#!/usr/bin/env python3
"""
Multi-Machine Multi-GPU Data Generation with Optimal Sampling HF

This script automatically detects available GPUs and launches one process per GPU.
Each process handles a different portion of the dataset.

Usage:
    # Machine 0 (out of 8 machines total)
    python generate_data_multigpu.py --data_slice 0::8

    # Machine 1 (out of 8 machines total)
    python generate_data_multigpu.py --data_slice 1::8

    # Specify GPU list explicitly
    CUDA_VISIBLE_DEVICES=0,1,2,3 python generate_data_multigpu.py --data_slice 0::8

Data Distribution:
    - If you have 8 machines, each with 4 GPUs (total 32 workers):
      - Machine 0, GPU 0 processes: slice_0_gpu_0 (indices 0-N/32)
      - Machine 0, GPU 1 processes: slice_0_gpu_1 (indices 1*N/32-2*N/32)
      - ...
      - Machine 7, GPU 3 processes: slice_7_gpu_3 (indices 31*N/32-N)

Output Files:
    - Data: slice_0_gpu_0.jsonl, slice_0_gpu_1.jsonl, ...
    - Checkpoints: checkpoint_slice_0_gpu_0.json, ...
    - Logs: logs/slice_0_gpu_0.log, ...
"""

import os
import sys
import subprocess
import argparse
import time
import signal
from pathlib import Path

def get_available_gpus():
    """Get list of available GPU IDs from CUDA_VISIBLE_DEVICES or nvidia-smi"""
    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', None)

    if cuda_visible is not None:
        # Parse CUDA_VISIBLE_DEVICES
        if cuda_visible == '':
            return []
        gpu_ids = [int(x.strip()) for x in cuda_visible.split(',') if x.strip().isdigit()]
        return list(range(len(gpu_ids)))  # Return 0, 1, 2, ... based on number of GPUs

    # Try to detect GPUs using nvidia-smi
    try:
        result = subprocess.run(
            ['nvidia-smi', '--list-gpus'],
            capture_output=True,
            text=True,
            check=True
        )
        num_gpus = len([line for line in result.stdout.strip().split('\n') if line])
        return list(range(num_gpus))
    except:
        # Fallback: assume single GPU
        return [0]


def parse_data_slice(slice_str):
    """Parse data slice string 'slice_id::total_slices'"""
    try:
        parts = slice_str.split('::')
        if len(parts) != 2:
            raise ValueError(f"Invalid format: {slice_str}")

        slice_id = int(parts[0])
        total_slices = int(parts[1])

        if slice_id < 0 or slice_id >= total_slices:
            raise ValueError(f"slice_id {slice_id} must be in range [0, {total_slices})")

        return slice_id, total_slices
    except Exception as e:
        raise ValueError(f"Failed to parse data_slice '{slice_str}': {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Multi-GPU Optimal Sampling Data Generation (Multi-Machine Support)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Machine 0 out of 8 machines (auto-detect GPUs)
  python generate_data_multigpu.py --data_slice 0::8 --output_dir ./output

  # Machine 1 out of 8 machines with explicit GPU selection
  CUDA_VISIBLE_DEVICES=0,1,2,3 python generate_data_multigpu.py --data_slice 1::8

  # Single machine mode (no slicing)
  python generate_data_multigpu.py --output_dir ./output
        """
    )

    # Model config
    parser.add_argument("--model_teacher", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--model_theta", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--alpha_method", type=str, default="kl_symmetry",
                        choices=["kl_symmetry", "ess_balance", "entropy", "fixed", "reverse_kl_symmetry"])
    parser.add_argument("--alpha_min", type=float, default=0.0)
    parser.add_argument("--alpha_max", type=float, default=1.0)

    # System prompts
    parser.add_argument("--teacher_system_prompt", type=str,
                        default="You are an analytical thinker who breaks down problems methodically. "
                                "You have insight into effective solution strategies. "
                                "Demonstrate clear thinking that naturally reaches the correct answer, "
                                "and put your final answer within \\boxed{}")
    parser.add_argument("--theta_system_prompt", type=str,
                        default="Please reason step by step, and put your final answer within \\boxed{}")

    # Generation params
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--batch_size", type=int, default=8)

    # Dataset config
    parser.add_argument("--dataset", type=str, default="agentica-org/DeepScaleR-Preview-Dataset")
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--num_examples", type=int, default=None)

    # Output config
    parser.add_argument("--output_dir", type=str, default="./data/optimal_sampling_output")
    parser.add_argument("--checkpoint_interval", type=int, default=100,
                        help="(Deprecated - now saves after every batch) Kept for backward compatibility")

    # Multi-machine config
    parser.add_argument("--data_slice", type=str, default=None,
                        help="Machine slice in format 'slice_id::total_slices' (e.g., '0::8' for machine 0 of 8)")
    parser.add_argument("--resume", action="store_true")

    # Dual prompts config
    parser.add_argument("--use_solution_hint", action="store_true", default=True)
    parser.add_argument("--force_nlt_in_theta", action="store_true", default=False)
    parser.add_argument("--add_think_token", action="store_true", default=False)

    # Device config
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["float16", "bfloat16", "float32"])

    args = parser.parse_args()

    # Parse machine slice
    if args.data_slice:
        machine_slice_id, total_machines = parse_data_slice(args.data_slice)
    else:
        machine_slice_id, total_machines = 0, 1

    # Detect available GPUs
    gpu_ids = get_available_gpus()
    num_gpus = len(gpu_ids)

    if num_gpus == 0:
        print("❌ No GPUs detected! Please check CUDA_VISIBLE_DEVICES or nvidia-smi")
        sys.exit(1)

    # Print configuration
    print("=" * 80)
    print("Multi-Machine Multi-GPU Optimal Sampling Data Generation")
    print("=" * 80)
    print(f"\n📍 Machine Configuration:")
    print(f"   Machine ID: {machine_slice_id} / {total_machines}")
    print(f"   GPUs on this machine: {num_gpus} (IDs: {gpu_ids})")
    print(f"   Total workers: {total_machines * num_gpus}")
    print(f"\n🔧 Model Configuration:")
    print(f"   Teacher: {args.model_teacher}")
    print(f"   Theta: {args.model_theta}")
    print(f"   Alpha method: {args.alpha_method}")
    print(f"\n📊 Generation Parameters:")
    print(f"   Dataset: {args.dataset}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Max tokens: {args.max_tokens}")
    print(f"   Temperature: {args.temperature}")
    print(f"\n📁 Output:")
    print(f"   Directory: {args.output_dir}")
    print(f"   Files: slice_{machine_slice_id}_gpu_0.jsonl, slice_{machine_slice_id}_gpu_1.jsonl, ...")
    print()

    # Create output directory and logs directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Build base command
    base_cmd = [
        sys.executable,
        str(Path(__file__).parent / "generate_data.py"),
        "--model_teacher", args.model_teacher,
        "--model_theta", args.model_theta,
        "--alpha_method", args.alpha_method,
        "--alpha_min", str(args.alpha_min),
        "--alpha_max", str(args.alpha_max),
        "--teacher_system_prompt", args.teacher_system_prompt,
        "--theta_system_prompt", args.theta_system_prompt,
        "--max_tokens", str(args.max_tokens),
        "--temperature", str(args.temperature),
        "--batch_size", str(args.batch_size),
        "--dataset", args.dataset,
        "--dataset_split", args.dataset_split,
        "--output_dir", args.output_dir,
        "--checkpoint_interval", str(args.checkpoint_interval),
        "--device", args.device,
        "--dtype", args.dtype,
    ]

    if args.num_examples is not None:
        base_cmd.extend(["--num_examples", str(args.num_examples)])

    if args.resume:
        base_cmd.append("--resume")

    if args.use_solution_hint:
        base_cmd.append("--use_solution_hint")

    if args.force_nlt_in_theta:
        base_cmd.append("--force_nlt_in_theta")

    if args.add_think_token:
        base_cmd.append("--add_think_token")

    # Launch processes
    processes = []
    print("🚀 Launching GPU processes...")
    print()

    for local_gpu_id in gpu_ids:
        # Calculate global GPU ID
        global_gpu_id = machine_slice_id * num_gpus + local_gpu_id
        total_gpus = total_machines * num_gpus

        # Build command for this GPU
        cmd = base_cmd.copy()
        cmd.extend([
            "--data_slice_internal",  # Special internal flag
            f"{global_gpu_id}::{total_gpus}::{machine_slice_id}::{local_gpu_id}"
        ])

        # Set environment
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_ids[local_gpu_id])

        # Output files
        log_file = logs_dir / f"slice_{machine_slice_id}_gpu_{local_gpu_id}.log"

        print(f"   GPU {local_gpu_id} (global {global_gpu_id}/{total_gpus}):")
        print(f"      Log: {log_file}")
        print(f"      Output: slice_{machine_slice_id}_gpu_{local_gpu_id}.jsonl")

        # Launch process
        with open(log_file, 'w') as f:
            process = subprocess.Popen(
                cmd,
                env=env,
                stdout=f,
                stderr=subprocess.STDOUT,
                preexec_fn=os.setsid if hasattr(os, 'setsid') else None
            )

        processes.append({
            'process': process,
            'gpu_id': local_gpu_id,
            'global_gpu_id': global_gpu_id,
            'log_file': log_file
        })

        # Small delay to avoid race conditions
        time.sleep(1)

    print()
    print("=" * 80)
    print(f"✅ Launched {len(processes)} processes")
    print("=" * 80)
    print()
    print("💡 Monitor progress:")
    print(f"   tail -f {logs_dir}/slice_{machine_slice_id}_gpu_0.log")
    print()
    print("📊 Check output files:")
    print(f"   ls -lh {output_dir}/slice_{machine_slice_id}_gpu_*.jsonl")
    print()
    print("⏳ Waiting for all processes to complete...")
    print("   Press Ctrl+C to stop all processes")
    print()

    # Handle Ctrl+C
    def signal_handler(sig, frame):
        print("\n⚠️  Caught interrupt signal. Stopping all processes...")
        for p_info in processes:
            try:
                if hasattr(os, 'killpg'):
                    os.killpg(os.getpgid(p_info['process'].pid), signal.SIGTERM)
                else:
                    p_info['process'].terminate()
            except:
                pass

        # Wait for processes to die
        time.sleep(2)
        for p_info in processes:
            try:
                p_info['process'].kill()
            except:
                pass

        print("✅ All processes stopped")
        sys.exit(1)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Wait for all processes
    failed = 0
    completed = 0

    for p_info in processes:
        returncode = p_info['process'].wait()
        gpu_id = p_info['gpu_id']

        if returncode == 0:
            print(f"✅ GPU {gpu_id} completed successfully")
            completed += 1
        else:
            print(f"❌ GPU {gpu_id} failed with code {returncode}")
            print(f"   Check log: {p_info['log_file']}")
            failed += 1

    print()
    print("=" * 80)
    print("Results Summary")
    print("=" * 80)

    # Check output files
    for p_info in processes:
        gpu_id = p_info['gpu_id']
        output_file = output_dir / f"slice_{machine_slice_id}_gpu_{gpu_id}.jsonl"

        if output_file.exists():
            lines = sum(1 for _ in open(output_file))
            size = output_file.stat().st_size / 1024 / 1024
            print(f"✅ GPU {gpu_id}: {lines} examples, {size:.2f} MB")
        else:
            print(f"❌ GPU {gpu_id}: No output file")

    print()
    print(f"Completed: {completed} / {len(processes)}")
    print(f"Failed: {failed} / {len(processes)}")
    print()

    if failed == 0:
        print("=" * 80)
        print("🎉 All GPU processes completed successfully!")
        print("=" * 80)
        print()
        print("📦 Next steps:")
        print(f"   1. Copy results from all machines to one location")
        print(f"   2. Merge all slices:")
        print(f"      cat {output_dir}/slice_*_gpu_*.jsonl > {output_dir}/all_data.jsonl")
        print()
        sys.exit(0)
    else:
        print("=" * 80)
        print("⚠️  Some GPU processes failed")
        print("=" * 80)
        print()
        print(f"Check logs in: {logs_dir}/")
        print()
        sys.exit(1)


if __name__ == "__main__":
    main()
