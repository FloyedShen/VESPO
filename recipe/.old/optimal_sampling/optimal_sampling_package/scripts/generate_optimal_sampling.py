#!/usr/bin/env python3
"""
Large-Scale Data Processing with Optimal Sampling

Generates training data using optimal sampling (teacher + theta mixing).
Uses all visible GPUs (controlled by CUDA_VISIBLE_DEVICES).

Usage:
    # Process full dataset (single instance, uses all visible GPUs)
    CUDA_VISIBLE_DEVICES=0,1,2,3 python generate_optimal_sampling.py

    # Process slice 1 out of 8 (for distributed processing)
    CUDA_VISIBLE_DEVICES=0,1 python generate_optimal_sampling.py --data_slice 1::8

    # Resume from checkpoint
    CUDA_VISIBLE_DEVICES=0 python generate_optimal_sampling.py --data_slice 1::8 --resume

    # Multiple terminals/machines example:
    # Terminal 1 (machine A, slice 0-3):
    CUDA_VISIBLE_DEVICES=0,1 python generate_optimal_sampling.py --data_slice 0::8
    CUDA_VISIBLE_DEVICES=2,3 python generate_optimal_sampling.py --data_slice 1::8
    # Terminal 2 (machine B, slice 4-7):
    CUDA_VISIBLE_DEVICES=0,1 python generate_optimal_sampling.py --data_slice 4::8
    CUDA_VISIBLE_DEVICES=2,3 python generate_optimal_sampling.py --data_slice 5::8
"""

import os
import sys
import json
import argparse
import time
import re
import subprocess
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
import datasets
from tqdm import tqdm

# Add package to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from optimal_sampling import OptimalSamplingV1


@dataclass
class GenerationConfig:
    """Configuration for data generation"""
    # Model config
    model_teacher: str = "Qwen/Qwen2.5-3B-Instruct"
    model_theta: str = "Qwen/Qwen2.5-1.5B-Instruct"
    alpha_method: str = "kl_symmetry"

    # System prompts
    teacher_system_prompt: str = (
        "You are an analytical thinker who breaks down problems methodically. "
        "You have insight into effective solution strategies. "
        "Demonstrate clear thinking that naturally reaches the correct answer, "
        "and put your final answer within \\boxed{}"
    )
    theta_system_prompt: str = "Please reason step by step, and put your final answer within \\boxed{}"

    # Generation params
    max_tokens: int = 4096
    temperature: float = 0.8
    batch_size: int = 8

    # GPU config
    gpu_memory_utilization: float = 0.45
    max_num_seqs: int = 16

    # Dataset config
    dataset_name: str = "agentica-org/DeepScaleR-Preview-Dataset"
    dataset_split: str = "train"

    # Output config
    output_dir: str = "./data/optimal_sampling_output"
    checkpoint_interval: int = 100

    # Data slice config
    slice_id: int = 0
    total_slices: int = 1


def parse_data_slice(slice_str: str) -> tuple[int, int]:
    """
    Parse data slice string in format 'slice_id::total_slices'

    Examples:
        '0::8' -> (0, 8)  # First slice of 8
        '3::10' -> (3, 10)  # Fourth slice of 10
    """
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


def fix_solution_answer(solution: str, correct_answer: str) -> str:
    """
    Fix the answer in solution if it contains \\boxed{}

    Some solutions in the dataset have incorrect answers in \\boxed{}.
    This function replaces the boxed answer with the correct one from the 'answer' field.

    Args:
        solution: Original solution text
        correct_answer: Correct answer from dataset

    Returns:
        Fixed solution with correct answer
    """
    # Find and replace \boxed{...} with \boxed{correct_answer}
    # Use raw string pattern to match \boxed{anything}
    pattern = r'\\boxed\{[^}]*\}'

    # Use a function to avoid escape sequence issues
    def make_replacement(match):
        return '\\boxed{' + correct_answer + '}'

    fixed_solution = re.sub(pattern, make_replacement, solution)
    return fixed_solution


def create_prompts(example: Dict[str, Any]) -> tuple[str, str]:
    """
    Create teacher and theta prompts from dataset example

    Args:
        example: Dataset example with 'problem', 'solution', and 'answer' fields

    Returns:
        (teacher_prompt, theta_prompt)
    """
    problem = example['problem']
    solution = example['solution']
    correct_answer = example.get('answer', '')

    # Fix solution answer if needed
    if correct_answer:
        solution = fix_solution_answer(solution, correct_answer)

    # Teacher sees problem + solution hint (with corrected answer)
    teacher_prompt = f"{problem}\n\n##Hint\n{solution}"

    # Theta only sees problem (learns to reason from scratch)
    theta_prompt = problem

    return teacher_prompt, theta_prompt


def load_checkpoint(checkpoint_path: Path) -> Dict[str, Any]:
    """Load checkpoint file"""
    if checkpoint_path.exists():
        with open(checkpoint_path, 'r') as f:
            return json.load(f)
    return {"processed_indices": [], "last_index": -1}


def save_checkpoint(checkpoint_path: Path, checkpoint_data: Dict[str, Any]):
    """Save checkpoint file"""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with open(checkpoint_path, 'w') as f:
        json.dump(checkpoint_data, f, indent=2)


def save_output(output_path: Path, examples: List[Dict[str, Any]]):
    """Append generated examples to output JSONL file"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'a') as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')


def get_visible_gpus() -> List[int]:
    """Get list of visible GPU IDs from CUDA_VISIBLE_DEVICES"""
    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if not cuda_visible:
        # No CUDA_VISIBLE_DEVICES set, try to detect all GPUs
        try:
            result = subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True)
            gpu_count = len([line for line in result.stdout.strip().split('\n') if line])
            return list(range(gpu_count))
        except:
            return [0]  # Assume at least one GPU

    # Parse CUDA_VISIBLE_DEVICES
    gpu_ids = [int(gpu_id.strip()) for gpu_id in cuda_visible.split(',') if gpu_id.strip()]
    return gpu_ids if gpu_ids else [0]


def run_worker(gpu_id: int, gpu_index: int, num_gpus: int, args, start_idx: int, end_idx: int):
    """
    Worker function to process a GPU-specific sub-slice of data

    Args:
        gpu_id: Physical GPU ID
        gpu_index: Index in the list of visible GPUs (0, 1, 2, ...)
        num_gpus: Total number of GPUs being used
        args: Command line arguments
        start_idx: Start index for this GPU's sub-slice
        end_idx: End index for this GPU's sub-slice
    """
    # Set this worker to use only its assigned GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    # Parse data slice
    if args.data_slice:
        slice_id, total_slices = parse_data_slice(args.data_slice)
    else:
        slice_id, total_slices = 0, 1

    # Create config
    config = GenerationConfig(
        model_teacher=args.model_teacher,
        model_theta=args.model_theta,
        alpha_method=args.alpha_method,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        batch_size=args.batch_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        dataset_name=args.dataset,
        dataset_split=args.dataset_split,
        output_dir=args.output_dir,
        checkpoint_interval=args.checkpoint_interval,
        slice_id=slice_id,
        total_slices=total_slices,
    )

    print(f"🚀 GPU {gpu_index}/{num_gpus} (GPU ID {gpu_id}): Processing range [{start_idx}, {end_idx})")

    # Load dataset
    dataset = datasets.load_dataset(config.dataset_name, split=config.dataset_split)

    # Setup output paths with GPU info
    output_dir = Path(config.output_dir)
    if num_gpus > 1:
        if config.total_slices > 1:
            output_file = output_dir / f"slice_{config.slice_id}_of_{config.total_slices}_gpu_{gpu_index}_of_{num_gpus}.jsonl"
            checkpoint_file = output_dir / f"checkpoint_slice_{config.slice_id}_of_{config.total_slices}_gpu_{gpu_index}_of_{num_gpus}.json"
        else:
            output_file = output_dir / f"gpu_{gpu_index}_of_{num_gpus}.jsonl"
            checkpoint_file = output_dir / f"checkpoint_gpu_{gpu_index}_of_{num_gpus}.json"
    else:
        if config.total_slices > 1:
            output_file = output_dir / f"slice_{config.slice_id}_of_{config.total_slices}.jsonl"
            checkpoint_file = output_dir / f"checkpoint_slice_{config.slice_id}_of_{config.total_slices}.json"
        else:
            output_file = output_dir / "output.jsonl"
            checkpoint_file = output_dir / "checkpoint.json"

    # Load checkpoint if resuming
    checkpoint = load_checkpoint(checkpoint_file) if args.resume else {"processed_indices": [], "last_index": -1}
    processed_indices = set(checkpoint['processed_indices'])

    if args.resume:
        print(f"   GPU {gpu_index}: Resuming from checkpoint: {len(processed_indices)} examples already processed")

    # Initialize sampler
    sampler = OptimalSamplingV1(
        model_teacher=config.model_teacher,
        model_theta=config.model_theta,
        alpha_method=config.alpha_method,
        teacher_system_prompt=config.teacher_system_prompt,
        theta_system_prompt=config.theta_system_prompt,
        enable_chat_template=True,
        track_alpha_stats=False,
        gpu_memory_utilization=config.gpu_memory_utilization,
        max_num_seqs=config.max_num_seqs,
        enforce_eager=True
    )

    # Process in batches
    pending_save = []
    num_processed = 0
    start_time = time.time()

    # Create progress bar
    total_to_process = end_idx - start_idx - len([i for i in processed_indices if start_idx <= i < end_idx])
    pbar = tqdm(
        total=total_to_process,
        desc=f"GPU {gpu_index}/{num_gpus}",
        position=gpu_index,
    )

    # Collect batch
    batch_indices = []
    batch_teacher_prompts = []
    batch_theta_prompts = []
    batch_examples = []

    for idx in range(start_idx, end_idx):
        if idx in processed_indices:
            continue

        example = dataset[idx]
        teacher_prompt, theta_prompt = create_prompts(example)

        batch_indices.append(idx)
        batch_teacher_prompts.append(teacher_prompt)
        batch_theta_prompts.append(theta_prompt)
        batch_examples.append(example)

        # Process batch when full
        if len(batch_indices) >= config.batch_size:
            try:
                outputs = sampler.generate(
                    prompts=batch_teacher_prompts,
                    theta_prompts=batch_theta_prompts,
                    max_tokens=config.max_tokens,
                    temperature=config.temperature
                )

                for i, (bidx, bexample) in enumerate(zip(batch_indices, batch_examples)):
                    solution = bexample.get('solution', '')
                    if bexample.get('answer', ''):
                        solution = fix_solution_answer(solution, bexample['answer'])

                    output_record = {
                        "index": bidx,
                        "problem": bexample['problem'],
                        "solution": solution,
                        "generated_reasoning": outputs.generated_texts[i],
                        "num_tokens": outputs.num_tokens[i],
                        "method": "optimal_sampling",
                        "teacher_model": config.model_teacher,
                        "theta_model": config.model_theta,
                        "alpha_method": config.alpha_method,
                        "slice_id": config.slice_id,
                        "total_slices": config.total_slices,
                        "gpu_id": gpu_index,
                        "num_gpus": num_gpus,
                    }
                    pending_save.append(output_record)
                    processed_indices.add(bidx)
                    num_processed += 1

                pbar.update(len(batch_indices))

                # Save periodically
                if num_processed % config.checkpoint_interval == 0:
                    save_output(output_file, pending_save)
                    pending_save = []

                    checkpoint_data = {
                        "processed_indices": list(processed_indices),
                        "last_index": max(processed_indices),
                        "num_processed": num_processed,
                        "slice_id": config.slice_id,
                        "total_slices": config.total_slices,
                        "gpu_id": gpu_index,
                        "num_gpus": num_gpus,
                        "timestamp": time.time()
                    }
                    save_checkpoint(checkpoint_file, checkpoint_data)

            except Exception as e:
                print(f"\n❌ GPU {gpu_index}: Error processing batch starting at index {batch_indices[0]}: {e}")
                import traceback
                traceback.print_exc()

            # Clear batch
            batch_indices = []
            batch_teacher_prompts = []
            batch_theta_prompts = []
            batch_examples = []

    # Process remaining batch
    if batch_indices:
        try:
            outputs = sampler.generate(
                prompts=batch_teacher_prompts,
                theta_prompts=batch_theta_prompts,
                max_tokens=config.max_tokens,
                temperature=config.temperature
            )

            for i, (bidx, bexample) in enumerate(zip(batch_indices, batch_examples)):
                solution = bexample.get('solution', '')
                if bexample.get('answer', ''):
                    solution = fix_solution_answer(solution, bexample['answer'])

                output_record = {
                    "index": bidx,
                    "problem": bexample['problem'],
                    "solution": solution,
                    "generated_reasoning": outputs.generated_texts[i],
                    "num_tokens": outputs.num_tokens[i],
                    "method": "optimal_sampling",
                    "teacher_model": config.model_teacher,
                    "theta_model": config.model_theta,
                    "alpha_method": config.alpha_method,
                    "slice_id": config.slice_id,
                    "total_slices": config.total_slices,
                    "gpu_id": gpu_index,
                    "num_gpus": num_gpus,
                }
                pending_save.append(output_record)
                processed_indices.add(bidx)
                num_processed += 1

            pbar.update(len(batch_indices))

        except Exception as e:
            print(f"\n❌ GPU {gpu_index}: Error processing final batch: {e}")
            import traceback
            traceback.print_exc()

    # Final save
    if pending_save:
        save_output(output_file, pending_save)

    # Save final checkpoint
    checkpoint_data = {
        "processed_indices": list(processed_indices),
        "last_index": max(processed_indices) if processed_indices else -1,
        "num_processed": num_processed,
        "slice_id": config.slice_id,
        "total_slices": config.total_slices,
        "gpu_id": gpu_index,
        "num_gpus": num_gpus,
        "timestamp": time.time(),
        "completed": True
    }
    save_checkpoint(checkpoint_file, checkpoint_data)

    # Cleanup
    sampler.cleanup()
    pbar.close()

    elapsed = time.time() - start_time
    print(f"\n✅ GPU {gpu_index}/{num_gpus}: Completed {num_processed} examples in {elapsed:.1f}s ({num_processed/elapsed:.2f} ex/s)")
    print(f"   Output: {output_file}\n")


def main():
    parser = argparse.ArgumentParser(description="Generate training data with Optimal Sampling")

    # Model config
    parser.add_argument("--model_teacher", type=str, default="Qwen/Qwen2.5-3B-Instruct",
                        help="Teacher model path")
    parser.add_argument("--model_theta", type=str, default="Qwen/Qwen2.5-1.5B-Instruct",
                        help="Theta model path")
    parser.add_argument("--alpha_method", type=str, default="kl_symmetry",
                        choices=["kl_symmetry", "ess_balance", "entropy", "fixed"],
                        help="Alpha computation method")

    # Generation params
    parser.add_argument("--max_tokens", type=int, default=4096,
                        help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for generation")

    # GPU config
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.45,
                        help="GPU memory utilization fraction")
    parser.add_argument("--max_num_seqs", type=int, default=16,
                        help="Maximum number of sequences")

    # Dataset config
    parser.add_argument("--dataset", type=str, default="agentica-org/DeepScaleR-Preview-Dataset",
                        help="HuggingFace dataset name")
    parser.add_argument("--dataset_split", type=str, default="train",
                        help="Dataset split to use")
    parser.add_argument("--num_examples", type=int, default=None,
                        help="Number of examples to process (default: all)")

    # Output config
    parser.add_argument("--output_dir", type=str, default="./data/optimal_sampling_output",
                        help="Output directory")
    parser.add_argument("--checkpoint_interval", type=int, default=100,
                        help="Save checkpoint every N examples")

    # Data slice config
    parser.add_argument("--data_slice", type=str, default=None,
                        help="Data slice in format 'slice_id::total_slices' (e.g., '1::8' for 2nd slice of 8)")

    # Resume
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoint")

    args = parser.parse_args()

    # Parse data slice
    if args.data_slice:
        slice_id, total_slices = parse_data_slice(args.data_slice)
    else:
        slice_id, total_slices = 0, 1

    # Create config
    config = GenerationConfig(
        model_teacher=args.model_teacher,
        model_theta=args.model_theta,
        alpha_method=args.alpha_method,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        batch_size=args.batch_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        dataset_name=args.dataset,
        dataset_split=args.dataset_split,
        output_dir=args.output_dir,
        checkpoint_interval=args.checkpoint_interval,
        slice_id=slice_id,
        total_slices=total_slices,
    )

    # Load dataset to get size
    print(f"📦 Loading dataset: {config.dataset_name}")
    dataset = datasets.load_dataset(config.dataset_name, split=config.dataset_split)
    total_size = len(dataset)

    if args.num_examples is not None:
        total_size = min(total_size, args.num_examples)

    # Calculate slice range
    slice_size = total_size // config.total_slices
    start_idx = config.slice_id * slice_size
    end_idx = start_idx + slice_size if config.slice_id < config.total_slices - 1 else total_size

    print(f"   Total examples in dataset: {total_size}")
    print(f"   Data slice: {config.slice_id} / {config.total_slices}")
    print(f"   Processing range: [{start_idx}, {end_idx}) ({end_idx - start_idx} examples)")
    print()

    # Check for multi-GPU setup
    visible_gpus = get_visible_gpus()
    num_gpus = len(visible_gpus)

    if num_gpus > 1:
        # Multi-GPU mode: spawn multiple processes
        print(f"🔥 Multi-GPU mode detected: {num_gpus} GPUs available")
        print(f"   GPUs: {visible_gpus}")
        print(f"   Distributing slice [{start_idx}, {end_idx}) across {num_gpus} GPUs")
        print()

        # Calculate sub-slice ranges for each GPU
        slice_total = end_idx - start_idx
        gpu_slice_size = slice_total // num_gpus

        # Use multiprocessing to spawn workers
        import multiprocessing
        multiprocessing.set_start_method('spawn', force=True)
        processes = []

        for gpu_index, gpu_id in enumerate(visible_gpus):
            gpu_start = start_idx + gpu_index * gpu_slice_size
            gpu_end = gpu_start + gpu_slice_size if gpu_index < num_gpus - 1 else end_idx

            print(f"   GPU {gpu_index}/{num_gpus} (ID {gpu_id}): [{gpu_start}, {gpu_end}) ({gpu_end - gpu_start} examples)")

            p = multiprocessing.Process(
                target=run_worker,
                args=(gpu_id, gpu_index, num_gpus, args, gpu_start, gpu_end)
            )
            p.start()
            processes.append(p)

        print()
        print(f"✅ Launched {num_gpus} workers")
        print()

        # Wait for all processes to complete
        for p in processes:
            p.join()

        print()
        print(f"{'=' * 80}")
        print(f"✅ All {num_gpus} GPUs completed!")
        print(f"   Slice: {config.slice_id} / {config.total_slices}")
        print(f"   Total range: [{start_idx}, {end_idx})")
        print(f"{'=' * 80}")
        print()

        return  # Exit main after multi-GPU processing

    # Single-GPU mode: continue with original logic
    print(f"💻 Single-GPU mode")
    print()

    # Setup output paths with slice info
    output_dir = Path(config.output_dir)
    if config.total_slices > 1:
        output_file = output_dir / f"slice_{config.slice_id}_of_{config.total_slices}.jsonl"
        checkpoint_file = output_dir / f"checkpoint_slice_{config.slice_id}_of_{config.total_slices}.json"
    else:
        output_file = output_dir / "output.jsonl"
        checkpoint_file = output_dir / "checkpoint.json"

    print(f"📝 Output file: {output_file}")
    print(f"💾 Checkpoint file: {checkpoint_file}")
    print()

    # Load checkpoint if resuming
    checkpoint = load_checkpoint(checkpoint_file) if args.resume else {"processed_indices": [], "last_index": -1}
    processed_indices = set(checkpoint['processed_indices'])

    if args.resume:
        print(f"📂 Resuming from checkpoint: {len(processed_indices)} examples already processed")

    # Initialize sampler
    print(f"🔧 Initializing Optimal Sampling V1...")
    print(f"   Teacher: {config.model_teacher}")
    print(f"   Theta: {config.model_theta}")
    print(f"   Alpha method: {config.alpha_method}")
    print()

    sampler = OptimalSamplingV1(
        model_teacher=config.model_teacher,
        model_theta=config.model_theta,
        alpha_method=config.alpha_method,
        teacher_system_prompt=config.teacher_system_prompt,
        theta_system_prompt=config.theta_system_prompt,
        enable_chat_template=True,
        track_alpha_stats=False,  # Disable for faster generation
        gpu_memory_utilization=config.gpu_memory_utilization,
        max_num_seqs=config.max_num_seqs,
        enforce_eager=True
    )
    print(f"✅ Sampler initialized\n")

    # Process in batches
    pending_save = []
    num_processed = 0
    start_time = time.time()

    # Create progress bar
    total_to_process = end_idx - start_idx - len([i for i in processed_indices if start_idx <= i < end_idx])
    pbar = tqdm(
        total=total_to_process,
        desc=f"Slice {config.slice_id}/{config.total_slices}",
    )

    # Collect batch
    batch_indices = []
    batch_teacher_prompts = []
    batch_theta_prompts = []
    batch_examples = []

    for idx in range(start_idx, end_idx):
        # Skip if already processed
        if idx in processed_indices:
            continue

        # Get example
        example = dataset[idx]

        # Create prompts
        teacher_prompt, theta_prompt = create_prompts(example)

        # Add to batch
        batch_indices.append(idx)
        batch_teacher_prompts.append(teacher_prompt)
        batch_theta_prompts.append(theta_prompt)
        batch_examples.append(example)

        # Process batch when full
        if len(batch_indices) >= config.batch_size:
            try:
                # Generate
                outputs = sampler.generate(
                    prompts=batch_teacher_prompts,
                    theta_prompts=batch_theta_prompts,
                    max_tokens=config.max_tokens,
                    temperature=config.temperature
                )

                # Prepare output records
                for i, (bidx, bexample) in enumerate(zip(batch_indices, batch_examples)):
                    # Fix solution answer if needed
                    solution = bexample.get('solution', '')
                    if bexample.get('answer', ''):
                        solution = fix_solution_answer(solution, bexample['answer'])

                    output_record = {
                        "index": bidx,
                        "problem": bexample['problem'],
                        "solution": solution,  # Use fixed solution
                        "generated_reasoning": outputs.generated_texts[i],
                        "num_tokens": outputs.num_tokens[i],
                        "method": "optimal_sampling",
                        "teacher_model": config.model_teacher,
                        "theta_model": config.model_theta,
                        "alpha_method": config.alpha_method,
                        "slice_id": config.slice_id,
                        "total_slices": config.total_slices,
                    }
                    pending_save.append(output_record)
                    processed_indices.add(bidx)
                    num_processed += 1

                # Update progress
                pbar.update(len(batch_indices))

                # Save periodically
                if num_processed % config.checkpoint_interval == 0:
                    # Save outputs
                    save_output(output_file, pending_save)
                    pending_save = []

                    # Save checkpoint
                    checkpoint_data = {
                        "processed_indices": list(processed_indices),
                        "last_index": max(processed_indices),
                        "num_processed": num_processed,
                        "slice_id": config.slice_id,
                        "total_slices": config.total_slices,
                        "timestamp": time.time()
                    }
                    save_checkpoint(checkpoint_file, checkpoint_data)

            except Exception as e:
                print(f"\n❌ Error processing batch starting at index {batch_indices[0]}: {e}")
                import traceback
                traceback.print_exc()

            # Clear batch
            batch_indices = []
            batch_teacher_prompts = []
            batch_theta_prompts = []
            batch_examples = []

    # Process remaining batch
    if batch_indices:
        try:
            outputs = sampler.generate(
                prompts=batch_teacher_prompts,
                theta_prompts=batch_theta_prompts,
                max_tokens=config.max_tokens,
                temperature=config.temperature
            )

            for i, (bidx, bexample) in enumerate(zip(batch_indices, batch_examples)):
                # Fix solution answer if needed
                solution = bexample.get('solution', '')
                if bexample.get('answer', ''):
                    solution = fix_solution_answer(solution, bexample['answer'])

                output_record = {
                    "index": bidx,
                    "problem": bexample['problem'],
                    "solution": solution,  # Use fixed solution
                    "generated_reasoning": outputs.generated_texts[i],
                    "num_tokens": outputs.num_tokens[i],
                    "method": "optimal_sampling",
                    "teacher_model": config.model_teacher,
                    "theta_model": config.model_theta,
                    "alpha_method": config.alpha_method,
                    "slice_id": config.slice_id,
                    "total_slices": config.total_slices,
                }
                pending_save.append(output_record)
                processed_indices.add(bidx)
                num_processed += 1

            pbar.update(len(batch_indices))

        except Exception as e:
            print(f"\n❌ Error processing final batch: {e}")
            import traceback
            traceback.print_exc()

    # Final save
    if pending_save:
        save_output(output_file, pending_save)

    # Save final checkpoint
    checkpoint_data = {
        "processed_indices": list(processed_indices),
        "last_index": max(processed_indices) if processed_indices else -1,
        "num_processed": num_processed,
        "slice_id": config.slice_id,
        "total_slices": config.total_slices,
        "timestamp": time.time(),
        "completed": True
    }
    save_checkpoint(checkpoint_file, checkpoint_data)

    # Cleanup
    sampler.cleanup()
    pbar.close()

    # Print summary
    elapsed = time.time() - start_time
    print(f"\n{'=' * 80}")
    print(f"✅ Completed!")
    print(f"   Slice: {config.slice_id} / {config.total_slices}")
    print(f"   Processed: {num_processed} examples")
    print(f"   Time: {elapsed:.1f}s ({num_processed/elapsed:.2f} examples/s)")
    print(f"   Output: {output_file}")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
