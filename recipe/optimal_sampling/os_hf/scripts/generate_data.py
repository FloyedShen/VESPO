#!/usr/bin/env python3
"""
Data Generation with Optimal Sampling HF

Generates training data using optimal sampling (teacher + theta mixing) with HuggingFace + Flash Attention.
Supports distributed processing across multiple machines with data slicing.

Usage:
    # Process full dataset (single instance)
    python generate_data.py

    # Process slice 1 out of 8 (for distributed processing)
    python generate_data.py --data_slice 1::8

    # Resume from checkpoint
    python generate_data.py --data_slice 1::8 --resume

    # Multiple machines example:
    # Machine A:
    python generate_data.py --data_slice 0::8
    python generate_data.py --data_slice 1::8
    # Machine B:
    python generate_data.py --data_slice 2::8
    python generate_data.py --data_slice 3::8
"""

import os
import sys
import json
import argparse
import time
import re
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
import torch
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from optimal_sampling_hf import OptimalSamplingModel, create_dual_prompts, NATURAL_LANGUAGE_TEMPLATE


@dataclass
class GenerationConfig:
    """Configuration for data generation"""
    # Model config
    model_teacher: str = "Qwen/Qwen2.5-3B-Instruct"
    model_theta: str = "Qwen/Qwen2.5-1.5B-Instruct"
    alpha_method: str = "kl_symmetry"
    alpha_min: float = 0.0
    alpha_max: float = 1.0

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

    # Dataset config
    dataset_name: str = "agentica-org/DeepScaleR-Preview-Dataset"
    dataset_split: str = "train"

    # Output config
    output_dir: str = "./data/optimal_sampling_output"
    checkpoint_interval: int = 128  # Deprecated - now saves after every batch

    # Data slice config
    slice_id: int = 0
    total_slices: int = 1

    # Dual prompts config
    use_solution_hint: bool = True
    force_nlt_in_theta: bool = True
    add_think_token: bool = False

    # Device config
    device: str = "cuda"
    dtype: str = "bfloat16"


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
    Fix solution to ensure the boxed answer matches the correct answer

    Args:
        solution: Original solution text
        correct_answer: The correct answer

    Returns:
        Fixed solution with corrected answer
    """
    # Find all \\boxed{...} patterns
    pattern = r'\\boxed\{[^}]*\}'

    # Replace with correct answer using a function to avoid escape issues
    def make_replacement(match):
        return '\\boxed{' + correct_answer + '}'

    fixed_solution = re.sub(pattern, make_replacement, solution)
    return fixed_solution


def create_prompts_from_dataset(
    example: Dict[str, Any],
    teacher_system_prompt: str,
    theta_system_prompt: str,
    use_solution_hint: bool = True
) -> tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """
    Create message lists for teacher and theta from dataset example

    Args:
        example: Dataset example with 'problem', 'solution', and 'answer' fields
        teacher_system_prompt: System prompt for teacher
        theta_system_prompt: System prompt for base model
        use_solution_hint: Whether to include solution hint for teacher

    Returns:
        (messages_teacher, messages_theta): Message lists for both models
    """
    problem = example.get('problem', example.get('question', example.get('prompt', '')))
    solution = example.get('solution', '')
    correct_answer = example.get('answer', '')

    # Fix solution answer if needed
    if correct_answer and solution:
        solution = fix_solution_answer(solution, correct_answer)

    # Teacher messages
    if use_solution_hint and solution:
        # Teacher sees problem + solution hint (with corrected answer)
        teacher_user_content = f"{problem}\n\n##Hint\n{solution}"
    else:
        teacher_user_content = problem

    messages_teacher = [
        {"role": "system", "content": teacher_system_prompt},
        {"role": "user", "content": teacher_user_content}
    ]

    # Theta messages (only sees problem, learns to reason from scratch)
    messages_theta = [
        {"role": "system", "content": theta_system_prompt},
        {"role": "user", "content": problem}
    ]

    return messages_teacher, messages_theta


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


def main():
    parser = argparse.ArgumentParser(
        description="Generate training data with Optimal Sampling HF (HuggingFace + Flash Attention)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single machine, full dataset
  python generate_data.py

  # Distributed processing - Machine 1 processes slice 0
  python generate_data.py --data_slice 0::4

  # Distributed processing - Machine 2 processes slice 1
  python generate_data.py --data_slice 1::4

  # Resume from checkpoint
  python generate_data.py --data_slice 1::4 --resume
        """
    )

    # Model config
    parser.add_argument("--model_teacher", type=str, default="Qwen/Qwen2.5-3B-Instruct",
                        help="Teacher model path")
    parser.add_argument("--model_theta", type=str, default="Qwen/Qwen2.5-1.5B-Instruct",
                        help="Theta (base) model path")
    parser.add_argument("--alpha_method", type=str, default="kl_symmetry",
                        choices=["kl_symmetry", "ess_balance", "entropy", "fixed", "reverse_kl_symmetry"],
                        help="Alpha computation method")
    parser.add_argument("--alpha_min", type=float, default=0.0,
                        help="Minimum alpha value")
    parser.add_argument("--alpha_max", type=float, default=1.0,
                        help="Maximum alpha value")

    # System prompts
    parser.add_argument("--teacher_system_prompt", type=str,
                        default="You are an analytical thinker who breaks down problems methodically. "
                                "You have insight into effective solution strategies. "
                                "Demonstrate clear thinking that naturally reaches the correct answer, "
                                "and put your final answer within \\boxed{}",
                        help="System prompt for teacher model")
    parser.add_argument("--theta_system_prompt", type=str,
                        default="Please reason step by step, and put your final answer within \\boxed{}",
                        help="System prompt for base model")

    # Generation params
    parser.add_argument("--max_tokens", type=int, default=16384,
                        help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Batch size for generation")

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
                        help="(Deprecated - now saves after every batch) Kept for backward compatibility")

    # Data slice config
    parser.add_argument("--data_slice", type=str, default=None,
                        help="Data slice in format 'slice_id::total_slices' (e.g., '1::8' for 2nd slice of 8)")

    # Internal multi-GPU config (used by generate_data_multigpu.py)
    parser.add_argument("--data_slice_internal", type=str, default=None,
                        help=argparse.SUPPRESS)  # Hidden from help

    # Resume
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoint")

    # Dual prompts config
    parser.add_argument("--use_solution_hint", action="store_true", default=True,
                        help="Whether teacher sees solution hint")
    parser.add_argument("--force_nlt_in_theta", action="store_true", default=False,
                        help="Force natural language template for base model")
    parser.add_argument("--add_think_token", action="store_true", default=False,
                        help="Add <think> token to prompts")

    # Device config
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda or cpu)")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["float16", "bfloat16", "float32"],
                        help="Data type for model")

    args = parser.parse_args()

    # Parse data slice (internal multi-GPU takes precedence)
    machine_slice_id = None
    local_gpu_id = None

    if args.data_slice_internal:
        # Format: global_gpu_id::total_gpus::machine_slice_id::local_gpu_id
        parts = args.data_slice_internal.split('::')
        if len(parts) == 4:
            slice_id = int(parts[0])
            total_slices = int(parts[1])
            machine_slice_id = int(parts[2])
            local_gpu_id = int(parts[3])
        else:
            raise ValueError(f"Invalid --data_slice_internal format: {args.data_slice_internal}")
    elif args.data_slice:
        slice_id, total_slices = parse_data_slice(args.data_slice)
    else:
        slice_id, total_slices = 0, 1

    # Create config
    config = GenerationConfig(
        model_teacher=args.model_teacher,
        model_theta=args.model_theta,
        alpha_method=args.alpha_method,
        alpha_min=args.alpha_min,
        alpha_max=args.alpha_max,
        teacher_system_prompt=args.teacher_system_prompt,
        theta_system_prompt=args.theta_system_prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        batch_size=args.batch_size,
        dataset_name=args.dataset,
        dataset_split=args.dataset_split,
        output_dir=args.output_dir,
        checkpoint_interval=args.checkpoint_interval,
        slice_id=slice_id,
        total_slices=total_slices,
        use_solution_hint=args.use_solution_hint,
        force_nlt_in_theta=args.force_nlt_in_theta,
        add_think_token=args.add_think_token,
        device=args.device,
        dtype=args.dtype,
    )

    # Convert dtype
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map[config.dtype]

    # Print configuration
    print("=" * 80)
    print("Optimal Sampling HF - Data Generation")
    print("=" * 80)
    print("\nConfiguration:")
    print(f"  Teacher model (π_t): {config.model_teacher}")
    print(f"  Base model (π_θ): {config.model_theta}")
    print(f"  Alpha method: {config.alpha_method}")
    print(f"  Alpha range: [{config.alpha_min}, {config.alpha_max}]")
    print(f"  Max tokens: {config.max_tokens}")
    print(f"  Temperature: {config.temperature}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Dataset: {config.dataset_name}")
    print(f"  Use solution hint: {config.use_solution_hint}")
    print(f"  Force NLT for θ: {config.force_nlt_in_theta}")
    print(f"  Device: {config.device}")
    print(f"  Dtype: {config.dtype}")

    # Load dataset to get size
    print(f"\n📦 Loading dataset: {config.dataset_name}")
    from datasets import load_dataset
    dataset = load_dataset(config.dataset_name, split=config.dataset_split)
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

    # Setup output paths
    output_dir = Path(config.output_dir)
    if machine_slice_id is not None and local_gpu_id is not None:
        # Multi-machine multi-GPU mode
        output_file = output_dir / f"slice_{machine_slice_id}_gpu_{local_gpu_id}.jsonl"
        checkpoint_file = output_dir / f"checkpoint_slice_{machine_slice_id}_gpu_{local_gpu_id}.json"
    elif config.total_slices > 1:
        # Single-machine multi-slice mode
        output_file = output_dir / f"slice_{config.slice_id}_of_{config.total_slices}.jsonl"
        checkpoint_file = output_dir / f"checkpoint_slice_{config.slice_id}_of_{config.total_slices}.json"
    else:
        # Single machine single GPU mode
        output_file = output_dir / "output.jsonl"
        checkpoint_file = output_dir / "checkpoint.json"

    print(f"📁 Output file: {output_file}")
    print(f"📁 Checkpoint file: {checkpoint_file}")
    print()

    # Load checkpoint if resuming
    checkpoint = load_checkpoint(checkpoint_file) if args.resume else {"processed_indices": [], "last_index": -1}
    processed_indices = set(checkpoint['processed_indices'])

    if args.resume:
        print(f"📂 Resuming from checkpoint: {len(processed_indices)} examples already processed\n")

    # Initialize model
    print("🔧 Initializing model...")
    model = OptimalSamplingModel(
        model_theta_path=config.model_theta,
        model_t_path=config.model_teacher,
        alpha_method=config.alpha_method,
        alpha_min=config.alpha_min,
        alpha_max=config.alpha_max,
        device=config.device,
        dtype=dtype,
    )

    # Process in batches
    pending_save = []
    num_processed = 0
    start_time = time.time()

    # Create progress bar
    total_to_process = end_idx - start_idx - len([i for i in processed_indices if start_idx <= i < end_idx])
    pbar = tqdm(total=total_to_process, desc="Generating")

    # Collect batch
    batch_indices = []
    batch_messages_teacher = []
    batch_messages_theta = []
    batch_examples = []

    for idx in range(start_idx, end_idx):
        if idx in processed_indices:
            continue

        example = dataset[idx]

        # Create messages
        msg_teacher, msg_theta = create_prompts_from_dataset(
            example,
            config.teacher_system_prompt,
            config.theta_system_prompt,
            config.use_solution_hint
        )

        batch_indices.append(idx)
        batch_messages_teacher.append(msg_teacher)
        batch_messages_theta.append(msg_theta)
        batch_examples.append(example)

        # Process batch when full
        if len(batch_indices) >= config.batch_size:
            try:
                # Create dual prompts using chat templates
                prompts_theta, _ = create_dual_prompts(
                    batch_messages_theta,
                    model.tokenizer_theta,
                    model.tokenizer_t,
                    force_nlt_in_theta=config.force_nlt_in_theta,
                    base_template=NATURAL_LANGUAGE_TEMPLATE,
                    add_generation_prompt=True,
                    add_think_token=config.add_think_token
                )

                _, prompts_teacher = create_dual_prompts(
                    batch_messages_teacher,
                    model.tokenizer_theta,
                    model.tokenizer_t,
                    force_nlt_in_theta=False,  # Teacher uses standard template
                    add_generation_prompt=True,
                    add_think_token=config.add_think_token
                )

                # Generate
                outputs = model.generate(
                    prompts=prompts_theta,
                    prompts_t=prompts_teacher,
                    max_new_tokens=config.max_tokens,
                    temperature=config.temperature,
                    return_diagnostics=False,
                    skip_decode=False,
                )

                # Save outputs
                for i, (bidx, bexample) in enumerate(zip(batch_indices, batch_examples)):
                    solution = bexample.get('solution', '')
                    if bexample.get('answer', ''):
                        solution = fix_solution_answer(solution, bexample['answer'])

                    output_record = {
                        "index": bidx,
                        "problem": bexample.get('problem', bexample.get('question', bexample.get('prompt', ''))),
                        "solution": solution,
                        "answer": bexample.get('answer', ''),
                        "generated_reasoning": outputs.generated_texts[i],
                        "num_tokens": outputs.generated_ids[i].shape[0],
                        "method": "optimal_sampling_hf",
                        "teacher_model": config.model_teacher,
                        "theta_model": config.model_theta,
                        "alpha_method": config.alpha_method,
                        "slice_id": config.slice_id,
                        "total_slices": config.total_slices,
                    }

                    # Add multi-machine multi-GPU metadata if applicable
                    if machine_slice_id is not None and local_gpu_id is not None:
                        output_record["machine_slice_id"] = machine_slice_id
                        output_record["local_gpu_id"] = local_gpu_id

                    pending_save.append(output_record)
                    processed_indices.add(bidx)
                    num_processed += 1

                pbar.update(len(batch_indices))

                # Save after every batch (instead of waiting for checkpoint_interval)
                save_output(output_file, pending_save)
                pending_save = []

                checkpoint_data = {
                    "processed_indices": list(processed_indices),
                    "last_index": max(processed_indices),
                    "num_processed": num_processed,
                    "slice_id": config.slice_id,
                    "total_slices": config.total_slices,
                    "timestamp": time.time()
                }

                # Add multi-machine multi-GPU metadata if applicable
                if machine_slice_id is not None and local_gpu_id is not None:
                    checkpoint_data["machine_slice_id"] = machine_slice_id
                    checkpoint_data["local_gpu_id"] = local_gpu_id

                save_checkpoint(checkpoint_file, checkpoint_data)

            except Exception as e:
                print(f"\n❌ Error processing batch starting at index {batch_indices[0]}: {e}")
                import traceback
                traceback.print_exc()

            # Clear batch
            batch_indices = []
            batch_messages_teacher = []
            batch_messages_theta = []
            batch_examples = []

    # Process remaining batch
    if batch_indices:
        try:
            # Create dual prompts using chat templates
            prompts_theta, _ = create_dual_prompts(
                batch_messages_theta,
                model.tokenizer_theta,
                model.tokenizer_t,
                force_nlt_in_theta=config.force_nlt_in_theta,
                base_template=NATURAL_LANGUAGE_TEMPLATE,
                add_generation_prompt=True,
                add_think_token=config.add_think_token
            )

            _, prompts_teacher = create_dual_prompts(
                batch_messages_teacher,
                model.tokenizer_theta,
                model.tokenizer_t,
                force_nlt_in_theta=False,
                add_generation_prompt=True,
                add_think_token=config.add_think_token
            )

            # Generate
            outputs = model.generate(
                prompts=prompts_theta,
                prompts_t=prompts_teacher,
                max_new_tokens=config.max_tokens,
                temperature=config.temperature,
                return_diagnostics=False,
                skip_decode=False,
            )

            # Save outputs
            for i, (bidx, bexample) in enumerate(zip(batch_indices, batch_examples)):
                solution = bexample.get('solution', '')
                if bexample.get('answer', ''):
                    solution = fix_solution_answer(solution, bexample['answer'])

                output_record = {
                    "index": bidx,
                    "problem": bexample.get('problem', bexample.get('question', bexample.get('prompt', ''))),
                    "solution": solution,
                    "answer": bexample.get('answer', ''),
                    "generated_reasoning": outputs.generated_texts[i],
                    "num_tokens": outputs.generated_ids[i].shape[0],
                    "method": "optimal_sampling_hf",
                    "teacher_model": config.model_teacher,
                    "theta_model": config.model_theta,
                    "alpha_method": config.alpha_method,
                    "slice_id": config.slice_id,
                    "total_slices": config.total_slices,
                }

                # Add multi-machine multi-GPU metadata if applicable
                if machine_slice_id is not None and local_gpu_id is not None:
                    output_record["machine_slice_id"] = machine_slice_id
                    output_record["local_gpu_id"] = local_gpu_id

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

    # Add multi-machine multi-GPU metadata if applicable
    if machine_slice_id is not None and local_gpu_id is not None:
        checkpoint_data["machine_slice_id"] = machine_slice_id
        checkpoint_data["local_gpu_id"] = local_gpu_id

    save_checkpoint(checkpoint_file, checkpoint_data)

    pbar.close()

    elapsed = time.time() - start_time
    print(f"\n✅ Completed {num_processed} examples in {elapsed:.1f}s ({num_processed/elapsed:.2f} ex/s)")
    print(f"   Output: {output_file}\n")

    # Print summary
    print("=" * 80)
    print("Generation Summary")
    print("=" * 80)
    print(f"Total examples processed: {num_processed}")
    print(f"Output file: {output_file}")
    if output_file.exists():
        print(f"File size: {output_file.stat().st_size / 1024 / 1024:.2f} MB")
    print("=" * 80)


if __name__ == "__main__":
    main()
