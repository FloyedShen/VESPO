#!/usr/bin/env python3
"""
Large-Scale Data Processing with Single Model Baseline

Generates training data using a single model (no optimal sampling).
Supports two modes:
    - with_solution: Model sees problem + solution hint (like teacher)
    - without_solution: Model only sees problem (like theta/student)

Uses all visible GPUs (controlled by CUDA_VISIBLE_DEVICES).

Usage:
    # Process full dataset, without solution
    CUDA_VISIBLE_DEVICES=0,1,2,3 python generate_baseline.py --mode without_solution

    # Process slice 1 out of 8
    CUDA_VISIBLE_DEVICES=0,1 python generate_baseline.py --mode without_solution --data_slice 1::8

    # Resume from checkpoint
    CUDA_VISIBLE_DEVICES=0 python generate_baseline.py --mode with_solution --data_slice 1::8 --resume

    # Multiple terminals/machines example:
    # Terminal 1 (machine A):
    CUDA_VISIBLE_DEVICES=0,1 python generate_baseline.py --mode without_solution --data_slice 0::8
    CUDA_VISIBLE_DEVICES=2,3 python generate_baseline.py --mode without_solution --data_slice 1::8
    # Terminal 2 (machine B):
    CUDA_VISIBLE_DEVICES=0,1 python generate_baseline.py --mode without_solution --data_slice 4::8
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
import datasets
from tqdm import tqdm
from vllm import LLM, SamplingParams

# Add package to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class GenerationConfig:
    """Configuration for data generation"""
    # Model config
    model: str = "Qwen/Qwen2.5-3B-Instruct"

    # Mode: with_solution or without_solution
    mode: str = "without_solution"  # "with_solution" or "without_solution"

    # System prompts
    with_solution_system_prompt: str = (
        "You are an analytical thinker who breaks down problems methodically. "
        "You have insight into effective solution strategies. "
        "Demonstrate clear thinking that naturally reaches the correct answer, "
        "and put your final answer within \\boxed{}"
    )
    without_solution_system_prompt: str = "Please reason step by step, and put your final answer within \\boxed{}"

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
    output_dir: str = "./data/baseline_output"
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

    # Use a lambda function to avoid escape sequence issues in replacement
    def make_replacement(match):
        return '\\boxed{' + correct_answer + '}'

    fixed_solution = re.sub(pattern, make_replacement, solution)
    return fixed_solution


def create_prompt(example: Dict[str, Any], mode: str) -> str:
    """
    Create prompt from dataset example based on mode

    Args:
        example: Dataset example with 'problem', 'solution', and 'answer' fields
        mode: "with_solution" or "without_solution"

    Returns:
        Prompt string
    """
    problem = example['problem']

    if mode == "with_solution":
        # Model sees problem + solution hint (like teacher)
        solution = example['solution']
        correct_answer = example.get('answer', '')

        # Fix solution answer if needed
        if correct_answer:
            solution = fix_solution_answer(solution, correct_answer)

        prompt = f"{problem}\n\n##Hint\n{solution}"
    else:
        # Model only sees problem (like theta/student)
        prompt = problem

    return prompt


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
    parser = argparse.ArgumentParser(description="Generate training data with single model baseline")

    # Model config
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct",
                        help="Model path")

    # Mode
    parser.add_argument("--mode", type=str, required=True,
                        choices=["with_solution", "without_solution"],
                        help="Generation mode: with_solution (sees hint) or without_solution (no hint)")

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
    parser.add_argument("--output_dir", type=str, default="./data/baseline_output",
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
        model=args.model,
        mode=args.mode,
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
    print(f"   Mode: {config.mode}")
    print(f"   Data slice: {config.slice_id} / {config.total_slices}")
    print(f"   Processing range: [{start_idx}, {end_idx}) ({end_idx - start_idx} examples)")
    print()

    # Setup output paths with slice and mode info
    output_dir = Path(config.output_dir)
    if config.total_slices > 1:
        output_file = output_dir / f"slice_{config.slice_id}_of_{config.total_slices}_{config.mode}.jsonl"
        checkpoint_file = output_dir / f"checkpoint_slice_{config.slice_id}_of_{config.total_slices}_{config.mode}.json"
    else:
        output_file = output_dir / f"{config.mode}.jsonl"
        checkpoint_file = output_dir / f"checkpoint_{config.mode}.json"

    print(f"📝 Output file: {output_file}")
    print(f"💾 Checkpoint file: {checkpoint_file}")
    print()

    # Load checkpoint if resuming
    checkpoint = load_checkpoint(checkpoint_file) if args.resume else {"processed_indices": [], "last_index": -1}
    processed_indices = set(checkpoint['processed_indices'])

    if args.resume:
        print(f"📂 Resuming from checkpoint: {len(processed_indices)} examples already processed")

    # Select system prompt based on mode
    system_prompt = (
        config.with_solution_system_prompt if config.mode == "with_solution"
        else config.without_solution_system_prompt
    )

    # Initialize LLM
    print(f"🔧 Initializing LLM...")
    print(f"   Model: {config.model}")
    print(f"   Mode: {config.mode}")
    print(f"   System prompt: {system_prompt[:50]}...")
    print()

    llm = LLM(
        model=config.model,
        gpu_memory_utilization=config.gpu_memory_utilization,
        trust_remote_code=True,
        enable_prefix_caching=True,
        disable_log_stats=True,
        max_num_seqs=config.max_num_seqs,
        enforce_eager=True
    )

    # Get tokenizer for chat template
    tokenizer = llm.get_tokenizer()
    print(f"✅ LLM initialized\n")

    # Process in batches
    pending_save = []
    num_processed = 0
    start_time = time.time()

    # Create progress bar
    total_to_process = end_idx - start_idx - len([i for i in processed_indices if start_idx <= i < end_idx])
    pbar = tqdm(
        total=total_to_process,
        desc=f"Slice {config.slice_id}/{config.total_slices} ({config.mode})",
    )

    # Collect batch
    batch_indices = []
    batch_prompts = []
    batch_examples = []

    for idx in range(start_idx, end_idx):
        # Skip if already processed
        if idx in processed_indices:
            continue

        # Get example
        example = dataset[idx]

        # Create prompt based on mode
        prompt = create_prompt(example, config.mode)

        # Apply chat template
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Add to batch
        batch_indices.append(idx)
        batch_prompts.append(formatted_prompt)
        batch_examples.append(example)

        # Process batch when full
        if len(batch_indices) >= config.batch_size:
            try:
                # Create sampling params
                sampling_params = SamplingParams(
                    max_tokens=config.max_tokens,
                    temperature=config.temperature,
                )

                # Generate
                outputs = llm.generate(
                    prompts=batch_prompts,
                    sampling_params=sampling_params,
                    use_tqdm=False
                )

                # Prepare output records
                for i, (bidx, bexample) in enumerate(zip(batch_indices, batch_examples)):
                    # Fix solution answer if needed
                    solution = bexample.get('solution', '')
                    if config.mode == "with_solution" and bexample.get('answer', ''):
                        solution = fix_solution_answer(solution, bexample['answer'])

                    output_record = {
                        "index": bidx,
                        "problem": bexample['problem'],
                        "solution": solution,  # Use fixed solution
                        "generated_reasoning": outputs[i].outputs[0].text,
                        "num_tokens": len(outputs[i].outputs[0].token_ids),
                        "method": f"baseline_{config.mode}",
                        "model": config.model,
                        "mode": config.mode,
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
                        "mode": config.mode,
                        "timestamp": time.time()
                    }
                    save_checkpoint(checkpoint_file, checkpoint_data)

            except Exception as e:
                print(f"\n❌ Error processing batch starting at index {batch_indices[0]}: {e}")
                import traceback
                traceback.print_exc()

            # Clear batch
            batch_indices = []
            batch_prompts = []
            batch_examples = []

    # Process remaining batch
    if batch_indices:
        try:
            sampling_params = SamplingParams(
                max_tokens=config.max_tokens,
                temperature=config.temperature,
            )

            outputs = llm.generate(
                prompts=batch_prompts,
                sampling_params=sampling_params,
                use_tqdm=False
            )

            for i, (bidx, bexample) in enumerate(zip(batch_indices, batch_examples)):
                # Fix solution answer if needed
                solution = bexample.get('solution', '')
                if config.mode == "with_solution" and bexample.get('answer', ''):
                    solution = fix_solution_answer(solution, bexample['answer'])

                output_record = {
                    "index": bidx,
                    "problem": bexample['problem'],
                    "solution": solution,  # Use fixed solution
                    "generated_reasoning": outputs[i].outputs[0].text,
                    "num_tokens": len(outputs[i].outputs[0].token_ids),
                    "method": f"baseline_{config.mode}",
                    "model": config.model,
                    "mode": config.mode,
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
        "mode": config.mode,
        "timestamp": time.time(),
        "completed": True
    }
    save_checkpoint(checkpoint_file, checkpoint_data)

    pbar.close()

    # Print summary
    elapsed = time.time() - start_time
    print(f"\n{'=' * 80}")
    print(f"✅ Completed!")
    print(f"   Mode: {config.mode}")
    print(f"   Slice: {config.slice_id} / {config.total_slices}")
    print(f"   Processed: {num_processed} examples")
    print(f"   Time: {elapsed:.1f}s ({num_processed/elapsed:.2f} examples/s)")
    print(f"   Output: {output_file}")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
