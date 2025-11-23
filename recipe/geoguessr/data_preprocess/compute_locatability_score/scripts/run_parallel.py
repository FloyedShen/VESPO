#!/usr/bin/env python3
"""
Multi-GPU parallel coordinator for computing locatability scores.

This script:
1. Detects available GPUs
2. Spawns multiple processes to process shards in parallel
3. Merges results and creates final dataset
"""

import os
import sys
import argparse
import subprocess
import pickle
import time
from pathlib import Path
from typing import List, Dict

import torch
import datasets


def detect_gpus() -> int:
    """Detect number of available GPUs."""
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available, using CPU")
        return 1

    n_gpus = torch.cuda.device_count()
    print(f"Detected {n_gpus} GPUs:")
    for i in range(n_gpus):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name} ({props.total_memory / 1024**3:.1f} GB)")

    return n_gpus


def launch_shard_workers(
    input_dataset: str,
    output_dir: str,
    n_gpus: int,
    batch_size: int = 20,
    num_workers: int = 2,
    model_id: str = "facebook/mask2former-swin-large-ade-semantic",
    no_amp: bool = False,
    checkpoint_interval: int = 100,
    no_resume: bool = False
) -> List[subprocess.Popen]:
    """
    Launch worker processes for each GPU.

    Returns:
        List of subprocess handles
    """
    script_dir = Path(__file__).parent
    worker_script = script_dir / "process_shard.py"

    if not worker_script.exists():
        raise FileNotFoundError(f"Worker script not found: {worker_script}")

    processes = []

    for shard_id in range(n_gpus):
        gpu_id = shard_id

        cmd = [
            "python3",
            str(worker_script),
            "--input_dataset", input_dataset,
            "--shard_id", str(shard_id),
            "--total_shards", str(n_gpus),
            "--output_dir", output_dir,
            "--gpu_id", str(gpu_id),
            "--batch_size", str(batch_size),
            "--num_workers", str(num_workers),
            "--model_id", model_id,
            "--checkpoint_interval", str(checkpoint_interval),
        ]

        if no_amp:
            cmd.append("--no_amp")
        if no_resume:
            cmd.append("--no_resume")

        print(f"\nLaunching worker for Shard {shard_id} on GPU {gpu_id}")
        print(f"Command: {' '.join(cmd)}")

        # Create log file
        log_file = open(os.path.join(output_dir, f"shard_{shard_id}.log"), "w")

        proc = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT
        )

        processes.append((proc, log_file, shard_id))
        time.sleep(2)  # Stagger launches

    return processes


def wait_for_workers(processes: List) -> bool:
    """
    Wait for all worker processes to complete.

    Returns:
        True if all succeeded, False otherwise
    """
    print(f"\nWaiting for {len(processes)} workers to complete...")
    print("You can monitor progress in the log files:")
    for _, _, shard_id in processes:
        print(f"  tail -f <output_dir>/shard_{shard_id}.log")

    all_success = True

    for proc, log_file, shard_id in processes:
        returncode = proc.wait()
        log_file.close()

        if returncode != 0:
            print(f"✗ Shard {shard_id} failed with return code {returncode}")
            all_success = False
        else:
            print(f"✓ Shard {shard_id} completed successfully")

    return all_success


def merge_shards(
    input_dataset: str,
    output_dataset: str,
    temp_dir: str,
    n_shards: int
):
    """
    Merge shard results and create final dataset with locatability scores.

    Args:
        input_dataset: Path to input dataset
        output_dataset: Path to save output dataset
        temp_dir: Directory with shard results
        n_shards: Number of shards
    """
    print(f"\nMerging {n_shards} shards...")

    # Load original dataset
    print(f"Loading original dataset from: {input_dataset}")
    dataset = datasets.load_from_disk(input_dataset)
    total_samples = len(dataset)
    print(f"Original dataset size: {total_samples}")

    # Load all shard results
    all_scores = {}
    all_mappings = {}

    for shard_id in range(n_shards):
        shard_file = os.path.join(temp_dir, f"shard_{shard_id}.pkl")

        if not os.path.exists(shard_file):
            raise FileNotFoundError(f"Shard file not found: {shard_file}")

        print(f"Loading shard {shard_id}...")
        with open(shard_file, 'rb') as f:
            shard_data = pickle.load(f)

        all_scores.update(shard_data['scores'])
        all_mappings.update(shard_data['mappings'])

    print(f"Loaded scores for {len(all_scores)} samples")

    # Create ordered lists
    locatability_scores = []
    class_mappings = []

    for i in range(total_samples):
        locatability_scores.append(all_scores.get(i, 0.0))
        class_mappings.append(all_mappings.get(i, "{}"))

    # Add columns to dataset
    print("Adding locatability scores to dataset...")
    dataset = dataset.add_column("locatability_score", locatability_scores)
    dataset = dataset.add_column("class_mapping", class_mappings)

    # Save final dataset
    print(f"Saving final dataset to: {output_dataset}")
    os.makedirs(os.path.dirname(output_dataset), exist_ok=True)
    dataset.save_to_disk(output_dataset)

    # Statistics
    valid_scores = [s for s in locatability_scores if s > 0]
    print(f"\n{'='*60}")
    print(f"Final Dataset Statistics:")
    print(f"  Total samples: {total_samples}")
    print(f"  Valid scores: {len(valid_scores)}")
    print(f"  Failed/Invalid: {total_samples - len(valid_scores)}")
    if valid_scores:
        print(f"  Score range: [{min(valid_scores):.4f}, {max(valid_scores):.4f}]")
        print(f"  Mean score: {sum(valid_scores)/len(valid_scores):.4f}")
        print(f"  Median score: {sorted(valid_scores)[len(valid_scores)//2]:.4f}")
    print(f"{'='*60}")

    print(f"\n✓ Dataset saved successfully!")
    print(f"✓ Output: {output_dataset}")


def main():
    parser = argparse.ArgumentParser(
        description="Multi-GPU parallel locatability score computation",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        "--input_dataset",
        type=str,
        required=True,
        help="Path to input HuggingFace dataset"
    )

    parser.add_argument(
        "--output_dataset",
        type=str,
        required=True,
        help="Path to save output dataset with scores"
    )

    parser.add_argument(
        "--temp_dir",
        type=str,
        default=None,
        help="Temporary directory for intermediate results (default: output_dataset + '_temp')"
    )

    parser.add_argument(
        "--n_gpus",
        type=int,
        default=None,
        help="Number of GPUs to use (default: auto-detect)"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=20,
        help="Batch size per GPU"
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="Number of dataloader workers per GPU"
    )

    parser.add_argument(
        "--model_id",
        type=str,
        default="facebook/mask2former-swin-large-ade-semantic",
        help="HuggingFace model ID"
    )

    parser.add_argument(
        "--no_amp",
        action="store_true",
        help="Disable automatic mixed precision"
    )

    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=100,
        help="Save checkpoint every N batches"
    )

    parser.add_argument(
        "--no_resume",
        action="store_true",
        help="Don't resume from checkpoint"
    )

    parser.add_argument(
        "--skip_merge",
        action="store_true",
        help="Skip merging step (useful for debugging)"
    )

    args = parser.parse_args()

    # Setup temp directory
    if args.temp_dir is None:
        args.temp_dir = args.output_dataset + "_temp"

    os.makedirs(args.temp_dir, exist_ok=True)
    print(f"Temporary directory: {args.temp_dir}")

    # Detect GPUs
    if args.n_gpus is None:
        args.n_gpus = detect_gpus()
    else:
        print(f"Using {args.n_gpus} GPUs (manual override)")

    # Launch workers
    print(f"\n{'='*60}")
    print(f"Starting parallel processing with {args.n_gpus} workers")
    print(f"Input dataset: {args.input_dataset}")
    print(f"Output dataset: {args.output_dataset}")
    print(f"{'='*60}")

    processes = launch_shard_workers(
        input_dataset=args.input_dataset,
        output_dir=args.temp_dir,
        n_gpus=args.n_gpus,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        model_id=args.model_id,
        no_amp=args.no_amp,
        checkpoint_interval=args.checkpoint_interval,
        no_resume=args.no_resume
    )

    # Wait for completion
    success = wait_for_workers(processes)

    if not success:
        print("\n✗ Some workers failed. Check log files for details.")
        sys.exit(1)

    # Merge results
    if not args.skip_merge:
        try:
            merge_shards(
                input_dataset=args.input_dataset,
                output_dataset=args.output_dataset,
                temp_dir=args.temp_dir,
                n_shards=args.n_gpus
            )
        except Exception as e:
            print(f"\n✗ Failed to merge results: {e}")
            sys.exit(1)
    else:
        print("\nSkipping merge step (--skip_merge)")

    print("\n✓ All done!")


if __name__ == "__main__":
    main()
