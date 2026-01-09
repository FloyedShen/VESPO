#!/usr/bin/env python3
"""
Production-ready distillation with checkpoint/resume support.

Features:
- Checkpoint/resume: Skip already processed samples
- Real-time save: Save each trace immediately upon completion
- Flexible configuration: Specify dataset, workers, sampling strategy
- Progress tracking: Detailed statistics and resumability
"""

import os
import sys
import json
import time
import fcntl
import threading
from typing import Dict, List, Optional, Set, Any
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import datasets
from tqdm import tqdm

# Add parent path
sys.path.insert(0, str(Path(__file__).parent))
from demo_concurrent import process_single_sample


# ============================================================================
# Checkpoint Manager
# ============================================================================

class CheckpointManager:
    """Manage checkpoint for resume support."""

    def __init__(self, checkpoint_path: str):
        """
        Args:
            checkpoint_path: Path to checkpoint file
        """
        self.checkpoint_path = checkpoint_path
        self.lock = threading.Lock()
        self.processed_indices: Set[int] = set()
        self.failed_indices: Set[int] = set()
        self.total_processed = 0
        self.total_failed = 0

        # Load existing checkpoint
        self._load_checkpoint()

    def _load_checkpoint(self):
        """Load checkpoint from disk."""
        if os.path.exists(self.checkpoint_path):
            try:
                with open(self.checkpoint_path, 'r') as f:
                    data = json.load(f)
                    self.processed_indices = set(data.get('processed_indices', []))
                    self.failed_indices = set(data.get('failed_indices', []))
                    self.total_processed = len(self.processed_indices)
                    self.total_failed = len(self.failed_indices)
                    print(f"[Checkpoint] Loaded: {self.total_processed} processed, {self.total_failed} failed")
            except Exception as e:
                print(f"[Checkpoint] Failed to load: {e}")
                self.processed_indices = set()
                self.failed_indices = set()

    def _save_checkpoint(self):
        """Save checkpoint to disk."""
        try:
            checkpoint_data = {
                'processed_indices': list(self.processed_indices),
                'failed_indices': list(self.failed_indices),
                'total_processed': self.total_processed,
                'total_failed': self.total_failed,
                'timestamp': time.time()
            }

            # Atomic write
            temp_path = self.checkpoint_path + '.tmp'
            with open(temp_path, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
            os.rename(temp_path, self.checkpoint_path)

        except Exception as e:
            print(f"[Checkpoint] Failed to save: {e}")

    def mark_processed(self, index: int):
        """Mark an index as successfully processed."""
        with self.lock:
            self.processed_indices.add(index)
            self.total_processed = len(self.processed_indices)
            self._save_checkpoint()

    def mark_failed(self, index: int):
        """Mark an index as failed."""
        with self.lock:
            self.failed_indices.add(index)
            self.total_failed = len(self.failed_indices)
            self._save_checkpoint()

    def is_processed(self, index: int) -> bool:
        """Check if an index has been processed."""
        return index in self.processed_indices

    def is_failed(self, index: int) -> bool:
        """Check if an index has failed."""
        return index in self.failed_indices

    def get_remaining_indices(self, all_indices: List[int]) -> List[int]:
        """Get list of remaining indices to process."""
        return [idx for idx in all_indices if not self.is_processed(idx)]


# ============================================================================
# Real-time Trace Saver
# ============================================================================

class RealtimeTraceSaver:
    """Save traces immediately upon completion."""

    def __init__(self, output_dir: str):
        """
        Args:
            output_dir: Directory to save traces
        """
        self.output_dir = output_dir
        self.lock = threading.Lock()
        os.makedirs(output_dir, exist_ok=True)

    def save_trace(self, trace: Dict[str, Any], index: int) -> bool:
        """
        Save a trace immediately.

        Args:
            trace: Trace data
            index: Sample index

        Returns:
            True if successful, False otherwise
        """
        try:
            output_path = os.path.join(self.output_dir, f"trace_{index:05d}.json")

            # Use file lock to prevent concurrent writes
            with self.lock:
                with open(output_path, 'w', encoding='utf-8') as f:
                    # Acquire exclusive lock
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                    try:
                        json.dump(trace, f, indent=2, ensure_ascii=False)
                    finally:
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)

            return True

        except Exception as e:
            print(f"[Saver] Failed to save trace {index}: {e}")
            return False


# ============================================================================
# Production Distillation Pipeline
# ============================================================================

def run_production_distillation(
    dataset_path: str,
    output_dir: str,
    num_samples: int = 1000,
    max_workers: int = 4,
    max_turns: int = 10,
    temperature: float = 0.7,
    max_tokens: int = 2048,
    sampling_strategy: str = "random",
    target_acc: float = 0.6,
    resume: bool = True
):
    """
    Run production distillation with checkpoint and real-time save.

    Args:
        dataset_path: Path to dataset
        output_dir: Output directory
        num_samples: Number of samples to generate
        max_workers: Number of concurrent workers
        max_turns: Max turns per sample
        temperature: Sampling temperature
        max_tokens: Max tokens per turn
        sampling_strategy: "random" or "adaptive" or "hardest"
        target_acc: Target acc@25km (for adaptive sampling)
        resume: Whether to resume from checkpoint
    """
    print("=" * 80)
    print("Production GeoGuessr Trace Distillation")
    print("=" * 80)
    print(f"Dataset: {dataset_path}")
    print(f"Output: {output_dir}")
    print(f"Samples: {num_samples}")
    print(f"Workers: {max_workers}")
    print(f"Strategy: {sampling_strategy}")
    print(f"Resume: {resume}")
    print("=" * 80 + "\n")

    # Initialize managers
    checkpoint_path = os.path.join(output_dir, "checkpoint.json")
    checkpoint_mgr = CheckpointManager(checkpoint_path)
    trace_saver = RealtimeTraceSaver(output_dir)

    # Load dataset
    print(f"Loading dataset...")
    full_dataset = datasets.load_from_disk(dataset_path)
    print(f"Dataset loaded: {len(full_dataset)} samples")

    # Select indices based on strategy
    all_indices = list(range(len(full_dataset)))

    if sampling_strategy == "hardest":
        # Sort by locatability_score (descending = hardest first)
        if 'locatability_score' in full_dataset.column_names:
            scores = [(i, full_dataset[i]['locatability_score']) for i in all_indices]
            scores.sort(key=lambda x: x[1], reverse=True)
            all_indices = [i for i, _ in scores]
            print(f"Sorted by difficulty (hardest first)")
        else:
            print(f"[WARNING] No locatability_score, using random order")

    elif sampling_strategy == "easiest":
        # Sort by locatability_score (ascending = easiest first)
        if 'locatability_score' in full_dataset.column_names:
            scores = [(i, full_dataset[i]['locatability_score']) for i in all_indices]
            scores.sort(key=lambda x: x[1])
            all_indices = [i for i, _ in scores]
            print(f"Sorted by difficulty (easiest first)")

    elif sampling_strategy == "random":
        # Random shuffle
        import random
        random.shuffle(all_indices)
        print(f"Random order")

    # Limit to num_samples
    if num_samples < len(all_indices) and num_samples != -1:
        all_indices = all_indices[:num_samples]

    # Filter out already processed
    if resume:
        remaining_indices = checkpoint_mgr.get_remaining_indices(all_indices)
        print(f"\nResume from checkpoint:")
        print(f"  Already processed: {checkpoint_mgr.total_processed}")
        print(f"  Already failed: {checkpoint_mgr.total_failed}")
        print(f"  Remaining: {len(remaining_indices)}/{len(all_indices)}")
    else:
        remaining_indices = all_indices
        print(f"\nStarting fresh (no resume)")

    if len(remaining_indices) == 0:
        print("\n✅ All samples already processed!")
        return

    print(f"\n{'=' * 80}")
    print(f"Processing {len(remaining_indices)} samples with {max_workers} workers")
    print(f"{'=' * 80}\n")

    # Statistics
    success_count = 0
    failed_count = 0
    lock = threading.Lock()

    # Process with progress bar
    pbar = tqdm(total=len(remaining_indices), desc="Generating traces")

    def process_and_save(sample_index: int):
        """Process a single sample and save immediately."""
        nonlocal success_count, failed_count

        try:
            # Get sample
            sample = full_dataset[int(sample_index)]

            # Process
            trace = process_single_sample(
                sample,
                sample_index,
                dataset_path,
                max_turns,
                temperature,
                max_tokens
            )

            if trace is not None:
                # Save immediately
                if trace_saver.save_trace(trace, sample_index):
                    # Mark as processed
                    checkpoint_mgr.mark_processed(sample_index)

                    with lock:
                        success_count += 1

                    # Log
                    reward = trace['reward_score']
                    tqdm.write(
                        f"[✓] Sample {sample_index}: "
                        f"Score={reward.get('score', 0):.4f}, "
                        f"Distance={reward.get('distance@km', 0):.2f}km"
                    )
                else:
                    # Save failed
                    checkpoint_mgr.mark_failed(sample_index)
                    with lock:
                        failed_count += 1
                    tqdm.write(f"[✗] Sample {sample_index}: Failed to save")
            else:
                # Processing failed
                checkpoint_mgr.mark_failed(sample_index)
                with lock:
                    failed_count += 1
                tqdm.write(f"[✗] Sample {sample_index}: Processing failed")

        except Exception as e:
            checkpoint_mgr.mark_failed(sample_index)
            with lock:
                failed_count += 1
            tqdm.write(f"[✗] Sample {sample_index}: Exception - {e}")

        finally:
            pbar.update(1)

    # Execute with thread pool
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = {
            executor.submit(process_and_save, idx): idx
            for idx in remaining_indices
        }

        # Wait for completion
        for future in as_completed(futures):
            pass  # Progress is tracked in process_and_save

    pbar.close()

    # Final summary
    print(f"\n{'=' * 80}")
    print("Final Summary")
    print(f"{'=' * 80}")
    print(f"Total requested: {num_samples}")
    print(f"Already processed (before): {checkpoint_mgr.total_processed - success_count}")
    print(f"Newly processed: {success_count}")
    print(f"Total processed: {checkpoint_mgr.total_processed}")
    print(f"Failed: {checkpoint_mgr.total_failed}")
    print(f"Remaining: {len(all_indices) - checkpoint_mgr.total_processed - checkpoint_mgr.total_failed}")
    print(f"Output directory: {output_dir}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"{'=' * 80}\n")

    # Calculate statistics
    if checkpoint_mgr.total_processed > 0:
        print("Loading traces for statistics...")
        trace_files = [
            os.path.join(output_dir, f"trace_{idx:05d}.json")
            for idx in checkpoint_mgr.processed_indices
            if os.path.exists(os.path.join(output_dir, f"trace_{idx:05d}.json"))
        ]

        if trace_files:
            parse_success = 0
            total_distance = 0
            distances = []

            for trace_file in trace_files[:100]:  # Sample for speed
                try:
                    with open(trace_file) as f:
                        trace = json.load(f)
                    reward = trace['reward_score']
                    if reward.get('parse_success'):
                        parse_success += 1
                        dist = reward.get('distance@km', 0)
                        total_distance += dist
                        distances.append(dist)
                except:
                    pass

            if parse_success > 0:
                print(f"\nStatistics (sampled from {len(trace_files[:100])} traces):")
                print(f"  Parse success rate: {parse_success}/{len(trace_files[:100])} ({parse_success/len(trace_files[:100])*100:.1f}%)")
                print(f"  Average distance: {total_distance/parse_success:.2f} km")
                if len(distances) > 0:
                    distances.sort()
                    print(f"  Median distance: {distances[len(distances)//2]:.2f} km")


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Production distillation with checkpoint/resume"
    )

    parser.add_argument("--dataset_path", type=str, required=True,
        help="Path to dataset")
    parser.add_argument("--output_dir", type=str, required=True,
        help="Output directory (will resume from here if exists)")

    parser.add_argument("--num_samples", type=int, default=-1,
        help="Number of samples to generate (default: 1000)")
    parser.add_argument("--max_workers", type=int, default=4,
        help="Number of concurrent workers (default: 4)")

    parser.add_argument("--max_turns", type=int, default=10,
        help="Max turns per sample (default: 10)")
    parser.add_argument("--temperature", type=float, default=0.7,
        help="Sampling temperature (default: 0.7)")
    parser.add_argument("--max_tokens", type=int, default=16384,
        help="Max tokens per turn (default: 16384)")

    parser.add_argument("--sampling_strategy", type=str,
        choices=["random", "hardest", "easiest"],
        default="random",
        help="Sampling strategy: random, hardest (by locatability_score), or easiest")

    parser.add_argument("--no_resume", action="store_true",
        help="Don't resume from checkpoint (start fresh)")

    args = parser.parse_args()

    run_production_distillation(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        max_workers=args.max_workers,
        max_turns=args.max_turns,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        sampling_strategy=args.sampling_strategy,
        resume=not args.no_resume
    )


if __name__ == "__main__":
    main()
