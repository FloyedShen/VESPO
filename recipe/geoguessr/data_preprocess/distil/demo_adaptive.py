#!/usr/bin/env python3
"""
Adaptive sampling with heuristic rules.

Features:
- Dynamically learn relationship between locatability_score and acc@25km
- Target-oriented sampling (e.g., sample to achieve 60% acc@25km)
- Prioritize most challenging samples
- Balance exploration and exploitation
"""

import os
import sys
import json
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import datasets
from tqdm import tqdm

# Add parent path
sys.path.insert(0, str(Path(__file__).parent))
from demo_concurrent import (
    process_single_sample,
    call_qwen_api,
    encode_image_to_base64,
    SYSTEM_PROMPT
)


# ============================================================================
# Adaptive Sampler with Heuristic Rules
# ============================================================================

class AdaptiveSampler:
    """
    Adaptive sampler that learns the relationship between locatability_score
    and accuracy, then samples the most challenging examples to meet a target.
    """

    def __init__(
        self,
        target_acc_at_25km: float = 0.6,
        exploration_rate: float = 0.2,
        score_bins: int = 20,
        warmup_samples: int = 50
    ):
        """
        Args:
            target_acc_at_25km: Target accuracy @25km (e.g., 0.6 = 60%)
            exploration_rate: Probability of exploring unknown regions
            score_bins: Number of bins to discretize locatability_score
            warmup_samples: Number of random samples for initial estimation
        """
        self.target_acc = target_acc_at_25km
        self.exploration_rate = exploration_rate
        self.score_bins = score_bins
        self.warmup_samples = warmup_samples

        # Statistics: score_bin -> {count, acc_sum, acc_samples}
        self.bin_stats = defaultdict(lambda: {
            'count': 0,
            'acc_sum': 0.0,
            'acc_samples': []
        })

        # Score ranges for each bin
        self.score_min = 0.0
        self.score_max = 1.0
        self.bin_edges = np.linspace(self.score_min, self.score_max, self.score_bins + 1)

        # Sampling weights for each bin
        self.bin_weights = np.ones(self.score_bins) / self.score_bins

        self.total_samples = 0
        self.warmup_done = False

    def get_bin_index(self, score: float) -> int:
        """Get bin index for a given locatability_score."""
        if score <= self.score_min:
            return 0
        if score >= self.score_max:
            return self.score_bins - 1
        return int((score - self.score_min) / (self.score_max - self.score_min) * self.score_bins)

    def get_bin_center(self, bin_idx: int) -> float:
        """Get center value of a bin."""
        return (self.bin_edges[bin_idx] + self.bin_edges[bin_idx + 1]) / 2

    def update_statistics(self, score: float, acc_at_25km: float):
        """Update statistics with a new sample."""
        bin_idx = self.get_bin_index(score)
        stats = self.bin_stats[bin_idx]

        stats['count'] += 1
        stats['acc_sum'] += acc_at_25km
        stats['acc_samples'].append(acc_at_25km)

        self.total_samples += 1

        # Check if warmup is done
        if self.total_samples >= self.warmup_samples and not self.warmup_done:
            self.warmup_done = True
            print(f"\n[Sampler] Warmup complete with {self.total_samples} samples. Switching to adaptive mode.")
            self._update_sampling_weights()

    def get_bin_mean_acc(self, bin_idx: int) -> Optional[float]:
        """Get mean accuracy for a bin."""
        stats = self.bin_stats[bin_idx]
        if stats['count'] == 0:
            return None
        return stats['acc_sum'] / stats['count']

    def get_bin_confidence(self, bin_idx: int) -> float:
        """
        Get confidence (inverse of uncertainty) for a bin.
        More samples = higher confidence.
        """
        stats = self.bin_stats[bin_idx]
        # Use sqrt(n) as confidence measure (UCB-like)
        return np.sqrt(stats['count']) if stats['count'] > 0 else 0.0

    def _update_sampling_weights(self):
        """
        Update sampling weights based on:
        1. Distance to target accuracy (closer = higher weight)
        2. Uncertainty (higher uncertainty = higher exploration weight)
        3. Challenge level (medium difficulty = higher weight)
        """
        if not self.warmup_done:
            # During warmup, use uniform sampling
            self.bin_weights = np.ones(self.score_bins) / self.score_bins
            return

        weights = np.zeros(self.score_bins)

        for bin_idx in range(self.score_bins):
            # Component 1: Distance to target
            mean_acc = self.get_bin_mean_acc(bin_idx)
            if mean_acc is None:
                # Unknown region - high exploration weight
                target_weight = 1.0
            else:
                # Closer to target = higher weight
                # Use Gaussian-like function centered at target
                distance = abs(mean_acc - self.target_acc)
                target_weight = np.exp(-5 * distance)  # Sharp peak at target

            # Component 2: Uncertainty bonus (exploration)
            confidence = self.get_bin_confidence(bin_idx)
            uncertainty_weight = 1.0 / (1.0 + confidence)  # Higher weight for uncertain bins

            # Component 3: Challenge level
            # We want samples that are challenging but not impossible
            # Prefer medium locatability_scores (around 0.3-0.6)
            score = self.get_bin_center(bin_idx)
            if 0.2 <= score <= 0.7:
                challenge_weight = 1.5  # Boost for challenging range
            elif 0.1 <= score <= 0.8:
                challenge_weight = 1.0
            else:
                challenge_weight = 0.5  # Lower weight for too easy/hard

            # Combine components
            weight = (
                target_weight * 0.6 +
                uncertainty_weight * 0.2 +
                challenge_weight * 0.2
            )

            weights[bin_idx] = weight

        # Normalize
        weights_sum = weights.sum()
        if weights_sum > 0:
            self.bin_weights = weights / weights_sum
        else:
            self.bin_weights = np.ones(self.score_bins) / self.score_bins

    def sample_indices(
        self,
        dataset: datasets.Dataset,
        num_samples: int,
        score_key: str = 'locatability_score'
    ) -> List[int]:
        """
        Sample indices from dataset based on adaptive strategy.

        Args:
            dataset: HuggingFace dataset with locatability_score
            num_samples: Number of samples to draw
            score_key: Column name for locatability_score

        Returns:
            List of sampled indices
        """
        # Get scores for all samples
        all_scores = np.array([sample[score_key] for sample in dataset])
        n_total = len(all_scores)

        # Assign each sample to a bin
        sample_bins = np.array([self.get_bin_index(s) for s in all_scores])

        # Exploration: randomly sample some bins
        if np.random.random() < self.exploration_rate or not self.warmup_done:
            # Uniform sampling during exploration
            sampled_indices = np.random.choice(n_total, size=num_samples, replace=False)
        else:
            # Exploitation: sample based on weights
            # First sample bins according to weights
            sampled_bins = np.random.choice(
                self.score_bins,
                size=num_samples,
                p=self.bin_weights,
                replace=True
            )

            # Then sample indices from selected bins
            sampled_indices = []
            for target_bin in sampled_bins:
                # Get all indices in this bin
                bin_indices = np.where(sample_bins == target_bin)[0]

                if len(bin_indices) > 0:
                    # Sample one from this bin
                    idx = np.random.choice(bin_indices)
                    sampled_indices.append(idx)
                else:
                    # Bin is empty, sample randomly
                    idx = np.random.choice(n_total)
                    sampled_indices.append(idx)

            sampled_indices = np.array(sampled_indices)

        return sampled_indices.tolist()

    def print_statistics(self):
        """Print current statistics."""
        print("\n" + "=" * 80)
        print("Adaptive Sampler Statistics")
        print("=" * 80)

        print(f"Total samples: {self.total_samples}")
        print(f"Target acc@25km: {self.target_acc:.1%}")
        print(f"Warmup done: {self.warmup_done}")
        print(f"\nScore Range -> Mean Acc@25km (Count)")
        print("-" * 80)

        for bin_idx in range(self.score_bins):
            score_start = self.bin_edges[bin_idx]
            score_end = self.bin_edges[bin_idx + 1]
            mean_acc = self.get_bin_mean_acc(bin_idx)
            count = self.bin_stats[bin_idx]['count']
            weight = self.bin_weights[bin_idx]

            if count > 0:
                acc_str = f"{mean_acc:.1%}"
            else:
                acc_str = "N/A"

            print(f"  [{score_start:.2f}, {score_end:.2f}): {acc_str:>6} (n={count:3d}, w={weight:.3f})")

        print("=" * 80)

    def get_current_mean_acc(self) -> Optional[float]:
        """Get current overall mean acc@25km."""
        total_acc = 0.0
        total_count = 0

        for stats in self.bin_stats.values():
            if stats['count'] > 0:
                total_acc += stats['acc_sum']
                total_count += stats['count']

        if total_count == 0:
            return None

        return total_acc / total_count

    def get_target_gap(self) -> Optional[float]:
        """Get gap between current acc and target."""
        current_acc = self.get_current_mean_acc()
        if current_acc is None:
            return None
        return self.target_acc - current_acc


# ============================================================================
# Adaptive Distillation Pipeline
# ============================================================================

def run_adaptive_distillation(
    dataset_path: str,
    output_dir: str,
    total_samples: int = 1000,
    target_acc_at_25km: float = 0.6,
    batch_size: int = 50,
    max_workers: int = 4,
    max_turns: int = 10,
    temperature: float = 0.7,
    max_tokens: int = 2048,
    exploration_rate: float = 0.2
):
    """
    Run adaptive distillation with heuristic sampling.

    Args:
        dataset_path: Path to dataset
        output_dir: Output directory
        total_samples: Total number of samples to generate
        target_acc_at_25km: Target accuracy @25km (0.6 = 60%)
        batch_size: Samples per batch (for adaptive updates)
        max_workers: Concurrent workers
        max_turns: Max turns per sample
        temperature: Sampling temperature
        max_tokens: Max tokens per turn
        exploration_rate: Exploration rate (0.2 = 20% exploration)
    """
    print("=" * 80)
    print("Adaptive GeoGuessr Trace Distillation")
    print("=" * 80)
    print(f"Target acc@25km: {target_acc_at_25km:.1%}")
    print(f"Total samples: {total_samples}")
    print(f"Batch size: {batch_size}")
    print(f"Exploration rate: {exploration_rate:.1%}")
    print("=" * 80 + "\n")

    # Load dataset
    print(f"Loading dataset from: {dataset_path}")
    full_dataset = datasets.load_from_disk(dataset_path)
    print(f"Dataset loaded: {len(full_dataset)} samples\n")

    # Check for locatability_score
    if 'locatability_score' not in full_dataset.column_names:
        print("[ERROR] Dataset does not have 'locatability_score' column!")
        return

    # Initialize sampler
    sampler = AdaptiveSampler(
        target_acc_at_25km=target_acc_at_25km,
        exploration_rate=exploration_rate,
        warmup_samples=min(batch_size, total_samples // 10)
    )

    os.makedirs(output_dir, exist_ok=True)

    # Track all results
    all_results = {}
    generated_count = 0
    lock = threading.Lock()

    # Batch processing
    num_batches = (total_samples + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        remaining = total_samples - generated_count
        current_batch_size = min(batch_size, remaining)

        print(f"\n{'=' * 80}")
        print(f"Batch {batch_idx + 1}/{num_batches}: Sampling {current_batch_size} samples")
        print(f"{'=' * 80}")

        # Sample indices using adaptive strategy
        sampled_indices = sampler.sample_indices(full_dataset, current_batch_size)

        # Get samples
        batch_samples = [full_dataset[int(idx)] for idx in sampled_indices]

        # Print sampling info
        batch_scores = [s['locatability_score'] for s in batch_samples]
        print(f"Sampled scores - Min: {min(batch_scores):.3f}, "
              f"Max: {max(batch_scores):.3f}, "
              f"Mean: {np.mean(batch_scores):.3f}")

        # Process batch concurrently
        tasks = [
            (batch_samples[i], sampled_indices[i], dataset_path, max_turns, temperature, max_tokens)
            for i in range(current_batch_size)
        ]

        batch_results = {}
        pbar = tqdm(total=current_batch_size, desc=f"Batch {batch_idx + 1}")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(process_sample_wrapper, task): task[1]
                for task in tasks
            }

            for future in as_completed(future_to_idx):
                sample_idx = future_to_idx[future]

                try:
                    idx, trace = future.result()

                    with lock:
                        if trace is not None:
                            batch_results[idx] = trace

                            # Update sampler statistics
                            score = trace['sample_data']['locatability_score']
                            acc_at_25km = trace['reward_score'].get('acc@25km', 0.0)
                            sampler.update_statistics(score, acc_at_25km)

                            generated_count += 1

                        pbar.update(1)

                except Exception as e:
                    with lock:
                        tqdm.write(f"[ERROR] Sample {sample_idx}: {e}")
                        pbar.update(1)

        pbar.close()

        # Add to all results
        all_results.update(batch_results)

        # Update sampling weights after each batch
        sampler._update_sampling_weights()

        # Print current statistics
        sampler.print_statistics()

        current_mean_acc = sampler.get_current_mean_acc()
        target_gap = sampler.get_target_gap()

        if current_mean_acc is not None:
            print(f"\nCurrent mean acc@25km: {current_mean_acc:.1%}")
            if target_gap is not None:
                print(f"Gap to target: {target_gap:+.1%}")

    # Save all results
    print(f"\n{'=' * 80}")
    print("Saving results...")
    print(f"{'=' * 80}")

    for idx in sorted(all_results.keys()):
        trace = all_results[idx]
        output_path = os.path.join(output_dir, f"trace_{idx:05d}.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(trace, f, indent=2, ensure_ascii=False)

    # Final summary
    print(f"\n{'=' * 80}")
    print("Final Summary")
    print(f"{'=' * 80}")
    print(f"Total generated: {generated_count}/{total_samples}")

    final_mean_acc = sampler.get_current_mean_acc()
    if final_mean_acc is not None:
        print(f"Final mean acc@25km: {final_mean_acc:.1%}")
        print(f"Target acc@25km: {target_acc_at_25km:.1%}")
        print(f"Gap: {sampler.get_target_gap():+.1%}")

    # Calculate other metrics
    parse_success = sum(
        1 for trace in all_results.values()
        if trace['metadata']['parse_success']
    )
    print(f"Parse success: {parse_success}/{generated_count} ({parse_success/generated_count*100:.1f}%)")

    print(f"Output directory: {output_dir}")
    print(f"{'=' * 80}\n")

    # Save sampler statistics
    stats_path = os.path.join(output_dir, "sampler_statistics.json")
    with open(stats_path, 'w') as f:
        json.dump({
            'target_acc': target_acc_at_25km,
            'final_mean_acc': final_mean_acc,
            'total_samples': generated_count,
            'bin_stats': {
                f"bin_{i}": {
                    'score_range': [float(sampler.bin_edges[i]), float(sampler.bin_edges[i+1])],
                    'count': sampler.bin_stats[i]['count'],
                    'mean_acc': sampler.get_bin_mean_acc(i)
                }
                for i in range(sampler.score_bins)
            }
        }, f, indent=2)
    print(f"Sampler statistics saved to: {stats_path}")


def process_sample_wrapper(args):
    """Wrapper for processing."""
    sample, sample_index, dataset_path, max_turns, temperature, max_tokens = args
    return sample_index, process_single_sample(
        sample, sample_index, dataset_path, max_turns, temperature, max_tokens
    )


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Adaptive distillation with heuristic sampling"
    )

    parser.add_argument("--dataset_path", type=str,
        default="/mnt/tidal-alsh-hilab/usr/shenguobin/geoguessr/processed/gaea_wlp/train")
    parser.add_argument("--output_dir", type=str,
        default="/mnt/tidal-alsh-hilab/usr/shenguobin/verl/recipe/geoguessr/data_preprocess/distil/traces_adaptive")

    parser.add_argument("--total_samples", type=int, default=1000,
        help="Total number of samples to generate")
    parser.add_argument("--target_acc", type=float, default=0.6,
        help="Target accuracy @25km (0.6 = 60%%)")
    parser.add_argument("--batch_size", type=int, default=50,
        help="Samples per batch (for adaptive updates)")

    parser.add_argument("--max_workers", type=int, default=4)
    parser.add_argument("--max_turns", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_tokens", type=int, default=2048)

    parser.add_argument("--exploration_rate", type=float, default=0.2,
        help="Exploration rate (0.2 = 20%% exploration)")

    args = parser.parse_args()

    run_adaptive_distillation(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        total_samples=args.total_samples,
        target_acc_at_25km=args.target_acc,
        batch_size=args.batch_size,
        max_workers=args.max_workers,
        max_turns=args.max_turns,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        exploration_rate=args.exploration_rate
    )


if __name__ == "__main__":
    main()
