#!/usr/bin/env python3
"""
Heavy Benchmark: Long Sequences & Large Batch Sizes

Tests optimal sampling performance under production workloads:
- Long sequences (512, 1024, 2048 tokens)
- Large batch sizes (8, 16, 32, 64)
- Memory usage tracking
- Throughput and latency analysis
- Optimal Sampling vs Teacher-only comparison
"""

import time
import psutil
import torch
import json
from typing import List, Dict, Tuple
from optimal_sampling import OptimalSamplingV1
import numpy as np


def get_gpu_memory():
    """Get current GPU memory usage in GB"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**3
    return 0


def generate_long_prompts(num_prompts: int, complexity: str = "medium") -> List[str]:
    """Generate realistic math problems of varying complexity"""

    if complexity == "short":
        # Short problems (~50 tokens)
        templates = [
            "Solve for x: {a}x + {b} = {c}",
            "What is {a}% of {b}?",
            "Calculate: {a} × {b} + {c}",
            "Find the area of a rectangle with length {a} and width {b}.",
        ]
    elif complexity == "medium":
        # Medium problems (~150 tokens)
        templates = [
            "A train travels at {a} km/h for {b} hours, then at {c} km/h for {d} hours. "
            "What is the total distance traveled? Show your step-by-step calculation.",

            "In a class of {a} students, {b}% are boys. If {c} more girls join, "
            "what percentage of the class are girls now? Explain your reasoning.",

            "A store sells apples at ${a} each and oranges at ${b} each. "
            "If you buy {c} apples and {d} oranges, what is the total cost? "
            "Show the calculation process.",

            "The sum of three consecutive integers is {a}. What are the three numbers? "
            "Explain how you found them step by step.",
        ]
    else:  # long
        # Long problems (~300 tokens)
        templates = [
            "A company has {a} employees. Each employee works {b} hours per week at ${c} per hour. "
            "The company also pays ${d} per employee per month for benefits. "
            "If the company operates for {e} weeks, what is the total cost? "
            "Break down the calculation into salary costs and benefit costs, then sum them up. "
            "Show all intermediate steps clearly.",

            "A rectangular garden is {a} meters long and {b} meters wide. "
            "A path of {c} meters width surrounds the garden on all sides. "
            "First, calculate the area of the garden. "
            "Then, calculate the total area including the path. "
            "Finally, find the area of just the path. "
            "Show your work for each step and explain the geometric reasoning.",

            "In a school, there are {a} classes. Each class has {b} students on average. "
            "The school wants to organize a field trip where each bus can hold {c} students. "
            "Each bus costs ${d} to rent. If only {e}% of students attend, "
            "how many buses are needed and what is the total cost? "
            "Calculate: total students → attending students → buses needed → total cost. "
            "Round up buses to the nearest whole number.",

            "A water tank can hold {a} liters. It is currently {b}% full. "
            "Water is being added at {c} liters per minute and drained at {d} liters per minute. "
            "If this continues for {e} minutes, what percentage full will the tank be? "
            "First calculate current water, then net flow rate, then final water, then percentage. "
            "Show all calculations clearly.",
        ]

    prompts = []
    for i in range(num_prompts):
        template = templates[i % len(templates)]
        # Random parameters
        params = {
            'a': np.random.randint(2, 20),
            'b': np.random.randint(2, 20),
            'c': np.random.randint(2, 20),
            'd': np.random.randint(2, 20),
            'e': np.random.randint(2, 20),
        }
        prompts.append(template.format(**params))

    return prompts


def run_heavy_benchmark(
    sampler: OptimalSamplingV1,
    batch_size: int,
    max_tokens: int,
    num_batches: int = 3,
    use_optimal_sampling: bool = True
) -> Dict:
    """
    Run heavy benchmark with specified parameters

    Args:
        sampler: Initialized sampler
        batch_size: Number of prompts per batch
        max_tokens: Maximum tokens to generate
        num_batches: Number of batches to run
        use_optimal_sampling: Use optimal sampling or teacher-only

    Returns:
        Dictionary with benchmark results
    """
    mode = "Optimal Sampling" if use_optimal_sampling else "Teacher-only"

    # Determine complexity based on max_tokens
    if max_tokens <= 256:
        complexity = "short"
    elif max_tokens <= 512:
        complexity = "medium"
    else:
        complexity = "long"

    print(f"\n{'=' * 80}")
    print(f"Benchmark: {mode}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Max Tokens: {max_tokens}")
    print(f"  Num Batches: {num_batches}")
    print(f"  Complexity: {complexity}")
    print(f"{'=' * 80}")

    results = {
        "mode": mode,
        "batch_size": batch_size,
        "max_tokens": max_tokens,
        "num_batches": num_batches,
        "batches": [],
        "total_time": 0,
        "total_tokens": 0,
        "peak_memory_gb": 0,
    }

    for batch_idx in range(num_batches):
        print(f"\n[Batch {batch_idx + 1}/{num_batches}]")

        # Generate prompts
        prompts = generate_long_prompts(batch_size, complexity)

        # Measure memory before
        torch.cuda.reset_peak_memory_stats()
        mem_before = get_gpu_memory()

        # Measure time
        start_time = time.time()

        # Generate
        try:
            outputs = sampler.generate(
                prompts=prompts,
                max_tokens=max_tokens,
                temperature=0.8,
                use_optimal_sampling=use_optimal_sampling
            )

            end_time = time.time()
            elapsed = end_time - start_time

            # Measure memory after
            mem_after = get_gpu_memory()
            mem_peak = torch.cuda.max_memory_allocated() / 1024**3

            # Calculate metrics
            total_tokens = sum(outputs.num_tokens)
            throughput = total_tokens / elapsed
            latency_per_request = elapsed / batch_size
            avg_tokens_per_request = total_tokens / batch_size

            batch_result = {
                "batch_idx": batch_idx + 1,
                "elapsed_time": elapsed,
                "total_tokens": total_tokens,
                "throughput": throughput,
                "latency_per_request": latency_per_request,
                "avg_tokens_per_request": avg_tokens_per_request,
                "memory_before_gb": mem_before,
                "memory_after_gb": mem_after,
                "memory_peak_gb": mem_peak,
                "success": True,
            }

            results["batches"].append(batch_result)
            results["total_time"] += elapsed
            results["total_tokens"] += total_tokens
            results["peak_memory_gb"] = max(results["peak_memory_gb"], mem_peak)

            print(f"  Time: {elapsed:.2f}s")
            print(f"  Tokens: {total_tokens}")
            print(f"  Throughput: {throughput:.2f} tok/s")
            print(f"  Latency/req: {latency_per_request:.3f}s")
            print(f"  Avg tokens/req: {avg_tokens_per_request:.1f}")
            print(f"  Memory: {mem_before:.2f} → {mem_after:.2f} GB (peak: {mem_peak:.2f} GB)")

        except Exception as e:
            print(f"  ❌ Failed: {e}")
            batch_result = {
                "batch_idx": batch_idx + 1,
                "error": str(e),
                "success": False,
            }
            results["batches"].append(batch_result)
            continue

    # Calculate aggregate metrics
    successful_batches = [b for b in results["batches"] if b.get("success", False)]

    if successful_batches:
        results["avg_throughput"] = results["total_tokens"] / results["total_time"]
        results["avg_latency"] = results["total_time"] / (len(successful_batches) * batch_size)
        results["avg_tokens_per_request"] = results["total_tokens"] / (len(successful_batches) * batch_size)
        results["success_rate"] = len(successful_batches) / num_batches * 100
    else:
        results["avg_throughput"] = 0
        results["avg_latency"] = 0
        results["avg_tokens_per_request"] = 0
        results["success_rate"] = 0

    print(f"\n{'─' * 80}")
    print(f"AGGREGATE RESULTS: {mode}")
    print(f"{'─' * 80}")
    print(f"Total Time: {results['total_time']:.2f}s")
    print(f"Total Tokens: {results['total_tokens']}")
    print(f"Avg Throughput: {results['avg_throughput']:.2f} tok/s")
    print(f"Avg Latency: {results['avg_latency']:.3f} s/request")
    print(f"Avg Tokens/Request: {results['avg_tokens_per_request']:.1f}")
    print(f"Peak Memory: {results['peak_memory_gb']:.2f} GB")
    print(f"Success Rate: {results['success_rate']:.1f}%")
    print(f"{'─' * 80}")

    return results


def main():
    print("=" * 80)
    print("HEAVY BENCHMARK: Long Sequences & Large Batch Sizes")
    print("=" * 80)

    # Test configurations
    configs = [
        # (batch_size, max_tokens, num_batches)
        (8, 256, 2),      # Warm-up: Small batch, medium length
        (16, 512, 2),     # Medium: Medium batch, medium length
        (32, 512, 2),     # Heavy: Large batch, medium length
        (16, 1024, 2),    # Long: Medium batch, long sequence
        (8, 2048, 1),     # Very long: Small batch, very long sequence
    ]

    all_results = []

    # Initialize sampler (reuse for all tests)
    print("\n[Initialization] Loading models...")
    print("  (This will take ~30-60s for model loading and compilation)")

    sampler = OptimalSamplingV1(
        model_teacher="Qwen/Qwen2.5-3B-Instruct",
        model_theta="Qwen/Qwen2.5-1.5B-Instruct",
        alpha_method="kl_symmetry",
        gpu_memory_utilization=0.45,
        track_alpha_stats=False,  # Disable for performance
    )

    print("✅ Models loaded!")

    # Run benchmarks
    for config_idx, (batch_size, max_tokens, num_batches) in enumerate(configs):
        print(f"\n\n{'#' * 80}")
        print(f"CONFIG {config_idx + 1}/{len(configs)}")
        print(f"{'#' * 80}")

        # Test with Optimal Sampling
        optimal_results = run_heavy_benchmark(
            sampler,
            batch_size=batch_size,
            max_tokens=max_tokens,
            num_batches=num_batches,
            use_optimal_sampling=True
        )

        # Test with Teacher-only (baseline)
        baseline_results = run_heavy_benchmark(
            sampler,
            batch_size=batch_size,
            max_tokens=max_tokens,
            num_batches=num_batches,
            use_optimal_sampling=False
        )

        all_results.append({
            "config": {
                "batch_size": batch_size,
                "max_tokens": max_tokens,
                "num_batches": num_batches,
            },
            "optimal_sampling": optimal_results,
            "teacher_only": baseline_results,
        })

    # Summary comparison
    print("\n\n" + "=" * 80)
    print("SUMMARY: Optimal Sampling vs Teacher-only")
    print("=" * 80)
    print(f"{'Config':<25} {'Metric':<20} {'Optimal':<15} {'Baseline':<15} {'Ratio':<10}")
    print("─" * 80)

    for result in all_results:
        config = result["config"]
        optimal = result["optimal_sampling"]
        baseline = result["teacher_only"]

        config_str = f"BS={config['batch_size']}, MT={config['max_tokens']}"

        # Throughput
        if baseline["avg_throughput"] > 0:
            throughput_ratio = optimal["avg_throughput"] / baseline["avg_throughput"]
            print(f"{config_str:<25} {'Throughput (tok/s)':<20} "
                  f"{optimal['avg_throughput']:>13.2f} {baseline['avg_throughput']:>13.2f} "
                  f"{throughput_ratio:>8.2f}x")

        # Latency
        if baseline["avg_latency"] > 0:
            latency_ratio = optimal["avg_latency"] / baseline["avg_latency"]
            print(f"{'':<25} {'Latency (s/req)':<20} "
                  f"{optimal['avg_latency']:>13.3f} {baseline['avg_latency']:>13.3f} "
                  f"{latency_ratio:>8.2f}x")

        # Memory
        print(f"{'':<25} {'Peak Memory (GB)':<20} "
              f"{optimal['peak_memory_gb']:>13.2f} {baseline['peak_memory_gb']:>13.2f} "
              f"{'':>10}")

        print("─" * 80)

    # Save results
    output_file = "heavy_benchmark_results.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n✅ Results saved to: {output_file}")

    # Performance recommendations
    print("\n" + "=" * 80)
    print("PERFORMANCE RECOMMENDATIONS")
    print("=" * 80)

    # Analyze results
    best_throughput = max(r["optimal_sampling"]["avg_throughput"] for r in all_results)
    best_config = next(r for r in all_results
                      if r["optimal_sampling"]["avg_throughput"] == best_throughput)

    print(f"\n[Best Throughput Configuration]")
    print(f"  Batch Size: {best_config['config']['batch_size']}")
    print(f"  Max Tokens: {best_config['config']['max_tokens']}")
    print(f"  Throughput: {best_throughput:.2f} tok/s")

    # Memory efficiency
    memory_efficient = min(all_results,
                          key=lambda r: r["optimal_sampling"]["peak_memory_gb"])

    print(f"\n[Most Memory Efficient]")
    print(f"  Batch Size: {memory_efficient['config']['batch_size']}")
    print(f"  Max Tokens: {memory_efficient['config']['max_tokens']}")
    print(f"  Peak Memory: {memory_efficient['optimal_sampling']['peak_memory_gb']:.2f} GB")

    # Practical recommendations
    print(f"\n[General Recommendations]")
    print(f"  • For production: Use batch size 16-32 with max_tokens 512")
    print(f"  • For quality: Use max_tokens ≥ 512 for detailed reasoning")
    print(f"  • For memory: Keep batch_size × max_tokens < 16384")
    print(f"  • Expect ~20-50x slower than teacher-only (normal for dual models)")

    print("\n" + "=" * 80)
    print("✅ HEAVY BENCHMARK COMPLETED!")
    print("=" * 80)


if __name__ == '__main__':
    main()
