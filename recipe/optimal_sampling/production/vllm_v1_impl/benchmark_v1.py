"""
Benchmark for Optimal Sampling V1 with vLLM V1 Engine

Tests performance of optimal sampling with different configurations:
- Batch sizes
- With/without optimal sampling
- Different alpha methods
- Different sequence lengths
"""

import time
import sys
from typing import List, Dict
import numpy as np

sys.path.insert(0, '/diancpfs/user/guobin/verl/recipe/optimal_sampling')

from production.vllm_v1_impl import OptimalSamplingV1


class BenchmarkConfig:
    """Benchmark configuration"""
    def __init__(
        self,
        model_teacher: str = "Qwen/Qwen2.5-1.5B",
        model_theta: str = "Qwen/Qwen2.5-0.5B",
        batch_sizes: List[int] = [1, 2, 4, 8],
        max_tokens: List[int] = [50, 100],
        num_warmup: int = 2,
        num_runs: int = 5,
        gpu_memory_utilization: float = 0.4
    ):
        self.model_teacher = model_teacher
        self.model_theta = model_theta
        self.batch_sizes = batch_sizes
        self.max_tokens = max_tokens
        self.num_warmup = num_warmup
        self.num_runs = num_runs
        self.gpu_memory_utilization = gpu_memory_utilization


def generate_prompts(batch_size: int) -> List[str]:
    """Generate diverse test prompts"""
    base_prompts = [
        "What is artificial intelligence?",
        "Explain machine learning in simple terms.",
        "How does deep learning work?",
        "What are neural networks?",
        "Describe natural language processing.",
        "What is computer vision?",
        "Explain reinforcement learning.",
        "How do transformers work?",
    ]

    # Cycle through prompts to match batch size
    return [base_prompts[i % len(base_prompts)] for i in range(batch_size)]


def run_benchmark(
    sampler: OptimalSamplingV1,
    prompts: List[str],
    max_tokens: int,
    use_optimal_sampling: bool,
    num_runs: int
) -> Dict:
    """Run benchmark for a specific configuration"""
    latencies = []
    total_tokens = []

    for i in range(num_runs):
        start_time = time.time()

        outputs = sampler.generate(
            prompts=prompts,
            max_tokens=max_tokens,
            temperature=0.8,
            use_optimal_sampling=use_optimal_sampling
        )

        end_time = time.time()
        latency = end_time - start_time
        latencies.append(latency)

        # Count total generated tokens
        tokens = sum(outputs.num_tokens)
        total_tokens.append(tokens)

        print(f"[PROMPT] {prompts}\n\n[Response] {outputs.generated_texts}")

    # Compute statistics
    latencies = np.array(latencies)
    total_tokens = np.array(total_tokens)

    return {
        "latency_mean": float(np.mean(latencies)),
        "latency_std": float(np.std(latencies)),
        "latency_min": float(np.min(latencies)),
        "latency_max": float(np.max(latencies)),
        "tokens_mean": float(np.mean(total_tokens)),
        "throughput": float(np.mean(total_tokens) / np.mean(latencies)),  # tokens/sec
    }


def print_results(results: Dict, config: BenchmarkConfig):
    """Print benchmark results in a formatted table"""
    print("\n" + "=" * 100)
    print("BENCHMARK RESULTS")
    print("=" * 100)
    print(f"Teacher Model: {config.model_teacher}")
    print(f"Theta Model: {config.model_theta}")
    print(f"GPU Memory: {config.gpu_memory_utilization:.1%}")
    print(f"Warmup Runs: {config.num_warmup}")
    print(f"Benchmark Runs: {config.num_runs}")
    print("=" * 100)

    # Header
    print(f"\n{'Batch':<7} {'MaxTok':<8} {'Mode':<12} {'Latency(s)':<15} {'Tokens':<10} {'Throughput':<15}")
    print(f"{'Size':<7} {'ens':<8} {'':<12} {'Mean±Std':<15} {'Mean':<10} {'(tok/s)':<15}")
    print("-" * 100)

    # Print results
    for key, result in sorted(results.items()):
        batch_size, max_tokens, mode = key
        mode_str = "Optimal" if mode else "Baseline"

        latency_str = f"{result['latency_mean']:.2f}±{result['latency_std']:.2f}"
        tokens_str = f"{result['tokens_mean']:.1f}"
        throughput_str = f"{result['throughput']:.1f}"

        print(f"{batch_size:<7} {max_tokens:<8} {mode_str:<12} {latency_str:<15} {tokens_str:<10} {throughput_str:<15}")

    print("=" * 100)

    # Compute speedup statistics
    print("\n" + "=" * 100)
    print("SPEEDUP ANALYSIS (Optimal vs Baseline)")
    print("=" * 100)
    print(f"{'Batch':<7} {'MaxTok':<8} {'Baseline':<15} {'Optimal':<15} {'Speedup':<10}")
    print(f"{'Size':<7} {'ens':<8} {'(tok/s)':<15} {'(tok/s)':<15} {'Ratio':<10}")
    print("-" * 100)

    for batch_size in config.batch_sizes:
        for max_tokens in config.max_tokens:
            baseline_key = (batch_size, max_tokens, False)
            optimal_key = (batch_size, max_tokens, True)

            if baseline_key in results and optimal_key in results:
                baseline_throughput = results[baseline_key]['throughput']
                optimal_throughput = results[optimal_key]['throughput']
                speedup = optimal_throughput / baseline_throughput

                print(f"{batch_size:<7} {max_tokens:<8} {baseline_throughput:<15.1f} {optimal_throughput:<15.1f} {speedup:<10.2f}x")

    print("=" * 100)


def main():
    """Main benchmark function"""
    print("=" * 100)
    print("🚀 Optimal Sampling V1 Benchmark")
    print("=" * 100)

    # Configuration
    config = BenchmarkConfig(
        model_teacher="Qwen/Qwen3-8B",
        model_theta="Qwen/Qwen3-0.6B",
        batch_sizes=[1, 4, 16, 128],
        max_tokens=[512, 1024, 4096, 8192],
        num_warmup=0,
        num_runs=1,
        gpu_memory_utilization=0.4
    )

    print(f"\n📦 Initializing sampler...")
    sampler = OptimalSamplingV1(
        model_teacher=config.model_teacher,
        model_theta=config.model_theta,
        alpha_method="kl_symmetry",
        gpu_memory_utilization=config.gpu_memory_utilization
    )

    # Store results
    results = {}

    # Run benchmarks
    total_tests = len(config.batch_sizes) * len(config.max_tokens) * 2  # 2 modes
    current_test = 0

    for batch_size in config.batch_sizes:
        for max_tokens in config.max_tokens:
            for use_optimal in [False, True]:
                current_test += 1
                mode = "Optimal" if use_optimal else "Baseline"

                print(f"\n[{current_test}/{total_tests}] Testing: Batch={batch_size}, MaxTokens={max_tokens}, Mode={mode}")

                # Generate prompts
                prompts = generate_prompts(batch_size)

                # Warmup
                print(f"  Warming up ({config.num_warmup} runs)...")
                for _ in range(config.num_warmup):
                    sampler.generate(
                        prompts=prompts,
                        max_tokens=max_tokens,
                        temperature=0.8,
                        use_optimal_sampling=use_optimal
                    )

                # Benchmark
                print(f"  Running benchmark ({config.num_runs} runs)...")
                result = run_benchmark(
                    sampler=sampler,
                    prompts=prompts,
                    max_tokens=max_tokens,
                    use_optimal_sampling=use_optimal,
                    num_runs=config.num_runs
                )

                results[(batch_size, max_tokens, use_optimal)] = result
                print(f"  ✅ Throughput: {result['throughput']:.1f} tokens/s")

    # Print results
    print_results(results, config)

    print("\n✅ Benchmark completed!")


if __name__ == "__main__":
    main()
