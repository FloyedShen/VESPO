#!/usr/bin/env python3
"""
Benchmark example for Optimal Sampling HF

This example demonstrates:
1. Performance testing with different configurations
2. Comparing different alpha methods
3. Measuring throughput and latency
"""

import torch
import time
from optimal_sampling_hf import OptimalSamplingModel

def benchmark_config(config_name, model_theta, model_t, alpha_method, num_examples=10, max_tokens=128):
    """Benchmark a specific configuration"""
    print(f"\n{'='*60}")
    print(f"Benchmarking: {config_name}")
    print(f"{'='*60}")

    # Initialize
    start_init = time.time()
    model = OptimalSamplingModel(
        model_theta_path=model_theta,
        model_t_path=model_t,
        alpha_method=alpha_method,
        dtype=torch.bfloat16,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    init_time = time.time() - start_init
    print(f"Initialization time: {init_time:.2f}s")

    # Prepare prompts
    prompts = [f"Question {i}: What is machine learning?" for i in range(num_examples)]

    # Warmup
    _ = model.generate(prompts=prompts[:2], max_new_tokens=10, temperature=0.8)

    # Benchmark
    start_gen = time.time()
    outputs = model.generate(
        prompts=prompts,
        max_new_tokens=max_tokens,
        temperature=0.8,
        return_diagnostics=True
    )
    gen_time = time.time() - start_gen

    # Calculate metrics
    total_tokens = sum(outputs.generated_ids[i].shape[0] for i in range(len(prompts)))
    throughput = total_tokens / gen_time
    latency = gen_time / len(prompts)

    # Print results
    print(f"\nResults:")
    print(f"  Total time: {gen_time:.2f}s")
    print(f"  Total tokens: {total_tokens}")
    print(f"  Throughput: {throughput:.2f} tokens/s")
    print(f"  Latency per example: {latency:.3f}s")
    print(f"  Average α: {outputs.alpha_values.mean():.3f}")
    print(f"  Average ESS ratio: {outputs.ess_ratios.mean():.3f}")

    return {
        "config": config_name,
        "init_time": init_time,
        "gen_time": gen_time,
        "total_tokens": total_tokens,
        "throughput": throughput,
        "latency": latency,
        "alpha_mean": float(outputs.alpha_values.mean()),
    }

def main():
    print("=" * 80)
    print("Optimal Sampling HF - Benchmark")
    print("=" * 80)

    # Test configurations
    configs = [
        {
            "name": "KL Symmetry (Qwen 1.5B + 3B)",
            "theta": "Qwen/Qwen2.5-1.5B-Instruct",
            "t": "Qwen/Qwen2.5-3B-Instruct",
            "alpha_method": "kl_symmetry",
        },
        {
            "name": "Fixed Alpha 0.5 (Qwen 1.5B + 3B)",
            "theta": "Qwen/Qwen2.5-1.5B-Instruct",
            "t": "Qwen/Qwen2.5-3B-Instruct",
            "alpha_method": "fixed",
        },
        {
            "name": "Entropy (Qwen 1.5B + 3B)",
            "theta": "Qwen/Qwen2.5-1.5B-Instruct",
            "t": "Qwen/Qwen2.5-3B-Instruct",
            "alpha_method": "entropy",
        },
    ]

    # Run benchmarks
    results = []
    for config in configs:
        result = benchmark_config(
            config_name=config["name"],
            model_theta=config["theta"],
            model_t=config["t"],
            alpha_method=config["alpha_method"],
            num_examples=5,  # Small number for quick test
            max_tokens=64,
        )
        results.append(result)

    # Print comparison
    print("\n" + "=" * 80)
    print("Benchmark Summary")
    print("=" * 80)
    print(f"\n{'Configuration':<40} {'Throughput':<15} {'Latency':<15} {'Alpha':<10}")
    print("-" * 80)
    for result in results:
        print(f"{result['config']:<40} "
              f"{result['throughput']:>10.2f} tok/s  "
              f"{result['latency']:>10.3f}s     "
              f"{result['alpha_mean']:>6.3f}")

    print("\n" + "=" * 80)
    print("✓ Benchmark completed!")
    print("=" * 80)

if __name__ == "__main__":
    main()
