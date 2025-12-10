#!/usr/bin/env python3
"""
Basic Usage Example for Optimal Sampling

This example shows the simplest way to use optimal sampling.
"""

from optimal_sampling import OptimalSamplingV1


def main():
    print("=" * 80)
    print("BASIC USAGE EXAMPLE")
    print("=" * 80)

    # Initialize sampler
    print("\n[1/3] Initializing Optimal Sampling...")
    sampler = OptimalSamplingV1(
        model_teacher="Qwen/Qwen2.5-3B-Instruct",
        model_theta="Qwen/Qwen2.5-1.5B-Instruct",
        alpha_method="kl_symmetry",
        gpu_memory_utilization=0.45,
    )
    print("✅ Initialization complete!")

    # Generate with optimal sampling
    print("\n[2/3] Generating with optimal sampling...")
    prompts = [
        "What is the capital of France?",
        "Explain what machine learning is in simple terms.",
        "Write a haiku about coding.",
    ]

    outputs = sampler.generate(
        prompts=prompts,
        max_tokens=128,
        temperature=0.8,
        use_optimal_sampling=True
    )

    print("✅ Generation complete!")

    # Display results
    print("\n[3/3] Results:")
    print("=" * 80)
    for i, (prompt, text) in enumerate(zip(prompts, outputs.generated_texts)):
        print(f"\n[Prompt {i+1}]: {prompt}")
        print(f"[Response]: {text}")
        print(f"[Tokens]: {outputs.num_tokens[i]}")
        print("-" * 80)

    # Show alpha statistics
    if outputs.alpha_stats:
        print("\n[Alpha Statistics]:")
        for key, value in outputs.alpha_stats.items():
            if key != "history":
                print(f"  {key}: {value:.4f}")

    print("\n" + "=" * 80)
    print("✅ EXAMPLE COMPLETED!")
    print("=" * 80)


if __name__ == '__main__':
    main()
