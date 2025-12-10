#!/usr/bin/env python3
"""
Basic usage example for Optimal Sampling HF

This example demonstrates:
1. Loading the model with different configurations
2. Generating text with optimal sampling
3. Accessing diagnostics (alpha values, ESS, etc.)
"""

import torch
from optimal_sampling_hf import OptimalSamplingModel

def main():
    print("=" * 80)
    print("Optimal Sampling HF - Basic Usage Example")
    print("=" * 80)

    # Initialize model
    print("\n[1/3] Initializing model...")
    model = OptimalSamplingModel(
        model_theta_path="Qwen/Qwen2.5-1.5B-Instruct",  # Base model
        model_t_path="Qwen/Qwen2.5-3B-Instruct",        # Teacher model
        alpha_method="kl_symmetry",                      # Alpha computation method
        dtype=torch.bfloat16,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # Prepare prompts
    prompts = [
        "What is 2+2?",
        "Explain quantum computing in simple terms",
        "Write a haiku about artificial intelligence",
    ]

    # Generate
    print("\n[2/3] Generating with optimal sampling...")
    outputs = model.generate(
        prompts=prompts,
        max_new_tokens=100,
        temperature=0.8,
        return_diagnostics=True
    )

    # Print results
    print("\n[3/3] Results:")
    print("=" * 80)
    for i, text in enumerate(outputs.generated_texts):
        print(f"\n--- Example {i+1} ---")
        print(f"Prompt: {prompts[i]}")
        print(f"Generated: {text}")
        print(f"\nDiagnostics:")
        print(f"  - Generated tokens: {outputs.generated_ids[i].shape[0]}")
        print(f"  - Average α: {outputs.alpha_values[i].mean():.3f}")
        print(f"  - α range: [{outputs.alpha_values[i].min():.3f}, {outputs.alpha_values[i].max():.3f}]")
        print(f"  - Average ESS ratio: {outputs.ess_ratios[i].mean():.3f}")

        if outputs.diagnostics:
            kl_theta = outputs.diagnostics.get("kl_theta")
            kl_t = outputs.diagnostics.get("kl_t")
            if kl_theta is not None and kl_t is not None:
                print(f"  - KL(q||π_θ): {kl_theta[i].mean():.4f}")
                print(f"  - KL(q||π_t): {kl_t[i].mean():.4f}")

    print("\n" + "=" * 80)
    print("✓ Example completed successfully!")
    print("=" * 80)

if __name__ == "__main__":
    main()
