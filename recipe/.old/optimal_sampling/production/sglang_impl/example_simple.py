"""
Simple example using SGLang-style Optimal Sampling

Clean and straightforward!
"""

import os

from production.sglang_impl import OptimalSamplingSGLang

print("=" * 80)
print("🚀 Optimal Sampling - Simple Example")
print("=" * 80)

# Step 1: Create sampler
print("\n📦 Loading models...")
sampler = OptimalSamplingSGLang(
    model_base="Qwen/Qwen2.5-0.5B",
    model_teacher="Qwen/Qwen2.5-1.5B",
    alpha_method="kl_symmetry",
    gpu_memory_utilization=0.85
)

# Step 2: Generate
prompts = [
    "What is artificial intelligence?",
    "Explain quantum computing in simple terms.",
]

print("\n🔄 Generating...")
outputs = sampler.generate(
    prompts=prompts,
    max_tokens=80,
    temperature=0.8,
    collect_diagnostics=True,
    verbose=True
)

# Step 3: Show results
print("\n" + "=" * 80)
print("📝 Results")
print("=" * 80)

for i, (prompt, text) in enumerate(zip(prompts, outputs.generated_texts)):
    print(f"\n{'─' * 80}")
    print(f"Prompt {i+1}: {prompt}")
    print(f"\nGenerated ({outputs.num_tokens[i]} tokens):")
    print(text)

    if outputs.alphas:
        import numpy as np
        alpha_mean = np.mean(outputs.alphas[i])
        alpha_min = min(outputs.alphas[i])
        alpha_max = max(outputs.alphas[i])
        print(f"\nAlpha statistics:")
        print(f"  Mean: {alpha_mean:.3f}")
        print(f"  Range: [{alpha_min:.3f}, {alpha_max:.3f}]")

print("\n" + "=" * 80)
print("✅ Done!")
print("=" * 80)
