#!/usr/bin/env python3
"""
Demonstration of Dual VLLM Optimal Sampling with Qwen3-4B-Base + Qwen3-14B

This script demonstrates all the key features:
1. Dual prompts (different templates for base and teacher)
2. First token forcing (α=1 for first token)
3. Support constraint (trust region limiting)
4. KL symmetry (theoretical guarantee)
5. Real-time generation with progress tracking
"""

import asyncio
import numpy as np
from coordinator_enhanced import EnhancedDualVLLMCoordinator
from config_enhanced import EnhancedCoordinatorConfig


async def demo_basic():
    """Basic demonstration"""
    print("\n" + "="*80)
    print("🚀 Demo 1: Basic Optimal Sampling")
    print("="*80)

    config = EnhancedCoordinatorConfig(
        theta_url="http://localhost:9000",
        t_url="http://localhost:9001",
        top_k=20,
        force_first_token=True,
        constraint_to_target=True,
        target_top_p=0.95,
        enable_logging=False,
    )

    prompts_theta = [
        "Q: What is deep learning?\nA:",
    ]

    prompts_t = [
        "<|im_start|>user\nWhat is deep learning?<|im_end|>\n<|im_start|>assistant\n",
    ]

    async with EnhancedDualVLLMCoordinator(config) as coordinator:
        results = await coordinator.generate_batch_dual_prompts(
            prompts_theta=prompts_theta,
            prompts_t=prompts_t,
            max_tokens=80,
            temperature=1.0,
            return_diagnostics=True,
            show_progress=True
        )

        for result in results:
            print(f"\n{'='*80}")
            print("📊 Results")
            print("="*80)
            print(f"Tokens: {len(result.generated_tokens)}")
            print(f"Alpha: {np.mean(result.alpha_history):.3f} ± {np.std(result.alpha_history):.3f}")
            print(f"First Alpha: {result.alpha_history[0]:.3f} (should be 1.000)")
            print(f"KL Symmetry Error: {result.diagnostics['kl_diff_mean']:.6f} (should be < 0.001)")
            print(f"ESS Ratio: {result.diagnostics['ess_ratio_mean']:.3f} (should be ≈ 1.0)")

            print(f"\n{'='*80}")
            print("📝 Generated Text")
            print("="*80)
            generated_only = result.generated_text[len(result.prompt):]
            print(generated_only)


async def demo_comparison():
    """Compare different configurations"""
    print("\n" + "="*80)
    print("🚀 Demo 2: Configuration Comparison")
    print("="*80)

    configs = [
        ("Conservative (Strong Teacher)", EnhancedCoordinatorConfig(
            theta_url="http://localhost:9000",
            t_url="http://localhost:9001",
            top_k=20,
            force_first_token=True,
            constraint_to_target=True,
            target_top_p=0.90,  # Strict constraint
            enable_logging=False,
        )),
        ("Balanced (Recommended)", EnhancedCoordinatorConfig(
            theta_url="http://localhost:9000",
            t_url="http://localhost:9001",
            top_k=20,
            force_first_token=True,
            constraint_to_target=True,
            target_top_p=0.95,  # Moderate constraint
            enable_logging=False,
        )),
        ("Exploratory (More Diversity)", EnhancedCoordinatorConfig(
            theta_url="http://localhost:9000",
            t_url="http://localhost:9001",
            top_k=20,
            force_first_token=False,
            constraint_to_target=False,  # No constraint
            enable_logging=False,
        )),
    ]

    prompt_theta = "Q: Explain attention mechanism in transformers.\nA:"
    prompt_t = "<|im_start|>user\nExplain attention mechanism in transformers.<|im_end|>\n<|im_start|>assistant\n"

    for name, config in configs:
        print(f"\n{'='*80}")
        print(f"🔧 Configuration: {name}")
        print("="*80)

        async with EnhancedDualVLLMCoordinator(config) as coordinator:
            results = await coordinator.generate_batch_dual_prompts(
                prompts_theta=[prompt_theta],
                prompts_t=[prompt_t],
                max_tokens=50,
                temperature=1.0,
                return_diagnostics=True,
                show_progress=False
            )

            result = results[0]
            print(f"Alpha: {np.mean(result.alpha_history):.3f} ± {np.std(result.alpha_history):.3f}")
            print(f"First Alpha: {result.alpha_history[0]:.3f}")
            print(f"KL Error: {result.diagnostics['kl_diff_mean']:.6f}")

            stats = coordinator.get_statistics()
            print(f"Constraint Applied: {stats['constraint_applied']} times")

            # Show first 100 chars
            generated = result.generated_text[len(result.prompt):][:100]
            print(f"\nText Preview: {generated}...")


async def demo_batch():
    """Batch generation demonstration"""
    print("\n" + "="*80)
    print("🚀 Demo 3: Batch Generation")
    print("="*80)

    config = EnhancedCoordinatorConfig(
        theta_url="http://localhost:9000",
        t_url="http://localhost:9001",
        top_k=20,
        force_first_token=True,
        constraint_to_target=True,
        target_top_p=0.95,
        enable_logging=False,
    )

    questions = [
        "What is reinforcement learning?",
        "How does backpropagation work?",
        "What are transformers?",
        "Explain gradient descent.",
    ]

    prompts_theta = [f"Q: {q}\nA:" for q in questions]
    prompts_t = [f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n" for q in questions]

    print(f"\nGenerating responses for {len(questions)} questions...")

    async with EnhancedDualVLLMCoordinator(config) as coordinator:
        results = await coordinator.generate_batch_dual_prompts(
            prompts_theta=prompts_theta,
            prompts_t=prompts_t,
            max_tokens=60,
            temperature=1.0,
            return_diagnostics=True,
            show_progress=True
        )

        print(f"\n{'='*80}")
        print("📊 Batch Results Summary")
        print("="*80)

        for i, (question, result) in enumerate(zip(questions, results)):
            print(f"\n[{i+1}] {question}")
            print(f"    Tokens: {len(result.generated_tokens)}")
            print(f"    Alpha: {np.mean(result.alpha_history):.3f}")
            print(f"    Preview: {result.generated_text[len(result.prompt):][:80]}...")

        # Overall statistics
        stats = coordinator.get_statistics()
        print(f"\n{'='*80}")
        print("📈 Overall Statistics")
        print("="*80)
        print(f"Total Requests: {stats['total_requests']}")
        print(f"Total Tokens: {stats['total_tokens']}")
        print(f"Success Rate: {stats['success_rate']:.1%}")
        print(f"First Token Forced: {stats['first_token_forced']} times")
        print(f"Constraint Applied: {stats['constraint_applied']} times")


async def main():
    """Run all demos"""
    print("\n" + "="*80)
    print("🎯 Dual VLLM Optimal Sampling - Qwen3 Demonstration")
    print("="*80)
    print("\nFeatures:")
    print("  ✅ Dual Prompts (different templates for base and teacher)")
    print("  ✅ First Token Forcing (α=1 for first token)")
    print("  ✅ Support Constraint (trust region limiting)")
    print("  ✅ KL Symmetry (theoretical guarantee)")
    print("  ✅ Efficient Implementation (<1ms overhead/token)")

    try:
        # Run demos
        await demo_basic()
        await demo_comparison()
        await demo_batch()

        print("\n" + "="*80)
        print("🎉 All Demos Completed Successfully!")
        print("="*80)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
