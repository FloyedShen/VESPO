#!/usr/bin/env python3
"""
Enhanced Dual VLLM Examples

Demonstrates advanced features:
1. Dual prompts (different templates for base and teacher)
2. First token forcing
3. Support constraint (trust region)
4. Different use cases
"""

import asyncio
import numpy as np
from coordinator_enhanced import EnhancedDualVLLMCoordinator, generate_with_dual_prompts
from config_enhanced import EnhancedCoordinatorConfig


# ============================================
# Example 1: Dual Prompts with Different Templates
# ============================================

async def example_dual_prompts():
    """
    Use different chat templates for base and teacher models

    Scenario:
    - Base model (π_θ): Trained with simple Q&A format
    - Teacher model (π_t): Trained with ChatML format

    We want to sample from q* while respecting both formats!
    """
    print("\n" + "="*70)
    print("Example 1: Dual Prompts with Different Templates")
    print("="*70)

    # Base model uses simple format
    prompts_theta = [
        "Question: What is machine learning?\n\nAnswer:",
        "Question: Explain neural networks in simple terms.\n\nAnswer:",
    ]

    # Teacher model uses ChatML format
    prompts_t = [
        "<|im_start|>user\nWhat is machine learning?<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nExplain neural networks in simple terms.<|im_end|>\n<|im_start|>assistant\n",
    ]

    # Create coordinator with enhanced features
    config = EnhancedCoordinatorConfig(
        theta_url="http://localhost:8000",
        t_url="http://localhost:8001",
        top_k=100,
        force_first_token=True,  # Use π_t for first token
        constraint_to_target=False,  # No constraint for now
        enable_logging=True,
    )

    async with EnhancedDualVLLMCoordinator(config) as coordinator:
        results = await coordinator.generate_batch_dual_prompts(
            prompts_theta=prompts_theta,
            prompts_t=prompts_t,
            max_tokens=50,
            temperature=1.0,
            return_diagnostics=True,
            show_progress=True
        )

        # Display results
        for i, result in enumerate(results):
            print(f"\n{'─'*70}")
            print(f"Prompt {i+1}:")
            print(f"  Base format: {prompts_theta[i][:50]}...")
            print(f"  Teacher format: {prompts_t[i][:50]}...")

            if result.error:
                print(f"  ❌ Error: {result.error}")
            else:
                print(f"  ✅ Generated {len(result.generated_tokens)} tokens")
                print(f"  📊 Average α: {np.mean(result.alpha_history):.3f}")
                print(f"     First α: {result.alpha_history[0]:.3f} (should be 1.0)")

                if result.diagnostics:
                    print(f"  📈 KL symmetry error: {result.diagnostics['kl_diff_mean']:.6f}")
                    print(f"     ESS ratio: {result.diagnostics['ess_ratio_mean']:.3f}")

        # Show statistics
        stats = coordinator.get_statistics()
        print(f"\n{'─'*70}")
        print(f"Coordinator Statistics:")
        print(f"  First tokens forced: {stats['first_token_forced']}")
        print(f"  Total tokens: {stats['total_tokens']}")


# ============================================
# Example 2: Support Constraint (Trust Region)
# ============================================

async def example_support_constraint():
    """
    Limit sampling to π_t's trust region

    Why use this?
    - Prevents sampling tokens that π_t considers very unlikely
    - Better numerical stability
    - Stronger alignment with teacher

    Trade-off:
    - Less exploration from π_θ
    - May reduce diversity
    """
    print("\n" + "="*70)
    print("Example 2: Support Constraint (Trust Region)")
    print("="*70)

    prompts_theta = ["Explain quantum computing:"]
    prompts_t = ["<|im_start|>user\nExplain quantum computing<|im_end|>\n<|im_start|>assistant\n"]

    # Compare with and without constraint
    configs = [
        ("No Constraint", EnhancedCoordinatorConfig(
            theta_url="http://localhost:8000",
            t_url="http://localhost:8001",
            constraint_to_target=False,
        )),
        ("With Constraint (p=0.95)", EnhancedCoordinatorConfig(
            theta_url="http://localhost:8000",
            t_url="http://localhost:8001",
            constraint_to_target=True,
            target_top_p=0.95,
        )),
        ("With Constraint (p=0.90)", EnhancedCoordinatorConfig(
            theta_url="http://localhost:8000",
            t_url="http://localhost:8001",
            constraint_to_target=True,
            target_top_p=0.90,
        )),
    ]

    for name, config in configs:
        print(f"\n{'─'*70}")
        print(f"Configuration: {name}")

        async with EnhancedDualVLLMCoordinator(config) as coordinator:
            results = await coordinator.generate_batch_dual_prompts(
                prompts_theta=prompts_theta,
                prompts_t=prompts_t,
                max_tokens=30,
                temperature=1.0,
                return_diagnostics=True,
                show_progress=False
            )

            result = results[0]
            if not result.error:
                print(f"  Generated tokens: {len(result.generated_tokens)}")
                print(f"  Average α: {np.mean(result.alpha_history):.3f}")
                print(f"  α range: [{np.min(result.alpha_history):.3f}, {np.max(result.alpha_history):.3f}]")

                if result.diagnostics:
                    print(f"  ESS_θ: {result.diagnostics['ess_theta_mean']:.2f}")
                    print(f"  ESS_t: {result.diagnostics['ess_t_mean']:.2f}")

            stats = coordinator.get_statistics()
            print(f"  Constraints applied: {stats['constraint_applied']}")


# ============================================
# Example 3: First Token Forcing
# ============================================

async def example_first_token_forcing():
    """
    Force first token to use π_t (α=1)

    Theory:
    - First token sets the initial direction
    - Using π_t ensures we start in a good region
    - Subsequent tokens can balance both models
    """
    print("\n" + "="*70)
    print("Example 3: First Token Forcing")
    print("="*70)

    prompts_theta = ["Write a poem about AI:"]
    prompts_t = ["<|im_start|>user\nWrite a poem about AI<|im_end|>\n<|im_start|>assistant\n"]

    # Compare with and without first token forcing
    configs = [
        ("Without Forcing", EnhancedCoordinatorConfig(
            theta_url="http://localhost:8000",
            t_url="http://localhost:8001",
            force_first_token=False,
        )),
        ("With Forcing", EnhancedCoordinatorConfig(
            theta_url="http://localhost:8000",
            t_url="http://localhost:8001",
            force_first_token=True,
        )),
    ]

    for name, config in configs:
        print(f"\n{'─'*70}")
        print(f"Configuration: {name}")

        async with EnhancedDualVLLMCoordinator(config) as coordinator:
            results = await coordinator.generate_batch_dual_prompts(
                prompts_theta=prompts_theta,
                prompts_t=prompts_t,
                max_tokens=20,
                temperature=1.0,
                show_progress=False
            )

            result = results[0]
            if not result.error:
                print(f"  First token α: {result.alpha_history[0]:.3f}")
                print(f"  Average α (all): {np.mean(result.alpha_history):.3f}")
                print(f"  Average α (rest): {np.mean(result.alpha_history[1:]):.3f}")

                # Show α evolution
                alpha_str = ", ".join([f"{a:.2f}" for a in result.alpha_history[:10]])
                print(f"  α evolution: [{alpha_str}, ...]")


# ============================================
# Example 4: Combined Features (Recommended)
# ============================================

async def example_combined():
    """
    Use all features together (recommended for production)

    This configuration gives:
    1. Better initial direction (first token forcing)
    2. Numerical stability (support constraint)
    3. Flexibility for different models (dual prompts)
    """
    print("\n" + "="*70)
    print("Example 4: Combined Features (Recommended)")
    print("="*70)

    # Real-world scenario: MATH problem solving
    prompts_theta = [
        "Problem: Solve for x: 2x + 5 = 13\n\nSolution:",
        "Problem: What is the derivative of x^2?\n\nSolution:",
    ]

    prompts_t = [
        "<|im_start|>user\nSolve for x: 2x + 5 = 13<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nWhat is the derivative of x^2?<|im_end|>\n<|im_start|>assistant\n",
    ]

    # Recommended configuration
    config = EnhancedCoordinatorConfig(
        theta_url="http://localhost:8000",
        t_url="http://localhost:8001",
        top_k=100,
        force_first_token=True,       # ✅ Better start
        constraint_to_target=True,    # ✅ Stability
        target_top_p=0.95,            # Keep 95% of π_t's mass
        alpha_tol=1e-6,               # High precision
        enable_logging=True,
    )

    async with EnhancedDualVLLMCoordinator(config) as coordinator:
        results = await coordinator.generate_batch_dual_prompts(
            prompts_theta=prompts_theta,
            prompts_t=prompts_t,
            max_tokens=100,
            temperature=1.0,
            return_diagnostics=True,
            show_progress=True
        )

        print(f"\n{'─'*70}")
        print(f"Results:")

        for i, result in enumerate(results):
            print(f"\nProblem {i+1}:")
            if result.error:
                print(f"  ❌ Error: {result.error}")
            else:
                print(f"  ✅ Tokens: {len(result.generated_tokens)}")
                print(f"  📊 α: {np.mean(result.alpha_history):.3f} ± {np.std(result.alpha_history):.3f}")

                if result.diagnostics:
                    print(f"  📈 KL error: {result.diagnostics['kl_diff_mean']:.6f}")
                    print(f"     ESS balance: {result.diagnostics['ess_ratio_mean']:.3f}")
                    print(f"     Entropy: {result.diagnostics['entropy_q_mean']:.3f}")

        # Overall statistics
        stats = coordinator.get_statistics()
        print(f"\n{'─'*70}")
        print(f"Overall Statistics:")
        print(f"  Total requests: {stats['total_requests']}")
        print(f"  Total tokens: {stats['total_tokens']}")
        print(f"  First tokens forced: {stats['first_token_forced']}")
        print(f"  Constraints applied: {stats['constraint_applied']}")
        print(f"  Success rate: {stats.get('success_rate', 0):.1%}")


# ============================================
# Example 5: Convenience Function
# ============================================

async def example_convenience_function():
    """
    Quick usage with convenience function
    """
    print("\n" + "="*70)
    print("Example 5: Convenience Function")
    print("="*70)

    prompts_theta = ["Hello, how are you?"]
    prompts_t = ["<|im_start|>user\nHello, how are you?<|im_end|>\n<|im_start|>assistant\n"]

    # One-liner usage
    results = await generate_with_dual_prompts(
        prompts_theta=prompts_theta,
        prompts_t=prompts_t,
        theta_url="http://localhost:8000",
        t_url="http://localhost:8001",
        max_tokens=50,
        force_first_token=True,
        constraint_to_target=True,
        target_top_p=0.95,
        show_progress=True
    )

    result = results[0]
    if not result.error:
        print(f"\n✅ Generated {len(result.generated_tokens)} tokens")
        print(f"📊 Average α: {np.mean(result.alpha_history):.3f}")


# ============================================
# Main
# ============================================

async def main():
    """Run all examples"""
    print("="*70)
    print("🎯 Enhanced Dual VLLM Examples")
    print("="*70)
    print("\nThese examples demonstrate advanced features:")
    print("  1. Dual prompts (different templates)")
    print("  2. Support constraint (trust region)")
    print("  3. First token forcing")
    print("  4. Combined features (recommended)")
    print("  5. Convenience function")
    print("\nNote: Requires two vLLM instances running on ports 8000 and 8001")
    print("="*70)

    # Uncomment the examples you want to run

    # await example_dual_prompts()
    # await example_support_constraint()
    # await example_first_token_forcing()
    # await example_combined()
    # await example_convenience_function()

    print("\n" + "="*70)
    print("💡 Tip: Uncomment the examples you want to run in main()")
    print("="*70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
