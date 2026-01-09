"""
Example usage of DualVLLMCoordinator

This script demonstrates how to use the dual-vllm approach for optimal sampling.

Prerequisites:
    1. Start two vLLM instances:

       # Terminal 1: Start π_θ (base model)
       python -m vllm.entrypoints.api_server \\
           --model meta-llama/Llama-2-7b-hf \\
           --port 8000

       # Terminal 2: Start π_t (teacher/instruct model)
       python -m vllm.entrypoints.api_server \\
           --model meta-llama/Llama-2-7b-chat-hf \\
           --port 8001

    2. Run this script:
       python example.py
"""

import asyncio
import numpy as np
from coordinator import DualVLLMCoordinator, CoordinatorConfig, generate_with_optimal_sampling


async def example_basic():
    """Basic usage example"""
    print("=" * 80)
    print("Example 1: Basic Usage")
    print("=" * 80)

    # Using the convenience function
    results = await generate_with_optimal_sampling(
        prompts=[
            "What is the capital of France?",
            "Explain quantum computing in simple terms.",
        ],
        theta_url="http://localhost:8000",
        t_url="http://localhost:8001",
        max_tokens=50,
        return_diagnostics=True
    )

    for i, result in enumerate(results):
        if result.error:
            print(f"\nPrompt {i+1}: ERROR - {result.error}")
            continue

        print(f"\n{'='*60}")
        print(f"Prompt {i+1}: {result.prompt}")
        print(f"{'='*60}")
        print(f"Generated: {result.generated_text}")
        print(f"\nStatistics:")
        print(f"  Tokens generated: {len(result.generated_tokens)}")
        print(f"  Alpha (mean ± std): {np.mean(result.alpha_history):.3f} ± {np.std(result.alpha_history):.3f}")
        print(f"  Alpha range: [{np.min(result.alpha_history):.3f}, {np.max(result.alpha_history):.3f}]")

        if result.diagnostics:
            print(f"\nDiagnostics:")
            print(f"  KL symmetry error: {result.diagnostics['kl_diff_mean']:.6f}")
            print(f"  ESS ratio: {result.diagnostics['ess_ratio_mean']:.3f}")


async def example_advanced():
    """Advanced usage with custom configuration"""
    print("\n\n" + "=" * 80)
    print("Example 2: Advanced Usage with Custom Config")
    print("=" * 80)

    # Custom configuration
    config = CoordinatorConfig(
        theta_url="http://localhost:8000",
        t_url="http://localhost:8001",
        top_k=200,  # More candidates for better approximation
        alpha_tol=1e-8,  # Tighter tolerance
        alpha_max_iter=30,
        max_retries=5,
        request_timeout=120.0,
        enable_logging=True,
        log_level="INFO"
    )

    # Create coordinator with context manager
    async with DualVLLMCoordinator(config) as coordinator:
        # Batch generation
        prompts = [
            "Write a short poem about artificial intelligence.",
            "What are the main differences between Python and JavaScript?",
            "Explain the theory of relativity.",
        ]

        results = await coordinator.generate_batch(
            prompts=prompts,
            max_tokens=100,
            temperature=0.8,
            return_diagnostics=True,
            show_progress=True
        )

        # Process results
        print(f"\n\nGeneration complete!")
        print(f"Successfully generated: {sum(1 for r in results if r.error is None)}/{len(results)}")

        # Statistics
        stats = coordinator.get_statistics()
        print(f"\nCoordinator Statistics:")
        print(f"  Total requests: {stats['total_requests']}")
        print(f"  Failed requests: {stats['failed_requests']}")
        print(f"  Total tokens: {stats['total_tokens']}")

        # Detailed analysis
        print(f"\n\nDetailed Analysis:")
        for i, result in enumerate(results):
            if result.error:
                continue

            print(f"\n{'='*60}")
            print(f"Prompt {i+1}: {result.prompt[:50]}...")
            print(f"{'='*60}")

            # Alpha analysis
            alphas = result.alpha_history
            print(f"Alpha statistics:")
            print(f"  Mean: {np.mean(alphas):.4f}")
            print(f"  Std:  {np.std(alphas):.4f}")
            print(f"  Min:  {np.min(alphas):.4f}")
            print(f"  Max:  {np.max(alphas):.4f}")

            # Diagnostics
            if result.diagnostics:
                diag = result.diagnostics
                print(f"\nSampling quality:")
                print(f"  KL(q*||π_θ): {diag['kl_theta_mean']:.4f} ± {diag['kl_theta_std']:.4f}")
                print(f"  KL(q*||π_t):  {diag['kl_t_mean']:.4f} ± {diag['kl_t_std']:.4f}")
                print(f"  KL symmetry error: {diag['kl_diff_mean']:.6f}")
                print(f"  ESS ratio: {diag['ess_ratio_mean']:.3f} (ideal: 1.0)")

            # Show first few tokens
            print(f"\nFirst 10 alphas:")
            for j, alpha in enumerate(alphas[:10]):
                print(f"  Token {j+1}: α={alpha:.4f}")


async def example_batch_processing():
    """Example: Processing a large batch of prompts"""
    print("\n\n" + "=" * 80)
    print("Example 3: Batch Processing")
    print("=" * 80)

    # Simulate a dataset
    prompts = [
        f"Question {i}: What is {i} + {i}?"
        for i in range(1, 21)  # 20 prompts
    ]

    config = CoordinatorConfig(
        theta_url="http://localhost:8000",
        t_url="http://localhost:8001",
        top_k=100,
        connection_pool_size=50,  # Large pool for concurrent requests
    )

    async with DualVLLMCoordinator(config) as coordinator:
        results = await coordinator.generate_batch(
            prompts=prompts,
            max_tokens=20,
            temperature=1.0,
            return_diagnostics=False,  # Disable for faster processing
            show_progress=True
        )

        # Summary
        successful = [r for r in results if r.error is None]
        failed = [r for r in results if r.error is not None]

        print(f"\n\nBatch Processing Summary:")
        print(f"  Total prompts: {len(prompts)}")
        print(f"  Successful: {len(successful)}")
        print(f"  Failed: {len(failed)}")

        if successful:
            all_alphas = [alpha for r in successful for alpha in r.alpha_history]
            print(f"\n  Overall alpha statistics:")
            print(f"    Mean: {np.mean(all_alphas):.3f}")
            print(f"    Std:  {np.std(all_alphas):.3f}")

        if failed:
            print(f"\n  Failed prompts:")
            for r in failed[:5]:  # Show first 5 failures
                print(f"    - {r.prompt[:50]}... : {r.error}")


async def example_error_handling():
    """Example: Error handling"""
    print("\n\n" + "=" * 80)
    print("Example 4: Error Handling")
    print("=" * 80)

    # Test with potentially bad configuration
    config = CoordinatorConfig(
        theta_url="http://localhost:9999",  # Wrong port (for testing)
        t_url="http://localhost:8001",
        max_retries=2,
        retry_delay=0.5,
    )

    async with DualVLLMCoordinator(config) as coordinator:
        results = await coordinator.generate_batch(
            prompts=["Test prompt"],
            max_tokens=10,
            show_progress=False
        )

        for result in results:
            if result.error:
                print(f"Expected error caught: {result.error}")
            else:
                print(f"Unexpectedly succeeded: {result.generated_text}")


async def main():
    """Run all examples"""
    print("\n" + "="*80)
    print("Dual VLLM Coordinator - Examples")
    print("="*80)
    print("\nMake sure you have two vLLM instances running:")
    print("  - π_θ (base): http://localhost:8000")
    print("  - π_t (teacher): http://localhost:8001")
    print("\nStarting examples...\n")

    try:
        # Run examples
        await example_basic()
        await example_advanced()
        await example_batch_processing()

        # Uncomment to test error handling
        # await example_error_handling()

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
