#!/usr/bin/env python3
"""Show generated text from Qwen3 models"""

import asyncio
import numpy as np
from coordinator_enhanced import EnhancedDualVLLMCoordinator
from config_enhanced import EnhancedCoordinatorConfig


async def test():
    """Test and show generated text"""
    print("\n" + "="*70)
    print("🧪 Qwen3-4B-Base + Qwen3-14B 文本生成测试")
    print("="*70)

    config = EnhancedCoordinatorConfig(
        theta_url="http://localhost:9000",
        t_url="http://localhost:9001",
        top_k=20,
        force_first_token=True,
        constraint_to_target=True,
        target_top_p=0.95,
        enable_logging=False,
    )

    prompts_theta = ["Q: What is machine learning?\nA:"]
    prompts_t = ["<|im_start|>user\nWhat is machine learning?<|im_end|>\n<|im_start|>assistant\n"]

    async with EnhancedDualVLLMCoordinator(config) as coordinator:
        results = await coordinator.generate_batch_dual_prompts(
            prompts_theta=prompts_theta,
            prompts_t=prompts_t,
            max_tokens=100,
            temperature=1.0,
            return_diagnostics=False,
            show_progress=False
        )

        for i, result in enumerate(results):
            print(f"\n{'='*70}")
            print(f"生成结果 #{i+1}")
            print("="*70)
            print(f"\nPrompt (Base):\n{result.prompt}")
            print(f"\nGenerated Tokens: {len(result.generated_tokens)}")
            print(f"Alpha 平均值: {np.mean(result.alpha_history):.3f} ± {np.std(result.alpha_history):.3f}")
            print(f"首 Alpha: {result.alpha_history[0]:.3f}")
            print(f"\n生成的完整文本:")
            print("-" * 70)
            # Show just the generated part (after the prompt)
            generated_only = result.generated_text[len(result.prompt):]
            print(generated_only)
            print("-" * 70)

if __name__ == "__main__":
    asyncio.run(test())
