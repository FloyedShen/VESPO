#!/usr/bin/env python3
"""
简单测试 Qwen3-4B-Base + Qwen3-14B
假设两个 vLLM 实例已经在运行
"""

import asyncio
import numpy as np
from coordinator_enhanced import EnhancedDualVLLMCoordinator
from config_enhanced import EnhancedCoordinatorConfig


async def test():
    """简单测试"""
    print("\n" + "="*70)
    print("🧪 Qwen3-4B-Base + Qwen3-14B 简单测试")
    print("="*70)

    # 配置
    config = EnhancedCoordinatorConfig(
        theta_url="http://localhost:9000",
        t_url="http://localhost:9001",
        top_k=20,  # vLLM 0.11.0 限制最大为 20
        force_first_token=True,
        constraint_to_target=True,
        target_top_p=0.95,
        enable_logging=False,
    )

    # 测试提示
    prompts_theta = [
        "Q: What is machine learning?\nA:",
        "Q: Explain neural networks.\nA:",
    ]

    prompts_t = [
        "<|im_start|>user\nWhat is machine learning?<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nExplain neural networks.<|im_end|>\n<|im_start|>assistant\n",
    ]

    print(f"\n📝 测试 {len(prompts_theta)} 个提示...")
    print(f"   Base 格式: {prompts_theta[0][:40]}...")
    print(f"   Instruct 格式: {prompts_t[0][:40]}...")

    try:
        async with EnhancedDualVLLMCoordinator(config) as coordinator:
            results = await coordinator.generate_batch_dual_prompts(
                prompts_theta=prompts_theta,
                prompts_t=prompts_t,
                max_tokens=2000,
                temperature=1.0,
                return_diagnostics=True,
                show_progress=True
            )

            # 分析结果
            print(f"\n{'='*70}")
            print("📊 结果")
            print("="*70)

            print(results)
            for i, result in enumerate(results):
                print(f"\n[{i+1}] {prompts_theta[i][:30]}...")

                if result.error:
                    print(f"  ❌ 错误: {result.error}")
                else:
                    alpha_mean = np.mean(result.alpha_history)
                    alpha_std = np.std(result.alpha_history)

                    print(f"  ✅ Tokens: {len(result.generated_tokens)}")
                    print(f"  📊 α: {alpha_mean:.3f} ± {alpha_std:.3f}")
                    print(f"     首 α: {result.alpha_history[0]:.3f}")

                    if result.diagnostics:
                        print(f"  📈 KL 对称误差: {result.diagnostics['kl_diff_mean']:.6f}")
                        print(f"     ESS 比例: {result.diagnostics['ess_ratio_mean']:.3f}")

            # 统计
            stats = coordinator.get_statistics()
            print(f"\n{'='*70}")
            print("📈 统计")
            print("="*70)
            print(f"  请求数: {stats['total_requests']}")
            print(f"  Token 数: {stats['total_tokens']}")
            print(f"  首 token 强制次数: {stats['first_token_forced']}")
            print(f"  约束应用次数: {stats['constraint_applied']}")

            print("\n" + "="*70)
            print("🎉 测试完成！")
            print("="*70)

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test())
