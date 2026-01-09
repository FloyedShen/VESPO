#!/usr/bin/env python3
"""
在实际 Qwen3 模型上测试稳定性检测

这个脚本会：
1. 连接到两个 vLLM 服务器
2. 对不同类型的 prompts 测试稳定性
3. 收集并分析稳定性统计
"""

import asyncio
import aiohttp
import numpy as np
from typing import List, Dict
from utils_stability import (
    merge_top_k_candidates_with_stability,
    solve_kl_symmetry_with_fallback
)


async def get_logprobs(
    session: aiohttp.ClientSession,
    url: str,
    model_name: str,
    prompt: str,
    top_k: int = 20
) -> Dict[str, float]:
    """从 vLLM 获取 logprobs"""
    payload = {
        "model": model_name,
        "prompt": prompt,
        "max_tokens": 1,
        "temperature": 1.0,
        "logprobs": top_k,
        "echo": False,
    }

    async with session.post(f"{url}/v1/completions", json=payload) as resp:
        data = await resp.json()
        return data["choices"][0]["logprobs"]["top_logprobs"][0]


async def analyze_prompt_stability(
    session: aiohttp.ClientSession,
    prompt_theta: str,
    prompt_t: str,
    theta_url: str = "http://localhost:9000",
    t_url: str = "http://localhost:9001",
    theta_model: str = "Qwen/Qwen3-4B-Base",
    t_model: str = "Qwen/Qwen3-14B"
) -> Dict:
    """分析单个 prompt 的稳定性"""

    # 获取 logprobs
    logprobs_theta, logprobs_t = await asyncio.gather(
        get_logprobs(session, theta_url, theta_model, prompt_theta),
        get_logprobs(session, t_url, t_model, prompt_t)
    )

    # 稳定性检测
    candidates, probs_theta, probs_t, diag = merge_top_k_candidates_with_stability(
        logprobs_theta, logprobs_t,
        stability_threshold_js=0.5,
        stability_threshold_overlap=0.1,
        auto_fallback=True
    )

    # 计算 α
    alpha, did_fallback = solve_kl_symmetry_with_fallback(
        probs_theta, probs_t,
        stability_diagnostics=diag
    )

    # 合并结果
    result = {
        **diag,
        "alpha": alpha,
        "did_fallback": did_fallback,
        "num_candidates": len(candidates),
        "top_5_tokens_theta": list(logprobs_theta.keys())[:5],
        "top_5_tokens_t": list(logprobs_t.keys())[:5],
    }

    return result


async def main():
    print("\n" + "="*80)
    print("🔬 Qwen3 稳定性测试")
    print("="*80)

    # 测试不同类型的 prompts
    test_cases = [
        {
            "name": "技术问题（应该稳定）",
            "theta": "Q: What is machine learning?\nA:",
            "t": "<|im_start|>user\nWhat is machine learning?<|im_end|>\n<|im_start|>assistant\n",
        },
        {
            "name": "简单问题（应该稳定）",
            "theta": "Q: What is 2+2?\nA:",
            "t": "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n",
        },
        {
            "name": "开放性问题（可能不太稳定）",
            "theta": "Q: Tell me a creative story.\nA:",
            "t": "<|im_start|>user\nTell me a creative story.<|im_end|>\n<|im_start|>assistant\n",
        },
        {
            "name": "中文问题（可能不太稳定）",
            "theta": "Q: 什么是人工智能？\nA:",
            "t": "<|im_start|>user\n什么是人工智能？<|im_end|>\n<|im_start|>assistant\n",
        },
    ]

    timeout = aiohttp.ClientTimeout(total=60)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        results = []

        for i, case in enumerate(test_cases):
            print(f"\n{'='*80}")
            print(f"测试 {i+1}/{len(test_cases)}: {case['name']}")
            print("="*80)

            try:
                result = await analyze_prompt_stability(
                    session, case['theta'], case['t']
                )

                results.append({**case, **result})

                # 打印结果
                print(f"\n📊 稳定性分析:")
                print(f"  Overlap Count: {result['overlap_count']}")
                print(f"  Overlap Mass (θ): {result['overlap_mass_theta']:.3f}")
                print(f"  Overlap Mass (t): {result['overlap_mass_t']:.3f}")
                print(f"  JS Divergence: {result['js_divergence']:.3f}")
                print(f"  Is Stable: {result['is_stable']}")
                print(f"  Alpha: {result['alpha']:.3f}")
                print(f"  Did Fallback: {result['did_fallback']}")

                print(f"\n📝 Top-5 Tokens:")
                print(f"  π_θ: {', '.join(result['top_5_tokens_theta'])}")
                print(f"  π_t: {', '.join(result['top_5_tokens_t'])}")

                # 判断
                if result['did_fallback']:
                    print(f"\n⚠️  分布不稳定，已 fallback 到 π_t")
                else:
                    print(f"\n✅ 分布稳定，正常混合")

            except Exception as e:
                print(f"\n❌ 错误: {e}")
                import traceback
                traceback.print_exc()

        # 汇总统计
        print(f"\n{'='*80}")
        print("📈 汇总统计")
        print("="*80)

        if results:
            fallback_count = sum(1 for r in results if r['did_fallback'])
            avg_js = np.mean([r['js_divergence'] for r in results])
            avg_overlap = np.mean([r['overlap_mass_theta'] for r in results])
            avg_alpha = np.mean([r['alpha'] for r in results])

            print(f"\n总测试数: {len(results)}")
            print(f"Fallback 次数: {fallback_count} ({fallback_count/len(results)*100:.1f}%)")
            print(f"平均 JS Divergence: {avg_js:.3f}")
            print(f"平均 Overlap Mass: {avg_overlap:.3f}")
            print(f"平均 Alpha: {avg_alpha:.3f}")

            print(f"\n{'='*80}")
            print("💡 分析")
            print("="*80)

            if fallback_count == 0:
                print("✅ 所有测试都稳定，两个模型高度一致")
                print("   建议: 使用默认配置即可")
            elif fallback_count < len(results) * 0.2:
                print("✅ 大部分测试稳定，偶尔 fallback")
                print("   建议: 当前稳定性阈值合适")
            else:
                print("⚠️  Fallback 较频繁，模型差异较大")
                print("   建议: 考虑调整阈值或使用更相似的模型")

            # 稳定性建议
            if avg_js < 0.3:
                print(f"\n📊 JS Divergence 分析: {avg_js:.3f} < 0.3 (低)")
                print("   → 两个模型非常一致")
            elif avg_js < 0.5:
                print(f"\n📊 JS Divergence 分析: {avg_js:.3f} < 0.5 (中等)")
                print("   → 两个模型适度一致")
            else:
                print(f"\n📊 JS Divergence 分析: {avg_js:.3f} > 0.5 (高)")
                print("   → 两个模型差异较大")

            if avg_overlap > 0.2:
                print(f"\n📊 Overlap 分析: {avg_overlap:.3f} > 0.2 (高)")
                print("   → Top-k 有大量重叠")
            elif avg_overlap > 0.1:
                print(f"\n📊 Overlap 分析: {avg_overlap:.3f} > 0.1 (中等)")
                print("   → Top-k 有适度重叠")
            else:
                print(f"\n📊 Overlap 分析: {avg_overlap:.3f} < 0.1 (低)")
                print("   → Top-k 重叠很少")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n中断测试")
    except Exception as e:
        print(f"\n\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
