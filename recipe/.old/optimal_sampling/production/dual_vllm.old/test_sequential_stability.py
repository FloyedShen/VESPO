#!/usr/bin/env python3
"""
测试后续 token 的稳定性

检查在第一个 token 之后，两个模型的分布是否会变稳定
"""

import asyncio
import aiohttp
import numpy as np
from utils_stability import (
    merge_top_k_candidates_with_stability,
    solve_kl_symmetry_with_fallback
)


async def get_logprobs(session, url, model_name, prompt, top_k=20):
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
        logprobs_dict = data["choices"][0]["logprobs"]["top_logprobs"][0]
        token = data["choices"][0]["text"]
        return logprobs_dict, token


async def test_sequential_stability():
    """测试连续生成时的稳定性"""
    print("\n" + "="*80)
    print("🔬 测试后续 Token 稳定性")
    print("="*80)

    theta_url = "http://localhost:9000"
    t_url = "http://localhost:9001"
    theta_model = "Qwen/Qwen3-4B-Base"
    t_model = "Qwen/Qwen3-14B"

    # 初始 prompts
    prompt_theta = "Q: What is 2+2?\nA:"
    prompt_t = "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n"

    # 生成 10 个 token
    num_tokens = 10

    timeout = aiohttp.ClientTimeout(total=60)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        context_theta = prompt_theta
        context_t = prompt_t

        stability_history = []

        for step in range(num_tokens):
            print(f"\n{'='*80}")
            print(f"Token {step+1}/{num_tokens}")
            print("="*80)

            # 获取 logprobs
            (logprobs_theta, token_theta), (logprobs_t, token_t) = await asyncio.gather(
                get_logprobs(session, theta_url, theta_model, context_theta),
                get_logprobs(session, t_url, t_model, context_t)
            )

            # 稳定性检测
            candidates, probs_theta, probs_t, diag = merge_top_k_candidates_with_stability(
                logprobs_theta, logprobs_t,
                stability_threshold_js=0.5,
                stability_threshold_overlap=0.1,
                auto_fallback=True
            )

            alpha, did_fallback = solve_kl_symmetry_with_fallback(
                probs_theta, probs_t,
                stability_diagnostics=diag
            )

            # 记录
            stability_history.append({
                "step": step,
                "overlap_count": diag["overlap_count"],
                "overlap_mass": diag["overlap_mass_theta"],
                "js_divergence": diag["js_divergence"],
                "is_stable": diag["is_stable"],
                "alpha": alpha,
                "did_fallback": did_fallback,
                "token_theta": token_theta,
                "token_t": token_t,
            })

            # 打印
            print(f"π_θ token: '{token_theta}'")
            print(f"π_t token: '{token_t}'")
            print(f"Overlap: {diag['overlap_count']} tokens ({diag['overlap_mass_theta']:.3f} mass)")
            print(f"JS Div: {diag['js_divergence']:.3f}")
            print(f"Alpha: {alpha:.3f}")
            print(f"Fallback: {did_fallback}")

            # 使用 π_t 的 token 更新上下文（因为我们用它来采样）
            # 注意：这里简化了，实际应该从 q* 采样
            if did_fallback or step == 0:
                # 使用 teacher token
                next_token = token_t
            else:
                # 简化：仍然使用 teacher token（实际应该从 q* 采样）
                next_token = token_t

            context_theta += next_token
            context_t += next_token

        # 分析
        print(f"\n{'='*80}")
        print("📊 稳定性演化分析")
        print("="*80)

        fallback_count = sum(1 for h in stability_history if h['did_fallback'])
        stable_count = sum(1 for h in stability_history if h['is_stable'])

        print(f"\n总 Tokens: {num_tokens}")
        print(f"Fallback 次数: {fallback_count} ({fallback_count/num_tokens*100:.1f}%)")
        print(f"稳定次数: {stable_count} ({stable_count/num_tokens*100:.1f}%)")

        # 分步骤分析
        print(f"\n逐步骤分析:")
        print(f"{'Step':<6} {'Overlap':<8} {'JS Div':<8} {'Alpha':<8} {'Stable':<8} {'Fallback':<10}")
        print("-" * 60)
        for h in stability_history:
            print(
                f"{h['step']:<6} "
                f"{h['overlap_count']:<8} "
                f"{h['js_divergence']:<8.3f} "
                f"{h['alpha']:<8.3f} "
                f"{str(h['is_stable']):<8} "
                f"{str(h['did_fallback']):<10}"
            )

        # 趋势分析
        avg_js_first_3 = np.mean([h['js_divergence'] for h in stability_history[:3]])
        avg_js_last_7 = np.mean([h['js_divergence'] for h in stability_history[3:]])

        print(f"\n趋势分析:")
        print(f"  前 3 个 token 平均 JS: {avg_js_first_3:.3f}")
        print(f"  后 7 个 token 平均 JS: {avg_js_last_7:.3f}")

        if avg_js_last_7 < avg_js_first_3 * 0.8:
            print(f"  ✅ 稳定性提升了 {(1-avg_js_last_7/avg_js_first_3)*100:.1f}%")
            print(f"  → 建议: 前几个 token 使用 π_t，后续可以混合")
        elif avg_js_last_7 < 0.5:
            print(f"  ✅ 后续 token 已经稳定")
            print(f"  → 建议: 可以正常混合")
        else:
            print(f"  ⚠️  始终不稳定")
            print(f"  → 建议: 始终使用 π_t 或调整阈值")

        # 生成的文本
        print(f"\n{'='*80}")
        print("📝 生成的文本")
        print("="*80)
        print(f"θ: {context_theta[:200]}")
        print(f"t: {context_t[:200]}")


if __name__ == "__main__":
    try:
        asyncio.run(test_sequential_stability())
    except KeyboardInterrupt:
        print("\n\n中断测试")
    except Exception as e:
        print(f"\n\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
