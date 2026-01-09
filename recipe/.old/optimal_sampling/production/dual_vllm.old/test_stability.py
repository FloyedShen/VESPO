#!/usr/bin/env python3
"""
测试稳定性：检查两个模型分布差异大时的行为
"""

import numpy as np
from utils import merge_top_k_candidates, solve_kl_symmetry, compute_q_star


def test_no_overlap():
    """测试完全没有 overlap 的情况"""
    print("\n" + "="*70)
    print("🧪 测试 1: 完全没有 overlap")
    print("="*70)

    # π_θ 的 top-5: tokens 0-4
    logprobs_theta = {
        "token_0": -0.1,
        "token_1": -0.5,
        "token_2": -1.0,
        "token_3": -1.5,
        "token_4": -2.0,
    }

    # π_t 的 top-5: tokens 5-9 (完全不同!)
    logprobs_t = {
        "token_5": -0.1,
        "token_6": -0.5,
        "token_7": -1.0,
        "token_8": -1.5,
        "token_9": -2.0,
    }

    # 合并
    candidates, probs_theta, probs_t = merge_top_k_candidates(
        logprobs_theta, logprobs_t
    )

    print(f"并集大小: {len(candidates)}")
    print(f"交集大小: 0")

    # 检查归一化
    print(f"\nπ_θ 概率和: {probs_theta.sum():.6f}")
    print(f"π_t 概率和: {probs_t.sum():.6f}")

    # 检查 overlap（实际概率）
    overlap_mass_theta = sum(probs_theta[i] for i, tok in enumerate(candidates) if tok.startswith("token_5"))
    overlap_mass_t = sum(probs_t[i] for i, tok in enumerate(candidates) if tok.startswith("token_0"))
    print(f"\nπ_θ 在 π_t 的 top-k 上的概率质量: {overlap_mass_theta:.6e}")
    print(f"π_t 在 π_θ 的 top-k 上的概率质量: {overlap_mass_t:.6e}")

    # 计算 JS divergence
    js_div = compute_js_divergence(probs_theta, probs_t)
    print(f"\nJS Divergence: {js_div:.6f} (max=ln(2)≈0.693)")

    # KL 对称
    try:
        alpha = solve_kl_symmetry(probs_theta, probs_t)
        print(f"\n最优 α: {alpha:.3f}")

        q_star = compute_q_star(probs_theta, probs_t, alpha)

        # 检查 q* 的分布
        print(f"\nq* 的前 5 个概率:")
        top_5_idx = np.argsort(-q_star)[:5]
        for idx in top_5_idx:
            print(f"  {candidates[idx]}: {q_star[idx]:.4f}")

    except Exception as e:
        print(f"\n❌ KL 对称失败: {e}")


def test_small_overlap():
    """测试小 overlap 的情况"""
    print("\n" + "="*70)
    print("🧪 测试 2: 小 overlap (1 个共同 token)")
    print("="*70)

    # π_θ 的 top-5: tokens 0-4
    logprobs_theta = {
        "token_0": -0.1,
        "token_1": -0.5,
        "token_2": -1.0,
        "token_3": -1.5,
        "token_common": -2.0,  # 共同的 token，但概率低
    }

    # π_t 的 top-5: token_common + tokens 5-8
    logprobs_t = {
        "token_5": -0.1,
        "token_6": -0.5,
        "token_7": -1.0,
        "token_8": -1.5,
        "token_common": -2.0,  # 共同的 token，但概率低
    }

    candidates, probs_theta, probs_t = merge_top_k_candidates(
        logprobs_theta, logprobs_t
    )

    print(f"并集大小: {len(candidates)}")
    print(f"交集大小: 1")

    # 计算 overlap
    overlap_tokens = set(logprobs_theta.keys()) & set(logprobs_t.keys())
    overlap_mass_theta = sum(probs_theta[i] for i, tok in enumerate(candidates) if tok in overlap_tokens)
    overlap_mass_t = sum(probs_t[i] for i, tok in enumerate(candidates) if tok in overlap_tokens)

    print(f"\nOverlap 概率质量:")
    print(f"  π_θ: {overlap_mass_theta:.4f}")
    print(f"  π_t: {overlap_mass_t:.4f}")

    js_div = compute_js_divergence(probs_theta, probs_t)
    print(f"\nJS Divergence: {js_div:.6f}")

    alpha = solve_kl_symmetry(probs_theta, probs_t)
    print(f"\n最优 α: {alpha:.3f}")


def test_good_overlap():
    """测试好的 overlap 情况"""
    print("\n" + "="*70)
    print("🧪 测试 3: 好的 overlap (50% 共同)")
    print("="*70)

    # 共同的 tokens
    logprobs_theta = {
        "common_0": -0.1,
        "common_1": -0.5,
        "common_2": -1.0,
        "theta_only_0": -1.5,
        "theta_only_1": -2.0,
    }

    logprobs_t = {
        "common_0": -0.2,   # 稍微不同的概率
        "common_1": -0.6,
        "common_2": -1.1,
        "t_only_0": -1.5,
        "t_only_1": -2.0,
    }

    candidates, probs_theta, probs_t = merge_top_k_candidates(
        logprobs_theta, logprobs_t
    )

    print(f"并集大小: {len(candidates)}")
    overlap_tokens = set(logprobs_theta.keys()) & set(logprobs_t.keys())
    print(f"交集大小: {len(overlap_tokens)}")

    overlap_mass_theta = sum(probs_theta[i] for i, tok in enumerate(candidates) if tok in overlap_tokens)
    overlap_mass_t = sum(probs_t[i] for i, tok in enumerate(candidates) if tok in overlap_tokens)

    print(f"\nOverlap 概率质量:")
    print(f"  π_θ: {overlap_mass_theta:.4f}")
    print(f"  π_t: {overlap_mass_t:.4f}")

    js_div = compute_js_divergence(probs_theta, probs_t)
    print(f"\nJS Divergence: {js_div:.6f}")

    alpha = solve_kl_symmetry(probs_theta, probs_t)
    print(f"\n最优 α: {alpha:.3f}")


def compute_js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """计算 Jensen-Shannon Divergence

    JS(P||Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M)
    where M = 0.5 * (P + Q)

    范围: [0, ln(2)] ≈ [0, 0.693]
    - 0 表示完全相同
    - ln(2) 表示完全不同
    """
    eps = 1e-10
    m = 0.5 * (p + q)

    kl_pm = (p * (np.log(p + eps) - np.log(m + eps))).sum()
    kl_qm = (q * (np.log(q + eps) - np.log(m + eps))).sum()

    return 0.5 * kl_pm + 0.5 * kl_qm


if __name__ == "__main__":
    print("\n" + "="*70)
    print("🔬 稳定性测试：两个模型分布差异分析")
    print("="*70)

    test_no_overlap()
    test_small_overlap()
    test_good_overlap()

    print("\n" + "="*70)
    print("📊 总结")
    print("="*70)
    print("""
关键发现:
1. 完全没有 overlap 时，两个分布在合并后几乎不相交
2. exp(-100) 赋值会导致 missing tokens 概率 ≈ 0
3. JS divergence 可以量化分布差异
4. 当 JS divergence > 阈值（如 0.5）时，可能需要切换策略

建议:
- 添加 overlap ratio 检测
- 添加 JS divergence 检测
- 当不稳定时（overlap < 5 tokens 或 JS > 0.5），直接使用 π_t
    """)
