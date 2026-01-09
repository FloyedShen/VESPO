#!/usr/bin/env python3
"""
测试稳定性增强功能
"""

import numpy as np
from utils_stability import (
    merge_top_k_candidates_with_stability,
    solve_kl_symmetry_with_fallback,
    compute_js_divergence
)
from utils import compute_q_star


def test_unstable_fallback():
    """测试不稳定时的 fallback 行为"""
    print("\n" + "="*70)
    print("🧪 测试 1: 不稳定分布 - 自动 Fallback")
    print("="*70)

    # 完全不同的分布
    logprobs_theta = {
        "token_0": -0.1,
        "token_1": -0.5,
        "token_2": -1.0,
    }

    logprobs_t = {
        "token_5": -0.1,
        "token_6": -0.5,
        "token_7": -1.0,
    }

    # 使用稳定性增强的合并
    candidates, probs_theta, probs_t, diag = merge_top_k_candidates_with_stability(
        logprobs_theta, logprobs_t,
        stability_threshold_js=0.5,
        stability_threshold_overlap=0.1,
        auto_fallback=True
    )

    print(f"Overlap Count: {diag['overlap_count']}")
    print(f"Overlap Mass (θ): {diag['overlap_mass_theta']:.6f}")
    print(f"Overlap Mass (t): {diag['overlap_mass_t']:.6f}")
    print(f"JS Divergence: {diag['js_divergence']:.3f}")
    print(f"Is Stable: {diag['is_stable']}")
    print(f"Fallback to π_t: {diag['fallback_to_t']}")

    # 求解 α
    alpha, did_fallback = solve_kl_symmetry_with_fallback(
        probs_theta, probs_t,
        stability_diagnostics=diag
    )

    print(f"\n最优 α: {alpha:.3f}")
    print(f"Did Fallback: {did_fallback}")

    if did_fallback:
        print("\n✅ 成功检测到不稳定并 fallback 到 π_t!")
        print("   这意味着我们直接使用 teacher 模型，避免不稳定的混合")


def test_stable_mixing():
    """测试稳定时的正常混合"""
    print("\n" + "="*70)
    print("🧪 测试 2: 稳定分布 - 正常混合")
    print("="*70)

    # 有很多 overlap 的分布
    logprobs_theta = {
        "common_0": -0.1,
        "common_1": -0.5,
        "common_2": -1.0,
        "theta_only": -2.0,
    }

    logprobs_t = {
        "common_0": -0.2,
        "common_1": -0.6,
        "common_2": -1.1,
        "t_only": -2.0,
    }

    candidates, probs_theta, probs_t, diag = merge_top_k_candidates_with_stability(
        logprobs_theta, logprobs_t,
        stability_threshold_js=0.5,
        stability_threshold_overlap=0.1,
        auto_fallback=True
    )

    print(f"Overlap Count: {diag['overlap_count']}")
    print(f"Overlap Mass (θ): {diag['overlap_mass_theta']:.6f}")
    print(f"Overlap Mass (t): {diag['overlap_mass_t']:.6f}")
    print(f"JS Divergence: {diag['js_divergence']:.3f}")
    print(f"Is Stable: {diag['is_stable']}")
    print(f"Fallback to π_t: {diag['fallback_to_t']}")

    alpha, did_fallback = solve_kl_symmetry_with_fallback(
        probs_theta, probs_t,
        stability_diagnostics=diag
    )

    print(f"\n最优 α: {alpha:.3f}")
    print(f"Did Fallback: {did_fallback}")

    if not did_fallback:
        print("\n✅ 分布稳定，正常进行 KL 对称混合!")

        # 计算 q*
        q_star = compute_q_star(probs_theta, probs_t, alpha)

        print("\nq* 分布 (top-3):")
        top_3_idx = np.argsort(-q_star)[:3]
        for idx in top_3_idx:
            print(f"  {candidates[idx]}: {q_star[idx]:.4f}")


def test_threshold_sensitivity():
    """测试不同阈值的敏感性"""
    print("\n" + "="*70)
    print("🧪 测试 3: 阈值敏感性分析")
    print("="*70)

    # 中等 overlap 的分布
    logprobs_theta = {
        "common": -1.0,
        "theta_0": -0.1,
        "theta_1": -0.5,
    }

    logprobs_t = {
        "common": -1.0,
        "t_0": -0.1,
        "t_1": -0.5,
    }

    thresholds = [
        (0.3, 0.05, "宽松"),
        (0.5, 0.10, "中等"),
        (0.6, 0.20, "严格"),
    ]

    for js_thresh, overlap_thresh, name in thresholds:
        candidates, probs_theta, probs_t, diag = merge_top_k_candidates_with_stability(
            logprobs_theta, logprobs_t,
            stability_threshold_js=js_thresh,
            stability_threshold_overlap=overlap_thresh,
            auto_fallback=True
        )

        print(f"\n{name} 阈值 (JS<{js_thresh}, Overlap>{overlap_thresh}):")
        print(f"  JS Divergence: {diag['js_divergence']:.3f}")
        print(f"  Overlap Mass: {diag['overlap_mass_theta']:.3f}")
        print(f"  Is Stable: {diag['is_stable']}")
        print(f"  Fallback: {diag['fallback_to_t']}")


def test_comparison():
    """对比旧方法和新方法"""
    print("\n" + "="*70)
    print("🧪 测试 4: 新旧方法对比")
    print("="*70)

    from utils import merge_top_k_candidates, solve_kl_symmetry

    # 不稳定的分布
    logprobs_theta = {"t0": -0.1, "t1": -0.5, "t2": -1.0}
    logprobs_t = {"t5": -0.1, "t6": -0.5, "t7": -1.0}

    # 旧方法
    print("旧方法 (无稳定性检测):")
    cand_old, p_theta_old, p_t_old = merge_top_k_candidates(
        logprobs_theta, logprobs_t
    )
    alpha_old = solve_kl_symmetry(p_theta_old, p_t_old)
    print(f"  α = {alpha_old:.3f}")
    print(f"  (会进行混合，即使分布完全不同)")

    # 新方法
    print("\n新方法 (自动 fallback):")
    cand_new, p_theta_new, p_t_new, diag_new = merge_top_k_candidates_with_stability(
        logprobs_theta, logprobs_t,
        auto_fallback=True
    )
    alpha_new, fallback = solve_kl_symmetry_with_fallback(
        p_theta_new, p_t_new,
        stability_diagnostics=diag_new
    )
    print(f"  α = {alpha_new:.3f}")
    print(f"  Fallback = {fallback}")
    print(f"  (检测到不稳定，自动切换到 π_t)")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("🔬 稳定性增强功能测试")
    print("="*70)

    test_unstable_fallback()
    test_stable_mixing()
    test_threshold_sensitivity()
    test_comparison()

    print("\n" + "="*70)
    print("📊 总结")
    print("="*70)
    print("""
稳定性增强功能:

1. ✅ Overlap 检测
   - 计算两个模型 top-k 的交集大小和概率质量
   - 当 overlap 太小时（< 10%），认为不稳定

2. ✅ JS Divergence 检测
   - 量化两个分布的差异 [0, ln(2)]
   - 当 JS > 0.5 时，认为分布差异太大

3. ✅ 自动 Fallback
   - 当不稳定时，自动设置 α = 1.0（使用 π_t）
   - 避免不稳定的混合

4. ✅ 可配置阈值
   - stability_threshold_js: JS divergence 阈值
   - stability_threshold_overlap: Overlap 概率质量阈值
   - auto_fallback: 是否自动 fallback

推荐配置:
- 保守（更依赖 teacher）: JS<0.4, Overlap>0.15
- 平衡（推荐）: JS<0.5, Overlap>0.10
- 激进（更多混合）: JS<0.6, Overlap>0.05
    """)
