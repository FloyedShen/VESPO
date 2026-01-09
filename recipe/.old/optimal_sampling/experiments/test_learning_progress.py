"""
探索：从"学习进展+鲁棒性"能得到什么？

核心问题：这条路径能否直接推导出几何平均族，而不需要"双重目标"的中间步骤？
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar, minimize
from scipy.special import softmax

def geometric_mean(pi_theta, pi_t, alpha):
    """几何平均"""
    q = (pi_theta ** alpha) * (pi_t ** (1 - alpha))
    return q / (q.sum() + 1e-10)

def compute_progress(pi_theta, pi_t, q, eta, r=None):
    """
    计算学习进展（简化版）

    Progress(q; η) ≈ η·Φ₁(q) - (η²/2)·Φ₂(q)

    其中：
    Φ₁(q) = Σ_y [π_t(y) - π_θ(y)] · w(y) · r(y)
    Φ₂(q) = Var_q[w(y)·r(y)]  (简化假设)
    """
    if r is None:
        # 默认：r(y) = log(π_t(y)/π_θ(y))
        r = np.log((pi_t + 1e-10) / (pi_theta + 1e-10))

    # 重要性权重
    w = pi_theta / (q + 1e-10)

    # 一阶项（信号）
    phi_1 = np.sum((pi_t - pi_theta) * w * r)

    # 二阶项（方差，简化计算）
    weighted_r = w * r
    phi_2 = np.sum(pi_theta * weighted_r**2) - np.sum(pi_theta * weighted_r)**2

    # 学习进展
    progress = eta * phi_1 - 0.5 * eta**2 * phi_2

    return progress, phi_1, phi_2

def optimal_eta(phi_1, phi_2):
    """给定Φ₁和Φ₂，最优学习率"""
    if phi_2 <= 0:
        return 0.0
    return phi_1 / phi_2

def test_learning_progress_framework():
    """测试：从学习进展出发的完整推导"""

    print("="*70)
    print("测试：学习进展 + 鲁棒性框架")
    print("="*70)

    # 测试分布
    pi_theta = softmax(np.array([3, 2, 1, 0.5, 0.2]))
    pi_t = softmax(np.array([0.5, 1, 2, 3, 2]))

    print(f"\nπ_θ = {pi_theta}")
    print(f"π_t = {pi_t}")

    # ========================================
    # 方法1：给定η，直接最大化Progress
    # ========================================
    print(f"\n{'='*70}")
    print("方法1：给定η，最大化Progress(q; η)")
    print(f"{'='*70}")

    for eta in [0.1, 0.5, 1.0]:
        # 在几何平均族内搜索
        def objective_fixed_eta(alpha):
            q = geometric_mean(pi_theta, pi_t, alpha)
            progress, _, _ = compute_progress(pi_theta, pi_t, q, eta)
            return -progress  # 最小化负值 = 最大化

        result = minimize_scalar(objective_fixed_eta, bounds=(0, 1), method='bounded')
        alpha_opt = result.x

        q_opt = geometric_mean(pi_theta, pi_t, alpha_opt)
        progress, phi_1, phi_2 = compute_progress(pi_theta, pi_t, q_opt, eta)

        print(f"\nη = {eta:.2f}:")
        print(f"  最优α = {alpha_opt:.4f}")
        print(f"  Progress = {progress:.6f}")
        print(f"  Φ₁ = {phi_1:.6f}, Φ₂ = {phi_2:.6f}")

    print(f"\n关键观察：最优α依赖于η的取值！")
    print(f"  → η越大，α越趋向于某个值（受方差项主导）")
    print(f"  → 如果η未知，无法确定唯一的α")

    # ========================================
    # 方法2：自适应最优η*
    # ========================================
    print(f"\n{'='*70}")
    print("方法2：对每个q，选择最优η*(q)，然后最大化Progress(q; η*(q))")
    print(f"{'='*70}")

    print(f"\n理论推导：")
    print(f"  ∂Progress/∂η = Φ₁(q) - η·Φ₂(q) = 0")
    print(f"  ⇒ η*(q) = Φ₁(q)/Φ₂(q)")
    print(f"  ")
    print(f"  代入：Progress(q; η*(q)) = Φ₁²(q)/(2Φ₂(q))")
    print(f"  ")
    print(f"  ⇒ max_q Progress(q; η*(q)) = max_q Φ₁²(q)/Φ₂(q)")
    print(f"                              = max_q SNR²(q)")
    print(f"                              ≡ max_q SNR(q)")

    # 在几何平均族内搜索
    def objective_optimal_eta(alpha):
        q = geometric_mean(pi_theta, pi_t, alpha)
        _, phi_1, phi_2 = compute_progress(pi_theta, pi_t, q, eta=1.0)  # eta doesn't matter

        if phi_2 <= 0:
            return 1e10

        # 最优进展
        optimal_progress = phi_1**2 / (2 * phi_2)
        return -optimal_progress

    result = minimize_scalar(objective_optimal_eta, bounds=(0, 1), method='bounded')
    alpha_snr = result.x

    q_snr = geometric_mean(pi_theta, pi_t, alpha_snr)
    _, phi_1_snr, phi_2_snr = compute_progress(pi_theta, pi_t, q_snr, eta=1.0)
    eta_opt = optimal_eta(phi_1_snr, phi_2_snr)
    snr = phi_1_snr / np.sqrt(phi_2_snr)

    print(f"\n结果：")
    print(f"  最优α (SNR最大化) = {alpha_snr:.4f}")
    print(f"  Φ₁ = {phi_1_snr:.6f}, Φ₂ = {phi_2_snr:.6f}")
    print(f"  SNR = {snr:.6f}")
    print(f"  对应的最优η* = {eta_opt:.6f}")
    print(f"  最优进展 = {phi_1_snr**2/(2*phi_2_snr):.6f}")

    # ========================================
    # 方法3：对比KL对称
    # ========================================
    print(f"\n{'='*70}")
    print("方法3：对比KL对称原则")
    print(f"{'='*70}")

    def kl_divergence(p, q):
        return np.sum(p * np.log((p + 1e-10) / (q + 1e-10)))

    def kl_symmetry_objective(alpha):
        q = geometric_mean(pi_theta, pi_t, alpha)
        kl_theta = kl_divergence(q, pi_theta)
        kl_t = kl_divergence(q, pi_t)
        return abs(kl_theta - kl_t)

    result = minimize_scalar(kl_symmetry_objective, bounds=(0, 1), method='bounded')
    alpha_kl = result.x

    q_kl = geometric_mean(pi_theta, pi_t, alpha_kl)
    _, phi_1_kl, phi_2_kl = compute_progress(pi_theta, pi_t, q_kl, eta=1.0)
    snr_kl = phi_1_kl / np.sqrt(phi_2_kl)

    print(f"\n最优α (KL对称) = {alpha_kl:.4f}")
    print(f"  Φ₁ = {phi_1_kl:.6f}, Φ₂ = {phi_2_kl:.6f}")
    print(f"  SNR = {snr_kl:.6f}")

    print(f"\n对比：")
    print(f"  |α_SNR - α_KL| = {abs(alpha_snr - alpha_kl):.4f}")
    print(f"  |SNR_SNR - SNR_KL| / SNR_SNR = {abs(snr - snr_kl)/snr * 100:.2f}%")

    # ========================================
    # 核心问题：几何平均族从哪里来？
    # ========================================
    print(f"\n{'='*70}")
    print("核心问题：SNR最大化为什么要在几何平均族内搜索？")
    print(f"{'='*70}")

    print(f"\n当前做法：")
    print(f"  1. 假设 q ∈ {{q_α: α∈[0,1]}} (几何平均族)")
    print(f"  2. 在此族内最大化SNR → α_SNR")
    print(f"  ")
    print(f"但问题是：")
    print(f"  - 为什么要限制在几何平均族？")
    print(f"  - SNR最大化本身不能导出这个族")
    print(f"  - 需要其他原则来justify几何平均族")

    # ========================================
    # 尝试：不限制族，直接优化
    # ========================================
    print(f"\n{'='*70}")
    print("尝试：在全空间优化SNR（不限制为几何平均）")
    print(f"{'='*70}")

    V = len(pi_theta)

    def snr_objective_full(q_unnorm):
        # q_unnorm: 未归一化的log(q)
        q = softmax(q_unnorm)

        _, phi_1, phi_2 = compute_progress(pi_theta, pi_t, q, eta=1.0)

        if phi_2 <= 0 or np.abs(phi_1) < 1e-10:
            return 1e10

        snr = phi_1 / np.sqrt(phi_2)
        return -snr  # 最小化负值

    # 初始化为几何平均 α=0.5
    q_init = geometric_mean(pi_theta, pi_t, 0.5)
    q_init_unnorm = np.log(q_init + 1e-10)

    result = minimize(snr_objective_full, q_init_unnorm, method='BFGS')
    q_full = softmax(result.x)

    _, phi_1_full, phi_2_full = compute_progress(pi_theta, pi_t, q_full, eta=1.0)
    snr_full = phi_1_full / np.sqrt(phi_2_full)

    print(f"\n全空间优化结果：")
    print(f"  SNR = {snr_full:.6f}")
    print(f"  q_full = {q_full}")

    # 检查是否接近几何平均族
    def find_closest_alpha(q):
        """找到最接近q的几何平均"""
        def distance(alpha):
            q_alpha = geometric_mean(pi_theta, pi_t, alpha)
            return np.sum((q - q_alpha)**2)
        result = minimize_scalar(distance, bounds=(0, 1), method='bounded')
        return result.x

    alpha_closest = find_closest_alpha(q_full)
    q_closest = geometric_mean(pi_theta, pi_t, alpha_closest)

    print(f"\n最接近的几何平均：")
    print(f"  α = {alpha_closest:.4f}")
    print(f"  ||q_full - q_α||² = {np.sum((q_full - q_closest)**2):.8f}")

    if np.sum((q_full - q_closest)**2) < 1e-4:
        print(f"  ✓ 全空间最优解在几何平均族内！")
    else:
        print(f"  ✗ 全空间最优解偏离几何平均族")

    print(f"\n{'='*70}")
    print("结论")
    print(f"{'='*70}")
    print(f"1. 学习进展+自适应η → SNR最大化 ✓ (理论严格)")
    print(f"2. SNR最大化 → 几何平均族 ??? (无法自动推出)")
    print(f"3. 需要额外原则justify几何平均族：")
    print(f"   - Pareto最优性（双重目标）")
    print(f"   - 信息几何（测地线）")
    print(f"   - 或者直接在全空间优化（但计算复杂）")

if __name__ == "__main__":
    test_learning_progress_framework()
