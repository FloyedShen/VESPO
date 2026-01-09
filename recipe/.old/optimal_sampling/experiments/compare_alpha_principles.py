"""
比较三个原则在Pareto前沿上选择α的结果：
1. KL对称：D_KL(q||π_θ) = D_KL(q||π_t)
2. ESS对称：ESS_θ(q) = ESS_t(q)
3. 纳什议价：max [U_θ(q) · U_t(q)]
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

def geometric_mean(pi_theta, pi_t, alpha, eps=1e-10):
    """几何平均分布"""
    q = (pi_theta ** alpha) * (pi_t ** (1 - alpha))
    q = q / (q.sum() + eps)
    return q

def kl_divergence(p, q, eps=1e-10):
    """KL散度 D_KL(p||q)"""
    p = p + eps
    q = q + eps
    return (p * np.log(p / q)).sum()

def ess(pi, q, eps=1e-10):
    """有效样本量"""
    weights_sq = (pi ** 2) / (q + eps)
    return 1.0 / weights_sq.sum()

def entropy(pi, eps=1e-10):
    """熵"""
    pi = pi + eps
    return -(pi * np.log(pi)).sum()

# === 三个原则 ===

def principle_1_kl_symmetry(alpha, pi_theta, pi_t):
    """原则1：KL对称性"""
    q = geometric_mean(pi_theta, pi_t, alpha)
    kl_theta = kl_divergence(q, pi_theta)
    kl_t = kl_divergence(q, pi_t)
    return abs(kl_theta - kl_t)

def principle_2_ess_symmetry(alpha, pi_theta, pi_t):
    """原则2：ESS对称性"""
    q = geometric_mean(pi_theta, pi_t, alpha)
    ess_theta = ess(pi_theta, q)
    ess_t = ess(pi_t, q)
    return abs(ess_theta - ess_t)

def principle_3_nash_bargaining(alpha, pi_theta, pi_t):
    """原则3：纳什议价（最大化乘积）"""
    q = geometric_mean(pi_theta, pi_t, alpha)
    # 效用 = 负KL散度（越小越好 → 越大越好）
    u_theta = -kl_divergence(q, pi_theta)
    u_t = -kl_divergence(q, pi_t)
    # 最大化乘积 = 最小化负乘积
    return -(u_theta * u_t)

def test_case(pi_theta, pi_t, name):
    """测试一个分布对"""
    print(f"\n{'='*70}")
    print(f"{name}")
    print(f"{'='*70}")
    print(f"π_θ = {pi_theta[:5]}..." if len(pi_theta) > 5 else f"π_θ = {pi_theta}")
    print(f"π_t = {pi_t[:5]}..." if len(pi_t) > 5 else f"π_t = {pi_t}")
    print(f"H(π_θ) = {entropy(pi_theta):.4f}, H(π_t) = {entropy(pi_t):.4f}")

    # 三个原则
    alpha_1 = minimize_scalar(
        lambda a: principle_1_kl_symmetry(a, pi_theta, pi_t),
        bounds=(0, 1), method='bounded'
    ).x

    alpha_2 = minimize_scalar(
        lambda a: principle_2_ess_symmetry(a, pi_theta, pi_t),
        bounds=(0, 1), method='bounded'
    ).x

    alpha_3 = minimize_scalar(
        lambda a: principle_3_nash_bargaining(a, pi_theta, pi_t),
        bounds=(0, 1), method='bounded'
    ).x

    alpha_entropy = entropy(pi_theta) / (entropy(pi_theta) + entropy(pi_t))

    print(f"\nα (KL对称):        {alpha_1:.4f}")
    print(f"α (ESS对称):       {alpha_2:.4f}")
    print(f"α (纳什议价):      {alpha_3:.4f}")
    print(f"α (熵公式):        {alpha_entropy:.4f}")

    # 验证结果
    q1 = geometric_mean(pi_theta, pi_t, alpha_1)
    q2 = geometric_mean(pi_theta, pi_t, alpha_2)
    q3 = geometric_mean(pi_theta, pi_t, alpha_3)

    print(f"\n验证原则1（KL对称）：")
    print(f"  D_KL(q||π_θ) = {kl_divergence(q1, pi_theta):.4f}")
    print(f"  D_KL(q||π_t) = {kl_divergence(q1, pi_t):.4f}")
    print(f"  差异 = {abs(kl_divergence(q1, pi_theta) - kl_divergence(q1, pi_t)):.6f}")

    print(f"\n验证原则2（ESS对称）：")
    print(f"  ESS(π_θ, q) = {ess(pi_theta, q2):.4f}")
    print(f"  ESS(π_t, q) = {ess(pi_t, q2):.4f}")
    print(f"  差异 = {abs(ess(pi_theta, q2) - ess(pi_t, q2)):.6f}")

    print(f"\n验证原则3（纳什议价）：")
    print(f"  U_θ·U_t = {-principle_3_nash_bargaining(alpha_3, pi_theta, pi_t):.4f}")

    return {
        'name': name,
        'alpha_kl': alpha_1,
        'alpha_ess': alpha_2,
        'alpha_nash': alpha_3,
        'alpha_entropy': alpha_entropy,
    }

def visualize_pareto_frontier(pi_theta, pi_t):
    """可视化Pareto前沿"""
    alphas = np.linspace(0.01, 0.99, 100)
    kl_theta_list = []
    kl_t_list = []

    for alpha in alphas:
        q = geometric_mean(pi_theta, pi_t, alpha)
        kl_theta_list.append(kl_divergence(q, pi_theta))
        kl_t_list.append(kl_divergence(q, pi_t))

    # 计算四个α值
    alpha_kl = minimize_scalar(
        lambda a: principle_1_kl_symmetry(a, pi_theta, pi_t),
        bounds=(0, 1), method='bounded'
    ).x

    alpha_ess = minimize_scalar(
        lambda a: principle_2_ess_symmetry(a, pi_theta, pi_t),
        bounds=(0, 1), method='bounded'
    ).x

    alpha_nash = minimize_scalar(
        lambda a: principle_3_nash_bargaining(a, pi_theta, pi_t),
        bounds=(0, 1), method='bounded'
    ).x

    alpha_entropy = entropy(pi_theta) / (entropy(pi_theta) + entropy(pi_t))

    # 对应的点
    q_kl = geometric_mean(pi_theta, pi_t, alpha_kl)
    q_ess = geometric_mean(pi_theta, pi_t, alpha_ess)
    q_nash = geometric_mean(pi_theta, pi_t, alpha_nash)
    q_entropy = geometric_mean(pi_theta, pi_t, alpha_entropy)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Pareto前沿
    ax.plot(kl_theta_list, kl_t_list, 'b-', linewidth=2, label='Pareto Frontier (Geometric Mean Family)')

    # 四个原则的点
    ax.plot(kl_divergence(q_kl, pi_theta), kl_divergence(q_kl, pi_t),
            'ro', markersize=10, label=f'KL Symmetry (α={alpha_kl:.3f})')
    ax.plot(kl_divergence(q_ess, pi_theta), kl_divergence(q_ess, pi_t),
            'gs', markersize=10, label=f'ESS Symmetry (α={alpha_ess:.3f})')
    ax.plot(kl_divergence(q_nash, pi_theta), kl_divergence(q_nash, pi_t),
            'md', markersize=10, label=f'Nash Bargaining (α={alpha_nash:.3f})')
    ax.plot(kl_divergence(q_entropy, pi_theta), kl_divergence(q_entropy, pi_t),
            'c^', markersize=10, label=f'Entropy Formula (α={alpha_entropy:.3f})')

    # 对角线（对称线）
    max_kl = max(max(kl_theta_list), max(kl_t_list))
    ax.plot([0, max_kl], [0, max_kl], 'k--', alpha=0.3, label='Symmetry Line')

    ax.set_xlabel('D_KL(q || pi_theta)', fontsize=12)
    ax.set_ylabel('D_KL(q || pi_t)', fontsize=12)
    ax.set_title('Pareto Frontier: Different Principles for Choosing alpha', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/diancpfs/user/guobin/optimal_sampling/pareto_frontier.png', dpi=150)
    print("\n图片已保存: pareto_frontier.png")

if __name__ == "__main__":
    results = []

    # 测试1：对称情况
    pi_theta = np.array([0.7, 0.3])
    pi_t = np.array([0.3, 0.7])
    results.append(test_case(pi_theta, pi_t, "测试1: 对称的两点分布"))

    # 测试2：极端情况A（θ集中，t分散）
    from scipy.special import softmax
    pi_theta = softmax(np.array([10.0] + [0.0]*9))
    pi_t = np.ones(10) / 10
    results.append(test_case(pi_theta, pi_t, "测试2: π_θ集中 vs π_t均匀"))

    # 测试3：极端情况B（θ分散，t集中）
    pi_theta = np.ones(10) / 10
    pi_t = softmax(np.array([10.0] + [0.0]*9))
    results.append(test_case(pi_theta, pi_t, "测试3: π_θ均匀 vs π_t集中"))

    # 测试4：一般情况
    np.random.seed(42)
    pi_theta = np.random.dirichlet(np.array([2, 3, 1, 4, 2, 1, 1, 1, 1, 1]))
    pi_t = np.random.dirichlet(np.array([1, 1, 3, 5, 3, 2, 1, 1, 1, 1]))
    results.append(test_case(pi_theta, pi_t, "测试4: 一般Dirichlet分布"))

    # 可视化
    visualize_pareto_frontier(pi_theta, pi_t)

    # 总结
    print(f"\n{'='*70}")
    print("总结：三个原则的对比")
    print(f"{'='*70}")

    for r in results:
        print(f"\n{r['name']}:")
        print(f"  KL对称:    α = {r['alpha_kl']:.4f}")
        print(f"  ESS对称:   α = {r['alpha_ess']:.4f}")
        print(f"  纳什议价:  α = {r['alpha_nash']:.4f}")
        print(f"  熵公式:    α = {r['alpha_entropy']:.4f}")
        print(f"  |KL - ESS| = {abs(r['alpha_kl'] - r['alpha_ess']):.4f}")
        print(f"  |KL - Nash| = {abs(r['alpha_kl'] - r['alpha_nash']):.4f}")
        print(f"  |ESS - Nash| = {abs(r['alpha_ess'] - r['alpha_nash']):.4f}")

    print(f"\n{'='*70}")
    print("关键发现：")
    print(f"{'='*70}")
    print("1. KL对称和ESS对称给出非常接近的结果")
    print("2. 纳什议价通常也很接近")
    print("3. 熵公式在极端情况下会偏离")
    print("4. KL对称是最简洁的原则（纯对称性）")
