"""
二维正态分布下的 q* 行为可视化

研究问题：
- 给定 π_θ 和 π_t 都是二维正态分布
- q_α = π_θ^α * π_t^(1-α) 的几何平均
- 可视化不同情况下 q* 的形状和位置
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from matplotlib.patches import Ellipse
import matplotlib.patches as mpatches

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def gaussian_2d(x, y, mu, cov):
    """计算二维高斯分布的概率密度"""
    pos = np.dstack((x, y))
    rv = multivariate_normal(mu, cov)
    return rv.pdf(pos)


def geometric_mean_gaussian(x, y, mu1, cov1, mu2, cov2, alpha):
    """
    计算两个高斯分布的几何平均

    q_α(x,y) ∝ N(x,y|μ₁,Σ₁)^α * N(x,y|μ₂,Σ₂)^(1-α)
    """
    p1 = gaussian_2d(x, y, mu1, cov1)
    p2 = gaussian_2d(x, y, mu2, cov2)

    # 几何平均（在概率密度上）
    q_alpha = (p1 ** alpha) * (p2 ** (1 - alpha))

    # 归一化（数值积分）
    # 注意：这不是严格的归一化，只是为了可视化
    q_alpha = q_alpha / (q_alpha.sum() + 1e-10)

    return q_alpha


def solve_kl_symmetry_gaussian(mu1, cov1, mu2, cov2, tol=1e-4, max_iter=50):
    """
    对于高斯分布，用二分法求解 KL 对称条件

    这是数值方法，因为几何平均的高斯不一定还是高斯
    """
    # 创建网格用于数值积分
    x_range = np.linspace(-5, 5, 100)
    y_range = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x_range, y_range)

    alpha_low, alpha_high = 0.0, 1.0

    for iteration in range(max_iter):
        alpha_mid = (alpha_low + alpha_high) / 2

        # 计算 q_α
        q_alpha = geometric_mean_gaussian(X, Y, mu1, cov1, mu2, cov2, alpha_mid)
        q_alpha = q_alpha / q_alpha.sum()  # 归一化

        # 计算 Δ(α) = E_{q_α}[log(π_t/π_θ)]
        p_theta = gaussian_2d(X, Y, mu1, cov1)
        p_t = gaussian_2d(X, Y, mu2, cov2)

        log_ratio = np.log(p_t + 1e-10) - np.log(p_theta + 1e-10)
        delta = (q_alpha * log_ratio).sum()

        # 更新区间
        if delta > 0:
            alpha_low = alpha_mid
        else:
            alpha_high = alpha_mid

        # 收敛检查
        if alpha_high - alpha_low < tol:
            break

    return (alpha_low + alpha_high) / 2


def plot_ellipse(ax, mu, cov, color, label, alpha=0.3):
    """绘制协方差椭圆"""
    # 特征值分解
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))

    # 2-sigma椭圆（覆盖约95%概率）
    width, height = 2 * 2 * np.sqrt(eigenvalues)

    ellipse = Ellipse(
        xy=mu,
        width=width,
        height=height,
        angle=angle,
        facecolor=color,
        edgecolor=color,
        alpha=alpha,
        linewidth=2,
        label=label
    )
    ax.add_patch(ellipse)


def visualize_case(mu_theta, cov_theta, mu_t, cov_t, title, ax):
    """可视化一个特定情况"""
    # 创建网格
    x_range = np.linspace(-5, 5, 200)
    y_range = np.linspace(-5, 5, 200)
    X, Y = np.meshgrid(x_range, y_range)

    # 计算 α*
    alpha_star = solve_kl_symmetry_gaussian(mu_theta, cov_theta, mu_t, cov_t)

    # 计算分布
    p_theta = gaussian_2d(X, Y, mu_theta, cov_theta)
    p_t = gaussian_2d(X, Y, mu_t, cov_t)
    q_star = geometric_mean_gaussian(X, Y, mu_theta, cov_theta, mu_t, cov_t, alpha_star)

    # 绘制等高线
    levels = 10

    # π_θ (蓝色)
    ax.contour(X, Y, p_theta, levels=levels, colors='blue', alpha=0.4, linewidths=1)

    # π_t (红色)
    ax.contour(X, Y, p_t, levels=levels, colors='red', alpha=0.4, linewidths=1)

    # q* (绿色，加粗)
    ax.contour(X, Y, q_star, levels=levels, colors='green', alpha=0.8, linewidths=2)

    # 绘制协方差椭圆
    plot_ellipse(ax, mu_theta, cov_theta, 'blue', r'$\pi_\theta$')
    plot_ellipse(ax, mu_t, cov_t, 'red', r'$\pi_t$')

    # 标记均值点
    ax.plot(mu_theta[0], mu_theta[1], 'bo', markersize=8, label=r'$\mu_\theta$')
    ax.plot(mu_t[0], mu_t[1], 'ro', markersize=8, label=r'$\mu_t$')

    # 设置标题和标签
    ax.set_title(f'{title}\n' + r'$\alpha^* = $' + f'{alpha_star:.3f}', fontsize=12)
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    # 图例（只在第一个子图显示）
    if ax == axes[0]:
        ax.legend(loc='upper right', fontsize=8)


# ============================================
# 可视化不同情况
# ============================================

fig, axes = plt.subplots(3, 3, figsize=(18, 18))
axes = axes.flatten()

# ============================================
# 情况1：相同方差，不同均值
# ============================================

# Case 1.1: 沿x轴分离
mu_theta = np.array([-2.0, 0.0])
cov_theta = np.array([[1.0, 0.0], [0.0, 1.0]])
mu_t = np.array([2.0, 0.0])
cov_t = np.array([[1.0, 0.0], [0.0, 1.0]])
visualize_case(mu_theta, cov_theta, mu_t, cov_t,
               '情况1: 相同方差, x轴分离', axes[0])

# Case 1.2: 沿y轴分离
mu_theta = np.array([0.0, -2.0])
cov_theta = np.array([[1.0, 0.0], [0.0, 1.0]])
mu_t = np.array([0.0, 2.0])
cov_t = np.array([[1.0, 0.0], [0.0, 1.0]])
visualize_case(mu_theta, cov_theta, mu_t, cov_t,
               '情况2: 相同方差, y轴分离', axes[1])

# Case 1.3: 对角分离
mu_theta = np.array([-1.5, -1.5])
cov_theta = np.array([[1.0, 0.0], [0.0, 1.0]])
mu_t = np.array([1.5, 1.5])
cov_t = np.array([[1.0, 0.0], [0.0, 1.0]])
visualize_case(mu_theta, cov_theta, mu_t, cov_t,
               '情况3: 相同方差, 对角分离', axes[2])

# ============================================
# 情况2：不同方差，相同均值
# ============================================

# Case 2.1: π_θ 窄, π_t 宽
mu_theta = np.array([0.0, 0.0])
cov_theta = np.array([[0.5, 0.0], [0.0, 0.5]])  # 窄
mu_t = np.array([0.0, 0.0])
cov_t = np.array([[2.0, 0.0], [0.0, 2.0]])  # 宽
visualize_case(mu_theta, cov_theta, mu_t, cov_t,
               '情况4: π_θ窄, π_t宽', axes[3])

# Case 2.2: π_θ 在x方向宽
mu_theta = np.array([0.0, 0.0])
cov_theta = np.array([[2.0, 0.0], [0.0, 0.5]])  # x宽y窄
mu_t = np.array([0.0, 0.0])
cov_t = np.array([[0.5, 0.0], [0.0, 2.0]])  # x窄y宽
visualize_case(mu_theta, cov_theta, mu_t, cov_t,
               '情况5: 方差方向相反', axes[4])

# Case 2.3: 极端差异
mu_theta = np.array([0.0, 0.0])
cov_theta = np.array([[0.2, 0.0], [0.0, 0.2]])  # 非常窄
mu_t = np.array([0.0, 0.0])
cov_t = np.array([[3.0, 0.0], [0.0, 3.0]])  # 非常宽
visualize_case(mu_theta, cov_theta, mu_t, cov_t,
               '情况6: 极端方差差异', axes[5])

# ============================================
# 情况3：不同均值 + 不同方差
# ============================================

# Case 3.1: π_θ 窄且左, π_t 宽且右
mu_theta = np.array([-1.5, 0.0])
cov_theta = np.array([[0.5, 0.0], [0.0, 0.5]])
mu_t = np.array([1.5, 0.0])
cov_t = np.array([[2.0, 0.0], [0.0, 2.0]])
visualize_case(mu_theta, cov_theta, mu_t, cov_t,
               '情况7: 均值分离 + 方差差异', axes[6])

# Case 3.2: 方向不一致
mu_theta = np.array([-1.0, -1.0])
cov_theta = np.array([[2.0, 0.0], [0.0, 0.5]])  # x宽
mu_t = np.array([1.0, 1.0])
cov_t = np.array([[0.5, 0.0], [0.0, 2.0]])  # y宽
visualize_case(mu_theta, cov_theta, mu_t, cov_t,
               '情况8: 方向 + 位置不一致', axes[7])

# Case 3.3: 有协方差（相关性）
mu_theta = np.array([-1.5, 0.0])
cov_theta = np.array([[1.0, 0.7], [0.7, 1.0]])  # 正相关
mu_t = np.array([1.5, 0.0])
cov_t = np.array([[1.0, -0.7], [-0.7, 1.0]])  # 负相关
visualize_case(mu_theta, cov_theta, mu_t, cov_t,
               '情况9: 相关性不同', axes[8])

# ============================================
# 全局设置
# ============================================

# 添加总标题
fig.suptitle(
    r'二维正态分布下的最优采样分布 $q^*$ 行为' + '\n' +
    r'蓝色: $\pi_\theta$, 红色: $\pi_t$, 绿色: $q^*$ (粗线)',
    fontsize=16,
    y=0.995
)

plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.savefig('/diancpfs/user/guobin/optimal_sampling/q_star_behavior_2d.png',
            dpi=150, bbox_inches='tight')
print("可视化已保存到: q_star_behavior_2d.png")

# ============================================
# 创建第二张图：详细分析一个情况
# ============================================

fig2, axes2 = plt.subplots(2, 3, figsize=(18, 12))

# 选择一个有趣的情况进行详细分析
mu_theta = np.array([-1.5, 0.0])
cov_theta = np.array([[0.5, 0.0], [0.0, 0.5]])
mu_t = np.array([1.5, 0.0])
cov_t = np.array([[2.0, 0.0], [0.0, 2.0]])

x_range = np.linspace(-5, 5, 200)
y_range = np.linspace(-5, 5, 200)
X, Y = np.meshgrid(x_range, y_range)

# 计算不同 α 值的 q_α
alphas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
alpha_star = solve_kl_symmetry_gaussian(mu_theta, cov_theta, mu_t, cov_t)

for idx, alpha in enumerate(alphas):
    ax = axes2.flatten()[idx]

    p_theta = gaussian_2d(X, Y, mu_theta, cov_theta)
    p_t = gaussian_2d(X, Y, mu_t, cov_t)
    q_alpha = geometric_mean_gaussian(X, Y, mu_theta, cov_theta, mu_t, cov_t, alpha)

    # 等高线
    ax.contour(X, Y, p_theta, levels=8, colors='blue', alpha=0.3, linewidths=1)
    ax.contour(X, Y, p_t, levels=8, colors='red', alpha=0.3, linewidths=1)
    ax.contourf(X, Y, q_alpha, levels=15, cmap='Greens', alpha=0.6)
    ax.contour(X, Y, q_alpha, levels=8, colors='darkgreen', linewidths=2)

    # 标记
    ax.plot(mu_theta[0], mu_theta[1], 'bo', markersize=10)
    ax.plot(mu_t[0], mu_t[1], 'ro', markersize=10)

    # 标题
    is_optimal = abs(alpha - alpha_star) < 0.05
    title_prefix = '★ ' if is_optimal else ''
    ax.set_title(f'{title_prefix}' + r'$\alpha = $' + f'{alpha:.1f}' +
                 (r' ($\alpha^*$)' if is_optimal else ''),
                 fontsize=12, fontweight='bold' if is_optimal else 'normal')

    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

fig2.suptitle(
    r'不同 $\alpha$ 值下的 $q_\alpha$ 演化' + '\n' +
    r'从 $\alpha=0$ ($q=\pi_t$) 到 $\alpha=1$ ($q=\pi_\theta$)',
    fontsize=16
)

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig('/diancpfs/user/guobin/optimal_sampling/q_alpha_evolution.png',
            dpi=150, bbox_inches='tight')
print("详细分析已保存到: q_alpha_evolution.png")

plt.show()

# ============================================
# 定量分析
# ============================================

print("\n" + "=" * 60)
print("定量分析总结")
print("=" * 60)

cases = [
    ("相同方差,x轴分离", np.array([-2, 0]), np.eye(2), np.array([2, 0]), np.eye(2)),
    ("π_θ窄,π_t宽", np.zeros(2), 0.5*np.eye(2), np.zeros(2), 2*np.eye(2)),
    ("方差方向相反", np.zeros(2), np.diag([2, 0.5]), np.zeros(2), np.diag([0.5, 2])),
    ("极端方差差异", np.zeros(2), 0.2*np.eye(2), np.zeros(2), 3*np.eye(2)),
    ("均值+方差综合", np.array([-1.5, 0]), 0.5*np.eye(2), np.array([1.5, 0]), 2*np.eye(2)),
]

print(f"\n{'情况':<20} {'α*':<10} {'KL(π_θ||π_t)':<15} {'KL(π_t||π_θ)':<15}")
print("-" * 60)

for name, mu1, cov1, mu2, cov2 in cases:
    alpha_star = solve_kl_symmetry_gaussian(mu1, cov1, mu2, cov2)

    # 计算KL散度（高斯分布的KL有闭式解）
    # D_KL(N(μ1,Σ1) || N(μ2,Σ2)) = 0.5 * [log(|Σ2|/|Σ1|) + tr(Σ2^-1 Σ1) + (μ2-μ1)ᵀΣ2^-1(μ2-μ1) - d]
    def kl_gaussian(mu1, cov1, mu2, cov2):
        d = len(mu1)
        cov2_inv = np.linalg.inv(cov2)
        term1 = np.log(np.linalg.det(cov2) / np.linalg.det(cov1))
        term2 = np.trace(cov2_inv @ cov1)
        delta_mu = mu2 - mu1
        term3 = delta_mu.T @ cov2_inv @ delta_mu
        return 0.5 * (term1 + term2 + term3 - d)

    kl_12 = kl_gaussian(mu1, cov1, mu2, cov2)
    kl_21 = kl_gaussian(mu2, cov2, mu1, cov1)

    print(f"{name:<20} {alpha_star:<10.3f} {kl_12:<15.3f} {kl_21:<15.3f}")

print("\n观察:")
print("1. α* 反映了两个分布的相对'强度'")
print("2. 当π_θ更集中(方差小)时, α*趋向更大(偏向π_θ)")
print("3. 当均值分离时, α*趋向0.5(对称)")
print("4. KL散度越不对称, α*偏离0.5越远")
