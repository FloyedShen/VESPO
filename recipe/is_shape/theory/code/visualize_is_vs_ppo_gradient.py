"""
IS Reshape vs PPO Clip: 梯度影响的完整可视化

重点分析：
1. 权重函数 f(w) 的形状
2. 梯度贡献 f(w) * A
3. 梯度方向和大小的变化
4. 训练动态中的行为差异
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os

# 创建输出目录
OUTPUT_DIR = '/Users/floyed/Downloads/offline_rl/figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 样式设置
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['font.size'] = 11

# 颜色方案
COLORS = {
    'gamma_0': '#27ae60',      # 绿色 - γ=0 (SFT)
    'gamma_0.3': '#3498db',    # 蓝色
    'gamma_0.5': '#9b59b6',    # 紫色
    'gamma_0.7': '#e74c3c',    # 红色
    'gamma_1': '#e67e22',      # 橙色 - γ=1 (RL)
    'ppo': '#1abc9c',          # 青色 - PPO
    'ppo_neg': '#e91e63',      # 粉色 - PPO A<0
    'no_clip': '#95a5a6',      # 灰色 - 无处理
}


# ============================================================
# 核心函数
# ============================================================

def f_is_reshape(w, gamma):
    """IS Reshape: f(w) = w^γ"""
    return np.power(np.maximum(w, 1e-10), gamma)

def f_ppo_clip(w, epsilon=0.2, A_sign='pos'):
    """PPO Clip 权重函数"""
    if A_sign == 'pos':
        return np.minimum(w, 1 + epsilon)
    else:
        return np.maximum(w, 1 - epsilon)

def gradient_is_reshape(w, gamma, A):
    """
    IS Reshape 的梯度贡献
    ∂L/∂θ ∝ f(w) * A * ∇log π

    这里我们分析 f(w) * A 的部分
    """
    return f_is_reshape(w, gamma) * A

def gradient_ppo(w, epsilon, A):
    """
    PPO 的梯度贡献
    当 A > 0: 使用 min(w, 1+ε)
    当 A < 0: 使用 max(w, 1-ε)
    """
    if A > 0:
        return f_ppo_clip(w, epsilon, 'pos') * A
    else:
        return f_ppo_clip(w, epsilon, 'neg') * A

def effective_gradient_direction(w, gamma):
    """
    有效梯度方向：∂f(w)/∂w
    IS Reshape: γ * w^(γ-1)
    """
    return gamma * np.power(np.maximum(w, 1e-10), gamma - 1)


# ============================================================
# 图1: 权重函数对比
# ============================================================
def plot_weight_functions():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    w = np.linspace(0.01, 3, 500)

    # --- (a) IS Reshape ---
    ax = axes[0]
    for gamma, color, label in [
        (0, COLORS['gamma_0'], 'γ=0 (SFT)'),
        (0.3, COLORS['gamma_0.3'], 'γ=0.3'),
        (0.5, COLORS['gamma_0.5'], 'γ=0.5'),
        (0.7, COLORS['gamma_0.7'], 'γ=0.7'),
        (1.0, COLORS['gamma_1'], 'γ=1 (RL)'),
    ]:
        ax.plot(w, f_is_reshape(w, gamma), color=color, linewidth=2.5, label=label)

    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=1, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('w = π_θ / μ', fontsize=12)
    ax.set_ylabel('f(w) = w^γ', fontsize=12)
    ax.set_title('(a) IS Reshape Weight Function', fontsize=13, fontweight='bold')
    ax.legend(loc='upper left')
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 3)

    # --- (b) PPO Clip ---
    ax = axes[1]
    ax.plot(w, f_ppo_clip(w, 0.2, 'pos'), color=COLORS['ppo'],
            linewidth=2.5, label='A > 0: min(w, 1+ε)')
    ax.plot(w, f_ppo_clip(w, 0.2, 'neg'), color=COLORS['ppo_neg'],
            linewidth=2.5, label='A < 0: max(w, 1-ε)')
    ax.plot(w, w, color=COLORS['no_clip'], linestyle=':', linewidth=2,
            alpha=0.7, label='w (no clip)')

    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=1, color='gray', linestyle='--', alpha=0.5)
    ax.fill_between([1.2, 3], [0, 0], [3, 3], color=COLORS['ppo'], alpha=0.1)
    ax.fill_between([0, 0.8], [0, 0], [3, 3], color=COLORS['ppo_neg'], alpha=0.1)

    ax.set_xlabel('w = π_θ / π_old', fontsize=12)
    ax.set_ylabel('f_PPO(w)', fontsize=12)
    ax.set_title('(b) PPO Clip Weight Function', fontsize=13, fontweight='bold')
    ax.legend(loc='upper left')
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 3)

    # --- (c) Direct Comparison ---
    ax = axes[2]
    ax.plot(w, f_is_reshape(w, 0.5), color=COLORS['gamma_0.5'],
            linewidth=2.5, label='IS γ=0.5')
    ax.plot(w, f_is_reshape(w, 0.3), color=COLORS['gamma_0.3'],
            linewidth=2.5, linestyle='--', label='IS γ=0.3')
    ax.plot(w, np.clip(w, 0.8, 1.2), color=COLORS['no_clip'],
            linewidth=2.5, label='PPO Clip (symmetric)')

    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=1, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('w', fontsize=12)
    ax.set_ylabel('f(w)', fontsize=12)
    ax.set_title('(c) IS Reshape vs PPO Clip', fontsize=13, fontweight='bold')
    ax.legend(loc='upper left')
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 2)

    # 添加注释
    ax.annotate('Smooth\ncompression', xy=(2.5, f_is_reshape(2.5, 0.5)),
                xytext=(2.0, 1.7), fontsize=10,
                arrowprops=dict(arrowstyle='->', color=COLORS['gamma_0.5']))
    ax.annotate('Hard\nclipping', xy=(2.5, 1.2),
                xytext=(2.5, 0.7), fontsize=10,
                arrowprops=dict(arrowstyle='->', color=COLORS['no_clip']))

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/fig1_weight_functions.png', dpi=150, bbox_inches='tight')
    plt.savefig(f'{OUTPUT_DIR}/fig1_weight_functions.pdf', bbox_inches='tight')
    print(f"Saved: fig1_weight_functions.png/pdf")
    plt.close()


# ============================================================
# 图2: 梯度贡献详细分析
# ============================================================
def plot_gradient_analysis():
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

    w = np.linspace(0.01, 3, 500)
    A_pos, A_neg = 1.0, -1.0
    epsilon = 0.2

    # ============ Row 1: Gradient contribution f(w)*A ============

    # (a) IS Reshape, A > 0
    ax = fig.add_subplot(gs[0, 0])
    for gamma, color in [(0.3, COLORS['gamma_0.3']), (0.5, COLORS['gamma_0.5']),
                          (0.7, COLORS['gamma_0.7']), (1.0, COLORS['gamma_1'])]:
        grad = gradient_is_reshape(w, gamma, A_pos)
        ax.plot(w, grad, color=color, linewidth=2, label=f'γ={gamma}')
    ax.axvline(x=1, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel('w')
    ax.set_ylabel('Gradient: f(w)·A')
    ax.set_title('(a) IS Reshape Gradient (A > 0)', fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 3)

    # (b) IS Reshape, A < 0
    ax = fig.add_subplot(gs[0, 1])
    for gamma, color in [(0.3, COLORS['gamma_0.3']), (0.5, COLORS['gamma_0.5']),
                          (0.7, COLORS['gamma_0.7']), (1.0, COLORS['gamma_1'])]:
        grad = gradient_is_reshape(w, gamma, A_neg)
        ax.plot(w, grad, color=color, linewidth=2, label=f'γ={gamma}')
    ax.axvline(x=1, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel('w')
    ax.set_ylabel('Gradient: f(w)·A')
    ax.set_title('(b) IS Reshape Gradient (A < 0)', fontweight='bold')
    ax.legend(fontsize=9, loc='lower left')
    ax.set_xlim(0, 3)
    ax.set_ylim(-3, 0)

    # (c) PPO Clip comparison
    ax = fig.add_subplot(gs[0, 2])
    # PPO A > 0
    grad_ppo_pos = np.array([gradient_ppo(wi, epsilon, A_pos) for wi in w])
    ax.plot(w, grad_ppo_pos, color=COLORS['ppo'], linewidth=2.5, label='PPO (A>0)')
    # PPO A < 0
    grad_ppo_neg = np.array([gradient_ppo(wi, epsilon, A_neg) for wi in w])
    ax.plot(w, grad_ppo_neg, color=COLORS['ppo_neg'], linewidth=2.5, label='PPO (A<0)')
    # No clip reference
    ax.plot(w, w * A_pos, color=COLORS['no_clip'], linestyle=':', linewidth=1.5, alpha=0.7, label='No clip')
    ax.plot(w, w * A_neg, color=COLORS['no_clip'], linestyle=':', linewidth=1.5, alpha=0.7)

    ax.axvline(x=1, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.axvline(x=1.2, color=COLORS['ppo'], linestyle='--', alpha=0.3)
    ax.axvline(x=0.8, color=COLORS['ppo_neg'], linestyle='--', alpha=0.3)
    ax.set_xlabel('w')
    ax.set_ylabel('Gradient: f(w)·A')
    ax.set_title('(c) PPO Clip Gradient', fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_xlim(0, 3)
    ax.set_ylim(-3, 3)

    # ============ Row 2: Gradient magnitude comparison ============

    # (d) Gradient magnitude |f(w)*A| for A > 0
    ax = fig.add_subplot(gs[1, 0])
    for gamma, color, ls in [(0.3, COLORS['gamma_0.3'], '-'),
                               (0.5, COLORS['gamma_0.5'], '-'),
                               (0.7, COLORS['gamma_0.7'], '--')]:
        ax.plot(w, np.abs(gradient_is_reshape(w, gamma, A_pos)),
                color=color, linewidth=2, linestyle=ls, label=f'IS γ={gamma}')
    ax.plot(w, np.abs(grad_ppo_pos), color=COLORS['ppo'], linewidth=2.5,
            linestyle=':', label='PPO Clip')

    ax.axvline(x=1, color='gray', linestyle='--', alpha=0.5)
    ax.fill_between(w[w > 1.2], np.abs(grad_ppo_pos)[w > 1.2],
                     np.abs(gradient_is_reshape(w, 0.5, A_pos))[w > 1.2],
                     color=COLORS['gamma_0.5'], alpha=0.2, label='IS preserves more')
    ax.set_xlabel('w')
    ax.set_ylabel('|Gradient|')
    ax.set_title('(d) Gradient Magnitude (A > 0)', fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 2.5)

    # (e) Relative gradient: f(w)/w (how much is preserved)
    ax = fig.add_subplot(gs[1, 1])
    for gamma, color in [(0.3, COLORS['gamma_0.3']), (0.5, COLORS['gamma_0.5']),
                          (0.7, COLORS['gamma_0.7'])]:
        ratio = f_is_reshape(w, gamma) / np.maximum(w, 1e-10)
        ax.plot(w, ratio, color=color, linewidth=2, label=f'IS γ={gamma}')

    # PPO ratio
    ppo_ratio = f_ppo_clip(w, epsilon, 'pos') / np.maximum(w, 1e-10)
    ax.plot(w, ppo_ratio, color=COLORS['ppo'], linewidth=2.5, linestyle='--', label='PPO')

    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5, label='Full IS (γ=1)')
    ax.axvline(x=1, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('w')
    ax.set_ylabel('f(w) / w')
    ax.set_title('(e) Gradient Retention Ratio', fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 1.5)

    # (f) Effective sample weight distribution
    ax = fig.add_subplot(gs[1, 2])
    # Simulate a log-normal w distribution
    np.random.seed(42)
    w_samples = np.random.lognormal(mean=0.3, sigma=0.5, size=1000)
    w_samples = np.clip(w_samples, 0.1, 5)

    for gamma, color in [(0.3, COLORS['gamma_0.3']), (0.5, COLORS['gamma_0.5']),
                          (0.7, COLORS['gamma_0.7'])]:
        weights = f_is_reshape(w_samples, gamma)
        weights = weights / weights.sum()  # Normalize
        ax.hist(weights * len(w_samples), bins=50, alpha=0.4, color=color,
                label=f'IS γ={gamma}', density=True)

    ax.set_xlabel('Normalized Weight')
    ax.set_ylabel('Density')
    ax.set_title('(f) Weight Distribution', fontweight='bold')
    ax.legend(fontsize=9)

    # ============ Row 3: Training dynamics ============

    # (g) Gradient variance as function of γ
    ax = fig.add_subplot(gs[2, 0])
    gammas = np.linspace(0, 1, 50)

    # Theoretical variance: E[w^{2γ}]
    # Under log-normal assumption: E[w^{2γ}] = exp(σ² * γ(2γ-1))
    sigma_sq_values = [0.5, 1.0, 2.0]
    for sigma_sq, ls in zip(sigma_sq_values, ['-', '--', ':']):
        var = np.exp(sigma_sq * gammas * (2*gammas - 1))
        ax.plot(gammas, var, linewidth=2, linestyle=ls, label=f'σ²={sigma_sq}')

    ax.axvline(x=0.25, color='red', linestyle='--', alpha=0.5)
    ax.text(0.27, 0.5, 'Var min\nat γ=0.25', fontsize=9, color='red')
    ax.set_xlabel('γ')
    ax.set_ylabel('Relative Variance E[w^{2γ}]')
    ax.set_title('(g) Gradient Variance vs γ', fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_yscale('log')

    # (h) ESS as function of γ
    ax = fig.add_subplot(gs[2, 1])
    for sigma_sq, ls, color in zip([0.5, 1.0, 2.0], ['-', '--', ':'],
                                     [COLORS['gamma_0.3'], COLORS['gamma_0.5'], COLORS['gamma_0.7']]):
        # ESS/n ≈ exp(-σ²γ²)
        ess_ratio = np.exp(-sigma_sq * gammas**2)
        ax.plot(gammas, ess_ratio, linewidth=2, linestyle=ls, color=color, label=f'σ²={sigma_sq}')

    ax.axhline(y=0.3, color='red', linestyle='--', alpha=0.5)
    ax.text(0.7, 0.35, 'ESS threshold', fontsize=9, color='red')
    ax.set_xlabel('γ')
    ax.set_ylabel('ESS / n')
    ax.set_title('(h) Effective Sample Size vs γ', fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # (i) Comparison summary
    ax = fig.add_subplot(gs[2, 2])
    ax.axis('off')

    summary_text = """
    Key Differences in Gradient Behavior:

    ┌─────────────────┬────────────────────┬────────────────────┐
    │                 │   IS Reshape       │    PPO Clip        │
    ├─────────────────┼────────────────────┼────────────────────┤
    │ High w samples  │ Smooth decay       │ Hard truncation    │
    │                 │ (preserves order)  │ (gradient = 0)     │
    ├─────────────────┼────────────────────┼────────────────────┤
    │ Low w samples   │ Symmetric decay    │ Only clip when A<0 │
    ├─────────────────┼────────────────────┼────────────────────┤
    │ Variance        │ Controllable by γ  │ Reduced by clip    │
    ├─────────────────┼────────────────────┼────────────────────┤
    │ Bias            │ Target shifts      │ Loses information  │
    │                 │ with γ             │ beyond threshold   │
    ├─────────────────┼────────────────────┼────────────────────┤
    │ Adaptivity      │ γ adjusts with ESS │ ε usually fixed    │
    └─────────────────┴────────────────────┴────────────────────┘
    """
    ax.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
            verticalalignment='center', transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    ax.set_title('(i) Comparison Summary', fontweight='bold')

    plt.savefig(f'{OUTPUT_DIR}/fig2_gradient_analysis.png', dpi=150, bbox_inches='tight')
    plt.savefig(f'{OUTPUT_DIR}/fig2_gradient_analysis.pdf', bbox_inches='tight')
    print(f"Saved: fig2_gradient_analysis.png/pdf")
    plt.close()


# ============================================================
# 图3: 梯度方向可视化 (向量场)
# ============================================================
def plot_gradient_vector_field():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 创建网格
    w_range = np.linspace(0.2, 2.5, 15)
    A_range = np.linspace(-1.5, 1.5, 15)
    W, A = np.meshgrid(w_range, A_range)

    epsilon = 0.2

    # --- (a) IS Reshape γ=0.5 ---
    ax = axes[0]
    gamma = 0.5

    # 梯度方向：∂L/∂θ ∝ w^γ * A * ∇log π
    # 简化为 (w^γ * A) 作为"梯度强度"
    grad_magnitude = f_is_reshape(W, gamma) * A

    # 归一化显示
    U = np.ones_like(grad_magnitude) * 0.1  # w 方向保持不变
    V = grad_magnitude / (np.abs(grad_magnitude).max() + 1e-10) * 0.3

    ax.quiver(W, A, U, V, grad_magnitude, cmap='RdYlGn', alpha=0.8)
    ax.axhline(y=0, color='black', linewidth=1)
    ax.axvline(x=1, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('w = π_θ / μ')
    ax.set_ylabel('Advantage A')
    ax.set_title(f'(a) IS Reshape γ={gamma}\nGradient Direction', fontweight='bold')

    # 添加颜色条
    sm = plt.cm.ScalarMappable(cmap='RdYlGn',
                                norm=plt.Normalize(vmin=-2, vmax=2))
    plt.colorbar(sm, ax=ax, label='Gradient magnitude')

    # --- (b) IS Reshape γ=0.3 ---
    ax = axes[1]
    gamma = 0.3

    grad_magnitude = f_is_reshape(W, gamma) * A
    U = np.ones_like(grad_magnitude) * 0.1
    V = grad_magnitude / (np.abs(grad_magnitude).max() + 1e-10) * 0.3

    ax.quiver(W, A, U, V, grad_magnitude, cmap='RdYlGn', alpha=0.8)
    ax.axhline(y=0, color='black', linewidth=1)
    ax.axvline(x=1, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('w = π_θ / μ')
    ax.set_ylabel('Advantage A')
    ax.set_title(f'(b) IS Reshape γ={gamma}\nGradient Direction', fontweight='bold')
    plt.colorbar(sm, ax=ax, label='Gradient magnitude')

    # --- (c) PPO Clip ---
    ax = axes[2]

    # PPO 的梯度
    grad_magnitude = np.zeros_like(W)
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            grad_magnitude[i, j] = gradient_ppo(W[i, j], epsilon, A[i, j])

    U = np.ones_like(grad_magnitude) * 0.1
    V = grad_magnitude / (np.abs(grad_magnitude).max() + 1e-10) * 0.3

    q = ax.quiver(W, A, U, V, grad_magnitude, cmap='RdYlGn', alpha=0.8)
    ax.axhline(y=0, color='black', linewidth=1)
    ax.axvline(x=1, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=1+epsilon, color=COLORS['ppo'], linestyle='--', alpha=0.5)
    ax.axvline(x=1-epsilon, color=COLORS['ppo_neg'], linestyle='--', alpha=0.5)

    # 标记截断区域
    ax.fill_betweenx(A_range, 1+epsilon, 2.5, alpha=0.1, color=COLORS['ppo'])
    ax.fill_betweenx(A_range, 0.2, 1-epsilon, alpha=0.1, color=COLORS['ppo_neg'])
    ax.text(2.0, 1.2, 'Clipped\n(A>0)', fontsize=9, ha='center', color=COLORS['ppo'])
    ax.text(0.5, -1.2, 'Clipped\n(A<0)', fontsize=9, ha='center', color=COLORS['ppo_neg'])

    ax.set_xlabel('w = π_θ / π_old')
    ax.set_ylabel('Advantage A')
    ax.set_title('(c) PPO Clip\nGradient Direction', fontweight='bold')
    plt.colorbar(sm, ax=ax, label='Gradient magnitude')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/fig3_gradient_vector_field.png', dpi=150, bbox_inches='tight')
    plt.savefig(f'{OUTPUT_DIR}/fig3_gradient_vector_field.pdf', bbox_inches='tight')
    print(f"Saved: fig3_gradient_vector_field.png/pdf")
    plt.close()


# ============================================================
# 图4: 训练动态对比
# ============================================================
def plot_training_dynamics():
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    w = np.linspace(0.01, 4, 500)

    # --- (a) w 分布随训练演变 ---
    ax = axes[0, 0]

    from scipy.stats import lognorm

    stages = [
        ('Step 0 (Initial)', 0.0, 0.3, COLORS['gamma_0']),
        ('Step 100', 0.2, 0.4, COLORS['gamma_0.3']),
        ('Step 500', 0.5, 0.6, COLORS['gamma_0.5']),
        ('Step 1000 (Late)', 0.8, 0.8, COLORS['gamma_0.7']),
    ]

    for label, mean_log, sigma, color in stages:
        dist = lognorm(s=sigma, scale=np.exp(mean_log))
        y = dist.pdf(w)
        ax.fill_between(w, 0, y, alpha=0.3, color=color, label=label)
        ax.plot(w, y, color=color, linewidth=2)

    ax.axvline(x=1.2, color='red', linestyle='--', linewidth=2, label='PPO clip (1+ε)')
    ax.set_xlabel('w = π_θ / μ')
    ax.set_ylabel('Density')
    ax.set_title('(a) Evolution of w Distribution During Training', fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 1.5)

    # --- (b) 被截断的样本比例 ---
    ax = axes[0, 1]

    steps = np.arange(0, 1001, 50)
    mean_logs = np.linspace(0, 0.8, len(steps))
    sigmas = np.linspace(0.3, 0.8, len(steps))

    clip_ratios = []
    for mean_log, sigma in zip(mean_logs, sigmas):
        dist = lognorm(s=sigma, scale=np.exp(mean_log))
        # P(w > 1.2)
        clip_ratio = 1 - dist.cdf(1.2)
        clip_ratios.append(clip_ratio)

    ax.plot(steps, clip_ratios, color=COLORS['ppo'], linewidth=2.5, label='PPO: P(w > 1+ε)')
    ax.fill_between(steps, 0, clip_ratios, color=COLORS['ppo'], alpha=0.2)

    ax.set_xlabel('Training Step')
    ax.set_ylabel('Fraction of Clipped Samples')
    ax.set_title('(b) Fraction of Samples Clipped by PPO', fontweight='bold')
    ax.legend()
    ax.set_xlim(0, 1000)
    ax.set_ylim(0, 0.8)

    # --- (c) 自适应 γ 的变化 ---
    ax = axes[1, 0]

    # 模拟 σ² 随训练增加
    sigma_sqs = np.linspace(0.1, 2.5, len(steps))

    # γ* = max(0, 1 - σ²/(2δ)), 假设 δ = 1
    delta = 1.0
    gamma_opt = np.maximum(0, 1 - sigma_sqs / (2 * delta))

    ax.plot(steps, gamma_opt, color=COLORS['gamma_0.5'], linewidth=2.5,
            label='Adaptive γ* (ESS-based)')
    ax.axhline(y=0.5, color=COLORS['gamma_0.3'], linestyle='--', alpha=0.7,
               label='Fixed γ=0.5')

    ax.fill_between(steps, gamma_opt, 0.5, where=gamma_opt < 0.5,
                     color=COLORS['gamma_0.5'], alpha=0.2, label='Adaptation benefit')

    ax.set_xlabel('Training Step')
    ax.set_ylabel('γ value')
    ax.set_title('(c) Adaptive γ During Training', fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1000)
    ax.set_ylim(0, 1)

    # --- (d) 梯度方差对比 ---
    ax = axes[1, 1]

    # IS Reshape 方差: E[w^{2γ}] ≈ exp(σ² * γ(2γ-1))
    gamma_fixed = 0.5
    var_is_fixed = np.exp(sigma_sqs * gamma_fixed * (2*gamma_fixed - 1))
    var_is_adaptive = np.exp(sigma_sqs * gamma_opt * (2*gamma_opt - 1))

    # PPO 方差 (简化模型)
    var_ppo = 1 + 0.3 * sigma_sqs  # 假设截断后方差线性增长但较慢

    ax.plot(steps, var_is_fixed, color=COLORS['gamma_0.5'], linewidth=2,
            linestyle='--', label='IS (fixed γ=0.5)')
    ax.plot(steps, var_is_adaptive, color=COLORS['gamma_0.5'], linewidth=2.5,
            label='IS (adaptive γ)')
    ax.plot(steps, var_ppo, color=COLORS['ppo'], linewidth=2.5,
            linestyle=':', label='PPO Clip')

    ax.set_xlabel('Training Step')
    ax.set_ylabel('Relative Gradient Variance')
    ax.set_title('(d) Gradient Variance During Training', fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1000)
    ax.set_yscale('log')
    ax.set_ylim(0.5, 10)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/fig4_training_dynamics.png', dpi=150, bbox_inches='tight')
    plt.savefig(f'{OUTPUT_DIR}/fig4_training_dynamics.pdf', bbox_inches='tight')
    print(f"Saved: fig4_training_dynamics.png/pdf")
    plt.close()


# ============================================================
# 图5: 综合对比大图
# ============================================================
def plot_comprehensive():
    fig = plt.figure(figsize=(18, 14))
    gs = GridSpec(4, 4, figure=fig, hspace=0.4, wspace=0.35)

    w = np.linspace(0.01, 3, 500)
    epsilon = 0.2

    # ============ Row 1: Weight functions ============
    ax1 = fig.add_subplot(gs[0, 0:2])
    for gamma, color, label in [(0, COLORS['gamma_0'], 'γ=0'),
                                  (0.3, COLORS['gamma_0.3'], 'γ=0.3'),
                                  (0.5, COLORS['gamma_0.5'], 'γ=0.5'),
                                  (0.7, COLORS['gamma_0.7'], 'γ=0.7'),
                                  (1.0, COLORS['gamma_1'], 'γ=1')]:
        ax1.plot(w, f_is_reshape(w, gamma), color=color, linewidth=2, label=label)
    ax1.axvline(x=1, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel('w')
    ax1.set_ylabel('f(w) = w^γ')
    ax1.set_title('(a) IS Reshape Weight Functions', fontweight='bold')
    ax1.legend(loc='upper left', ncol=2, fontsize=9)
    ax1.set_xlim(0, 3)
    ax1.set_ylim(0, 3)

    ax2 = fig.add_subplot(gs[0, 2:4])
    ax2.plot(w, f_ppo_clip(w, epsilon, 'pos'), color=COLORS['ppo'],
             linewidth=2.5, label='A > 0: min(w, 1+ε)')
    ax2.plot(w, f_ppo_clip(w, epsilon, 'neg'), color=COLORS['ppo_neg'],
             linewidth=2.5, label='A < 0: max(w, 1-ε)')
    ax2.plot(w, w, color=COLORS['no_clip'], linestyle=':', linewidth=2, alpha=0.7, label='No clip')
    ax2.axvline(x=1, color='gray', linestyle='--', alpha=0.5)
    ax2.fill_between([1+epsilon, 3], [0, 0], [3, 3], color=COLORS['ppo'], alpha=0.08)
    ax2.fill_between([0, 1-epsilon], [0, 0], [3, 3], color=COLORS['ppo_neg'], alpha=0.08)
    ax2.set_xlabel('w')
    ax2.set_ylabel('f_PPO(w)')
    ax2.set_title('(b) PPO Clip Weight Functions', fontweight='bold')
    ax2.legend(loc='upper left', fontsize=9)
    ax2.set_xlim(0, 3)
    ax2.set_ylim(0, 3)

    # ============ Row 2: Gradients (A > 0 and A < 0) ============
    ax3 = fig.add_subplot(gs[1, 0])
    for gamma, color in [(0.3, COLORS['gamma_0.3']), (0.5, COLORS['gamma_0.5']),
                          (0.7, COLORS['gamma_0.7']), (1.0, COLORS['gamma_1'])]:
        ax3.plot(w, gradient_is_reshape(w, gamma, 1.0), color=color, linewidth=2, label=f'γ={gamma}')
    ax3.axvline(x=1, color='gray', linestyle='--', alpha=0.5)
    ax3.set_xlabel('w')
    ax3.set_ylabel('f(w)·A')
    ax3.set_title('(c) IS Gradient (A>0)', fontweight='bold')
    ax3.legend(fontsize=8)
    ax3.set_xlim(0, 3)
    ax3.set_ylim(0, 3)

    ax4 = fig.add_subplot(gs[1, 1])
    for gamma, color in [(0.3, COLORS['gamma_0.3']), (0.5, COLORS['gamma_0.5']),
                          (0.7, COLORS['gamma_0.7']), (1.0, COLORS['gamma_1'])]:
        ax4.plot(w, gradient_is_reshape(w, gamma, -1.0), color=color, linewidth=2, label=f'γ={gamma}')
    ax4.axvline(x=1, color='gray', linestyle='--', alpha=0.5)
    ax4.set_xlabel('w')
    ax4.set_ylabel('f(w)·A')
    ax4.set_title('(d) IS Gradient (A<0)', fontweight='bold')
    ax4.legend(fontsize=8, loc='lower left')
    ax4.set_xlim(0, 3)
    ax4.set_ylim(-3, 0)

    ax5 = fig.add_subplot(gs[1, 2])
    grad_ppo_pos = np.array([gradient_ppo(wi, epsilon, 1.0) for wi in w])
    ax5.plot(w, grad_ppo_pos, color=COLORS['ppo'], linewidth=2.5, label='PPO')
    ax5.plot(w, w, color=COLORS['no_clip'], linestyle=':', linewidth=1.5, alpha=0.7, label='No clip')
    ax5.fill_between(w[w > 1+epsilon], grad_ppo_pos[w > 1+epsilon], w[w > 1+epsilon],
                      color=COLORS['ppo'], alpha=0.2)
    ax5.axvline(x=1+epsilon, color=COLORS['ppo'], linestyle='--', alpha=0.5)
    ax5.set_xlabel('w')
    ax5.set_ylabel('f(w)·A')
    ax5.set_title('(e) PPO Gradient (A>0)', fontweight='bold')
    ax5.legend(fontsize=8)
    ax5.set_xlim(0, 3)
    ax5.set_ylim(0, 3)

    ax6 = fig.add_subplot(gs[1, 3])
    grad_ppo_neg = np.array([gradient_ppo(wi, epsilon, -1.0) for wi in w])
    ax6.plot(w, grad_ppo_neg, color=COLORS['ppo_neg'], linewidth=2.5, label='PPO')
    ax6.plot(w, -w, color=COLORS['no_clip'], linestyle=':', linewidth=1.5, alpha=0.7, label='No clip')
    ax6.fill_between(w[w < 1-epsilon], grad_ppo_neg[w < 1-epsilon], -w[w < 1-epsilon],
                      color=COLORS['ppo_neg'], alpha=0.2)
    ax6.axvline(x=1-epsilon, color=COLORS['ppo_neg'], linestyle='--', alpha=0.5)
    ax6.set_xlabel('w')
    ax6.set_ylabel('f(w)·A')
    ax6.set_title('(f) PPO Gradient (A<0)', fontweight='bold')
    ax6.legend(fontsize=8, loc='lower left')
    ax6.set_xlim(0, 3)
    ax6.set_ylim(-3, 0)

    # ============ Row 3: Direct comparison and variance ============
    ax7 = fig.add_subplot(gs[2, 0:2])
    ax7.plot(w, f_is_reshape(w, 0.5), color=COLORS['gamma_0.5'], linewidth=2.5, label='IS γ=0.5')
    ax7.plot(w, f_is_reshape(w, 0.3), color=COLORS['gamma_0.3'], linewidth=2.5, linestyle='--', label='IS γ=0.3')
    ax7.plot(w, np.clip(w, 1-epsilon, 1+epsilon), color=COLORS['no_clip'], linewidth=2.5, label='PPO (symmetric)')
    ax7.axvline(x=1, color='gray', linestyle='--', alpha=0.5)
    ax7.set_xlabel('w')
    ax7.set_ylabel('f(w)')
    ax7.set_title('(g) Direct Comparison: IS vs PPO', fontweight='bold')
    ax7.legend(fontsize=10)
    ax7.set_xlim(0, 3)
    ax7.set_ylim(0, 2)
    ax7.annotate('Smooth\ncompression', xy=(2.3, f_is_reshape(2.3, 0.5)),
                 xytext=(1.8, 1.7), fontsize=10,
                 arrowprops=dict(arrowstyle='->', color=COLORS['gamma_0.5']))
    ax7.annotate('Hard\nclipping', xy=(2.3, 1.2), xytext=(2.5, 0.8), fontsize=10,
                 arrowprops=dict(arrowstyle='->', color=COLORS['no_clip']))

    ax8 = fig.add_subplot(gs[2, 2:4])
    gammas = np.linspace(0, 1, 50)
    for sigma_sq, color, ls in [(0.5, COLORS['gamma_0.3'], '-'),
                                  (1.0, COLORS['gamma_0.5'], '--'),
                                  (2.0, COLORS['gamma_0.7'], ':')]:
        var = np.exp(sigma_sq * gammas * (2*gammas - 1))
        ax8.plot(gammas, var, color=color, linewidth=2, linestyle=ls, label=f'σ²={sigma_sq}')
    ax8.axvline(x=0.25, color='red', linestyle='--', alpha=0.5)
    ax8.text(0.27, 0.3, 'Var min', fontsize=9, color='red')
    ax8.set_xlabel('γ')
    ax8.set_ylabel('Relative Variance E[w^{2γ}]')
    ax8.set_title('(h) Gradient Variance vs γ', fontweight='bold')
    ax8.legend(fontsize=10)
    ax8.set_yscale('log')
    ax8.set_xlim(0, 1)

    # ============ Row 4: Summary table ============
    ax9 = fig.add_subplot(gs[3, :])
    ax9.axis('off')

    table_data = [
        ['Property', 'IS Reshape (w^γ)', 'PPO Clip'],
        ['Weight function', 'f(w) = w^γ', 'clip(w, 1-ε, 1+ε)'],
        ['Transform type', 'Smooth power transform', 'Hard truncation'],
        ['Symmetry', 'Symmetric in w', 'Asymmetric (depends on A)'],
        ['High w handling', 'Smooth compression', 'Truncate to 1+ε'],
        ['Low w handling', 'Smooth compression', 'Truncate to 1-ε (only A<0)'],
        ['Theoretical basis', 'Generalized divergence D_α', 'Trust region constraint'],
        ['Parameter control', 'γ controls SFT-RL interpolation', 'ε controls trust region size'],
        ['Adaptivity', 'γ can adapt based on ESS', 'ε usually fixed'],
        ['Gradient info', 'Preserves relative ordering', 'Loses info beyond threshold'],
    ]

    table = ax9.table(cellText=table_data, loc='center', cellLoc='center',
                       colWidths=[0.22, 0.38, 0.38])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 2.0)

    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor('#34495e')
        table[(0, i)].set_text_props(color='white', fontweight='bold')

    for i in range(1, len(table_data)):
        for j in range(len(table_data[0])):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#ecf0f1')

    ax9.set_title('(i) Comprehensive Comparison Summary', fontweight='bold', fontsize=14, pad=20)

    plt.savefig(f'{OUTPUT_DIR}/fig5_comprehensive.png', dpi=150, bbox_inches='tight')
    plt.savefig(f'{OUTPUT_DIR}/fig5_comprehensive.pdf', bbox_inches='tight')
    print(f"Saved: fig5_comprehensive.png/pdf")
    plt.close()


# ============================================================
# 主函数
# ============================================================
if __name__ == '__main__':
    print("=" * 60)
    print("IS Reshape vs PPO Clip: Gradient Visualization")
    print("=" * 60)
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    plot_weight_functions()
    plot_gradient_analysis()
    plot_gradient_vector_field()
    plot_training_dynamics()
    plot_comprehensive()

    print()
    print("=" * 60)
    print("All figures generated successfully!")
    print(f"Location: {OUTPUT_DIR}/")
    print("=" * 60)
