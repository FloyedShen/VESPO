"""
IS Reshape vs PPO Clip 可视化对比

对比内容：
1. 权重函数 f(w) 的形状
2. 梯度贡献的可视化
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as mpatches

# 设置中文字体和样式
plt.rcParams['font.family'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

# 颜色方案
COLORS = {
    'gamma_0': '#2ecc71',      # 绿色 - γ=0 (SFT)
    'gamma_0.3': '#3498db',    # 蓝色
    'gamma_0.5': '#9b59b6',    # 紫色
    'gamma_0.7': '#e74c3c',    # 红色
    'gamma_1': '#e67e22',      # 橙色 - γ=1 (RL)
    'ppo_pos': '#1abc9c',      # 青色 - PPO A>0
    'ppo_neg': '#e91e63',      # 粉色 - PPO A<0
    'ppo_sym': '#607d8b',      # 灰色 - PPO 对称
}


def f_is_reshape(w, gamma):
    """IS Reshape 权重函数: f(w) = w^γ"""
    return np.power(w, gamma)


def f_ppo_clip(w, epsilon_low=0.2, epsilon_high=0.27, A_sign=None):
    """
    PPO Clip 权重函数
    A_sign: 'pos' for A>0, 'neg' for A<0, None for symmetric
    """
    if A_sign == 'pos':
        # A > 0: min(w, 1+ε)
        return np.minimum(w, 1 + epsilon_high)
    elif A_sign == 'neg':
        # A < 0: max(w, 1-ε)
        return np.maximum(w, 1 - epsilon_low)
    else:
        # 对称截断
        return np.clip(w, 1 - epsilon_low, 1 + epsilon_high)


def gradient_contribution(w, f_w, advantage):
    """
    梯度贡献 = f(w) * A * ∇log π
    这里我们可视化 f(w) * A 的部分（假设 ∇log π = 1）
    """
    return f_w * advantage


# ============================================================
# 图1: 权重函数 f(w) 对比
# ============================================================
def plot_weight_functions():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    w = np.linspace(0.01, 3, 500)

    # --- 子图1: IS Reshape 不同 γ ---
    ax1 = axes[0]
    gammas = [0, 0.3, 0.5, 0.7, 1.0]
    gamma_colors = ['gamma_0', 'gamma_0.3', 'gamma_0.5', 'gamma_0.7', 'gamma_1']

    for gamma, color_key in zip(gammas, gamma_colors):
        f_w = f_is_reshape(w, gamma)
        label = f'γ={gamma}' + (' (SFT)' if gamma == 0 else ' (RL)' if gamma == 1 else '')
        ax1.plot(w, f_w, color=COLORS[color_key], linewidth=2.5, label=label)

    ax1.axhline(y=1, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax1.axvline(x=1, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax1.set_xlabel('w = π_θ / μ', fontsize=12)
    ax1.set_ylabel('f(w) = w^γ', fontsize=12)
    ax1.set_title('IS Reshape: 不同 γ 的权重函数', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=10)
    ax1.set_xlim(0, 3)
    ax1.set_ylim(0, 3)

    # --- 子图2: PPO Clip (取决于 A 的符号) ---
    ax2 = axes[1]

    # PPO A > 0
    f_ppo_pos = f_ppo_clip(w, A_sign='pos')
    ax2.plot(w, f_ppo_pos, color=COLORS['ppo_pos'], linewidth=2.5,
             label='PPO (A > 0): min(w, 1+ε)')

    # PPO A < 0
    f_ppo_neg = f_ppo_clip(w, A_sign='neg')
    ax2.plot(w, f_ppo_neg, color=COLORS['ppo_neg'], linewidth=2.5,
             label='PPO (A < 0): max(w, 1-ε)')

    # 参考线: w 本身
    ax2.plot(w, w, color='gray', linestyle=':', linewidth=1.5, alpha=0.7, label='w (无截断)')

    ax2.axhline(y=1, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax2.axvline(x=1, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax2.axhline(y=1.27, color=COLORS['ppo_pos'], linestyle='--', alpha=0.3, linewidth=1)
    ax2.axhline(y=0.8, color=COLORS['ppo_neg'], linestyle='--', alpha=0.3, linewidth=1)

    ax2.set_xlabel('w = π_θ / π_old', fontsize=12)
    ax2.set_ylabel('f_PPO(w)', fontsize=12)
    ax2.set_title('PPO Clip: 依赖 Advantage 符号', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=10)
    ax2.set_xlim(0, 3)
    ax2.set_ylim(0, 3)

    # 添加截断区域标注
    ax2.fill_between([1.27, 3], [0, 0], [3, 3], color=COLORS['ppo_pos'], alpha=0.1)
    ax2.fill_between([0, 0.8], [0, 0], [3, 3], color=COLORS['ppo_neg'], alpha=0.1)
    ax2.text(2.0, 0.3, '截断区\n(A>0)', fontsize=9, ha='center', color=COLORS['ppo_pos'])
    ax2.text(0.4, 0.3, '截断区\n(A<0)', fontsize=9, ha='center', color=COLORS['ppo_neg'])

    # --- 子图3: 直接对比 ---
    ax3 = axes[2]

    # IS Reshape γ=0.5
    ax3.plot(w, f_is_reshape(w, 0.5), color=COLORS['gamma_0.5'], linewidth=2.5,
             label='IS Reshape γ=0.5')
    ax3.plot(w, f_is_reshape(w, 0.3), color=COLORS['gamma_0.3'], linewidth=2.5,
             label='IS Reshape γ=0.3', linestyle='--')

    # PPO 对称截断
    f_ppo_sym = f_ppo_clip(w, A_sign=None)
    ax3.plot(w, f_ppo_sym, color=COLORS['ppo_sym'], linewidth=2.5,
             label='PPO Clip (对称)')

    ax3.axhline(y=1, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax3.axvline(x=1, color='gray', linestyle='--', alpha=0.5, linewidth=1)

    ax3.set_xlabel('w', fontsize=12)
    ax3.set_ylabel('f(w)', fontsize=12)
    ax3.set_title('IS Reshape vs PPO Clip 对比', fontsize=14, fontweight='bold')
    ax3.legend(loc='upper left', fontsize=10)
    ax3.set_xlim(0, 3)
    ax3.set_ylim(0, 2)

    # 添加注释
    ax3.annotate('平滑压缩', xy=(2.5, f_is_reshape(2.5, 0.5)),
                 xytext=(2.2, 1.8), fontsize=9,
                 arrowprops=dict(arrowstyle='->', color=COLORS['gamma_0.5']))
    ax3.annotate('硬截断', xy=(2.5, f_ppo_clip(2.5, A_sign=None)),
                 xytext=(2.5, 0.8), fontsize=9,
                 arrowprops=dict(arrowstyle='->', color=COLORS['ppo_sym']))

    plt.tight_layout()
    plt.savefig('/Users/floyed/Downloads/offline_rl/fig1_weight_functions.png',
                dpi=150, bbox_inches='tight')
    plt.savefig('/Users/floyed/Downloads/offline_rl/fig1_weight_functions.pdf',
                bbox_inches='tight')
    print("图1已保存: fig1_weight_functions.png/pdf")
    plt.close()


# ============================================================
# 图2: 梯度贡献可视化
# ============================================================
def plot_gradient_contributions():
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    w = np.linspace(0.01, 3, 500)

    # 上排: A > 0 (正 Advantage)
    A_pos = 1.0

    # --- IS Reshape 梯度 (A > 0) ---
    ax = axes[0, 0]
    gammas = [0, 0.3, 0.5, 0.7, 1.0]
    gamma_colors = ['gamma_0', 'gamma_0.3', 'gamma_0.5', 'gamma_0.7', 'gamma_1']

    for gamma, color_key in zip(gammas, gamma_colors):
        f_w = f_is_reshape(w, gamma)
        grad = gradient_contribution(w, f_w, A_pos)
        ax.plot(w, grad, color=COLORS[color_key], linewidth=2, label=f'γ={gamma}')

    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3, linewidth=1)
    ax.axvline(x=1, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.set_xlabel('w', fontsize=11)
    ax.set_ylabel('梯度贡献 f(w)·A', fontsize=11)
    ax.set_title('IS Reshape 梯度 (A > 0)', fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 3)

    # --- PPO 梯度 (A > 0) ---
    ax = axes[0, 1]

    f_ppo = f_ppo_clip(w, A_sign='pos')
    grad_ppo = gradient_contribution(w, f_ppo, A_pos)
    ax.plot(w, grad_ppo, color=COLORS['ppo_pos'], linewidth=2.5, label='PPO Clip')

    # 对比: 无截断
    grad_no_clip = gradient_contribution(w, w, A_pos)
    ax.plot(w, grad_no_clip, color='gray', linestyle=':', linewidth=1.5,
            alpha=0.7, label='无截断 (w)')

    ax.fill_between(w[w > 1.27], grad_ppo[w > 1.27], grad_no_clip[w > 1.27],
                    color=COLORS['ppo_pos'], alpha=0.2)

    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3, linewidth=1)
    ax.axvline(x=1, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.axvline(x=1.27, color=COLORS['ppo_pos'], linestyle='--', alpha=0.5, linewidth=1)

    ax.set_xlabel('w', fontsize=11)
    ax.set_ylabel('梯度贡献 f(w)·A', fontsize=11)
    ax.set_title('PPO Clip 梯度 (A > 0)', fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 3)
    ax.text(2.0, 2.5, '被截断的\n梯度贡献', fontsize=9, ha='center',
            color=COLORS['ppo_pos'], alpha=0.8)

    # --- 对比图 (A > 0) ---
    ax = axes[0, 2]

    # IS Reshape
    for gamma, color_key in [(0.3, 'gamma_0.3'), (0.5, 'gamma_0.5')]:
        f_w = f_is_reshape(w, gamma)
        grad = gradient_contribution(w, f_w, A_pos)
        ax.plot(w, grad, color=COLORS[color_key], linewidth=2, label=f'IS γ={gamma}')

    # PPO
    ax.plot(w, grad_ppo, color=COLORS['ppo_pos'], linewidth=2,
            linestyle='--', label='PPO Clip')

    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3, linewidth=1)
    ax.axvline(x=1, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.set_xlabel('w', fontsize=11)
    ax.set_ylabel('梯度贡献', fontsize=11)
    ax.set_title('对比: IS Reshape vs PPO (A > 0)', fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 2)

    # 下排: A < 0 (负 Advantage)
    A_neg = -1.0

    # --- IS Reshape 梯度 (A < 0) ---
    ax = axes[1, 0]

    for gamma, color_key in zip(gammas, gamma_colors):
        f_w = f_is_reshape(w, gamma)
        grad = gradient_contribution(w, f_w, A_neg)
        ax.plot(w, grad, color=COLORS[color_key], linewidth=2, label=f'γ={gamma}')

    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3, linewidth=1)
    ax.axvline(x=1, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.set_xlabel('w', fontsize=11)
    ax.set_ylabel('梯度贡献 f(w)·A', fontsize=11)
    ax.set_title('IS Reshape 梯度 (A < 0)', fontsize=12, fontweight='bold')
    ax.legend(loc='lower left', fontsize=9)
    ax.set_xlim(0, 3)
    ax.set_ylim(-3, 0)

    # --- PPO 梯度 (A < 0) ---
    ax = axes[1, 1]

    f_ppo_neg = f_ppo_clip(w, A_sign='neg')
    grad_ppo_neg = gradient_contribution(w, f_ppo_neg, A_neg)
    ax.plot(w, grad_ppo_neg, color=COLORS['ppo_neg'], linewidth=2.5, label='PPO Clip')

    # 对比: 无截断
    grad_no_clip_neg = gradient_contribution(w, w, A_neg)
    ax.plot(w, grad_no_clip_neg, color='gray', linestyle=':', linewidth=1.5,
            alpha=0.7, label='无截断 (w)')

    ax.fill_between(w[w < 0.8], grad_ppo_neg[w < 0.8], grad_no_clip_neg[w < 0.8],
                    color=COLORS['ppo_neg'], alpha=0.2)

    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3, linewidth=1)
    ax.axvline(x=1, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.axvline(x=0.8, color=COLORS['ppo_neg'], linestyle='--', alpha=0.5, linewidth=1)

    ax.set_xlabel('w', fontsize=11)
    ax.set_ylabel('梯度贡献 f(w)·A', fontsize=11)
    ax.set_title('PPO Clip 梯度 (A < 0)', fontsize=12, fontweight='bold')
    ax.legend(loc='lower left', fontsize=9)
    ax.set_xlim(0, 3)
    ax.set_ylim(-3, 0)
    ax.text(0.4, -2.5, '被截断的\n梯度贡献', fontsize=9, ha='center',
            color=COLORS['ppo_neg'], alpha=0.8)

    # --- 对比图 (A < 0) ---
    ax = axes[1, 2]

    # IS Reshape
    for gamma, color_key in [(0.3, 'gamma_0.3'), (0.5, 'gamma_0.5')]:
        f_w = f_is_reshape(w, gamma)
        grad = gradient_contribution(w, f_w, A_neg)
        ax.plot(w, grad, color=COLORS[color_key], linewidth=2, label=f'IS γ={gamma}')

    # PPO
    ax.plot(w, grad_ppo_neg, color=COLORS['ppo_neg'], linewidth=2,
            linestyle='--', label='PPO Clip')

    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3, linewidth=1)
    ax.axvline(x=1, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.set_xlabel('w', fontsize=11)
    ax.set_ylabel('梯度贡献', fontsize=11)
    ax.set_title('对比: IS Reshape vs PPO (A < 0)', fontsize=12, fontweight='bold')
    ax.legend(loc='lower left', fontsize=9)
    ax.set_xlim(0, 3)
    ax.set_ylim(-2, 0)

    plt.tight_layout()
    plt.savefig('/Users/floyed/Downloads/offline_rl/fig2_gradient_contributions.png',
                dpi=150, bbox_inches='tight')
    plt.savefig('/Users/floyed/Downloads/offline_rl/fig2_gradient_contributions.pdf',
                bbox_inches='tight')
    print("图2已保存: fig2_gradient_contributions.png/pdf")
    plt.close()


# ============================================================
# 图3: 关键差异示意图
# ============================================================
def plot_key_differences():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    w = np.linspace(0.01, 3, 500)

    # --- 左图: 高 w 区域的处理差异 ---
    ax = axes[0]

    # 放大高 w 区域
    w_high = np.linspace(1, 3, 300)

    for gamma, color_key, ls in [(0.3, 'gamma_0.3', '-'),
                                   (0.5, 'gamma_0.5', '-'),
                                   (0.7, 'gamma_0.7', '-')]:
        f_w = f_is_reshape(w_high, gamma)
        ax.plot(w_high, f_w, color=COLORS[color_key], linewidth=2.5,
                linestyle=ls, label=f'IS Reshape γ={gamma}')

    # PPO clip (A > 0)
    f_ppo = f_ppo_clip(w_high, A_sign='pos')
    ax.plot(w_high, f_ppo, color=COLORS['ppo_pos'], linewidth=3,
            linestyle='--', label='PPO Clip (A>0)')

    # 无截断参考
    ax.plot(w_high, w_high, color='gray', linestyle=':', linewidth=1.5,
            alpha=0.5, label='w (无处理)')

    ax.axhline(y=1, color='gray', linestyle='-', alpha=0.3, linewidth=1)
    ax.axhline(y=1.27, color=COLORS['ppo_pos'], linestyle='--', alpha=0.3, linewidth=1)
    ax.axvline(x=1.27, color=COLORS['ppo_pos'], linestyle='--', alpha=0.3, linewidth=1)

    ax.set_xlabel('w (重要性权重)', fontsize=12)
    ax.set_ylabel('有效权重 f(w)', fontsize=12)
    ax.set_title('高 w 区域: 平滑压缩 vs 硬截断', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.set_xlim(1, 3)
    ax.set_ylim(0.8, 3)

    # 添加注释箭头
    ax.annotate('IS Reshape:\n平滑衰减，\n保留梯度信息',
                xy=(2.5, f_is_reshape(2.5, 0.5)),
                xytext=(2.0, 2.3),
                fontsize=10, ha='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                arrowprops=dict(arrowstyle='->', color='black'))

    ax.annotate('PPO Clip:\n硬截断，\nw>1+ε 时梯度为0',
                xy=(2.5, 1.27),
                xytext=(2.5, 0.95),
                fontsize=10, ha='center',
                bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.5),
                arrowprops=dict(arrowstyle='->', color='black'))

    # --- 右图: 训练动态示意 ---
    ax = axes[1]

    # 模拟训练过程中 w 分布的变化
    steps = ['Step 0\n(初始)', 'Step 100', 'Step 500', 'Step 1000\n(后期)']
    w_means = [1.0, 1.3, 1.8, 2.5]
    w_stds = [0.2, 0.4, 0.6, 0.8]

    colors_steps = ['#27ae60', '#2980b9', '#8e44ad', '#c0392b']

    x_plot = np.linspace(0, 4, 500)

    for i, (step, mean, std, color) in enumerate(zip(steps, w_means, w_stds, colors_steps)):
        # 对数正态分布模拟 w 的分布
        from scipy.stats import lognorm
        s = std / mean  # shape parameter
        scale = mean
        y = lognorm.pdf(x_plot, s, scale=scale)
        y = y / y.max() * (1 - i * 0.15)  # 归一化并稍微降低后期的峰值
        ax.fill_between(x_plot, y * 0, y, alpha=0.3, color=color)
        ax.plot(x_plot, y, color=color, linewidth=2, label=step)

    # 添加 PPO 截断线
    ax.axvline(x=1.27, color=COLORS['ppo_pos'], linewidth=2, linestyle='--',
               label='PPO 截断边界 (1+ε)')

    ax.set_xlabel('w = π_θ / μ', fontsize=12)
    ax.set_ylabel('w 的分布密度', fontsize=12)
    ax.set_title('训练过程中 w 分布的演变', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 1.1)

    # 添加说明文字
    ax.text(2.5, 0.85, '随着训练进行:\n• w 均值增大\n• w 方差增大\n• 更多样本被 PPO 截断\n• IS Reshape 自适应调整 γ',
            fontsize=10, ha='left', va='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    plt.savefig('/Users/floyed/Downloads/offline_rl/fig3_key_differences.png',
                dpi=150, bbox_inches='tight')
    plt.savefig('/Users/floyed/Downloads/offline_rl/fig3_key_differences.pdf',
                bbox_inches='tight')
    print("图3已保存: fig3_key_differences.png/pdf")
    plt.close()


# ============================================================
# 图4: 综合对比总图
# ============================================================
def plot_comprehensive_comparison():
    fig = plt.figure(figsize=(16, 12))

    # 创建复杂布局
    gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.3)

    w = np.linspace(0.01, 3, 500)

    # ============ 第一行: 权重函数 ============
    # (1,1) IS Reshape 权重函数
    ax1 = fig.add_subplot(gs[0, 0:2])
    gammas = [0, 0.3, 0.5, 0.7, 1.0]
    gamma_colors = ['gamma_0', 'gamma_0.3', 'gamma_0.5', 'gamma_0.7', 'gamma_1']
    for gamma, color_key in zip(gammas, gamma_colors):
        f_w = f_is_reshape(w, gamma)
        label = f'γ={gamma}'
        ax1.plot(w, f_w, color=COLORS[color_key], linewidth=2, label=label)
    ax1.axvline(x=1, color='gray', linestyle='--', alpha=0.5)
    ax1.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel('w', fontsize=11)
    ax1.set_ylabel('f(w) = w^γ', fontsize=11)
    ax1.set_title('(a) IS Reshape 权重函数', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9, ncol=2)
    ax1.set_xlim(0, 3)
    ax1.set_ylim(0, 3)

    # (1,2) PPO Clip 权重函数
    ax2 = fig.add_subplot(gs[0, 2:4])
    f_ppo_pos = f_ppo_clip(w, A_sign='pos')
    f_ppo_neg = f_ppo_clip(w, A_sign='neg')
    ax2.plot(w, f_ppo_pos, color=COLORS['ppo_pos'], linewidth=2.5, label='A > 0: min(w, 1+ε)')
    ax2.plot(w, f_ppo_neg, color=COLORS['ppo_neg'], linewidth=2.5, label='A < 0: max(w, 1-ε)')
    ax2.plot(w, w, color='gray', linestyle=':', linewidth=1.5, alpha=0.7, label='w (无截断)')
    ax2.axvline(x=1, color='gray', linestyle='--', alpha=0.5)
    ax2.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    ax2.fill_between([1.27, 3], [0, 0], [3, 3], color=COLORS['ppo_pos'], alpha=0.08)
    ax2.fill_between([0, 0.8], [0, 0], [3, 3], color=COLORS['ppo_neg'], alpha=0.08)
    ax2.set_xlabel('w', fontsize=11)
    ax2.set_ylabel('f_PPO(w)', fontsize=11)
    ax2.set_title('(b) PPO Clip 权重函数 (A-不对称)', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=9)
    ax2.set_xlim(0, 3)
    ax2.set_ylim(0, 3)

    # ============ 第二行: 梯度贡献 ============
    A_pos, A_neg = 1.0, -1.0

    # (2,1) IS Reshape 梯度 A>0
    ax3 = fig.add_subplot(gs[1, 0])
    for gamma, color_key in zip([0.3, 0.5, 0.7, 1.0],
                                 ['gamma_0.3', 'gamma_0.5', 'gamma_0.7', 'gamma_1']):
        grad = f_is_reshape(w, gamma) * A_pos
        ax3.plot(w, grad, color=COLORS[color_key], linewidth=2, label=f'γ={gamma}')
    ax3.axvline(x=1, color='gray', linestyle='--', alpha=0.5)
    ax3.set_xlabel('w', fontsize=10)
    ax3.set_ylabel('f(w)·A', fontsize=10)
    ax3.set_title('(c) IS 梯度 (A>0)', fontsize=11, fontweight='bold')
    ax3.legend(fontsize=8)
    ax3.set_xlim(0, 3)
    ax3.set_ylim(0, 3)

    # (2,2) IS Reshape 梯度 A<0
    ax4 = fig.add_subplot(gs[1, 1])
    for gamma, color_key in zip([0.3, 0.5, 0.7, 1.0],
                                 ['gamma_0.3', 'gamma_0.5', 'gamma_0.7', 'gamma_1']):
        grad = f_is_reshape(w, gamma) * A_neg
        ax4.plot(w, grad, color=COLORS[color_key], linewidth=2, label=f'γ={gamma}')
    ax4.axvline(x=1, color='gray', linestyle='--', alpha=0.5)
    ax4.set_xlabel('w', fontsize=10)
    ax4.set_ylabel('f(w)·A', fontsize=10)
    ax4.set_title('(d) IS 梯度 (A<0)', fontsize=11, fontweight='bold')
    ax4.legend(fontsize=8, loc='lower left')
    ax4.set_xlim(0, 3)
    ax4.set_ylim(-3, 0)

    # (2,3) PPO 梯度 A>0
    ax5 = fig.add_subplot(gs[1, 2])
    grad_ppo_pos = f_ppo_clip(w, A_sign='pos') * A_pos
    grad_no_clip = w * A_pos
    ax5.plot(w, grad_ppo_pos, color=COLORS['ppo_pos'], linewidth=2.5, label='PPO Clip')
    ax5.plot(w, grad_no_clip, color='gray', linestyle=':', linewidth=1.5, alpha=0.7, label='无截断')
    ax5.fill_between(w[w > 1.27], grad_ppo_pos[w > 1.27], grad_no_clip[w > 1.27],
                     color=COLORS['ppo_pos'], alpha=0.2)
    ax5.axvline(x=1.27, color=COLORS['ppo_pos'], linestyle='--', alpha=0.5)
    ax5.set_xlabel('w', fontsize=10)
    ax5.set_ylabel('f(w)·A', fontsize=10)
    ax5.set_title('(e) PPO 梯度 (A>0)', fontsize=11, fontweight='bold')
    ax5.legend(fontsize=8)
    ax5.set_xlim(0, 3)
    ax5.set_ylim(0, 3)

    # (2,4) PPO 梯度 A<0
    ax6 = fig.add_subplot(gs[1, 3])
    grad_ppo_neg = f_ppo_clip(w, A_sign='neg') * A_neg
    grad_no_clip_neg = w * A_neg
    ax6.plot(w, grad_ppo_neg, color=COLORS['ppo_neg'], linewidth=2.5, label='PPO Clip')
    ax6.plot(w, grad_no_clip_neg, color='gray', linestyle=':', linewidth=1.5, alpha=0.7, label='无截断')
    ax6.fill_between(w[w < 0.8], grad_ppo_neg[w < 0.8], grad_no_clip_neg[w < 0.8],
                     color=COLORS['ppo_neg'], alpha=0.2)
    ax6.axvline(x=0.8, color=COLORS['ppo_neg'], linestyle='--', alpha=0.5)
    ax6.set_xlabel('w', fontsize=10)
    ax6.set_ylabel('f(w)·A', fontsize=10)
    ax6.set_title('(f) PPO 梯度 (A<0)', fontsize=11, fontweight='bold')
    ax6.legend(fontsize=8, loc='lower left')
    ax6.set_xlim(0, 3)
    ax6.set_ylim(-3, 0)

    # ============ 第三行: 对比和总结 ============
    # (3,1-2) 直接对比
    ax7 = fig.add_subplot(gs[2, 0:2])

    # IS Reshape
    ax7.plot(w, f_is_reshape(w, 0.3), color=COLORS['gamma_0.3'], linewidth=2.5,
             label='IS γ=0.3')
    ax7.plot(w, f_is_reshape(w, 0.5), color=COLORS['gamma_0.5'], linewidth=2.5,
             label='IS γ=0.5')

    # PPO (对称版本用于对比)
    ax7.plot(w, f_ppo_clip(w, A_sign=None), color=COLORS['ppo_sym'], linewidth=2.5,
             linestyle='--', label='PPO Clip (对称)')

    ax7.axvline(x=1, color='gray', linestyle='--', alpha=0.5)
    ax7.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    ax7.set_xlabel('w', fontsize=11)
    ax7.set_ylabel('f(w)', fontsize=11)
    ax7.set_title('(g) IS Reshape vs PPO Clip 直接对比', fontsize=12, fontweight='bold')
    ax7.legend(loc='upper left', fontsize=10)
    ax7.set_xlim(0, 3)
    ax7.set_ylim(0, 2)

    # 添加关键差异说明
    ax7.annotate('IS: 平滑压缩\n保留相对大小', xy=(2.2, f_is_reshape(2.2, 0.5)),
                 xytext=(2.5, 1.6), fontsize=9,
                 arrowprops=dict(arrowstyle='->', color=COLORS['gamma_0.5']))
    ax7.annotate('PPO: 硬截断\n超过阈值归零', xy=(2.2, 1.27),
                 xytext=(2.5, 0.9), fontsize=9,
                 arrowprops=dict(arrowstyle='->', color=COLORS['ppo_sym']))

    # (3,3-4) 特性对比表
    ax8 = fig.add_subplot(gs[2, 2:4])
    ax8.axis('off')

    # 创建表格数据
    table_data = [
        ['特性', 'IS Reshape (w^γ)', 'PPO Clip'],
        ['权重函数', 'f(w) = w^γ', 'clip(w, 1-ε, 1+ε)'],
        ['变换类型', '平滑幂变换', '硬截断'],
        ['对称性', '关于 w 对称', 'A-不对称'],
        ['高 w 处理', '平滑压缩', '截断到 1+ε'],
        ['低 w 处理', '平滑压缩', '截断到 1-ε (A<0)'],
        ['理论基础', '优化广义散度 D_α', '信任域约束'],
        ['参数调节', 'γ 控制 SFT-RL 插值', 'ε 控制信任域大小'],
        ['自适应', 'γ 可基于 ESS 调整', 'ε 通常固定'],
    ]

    # 绘制表格
    table = ax8.table(cellText=table_data, loc='center', cellLoc='center',
                      colWidths=[0.25, 0.35, 0.35])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)

    # 设置表头样式
    for i in range(3):
        table[(0, i)].set_facecolor('#34495e')
        table[(0, i)].set_text_props(color='white', fontweight='bold')

    # 交替行颜色
    for i in range(1, len(table_data)):
        for j in range(3):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#ecf0f1')
            else:
                table[(i, j)].set_facecolor('white')

    ax8.set_title('(h) 特性对比总结', fontsize=12, fontweight='bold', pad=20)

    plt.savefig('/Users/floyed/Downloads/offline_rl/fig4_comprehensive_comparison.png',
                dpi=150, bbox_inches='tight')
    plt.savefig('/Users/floyed/Downloads/offline_rl/fig4_comprehensive_comparison.pdf',
                bbox_inches='tight')
    print("图4已保存: fig4_comprehensive_comparison.png/pdf")
    plt.close()


# ============================================================
# 主函数
# ============================================================
if __name__ == '__main__':
    print("=" * 60)
    print("IS Reshape vs PPO Clip 可视化")
    print("=" * 60)

    plot_weight_functions()
    plot_gradient_contributions()
    plot_key_differences()
    plot_comprehensive_comparison()

    print("\n所有图表已生成完成！")
    print("保存位置: /Users/floyed/Downloads/offline_rl/")
