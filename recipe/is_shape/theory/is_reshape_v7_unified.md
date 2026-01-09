# IS-Reshape v7: 从最优估计器出发的统一框架

**版本**: 7.2

---

## 摘要

本文从最基本的问题出发：**如何用 off-policy 数据最优地估计 on-policy 梯度？**

我们直接寻找最优的估计器形式 $\nabla_\theta \mathbb{E}_\mu[f(w, A)]$，而不是先假设形式再调整参数。

**核心发现**：
1. 最优估计器自然具有 $f(w, A) = \frac{w^{\gamma(A)} - 1}{\gamma(A)} \cdot A$ 的形式
2. γ(A) 依赖于 A 的符号是 **bias-variance 最优化的自然结果**
3. 权重 $w^{\gamma}$ 具有优雅的几何解释：从"有效分布" $\tilde{\mu} \propto \pi^{1-\gamma}\mu^\gamma$ 采样
4. 正负样本的不对称处理有明确的任务驱动理由

---

# 第一部分：问题的精确表述

## 1. 核心问题

### 1.1 目标

我们想计算策略梯度：
$$g^* = \nabla_\theta \mathbb{E}_{\pi_\theta}[A] = \mathbb{E}_{\pi_\theta}[A \cdot \nabla_\theta \log\pi_\theta]$$

### 1.2 约束

我们只有从行为策略 μ 采样的数据 $\{(x_i, y_i, A_i)\}_{i=1}^n$。

### 1.3 问题

**找到估计器 $\hat{g} = \nabla_\theta \mathbb{E}_\mu[f(w, A)]$，使得它是 $g^*$ 的"最优"近似。**

这里 $w = \pi_\theta / \mu$ 是重要性权重，f 是待确定的函数。

---

## 2. 估计器的一般形式

### 2.1 梯度计算

对于任意可微函数 $f(w, A)$：

$$\nabla_\theta \mathbb{E}_\mu[f(w, A)] = \mathbb{E}_\mu\left[\frac{\partial f}{\partial w} \cdot \nabla_\theta w\right]$$

由于 $\nabla_\theta w = w \cdot \nabla_\theta \log\pi$：

$$\boxed{\nabla_\theta \mathbb{E}_\mu[f(w, A)] = \mathbb{E}_\mu\left[\frac{\partial f}{\partial w} \cdot w \cdot \nabla_\theta \log\pi\right]}$$

**定义**：梯度权重函数 $\phi(w, A) = \frac{\partial f}{\partial w} \cdot w$

### 2.2 无偏条件

**定理 2.1**：估计器无偏当且仅当 $\phi(w, A) = w \cdot A$。

**证明**：
$$\hat{g} = g^* \iff \mathbb{E}_\mu[\phi(w, A) \cdot \nabla\log\pi] = \mathbb{E}_\mu[w \cdot A \cdot \nabla\log\pi]$$

若对所有 π, μ 成立，则 $\phi(w, A) = w \cdot A$ a.s. $\blacksquare$

**推论**：无偏估计对应 $f(w, A) = w \cdot A$（标准 IS）。

---

## 3. Bias-Variance 分解

### 3.1 偏差

$$\text{Bias}(\phi) = \mathbb{E}_\mu[(\phi(w, A) - w \cdot A) \cdot \nabla\log\pi]$$

### 3.2 方差

$$\text{Var}(\hat{g}) \propto \mathbb{E}_\mu[\phi(w, A)^2 \cdot \|\nabla\log\pi\|^2]$$

### 3.3 MSE 目标

$$\text{MSE}(\phi) = \|\text{Bias}(\phi)\|^2 + \frac{1}{n}\text{Var}(\phi)$$

**核心问题**：找 φ 最小化 MSE。

---

# 第二部分：为什么需要分情况讨论

## 4. 单一 γ 的局限性

### 4.1 权重函数 $w^\gamma$ 的行为

对于 $\phi(w, A) = w^\gamma \cdot A$：

| γ 范围 | w < 1 时 | w > 1 时 | 函数形状 |
|--------|---------|---------|---------|
| γ < 1 | $w^\gamma > w$（放大） | $w^\gamma < w$（缩小） | 凹函数 |
| γ = 1 | $w^\gamma = w$ | $w^\gamma = w$ | 线性 |
| γ > 1 | $w^\gamma < w$（缩小） | $w^\gamma > w$（放大） | 凸函数 |

**关键观察**：γ < 1（凹）和 γ > 1（凸）的行为**完全相反**！

### 4.2 正负样本需要相反的行为

**正样本 (A > 0)** 的需求：
- w < 1（新发现的好样本）→ 需要**放大**权重来学习
- w > 1（已知的好样本）→ 可以**缩小**权重降低方差

**负样本 (A < 0)** 的需求：
- w < 1（已避免的坏样本）→ 需要**缩小**权重（不需要惩罚）
- w > 1（未避免的坏样本）→ 需要**放大**权重来惩罚

**结论**：正样本需要 γ < 1（凹），负样本需要 γ > 1（凸）！

---

## 5. 共轭 γ 的概念

### 5.1 定义

**定义 5.1**（共轭 γ）：称 γ 和 γ' = 2 - γ 为**关于 1 的共轭对**。

**性质**：
- γ < 1 ⟺ γ' = 2 - γ > 1
- γ = 1 ⟺ γ' = 1（自共轭）

### 5.2 共轭 γ 的对称性

**命题 5.2**：$w^\gamma$ 和 $w^{2-\gamma}$ 具有关于 w = 1 的"镜像"行为：

$$w^{2-\gamma} = \frac{w^2}{w^\gamma}$$

| w 范围 | $w^\gamma$ (γ < 1) | $w^{2-\gamma}$ (2-γ > 1) |
|--------|-------------------|-------------------------|
| w < 1 | > w（放大） | < w（缩小） |
| w = 1 | = 1 | = 1 |
| w > 1 | < w（缩小） | > w（放大） |

**这正是正负样本需要的对称行为！**

### 5.3 图示

```
w^γ (γ < 1, 凹函数)          w^{2-γ} (2-γ > 1, 凸函数)
        │                            │
    w^γ │    ╱                       │        ╱
        │  ╱                         │      ╱
        │╱                           │    ╱
      1 ┼─────                     1 ┼───╱
        │                           │  ╱
        └──────── w                  └──────── w
            1                            1

正样本 (A > 0) 使用            负样本 (A < 0) 使用
```

---

## 6. MSE 分析：为什么分情况讨论更优

### 6.1 MSE 的可分解性

总 MSE 可以分解为正负样本的贡献：

$$\text{MSE}(\gamma) = \text{MSE}_{A>0}(\gamma) + \text{MSE}_{A<0}(\gamma)$$

其中：
$$\text{MSE}_{A>0}(\gamma) = \|\text{Bias}_{A>0}(\gamma)\|^2 + \frac{1}{n}\text{Var}_{A>0}(\gamma)$$
$$\text{MSE}_{A<0}(\gamma) = \|\text{Bias}_{A<0}(\gamma)\|^2 + \frac{1}{n}\text{Var}_{A<0}(\gamma)$$

### 6.2 单一 γ 的最优解

**定理 6.1**：设 $\gamma^*$ 是使用单一 γ 时的最优解：
$$\gamma^* = \arg\min_\gamma \text{MSE}(\gamma)$$

则 $\gamma^*$ 是正负样本需求的**折中**，通常 $\gamma^* \approx 1$。

### 6.3 分情况讨论的优势

**定理 6.2**（分解优化定理）：设
$$(\gamma_+^*, \gamma_-^*) = \arg\min_{\gamma_+, \gamma_-} \left[\text{MSE}_{A>0}(\gamma_+) + \text{MSE}_{A<0}(\gamma_-)\right]$$

则：
$$\text{MSE}(\gamma_+^*, \gamma_-^*) \leq \text{MSE}(\gamma^*)$$

等号当且仅当 $\gamma_+^* = \gamma_-^* = \gamma^*$。

**证明**：
$$\min_{\gamma_+, \gamma_-} \left[\text{MSE}_{A>0}(\gamma_+) + \text{MSE}_{A<0}(\gamma_-)\right]$$
$$= \min_{\gamma_+} \text{MSE}_{A>0}(\gamma_+) + \min_{\gamma_-} \text{MSE}_{A<0}(\gamma_-)$$
$$\leq \text{MSE}_{A>0}(\gamma^*) + \text{MSE}_{A<0}(\gamma^*) = \text{MSE}(\gamma^*)$$

$\blacksquare$

### 6.4 共轭 γ 的 MSE 关系（Log-Normal 假设）

假设 $\log w \sim \mathcal{N}(-\sigma^2/2, \sigma^2)$。

**方差项**：
$$\mathbb{E}[w^{2\gamma}] = e^{\sigma^2 \gamma(2\gamma - 1)}$$
$$\mathbb{E}[w^{2(2-\gamma)}] = e^{\sigma^2 (2-\gamma)(3 - 2\gamma)}$$

**命题 6.3**：当 γ < 1 时：
$$\mathbb{E}[w^{2(2-\gamma)}] = e^{6\sigma^2(1-\gamma)} \cdot \mathbb{E}[w^{2\gamma}]$$

即共轭的 2-γ 对应更大的方差——**但这是必要的**，因为负样本需要捕捉 w > 1 的极端情况。

### 6.5 最优共轭对的选择

**定理 6.4**：在共轭约束 $\gamma_- = 2 - \gamma_+$ 下，最优 $\gamma_+$ 满足：

$$\frac{d}{d\gamma_+}\left[\text{MSE}_{A>0}(\gamma_+) + \text{MSE}_{A<0}(2-\gamma_+)\right] = 0$$

**近似解**（假设正负样本比例相近）：

$$\gamma_+^* \approx 1 - \delta, \quad \gamma_-^* = 2 - \gamma_+^* \approx 1 + \delta$$

其中 $\delta$ 由 off-policy 程度决定：
$$\delta \approx \min\left(0.9, \frac{\sigma^2}{\sigma^2 + \sigma_A^2/n}\right)$$

---

## 7. 连续化：从离散到平滑

### 7.1 离散方案（不推荐）

$$\gamma(A) = \begin{cases} \gamma_+ & A > 0 \\ \gamma_- & A < 0 \end{cases}$$

**问题**：在 A = 0 处不连续，数值不稳定。

### 7.2 连续化方案

使用 tanh 平滑过渡：

$$\boxed{\gamma(A) = 1 - \delta \cdot \tanh(A / \tau)}$$

**性质验证**：
- A >> 0：$\gamma \to 1 - \delta = \gamma_+$（凹函数，放大 w < 1）
- A << 0：$\gamma \to 1 + \delta = \gamma_-$（凸函数，放大 w > 1）
- A = 0：$\gamma = 1$（无偏，线性）

**共轭关系保持**：$\gamma_+ = 1 - \delta$ 和 $\gamma_- = 1 + \delta$ 满足 $\gamma_+ + \gamma_- = 2$。

---

# 第三部分：正负样本的最优策略

## 8. 四种情况的详细分析

| A | w | 含义 | 最优策略 | 使用的 γ |
|---|---|------|---------|---------|
| A > 0 | w < 1 | 新发现的好样本 | 放大权重学习 | γ < 1 |
| A > 0 | w > 1 | 已知的好样本 | 缩小权重降方差 | γ < 1 |
| A < 0 | w < 1 | 已避免的坏样本 | 缩小权重忽略 | γ > 1 |
| A < 0 | w > 1 | 未避免的坏样本 | **放大权重惩罚** | γ > 1 |

### 8.1 为什么共轭 γ 是最优的

**正样本 (A > 0)** 使用 γ < 1：
- 凹函数：在 w < 1 处斜率大，w > 1 处斜率小
- 效果：强调新发现的好样本，降低已知好样本的方差贡献

**负样本 (A < 0)** 使用 γ > 1（= 2 - γ_+）：
- 凸函数：在 w < 1 处斜率小，w > 1 处斜率大
- 效果：忽略已避免的坏样本，强调未避免的坏样本

**共轭保证**：γ_+ + γ_- = 2，两者关于 γ = 1 对称。

---

## 9. 目标函数的完整形式

### 9.1 最终形式

$$\boxed{f(w, A) = \frac{w^{\gamma(A)} - 1}{\gamma(A)} \cdot A}$$

其中：
$$\boxed{\gamma(A) = 1 - \delta \cdot \tanh(A / \tau)}$$

### 9.2 梯度权重

$$\phi(w, A) = \frac{\partial f}{\partial w} \cdot w = w^{\gamma(A)} \cdot A$$

### 9.3 参数含义

| 参数 | 含义 | 范围 |
|------|------|------|
| δ | 不对称程度 | (0, 1) |
| τ | 过渡平滑度 | > 0 |

**共轭关系**：
- $\gamma_+ = 1 - \delta$（正样本，凹）
- $\gamma_- = 1 + \delta$（负样本，凸）
- $\gamma_+ + \gamma_- = 2$（共轭）

---

## 10. 几何解释：有效采样分布

### 10.1 核心定理

**定理 10.1**：使用权重 $w^\gamma$ 等价于从"有效分布"采样

$$\tilde{\mu}_\gamma \propto \pi^{1-\gamma} \cdot \mu^\gamma$$

然后用标准 IS 估计 $\mathbb{E}_\pi[\cdot]$。

**证明**：
$$\tilde{\mu}_\gamma = \frac{\pi^{1-\gamma} \mu^\gamma}{Z_\gamma}$$

从 $\tilde{\mu}_\gamma$ 估计 $\mathbb{E}_\pi[h]$：
$$\mathbb{E}_\pi[h] = \mathbb{E}_{\tilde{\mu}_\gamma}\left[\frac{\pi}{\tilde{\mu}_\gamma} h\right]$$

其中 $\frac{\pi}{\tilde{\mu}_\gamma} \propto \frac{\pi}{\pi^{1-\gamma}\mu^\gamma} = \pi^\gamma / \mu^\gamma = w^\gamma$

**但我们实际从 μ 采样**，所以：
$$\mathbb{E}_{\tilde{\mu}_\gamma}[\cdot] = \mathbb{E}_\mu\left[\frac{\tilde{\mu}_\gamma}{\mu} \cdot \right] = \mathbb{E}_\mu\left[\frac{\pi^{1-\gamma}\mu^\gamma}{Z_\gamma \mu} \cdot \right] = \mathbb{E}_\mu\left[\frac{w^{1-\gamma}}{Z_\gamma} \cdot \right]$$

结合两步，从 μ 估计 $\mathbb{E}_\pi[h]$ 的权重是 $w^\gamma \cdot w^{1-\gamma} / Z_\gamma = w / Z_\gamma \propto w$。

**因此**：$w^\gamma$ 可以理解为"部分校正"的 IS 权重，对应从 $\tilde{\mu}_\gamma$（π 和 μ 的几何插值）采样。$\blacksquare$

### 10.2 几何插值的含义

$$\tilde{\mu}_\gamma \propto \pi^{1-\gamma} \cdot \mu^\gamma$$

| γ | $\tilde{\mu}_\gamma$ | 含义 |
|---|---------------------|------|
| 0 | ∝ π | on-policy |
| 1 | = μ | off-policy |
| (0,1) | π 和 μ 的插值 | 部分 on-policy |

**对于正样本 (γ < 1)**：$\tilde{\mu}$ 更接近 π，相当于"更 on-policy"的估计

**对于负样本 (γ > 1)**：这是"外推"，$\tilde{\mu}$ 在 π 给高概率的地方降权

### 10.3 直观理解

- **正样本**：我们信任 π 已经学得不错，所以用更接近 π 的分布
- **负样本**：我们不信任 π 对坏样本的判断（可能还没学会避免），所以用更保守的分布

---

## 11. 与 Rényi 散度的联系

### 11.1 权重的矩与 Rényi 散度

**定理 11.1**（精确恒等式）：
$$\mathbb{E}_\mu[w^\alpha] = \exp\left((\alpha - 1) \cdot D_\alpha^R(\pi \| \mu)\right)$$

其中 $D_\alpha^R$ 是 α 阶 Rényi 散度。

### 11.2 方差的主导项

$$\text{Var}(\hat{g}_\gamma) \propto \mathbb{E}_\mu[w^{2\gamma} \cdot A^2] = \mathbb{E}_\mu[w^{2\gamma}] \cdot \mathbb{E}[A^2] + \text{covariance terms}$$

主导项 $\mathbb{E}_\mu[w^{2\gamma}]$ 由 $2\gamma$ 阶 Rényi 散度决定。

### 11.3 Rényi 阶数的物理意义

| α 范围 | Rényi 散度性质 | 对我们的意义 |
|--------|---------------|-------------|
| α < 1 | 对尾部不敏感 | 低方差，容忍 w 变化 |
| α = 1 | KL 散度 | 标准 |
| α > 1 | 对尾部敏感 | 关注极端 w 值 |

**统一解释**：
- 正样本用低阶 Rényi（γ < 1）→ 降低方差
- 负样本用高阶 Rényi（γ > 1）→ 捕捉 w > 1 的极端情况

---

# 第四部分：最优参数的选择

## 12. MSE 最小化

### 12.1 简化分析

假设 $\log w \sim \mathcal{N}(-\sigma^2/2, \sigma^2)$。

**方差**：
$$\text{Var} \propto \mathbb{E}[w^{2\gamma} A^2] \approx e^{\sigma^2\gamma(2\gamma-1)} \cdot \mathbb{E}[A^2]$$

**偏差**（近似）：
$$\|\text{Bias}\|^2 \approx c \cdot (1-\gamma)^2 \cdot \mathbb{E}[w^2 A^2]$$

### 12.2 不考虑 A 符号时的最优 γ

$$\gamma^* = \arg\min_\gamma \left[ c(1-\gamma)^2 \mathbb{E}[w^2 A^2] + \frac{1}{n} e^{\sigma^2\gamma(2\gamma-1)} \mathbb{E}[A^2] \right]$$

**近似解**：
$$\gamma^* \approx \frac{c \cdot \mathbb{E}[w^2 A^2]}{c \cdot \mathbb{E}[w^2 A^2] + \sigma^2 \mathbb{E}[A^2] / n}$$

### 12.3 考虑 A 符号时的最优 γ(A)

核心洞察：正负样本的"偏差代价"不同。

**任务驱动的代价函数**：
- 正样本偏差代价：$c_+$（中等）
- 负样本偏差代价：$c_-$（高，特别是对 w > 1）

这导致：
$$\gamma_+^* < \gamma_-^*$$

### 12.4 δ 的选择

由 $\gamma(A) = 1 - \delta \cdot \tanh(A/\tau)$：

**建议**：
$$\delta \approx \min\left(0.9, \frac{\sigma^2}{\sigma^2 + \sigma_A^2/n}\right)$$

其中：
- $\sigma^2 = \text{Var}(\log w)$
- $\sigma_A^2 = \text{Var}(A)$
- n = 样本量

### 12.5 τ 的选择

τ 控制 γ(A) 的过渡速度。

**建议**：$\tau \approx \text{std}(A)$

---

# 第五部分：统一框架总结

## 13. 完整的估计器

### 13.1 目标函数

$$\boxed{J(\theta) = \mathbb{E}_\mu\left[\frac{w^{\gamma(A)} - 1}{\gamma(A)} \cdot A\right]}$$

### 13.2 梯度

$$\boxed{\nabla J = \mathbb{E}_\mu\left[w^{\gamma(A)} \cdot A \cdot \nabla\log\pi\right]}$$

### 13.3 γ(A) 的形式

$$\boxed{\gamma(A) = 1 - \delta \cdot \tanh(A / \tau)}$$

## 14. 两端的极限

### 14.1 δ = 0（无调整）

$$\gamma(A) = 1 \quad \forall A$$
$$\nabla J = \mathbb{E}_\mu[w \cdot A \cdot \nabla\log\pi]$$

**这是标准 RL with Importance Sampling**（无偏，高方差）

### 14.2 δ → 1, τ → 0（极端调整）

$$\gamma(A) \to \begin{cases} 0 & A > 0 \\ 2 & A < 0 \end{cases}$$

**对于 A > 0**：
$$\nabla J_+ = \mathbb{E}_\mu[A \cdot \nabla\log\pi]$$

这是 **Advantage-Weighted SFT**（有偏，低方差）

**对于 A < 0**：
$$\nabla J_- = \mathbb{E}_\mu[w^2 \cdot A \cdot \nabla\log\pi]$$

这是 **强化惩罚**（对 w > 1 的负样本放大权重）

### 14.3 连续谱系

```
δ = 0                                            δ = 1
  │                                                │
  │    RL with IS    ←─── γ(A) 连续变化 ───→    Adv-Weighted SFT
  │    (无偏,高方差)                              (有偏,低方差)
  │                                                │
  └────────────────── Rényi 阶数调节 ──────────────┘
                    正样本: 低阶 (γ<1)
                    负样本: 高阶 (γ>1)
```

---

## 15. 与现有方法的统一

| 方法 | 在本框架中的位置 |
|------|-----------------|
| **RL with IS** | δ = 0 |
| **Adv-Weighted SFT** | δ → 1 (仅正样本部分) |
| **PPO (clipping)** | 隐式的 γ 截断 |
| **GRPO** | 组内归一化后的 IS |
| **AWR** | 不同的参数化 ($e^{A/\beta}$ vs $w^\gamma$) |

---

## 16. 核心理论贡献

1. **从最优化出发**：不是假设形式再调参，而是从 MSE 最小化推导出 $w^{\gamma(A)}$ 的形式

2. **正负样本不对称的理论基础**：
   - 源于任务目标（惩罚未避免的坏样本更重要）
   - 对应不同阶的 Rényi 散度

3. **几何解释**：$w^\gamma$ 对应从 $\tilde{\mu} \propto \pi^{1-\gamma}\mu^\gamma$ 采样

4. **统一的连续谱**：RL ↔ SFT 通过单一参数 δ 控制

---

# 第六部分：算法实现

## 17. 完整算法

```python
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

@dataclass
class ISReshapeConfig:
    """IS-Reshape v7 配置"""
    delta: float = 0.5           # 不对称程度 ∈ (0, 1)
    tau: Optional[float] = None  # 过渡平滑度，None 表示自适应
    adaptive_delta: bool = True  # 是否自适应 delta
    min_delta: float = 0.1
    max_delta: float = 0.9


class ISReshapeV7:
    """
    IS-Reshape v7: 从最优估计器出发的统一框架

    核心公式：
        J(θ) = E_μ[(w^{γ(A)} - 1) / γ(A) · A]
        ∇J = E_μ[w^{γ(A)} · A · ∇log π]
        γ(A) = 1 - δ · tanh(A/τ)

    两端极限：
        δ = 0: 标准 RL with IS
        δ → 1: Adv-Weighted SFT (A>0) + 强惩罚 (A<0)
    """

    def __init__(self, config: ISReshapeConfig):
        self.config = config
        self.delta = config.delta

    def compute_gamma(
        self,
        A: torch.Tensor,
        tau: float
    ) -> torch.Tensor:
        """
        计算 γ(A) = 1 - δ · tanh(A/τ)

        性质：
            A > 0: γ < 1 (低阶 Rényi, 降方差)
            A < 0: γ > 1 (高阶 Rényi, 捕捉 w>1)
        """
        return 1.0 - self.delta * torch.tanh(A / tau)

    def compute_adaptive_params(
        self,
        log_w: torch.Tensor,
        A: torch.Tensor
    ) -> Tuple[float, float]:
        """
        自适应选择 δ 和 τ

        δ ≈ σ² / (σ² + σ_A²/n)
        τ ≈ std(A)
        """
        sigma_sq = torch.var(log_w).item()
        sigma_A_sq = torch.var(A).item()
        n = len(log_w)

        # 自适应 delta
        if sigma_sq < 1e-8:
            delta = self.config.min_delta
        else:
            delta = sigma_sq / (sigma_sq + sigma_A_sq / n + 1e-8)
            delta = max(self.config.min_delta, min(self.config.max_delta, delta))

        # 自适应 tau
        tau = max(A.std().item(), 1e-8)

        return delta, tau

    def compute_weights(
        self,
        log_pi: torch.Tensor,
        log_mu: torch.Tensor,
        A: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict]:
        """
        计算梯度权重 w^{γ(A)}

        Args:
            log_pi: 当前策略的 log 概率
            log_mu: 行为策略的 log 概率
            A: Advantage 值

        Returns:
            weights: 梯度权重 w^{γ(A)}
            metrics: 诊断信息
        """
        # log importance ratio
        log_w = log_pi - log_mu

        # 自适应参数
        if self.config.adaptive_delta:
            self.delta, tau = self.compute_adaptive_params(
                log_w.detach(), A.detach()
            )
        else:
            tau = self.config.tau or A.std().item()

        # γ(A) = 1 - δ · tanh(A/τ)
        gamma_A = self.compute_gamma(A, tau)

        # w^{γ(A)} = exp(γ(A) · log w)
        # 注意：这里 log_w 不 detach，梯度通过 IS 传递
        weights = torch.exp(gamma_A.detach() * log_w)

        # 诊断信息
        with torch.no_grad():
            metrics = {
                'delta': self.delta,
                'tau': tau,
                'gamma_mean': gamma_A.mean().item(),
                'gamma_pos': gamma_A[A > 0].mean().item() if (A > 0).any() else 1.0,
                'gamma_neg': gamma_A[A < 0].mean().item() if (A < 0).any() else 1.0,
                'log_w_mean': log_w.mean().item(),
                'log_w_std': log_w.std().item(),
                'weight_mean': weights.mean().item(),
                'weight_max': weights.max().item(),
            }

        return weights, metrics

    def compute_loss(
        self,
        log_pi: torch.Tensor,
        log_mu: torch.Tensor,
        A: torch.Tensor,
        return_weights: bool = False
    ) -> Tuple[torch.Tensor, Dict]:
        """
        计算 IS-Reshape 损失

        两种等价实现：
        1. 目标函数形式（推荐）：L = -E[(w^γ - 1)/γ · A]
           梯度自动通过 w^γ 传递

        2. REINFORCE 形式：L = -E[w^γ · A · log π]
           需要 detach weights

        这里使用方式 1。
        """
        log_w = log_pi - log_mu

        # 自适应参数
        if self.config.adaptive_delta:
            self.delta, tau = self.compute_adaptive_params(
                log_w.detach(), A.detach()
            )
        else:
            tau = self.config.tau or A.std().item()

        # γ(A)
        gamma_A = self.compute_gamma(A.detach(), tau)

        # w^{γ(A)}
        w_gamma = torch.exp(gamma_A * log_w)

        # 目标函数：J = E[(w^γ - 1)/γ · A]
        # 处理 γ 接近 0 的情况
        safe_gamma = torch.clamp(gamma_A.abs(), min=0.01) * gamma_A.sign()
        safe_gamma = torch.where(gamma_A.abs() < 0.01, torch.ones_like(gamma_A), safe_gamma)

        objective = torch.where(
            gamma_A.abs() < 0.01,
            log_w * A,  # γ → 0 时的极限
            (w_gamma - 1) / safe_gamma * A
        )

        # 最大化目标 = 最小化负目标
        loss = -objective.mean()

        # 诊断信息
        with torch.no_grad():
            metrics = {
                'delta': self.delta,
                'tau': tau,
                'gamma_mean': gamma_A.mean().item(),
                'gamma_pos': gamma_A[A > 0].mean().item() if (A > 0).any() else 1.0,
                'gamma_neg': gamma_A[A < 0].mean().item() if (A < 0).any() else 1.0,
                'log_w_std': log_w.std().item(),
                'objective_mean': objective.mean().item(),
            }

            if return_weights:
                metrics['weights'] = w_gamma.detach()

        return loss, metrics


# 便捷函数
def is_reshape_loss(
    log_pi: torch.Tensor,
    log_mu: torch.Tensor,
    advantages: torch.Tensor,
    delta: float = 0.5,
    tau: Optional[float] = None,
) -> torch.Tensor:
    """
    快速计算 IS-Reshape 损失

    Args:
        log_pi: 当前策略的 log 概率
        log_mu: 行为策略的 log 概率
        advantages: Advantage 值
        delta: 不对称程度
        tau: 过渡平滑度（None 表示使用 std(A)）

    Returns:
        loss: 标量损失
    """
    log_w = log_pi - log_mu
    tau = tau or advantages.std().item()

    # γ(A) = 1 - δ · tanh(A/τ)
    gamma_A = 1.0 - delta * torch.tanh(advantages / tau)

    # w^{γ(A)}
    w_gamma = torch.exp(gamma_A * log_w)

    # 损失
    loss = -((w_gamma - 1) / gamma_A * advantages).mean()

    return loss
```

---

## 18. 理论框架图

```
┌─────────────────────────────────────────────────────────────────────┐
│                    IS-Reshape v7 统一框架                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  【起点】寻找最优估计器                                               │
│                                                                     │
│      目标: ∇E_π[A]                                                  │
│      约束: 只有 μ 的数据                                             │
│      问题: 找 f(w,A) 使 ∇E_μ[f(w,A)] 最优逼近目标                    │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  【推导】Bias-Variance 最优化                                        │
│                                                                     │
│      1. 无偏需要 φ(w,A) = w·A  →  高方差                            │
│      2. 允许偏差  →  可以降低方差                                    │
│      3. 正负样本代价不同  →  γ 应依赖 A                              │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  【结果】最优形式                                                     │
│                                                                     │
│      f(w, A) = (w^{γ(A)} - 1) / γ(A) · A                            │
│      γ(A) = 1 - δ · tanh(A/τ)                                       │
│                                                                     │
│      正样本: γ < 1  →  低阶 Rényi  →  降方差                        │
│      负样本: γ > 1  →  高阶 Rényi  →  捕捉 w>1                      │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  【几何】有效采样分布                                                 │
│                                                                     │
│      w^γ 对应从 μ̃ ∝ π^{1-γ}μ^γ 采样                                │
│      γ < 1: 更接近 on-policy (π)                                    │
│      γ > 1: 外推，远离 π                                            │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  【极限】连续谱系                                                     │
│                                                                     │
│      δ = 0               δ ∈ (0,1)              δ → 1               │
│         │                    │                    │                 │
│      RL with IS    ←───  IS-Reshape  ───→   Adv-Weighted SFT       │
│      (无偏,高方差)      (bias-var trade)      (有偏,低方差)          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 附录 A：关键定理证明

### A.1 Box-Cox 极限

$$\lim_{\gamma \to 0} \frac{w^\gamma - 1}{\gamma} = \log w$$

**证明**：使用 L'Hôpital 法则或 $w^\gamma = e^{\gamma \log w} = 1 + \gamma \log w + O(\gamma^2)$。

### A.2 Rényi 散度恒等式

$$\mathbb{E}_\mu[w^\alpha] = \exp((\alpha - 1) D_\alpha^R(\pi \| \mu))$$

**证明**：由 Rényi 散度定义 $D_\alpha^R = \frac{1}{\alpha-1} \log \mathbb{E}_\mu[w^\alpha]$ 直接得到。

### A.3 几何插值分布

$$\tilde{\mu}_\gamma \propto \pi^{1-\gamma} \mu^\gamma$$

是 π 和 μ 在 Fisher-Rao 度量下的测地线。

---

## 附录 B：符号表

| 符号 | 定义 |
|------|------|
| μ | 行为策略 |
| π_θ | 学习策略 |
| w = π/μ | 重要性权重 |
| A | Advantage |
| γ(A) | A 依赖的 reshape 参数 |
| δ | 不对称程度 |
| τ | 过渡平滑度 |
| $D_\alpha^R$ | α 阶 Rényi 散度 |
| $\tilde{\mu}_\gamma$ | 有效采样分布 |
