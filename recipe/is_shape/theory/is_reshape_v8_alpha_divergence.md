# IS-Reshape v8: 从梯度权重设计到 α-Divergence

**版本**: 8.0

---

## 摘要

本文从策略梯度估计器的设计出发，直接研究梯度权重函数 φ(w, A) 的性质，而非先假设目标函数形式。

**核心思路转变**：
- ~~从目标函数 f(w, A) 出发，推导梯度权重 φ~~
- **从理想的梯度权重 φ(w, A) 出发，分析其性质和理论联系**

**主要贡献**：
1. **梯度权重视角**：直接定义和分析 φ(w, A)，不预设目标函数形式
2. **可积性分析**：什么样的 φ 对应某个目标函数 f？不可积意味着什么？
3. **α-Divergence 联系**：证明 φ(w) = w^α 对应 α-divergence 的梯度结构
4. **设计原则**：从偏差-方差权衡、有界性、单调性等角度分析 φ 的设计
5. **SAPO 重新分析**：从 φ 的角度理解 SAPO 的设计
6. **SFT 特例**：A > 0 恒成立时的简化

---

# 第一部分：梯度权重视角

## 1. 核心对象：梯度估计器

### 1.1 问题设定

**目标**：估计 on-policy 策略梯度
$$g^* = \nabla_\theta \mathbb{E}_{\pi_\theta}[A] = \mathbb{E}_{\pi_\theta}[A \cdot \nabla_\theta \log \pi_\theta]$$

**约束**：只有 off-policy 数据，从行为策略 μ 采样

### 1.2 一般梯度估计器

定义一般形式的梯度估计器：

$$\boxed{\hat{g} = \mathbb{E}_\mu[\phi(w, A) \cdot \nabla_\theta \log \pi_\theta]}$$

其中：
- $w = \pi_\theta / \mu$ 是重要性权重
- $\phi: \mathbb{R}^+ \times \mathbb{R} \to \mathbb{R}$ 是**梯度权重函数**
- A 是优势值

**注意**：我们直接定义 φ，不预设它来自某个目标函数。

### 1.3 特殊情况

| φ(w, A) | 名称 | 性质 |
|---------|------|------|
| w · A | 标准 IS | 无偏，高方差 |
| A | Weighted SFT | 有偏（忽略分布偏移），低方差 |
| clip(w, 1-ε, 1+ε) · A | PPO | 有偏，截断方差 |
| w^γ · A | IS-Reshape | 参数化偏差-方差权衡 |

---

## 2. φ(w, A) 的基本性质

### 2.1 无偏条件

**定理 2.1**：$\hat{g}$ 是 $g^*$ 的无偏估计 当且仅当 $\phi(w, A) = w \cdot A$。

**证明**：
$$\hat{g} = \mathbb{E}_\mu[\phi(w, A) \cdot \nabla \log \pi] = g^* = \mathbb{E}_\mu[w \cdot A \cdot \nabla \log \pi]$$

对所有分布 μ, π 和优势 A 成立，必须 φ(w, A) = w · A。$\blacksquare$

### 2.2 偏差与方差

**偏差**：
$$\text{Bias}(\phi) = \mathbb{E}_\mu[(\phi(w, A) - w \cdot A) \cdot \nabla \log \pi]$$

**方差**：
$$\text{Var}(\phi) \propto \mathbb{E}_\mu[\phi(w, A)^2 \cdot \|\nabla \log \pi\|^2]$$

**权衡**：φ 偏离 w·A 会引入偏差，但可能降低方差。

### 2.3 可分离性

**定义**：φ(w, A) 是可分离的，如果存在 g, h 使得 φ(w, A) = g(w) · h(A)。

**常见假设**：φ(w, A) = g(w) · A（h(A) = A）

**问题**：这个假设合理吗？我们是否需要更一般的形式？

---

## 3. 可积性：φ 何时对应目标函数？

### 3.1 问题

给定 φ(w, A)，是否存在目标函数 f(w, A) 使得：
$$\phi(w, A) = \frac{\partial f(w, A)}{\partial w} \cdot w$$

### 3.2 可积性条件

**定理 3.1**：φ(w, A) 对应某个 f(w, A) 当且仅当 φ(w, A)/w 关于 w 可积分。

若可积，则：
$$f(w, A) = \int \frac{\phi(w, A)}{w} dw + h(A)$$

其中 h(A) 是任意只依赖于 A 的函数。

### 3.3 例子

**例 1**：φ(w, A) = w · A（标准 IS）
$$f(w, A) = \int A \, dw = w \cdot A + h(A)$$
✓ 可积

**例 2**：φ(w, A) = w^γ · A（IS-Reshape）
$$f(w, A) = \int w^{\gamma-1} \cdot A \, dw = \frac{w^\gamma}{\gamma} \cdot A + h(A)$$
✓ 可积（γ ≠ 0）

**例 3**：φ(w, A) = clip(w, 1-ε, 1+ε) · A（PPO）
$$f(w, A) = \int \frac{\text{clip}(w, 1-ε, 1+ε)}{w} \cdot A \, dw$$
✓ 可积（分段函数）

### 3.4 不可积的情况意味着什么？

如果 φ(w, A)/w 不可积分，则：
- 不存在对应的目标函数 f
- 梯度 $\hat{g}$ 不是任何标量目标的梯度
- 这不一定是坏事！我们关心的是梯度估计器的质量，不一定需要目标函数

**哲学问题**：RL 中我们真正优化的是什么？
- 如果我们只关心梯度方向，φ 不需要可积
- 如果我们需要目标函数值（如用于收敛判断），则需要可积

---

# 第二部分：φ(w, A) 的设计原则

## 4. 从偏差-方差权衡出发

### 4.1 参数化形式：φ(w, A) = w^γ · A

这是一个单参数族，γ 控制偏差-方差权衡：
- γ = 1：无偏，最高方差
- γ < 1：有偏（向 SFT 方向），较低方差
- γ > 1：有偏（超 IS 方向），更高方差

### 4.2 为什么考虑 γ > 1？

表面上看，γ > 1 增加方差，为什么还要用？

**答案**：对于负样本 (A < 0)，梯度方向相反！

| 样本类型 | 需要的行为 | 对应 γ |
|---------|-----------|--------|
| A > 0, w < 1 (新好样本) | 放大学习 | γ < 1 |
| A > 0, w > 1 (已知好样本) | 减少重复 | γ < 1 |
| A < 0, w < 1 (已避免坏样本) | 减少惩罚 | γ > 1 |
| A < 0, w > 1 (未避免坏样本) | 加强惩罚 | γ > 1 |

### 4.3 分组设计：γ(A)

自然的设计：根据 A 的符号选择不同的 γ

$$\gamma(A) = \begin{cases}
\gamma_+ \in (0, 1) & A > 0 \\
\gamma_- \in (1, 2) & A < 0
\end{cases}$$

**对应的 φ**：
$$\phi(w, A) = w^{\gamma(A)} \cdot A$$

---

## 5. 有界性分析

### 5.1 为什么需要有界？

当 w 很大时（π_θ >> μ），无界的 φ 会导致：
- 梯度爆炸
- 训练不稳定
- 单个样本主导更新

### 5.2 不同 φ 的有界性

| φ(w, A) | w → ∞ 时 | 有界？ |
|---------|----------|--------|
| w · A | → ∞ | ❌ |
| w^γ · A (γ < 1) | → ∞（慢） | ❌ |
| w^γ · A (γ > 1) | → ∞（快） | ❌ |
| clip(w) · A | → (1+ε)·A | ✓ |
| sigmoid gate · A | → 有界 | ✓ |

### 5.3 有界性的数学约束

**定理 5.1**：不存在 φ(w) 同时满足：
1. 严格单调增
2. 严格凸
3. 有界
4. φ(1) = 1

**证明**：见 v7 或前文。

**推论**：对于 A < 0 需要的"凸"权重，必须在有界性上妥协。

### 5.4 实用解决方案

**方案 1：截断**
$$\phi(w, A) = \min(w^{\gamma(A)}, M) \cdot A$$

**方案 2：软饱和**
$$\phi(w, A) = \frac{w^{\gamma(A)}}{1 + \lambda(w^{\gamma(A)} - 1)^+} \cdot A$$

**方案 3：S形函数**（sapo_is_mono 的做法）
$$\phi_-(w) = \frac{(1+\tau)w}{\tau + w}$$（Michaelis-Menten）

---

## 6. 单调性分析

### 6.1 为什么需要单调？

**重要性采样的基本原则**：如果 π_θ(a) > μ(a)，应该给予更高权重。

这要求 φ(w, A)/A 关于 w 单调增（对于固定的 A）。

### 6.2 SAPO-IS 的问题

回顾 sapo_is 的设计：
```python
gate_pos = 2 * sigmoid(-tau * (w - 1))  # 对 A > 0
gate_neg = 2 * sigmoid(tau * (w - 1))   # 对 A < 0
```

**问题**：gate_pos 关于 w **单调递减**！

这违反了 IS 的基本原则：w 越大，权重反而越小。

### 6.3 单调性 vs 凹凸性

| 需求 | 含义 | 对 γ 的影响 |
|------|------|------------|
| 单调增 | φ'(w) > 0 | 任意 γ > 0 满足 |
| 凹 (A > 0) | φ''(w) < 0 | γ < 1 |
| 凸 (A < 0) | φ''(w) > 0 | γ > 1 |

**结论**：w^γ 天然满足单调性，凹凸性由 γ 与 1 的关系决定。

---

# 第三部分：与 α-Divergence 的联系

## 7. 当 φ 可积分时

### 7.1 从 φ 到 f

对于 φ(w, A) = w^γ · A：

$$f(w, A) = \int \frac{w^\gamma \cdot A}{w} dw = \frac{w^\gamma}{\gamma} \cdot A + h(A)$$

取 h(A) = -A/γ（使得 f(1, A) = 0）：

$$f(w, A) = \frac{w^\gamma - 1}{\gamma} \cdot A$$

这是 **Box-Cox 变换** 乘以优势。

### 7.2 与 α-Divergence 的对应

**α-Divergence** 定义：
$$D_\alpha(\pi \| \mu) = \frac{1}{\alpha(\alpha-1)} \left( \mathbb{E}_\mu[w^\alpha] - 1 \right)$$

其梯度结构：
$$\nabla_\theta D_\alpha \propto \mathbb{E}_\mu[w^\alpha \cdot \nabla \log \pi]$$

**对应关系**：

$$\boxed{\phi(w) = w^\gamma \quad \Longleftrightarrow \quad \alpha\text{-divergence with } \alpha = \gamma}$$

### 7.3 α 的意义

| α (= γ) | Divergence 类型 | 行为 |
|---------|----------------|------|
| α → 0 | Forward KL | Mean-seeking（覆盖所有模式）|
| α = 1 | Reverse KL | Mode-seeking（聚焦主模式）|
| α > 1 | χ²-divergence 方向 | 强 mode-seeking |

### 7.4 Reward-Modulated α-Divergence

当 γ 依赖于 A 时：

$$\phi(w, A) = w^{\gamma(A)} \cdot A$$

这对应于**自适应的 α-divergence**：
- A > 0：使用 α_+ < 1（Forward KL 方向，mean-seeking）
- A < 0：使用 α_- > 1（χ² 方向，mode-seeking）

**直觉**：
- 学习好动作时，要覆盖所有好的模式（mean-seeking）
- 避免坏动作时，要聚焦最危险的模式（mode-seeking）

---

## 8. 当 φ 不直接对应标准 f-Divergence 时

### 8.1 SAPO 的情况

SAPO 的权重函数：
$$\phi_{SAPO}(w) = \sigma(\tau(w-1)) \cdot \frac{4}{\tau}$$

**检验可积性**：
$$\frac{\phi(w)}{w} = \frac{4\sigma(\tau(w-1))}{\tau \cdot w}$$

这可以积分，但结果不是任何标准 f-divergence 的形式。

### 8.2 这意味着什么？

SAPO 定义了一个合法的目标函数（因为可积），但：
- 不对应任何标准的 f-divergence
- 没有明确的信息论解释
- 是一种 **工程驱动的设计**，而非理论驱动

### 8.3 工程设计 vs 理论设计

| 方面 | 理论驱动 (IS-Reshape) | 工程驱动 (SAPO) |
|------|----------------------|----------------|
| 起点 | α-divergence | 有界性需求 |
| φ 形式 | w^γ | sigmoid gate |
| f-divergence | ✓ 对应 | ✗ 不对应标准形式 |
| 有界性 | 需要额外处理 | 天然满足 |
| 理论解释 | 明确 | 缺乏 |

**实用结论**：两种路径都有价值，取决于需求。

---

# 第四部分：具体设计

## 9. φ(w, A) 的推荐形式

### 9.1 基于 α-Divergence 的设计

$$\phi(w, A) = w^{\gamma(A)} \cdot A$$

其中：
$$\gamma(A) = \begin{cases}
\gamma_+ = \min(1-\epsilon, \sqrt{-\log\rho_{min}/\sigma_+^2}) & A > 0 \\
\gamma_- = 1 + \min(1-\epsilon, \sqrt{-\log\rho_{min}/\sigma_-^2}) & A < 0
\end{cases}$$

**优点**：有明确的 α-divergence 理论基础
**缺点**：γ_- > 1 时 φ 无界

### 9.2 带软饱和的设计

对于 A < 0，使用软饱和避免无界：

$$\phi_-(w, A) = \frac{w^{\gamma_-}}{1 + \lambda(w - w_{th})^+} \cdot A$$

或使用 S 形函数（如 Michaelis-Menten）：

$$\phi_-(w, A) = \frac{(1+\tau)w}{\tau + w} \cdot A$$

**特性**：
- 小 w 时近似 w（或 w^γ）
- 大 w 时饱和
- 损失部分凸性，换取有界性

### 9.3 混合设计

$$\phi(w, A) = \begin{cases}
w^{\gamma_+} \cdot A & A > 0 \\
[\beta(w) \cdot w^{\gamma_-} + (1-\beta(w)) \cdot w] \cdot A & A < 0
\end{cases}$$

其中 $\beta(w) = \sigma(-\lambda(w - w_{th}))$。

**含义**：
- 小 w 时使用凸函数 w^γ_-（mode-seeking）
- 大 w 时过渡到线性 w（有界增长）

---

## 10. SFT 特例：A > 0 恒成立

### 10.1 简化

当所有样本都是"好样本"（A > 0）时：
- 只需要 mean-seeking（γ < 1）
- 只需要凹函数
- **凹 + 有界 + 单调** 可以同时满足！

### 10.2 推荐的 φ

**选项 1：幂函数**
$$\phi(w) = w^\gamma, \quad \gamma \in (0, 1)$$

**选项 2：Michaelis-Menten**
$$\phi(w) = \frac{(1+\tau)w}{\tau + w}$$

性质：
- 凹 ✓
- 有界于 1+τ ✓
- 单调增 ✓
- φ(1) = 1 ✓

### 10.3 SFT Loss 改进

**标准 SFT**（隐式 φ = 1）：
$$L = -\mathbb{E}[\log \pi_\theta(y|x)]$$

**IS-Reshape SFT**（φ = w^γ）：
$$L = -\mathbb{E}[w^\gamma \cdot \log \pi_\theta(y|x)]$$

**效果**：
- 新样本（w < 1）获得相对更高权重
- 已学样本（w > 1）权重被压缩
- 自动的课程学习效果

---

# 第五部分：总结

## 11. 框架对比

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     梯度权重 φ(w, A) 设计框架                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  【传统路径】                                                             │
│      定义目标 f(w, A)  →  推导 φ = ∂f/∂w · w  →  分析性质               │
│                                                                         │
│  【本文路径】                                                             │
│      设计 φ(w, A)  →  检验可积性  →  （若可积）连接到 f-divergence        │
│           ↓                                                             │
│      直接分析 φ 的性质：偏差、方差、有界性、单调性、凹凸性                  │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  【核心结论】                                                             │
│                                                                         │
│  1. φ(w) = w^γ 对应 α = γ 的 α-divergence                               │
│                                                                         │
│  2. 正负样本需要不同的 γ：                                                │
│     - A > 0: γ < 1 (凹，mean-seeking)                                   │
│     - A < 0: γ > 1 (凸，mode-seeking)                                   │
│                                                                         │
│  3. 有界性困境：凸 + 有界 + 单调 不可兼得                                 │
│     → 需要软饱和或混合设计                                               │
│                                                                         │
│  4. SAPO 是工程驱动设计，不对应标准 f-divergence                          │
│                                                                         │
│  5. SFT (A > 0 only) 约束放松：凹 + 有界 + 单调 可以满足                  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## 12. 设计决策树

```
                        φ(w, A) 设计
                             │
                ┌────────────┴────────────┐
                │                         │
           需要理论基础？              需要工程稳定性？
                │                         │
                ↓                         ↓
          φ = w^γ · A               SAPO-style gate
          (α-divergence)            (sigmoid, 天然有界)
                │
        ┌───────┴───────┐
        │               │
      A > 0           A < 0
        │               │
        ↓               ↓
    γ < 1 (凹)      γ > 1 (凸)
    天然OK          需要处理有界性
                        │
              ┌─────────┼─────────┐
              │         │         │
           截断      软饱和     混合设计
```

---

## 附录 A：关键公式

**梯度估计器**：
$$\hat{g} = \mathbb{E}_\mu[\phi(w, A) \cdot \nabla_\theta \log \pi_\theta]$$

**可积性条件**：
$$f(w, A) = \int \frac{\phi(w, A)}{w} dw$$

**α-Divergence 对应**：
$$\phi(w) = w^\gamma \quad \Longleftrightarrow \quad D_\gamma(\pi \| \mu)$$

**ESS 约束下的 γ**：
$$\gamma = \sqrt{\frac{-\log \rho_{min}}{\sigma^2}}$$

---

## 附录 B：代码实现

```python
import torch
import math
from typing import Tuple, Dict, Optional

class GradientWeightDesign:
    """
    梯度权重 φ(w, A) 的设计和计算

    核心思想：
    - 直接设计 φ，不预设目标函数
    - φ = w^γ 对应 α-divergence (α = γ)
    - 正负样本使用不同的 γ
    """

    def __init__(
        self,
        rho_min: float = 0.3,
        gamma_pos_range: Tuple[float, float] = (0.1, 0.99),
        gamma_neg_range: Tuple[float, float] = (1.01, 1.99),
        use_soft_saturation: bool = True,
        saturation_threshold: float = 3.0,
        saturation_lambda: float = 1.0,
    ):
        self.rho_min = rho_min
        self.log_rho = -math.log(rho_min)
        self.gamma_pos_range = gamma_pos_range
        self.gamma_neg_range = gamma_neg_range
        self.use_soft_saturation = use_soft_saturation
        self.saturation_threshold = saturation_threshold
        self.saturation_lambda = saturation_lambda

    def compute_gamma(
        self,
        log_w: torch.Tensor,
        A: torch.Tensor
    ) -> Tuple[float, float]:
        """计算正负样本的 γ 值"""
        pos_mask = A > 0
        neg_mask = A < 0

        # 正样本 γ_+
        if pos_mask.any():
            sigma_sq_pos = log_w[pos_mask].var().item()
            sigma_sq_pos = max(sigma_sq_pos, 1e-8)
            gamma_pos = min(
                self.gamma_pos_range[1],
                math.sqrt(self.log_rho / sigma_sq_pos)
            )
            gamma_pos = max(self.gamma_pos_range[0], gamma_pos)
        else:
            gamma_pos = 0.5

        # 负样本 γ_-
        if neg_mask.any():
            sigma_sq_neg = log_w[neg_mask].var().item()
            sigma_sq_neg = max(sigma_sq_neg, 1e-8)
            gamma_neg = 1.0 + min(
                self.gamma_neg_range[1] - 1.0,
                math.sqrt(self.log_rho / sigma_sq_neg)
            )
            gamma_neg = max(self.gamma_neg_range[0], gamma_neg)
        else:
            gamma_neg = 1.5

        return gamma_pos, gamma_neg

    def compute_phi(
        self,
        w: torch.Tensor,
        A: torch.Tensor,
        gamma_pos: float,
        gamma_neg: float,
    ) -> torch.Tensor:
        """
        计算梯度权重 φ(w, A)

        φ(w, A) = w^{γ(A)} · A  （基础形式）

        对于 A < 0，可选软饱和处理
        """
        pos_mask = A > 0
        neg_mask = A < 0

        phi = torch.zeros_like(A)

        # 正样本：φ = w^{γ_+} · A
        if pos_mask.any():
            phi[pos_mask] = (w[pos_mask] ** gamma_pos) * A[pos_mask]

        # 负样本：φ = w^{γ_-} · A，可选软饱和
        if neg_mask.any():
            w_neg = w[neg_mask]
            A_neg = A[neg_mask]

            if self.use_soft_saturation:
                # 软饱和：大 w 时混入线性
                beta = torch.sigmoid(
                    -self.saturation_lambda * (w_neg - self.saturation_threshold)
                )
                weight = beta * (w_neg ** gamma_neg) + (1 - beta) * w_neg
            else:
                weight = w_neg ** gamma_neg

            phi[neg_mask] = weight * A_neg

        return phi

    def __call__(
        self,
        log_pi: torch.Tensor,
        log_mu: torch.Tensor,
        advantages: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        计算损失

        L = -E[φ(w, A)]  (最大化 φ·A 的期望)
        """
        log_w = log_pi - log_mu
        log_w = torch.clamp(log_w, -20.0, 20.0)
        w = torch.exp(log_w)

        # 计算 γ
        gamma_pos, gamma_neg = self.compute_gamma(log_w.detach(), advantages)

        # 计算 φ
        phi = self.compute_phi(w, advantages, gamma_pos, gamma_neg)

        # 损失
        if mask is not None:
            loss = -(phi * mask).sum() / (mask.sum() + 1e-8)
        else:
            loss = -phi.mean()

        # 诊断
        with torch.no_grad():
            metrics = {
                'gamma_pos': gamma_pos,
                'gamma_neg': gamma_neg,
                'phi_mean': phi.mean().item(),
                'phi_abs_max': phi.abs().max().item(),
                'w_mean': w.mean().item(),
                'w_max': w.max().item(),
            }

        return loss, metrics


# SFT 特例
class SFTGradientWeight:
    """
    SFT 场景（A > 0 恒成立）的简化版本

    只需要 γ < 1 的凹函数
    """

    def __init__(
        self,
        gamma: float = 0.5,
        use_michaelis_menten: bool = False,
        mm_tau: float = 1.0,
    ):
        self.gamma = gamma
        self.use_mm = use_michaelis_menten
        self.mm_tau = mm_tau

    def compute_phi(self, w: torch.Tensor) -> torch.Tensor:
        """计算 φ(w)"""
        if self.use_mm:
            # Michaelis-Menten: (1+τ)w / (τ+w)
            return (1 + self.mm_tau) * w / (self.mm_tau + w)
        else:
            # 幂函数: w^γ
            return w ** self.gamma

    def __call__(
        self,
        log_pi: torch.Tensor,
        log_mu: torch.Tensor,
    ) -> torch.Tensor:
        """
        SFT Loss with IS-Reshape

        L = -E[φ(w) · log π]
        """
        log_w = log_pi - log_mu
        log_w = torch.clamp(log_w, -20.0, 20.0)
        w = torch.exp(log_w)

        phi = self.compute_phi(w)

        loss = -(phi * log_pi).mean()
        return loss
```
