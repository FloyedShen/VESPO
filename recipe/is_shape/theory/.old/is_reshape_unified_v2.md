# IS Reshape 统一框架：完整理论与实践指南

**版本**: 2.0（整合经典IS理论）

---

## 摘要

本文从经典重要性采样（IS）理论出发，建立一个统一 SFT、RL 和蒸馏的理论框架。核心思想是：**幂次整形函数 f(w) = w^γ 在策略梯度中的作用，等价于在不同散度目标之间插值**。

**理论根基**：
- Power Tempering (Owen 2013): w^γ 形式的偏差-方差权衡
- Self-Normalized IS (Kong 1992): 精确的偏差公式
- Pareto Smoothed IS (Vehtari 2024): 尾部诊断与平滑
- α-Divergence (Amari 2016): 散度族的统一视角

**独特贡献**：
- 将经典 IS 理论与 **RLHF/LLM 对齐** 建立精确对应
- γ 参数的 **SFT-RL 谱系解释**
- 有理论保证的 **自适应 γ 选择方法**
- 与 **PPO Clip** 的系统对比

---

# 第一部分：理论基础

## 1. 从经典重要性采样出发

### 1.1 标准 IS 估计量

**问题设定**：给定目标分布 π 和提议分布 μ，估计 $\mathbb{E}_\pi[h(y)]$。

**标准 IS 估计量**：
$$\hat{\mu}_{\text{IS}} = \frac{1}{n} \sum_{i=1}^n w_i \cdot h(y_i), \quad w_i = \frac{\pi(y_i)}{\mu(y_i)}, \quad y_i \sim \mu$$

**性质**：
- 无偏：$\mathbb{E}[\hat{\mu}_{\text{IS}}] = \mathbb{E}_\pi[h]$
- 方差：$\text{Var}(\hat{\mu}_{\text{IS}}) = \frac{1}{n} \text{Var}_\mu[w \cdot h]$

**问题**：当 π 和 μ 差异大时，某些 $w_i$ 可能极大，导致高方差。

### 1.2 Power Tempering: w^γ 形式

**经典方法**（Owen 2013, §9.9）：使用 $w^\gamma$ 代替 $w$，其中 $\gamma \in (0, 1]$

$$\hat{\mu}_\gamma = \frac{\sum_i w_i^\gamma \cdot h(y_i)}{\sum_i w_i^\gamma}$$

**关键性质**（经典结果）：

| γ | 偏差 | 方差 | 等价目标 |
|---|------|------|---------|
| 0 | 最大（指向 μ） | 最小 | $\mathbb{E}_\mu[h]$ |
| 1 | 0 | 最大 | $\mathbb{E}_\pi[h]$ |
| (0,1) | 介于两者 | 介于两者 | 某种插值 |

### 1.3 自归一化 IS (SNIS)

**定义**（Kong 1992）：
$$\hat{\mu}_{\text{SNIS}} = \sum_i \bar{w}_i \cdot h(y_i), \quad \bar{w}_i = \frac{w_i}{\sum_j w_j}$$

**偏差公式**（Kong-Owen）：
$$\boxed{\text{Bias}(\hat{\mu}_{\text{SNIS}}) = -\frac{1}{n} \cdot \frac{\text{Cov}_\mu(w, h)}{\mathbb{E}_\mu[w]} + O(n^{-2})}$$

**关键洞察**：偏差是 $O(1/n)$，渐近可忽略，但有限样本下需要考虑。

### 1.4 PSIS 诊断

**Pareto k 参数**（Vehtari et al. 2024）：

权重尾部服从广义 Pareto 分布，形状参数 k 刻画尾部厚度：

| k̂ 值 | 有限矩阶数 | IS 可靠性 |
|------|-----------|----------|
| < 0.5 | > 2 | 非常可靠 |
| 0.5 - 0.7 | 1.4 - 2 | 可靠 |
| 0.7 - 1 | 1 - 1.4 | 不太可靠 |
| ≥ 1 | < 1 | 不可靠 |

---

## 2. 统一梯度框架

### 2.1 核心定义

**符号约定**：
| 符号 | 定义 |
|------|------|
| μ(y\|x) | 行为策略（数据来源）|
| π_θ(y\|x) | 待学习策略 |
| r(x,y) | 奖励函数 |
| w = π_θ/μ | 重要性权重 |
| f(w) | IS 整形函数 |

**定义 2.1（统一策略梯度）**：

$$\boxed{g(\theta) = \mathbb{E}_\mu\left[f(w) \cdot r(x,y) \cdot \nabla_\theta \log \pi_\theta(y|x)\right]}$$

这是整个框架的核心。不同的 f(w) 选择导致不同的优化行为。

### 2.2 两个极端的精确等价

**定理 2.2（f(w)=1 → Forward KL）**：

当 f(w) = 1 时，梯度 $g(\theta) = \mathbb{E}_\mu[r \cdot \nabla\log\pi_\theta]$ 对应目标：

$$\max_\theta \mathbb{E}_\mu[r \cdot \log\pi_\theta] \iff \min_\theta D_{KL}(\tilde{\pi} \| \pi_\theta)$$

其中 $\tilde{\pi}(y|x) \propto \mu(y|x) \cdot r(x,y)$ 是奖励倾斜分布。

**前提条件**：需要 **r(x,y) > 0** 对所有 (x,y) 成立，否则 $\tilde{\pi}$ 不是有效概率分布。对于可正可负的奖励，应使用指数形式 $\tilde{\pi}^{\exp} \propto \mu \cdot e^{r/\tau}$。

**证明**：Forward KL 展开：
$$D_{KL}(\tilde{\pi} \| \pi_\theta) = \mathbb{E}_{\tilde{\pi}}[\log\tilde{\pi}] - \mathbb{E}_{\tilde{\pi}}[\log\pi_\theta]$$

第二项：$\mathbb{E}_{\tilde{\pi}}[\log\pi_\theta] = \frac{\mathbb{E}_\mu[r \cdot \log\pi_\theta]}{\mathbb{E}_\mu[r]}$

因此 $\min D_{KL}(\tilde{\pi} \| \pi_\theta) \iff \max \mathbb{E}_\mu[r \cdot \log\pi_\theta]$。$\blacksquare$

**性质**：Forward KL 是 **mean-seeking**——π_θ 试图覆盖 $\tilde{\pi}$ 的所有模式。

---

**定理 2.3（f(w)=w → Reverse KL 方向）**：

当 f(w) = w 时，梯度变为 $g(\theta) = \mathbb{E}_{\pi_\theta}[r \cdot \nabla\log\pi_\theta]$（标准 REINFORCE）。

**三种相关目标**：

**(a) 纯策略梯度**（无正则化）：
$$g(\theta) = \nabla_\theta \mathbb{E}_{\pi_\theta}[r]$$

**(b) 熵正则化**（最大熵 RL）：
$$\max_\theta \left[\mathbb{E}_{\pi_\theta}[r] + \tau H(\pi_\theta)\right] \iff \min_\theta D_{KL}(\pi_\theta \| p^*)$$
其中 $p^*(y|x) = e^{r(x,y)/\tau}/Z$（**不含 μ**）。

**(c) KL 正则化**（RLHF 常用形式）：
$$\max_\theta \left[\mathbb{E}_{\pi_\theta}[r] - \beta \cdot D_{KL}(\pi_\theta \| \mu)\right] \iff \min_\theta D_{KL}(\pi_\theta \| p^*_\beta)$$
其中 $p^*_\beta(y|x) = \mu(y|x) \cdot e^{r(x,y)/\beta} / Z_\beta$（**含 μ**）。

**证明 (c)**：
$$D_{KL}(\pi_\theta \| p^*_\beta) = \mathbb{E}_{\pi_\theta}\left[\log\frac{\pi_\theta}{\mu \cdot e^{r/\beta}/Z_\beta}\right]$$
$$= \mathbb{E}_{\pi_\theta}[\log\pi_\theta - \log\mu - r/\beta + \log Z_\beta]$$
$$= D_{KL}(\pi_\theta \| \mu) - \frac{1}{\beta}\mathbb{E}_{\pi_\theta}[r] + \text{const}$$

因此 $\min D_{KL}(\pi_\theta \| p^*_\beta) \iff \max \left[\mathbb{E}_{\pi_\theta}[r] - \beta D_{KL}(\pi_\theta \| \mu)\right]$。$\blacksquare$

**重要说明**：
- f(w) = w 的梯度直接对应 (a)，即 $\nabla \mathbb{E}_{\pi_\theta}[r]$
- 目标 (c) 的完整梯度还包含 KL 惩罚项：$\nabla_\theta D_{KL}(\pi_\theta \| \mu) = \mathbb{E}_{\pi_\theta}[(1+\log\frac{\pi_\theta}{\mu})\nabla\log\pi_\theta]$
- 散度等价关系 (b)(c) 说明了**优化的最终目标**，但梯度形式需要分别计算

**性质**：Reverse KL 是 **mode-seeking**——π_θ 聚焦于 $p^*_\beta$ 的主要模式。

---

### 2.3 SFT 作为特殊情况

**命题 2.4**：标准 SFT 是 f(w) = 1, r ≡ 1 的特殊情况。

此时 $\tilde{\pi} = \mu$，目标变为：
$$\min_\theta D_{KL}(\mu \| \pi_\theta) = \max_\theta \mathbb{E}_\mu[\log\pi_\theta]$$

这正是**最大似然估计**。

### 2.4 小结：两极对比

| 特性 | f(w) = 1 | f(w) = w |
|------|----------|----------|
| 梯度采样 | 从 μ 采样 | 等价于从 π_θ 采样 |
| 散度目标 | min D_KL(π̃ \|\| π_θ) | min D_KL(π_θ \|\| p*) |
| 优化行为 | Mean-seeking | Mode-seeking |
| 方差 | 低（无 IS 校正） | 高（完整 IS 校正） |
| 偏差 | 高（目标偏向 μ） | 零（无偏） |
| 对应方法 | 奖励加权 SFT | 策略梯度 RL |

---

## 3. f(w) = w^γ：散度谱系的刻画

### 3.1 与 Rényi/α-散度的对应

**定义 3.1（α-散度，Amari 2016）**：

$$D_\alpha(p \| q) = \frac{1}{\alpha(1-\alpha)}\left(1 - \int p^\alpha q^{1-\alpha}\right)$$

**定理 3.2（f(w) = w^γ 与 α-散度的关系）**：

设 α = 1-γ，f(w) = w^γ 的梯度与 α-散度有以下关系：

**(a) 边界情况精确**：
- γ = 0 (α = 1)：Forward KL 的梯度
- γ = 1 (α = 0)：Reverse KL 的梯度

**(b) 中间值的形式对应**：
α-散度 $D_\alpha(\pi_\theta \| p^*)$ 对 θ 的梯度包含 $w^\alpha = w^{1-\gamma}$ 项。当我们使用 f(w) = w^γ 时，实际对应的是 α = γ（而非 α = 1-γ）的**加权方向**。

**(c) 近似条件**：
当奖励 r 相对于温度参数较小（$|r|/\beta \ll 1$）时，$e^{(1-\alpha)r/\beta} \approx 1$，此时 f(w) = w^γ 近似对应 α-散度的梯度方向。

**对应关系表**：

| γ | 梯度权重 | 定性行为 | 极限散度 |
|---|---------|---------|---------|
| 0 | $w^0 = 1$ | Mean-seeking | Forward KL |
| 0.5 | $w^{0.5}$ | 中间 | 近似 Hellinger |
| 1 | $w^1 = w$ | Mode-seeking | Reverse KL |

### 3.2 Rényi 散度视角

**定义**：α-Rényi 散度
$$D_\alpha^R(p \| q) = \frac{1}{\alpha-1} \log \mathbb{E}_q\left[\left(\frac{p}{q}\right)^\alpha\right]$$

**关键等式**：
$$\boxed{\mathbb{E}_\mu[w^\gamma] = \exp\left((\gamma-1) \cdot D_\gamma^R(\pi_\theta \| \mu)\right)}$$

**证明**：由定义，$D_\gamma^R(\pi_\theta \| \mu) = \frac{1}{\gamma-1}\log\mathbb{E}_\mu[w^\gamma]$，移项即得。$\blacksquare$

这建立了 **IS 整形函数与信息散度的直接联系**。

### 3.3 几何插值解释

**定理 3.3（分布族的几何路径）**：

定义分布族：
$$p_\gamma(y|x) \propto \mu(y|x)^{1-\gamma} \cdot p^*(y|x)^\gamma$$

其中 $p^* = \mu \cdot e^{r/\beta}/Z$。这是 μ 和 p* 之间的 **几何插值**。

**正确解释**：

**(a) 信息几何视角**：$p_\gamma$ 是 μ 和 p* 在 **Fisher-Rao 度量**下的测地线（geodesic）。

**(b) 散度凸组合**：$p_\gamma$ 是以下优化的解：
$$p_\gamma = \arg\min_p \left[(1-\gamma) D_{KL}(p \| \mu) + \gamma D_{KL}(p \| p^*)\right]$$

**(c) 与 IS Reshape 的关系**：f(w) = w^γ 的优化**沿着 $p_\gamma$ 方向移动**，但不一定精确收敛到 $p_\gamma$。这是一个梯度方向的对应，而非最优解的对应。

**验证 (b)**：对 $J(p) = (1-\gamma)D_{KL}(p\|\mu) + \gamma D_{KL}(p\|p^*)$ 求变分，
$$\frac{\delta J}{\delta p} = (1-\gamma)(\log p - \log\mu + 1) + \gamma(\log p - \log p^* + 1) = 0$$
$$\Rightarrow \log p = (1-\gamma)\log\mu + \gamma\log p^* + \text{const}$$
$$\Rightarrow p \propto \mu^{1-\gamma} (p^*)^\gamma$$
$\blacksquare$

---

## 4. 偏差-方差分析（融合经典理论）

### 4.1 方差分析

**定理 4.1（方差的 γ 依赖性）**：

IS Reshape 梯度的方差满足：
$$\text{Var}(g_\gamma) \propto \mathbb{E}_\mu[w^{2\gamma}] \cdot \mathbb{E}[r^2 \|\nabla\log\pi\|^2]$$

**Rényi 散度表示**：
$$\boxed{\mathbb{E}_\mu[w^{2\gamma}] = \exp\left((2\gamma-1) \cdot D_{2\gamma}^R(\pi_\theta \| \mu)\right)}$$

**推论 4.2（方差最小点）**：

在 Log-Normal 假设下（$\log w \sim \mathcal{N}(-\sigma^2/2, \sigma^2)$）：
$$V(\gamma) = e^{\sigma^2 \gamma(2\gamma-1)}$$

方差在 **γ = 0.25** 达到最小值（而非 γ = 0）。

**物理解释**：轻微的 IS 校正（γ ≈ 0.25）实际上**降低方差**，因为它平衡了正负贡献。

### 4.2 偏差分析（Kong-Owen 公式）

**定理 4.3（IS Reshape 的偏差）**：

应用 Kong-Owen 公式到 f(w) = w^γ：

$$\boxed{\text{Bias}(\gamma) = -\frac{1}{n} \cdot \frac{\text{Cov}_\mu(w^\gamma, r \cdot \nabla\log\pi)}{\mathbb{E}_\mu[w^\gamma]} + O(n^{-2})}$$

**性质**：
1. 当 γ = 0 时：Cov(1, ·) = 0，偏差为零（但目标本身偏向 μ）
2. 当 γ 增大时：Cov 增大，偏差（绝对值）先增后可能因 E[w^γ] 增大而减缓
3. 当 γ = 1 时：自归一化估计量，偏差为 O(1/n)

**Log-Normal 假设下的显式公式**：

$$\text{Bias}(\gamma) \approx -\frac{\rho \cdot \sigma_r \sigma_\nabla}{n} \cdot \left(e^{\sigma^2\gamma} - 1\right)$$

其中 ρ 是 r 和 ∇log π 的相关系数。

### 4.3 MSE 分解与最优 γ

**定理 4.4（完整 MSE 分解）**：

$$\text{MSE}(\gamma) = \underbrace{\frac{C_1}{n^2} \left(e^{\sigma^2\gamma} - 1\right)^2}_{\text{Bias}^2} + \underbrace{\frac{C_2}{n} e^{\sigma^2\gamma(2\gamma-1)}}_{\text{Variance}}$$

其中 $C_1, C_2$ 是与 γ 无关的常数。

**最优 γ 的存在性**：由于 MSE 在 γ ∈ [0,1] 上连续，且边界值有限，由 Weierstrass 定理知最优 γ* 存在。

**数值特性**：
- 小样本（n 小）：偏差项主导 → γ* 偏小
- 大样本（n 大）：方差项主导 → γ* 可以偏大

---

## 5. PSIS 诊断与稳定性理论

### 5.1 Pareto k 与 γ 的关系

**定理 5.1（Power Transform 对 Pareto k 的影响）**：

若 w 的尾部 Pareto 参数为 k（即 $P(w > t) \sim t^{-1/k}$ as $t \to \infty$），则 w^γ 的尾部参数为：

$$\boxed{\hat{k}(w^\gamma) = \gamma \cdot \hat{k}(w)}$$

**前提条件**：
1. **γ > 0**（否则 w^γ 不是单调变换）
2. **尾部精确服从 Pareto**（对于 sub-Pareto 尾部，关系是近似的）
3. **实际应用中**：$\hat{k}$ 是估计值，关系变为近似

**证明**：设 $P(w > t) \sim L(t) \cdot t^{-1/k}$，其中 L(t) 是慢变函数，则
$$P(w^\gamma > s) = P(w > s^{1/\gamma}) \sim L(s^{1/\gamma}) \cdot s^{-1/(\gamma k)}$$
由于 $L(s^{1/\gamma})$ 仍是 s 的慢变函数，新的 Pareto 指数为 $\gamma k$。$\blacksquare$

### 5.2 基于 PSIS 的 γ 上界

**推论 5.2（PSIS 约束的 γ 选择）**：

为保证 IS 可靠性（k̂ < 0.7），γ 应满足：

$$\boxed{\gamma \leq \gamma^*_{\text{PSIS}} = \frac{0.7}{\hat{k}(w)}}$$

**样本量修正**：对于有限样本 n，使用更严格的阈值
$$\gamma \leq \frac{1 - 1/\log_{10}(n)}{\hat{k}(w)}$$

### 5.3 ESS 约束

**定义 5.3（有效样本量）**：

$$\text{ESS}_\gamma = \frac{(\sum_i w_i^\gamma)^2}{\sum_i w_i^{2\gamma}} = \frac{n}{1 + \text{Var}(\bar{w}^\gamma)}$$

**约束**：要求 ESS/n ≥ ρ_min（通常取 0.3）

$$\gamma \leq \gamma^*_{\text{ESS}}: \text{ s.t. } \text{ESS}_\gamma / n \geq \rho_{\min}$$

### 5.4 联合 γ 选择（核心算法）

**定理 5.4（γ 选择的统一方法）**：

结合 PSIS 和 ESS 约束，最优 γ 为：

$$\boxed{\gamma^* = \min\left(\gamma^*_{\text{PSIS}}, \gamma^*_{\text{ESS}}, 1\right)}$$

这保证了：
1. 尾部行为良好（k̂ < 0.7）
2. 有效样本量充足（ESS/n ≥ ρ_min）
3. 不超过完整 IS（γ ≤ 1）

---

## 6. 稳定化技术

### 6.1 Defensive IS Reshape

**定义 6.1**（融合 Hesterberg 1995 的思想）：

$$f_{\text{def}}(w; \gamma, \alpha) = \frac{w^\gamma}{\alpha w^\gamma + (1-\alpha)}$$

**性质**：
- 权重有界：$f_{\text{def}} \leq 1/\alpha$
- α = 0：退化为标准 IS Reshape
- α = 1：退化为 unweighted（f ≡ 1）

**适用场景**：当 PSIS 诊断显示 k̂ > 1（极端 heavy-tail）时使用。

### 6.2 PSIS 平滑

**方法**（Vehtari et al. 2024）：

1. 拟合 w^γ 尾部的 Pareto 分布
2. 用拟合分布的期望顺序统计量替换最大的 M 个权重
3. 重新归一化

**优点**：保留更多信息（相比直接截断），同时控制方差。

### 6.3 Control Variates

**方法**（Owen & Zhou 2000）：

$$\hat{g}_{CV} = \frac{1}{n}\sum_i w_i^\gamma r_i \nabla\log\pi - c \cdot \left(\frac{1}{n}\sum_i w_i^\gamma - \hat{\mathbb{E}}[w^\gamma]\right)$$

最优系数：$c^* = \text{Cov}(w^\gamma r \nabla\log\pi, w^\gamma) / \text{Var}(w^\gamma)$

**实践**：在双样本估计中使用，进一步降低方差。

---

# 第二部分：实践方法论

## 7. γ-Annealing 训练策略

### 7.1 理论动机

**定理 7.1（几何路径的稳定性）**：

直接从 γ=0 跳到 γ=1 相当于从 μ 直接跳到 p*，可能导致：
- 梯度方差突变
- 训练不稳定
- 陷入局部最优

沿几何路径 p_γ 逐步移动可以：
- 平滑过渡
- 保持 ESS 稳定
- 利用中间分布的"搭桥"作用

### 7.2 Annealing 策略

**策略 1：线性 Annealing**
$$\gamma(t) = \gamma_{\min} + (\gamma_{\max} - \gamma_{\min}) \cdot \frac{t}{T}$$

**策略 2：Cosine Annealing**
$$\gamma(t) = \gamma_{\min} + (\gamma_{\max} - \gamma_{\min}) \cdot \frac{1 - \cos(\pi t/T)}{2}$$

**策略 3：ESS-Adaptive Annealing**
```
if ESS_ratio > 1.5 * ρ_min:
    γ ← min(γ + Δγ, γ_max)
elif ESS_ratio < 0.8 * ρ_min:
    γ ← max(γ - Δγ/2, γ_min)
```

### 7.3 与 PPO 的对比

| 方面 | PPO Clip | γ-Annealing |
|-----|----------|-------------|
| 约束方式 | 硬截断 | 软压缩 |
| 参数演化 | ε 固定 | γ 逐渐增大 |
| 理论基础 | Trust region | 几何插值 |
| 梯度信息 | 截断外为零 | 始终保留 |
| 极限行为 | 始终截断 | 恢复完整 IS |

---

## 8. 完整算法

### 8.1 IS Reshape 训练器

```python
class ISReshapeTrainer:
    """
    IS Reshape 训练器（完整版）

    融合:
    - 经典 IS 理论（Kong-Owen 偏差, PSIS 诊断）
    - 自适应 γ 选择（ESS + PSIS 联合）
    - γ-Annealing 策略
    - 可选稳定化技术
    """

    def __init__(
        self,
        model,
        ref_model,  # μ，冻结
        gamma_init: float = 0.3,
        gamma_max: float = 1.0,
        gamma_schedule: str = 'adaptive',  # 'fixed', 'linear', 'adaptive'
        ess_min_ratio: float = 0.3,
        psis_k_max: float = 0.7,
        use_defensive: bool = False,
        defensive_alpha: float = 0.1,
    ):
        self.model = model
        self.ref_model = ref_model
        self.gamma = gamma_init
        self.gamma_max = gamma_max
        self.gamma_schedule = gamma_schedule
        self.ess_min = ess_min_ratio
        self.psis_k_max = psis_k_max
        self.use_defensive = use_defensive
        self.alpha = defensive_alpha

        # 冻结参考模型
        for p in self.ref_model.parameters():
            p.requires_grad = False

        # 诊断历史
        self.diagnostics = {'gamma': [], 'ess_ratio': [], 'pareto_k': [], 'bias_proxy': []}

    def compute_log_importance_weights(self, x, y):
        """计算 log w = log π_θ - log μ"""
        with torch.no_grad():
            log_mu = self.ref_model.log_prob(y, x)
        log_pi = self.model.log_prob(y, x)
        return log_pi - log_mu, log_pi

    def estimate_pareto_k(self, w):
        """估计 Pareto k 参数"""
        # 取上 20% 作为尾部
        threshold = torch.quantile(w, 0.8)
        tail = w[w > threshold] - threshold

        if len(tail) < 10:
            return 0.0

        # 简化估计：k ≈ mean(log(tail)) / log(mean(tail))
        # 更精确的方法应使用 MLE
        log_tail = torch.log(tail + 1e-10)
        k_hat = log_tail.mean() / torch.log(tail.mean() + 1e-10)
        return max(0, min(k_hat.item(), 2.0))

    def compute_ess_ratio(self, log_w_gamma):
        """计算 ESS / n"""
        w = torch.exp(log_w_gamma - log_w_gamma.max())  # 数值稳定
        w_normalized = w / w.sum()
        ess = 1.0 / (w_normalized ** 2).sum()
        return (ess / len(w)).item()

    def select_gamma(self, log_w):
        """联合 γ 选择（核心算法）"""
        w = torch.exp(log_w)

        # 1. PSIS 约束
        k_hat = self.estimate_pareto_k(w)
        gamma_psis = self.psis_k_max / (k_hat + 1e-8)

        # 2. ESS 约束（二分搜索）
        gamma_low, gamma_high = 0.0, min(gamma_psis, self.gamma_max)
        for _ in range(15):
            gamma_mid = (gamma_low + gamma_high) / 2
            ess_ratio = self.compute_ess_ratio(gamma_mid * log_w)
            if ess_ratio >= self.ess_min:
                gamma_low = gamma_mid
            else:
                gamma_high = gamma_mid

        gamma_ess = gamma_low

        # 3. 联合选择
        return min(gamma_psis, gamma_ess, self.gamma_max), k_hat

    def compute_weights(self, log_w):
        """计算最终权重"""
        w_gamma = torch.exp(self.gamma * log_w)

        # 可选：Defensive mixing
        if self.use_defensive:
            w_gamma = w_gamma / (self.alpha * w_gamma + (1 - self.alpha))

        # 归一化
        weights = w_gamma / w_gamma.sum()
        return weights

    def compute_loss(self, x, y, rewards):
        """计算 IS Reshape 损失"""
        # 1. 计算 log importance weights
        log_w, log_pi = self.compute_log_importance_weights(x, y)

        # 2. 更新 γ（如果自适应）
        if self.gamma_schedule == 'adaptive':
            gamma_new, k_hat = self.select_gamma(log_w.detach())
            self.gamma = 0.9 * self.gamma + 0.1 * gamma_new  # 平滑更新
        else:
            k_hat = self.estimate_pareto_k(torch.exp(log_w.detach()))

        # 3. 计算权重
        weights = self.compute_weights(log_w.detach())

        # 4. 加权损失
        # 注意：rewards 可以是标量 1（纯 SFT）或实际奖励
        loss = -torch.sum(weights * rewards * log_pi)

        # 5. 记录诊断
        ess_ratio = self.compute_ess_ratio(self.gamma * log_w.detach())
        self.diagnostics['gamma'].append(self.gamma)
        self.diagnostics['ess_ratio'].append(ess_ratio)
        self.diagnostics['pareto_k'].append(k_hat)

        return loss, {
            'gamma': self.gamma,
            'ess_ratio': ess_ratio,
            'pareto_k': k_hat,
            'max_weight': weights.max().item(),
        }

    def update_gamma_schedule(self, step, total_steps):
        """更新 γ（非自适应模式）"""
        if self.gamma_schedule == 'linear':
            progress = step / total_steps
            self.gamma = 0.0 + self.gamma_max * progress
        elif self.gamma_schedule == 'cosine':
            progress = step / total_steps
            self.gamma = self.gamma_max * (1 - np.cos(np.pi * progress)) / 2
```

### 8.2 使用示例

```python
# 初始化
trainer = ISReshapeTrainer(
    model=policy_model,
    ref_model=reference_model,
    gamma_init=0.3,
    gamma_max=0.8,
    gamma_schedule='adaptive',
    ess_min_ratio=0.3,
    psis_k_max=0.7,
)

# 训练循环
for epoch in range(num_epochs):
    for batch in dataloader:
        x, y, rewards = batch['prompt'], batch['response'], batch['reward']

        loss, metrics = trainer.compute_loss(x, y, rewards)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 监控
        if step % 100 == 0:
            print(f"Step {step}: γ={metrics['gamma']:.3f}, "
                  f"ESS={metrics['ess_ratio']:.3f}, "
                  f"k̂={metrics['pareto_k']:.3f}")
```

---

## 9. 理论保证总结

### 9.1 收敛性

**定理 9.1（梯度估计的一致性）**：

在以下条件下，IS Reshape 梯度估计是渐近一致的：
1. ESS/n → ρ > 0（有效样本量非退化）
2. k̂ < 1（有限一阶矩）
3. rewards 有界

则 $\hat{g}_\gamma \xrightarrow{p} g_\gamma^*$ as n → ∞。

### 9.2 有限样本保证

**定理 9.2（集中不等式）**：

若 k̂ < 0.5（有限二阶矩），则以概率 1-δ：

$$\|\hat{g}_\gamma - g_\gamma^*\| \leq O\left(\sqrt{\frac{\log(1/\delta)}{n \cdot \text{ESS}/n}}\right)$$

### 9.3 γ 选择的最优性

**定理 9.3**：

联合 γ 选择方法 $\gamma^* = \min(\gamma^*_{\text{PSIS}}, \gamma^*_{\text{ESS}})$ 满足：
1. IS 估计量方差有限
2. 偏差可控
3. 在 Bias-Variance 权衡意义下近似最优

---

## 10. 与现有方法的关系

### 10.1 方法定位

| 方法 | IS Reshape 视角 |
|-----|----------------|
| 标准 SFT | f(w)=1, r≡1 |
| 奖励加权 SFT | f(w)=1, r=reward |
| REINFORCE | f(w)=w |
| PPO Clip | f(w) = clip(w, 1-ε, 1+ε)·sign(A) |
| **IS Reshape** | f(w) = w^γ, γ 自适应 |

### 10.2 独特优势

1. **统一视角**：SFT 和 RL 是同一框架的两端
2. **理论根基**：直接继承 30+ 年的经典 IS 理论
3. **可控性**：γ 提供精细的 mean-seeking ↔ mode-seeking 控制
4. **诊断完备**：ESS + PSIS k̂ 双重保障
5. **平滑过渡**：γ-annealing 避免训练不稳定

---

## 附录 A：符号汇总

| 符号 | 定义 |
|------|------|
| μ | 行为策略（数据分布）|
| π_θ | 待学习策略 |
| p* | KL 正则化最优策略 |
| w = π_θ/μ | 重要性权重 |
| γ | IS 整形指数 |
| k̂ | Pareto 尾部参数 |
| ESS | 有效样本量 |
| D_α | α-散度 |
| D^R_α | α-Rényi 散度 |

## 附录 B：关键公式速查

| 公式 | 表达式 |
|------|--------|
| 统一梯度 | $g = \mathbb{E}_\mu[w^\gamma r \nabla\log\pi]$ |
| 方差 | $\text{Var} \propto e^{\sigma^2\gamma(2\gamma-1)}$ |
| 偏差 | $\text{Bias} \approx -\frac{1}{n}\frac{\text{Cov}(w^\gamma, r\nabla\log\pi)}{\mathbb{E}[w^\gamma]}$ |
| Pareto k 变换 | $\hat{k}(w^\gamma) = \gamma \cdot \hat{k}(w)$ |
| ESS | $\text{ESS} = n/(1 + \text{Var}(\bar{w}))$ |
| γ 选择 | $\gamma^* = \min(\gamma^*_{\text{PSIS}}, \gamma^*_{\text{ESS}})$ |

## 附录 C：引用文献

1. Owen, A. B. (2013). Monte Carlo theory, methods and examples.
2. Kong, A. (1992). A note on importance sampling using standardized weights.
3. Hesterberg, T. (1995). Weighted average importance sampling and defensive mixture distributions.
4. Vehtari, A. et al. (2024). Pareto smoothed importance sampling. JMLR.
5. Amari, S. I. (2016). Information geometry and its applications.
6. Neal, R. M. (2001). Annealed importance sampling.
