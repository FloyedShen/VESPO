# 统一框架：从 IS Reshape 视角理解离线强化学习

## 摘要

本文从 **重要性采样整形（IS Reshape）** 的视角出发，建立一个统一 SFT、RL 和蒸馏的理论框架。核心思想是：不同的 IS 整形函数 f(w) 对应不同类型的散度优化，从而产生不同的学习行为。

**主要结论**：
- f(w) = 1 → Forward KL → Mean-seeking → **奖励加权 SFT**
- f(w) = w → Reverse KL → Mode-seeking → **RL**
- 一般的 f(w) → 广义散度 → **在两者之间插值**
- **纯 SFT** 是 f(w) = 1 且 r(x, y) ≡ const 的特例

---

# 第一部分：一般 f(w) 理论

## 1. 问题设定

### 1.1 符号定义

| 符号 | 定义 |
|------|------|
| $x$ | 上下文 / prompt |
| $y$ | 响应 / 动作 |
| $\mu(y\|x)$ | 行为策略（离线数据分布） |
| $\pi_\theta(y\|x)$ | 待学习的策略 |
| $r(x, y)$ | 奖励函数 |
| $\tau > 0$ | 温度参数 |
| $w = \frac{\pi_\theta(y\|x)}{\mu(y\|x)}$ | 重要性采样比率 |
| $f(w)$ | IS 整形函数 |

### 1.2 目标

给定从 $\mu$ 采样的离线数据集 $\mathcal{D} = \{(x_i, y_i, r_i)\}_{i=1}^n$，通过选择合适的 IS 整形函数 f(w)，学习一个最优策略 $\pi_\theta$。

### 1.3 统一梯度形式（核心定义）

**定义 1.1（统一策略梯度）**：带 IS 整形的策略梯度定义为：

$$\boxed{g(\theta) = \mathbb{E}_{y \sim \mu}\left[f(w) \cdot r(x,y) \cdot \nabla_\theta \log \pi_\theta(y|x)\right]}$$

其中 $w = \pi_\theta(y|x) / \mu(y|x)$。

**关键问题**：不同的 f(w) 选择会导致什么样的优化行为？

**备注**：我们从梯度形式出发定义框架，而非从目标函数出发。这是因为 SFT 和 RL 的目标函数形式本质不同，但它们的**梯度可以统一表示**。

---

## 2. 两个极端情况的精确分析

### 2.1 Case 1: f(w) = 1（无 IS 校正）

当 f(w) = 1 时，梯度简化为：

$$g(\theta) = \mathbb{E}_\mu[r(x,y) \cdot \nabla_\theta \log \pi_\theta(y|x)]$$

这对应于**奖励加权的最大似然估计**，其隐式目标函数为：

$$L(\theta) = \mathbb{E}_\mu[r(x,y) \cdot \log \pi_\theta(y|x)]$$

**定理 2.1（Forward KL 等价性）**：定义线性奖励倾斜分布（假设 $r(x,y) > 0$）：
$$\tilde{\pi}(y|x) = \frac{\mu(y|x) \cdot r(x,y)}{\mathbb{E}_\mu[r]}, \quad \text{其中 } \mathbb{E}_\mu[r] = \int \mu(y|x) r(x,y) dy$$

则最大化 $L(\theta)$ **严格等价于**最小化 Forward KL 散度：
$$\boxed{\max_\theta L(\theta) \iff \min_\theta D_{KL}(\tilde{\pi} \| \pi_\theta)}$$

**证明**：

Forward KL 散度展开为：
$$D_{KL}(\tilde{\pi} \| \pi_\theta) = \mathbb{E}_{\tilde{\pi}}[\log \tilde{\pi}] - \mathbb{E}_{\tilde{\pi}}[\log \pi_\theta]$$

第一项 $H(\tilde{\pi})$ 与 θ 无关。第二项：
$$\mathbb{E}_{\tilde{\pi}}[\log \pi_\theta] = \int \frac{\mu(y) \cdot r(y)}{\mathbb{E}_\mu[r]} \log \pi_\theta(y) dy = \frac{\mathbb{E}_\mu[r \cdot \log \pi_\theta]}{\mathbb{E}_\mu[r]}$$

因此：
$$\min_\theta D_{KL}(\tilde{\pi} \| \pi_\theta) \iff \max_\theta \mathbb{E}_\mu[r \cdot \log \pi_\theta] = \max_\theta L(\theta)$$

这是**严格等价**，不涉及任何近似。$\blacksquare$

**备注 2.1（线性 vs 指数奖励变换）**：

本文采用**线性形式** $\tilde{\pi} \propto \mu \cdot r$，但文献中也常见**指数形式** $\tilde{\pi}^{\exp} \propto \mu \cdot e^{r/\tau}$。

| 形式 | 对应目标 | 适用条件 | 一阶关系 |
|------|---------|---------|---------|
| 线性 | $\mathbb{E}_\mu[r \cdot \log \pi_\theta]$ | r > 0 | 当 $\|r - \bar{r}\| \ll \tau$ 时 |
| 指数 | $\mathbb{E}_\mu[e^{r/\tau} \cdot \log \pi_\theta]$ | 任意 r | 线性是指数的一阶近似 |

**两者在定性行为上一致**（都是 mean-seeking）。本文选择线性形式以简化主框架推导；§6 的 α-散度分析采用指数形式以对应 Boltzmann 分布。

可引入**广义奖励变换** $\phi(r)$，统一表示为 $\tilde{\pi}^\phi \propto \mu \cdot \phi(r)$，详见 `theory_refinements.md` 第 2 节。

**性质**：Forward KL 是 **mean-seeking** 的——$\pi_\theta$ 会试图覆盖 $\tilde{\pi}$ 的所有模式。

### 2.2 纯 SFT 作为特殊情况：r(x, y) ≡ const

**关键观察**：标准的监督微调（SFT）是 **f(w) = 1 且 r(x, y) ≡ 1** 的特殊情况。

**直觉**：
- 在实际中，用于 SFT 的数据是经过筛选的高质量数据
- 这相当于隐式假设：被选入训练集的样本都是"正确的"
- 等价于对所有训练样本赋予相同的奖励 r = 1

当 r ≡ 1 时：
- 线性奖励倾斜分布 $\tilde{\pi} = \mu \cdot 1 / \mathbb{E}_\mu[1] = \mu$
- 目标变为 $L(\theta) = \mathbb{E}_\mu[\log \pi_\theta]$

因此：

$$\boxed{L_{\text{SFT}}(\theta) = \mathbb{E}_\mu[\log \pi_\theta(y|x)] \iff \min_\theta D_{KL}(\mu \| \pi_\theta)}$$

这正是**标准的最大似然估计（MLE）/ 交叉熵损失**。

**命题 2.2**：SFT 是 IS Reshape 框架的特殊情况，对应于：
- f(w) = 1（无 IS 校正）
- r(x, y) ≡ 1（所有样本等权）

**推论**：f(w) = 1 对应的是**奖励加权 SFT**（Reward-Weighted SFT），纯 SFT 是其 r ≡ const 的退化情况。

### 2.3 Case 2: f(w) = w（完整 IS 校正）

当 f(w) = w 时，梯度变为：

$$g(\theta) = \mathbb{E}_\mu[w \cdot r \cdot \nabla_\theta \log \pi_\theta] = \mathbb{E}_{\pi_\theta}[r \cdot \nabla_\theta \log \pi_\theta]$$

这正是标准的 **REINFORCE 策略梯度**，对应目标：

$$L(\theta) = \mathbb{E}_{\pi_\theta}[r(x,y)]$$

**定理 2.3（Reverse KL 等价性）**：定义最优软策略（**纯由奖励决定**）：
$$p^*(y|x) = \frac{e^{r(x,y)/\tau}}{Z}, \quad Z = \int e^{r(y)/\tau} dy$$

则期望奖励最大化（带熵正则）等价于最小化 Reverse KL 散度：
$$\boxed{\max_\theta \left[\mathbb{E}_{\pi_\theta}[r] + \tau H(\pi_\theta)\right] \iff \min_\theta D_{KL}(\pi_\theta \| p^*)}$$

**证明**：

$$D_{KL}(\pi_\theta \| p^*) = \mathbb{E}_{\pi_\theta}\left[\log \frac{\pi_\theta}{p^*}\right]$$

$$= \mathbb{E}_{\pi_\theta}\left[\log \frac{\pi_\theta}{e^{r/\tau} / Z}\right]$$

$$= \mathbb{E}_{\pi_\theta}\left[\log \pi_\theta\right] - \frac{1}{\tau}\mathbb{E}_{\pi_\theta}[r] + \log Z$$

$$= -H(\pi_\theta) - \frac{1}{\tau}\mathbb{E}_{\pi_\theta}[r] + \log Z$$

由于 $\log Z$ 是常数：
$$\min_\theta D_{KL}(\pi_\theta \| p^*) \iff \max_\theta \left[\frac{1}{\tau}\mathbb{E}_{\pi_\theta}[r] + H(\pi_\theta)\right]$$

这正是**最大熵 RL**（Maximum Entropy RL）目标！$\blacksquare$

**备注 2.3（μ 的意义统一）**：注意 p* 的定义中**不含 μ**。这使得：
- **μ 在整个框架中只有一个意义**：采样分布（数据来源）
- **p* 纯由奖励决定**：目标策略不依赖于数据是从哪个分布采样的
- 避免了 μ 同时作为"采样分布"和"参考策略"的概念混淆

**备注 2.4（与 KL 正则化 RL 的关系）**：在 RLHF 实践中，更常见的是 **KL 正则化形式**：

$$p^*_\beta(y|x) = \frac{\mu(y|x) \cdot e^{r(x,y)/\beta}}{Z_\beta}, \quad \text{目标：} \max_\theta \left[\mathbb{E}_{\pi_\theta}[r] - \beta \cdot D_{KL}(\pi_\theta \| \mu)\right]$$

此时 μ 同时作为采样分布和 KL 惩罚的参考。**两种形式的关系**：
- 当 μ = uniform 时，$p^*_\beta$ 退化为 $p^* = e^{r/\tau}/Z$
- 在定性行为上两者一致（都是 mode-seeking）
- §6 的 α-散度分析采用 KL 正则化形式，与本节**定性一致**

详细分析见补充文档 `theory_refinements.md` 第 1 节。

**性质**：Reverse KL 是 **mode-seeking** 的——$\pi_\theta$ 会聚焦于 $p^*$ 的主要模式。

### 2.4 两个极端的对比

| 特性 | f(w) = 1 | f(w) = w |
|------|----------|----------|
| **梯度** | $\mathbb{E}_\mu[r \cdot \nabla \log \pi_\theta]$ | $\mathbb{E}_{\pi_\theta}[r \cdot \nabla \log \pi_\theta]$ |
| **隐式目标** | $\mathbb{E}_\mu[r \cdot \log \pi_\theta]$ | $\mathbb{E}_{\pi_\theta}[r] + \tau H(\pi_\theta)$ |
| **等价散度** | $D_{KL}(\tilde{\pi} \| \pi_\theta)$ | $D_{KL}(\pi_\theta \| p^*)$ |
| **目标分布** | $\tilde{\pi} = \mu \cdot r / Z$（依赖 μ） | $p^* = e^{r/\tau} / Z$（不依赖 μ） |
| **散度方向** | Forward KL | Reverse KL |
| **优化行为** | Mean-seeking | Mode-seeking |
| **分布覆盖** | 覆盖所有模式 | 聚焦最优模式 |
| **对应方法** | 奖励加权 SFT | 最大熵 RL |
| **梯度方差** | 低（无 IS） | 高（完整 IS） |
| **μ 的角色** | 采样分布 + 目标分布的一部分 | 仅采样分布 |

---

## 3. 一般 f(w) 的理论分析

### 3.1 梯度的一般形式

对于一般的可微 f(w)，统一梯度为：

$$g(\theta) = \mathbb{E}_\mu\left[f(w) \cdot r \cdot \nabla_\theta \log \pi_\theta\right]$$

这就是定义 1.1。对于 f(w) = w^γ 的情况，我们也可以从目标函数 $L(\theta) = \mathbb{E}_\mu[w^\gamma \cdot r / \gamma]$（γ > 0）推导出类似的梯度形式。

**验证特殊情况**：
- f(w) = 1：$g = \mathbb{E}_\mu[r \cdot \nabla_\theta \log \pi_\theta]$（奖励加权 SFT 梯度）
- f(w) = w：$g = \mathbb{E}_\mu[w \cdot r \cdot \nabla_\theta \log \pi_\theta] = \mathbb{E}_{\pi_\theta}[r \cdot \nabla_\theta \log \pi_\theta]$（RL 梯度）

### 3.2 有效权重表示

**定义 3.2（有效权重）**：对于一般的 f(w)，定义归一化的有效权重：

$$\tilde{w}_f(y) = \frac{f(w(y)) \cdot r(y)}{\mathbb{E}_\mu[f(w) \cdot r]}$$

**定理 3.3**：梯度可以统一写成归一化加权形式：
$$g(\theta) = c(f, \theta) \cdot \mathbb{E}_\mu\left[\tilde{w}_f \cdot \nabla_\theta \log \pi_\theta\right]$$

其中 $c(f, \theta) = \mathbb{E}_\mu[f(w) \cdot r]$ 是尺度因子。

**验证**：
$$c \cdot \mathbb{E}_\mu[\tilde{w}_f \cdot \nabla \log \pi] = \mathbb{E}_\mu[f(w) \cdot r] \cdot \mathbb{E}_\mu\left[\frac{f(w) \cdot r}{\mathbb{E}_\mu[f(w) \cdot r]} \cdot \nabla \log \pi\right] = \mathbb{E}_\mu[f(w) \cdot r \cdot \nabla \log \pi] = g(\theta) \quad \checkmark$$

### 3.3 有效目标分布

**定义 3.4（f-有效目标分布）**：假设 $r(x,y) > 0$，定义：

$$\pi_f^{\text{eff}}(y|x) = \frac{\mu(y|x) \cdot f(w(y)) \cdot r(y)}{Z_f}$$

其中 $Z_f = \int \mu(y) f(w(y)) r(y) dy = \mathbb{E}_\mu[f(w) \cdot r]$。

**关键观察**：$\pi_f^{\text{eff}}$ 是否依赖于 θ？

| f(w) | 有效目标 | 目标是否依赖 θ |
|------|----------|---------------|
| 1 | $\mu \cdot r / Z$ | **否**（固定目标） |
| w | $\pi_\theta \cdot r / Z'$ | **是**（移动目标） |
| $w^\gamma$ | $\pi_\theta^\gamma \mu^{1-\gamma} \cdot r / Z''$ | **是**（移动目标） |

**命题 3.5**：
- 当目标**固定**时，优化是"追赶"一个静态分布（类似 SFT）
- 当目标**移动**时，优化是"追逐"一个随 $\pi_\theta$ 变化的分布（类似 RL）

### 3.4 方差特性

**定理 3.6（梯度方差）**：梯度估计器的方差满足：

$$\text{Var}(\hat{g}) \propto \mathbb{E}_\mu\left[f(w)^2 \cdot r^2 \cdot \|\nabla_\theta \log \pi_\theta\|^2\right]$$

**推论 3.7**：f(w) 的增长速度决定了方差的大小：
- f(w) = 1：方差最小（不依赖 w）
- f(w) = w：方差可能很大（当 w 变化大时）
- f(w) = $w^\gamma$：方差介于两者之间

### 3.5 f(w) 的约束条件

**命题 3.8**：为保证优化良定义，f(w) 应满足：

1. **非负性**：$f(w) \geq 0, \forall w \geq 0$
2. **边界行为**：$f(0) = 0$ 或 $f(0)$ 有界
3. **单调性**（推荐）：$f'(w) \geq 0$（更大的 w 给予更大的权重）
4. **增长控制**：$f(w)$ 的增长不能太快，否则方差爆炸

### 3.6 广义散度解释

**定义 3.9（f-加权梯度对应的隐式散度）**：

f(w) 加权的梯度隐式地优化某种广义散度 $D_f$，满足：
$$g(\theta) = -\nabla_\theta D_f(\pi_\theta, p^*_f)$$

其中 $p^*_f$ 是 f 对应的有效目标分布。

**特殊情况**：
- f(w) = 1：$D_1 \propto D_{KL}(\tilde{\pi} \| \pi_\theta)$（Forward KL）
- f(w) = w：$D_w \propto D_{KL}(\pi_\theta \| p^*)$（Reverse KL）

### 3.7 f(w) 的谱系图

```
                            f(w) 的选择空间
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

   f(w) = 1                                                f(w) = w
      │                                                        │
      ▼                                                        ▼
┌─────────────┐                                        ┌─────────────┐
│ Forward KL  │                                        │ Reverse KL  │
│ D(p||π_θ)   │◄─────────── 一般 f(w) ──────────────►│ D(π_θ||p)   │
│ Mean-seeking│           广义散度 D_f                │ Mode-seeking│
└─────────────┘                                        └─────────────┘
      │                                                        │
      ▼                                                        ▼
┌─────────────┐                                        ┌─────────────┐
│   覆盖所有   │                                        │  聚焦最优   │
│    模式     │                                        │    模式     │
└─────────────┘                                        └─────────────┘
      │                                                        │
      ▼                                                        ▼
┌─────────────┐                                        ┌─────────────┐
│ 奖励加权 SFT │                                        │    RL       │
│ (r≡1时纯SFT)│                                        │   RLHF      │
└─────────────┘                                        └─────────────┘
      │                                                        │
      ▼                                                        ▼
  低方差/高偏差                                           高方差/低偏差

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### 3.8 与 f-divergence 的联系

标准的 f-divergence 定义为：
$$D_\phi(P \| Q) = \int Q(x) \cdot \phi\left(\frac{P(x)}{Q(x)}\right) dx$$

其中 $\phi$ 是凸函数，$\phi(1) = 0$。

| $\phi(t)$ | 对应散度 |
|-----------|---------|
| $t \log t$ | KL 散度 $D_{KL}(P \| Q)$ |
| $-\log t$ | 反向 KL $D_{KL}(Q \| P)$ |
| $(\sqrt{t} - 1)^2$ | Hellinger 距离 |
| $\|t - 1\|$ | Total Variation |
| $\frac{4}{1-\alpha^2}(1 - t^{(1+\alpha)/2})$ | Amari α-散度 |

**联系**：我们的 IS reshape 函数 f(w) 与 f-divergence 中的 $\phi$ 有对应关系，但多了奖励加权。

---

## 4. 蒸馏的统一视角

### 4.1 蒸馏作为 f(w) = 1 的变体

**标准蒸馏目标**：
$$\min_\theta D_{KL}(\pi_{\text{teacher}} \| \pi_\theta)$$

这可以看作 f(w) = 1 的情况，其中"奖励"隐式定义为：
$$r(x, y) = \tau \log \pi_{\text{teacher}}(y|x) + \text{const}$$

**关于负奖励的说明**：在蒸馏场景中，r(x, y) 可以取负值（当 $\pi_{\text{teacher}}(y|x)$ 很小时）。这与定理 2.1 中 $r > 0$ 的假设不同。但框架仍然适用：
- Forward KL 的等价性（定理 2.1）依赖于 r 作为权重，而非其符号
- 当 r 可正可负时，线性奖励倾斜分布 $\tilde{\pi} \propto \mu \cdot r$ 可能不是有效的概率分布
- 实际中，蒸馏直接使用交叉熵损失 $-\mathbb{E}_\mu[\log \pi_\theta]$ 加权，避免了这一问题

### 4.2 广义蒸馏

在统一框架下，蒸馏可以推广为：

$$g_{\text{distill}}(\theta) = \mathbb{E}_\mu[f(w) \cdot \log \pi_{\text{teacher}}(y|x) \cdot \nabla_\theta \log \pi_\theta]$$

不同的 f(w) 给出不同风格的蒸馏：
- f(w) = 1：标准蒸馏（Forward KL）
- f(w) = w：反向蒸馏（Reverse KL）
- f(w) = $w^\gamma$：插值蒸馏

### 4.3 统一视角

| 方法 | f(w) | r(x, y) | 散度类型 |
|------|------|---------|---------|
| 纯 SFT | 1 | ≡ 1 | Forward KL to μ |
| 奖励加权 SFT | 1 | reward | Forward KL to $\tilde{\pi}$ |
| 标准蒸馏 | 1 | $\log \pi_t$ | Forward KL to $\pi_t$ |
| RLHF | w | reward model | Reverse KL to p* |
| γ-插值 | $w^\gamma$ | reward | 广义散度 |

---

# 第二部分：具体形式 f(w) = w^γ

## 5. 幂函数形式的选择

### 5.1 为什么选择 f(w) = w^γ？

1. **自然的插值**：
   - γ = 0：f(w) = 1（奖励加权 SFT 端）
   - γ = 1：f(w) = w（RL 端）
   - γ ∈ (0, 1)：平滑插值

2. **数学简洁**：幂函数的导数仍是幂函数
   $$f'(w) = \gamma w^{\gamma - 1}$$

3. **对应已知散度族**：与 Rényi / Amari α-散度直接对应

4. **单参数控制**：γ 一个参数即可调节 SFT-RL 谱系

### 5.2 f(w) = w^γ 的梯度

**定理 5.1**：当 f(w) = w^γ 时，梯度为：

$$g(\theta) = \mathbb{E}_\mu\left[w^\gamma \cdot r \cdot \nabla_\theta \log \pi_\theta\right]$$

**归一化有效权重**：
$$\bar{w}_\gamma(y) = \frac{w(y)^\gamma \cdot r(y)}{\mathbb{E}_\mu[w^\gamma \cdot r]}$$

---

## 6. 与 Amari α-散度的对应

**重要说明**：本节使用**指数形式**的目标分布 $p_\beta = \mu e^{\beta r/\tau}/Z_\beta$ 来建立与 Amari α-散度的理论联系。这与 §2 中的主框架有重要区别：

| 视角 | SFT 端目标 | RL 端目标 | μ 的角色 |
|------|-----------|----------|---------|
| §2 主框架 | $\tilde{\pi} \propto \mu \cdot r$ | $p^* = e^{r/\tau}/Z$（**无 μ**） | 仅采样分布 |
| §6 理论分析 | $p_0 = \mu$ | $p_1 = \mu e^{r/\tau}/Z$（有 μ） | 采样 + 参考 |

**关键区别**：
- §2 的 RL 目标 $p^* = e^{r/\tau}/Z$ **不含 μ**（最大熵 RL），使得 μ 在整个框架中仅作为采样分布
- §6 的 $p_1 = \mu e^{r/\tau}/Z$ **含 μ**（KL 正则化 RL），这是为了建立与 Amari α-散度的精确数学对应

两种视角在**定性行为上一致**（都是从 mean-seeking 到 mode-seeking 的谱系）。本节的指数形式能够：
1. 与 Amari α-散度族建立精确对应
2. 利用变分推断（ELBO）的理论框架
3. 提供完整的 Bias-Variance 理论分析框架

**实践建议**：实际应用中推荐使用 §2 的 $p^* = e^{r/\tau}/Z$，避免 μ 的双重角色。

### 6.1 参数映射

**定义 6.1（参数映射）**：设 $\alpha \in [-1, +1]$，定义：

$$\gamma = \frac{1 + \alpha}{2} \in [0, 1]$$

等价地：$\alpha = 2\gamma - 1$。

| γ | α = 2γ - 1 | f(w) = w^γ | 对应方法 |
|---|------------|------------|---------|
| 0 | -1 | 1 | 奖励加权 SFT |
| 0.25 | -0.5 | $w^{0.25}$ | 保守插值 |
| 0.5 | 0 | $\sqrt{w}$ | Hellinger |
| 0.75 | 0.5 | $w^{0.75}$ | 激进插值 |
| 1 | +1 | w | RL |

### 6.2 Amari α-散度

**定义 6.2（Amari α-散度）**：对于 α ∈ (-1, +1)：

$$D_\alpha^{(A)}(P \| Q) = \frac{4}{1-\alpha^2}\left(1 - \int P(x)^{\frac{1+\alpha}{2}} Q(x)^{\frac{1-\alpha}{2}} dx\right)$$

**关键性质**：

$$\lim_{\alpha \to +1} D_\alpha^{(A)}(P \| Q) = D_{KL}(P \| Q) \quad \text{（Reverse KL）}$$

$$\lim_{\alpha \to -1} D_\alpha^{(A)}(P \| Q) = D_{KL}(Q \| P) \quad \text{（Forward KL）}$$

**证明**（α → +1 的情况）：

令 $\alpha = 1 - 2\epsilon$，当 $\epsilon \to 0^+$：

$$D_\alpha^{(A)} = \frac{4}{4\epsilon - 4\epsilon^2}\left(1 - \int P^{1-\epsilon} Q^{\epsilon} dx\right)$$

$$= \frac{1}{\epsilon(1-\epsilon)}\left(1 - \int P \cdot \left(\frac{Q}{P}\right)^{\epsilon} dx\right)$$

Taylor 展开 $\left(\frac{Q}{P}\right)^{\epsilon} \approx 1 + \epsilon \log\frac{Q}{P} + O(\epsilon^2)$：

$$= \frac{1}{\epsilon}\left(1 - 1 - \epsilon \int P \log\frac{Q}{P} dx + O(\epsilon^2)\right)$$

$$= -\int P \log\frac{Q}{P} dx = \int P \log\frac{P}{Q} dx = D_{KL}(P \| Q) \quad \blacksquare$$

### 6.3 耦合参数化

**核心创新**：不仅 f(w) 随 γ 变化，目标分布也应随 γ 变化。

**定义 6.3（β-倾斜分布）**：

$$p_\beta(y|x) = \frac{\mu(y|x) \cdot e^{\beta \cdot r(x,y)/\tau}}{Z_\beta}$$

其中 $Z_\beta = \int \mu(y|x) e^{\beta r/\tau} dy$。

**性质**：
- β = 0：$p_0 = \mu$
- β = 1：$p_1 = \mu e^{r/\tau}/Z$（§6 的 RL 目标，含 μ；区别于 §2 的 $p^* = e^{r/\tau}/Z$）

**耦合设置**：令 $\beta = \gamma = (1+\alpha)/2$。

### 6.4 统一目标函数

**定义 6.4（统一 γ-目标）**：

$$\boxed{\mathcal{L}_\gamma(\theta) = -D_{2\gamma-1}^{(A)}\left(\pi_\theta \| p_\gamma\right), \quad \gamma \in [0, 1]}$$

**定理 6.5（极限恢复）**：

1. **γ = 0（α = -1）**：
$$\mathcal{L}_0(\theta) = -D_{KL}(\mu \| \pi_\theta) = \mathbb{E}_\mu[\log \pi_\theta] + H(\mu)$$
这是**纯 SFT**（Forward KL to μ），因为在指数形式下 $p_0 = \mu$。

2. **γ = 1（α = +1）**：
$$\mathcal{L}_1(\theta) = -D_{KL}(\pi_\theta \| p_1) = \frac{1}{\tau}\mathbb{E}_{\pi_\theta}[r] - D_{KL}(\pi_\theta \| \mu) - \log Z$$
这是 **KL 正则化 RL**（Reverse KL to $p_1 = \mu e^{r/\tau}/Z$）。

**备注**：
- 在 §2 主框架中，RL 目标是 $p^* = e^{r/\tau}/Z$（最大熵 RL，**无 μ**）
- 在 §6 理论分析中，RL 目标是 $p_1 = \mu e^{r/\tau}/Z$（KL 正则化 RL，**有 μ**）
- 两者的定性行为一致（mode-seeking），但 §2 的定义更简洁，μ 仅作为采样分布
- §6 的公式中出现 $D_{KL}(\pi_\theta \| \mu)$ 项，体现了 μ 作为参考策略的角色

**证明**：

对于 γ = 0（α = -1）：
- 目标分布 $p_0 = \mu$
- $D_{-1}^{(A)}(\pi_\theta \| \mu) = D_{KL}(\mu \| \pi_\theta)$（Forward KL）
- 因此 $\mathcal{L}_0 = -D_{KL}(\mu \| \pi_\theta) = \mathbb{E}_\mu[\log \pi_\theta] + H(\mu)$

对于 γ = 1（α = +1）：
- 目标分布 $p_1 = \mu e^{r/\tau}/Z$（注：§6 使用的 KL 正则化形式）
- $D_{+1}^{(A)}(\pi_\theta \| p_1) = D_{KL}(\pi_\theta \| p_1)$（Reverse KL）
- 展开得 $\mathcal{L}_1 = -D_{KL}(\pi_\theta \| p_1) = \frac{1}{\tau}\mathbb{E}_{\pi_\theta}[r] - D_{KL}(\pi_\theta \| \mu) - \log Z$ $\blacksquare$

---

## 7. ELBO 视角

### 7.1 配分函数与 ELBO

**注**：本节沿用 §6 的 KL 正则化形式，定义 $p_1(y|x) = \mu(y|x) e^{r/\tau} / Z$，其中 $Z = \mathbb{E}_\mu[e^{r/\tau}]$。

**标准 ELBO**（γ = 1）：
$$\log Z \geq \mathcal{L}_1(\pi_\theta) = \frac{1}{\tau}\mathbb{E}_{\pi_\theta}[r] - D_{KL}(\pi_\theta \| \mu)$$

当 $\pi_\theta = p_1$ 时取等号。

### 7.2 α-ELBO

**定义 7.1（α-ELBO）**：对于 α ∈ (0, 1)（即 γ ∈ (0.5, 1)）：

$$\mathcal{L}_\alpha^{\text{ELBO}}(\pi_\theta) = \frac{1}{1-\alpha} \log \mathbb{E}_\mu\left[w^\alpha \cdot e^{(1-\alpha)r/\tau}\right]$$

**定理 7.2（ELBO 界）**：
$$\mathcal{L}_\alpha^{\text{ELBO}}(\pi_\theta) \leq \log Z$$

**证明**：

回顾 $Z = \mathbb{E}_\mu[e^{r/\tau}]$。我们需要证明：
$$\frac{1}{1-\alpha} \log \mathbb{E}_\mu\left[w^\alpha \cdot e^{(1-\alpha)r/\tau}\right] \leq \log \mathbb{E}_\mu[e^{r/\tau}]$$

**方法 1（Hölder 不等式）**：

设 $p = 1/\alpha$ 和 $q = 1/(1-\alpha)$，满足 $1/p + 1/q = 1$（共轭指数）。

对于任意非负函数 f, g，Hölder 不等式给出：
$$\mathbb{E}[f \cdot g] \leq \mathbb{E}[f^p]^{1/p} \cdot \mathbb{E}[g^q]^{1/q}$$

令 $f = w^\alpha$ 和 $g = e^{(1-\alpha)r/\tau}$，则：
- $f^p = f^{1/\alpha} = w$
- $g^q = g^{1/(1-\alpha)} = e^{r/\tau}$

因此：
$$\mathbb{E}_\mu[w^\alpha \cdot e^{(1-\alpha)r/\tau}] \leq \mathbb{E}_\mu[w]^\alpha \cdot \mathbb{E}_\mu[e^{r/\tau}]^{1-\alpha}$$

由于 $\mathbb{E}_\mu[w] = \mathbb{E}_\mu[\pi_\theta/\mu] = \int \pi_\theta(y) dy = 1$：
$$\mathbb{E}_\mu[w^\alpha \cdot e^{(1-\alpha)r/\tau}] \leq 1 \cdot Z^{1-\alpha} = Z^{1-\alpha}$$

取对数并乘以 $\frac{1}{1-\alpha} > 0$：
$$\mathcal{L}_\alpha^{\text{ELBO}} = \frac{1}{1-\alpha} \log \mathbb{E}_\mu[w^\alpha \cdot e^{(1-\alpha)r/\tau}] \leq \frac{1}{1-\alpha} \log Z^{1-\alpha} = \log Z \quad \blacksquare$$

**方法 2（变分视角）**：

$\mathcal{L}_\alpha^{\text{ELBO}}$ 可以写成 Rényi 散度的形式：
$$\mathcal{L}_\alpha^{\text{ELBO}} = \log Z - D_\alpha(\pi_\theta \| p_1)$$

其中 $D_\alpha$ 是 α 阶 Rényi 散度，$D_\alpha \geq 0$，因此 $\mathcal{L}_\alpha^{\text{ELBO}} \leq \log Z$。$\blacksquare$

**Gap 分析**：
$$\log Z - \mathcal{L}_\alpha^{\text{ELBO}} = D_\alpha(\pi_\theta \| p_1)$$

其中 $D_\alpha$ 是 α 阶 Rényi 散度。

### 7.3 ELBO 与 f(w) 的联系

| γ | α = 2γ-1 | ELBO 紧度 | 方差 |
|---|----------|----------|------|
| → 0 | → -1 | 不适用（Forward KL） | 最低 |
| 0.5 | 0 | 中等 | 中等 |
| → 1 | → +1 | 最紧（标准 ELBO） | 最高 |

---

## 8. Bias-Variance 分析

**注**：本节沿用 §6 的 KL 正则化形式，目标分布为 $p_1 = \mu e^{r/\tau}/Z$。Bias 定义为优化 $p_\gamma$ 而非 $p_1$ 导致的偏差。

### 8.1 MSE 分解

**定理 8.1（MSE 分解）**：梯度估计器的均方误差为：

$$\text{MSE}(\gamma) = \underbrace{\text{Bias}(\gamma)^2}_{\text{目标偏差的平方}} + \underbrace{\frac{1}{n} \cdot \text{Var}(\gamma)}_{\text{采样方差}}$$

其中：
- Bias(γ) = 优化 $p_\gamma$ 而非 $p_1$ 导致的偏差
- Var(γ) = 梯度估计器的方差

### 8.2 Bias 分析

**来源**：优化 $p_\gamma$ 而非 $p_1$ 导致的偏差。

**术语说明**：此处 "Bias" 指的是**目标偏移**（target mismatch）——当我们优化中间目标 $p_\gamma$ 而非真正目标 $p_1$ 时产生的次优性。这与统计学中估计量的偏差（estimator bias）概念不同。我们用 KL 散度来度量这种目标偏移的程度。

**定理 8.2（Bias 的精确形式）**：

$$\text{Bias}(\gamma) = D_{KL}(p_1 \| p_\gamma)$$

**展开**：

$$D_{KL}(p_1 \| p_\gamma) = \mathbb{E}_{p_1}\left[\log \frac{p_1}{p_\gamma}\right]$$

$$= \mathbb{E}_{p_1}\left[\log \frac{\mu e^{r/\tau}/Z_1}{\mu e^{\gamma r/\tau}/Z_\gamma}\right]$$

$$= \mathbb{E}_{p_1}\left[(1-\gamma)\frac{r}{\tau}\right] + \log Z_\gamma - \log Z_1$$

$$= \frac{1-\gamma}{\tau}\mathbb{E}_{p_1}[r] + \log Z_\gamma - \log Z_1$$

**定理 8.3（Bias 单调递减）**：

$$\frac{\partial \text{Bias}}{\partial \gamma} < 0$$

**证明**：

对 γ 求导：
$$\frac{\partial}{\partial \gamma} D_{KL}(p_1 \| p_\gamma) = -\frac{1}{\tau}\mathbb{E}_{p_1}[r] + \frac{\partial \log Z_\gamma}{\partial \gamma}$$

由于 $Z_\gamma = \mathbb{E}_\mu[e^{\gamma r/\tau}]$：
$$\frac{\partial \log Z_\gamma}{\partial \gamma} = \frac{\mathbb{E}_\mu[e^{\gamma r/\tau} \cdot r/\tau]}{Z_\gamma} = \frac{1}{\tau}\mathbb{E}_{p_\gamma}[r]$$

因此：
$$\frac{\partial \text{Bias}}{\partial \gamma} = \frac{1}{\tau}\left(\mathbb{E}_{p_\gamma}[r] - \mathbb{E}_{p_1}[r]\right)$$

由于 $p_1$ 比 $p_\gamma$（当 γ < 1）更偏向高奖励样本，有 $\mathbb{E}_{p_1}[r] > \mathbb{E}_{p_\gamma}[r]$。

因此 $\frac{\partial \text{Bias}}{\partial \gamma} < 0$。$\blacksquare$

**性质**：
- γ = 1：Bias = 0（无偏，因为 $p_1 = p_1$）
- γ = 0：Bias 最大（$p_0 = \mu$，与 $p_1$ 差距最大）

### 8.3 Variance 分析

**来源**：重要性权重 $w^\gamma$ 的变异性。

**定理 8.4（Variance 的形式）**：

$$\text{Var}(\gamma) \propto \mathbb{E}_\mu\left[w^{2\gamma} \cdot r^2 \cdot \|\nabla \log \pi_\theta\|^2\right]$$

主导项是 $\mathbb{E}_\mu[w^{2\gamma}]$。

**在 Log-Normal 假设下**：

假设 $\log w \sim \mathcal{N}(-\sigma^2/2, \sigma^2)$（保证 $\mathbb{E}[w] = 1$）。

则：
$$\mathbb{E}_\mu[w^{2\gamma}] = e^{2\gamma(-\sigma^2/2) + (2\gamma)^2 \sigma^2/2} = e^{\sigma^2 \cdot 2\gamma(2\gamma-1)/2} = e^{\sigma^2 \gamma(2\gamma-1)}$$

**定理 8.5（Variance 的单调性）**：

令 $V(\gamma) = e^{\sigma^2 \gamma(2\gamma-1)}$，则：
- 当 $\gamma < 0.25$ 时：$\frac{\partial V}{\partial \gamma} < 0$（Variance **递减**）
- 当 $\gamma = 0.25$ 时：$\frac{\partial V}{\partial \gamma} = 0$（Variance **达到最小值**）
- 当 $\gamma > 0.25$ 时：$\frac{\partial V}{\partial \gamma} > 0$（Variance **递增**）

**证明**：

$$\frac{d \log V}{d\gamma} = \sigma^2 (4\gamma - 1)$$

- $4\gamma - 1 < 0$ 当 $\gamma < 0.25$
- $4\gamma - 1 = 0$ 当 $\gamma = 0.25$
- $4\gamma - 1 > 0$ 当 $\gamma > 0.25$ $\blacksquare$

**具体数值**（在 Log-Normal 假设下）：

| γ | $V(\gamma) = e^{\sigma^2 \gamma(2\gamma-1)}$ | 相对于 V(0) |
|---|---------------------------------------------|-------------|
| 0 | $e^0 = 1$ | 基准 |
| 0.25 | $e^{-\sigma^2/8}$ | **最小值** |
| 0.5 | $e^0 = 1$ | 回到基准 |
| 1 | $e^{\sigma^2}$ | 最大 |

**物理解释**：

这一非直觉的结果（γ 增加时 Variance 先减后增）可以这样理解：
- 当 γ 很小时，$w^\gamma \approx 1 + \gamma \log w$，权重的变化主要来自 log w 的符号变化
- 当 γ = 0.25 时，正负贡献达到某种平衡，使得 $\mathbb{E}[w^{2\gamma}]$ 最小
- 当 γ > 0.25 时，高 w 样本的权重开始主导，方差单调增加

**实践意义**：由于 γ = 0.25 对应方差最小，而非 γ = 0，这表明**轻微的 IS 校正反而能降低方差**。但由于 γ < 0.25 区间很窄，实践中这一效应通常不显著。

**备注 8.6（非参数理论扩展）**：

上述结果依赖 Log-Normal 假设。对于一般分布（包括 heavy-tailed 情况），有以下**非参数结果**：

| 结论 | 假设 | 公式 |
|------|------|------|
| ESS 公式 | 仅需 $\mathbb{E}[w]=1$ | $\text{ESS} = n/(1+\chi^2(\pi\|\mu))$ |
| 权重尾部界 | 有限方差 V | $P(w>k) \leq (1+V)/k^2$（Chebyshev 型） |
| 截断比例 | 有限方差 V | 截断到 $w \leq M$ 只影响 $\leq (1+V)/M^2$ 样本 |

**Heavy-tailed 处理**：当怀疑 w 分布是 heavy-tailed 时：
1. 选择较小的 γ（如 0.2-0.3），利用 $w^\gamma$ 的"软截断"效应
2. 使用显式截断：$w_{\text{clip}} = \min(w, M)$
3. 监控 $w_{\max}/w_{\text{median}}$ 比值，超过 100 时需警惕

详细非参数理论见 `theory_refinements.md` 第 3 节。

### 8.4 Trade-off 可视化

```
MSE
 ↑
 │    ╲                      Variance 主导
 │     ╲                    /
 │      ╲                  /
 │       ╲    最优点      /
 │        ╲    ●        /
 │         ╲__________/
 │         /          ╲
 │        /            ╲
 │       / Bias 主导
 │      /
 └──────────────────────────────→ γ
   0              0.5              1
  (SFT)      (Hellinger)         (RL)
```

---

## 9. 单调性与有界性保证

### 9.1 单调性定理汇总

**定理 9.1（Bias 单调递减）**：
$$\frac{\partial \text{Bias}}{\partial \gamma} < 0$$
（证明见定理 8.3）

**定理 9.2（Variance 非单调性）**：
- 当 $\gamma < 0.25$ 时：$\frac{\partial \text{Var}}{\partial \gamma} < 0$（递减）
- 当 $\gamma = 0.25$ 时：$\frac{\partial \text{Var}}{\partial \gamma} = 0$（最小值）
- 当 $\gamma > 0.25$ 时：$\frac{\partial \text{Var}}{\partial \gamma} > 0$（递增）

（证明见定理 8.5）

**定理 9.3（ESS 单调递减）**：

定义有效样本量：
$$\text{ESS}_\gamma = \frac{(\sum_i w_i^\gamma)^2}{\sum_i w_i^{2\gamma}}$$

则：
$$\frac{\partial \text{ESS}_\gamma}{\partial \gamma} < 0$$

**证明**：

ESS 与权重的集中度成反比。当 γ 增大时：
1. 对于 $w_i > 1$ 的样本，$w_i^\gamma$ 增长更快
2. 权重分布变得更加集中（少数高 w 样本主导）
3. $\sum_i w_i^{2\gamma} / (\sum_i w_i^\gamma)^2$ 增大
4. ESS 减小

形式化地，在 Log-Normal 假设下：
$$\text{ESS}_\gamma / n \approx e^{-\sigma^2 \gamma^2}$$

这关于 γ 严格递减。$\blacksquare$

### 9.2 有界性保证

| 量 | 有界性 | 界 |
|---|-------|-----|
| γ | 定义域有界 | [0, 1] |
| $\bar{w}_{\gamma,i}$ | 归一化保证 | [0, 1]，且 $\sum_i \bar{w}_i = 1$ |
| $\|\hat{g}_\gamma\|$ | 梯度有界 | ≤ G（若 score 有界） |
| Var($\hat{g}$) | ESS 控制 | ≤ $4G^2/\text{ESS}_\gamma$ |

### 9.3 存在性定理

**定理 9.4（最优 γ 存在性）**：

定义 $\text{MSE}(\gamma) = \text{Bias}^2(\gamma) + \lambda \cdot \text{Var}(\gamma)$，其中 $\lambda > 0$ 是权衡参数。

则存在最优 $\gamma^* \in [0, 1]$ 使得 MSE 最小。

**证明**：

MSE(γ) 是 [0, 1] 上的连续函数（作为连续函数的复合）。由闭区间上连续函数的**极值定理**（Weierstrass），MSE 在 [0, 1] 上必达到最小值。

**最优点的位置分析**：

1. **边界行为**：
   - MSE(0) = Bias²(0) + λ·Var(0) = $B_0^2$ + λ·$V_0$（高 Bias，低 Var）
   - MSE(1) = 0 + λ·Var(1)（零 Bias，高 Var）

2. **内点分析**（γ > 0.25 时）：
   - Bias²(γ) 严格单调递减（由定理 8.3，因为 Bias > 0 且 ∂Bias/∂γ < 0）
   - Var(γ) 严格单调递增（由定理 8.5）

3. **单峰性**：当 γ > 0.25 时，MSE 的导数为：
   $$\frac{d\text{MSE}}{d\gamma} = 2\text{Bias} \cdot \frac{\partial \text{Bias}}{\partial \gamma} + \lambda \frac{\partial \text{Var}}{\partial \gamma}$$

   第一项 < 0（负），第二项 > 0（正）。在 γ 较小时第一项主导（MSE 递减），在 γ 较大时第二项主导（MSE 递增）。因此 MSE 是**单峰函数**（unimodal），最小值点唯一。

**备注**：当 γ ∈ (0, 0.25) 时，Var(γ) 实际上是递减的（见 §8.3），这使得 MSE 在此区间可能单调递减。综合考虑，最优 γ* 通常位于 (0, 1) 的内部。$\blacksquare$

---

## 10. 最优 γ 的选择

### 10.1 闭式解（Log-Normal 假设）

**定理 10.1**：在 $\log w \sim \mathcal{N}(-\sigma^2/2, \sigma^2)$ 假设下，使用**一阶近似**，MSE 最优的 γ 为：

$$\boxed{\gamma^* = \max\left(0, 1 - \frac{\sigma^2}{2\delta}\right)}$$

其中：
- $\sigma^2 = \text{Var}(\log w)$：分布偏移的度量
- $\delta = B_0^2 / (\lambda V_0) > 0$：bias-variance 权衡参数

**适用条件**：
- 此公式基于 Bias 的线性近似和 Variance 的一阶 Taylor 展开
- 当 $\sigma^2$ 较大（分布偏移严重）时，公式给出 γ* ≈ 0，此时近似有效
- 当 $\sigma^2$ 较小（$\pi_\theta \approx \mu$）时，公式给出 γ* ≈ 1，但此时 "小 γ 近似" 假设失效
- 对于 γ* 接近 1 的情况，建议使用 §10.2 的 ESS 自适应方法

**证明**：

在 Log-Normal 假设下：

**Bias 项**（一阶近似）：
$$\text{Bias}(\gamma) \approx (1-\gamma) \cdot B_0$$
其中 $B_0 = \text{Bias}(0) = D_{KL}(p_1 \| \mu)$ 是 γ = 0 时的最大偏差。这是 Bias(γ) 在 γ = 0 处的一阶 Taylor 展开。

**Variance 项**（一阶近似）：
$$\text{Var}(\gamma) \propto \mathbb{E}_\mu[w^{2\gamma}] = e^{\sigma^2 \gamma(2\gamma-1)}$$

对于小 γ（$\gamma \ll 1$），有 $2\gamma - 1 \approx -1$，因此：
$$\text{Var}(\gamma) \approx V_0 \cdot e^{-\sigma^2 \gamma}$$

进一步，当 $\sigma^2 \gamma \ll 1$ 时，$e^{-\sigma^2 \gamma} \approx 1 - \sigma^2 \gamma$。为简化分析，我们使用 $e^{\sigma^2 \gamma}$ 作为近似（保持指数形式但改变符号，这在小 γ 时引入的误差可控）。

**MSE 最小化**：
$$\text{MSE}(\gamma) \approx (1-\gamma)^2 B_0^2 + \lambda V_0 e^{\sigma^2 \gamma}$$

对 γ 求导并令其为零：
$$\frac{d\text{MSE}}{d\gamma} = -2(1-\gamma) B_0^2 + \lambda V_0 \sigma^2 e^{\sigma^2 \gamma} = 0$$

在 $\sigma^2 \gamma \ll 1$ 的近似下（$e^{\sigma^2 \gamma} \approx 1$）：
$$-2(1-\gamma) B_0^2 + \lambda V_0 \sigma^2 = 0$$

$$1 - \gamma = \frac{\lambda V_0 \sigma^2}{2 B_0^2}$$

$$\gamma = 1 - \frac{\lambda V_0 \sigma^2}{2 B_0^2}$$

定义 $\delta = B_0^2 / (\lambda V_0)$（bias 相对于 variance 的重要性），则：
$$\gamma^* = 1 - \frac{\sigma^2}{2\delta}$$

由于 γ 需要满足 $\gamma \in [0, 1]$：
$$\gamma^* = \max\left(0, 1 - \frac{\sigma^2}{2\delta}\right) \quad \blacksquare$$

**近似有效性分析**：

| 场景 | σ² 大小 | γ* 值 | 近似有效性 |
|------|--------|-------|-----------|
| π_θ 远离 μ | 大（≥ 2δ） | = 0 | ✓ 有效（小 γ） |
| 中等偏移 | ≈ δ | ≈ 0.5 | △ 部分有效 |
| π_θ ≈ μ | 小（→ 0） | → 1 | ✗ 近似失效 |

**实践建议**：当 σ² < δ 时（即公式给出 γ* > 0.5），建议使用 ESS 自适应方法（§10.2）或数值求解，而非依赖此闭式近似。

**性质**：

| 情况 | σ² | γ* | 行为 |
|-----|----|----|-----|
| π_θ ≈ μ | 小 → 0 | → 1 | 纯 RL |
| π_θ 远离 μ | σ² ≥ 2δ | = 0 | 纯 SFT |
| σ² = δ | 中等 | = 0.5 | Hellinger |

**备注**：当 σ² < 2δ 时，$\gamma^* \in (0, 1)$，实现 SFT-RL 插值。当 σ² ≥ 2δ 时，方差过大，应退化到纯 SFT（γ = 0）以保证稳定性。

### 10.2 ESS 自适应方法

**算法**：选择满足 ESS 约束的最大 γ：

$$\gamma^* = \max\left\{\gamma \in [0, 1] : \text{ESS}_\gamma \geq n \cdot \rho_{\min}\right\}$$

由 ESS 关于 γ 的单调性（定理 9.3），可用二分搜索高效求解。

**算法复杂度**：$O(\log(1/\epsilon))$ 次 ESS 计算。

### 10.3 自一致性问题与校准

**注**：本节使用 §6 的 KL 正则化形式 $p_1 = \mu e^{r/\tau}/Z$。

**问题**：原始 σ² = Var(log w) 在 π_θ → p_1 时不趋于零。

因为当 $\pi_\theta = p_1 = \mu e^{r/\tau}/Z$ 时：
$$\log w = \log \frac{p_1}{\mu} = \frac{r}{\tau} - \log Z$$
$$\text{Var}(\log w) = \text{Var}(r)/\tau^2 \neq 0$$

**解决方案**：使用校准权重：

$$\tilde{w} = \frac{w}{w^*} = \frac{w \cdot \bar{Z}}{e^{r/\tau}}$$

其中 $w^* = e^{r/\tau}/\bar{Z}$ 是"理想"权重，$\bar{Z} = \frac{1}{n}\sum_i e^{r_i/\tau}$。

**性质**：当 π_θ → p_1 时：
$$\tilde{w} = \frac{e^{r/\tau}/Z}{e^{r/\tau}/\bar{Z}} = \frac{\bar{Z}}{Z} \to 1$$

因此 Var(log $\tilde{w}$) → 0，由闭式解 γ → 1。✓

### 10.4 三层优雅设计

**最终的 f(w) 设计**：

$$\boxed{f_{\text{elegant}}(w; r, \gamma) = \text{Normalize}\left[\left(\frac{w \cdot \bar{Z}}{e^{r/\tau}}\right)^\gamma\right]}$$

其中 γ 由 ESS 约束确定。

**三层结构**：
1. **校准层**：$w \to \tilde{w} = w/w^*$（消除与 p* 的偏移）
2. **幂变换层**：$\tilde{w} \to \tilde{w}^\gamma$（由 ESS 单调性确定 γ）
3. **归一化层**：保证权重和为 1

---

## 11. 实现

### 11.1 PyTorch 实现

```python
import torch
import torch.nn.functional as F
import numpy as np

def compute_is_reshape_weights(
    log_pi: torch.Tensor,      # log π_θ(y|x)
    log_mu: torch.Tensor,      # log μ(y|x)
    rewards: torch.Tensor,     # r(x, y)
    tau: float,                # 温度
    gamma: float,              # IS reshape 参数 ∈ [0, 1]
    use_calibration: bool = True
) -> torch.Tensor:
    """
    计算 IS reshape 权重 f(w) = w^γ（可选校准）

    返回：归一化的权重
    """
    # 计算 log 重要性比率
    log_w = log_pi - log_mu

    if use_calibration:
        # 校准：log(w/w*) = log(w) - log(w*)
        # w* = e^{r/τ} / Z̄
        log_w_star = rewards / tau
        log_Z_bar = torch.logsumexp(log_w_star, dim=0) - np.log(len(rewards))
        log_w_star = log_w_star - log_Z_bar

        # 校准后的 log 权重
        log_w_calibrated = log_w - log_w_star
    else:
        log_w_calibrated = log_w

    # 幂变换：w^γ → γ * log(w)
    log_weights = gamma * log_w_calibrated

    # 归一化（数值稳定的 softmax）
    weights = F.softmax(log_weights, dim=0)

    return weights


def adaptive_gamma_selection(
    log_w: torch.Tensor,
    rho_min: float = 0.3,
    max_iter: int = 50
) -> float:
    """
    基于 ESS 约束自适应选择 γ

    利用 ESS(γ) 关于 γ 单调递减的性质进行二分搜索
    """
    n = len(log_w)

    def compute_ess_ratio(gamma):
        if gamma == 0:
            return 1.0
        log_weights = gamma * log_w
        weights = F.softmax(log_weights, dim=0)
        ess = 1.0 / torch.sum(weights ** 2)
        return (ess / n).item()

    # 二分搜索：找满足 ESS 约束的最大 γ
    gamma_low, gamma_high = 0.0, 1.0

    for _ in range(max_iter):
        gamma_mid = (gamma_low + gamma_high) / 2
        ess_ratio = compute_ess_ratio(gamma_mid)

        if ess_ratio >= rho_min:
            gamma_low = gamma_mid  # ESS 足够，尝试更大的 γ
        else:
            gamma_high = gamma_mid  # ESS 不足，需要更小的 γ

        if gamma_high - gamma_low < 1e-6:
            break

    return gamma_low


def is_reshape_loss(
    log_pi: torch.Tensor,
    log_mu: torch.Tensor,
    rewards: torch.Tensor,
    tau: float = 1.0,
    gamma: float = None,
    rho_min: float = 0.3
) -> tuple:
    """
    IS Reshape 损失函数

    参数：
        log_pi: 当前策略的 log 概率
        log_mu: 行为策略的 log 概率
        rewards: 奖励
        tau: 温度
        gamma: IS reshape 参数（None 则自适应选择）
        rho_min: 最小 ESS 比例

    返回：
        loss: 需要最小化的损失
        info: 调试信息
    """
    log_w = log_pi - log_mu

    # 自适应选择 γ
    if gamma is None:
        gamma = adaptive_gamma_selection(log_w.detach(), rho_min)

    # 计算权重
    weights = compute_is_reshape_weights(
        log_pi, log_mu, rewards, tau, gamma, use_calibration=True
    )

    # 损失：负的加权 log 似然
    loss = -torch.sum(weights.detach() * log_pi)

    # 计算 ESS
    ess = 1.0 / torch.sum(weights ** 2)

    info = {
        'gamma': gamma,
        'alpha': 2 * gamma - 1,
        'ess': ess.item(),
        'ess_ratio': ess.item() / len(log_pi),
        'max_weight': weights.max().item(),
        'sigma_sq': torch.var(log_w).item()
    }

    return loss, info
```

### 11.2 训练循环示例

```python
def train_with_is_reshape(
    policy,
    behavior_policy,
    dataloader,
    optimizer,
    tau: float = 1.0,
    rho_min: float = 0.3,
    num_epochs: int = 100
):
    """使用 IS Reshape 的离线 RL 训练"""

    for epoch in range(num_epochs):
        total_loss = 0
        total_gamma = 0
        num_batches = 0

        for batch in dataloader:
            x, y, r = batch['context'], batch['response'], batch['reward']

            # 计算 log 概率
            with torch.no_grad():
                log_mu = behavior_policy.log_prob(y, x)
            log_pi = policy.log_prob(y, x)

            # 计算 IS reshape 损失
            loss, info = is_reshape_loss(
                log_pi, log_mu, r, tau=tau, gamma=None, rho_min=rho_min
            )

            # 优化
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            total_gamma += info['gamma']
            num_batches += 1

        avg_loss = total_loss / num_batches
        avg_gamma = total_gamma / num_batches

        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, "
              f"γ={avg_gamma:.3f}, α={2*avg_gamma-1:.3f}")
```

---

## 12. 总结

### 12.1 主要贡献

1. **从 IS Reshape 出发的统一视角**：
   - f(w) = 1 → Forward KL → **奖励加权 SFT**
   - f(w) = w → Reverse KL → **RL**（最大熵 RL）
   - 一般 f(w) → 广义散度 → **插值**

2. **SFT 的精确定位**：
   - 奖励加权 SFT = f(w) = 1
   - 纯 SFT = f(w) = 1 且 r ≡ const

3. **μ 的统一角色**：
   - μ 在整个框架中**仅作为采样分布**
   - RL 目标 $p^* = e^{r/\tau}/Z$ **不含 μ**，纯由奖励决定
   - 避免了 μ 同时作为"采样分布"和"参考策略"的概念混淆

4. **f(w) = w^γ 的理论分析**：
   - 与 Amari α-散度的精确对应（§6-§8 使用 KL 正则化形式）
   - 耦合参数化
   - 完整的 Bias-Variance 分析

5. **理论保证**：
   - 单调性：Bias↓, Var↑, ESS↓ as γ↑（完整证明）
   - 有界性：所有关键量有界
   - 存在性：最优 γ 唯一存在

6. **实用的 γ 选择方法**：
   - 闭式解（带完整推导）
   - ESS 自适应
   - 三层校准设计

### 12.2 核心公式

**统一梯度**：
$$g(\theta) = \mathbb{E}_\mu[f(w) \cdot r \cdot \nabla_\theta \log \pi_\theta], \quad w = \pi_\theta / \mu$$

**两个极端**：
$$f(w) = 1 \Rightarrow \text{奖励加权 SFT（Forward KL）}$$
$$f(w) = w \Rightarrow \text{RL（Reverse KL）}$$

**幂函数形式**：
$$f(w) = w^\gamma, \quad \gamma \in [0, 1]$$

**最优 γ**：
$$\gamma^* = \max\left(0, 1 - \frac{\sigma^2}{2\delta}\right)$$

### 12.3 方法谱系图

```
        γ = 0                      γ = 0.5                    γ = 1
        α = -1                     α = 0                      α = +1
          │                          │                          │
          ▼                          ▼                          ▼
    ┌─────────┐               ┌─────────────┐              ┌─────────┐
    │奖励加权SFT│               │  Hellinger  │              │   RL    │
    │ f(w)=1  │───────────────│   f(w)=√w   │──────────────│ f(w)=w  │
    │ D(p||π) │               │   对称的    │              │ D(π||p*)│
    └─────────┘               └─────────────┘              └─────────┘
          │                          │                          │
          ▼                          ▼                          ▼
    Forward KL                   插值散度                  Reverse KL
    Mean-seeking                  平衡                    Mode-seeking
    覆盖所有模式              平衡覆盖与聚焦              聚焦最优模式
          │                          │                          │
          ▼                          ▼                          ▼
      低方差                      中等                       高方差
      高偏差                      中等                       低偏差
          │
          ▼
    ┌─────────┐
    │ 纯 SFT  │ (r ≡ const 的特例)
    │ D(μ||π) │
    └─────────┘
```

---

## 附录 A：符号表

| 符号 | 含义 |
|------|------|
| $w = \pi_\theta/\mu$ | 重要性采样比率 |
| $f(w)$ | IS 整形函数 |
| $\gamma \in [0, 1]$ | 幂函数指数（f(w) = w^γ） |
| $\alpha = 2\gamma - 1$ | Amari α-散度参数 |
| $\beta = \gamma$ | 目标分布倾斜度（§6） |
| $p_\beta = \mu e^{\beta r/\tau}/Z_\beta$ | β-倾斜目标分布（§6，含 μ） |
| $p^* = e^{r/\tau}/Z$ | 最优软策略（§2 主框架，**无 μ**） |
| $p_1 = \mu e^{r/\tau}/Z$ | §6 的 RL 目标（含 μ，用于理论分析） |
| $\tilde{\pi} = \mu \cdot r / \mathbb{E}_\mu[r]$ | 线性奖励倾斜分布（§2） |
| $\sigma^2 = \text{Var}(\log w)$ | 分布偏移度量 |
| $\delta$ | bias-variance 权衡参数 |
| $\text{ESS}_\gamma$ | 有效样本量 |
| $\rho = \text{ESS}/n$ | ESS 比例 |

**μ 的统一含义**：在整个框架中，μ **仅作为采样分布**（数据来源）。§2 的 p* 不含 μ，确保目标策略纯由奖励决定。§6-§8 的 p_1 含 μ 是为了建立与 Amari α-散度的数学对应。

## 附录 B：证明索引

| 定理 | 内容 | 位置 |
|------|------|------|
| 2.1 | Forward KL 等价性 | §2.1 |
| 2.3 | Reverse KL 等价性 | §2.3 |
| 6.2 | Amari α-散度极限 | §6.2 |
| 6.5 | 极限恢复 | §6.4 |
| 8.3 | Bias 单调递减 | §8.2 |
| 8.5 | Variance 单调递增 | §8.3 |
| 9.3 | ESS 单调递减 | §9.1 |
| 9.4 | 最优 γ 存在性 | §9.3 |
| 10.1 | 闭式最优 γ | §10.1 |
