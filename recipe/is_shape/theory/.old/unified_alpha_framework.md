# 统一框架：基于 Amari α-散度的离线强化学习

## 摘要

本文提出一个统一 SFT、RL 和蒸馏的理论框架。核心创新是使用 **Amari α-散度** 并引入**耦合参数化**，使得单一参数 α ∈ [-1, +1] 能够：
- α = -1：精确恢复 SFT（前向 KL）
- α = +1：精确恢复 RL（反向 KL）
- α ∈ (-1, +1)：在两者之间平滑插值

---

## 1. 问题回顾与动机

### 1.1 原有框架的局限

之前的 α-ELBO 框架基于 Rényi 散度：

$$\mathcal{L}_\alpha^{\text{old}} = \frac{1}{1-\alpha} \log \mathbb{E}_\mu\left[w^\alpha e^{(1-\alpha)r/\tau}\right]$$

**问题**：当 α → 0 时，得到的是**奖励加权的 MLE**，而非纯粹的 SFT：

$$\nabla_\theta \mathcal{L}_0 \propto \mathbb{E}_\mu\left[\frac{e^{r/\tau}}{Z} \nabla_\theta \log \pi_\theta\right] \neq \mathbb{E}_\mu[\nabla_\theta \log \pi_\theta]$$

**根本原因**：Rényi 散度只涵盖**反向 KL 方向**，无法表达前向 KL。

### 1.2 统一的需求

我们需要一个框架能够：
1. **精确恢复 SFT**：$\min D_{KL}(\mu \| \pi_\theta)$（前向 KL，mean-seeking）
2. **精确恢复 RL**：$\min D_{KL}(\pi_\theta \| p^*)$（反向 KL，mode-seeking）
3. **包含蒸馏**：作为中间状态
4. **平滑插值**：在上述极端之间连续过渡

---

## 2. Amari α-散度

### 2.1 定义

**定义 2.1（Amari α-散度）**：对于 α ∈ (-1, +1)：

$$D_\alpha^{(A)}(P \| Q) = \frac{4}{1-\alpha^2}\left(1 - \int P(x)^{\frac{1+\alpha}{2}} Q(x)^{\frac{1-\alpha}{2}} dx\right)$$

### 2.2 关键性质

**命题 2.2**：Amari α-散度具有以下极限行为：

1. **α → +1**（反向 KL）：
$$\lim_{\alpha \to 1^-} D_\alpha^{(A)}(P \| Q) = D_{KL}(P \| Q)$$

2. **α → -1**（前向 KL）：
$$\lim_{\alpha \to -1^+} D_\alpha^{(A)}(P \| Q) = D_{KL}(Q \| P)$$

3. **α = 0**（Hellinger 距离）：
$$D_0^{(A)}(P \| Q) = 4\left(1 - \int \sqrt{P \cdot Q} \, dx\right) = 4 H^2(P, Q)$$

**证明**：

**情况 1：α → +1**

令 $\alpha = 1 - 2\epsilon$，当 $\epsilon \to 0^+$：

$$D_\alpha^{(A)} = \frac{4}{1-(1-2\epsilon)^2}\left(1 - \int P^{1-\epsilon} Q^{\epsilon} dx\right)$$

$$= \frac{4}{4\epsilon - 4\epsilon^2}\left(1 - \int P \cdot \left(\frac{Q}{P}\right)^{\epsilon} dx\right)$$

$$= \frac{1}{\epsilon(1-\epsilon)}\left(1 - \int P \cdot e^{\epsilon \log(Q/P)} dx\right)$$

Taylor 展开 $e^{\epsilon \log(Q/P)} \approx 1 + \epsilon \log(Q/P) + O(\epsilon^2)$：

$$= \frac{1}{\epsilon}\left(1 - 1 - \epsilon \int P \log(Q/P) dx + O(\epsilon^2)\right)$$

$$= -\int P \log(Q/P) dx = \int P \log(P/Q) dx = D_{KL}(P \| Q) \quad \blacksquare$$

**情况 2：α → -1**

类似地，令 $\alpha = -1 + 2\epsilon$，可证 $D_\alpha^{(A)} \to D_{KL}(Q \| P)$。$\blacksquare$

### 2.3 与 f-散度的关系

**命题 2.3**：Amari α-散度是 f-散度的特例，对应于：

$$f_\alpha(t) = \frac{4}{1-\alpha^2}\left(1 - t^{\frac{1-\alpha}{2}}\right)$$

---

## 3. 耦合参数化：核心创新

### 3.1 关键洞察

**观察**：SFT 和 RL 的区别不仅在于**散度方向**，还在于**目标分布**：

| 方法 | 目标分布 | 散度方向 |
|------|---------|---------|
| SFT | μ（行为策略） | 前向 KL：D(μ \|\| π) |
| RL | p* = μ·e^{r/τ}/Z（最优策略） | 反向 KL：D(π \|\| p*) |

**核心创新**：定义**参数化的目标分布族**，并与散度参数**耦合**。

### 3.2 参数化目标分布

**定义 3.1（β-倾斜分布）**：

$$p_\beta(y|x) = \frac{\mu(y|x) \exp(\beta \cdot r(x,y)/\tau)}{Z_\beta(x)}$$

其中配分函数：
$$Z_\beta(x) = \int \mu(y|x) \exp(\beta \cdot r(x,y)/\tau) \, dy$$

**性质**：
- β = 0：$p_0 = \mu$（行为策略）
- β = 1：$p_1 = p^*$（最优软策略）
- β ∈ (0, 1)：在两者之间插值

### 3.3 耦合参数化

**定义 3.2（耦合参数）**：设置 **β = (1+α)/2**，使得：

| α | β = (1+α)/2 | 目标分布 |
|---|-------------|---------|
| -1 | 0 | p_0 = μ |
| 0 | 0.5 | p_{0.5}（中间分布） |
| +1 | 1 | p_1 = p* |

### 3.4 统一目标函数

**定义 3.3（统一 α-目标）**：

$$\boxed{\mathcal{L}_\alpha(\theta) = -D_\alpha^{(A)}\left(\pi_\theta \| p_{\frac{1+\alpha}{2}}\right), \quad \alpha \in [-1, +1]}$$

展开形式：

$$\mathcal{L}_\alpha(\theta) = -\frac{4}{1-\alpha^2}\left(1 - \int \pi_\theta(y|x)^{\frac{1+\alpha}{2}} p_{\frac{1+\alpha}{2}}(y|x)^{\frac{1-\alpha}{2}} dy\right)$$

---

## 4. 极限情况验证

### 4.1 α = +1：恢复标准 RL

**定理 4.1**：当 α → +1 时：

$$\mathcal{L}_{+1}(\theta) = -D_{KL}(\pi_\theta \| p^*) = \frac{1}{\tau}\mathbb{E}_{\pi_\theta}[r] - D_{KL}(\pi_\theta \| \mu) - \log Z$$

这正是**标准 RLHF 目标**（mode-seeking）。

**证明**：

由命题 2.2，当 α → +1：
$$D_\alpha^{(A)}(\pi_\theta \| p_\beta) \to D_{KL}(\pi_\theta \| p_\beta)$$

同时 β = (1+α)/2 → 1，所以 $p_\beta \to p_1 = p^*$。

因此：
$$\mathcal{L}_{+1} = -D_{KL}(\pi_\theta \| p^*)$$

展开 KL 散度：
$$D_{KL}(\pi_\theta \| p^*) = \mathbb{E}_{\pi_\theta}\left[\log \frac{\pi_\theta}{p^*}\right]$$

$$= \mathbb{E}_{\pi_\theta}\left[\log \frac{\pi_\theta}{\mu e^{r/\tau} / Z}\right]$$

$$= \mathbb{E}_{\pi_\theta}\left[\log \frac{\pi_\theta}{\mu}\right] - \frac{1}{\tau}\mathbb{E}_{\pi_\theta}[r] + \log Z$$

$$= D_{KL}(\pi_\theta \| \mu) - \frac{1}{\tau}\mathbb{E}_{\pi_\theta}[r] + \log Z$$

因此：
$$\mathcal{L}_{+1} = \frac{1}{\tau}\mathbb{E}_{\pi_\theta}[r] - D_{KL}(\pi_\theta \| \mu) - \log Z \quad \blacksquare$$

### 4.2 α = -1：恢复标准 SFT

**定理 4.2**：当 α → -1 时：

$$\mathcal{L}_{-1}(\theta) = -D_{KL}(\mu \| \pi_\theta) = \mathbb{E}_\mu[\log \pi_\theta] + H(\mu)$$

这正是**标准 SFT 目标**（mean-seeking），等价于最大化似然。

**证明**：

由命题 2.2，当 α → -1：
$$D_\alpha^{(A)}(\pi_\theta \| p_\beta) \to D_{KL}(p_\beta \| \pi_\theta)$$

同时 β = (1+α)/2 → 0，所以 $p_\beta \to p_0 = \mu$。

因此：
$$\mathcal{L}_{-1} = -D_{KL}(\mu \| \pi_\theta)$$

展开：
$$D_{KL}(\mu \| \pi_\theta) = \mathbb{E}_\mu\left[\log \frac{\mu}{\pi_\theta}\right] = -H(\mu) - \mathbb{E}_\mu[\log \pi_\theta]$$

因此：
$$\mathcal{L}_{-1} = \mathbb{E}_\mu[\log \pi_\theta] + H(\mu)$$

由于 $H(\mu)$ 是常数，最大化 $\mathcal{L}_{-1}$ 等价于：
$$\max_\theta \mathbb{E}_\mu[\log \pi_\theta]$$

这正是**标准 SFT（最大似然估计）**。$\blacksquare$

### 4.3 α = 0：Hellinger 距离（平衡点）

**定理 4.3**：当 α = 0 时：

$$\mathcal{L}_0(\theta) = -4H^2(\pi_\theta, p_{0.5}) = -4\left(1 - \int \sqrt{\pi_\theta \cdot p_{0.5}} \, dy\right)$$

其中 $p_{0.5} = \mu \cdot e^{r/(2\tau)} / Z_{0.5}$ 是"半倾斜"分布。

**性质**：
- Hellinger 距离是**对称的**：$H(\pi_\theta, p_{0.5}) = H(p_{0.5}, \pi_\theta)$
- 在 mean-seeking 和 mode-seeking 之间**平衡**
- 对异常值具有**鲁棒性**

---

## 5. 梯度分析

### 5.1 一般形式的梯度

**定理 5.1（统一梯度公式）**：

$$\nabla_\theta \mathcal{L}_\alpha = \frac{2}{1-\alpha} \cdot \frac{\mathbb{E}_{\pi_\theta}\left[\pi_\theta^{\frac{\alpha-1}{2}} p_\beta^{\frac{1-\alpha}{2}} \nabla_\theta \log \pi_\theta\right]}{\int \pi_\theta^{\frac{1+\alpha}{2}} p_\beta^{\frac{1-\alpha}{2}} dy}$$

其中 $\beta = (1+\alpha)/2$。

**证明**：

令 $I(\theta) = \int \pi_\theta^{\frac{1+\alpha}{2}} p_\beta^{\frac{1-\alpha}{2}} dy$，则：

$$\mathcal{L}_\alpha = -\frac{4}{1-\alpha^2}(1 - I(\theta))$$

对 θ 求导：
$$\nabla_\theta \mathcal{L}_\alpha = \frac{4}{1-\alpha^2} \nabla_\theta I(\theta)$$

计算 $\nabla_\theta I$：
$$\nabla_\theta I = \int \frac{1+\alpha}{2} \pi_\theta^{\frac{1+\alpha}{2}-1} p_\beta^{\frac{1-\alpha}{2}} \nabla_\theta \pi_\theta \, dy$$

使用 $\nabla_\theta \pi_\theta = \pi_\theta \nabla_\theta \log \pi_\theta$：

$$= \frac{1+\alpha}{2} \int \pi_\theta^{\frac{1+\alpha}{2}} p_\beta^{\frac{1-\alpha}{2}} \nabla_\theta \log \pi_\theta \, dy$$

$$= \frac{1+\alpha}{2} \mathbb{E}_{\pi_\theta}\left[\pi_\theta^{\frac{\alpha-1}{2}} p_\beta^{\frac{1-\alpha}{2}} \nabla_\theta \log \pi_\theta\right]$$

代入：
$$\nabla_\theta \mathcal{L}_\alpha = \frac{4}{1-\alpha^2} \cdot \frac{1+\alpha}{2} \mathbb{E}_{\pi_\theta}\left[\pi_\theta^{\frac{\alpha-1}{2}} p_\beta^{\frac{1-\alpha}{2}} \nabla_\theta \log \pi_\theta\right]$$

$$= \frac{2(1+\alpha)}{(1-\alpha)(1+\alpha)} \mathbb{E}_{\pi_\theta}\left[\cdot\right] = \frac{2}{1-\alpha} \mathbb{E}_{\pi_\theta}\left[\cdot\right] \quad \blacksquare$$

### 5.2 离线数据形式

**定理 5.2（离线梯度）**：利用重要性采样，梯度可以改写为：

$$\nabla_\theta \mathcal{L}_\alpha = \frac{2}{1-\alpha} \cdot \frac{\mathbb{E}_\mu\left[w^{\frac{1+\alpha}{2}} \cdot e^{\frac{(1-\alpha)\beta r}{2\tau}} \cdot \nabla_\theta \log \pi_\theta\right]}{\mathbb{E}_\mu\left[w^{\frac{1+\alpha}{2}} \cdot e^{\frac{(1-\alpha)\beta r}{2\tau}}\right]}$$

其中 $w = \pi_\theta / \mu$，$\beta = (1+\alpha)/2$。

**简化形式**：

定义**自归一化权重**：
$$\tilde{w}_\alpha(y) = \frac{w(y)^{\frac{1+\alpha}{2}} \cdot \exp\left(\frac{(1-\alpha)(1+\alpha) r}{4\tau}\right)}{\mathbb{E}_\mu\left[w^{\frac{1+\alpha}{2}} \cdot \exp\left(\frac{(1-\alpha)(1+\alpha) r}{4\tau}\right)\right]}$$

则：
$$\nabla_\theta \mathcal{L}_\alpha = \frac{2}{1-\alpha} \mathbb{E}_\mu\left[\tilde{w}_\alpha \nabla_\theta \log \pi_\theta\right]$$

### 5.3 特殊情况验证

**α = +1（RL）**：
- $w$ 的幂次：$(1+1)/2 = 1$
- 奖励权重：$(1-1)(1+1)/(4\tau) = 0$
- 梯度：$\mathbb{E}_\mu[w \nabla_\theta \log \pi_\theta] = \mathbb{E}_{\pi_\theta}[\nabla_\theta \log \pi_\theta]$

这需要配合 RL 的完整目标函数理解。

**α = -1（SFT）**：
- $w$ 的幂次：$(1-1)/2 = 0$
- 奖励权重：$(1-(-1))(1-1)/(4\tau) = 0$
- 梯度：$\mathbb{E}_\mu[1 \cdot \nabla_\theta \log \pi_\theta] = \mathbb{E}_\mu[\nabla_\theta \log \pi_\theta]$

这正是**标准 SFT 梯度**！$\checkmark$

---

## 6. 蒸馏的统一视角

### 6.1 蒸馏作为中间状态

**命题 6.1**：知识蒸馏可以视为 α ∈ (-1, 0) 的特殊情况。

**标准蒸馏目标**：
$$\min_\theta D_{KL}(\pi_t \| \pi_\theta)$$

其中 $\pi_t$ 是教师模型。

**在统一框架中**：
- 如果 $\pi_t = p_\beta$ 对于某个 β ∈ (0, 0.5)
- 则对应于 α = 2β - 1 ∈ (-1, 0)

### 6.2 广义蒸馏

**定义 6.2（广义蒸馏）**：

$$\mathcal{L}_\alpha^{\text{distill}}(\theta) = -D_\alpha^{(A)}(\pi_\theta \| \pi_t)$$

其中 $\pi_t$ 可以是：
- **纯教师**：$\pi_t = \pi_{\text{teacher}}$
- **混合分布**：$\pi_t = (1-\lambda)\mu + \lambda \pi_{\text{teacher}}$
- **倾斜分布**：$\pi_t = p_\beta$

### 6.3 统一视角

| 场景 | 目标分布 $p_\beta$ | α 值 | 方法名称 |
|-----|-------------------|-----|---------|
| 纯 SFT | μ | -1 | Maximum Likelihood |
| 保守蒸馏 | $\mu \cdot e^{r/(4\tau)}/Z$ | -0.5 | Conservative Distillation |
| 平衡点 | $\mu \cdot e^{r/(2\tau)}/Z$ | 0 | Hellinger Matching |
| 激进蒸馏 | $\mu \cdot e^{3r/(4\tau)}/Z$ | 0.5 | Aggressive Distillation |
| 纯 RL | p* = $\mu \cdot e^{r/\tau}/Z$ | +1 | RLHF |

---

## 7. Bias-Variance 分析

### 7.1 Gap 分析

**定理 7.1（统一 Gap）**：对于任意 α ∈ (-1, +1)：

$$\text{Gap}(\alpha) = D_\alpha^{(A)}(\pi_\theta \| p_\beta) - D_\alpha^{(A)}(\pi_\theta^* \| p_\beta)$$

其中 $\pi_\theta^*$ 是 $\mathcal{L}_\alpha$ 的最优解。

**性质**：
- 当 $\pi_\theta = p_\beta$ 时，Gap = 0
- Gap 关于 α 的行为取决于 $\pi_\theta$ 与 $p_\beta$ 的关系

### 7.2 方差分析

**定理 7.2（梯度估计器方差）**：

$$\text{Var}(\hat{\nabla}_\theta \mathcal{L}_\alpha) \propto \frac{1}{n} \cdot \mathbb{E}_\mu\left[w^{1+\alpha} \cdot e^{\frac{(1-\alpha^2)r}{2\tau}}\right]$$

**性质**：
- α → -1：$w^0 = 1$，方差最小（SFT 最稳定）
- α → +1：$w^2$ 项主导，方差可能很大（RL 需要方差控制）

### 7.3 Bias-Variance Trade-off

| α 值 | Bias（目标偏差） | Variance（梯度方差） | 行为模式 |
|-----|-----------------|-------------------|---------|
| α → -1 | 大（只拟合 μ） | 小（无 IS） | 保守/SFT |
| α = 0 | 中等 | 中等 | 平衡/Hellinger |
| α → +1 | 小（追踪 p*） | 大（完整 IS） | 激进/RL |

---

## 8. 最优 α 选择

### 8.1 基于目标的选择

根据任务需求选择 α：

```
α 选择指南：
┌─────────────────────────────────────────────────────┐
│  任务类型          │  推荐 α   │  原因              │
├─────────────────────────────────────────────────────┤
│  纯模仿学习        │  α ≈ -1   │  最大化似然        │
│  保守策略改进      │  α ∈ [-0.5, 0]  │  控制偏移   │
│  平衡探索/利用     │  α = 0    │  对称处理          │
│  激进策略优化      │  α ∈ [0, 0.5]   │  追踪高奖励  │
│  纯强化学习        │  α ≈ +1   │  最大化奖励        │
└─────────────────────────────────────────────────────┘
```

### 8.2 自适应选择

**算法 1：自适应 α 选择**

```python
def adaptive_alpha(log_w, rewards, tau, delta=0.1):
    """
    基于分布偏移程度自适应选择 α

    参数:
        log_w: log(π_θ/μ) 的样本
        rewards: 奖励样本
        tau: 温度参数
        delta: 可接受的散度上界

    返回:
        alpha: 最优 α 值
    """
    # 估计分布偏移
    sigma_sq = np.var(log_w)

    # 估计奖励信号强度
    reward_signal = np.var(rewards) / tau**2

    # 自适应公式
    # 当 σ² 大时，α 应该小（更保守）
    # 当奖励信号强时，α 可以大（更激进）
    alpha = np.tanh(reward_signal / (sigma_sq + 1e-8) - delta)

    return np.clip(alpha, -0.99, 0.99)
```

### 8.3 理论最优 α

**定理 8.1（MSE 最优 α）**：最小化梯度估计器 MSE 的最优 α 满足：

$$\alpha^* = \arg\min_\alpha \left[\text{Bias}^2(\alpha) + \text{Var}(\alpha)\right]$$

**近似闭式解**（在 log-normal 假设下）：

$$\alpha^* \approx \frac{2\delta - \sigma^2}{\sigma^2 + 2\delta}$$

其中：
- $\sigma^2 = \text{Var}(\log w)$：分布偏移度量
- $\delta$：可接受的 bias 上界

---

## 9. 与现有方法的联系

### 9.1 方法统一表

| 方法 | 在框架中的位置 | α 值 | β 值 |
|-----|--------------|-----|-----|
| SFT | 前向 KL 极限 | -1 | 0 |
| RWR (Reward-Weighted Regression) | 接近 SFT | ≈ -0.8 | 0.1 |
| AWR/AWAC | 中等偏保守 | ≈ -0.5 | 0.25 |
| Hellinger Matching | 平衡点 | 0 | 0.5 |
| Soft Q-Learning | 中等偏激进 | ≈ 0.5 | 0.75 |
| PPO/RLHF | 反向 KL 极限 | +1 | 1 |

### 9.2 PPO Clipping 的解释

**命题 9.1**：PPO 的 clipping 可以理解为**自适应 α 方案**：

$$L^{CLIP} \approx \mathcal{L}_{\alpha(w)}$$

其中：
$$\alpha(w) = \begin{cases} +1 & \text{if } w \in [1-\epsilon, 1+\epsilon] \\ \text{降低} & \text{if } w \text{ 超出范围} \end{cases}$$

### 9.3 KL 惩罚的解释

**命题 9.2**：带 KL 惩罚的 RL：
$$\max_\theta \mathbb{E}_{\pi_\theta}[r] - \beta_{KL} D_{KL}(\pi_\theta \| \mu)$$

对应于 α = +1 但增加了显式约束，等价于：
$$\mathcal{L}_{+1}(\theta) + \beta_{KL} D_{KL}(\pi_\theta \| \mu)$$

---

## 10. 实现指南

### 10.1 算法伪代码

**算法 2：统一 α-散度离线 RL**

```
输入：
    D = {(x_i, y_i, r_i)} - 离线数据集
    μ - 行为策略（或其估计）
    α ∈ [-1, +1] - 插值参数
    τ > 0 - 温度参数

初始化：π_θ

For t = 1, 2, ..., T:
    1. 计算 β = (1+α)/2

    2. 计算重要性权重：
       w_i = π_θ(y_i|x_i) / μ(y_i|x_i)

    3. 计算归一化权重：
       η_i = w_i^{(1+α)/2} · exp((1-α)·β·r_i / (2τ))
       η̃_i = η_i / Σ_j η_j

    4. 计算梯度：
       g = (2/(1-α)) · Σ_i η̃_i · ∇log π_θ(y_i|x_i)

    5. 更新：θ ← θ + lr · g

    6. (可选) 自适应更新 α：
       σ² = Var(log w)
       α ← clip(adaptive_alpha(σ²), α - Δα, α + Δα)

输出：π_θ
```

### 10.2 PyTorch 实现框架

```python
import torch
import torch.nn.functional as F

def unified_alpha_loss(
    log_pi_theta,    # 当前策略的 log 概率
    log_mu,          # 行为策略的 log 概率
    rewards,         # 奖励
    alpha,           # α ∈ [-1, +1]
    tau=1.0          # 温度
):
    """
    统一 α-散度损失函数

    返回：
        loss: 需要最小化的损失（负的 L_α）
        info: 调试信息
    """
    beta = (1 + alpha) / 2

    # 计算 log 重要性权重
    log_w = log_pi_theta - log_mu

    # 计算加权项（在 log 空间操作以提高数值稳定性）
    log_weights = (1 + alpha) / 2 * log_w + (1 - alpha) * beta * rewards / (2 * tau)

    # 归一化（log-sum-exp 技巧）
    log_Z = torch.logsumexp(log_weights, dim=0)
    log_normalized_weights = log_weights - log_Z
    normalized_weights = torch.exp(log_normalized_weights).detach()

    # 损失：负的加权 log 似然
    loss = -torch.sum(normalized_weights * log_pi_theta)

    # 调试信息
    info = {
        'effective_sample_size': 1.0 / torch.sum(normalized_weights**2).item(),
        'max_weight': normalized_weights.max().item(),
        'alpha': alpha,
        'beta': beta
    }

    return loss, info
```

---

## 11. 总结

### 主要贡献

1. **真正的统一框架**：通过 Amari α-散度和耦合参数化，首次实现 SFT、RL、蒸馏的完全统一

2. **精确的极限恢复**：
   - α = -1 → 精确的前向 KL（SFT）
   - α = +1 → 精确的反向 KL（RL）

3. **连续插值**：α ∈ [-1, +1] 提供 mean-seeking 到 mode-seeking 的平滑过渡

4. **理论清晰**：
   - 梯度公式统一
   - Bias-variance trade-off 明确
   - 与现有方法的关系清晰

### 核心公式

$$\boxed{\mathcal{L}_\alpha(\theta) = -D_\alpha^{(A)}\left(\pi_\theta \| p_{\frac{1+\alpha}{2}}\right), \quad \alpha \in [-1, +1]}$$

### 方法谱系图

```
        α = -1                     α = 0                      α = +1
          │                          │                          │
          ▼                          ▼                          ▼
    ┌─────────┐               ┌─────────────┐              ┌─────────┐
    │   SFT   │               │  Hellinger  │              │   RL    │
    │ D(μ||π) │───────────────│   平衡点    │──────────────│ D(π||p*)│
    │mean-seek│               │   对称的    │              │mode-seek│
    └─────────┘               └─────────────┘              └─────────┘
          │                          │                          │
          ▼                          ▼                          ▼
    覆盖所有模式              平衡覆盖与聚焦              聚焦最优模式
```

---

## 附录 A：Amari α-散度的性质

### A.1 完整定义

对于 α ∈ ℝ：

$$D_\alpha^{(A)}(P \| Q) = \begin{cases}
\frac{4}{1-\alpha^2}\left(1 - \int P^{\frac{1+\alpha}{2}} Q^{\frac{1-\alpha}{2}} dx\right) & \alpha \neq \pm 1 \\
D_{KL}(P \| Q) & \alpha = +1 \\
D_{KL}(Q \| P) & \alpha = -1
\end{cases}$$

### A.2 信息几何解释

Amari α-散度对应于**统计流形上的 α-连接**：

- α = +1：指数连接（e-connection）
- α = -1：混合连接（m-connection）
- α = 0：Levi-Civita 连接

### A.3 与其他散度的关系

| 散度类型 | 对应 α 值 |
|---------|----------|
| KL 散度 D(P\|\|Q) | +1 |
| 反向 KL 散度 D(Q\|\|P) | -1 |
| Hellinger 距离 | 0 |
| χ² 散度 | +3 |
| 反向 χ² 散度 | -3 |

---

## 附录 B：符号表

| 符号 | 含义 |
|------|------|
| $\alpha$ | 统一参数，∈ [-1, +1] |
| $\beta$ | 目标分布倾斜度，= (1+α)/2 |
| $\mu(y\|x)$ | 行为策略（离线数据分布） |
| $\pi_\theta(y\|x)$ | 待学习的策略 |
| $p^*(y\|x)$ | 最优软策略 = $\mu e^{r/\tau}/Z$ |
| $p_\beta(y\|x)$ | β-倾斜分布 = $\mu e^{\beta r/\tau}/Z_\beta$ |
| $D_\alpha^{(A)}(P\|\|Q)$ | Amari α-散度 |
| $w = \pi_\theta/\mu$ | 重要性采样比率 |
| $\tau$ | 温度参数 |
| $r(x,y)$ | 奖励函数 |
