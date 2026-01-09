# 从 IS Reshape 视角重新思考 SFT：理论洞见与优化策略

## 1. 问题背景

### 1.1 场景设定

我们考虑以下典型场景：
- 有一个 **well-trained 的基础模型** $\pi_0$（例如经过大规模预训练的 LLM）
- 有一份 **高质量 SFT 数据** $\mathcal{D} = \{(x_i, y_i)\}$（已清洗，或来自 RFT）
- 目标：让模型学会数据中的新能力，同时**最小化对已有能力的破坏**

### 1.2 传统 SFT 的问题

传统 SFT 的目标是最大似然估计：
$$L_{\text{SFT}}(\theta) = \mathbb{E}_{(x,y) \sim \mathcal{D}}[\log \pi_\theta(y|x)]$$

根据我们的统一框架（§2.2），这等价于：
$$\min_\theta D_{KL}(\mu \| \pi_\theta)$$

其中 $\mu$ 是数据分布。

**Forward KL 的 mean-seeking 特性**意味着：
- $\pi_\theta$ 会试图**覆盖 μ 的所有模式**
- 即使某些模式与 $\pi_0$ 的当前能力相距甚远
- 为了覆盖这些"远"模式，模型需要大幅调整参数
- 这种调整会破坏 $\pi_0$ 已有的能力（**灾难性遗忘**）

### 1.3 核心矛盾

| 传统 SFT 的行为 | 我们想要的行为 |
|----------------|---------------|
| 覆盖 μ 的所有模式 | 优先学习与 $\pi_0$ 接近的模式 |
| Mean-seeking | Mode-seeking（在某种意义上） |
| 对所有样本平等对待 | 对"近"样本给予更高权重 |
| 可能导致大幅参数变化 | 最小化参数变化 |
| 灾难性遗忘风险高 | 保留已有能力 |

---

## 2. 理论框架的洞见

### 2.1 从统一梯度公式出发

回顾我们的核心定义：
$$g(\theta) = \mathbb{E}_\mu\left[f(w) \cdot r(x,y) \cdot \nabla_\theta \log \pi_\theta(y|x)\right]$$

其中 $w = \pi_\theta(y|x) / \mu(y|x)$。

**关键观察**：重要性权重 $w$ 本身就编码了"$\pi_\theta$ 与 μ 的相似度"
- $w > 1$：$\pi_\theta$ 比 μ 更喜欢这个样本 → 与当前模型"近"
- $w < 1$：$\pi_\theta$ 比 μ 更不喜欢这个样本 → 与当前模型"远"
- $w \approx 1$：两者对这个样本的偏好相似

### 2.2 f(w) 的选择如何影响学习行为

| f(w) | 权重特性 | 学习行为 |
|------|---------|---------|
| f(w) = 1 | 所有样本等权 | 覆盖所有模式（传统 SFT） |
| f(w) = w | 高 w 样本权重大 | 聚焦于 $\pi_\theta$ 已高概率的样本 |
| f(w) = $w^\gamma$ | 介于两者之间 | 可控的聚焦程度 |

**核心洞见**：

> **当 γ > 0 时，优化自然地从"覆盖 μ"变成"聚焦于 $\pi_\theta$ 与 μ 重叠的部分"。**

这正是我们想要的行为！

### 2.3 为什么这能缓解灾难性遗忘？

设当前模型为 $\pi_\theta$（初始化为 $\pi_0$），考虑两类样本：

**类型 A：与 $\pi_0$ 接近的样本**
- $\pi_0(y|x)$ 较高 → w 较大 → 权重 $w^\gamma$ 较大
- 学习这些样本只需要小幅参数调整
- 对其他能力影响小

**类型 B：与 $\pi_0$ 远离的样本**
- $\pi_0(y|x)$ 较低 → w 较小 → 权重 $w^\gamma$ 较小
- 这些样本被自动降权
- 不强迫模型大幅调整参数

**结果**：模型优先学习"顺其自然"的模式，而非被迫覆盖所有模式。

### 2.4 与目标分布的关系

在纯 SFT 场景下（r ≡ 1），不同 γ 对应的有效目标分布：

| γ | 有效目标分布 | 直觉 |
|---|-------------|------|
| 0 | $\mu$（数据分布） | 学习数据的所有模式 |
| 0.5 | $\propto \sqrt{\pi_\theta \cdot \mu}$ | 几何平均，折中 |
| 1 | $\propto \pi_\theta$（自我强化） | 只强化已有能力 |

**实践中**，我们不想走到 γ = 1（那样模型不学新东西），而是选择一个中间值，在"学习新知识"和"保持已有能力"之间取得平衡。

---

## 3. 实践方案：Mode-Seeking SFT

### 3.1 方法概述

**核心思想**：用 IS reshape 权重替代传统 SFT 的等权重

**损失函数**：
$$L_{\gamma}(\theta) = -\sum_i w_i^\gamma \cdot \log \pi_\theta(y_i|x_i)$$

其中：
$$w_i = \frac{\pi_\theta(y_i|x_i)}{\mu_{\text{ref}}(y_i|x_i)}$$

### 3.2 参考分布的选择

**问题**：我们没有真实的 $\mu$，只有数据样本。

**解决方案**：用参考模型 $\mu_{\text{ref}}$ 近似

| 参考模型选择 | 优点 | 缺点 |
|-------------|------|------|
| $\pi_0$（初始模型） | 简单，自然的"距离"度量 | 随着训练进行，度量可能不准 |
| SFT checkpoint | 更接近数据分布 | 需要额外训练一个模型 |
| 冻结的 $\pi_0$ | 固定参考，稳定的度量 | 推荐 ✓ |

**推荐**：使用冻结的初始模型 $\pi_0$ 作为 $\mu_{\text{ref}}$。这样 $w = \pi_\theta / \pi_0$ 度量的是"相对于初始模型的变化"。

### 3.3 自适应 γ 选择

根据我们的理论（§10），最优 γ 取决于分布偏移程度：
$$\gamma^* = \max\left(0, 1 - \frac{\sigma^2}{2\delta}\right)$$

**实践中的 ESS 自适应方法**：

```python
def adaptive_gamma(log_w, rho_min=0.3):
    """
    选择满足 ESS 约束的最大 γ

    log_w: log(π_θ / π_0) for each sample
    rho_min: 最小 ESS 比例
    """
    n = len(log_w)

    def compute_ess_ratio(gamma):
        weights = softmax(gamma * log_w)
        ess = 1.0 / sum(weights ** 2)
        return ess / n

    # 二分搜索
    gamma_low, gamma_high = 0.0, 1.0
    while gamma_high - gamma_low > 1e-3:
        gamma_mid = (gamma_low + gamma_high) / 2
        if compute_ess_ratio(gamma_mid) >= rho_min:
            gamma_low = gamma_mid
        else:
            gamma_high = gamma_mid

    return gamma_low
```

### 3.4 完整算法

```python
def mode_seeking_sft(
    model,           # 待训练模型 π_θ，初始化为 π_0
    ref_model,       # 参考模型 π_0（冻结）
    data,            # SFT 数据
    gamma=None,      # IS reshape 参数（None 则自适应）
    rho_min=0.3,     # ESS 约束
    epochs=3
):
    """
    Mode-Seeking SFT: 优先学习与当前模型接近的模式
    """
    for epoch in range(epochs):
        for batch in data:
            x, y = batch

            # 计算 log 概率
            with torch.no_grad():
                log_pi_ref = ref_model.log_prob(y, x)
            log_pi = model.log_prob(y, x)

            # 计算 log 重要性权重
            log_w = log_pi - log_pi_ref

            # 自适应选择 γ
            if gamma is None:
                gamma_batch = adaptive_gamma(log_w.detach(), rho_min)
            else:
                gamma_batch = gamma

            # 计算归一化权重
            weights = F.softmax(gamma_batch * log_w.detach(), dim=0)

            # Mode-seeking SFT 损失
            loss = -torch.sum(weights * log_pi)

            # 更新
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model
```

---

## 4. 理论分析：为什么这能减少灾难性遗忘

### 4.1 梯度方向的改变

**传统 SFT 的梯度**：
$$g_{\text{SFT}} = \mathbb{E}_\mu[\nabla \log \pi_\theta]$$

这个梯度指向"覆盖 μ 所有模式"的方向。

**Mode-Seeking SFT 的梯度**：
$$g_\gamma = \mathbb{E}_\mu[w^\gamma \cdot \nabla \log \pi_\theta]$$

这个梯度被重新加权，指向"聚焦于 $\pi_\theta$ 已高概率样本"的方向。

### 4.2 参数变化量的对比

设 $\Delta\theta$ 为参数变化。直观上：

| 方法 | 参数变化特点 |
|------|-------------|
| 传统 SFT | 需要移动参数以覆盖所有样本，包括远离的样本 |
| Mode-Seeking SFT | 主要移动参数以更好地拟合"近"样本 |

**量化分析**：参数变化量与梯度的二阶矩相关。由于 Mode-Seeking SFT 降低了"远"样本的权重，梯度的方差更小，参数变化更稳定。

### 4.3 与 KL 正则化的联系

许多 RLHF 方法使用 KL 正则化：
$$L = L_{\text{task}} + \beta \cdot D_{KL}(\pi_\theta \| \pi_0)$$

这显式地惩罚偏离初始模型。

**我们的方法（Mode-Seeking SFT）提供了一个不同的视角**：
- 不是显式地惩罚偏离
- 而是通过样本加权，隐式地让优化聚焦于"近"模式
- 两种方法可以结合使用

---

## 5. 与现有方法的关系和比较

### 5.1 方法对比表

| 方法 | 核心思想 | 与 IS Reshape 的关系 |
|------|---------|---------------------|
| **传统 SFT** | 最大似然 | f(w)=1, r≡1 |
| **奖励加权 SFT** | 高奖励样本权重大 | f(w)=1, r=reward |
| **Mode-Seeking SFT**（本文） | 近模式优先 | f(w)=w^γ, r≡1 |
| **DPO** | 对比学习 | 隐式的 mode-seeking |
| **KL 正则化** | 惩罚偏离 | 与 mode-seeking 互补 |
| **LoRA/PEFT** | 限制参数更新 | 正交方法，可结合 |
| **Replay** | 混合旧数据 | 正交方法，可结合 |

### 5.2 与 DPO 的深层联系

DPO 的目标函数：
$$L_{DPO} = -\log \sigma\left(\beta \log \frac{\pi_\theta(y_w)}{\pi_0(y_w)} - \beta \log \frac{\pi_\theta(y_l)}{\pi_0(y_l)}\right)$$

**关键观察**：DPO 中 $\log \frac{\pi_\theta}{\pi_0}$ 正是我们的 log w！

DPO 隐式地实现了 mode-seeking：
- 优先增加那些"已经相对高概率"的好样本的概率
- 降低"已经相对低概率"的坏样本的概率

**我们的方法与 DPO 的区别**：
- DPO 需要偏好对 (y_w, y_l)
- Mode-Seeking SFT 只需要正样本
- 两者都利用了 w = π_θ/π_0 的信息

### 5.3 与课程学习的联系

Mode-Seeking SFT 可以看作一种**隐式的课程学习**：
- 训练初期，大部分样本 w ≈ 1，学习较均匀
- 随着训练，模型开始分化，自动聚焦于"擅长"的样本
- 困难/远离的样本被自动推迟或降权

---

## 6. 实践建议与最佳实践

### 6.1 何时使用 Mode-Seeking SFT

**推荐使用的场景**：
- 在 well-trained 模型上做领域微调
- 希望学习新能力的同时保持通用能力
- SFT 数据量较小，担心过拟合或遗忘
- 数据质量高但风格多样（不想覆盖所有风格）

**不推荐使用的场景**：
- 从头训练（此时应该覆盖所有模式）
- 数据量很大且同质（传统 SFT 足够）
- 需要模型学习与当前能力完全不同的新技能

### 6.2 超参数选择指南

| 参数 | 推荐值 | 说明 |
|------|-------|------|
| γ（固定） | 0.3 - 0.7 | 越大越 mode-seeking |
| γ（自适应） | 由 ESS 决定 | 推荐 |
| ρ_min（ESS 比例） | 0.2 - 0.5 | 越小允许越激进的 IS |
| 学习率 | 正常或略低 | 配合 mode-seeking 减少波动 |

### 6.3 与其他方法的结合

**推荐组合**：

1. **Mode-Seeking SFT + LoRA**
   - LoRA 限制参数变化的维度
   - Mode-Seeking 限制样本影响的分布
   - 双重保护，效果最好

2. **Mode-Seeking SFT + KL 正则化**
   ```python
   loss = mode_seeking_loss + beta * kl_divergence(pi_theta, pi_0)
   ```
   - 双重约束：样本加权 + 显式惩罚

3. **Mode-Seeking SFT + 数据混合**
   - 在 SFT 数据中混入少量通用数据
   - Mode-Seeking 确保新数据不会压倒旧能力

### 6.4 监控指标

训练过程中应监控：

1. **ESS 比例**：$\rho = \text{ESS}/n$
   - 过低（< 0.1）：γ 过大，实际只学少数样本
   - 适中（0.2-0.5）：理想范围
   - 过高（> 0.8）：接近普通 SFT

2. **权重分布**：观察 $w^\gamma$ 的分布
   - 应该是右偏但不过度集中

3. **验证损失**：在旧任务上的性能
   - 应该保持稳定或小幅下降

---

## 7. 扩展思考

### 7.1 Per-Sample γ

不同样本可能需要不同的 γ：
- 与 $\pi_0$ 很近的样本：可以用较大的 γ（强化）
- 与 $\pi_0$ 很远的样本：可能需要较小的 γ（需要学习）

$$\gamma_i = g(\text{distance}_i)$$

其中 distance 可以用 $-\log \pi_0(y_i|x_i)$ 或 KL 散度度量。

### 7.2 动态 γ 调度

训练过程中调整 γ：
- 初期：γ 较小，广泛学习
- 中期：γ 增大，开始聚焦
- 后期：γ 较大，精细打磨

类似于学习率调度，但作用于样本权重。

### 7.3 与主动学习的结合

Mode-Seeking SFT 的权重可以指导数据选择：
- 低 w 样本：模型不擅长，可能需要更多类似样本
- 高 w 样本：模型已擅长，可以减少采样

这可以用于迭代的数据收集和训练。

---

## 8. Sequence-level 与 Token-level 目标的关系

### 8.1 问题背景

我们的统一框架在 **sequence level** 定义：
$$g(\theta) = \mathbb{E}_\mu\left[f(w) \cdot r \cdot \nabla_\theta \log \pi_\theta(y|x)\right]$$

其中 $w = \pi_\theta(y|x) / \mu(y|x)$ 是**序列级别**的重要性权重。

然而，LLM 是 token-by-token 生成的：
$$\pi_\theta(y|x) = \prod_{t=1}^{|y|} \pi_\theta(y_t|x, y_{<t})$$

**核心问题**：在什么条件下，我们可以用 token-level 的实现来优化 sequence-level 的目标？

### 8.2 一阶近似理论

**关键洞见**（参考 Qwen 团队的工作）：Token-level 目标可以看作 sequence-level 目标的**一阶近似**。

**推导**：设 $\delta_t = \frac{\pi_\theta(y_t|x,y_{<t})}{\mu(y_t|x,y_{<t})} - 1$ 是每个 token 的重要性比率偏移量。

当 $\delta_t$ 较小时（即 $\pi_\theta \approx \mu$）：

$$w = \frac{\pi_\theta(y|x)}{\mu(y|x)} = \prod_{t=1}^{|y|}(1 + \delta_t) \approx 1 + \sum_{t=1}^{|y|}\delta_t + O(\delta^2)$$

忽略二阶及更高阶项，我们得到：

$$w \approx 1 + \sum_{t=1}^{|y|}\delta_t = \sum_{t=1}^{|y|}\frac{\pi_\theta(y_t|x,y_{<t})}{\mu(y_t|x,y_{<t})}$$

**结论**：序列级重要性权重可以近似为 token 级重要性权重之和。

### 8.3 对 Mode-Seeking SFT 的意义

在我们的 Mode-Seeking SFT 中，损失函数为：
$$L_\gamma(\theta) = -\sum_i w_i^\gamma \cdot \log \pi_\theta(y_i|x_i)$$

**Token-level 近似**：当 $\gamma$ 较小且 $\pi_\theta \approx \mu$ 时：

$$w^\gamma \approx \left(1 + \sum_t \delta_t\right)^\gamma \approx 1 + \gamma \sum_t \delta_t$$

这意味着我们可以用 token-level 的权重和来近似 sequence-level 的权重。

### 8.4 近似有效的条件

一阶近似成立需要满足两个条件：

| 条件 | 含义 | 如何满足 |
|-----|------|---------|
| **Training-Inference 一致性** | 训练引擎与推理引擎的数值计算一致 | 使用相同精度、相同 kernel |
| **Policy Staleness 较小** | rollout 策略与当前策略接近 | 控制 off-policy 程度、使用 clipping |

**对于 MoE 模型的特殊挑战**：
- Expert routing 可能在训练和推理时不一致
- 解决方案：Routing Replay（固定 expert 路由）

### 8.5 实践建议

**1. 监控 $\delta_t$ 的大小**：

```python
def monitor_token_delta(log_pi, log_mu):
    """监控 token-level 的偏移量"""
    delta = torch.exp(log_pi - log_mu) - 1
    return {
        'mean_abs_delta': delta.abs().mean().item(),
        'max_abs_delta': delta.abs().max().item(),
        'std_delta': delta.std().item(),
    }
```

- 当 `mean_abs_delta > 0.5` 时，一阶近似可能失效
- 建议通过减小学习率或增加 clipping 来控制

**2. Clipping 策略**：

参考 PPO 风格的 clipping，防止单步更新过大：

```python
def clipped_weight(log_pi, log_pi_old, epsilon_low=0.2, epsilon_high=0.27):
    """Clip token-level 权重以控制 policy staleness"""
    ratio = torch.exp(log_pi - log_pi_old)
    clipped_ratio = torch.clamp(ratio, 1 - epsilon_low, 1 + epsilon_high)
    return clipped_ratio
```

**3. 对于 MoE 模型**：

建议使用 Routing Replay：
- **R2 (Vanilla Routing Replay)**：重放训练引擎的 expert routing，减少 policy staleness
- **R3 (Rollout Routing Replay)**：重放推理引擎的 expert routing，同时减少 training-inference 差异和 policy staleness

### 8.6 与主框架的联系

| 层级 | 理论定义 | 实际实现 | 近似条件 |
|-----|---------|---------|---------|
| Sequence-level | $w = \pi_\theta(y|x)/\mu(y|x)$ | 需要完整序列概率 | 精确 |
| Token-level | $\approx \sum_t \pi_\theta(y_t)/\mu(y_t)$ | 逐 token 计算 | $\pi_\theta \approx \mu$ |

**实践指导**：
- 当 γ 较小（如 γ < 0.5）时，token-level 实现通常足够准确
- 当 γ 较大或分布偏移严重时，考虑使用 sequence-level 权重或增加约束

---

## 9. 直接求解的近似方法

### 9.1 迭代求解的问题

标准的 IS Reshape 优化采用策略梯度迭代更新：

$$\theta_{t+1} = \theta_t + \eta \cdot \mathbb{E}_\mu\left[f(w) \cdot r \cdot \nabla_\theta \log \pi_\theta\right]$$

**存在的问题**：
1. 每步需要计算当前策略的概率，计算开销大
2. 需要多次迭代才能收敛
3. 超参数（学习率、迭代次数）敏感
4. 在 offline 场景下，无法与环境交互来修正 off-policy 误差

**核心问题**：能否找到近似方法，直接（或近似直接）求出最优策略，避免迭代？

### 9.2 理论基础：最优分布的闭式解

#### 9.2.1 KL 散度约束下的最优解

考虑优化问题：
$$\max_\pi \mathbb{E}_\pi[r] - \beta \cdot D_{KL}(\pi \| \mu)$$

这有解析解：
$$\pi^*(y|x) = \frac{1}{Z} \mu(y|x) \cdot \exp\left(\frac{r(x,y)}{\beta}\right)$$

其中 $Z = \mathbb{E}_\mu[\exp(r/\beta)]$ 是归一化常数。

#### 9.2.2 α-散度约束下的最优解

对于广义的 α-散度（对应 IS Reshape 的 f(w) = w^γ，其中 γ = 1-α）：

$$\pi^*_\alpha(y|x) \propto \mu(y|x) \cdot \left[1 + \frac{\alpha}{\beta} r(x,y)\right]_+^{1/\alpha}$$

当 $\alpha \to 0$ 时，退化为 KL 情况的指数形式。

**关键洞见**：如果我们能直接利用这些闭式解，就不需要迭代！

### 9.3 近似方法一：奖励加权回归 (Reward-Weighted Regression)

#### 9.3.1 核心思想

既然最优分布有形式 $\pi^* \propto \mu \cdot g(r)$，我们可以直接用加权 MLE 来拟合：

$$\theta^* = \arg\max_\theta \sum_i g(r_i) \cdot \log \pi_\theta(y_i|x_i)$$

其中 $g(r)$ 是奖励变换函数。

#### 9.3.2 常见的 g(r) 选择

| 方法 | g(r) | 对应目标 |
|-----|------|---------|
| **指数权重** | $\exp(r/\beta)$ | KL 约束的最优解 |
| **截断指数** | $\exp(\min(r, r_{\max})/\beta)$ | 有界 KL 约束 |
| **幂函数** | $(r - r_{\min})^\alpha$ | 近似 α-散度 |
| **Top-K 指示器** | $\mathbb{1}[r \in \text{top-K}]$ | 最简单的近似 |
| **优势归一化** | $\exp((r-\bar{r})/\sigma_r)$ | 自适应缩放 |

#### 9.3.3 实现

```python
def reward_weighted_regression(
    model,
    data,  # [(x_i, y_i, r_i)]
    beta: float = 1.0,
    reward_transform: str = "exp",  # "exp", "power", "topk"
    topk_ratio: float = 0.2,
    num_epochs: int = 1,
):
    """
    直接求解：用奖励加权的 MLE 拟合最优分布

    优点：无需迭代的策略梯度，单轮训练即可
    """

    # 计算权重
    rewards = torch.tensor([r for _, _, r in data])

    if reward_transform == "exp":
        # 标准指数权重
        weights = torch.softmax(rewards / beta, dim=0)
    elif reward_transform == "power":
        # 幂函数权重（对应 α-散度）
        r_shifted = rewards - rewards.min()
        weights = r_shifted ** (1.0 / beta)
        weights = weights / weights.sum()
    elif reward_transform == "topk":
        # Top-K 简单近似
        k = int(len(rewards) * topk_ratio)
        topk_indices = torch.topk(rewards, k).indices
        weights = torch.zeros_like(rewards)
        weights[topk_indices] = 1.0 / k

    # 加权 MLE 训练
    for epoch in range(num_epochs):
        for batch_idx, (x, y, r) in enumerate(data):
            w = weights[batch_idx].item()

            log_prob = model.log_prob(y, x)
            loss = -w * log_prob

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model
```

### 9.4 近似方法二：自归一化重要性采样

#### 9.4.1 问题：归一化常数的计算

最优分布需要归一化常数：
$$Z = \sum_y \mu(y|x) \cdot \exp(r(x,y)/\beta)$$

这通常需要遍历所有可能的 y，计算量极大。

#### 9.4.2 自归一化技巧

用样本均值代替期望：

$$\hat{\pi}^*(y_i|x) \approx \frac{\mu(y_i|x) \cdot \exp(r_i/\beta)}{\sum_j \mu(y_j|x) \cdot \exp(r_j/\beta)} = \frac{\exp(r_i/\beta)}{\sum_j \exp(r_j/\beta)}$$

**关键**：$\mu(y|x)$ 在分子分母中抵消了！

最终权重简化为：
$$\hat{w}_i = \text{softmax}(r_i / \beta)$$

#### 9.4.3 实现

```python
def self_normalized_is_training(
    model,
    data_batches,
    beta: float = 1.0,
):
    """
    自归一化 IS：避免显式计算归一化常数

    每个 batch 内用 softmax(r/β) 作为权重
    """
    for batch in data_batches:
        x_batch, y_batch, r_batch = batch

        # 自归一化权重（batch 内）
        weights = F.softmax(r_batch / beta, dim=0)

        # 加权对数似然
        log_probs = model.log_prob(y_batch, x_batch)
        loss = -torch.sum(weights * log_probs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### 9.5 近似方法三：一阶泰勒展开

#### 9.5.1 适用场景

当策略变化较小时（$\pi_\theta \approx \mu$），可以用一阶近似。

#### 9.5.2 推导

设 $\pi_\theta(y|x) = \mu(y|x)(1 + \delta_\theta(y|x))$，其中 $\delta_\theta$ 是小扰动。

归一化约束：$\mathbb{E}_\mu[\delta_\theta] = 0$

优化目标（一阶展开）：
$$\max_{\delta} \mathbb{E}_\mu[(1+\delta) \cdot r] - \frac{\beta}{2}\mathbb{E}_\mu[\delta^2/\mu]$$

这是关于 $\delta$ 的二次优化问题，有闭式解：
$$\delta^*(y|x) = \frac{1}{\beta}(r(y|x) - \bar{r})$$

其中 $\bar{r} = \mathbb{E}_\mu[r]$ 确保归一化。

#### 9.5.3 对参数化策略的应用

如果 $\delta_\theta(y|x) = \theta^T \phi(y,x)$ 是线性参数化：

$$\theta^* = \frac{1}{\beta} \mathbb{E}_\mu[\phi(y,x) \cdot r(x,y)]$$

这是**单步闭式解**！

```python
def first_order_closed_form(
    feature_fn,  # φ(y, x)
    data,  # [(x_i, y_i, r_i)]
    beta: float = 1.0,
):
    """
    一阶近似的闭式解

    适用于：线性参数化或特征空间
    """
    # 计算 E[φ·r]
    features = torch.stack([feature_fn(y, x) for x, y, _ in data])
    rewards = torch.tensor([r for _, _, r in data])

    # 中心化奖励
    rewards_centered = rewards - rewards.mean()

    # 闭式解
    theta_star = (features.T @ rewards_centered) / (beta * len(data))

    return theta_star
```

### 9.6 近似方法四：变分拟合 (Variational Fitting)

#### 9.6.1 核心思想

将最优分布限制在某个参数化族内，然后找最近的近似。

设目标分布为 $\pi^* \propto \mu \cdot \exp(r/\beta)$，我们找 $\pi_\theta$ 使得：

$$\theta^* = \arg\min_\theta D_{KL}(\pi^* \| \pi_\theta)$$

#### 9.6.2 转化为加权 MLE

$$\theta^* = \arg\max_\theta \mathbb{E}_{\pi^*}[\log \pi_\theta] = \arg\max_\theta \mathbb{E}_\mu\left[\frac{\pi^*}{\mu} \log \pi_\theta\right]$$

由于 $\pi^*/\mu \propto \exp(r/\beta)$：

$$\theta^* = \arg\max_\theta \sum_i \exp(r_i/\beta) \cdot \log \pi_\theta(y_i|x_i)$$

这就是奖励加权回归！但有更精细的变体。

#### 9.6.3 迭代变分拟合（少量迭代）

```python
def iterative_variational_fitting(
    model,
    data,
    beta: float = 1.0,
    num_iterations: int = 3,  # 只需要少量迭代
):
    """
    迭代变分拟合：用少量迭代达到更好的近似

    关键：每次迭代用当前 π_θ 来重新估计目标分布
    """
    for iteration in range(num_iterations):
        # 计算当前的重要性权重
        with torch.no_grad():
            log_ratios = []
            for x, y, r in data:
                log_pi = model.log_prob(y, x)
                log_ratios.append(r / beta)  # 第一次迭代用奖励

        weights = F.softmax(torch.tensor(log_ratios), dim=0)

        # 一轮加权 MLE
        for epoch in range(1):
            for i, (x, y, r) in enumerate(data):
                log_prob = model.log_prob(y, x)
                loss = -weights[i] * log_prob

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    return model
```

### 9.7 近似方法五：Best-of-N 与 Top-K 采样

#### 9.7.1 最简单的近似

不需要训练，只需要在推理时做选择：

1. **Best-of-N (BoN)**：采样 N 个响应，选择奖励最高的
2. **Top-K 微调**：只用奖励最高的 K% 样本做 SFT

#### 9.7.2 理论联系

Best-of-N 近似的分布：
$$\pi_{\text{BoN}}(y|x) \approx \mu(y|x) \cdot N \cdot F_\mu(r(y))^{N-1}$$

其中 $F_\mu$ 是奖励的 CDF。

**与 IS Reshape 的联系**：当 N 较大时，BoN 近似 mode-seeking 行为。

#### 9.7.3 实现

```python
def best_of_n_inference(
    model,
    x,
    reward_model,
    n: int = 16,
):
    """Best-of-N 推理：零训练的近似"""
    candidates = [model.generate(x) for _ in range(n)]
    rewards = [reward_model(x, y) for y in candidates]
    best_idx = np.argmax(rewards)
    return candidates[best_idx]


def topk_sft(
    model,
    data,  # [(x_i, y_i, r_i)]
    k_ratio: float = 0.2,
):
    """Top-K SFT：只在最好的样本上训练"""
    # 选择 top K% 样本
    rewards = [r for _, _, r in data]
    threshold = np.percentile(rewards, 100 * (1 - k_ratio))
    filtered_data = [(x, y, r) for x, y, r in data if r >= threshold]

    # 标准 SFT
    for x, y, _ in filtered_data:
        log_prob = model.log_prob(y, x)
        loss = -log_prob

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model
```

### 9.8 方法对比与选择指南

| 方法 | 计算复杂度 | 近似精度 | 适用场景 |
|-----|-----------|---------|---------|
| **奖励加权回归** | O(1) 训练轮 | 中等 | 有明确奖励，数据质量高 |
| **自归一化 IS** | O(1) 训练轮 | 中等-高 | 批量数据可用 |
| **一阶闭式解** | O(1) 计算 | 低（$\pi \approx \mu$） | 微调场景，变化小 |
| **变分拟合** | O(k) 迭代，k 小 | 高 | 需要更精确的近似 |
| **Best-of-N** | 推理时 O(N) | 依赖 N | 无法微调时 |
| **Top-K SFT** | O(1) 训练轮 | 中等 | 最简单，数据量大 |

### 9.9 实践建议

#### 9.9.1 何时用直接方法 vs 迭代方法

**使用直接方法**：
- Offline 场景，无法与环境交互
- 计算资源有限，只能做少量训练
- 初步快速实验，验证方向
- 数据质量高，奖励信号清晰

**使用迭代方法**：
- 需要精确控制策略变化
- 分布偏移严重，需要多步调整
- 有在线交互的可能
- 对最终性能要求高

#### 9.9.2 温度参数 β 的选择

```python
def adaptive_beta(rewards, target_ess_ratio=0.3):
    """
    根据 ESS 自适应选择 β

    思路：β 控制权重的集中程度
    - β 大 → 权重更均匀
    - β 小 → 权重更集中在高奖励样本
    """
    def compute_ess_ratio(beta):
        weights = F.softmax(torch.tensor(rewards) / beta, dim=0)
        ess = 1.0 / (weights ** 2).sum()
        return ess / len(rewards)

    # 二分搜索
    beta_low, beta_high = 0.01, 10.0
    for _ in range(20):
        beta_mid = (beta_low + beta_high) / 2
        if compute_ess_ratio(beta_mid) >= target_ess_ratio:
            beta_high = beta_mid
        else:
            beta_low = beta_mid

    return beta_mid
```

#### 9.9.3 与 IS Reshape 的统一视角

所有直接方法都可以理解为 IS Reshape 的特例或近似：

| 直接方法 | IS Reshape 视角 |
|---------|----------------|
| 指数加权 RWR | f(w) = 1，但隐式引入了 exp(r/β) 变换 |
| Top-K SFT | f(w) = 𝟙[w ∈ top-K]，硬截断 |
| 一阶近似 | f(w) ≈ 1 + γ(w-1)，线性化 |
| BoN | 隐式的 f(w) = w^(N-1)·N |

---

## 10. 总结

### 10.1 核心洞见

1. **传统 SFT 的问题**：Forward KL 的 mean-seeking 特性导致模型被迫覆盖所有模式，引发灾难性遗忘

2. **IS Reshape 的解决方案**：通过 f(w) = w^γ 加权，将优化从 mean-seeking 转向 mode-seeking

3. **实践意义**：优先学习与当前模型接近的模式，最小化参数变化，保护已有能力

4. **理论保证**：γ 的选择有理论指导（Bias-Variance 权衡），可通过 ESS 自适应调整

5. **Sequence vs Token Level**：Token-level 目标是 sequence-level 目标的一阶近似，当 $\pi_\theta \approx \mu$ 时近似有效

6. **直接求解方法**：通过奖励加权回归、自归一化 IS 等技术，可以避免迭代求解，实现近似直接优化

### 10.2 公式总结

**Mode-Seeking SFT 损失**：
$$L_\gamma(\theta) = -\sum_i \bar{w}_i^\gamma \cdot \log \pi_\theta(y_i|x_i)$$

其中：
$$\bar{w}_i = \frac{\pi_\theta(y_i|x_i)}{\pi_0(y_i|x_i)}, \quad \gamma \in [0, 1]$$

**γ 的作用**：
- γ = 0：传统 SFT（mean-seeking）
- γ > 0：Mode-seeking，聚焦于近模式
- γ = 1：纯自我强化（不推荐）

### 10.3 实践 Checklist

- [ ] 保存初始模型 $\pi_0$ 作为参考
- [ ] 实现 IS 权重计算：$w = \pi_\theta / \pi_0$
- [ ] 实现自适应 γ 选择（基于 ESS）
- [ ] 监控 ESS 比例，确保在合理范围
- [ ] 监控 token-level $\delta_t$ 的大小，确保一阶近似有效
- [ ] 在旧任务上验证性能保持
- [ ] 考虑与 LoRA/KL 正则化结合
- [ ] 对于 MoE 模型，考虑使用 Routing Replay
- [ ] **（新增）** 评估是否可以使用直接求解方法（奖励加权回归、Top-K SFT）
- [ ] **（新增）** 根据场景选择合适的温度参数 β（或使用自适应方法）
- [ ] **（新增）** 对于 offline 场景，优先考虑自归一化 IS 或变分拟合

---

## 附录：PyTorch 实现

```python
import torch
import torch.nn.functional as F
from typing import Optional, Tuple

class ModeSeeking SFTTrainer:
    """
    Mode-Seeking SFT: 从 IS Reshape 视角优化的 SFT

    核心思想：通过 w^γ 加权，优先学习与当前模型接近的模式
    """

    def __init__(
        self,
        model: torch.nn.Module,
        ref_model: torch.nn.Module,  # 冻结的参考模型
        gamma: Optional[float] = None,  # None 则自适应
        rho_min: float = 0.3,  # ESS 约束
        kl_coef: float = 0.0,  # 可选的 KL 正则化
    ):
        self.model = model
        self.ref_model = ref_model
        self.gamma = gamma
        self.rho_min = rho_min
        self.kl_coef = kl_coef

        # 冻结参考模型
        for param in self.ref_model.parameters():
            param.requires_grad = False

    def compute_loss(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """
        计算 Mode-Seeking SFT 损失
        """
        # 当前模型的 log prob
        outputs = self.model(input_ids, attention_mask=attention_mask)
        log_probs = self._compute_log_probs(outputs.logits, labels)

        # 参考模型的 log prob
        with torch.no_grad():
            ref_outputs = self.ref_model(input_ids, attention_mask=attention_mask)
            ref_log_probs = self._compute_log_probs(ref_outputs.logits, labels)

        # 计算 log 重要性权重
        log_w = log_probs - ref_log_probs

        # 自适应选择 γ
        if self.gamma is None:
            gamma = self._adaptive_gamma(log_w.detach())
        else:
            gamma = self.gamma

        # 计算归一化权重（数值稳定）
        weights = F.softmax(gamma * log_w.detach(), dim=0)

        # Mode-Seeking SFT 损失
        loss = -torch.sum(weights * log_probs)

        # 可选：KL 正则化
        if self.kl_coef > 0:
            kl_div = torch.mean(log_probs - ref_log_probs)
            loss = loss + self.kl_coef * kl_div

        # 计算监控指标
        ess = 1.0 / torch.sum(weights ** 2)
        metrics = {
            'gamma': gamma,
            'ess': ess.item(),
            'ess_ratio': ess.item() / len(log_w),
            'max_weight': weights.max().item(),
            'mean_log_w': log_w.mean().item(),
            'std_log_w': log_w.std().item(),
        }

        return loss, metrics

    def _compute_log_probs(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """计算每个序列的 log probability"""
        # Shift for causal LM
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # 计算 token 级别的 log prob
        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_log_probs = torch.gather(
            log_probs,
            dim=-1,
            index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)

        # 对有效 token 求和（忽略 padding）
        mask = (shift_labels != -100).float()
        seq_log_probs = (token_log_probs * mask).sum(dim=-1)

        return seq_log_probs

    def _adaptive_gamma(self, log_w: torch.Tensor) -> float:
        """基于 ESS 约束自适应选择 γ"""
        n = len(log_w)

        def compute_ess_ratio(gamma):
            weights = F.softmax(gamma * log_w, dim=0)
            ess = 1.0 / torch.sum(weights ** 2)
            return (ess / n).item()

        # 二分搜索
        gamma_low, gamma_high = 0.0, 2.0
        for _ in range(20):
            gamma_mid = (gamma_low + gamma_high) / 2
            if compute_ess_ratio(gamma_mid) >= self.rho_min:
                gamma_low = gamma_mid
            else:
                gamma_high = gamma_mid

        return gamma_low


# 使用示例
def train_mode_seeking_sft(
    model,
    ref_model,
    train_dataloader,
    optimizer,
    num_epochs=3,
    rho_min=0.3,
):
    trainer = ModeSeekingSFTTrainer(
        model=model,
        ref_model=ref_model,
        gamma=None,  # 自适应
        rho_min=rho_min,
    )

    for epoch in range(num_epochs):
        total_loss = 0
        total_gamma = 0
        total_ess_ratio = 0
        num_batches = 0

        for batch in train_dataloader:
            loss, metrics = trainer.compute_loss(
                batch['input_ids'],
                batch['attention_mask'],
                batch['labels'],
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_gamma += metrics['gamma']
            total_ess_ratio += metrics['ess_ratio']
            num_batches += 1

        print(f"Epoch {epoch+1}: "
              f"Loss={total_loss/num_batches:.4f}, "
              f"γ={total_gamma/num_batches:.3f}, "
              f"ESS_ratio={total_ess_ratio/num_batches:.3f}")
```
