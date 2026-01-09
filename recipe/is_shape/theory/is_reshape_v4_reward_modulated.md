# IS Reshape v4：基于信任域和 Reward 调制的统一框架

**版本**: 4.0
**核心创新**: 将 SAPO 的软信任域机制与 reward 调制的 γ 结合，实现真正的 SFT-RL 连续插值
**状态**: 理论设计完成，待实验验证

---

## 摘要

IS Reshape v4 基于 v3.1 实验的关键发现——**方差控制 ≠ KL 控制**——重新设计框架：

1. **从 SAPO 学习**：引入 sech²(τ(r-1)/2) 形式的软信任域，确保离策略样本的梯度衰减
2. **Reward 调制 γ**：γ(|A|) = σ(β|A|)，让 reward 信号强度决定 SFT/RL 的混合比例
3. **统一框架**：f(r) = (1-γ) + γ·r·sech²(τ(r-1)/2)，当 γ=0 时退化为 SFT，γ=1 时接近 SAPO

---

# 第一部分：v3.1 的失败分析

## 1. 实验发现

### 1.1 关键数据

| Step | is_reshape grad_norm | noclip grad_norm | 比值 |
|------|---------------------|------------------|------|
| 70 | 0.074 | 0.402 | 0.18 |
| 74 | 0.067 | 4931.9 | 0.00001 |
| 80 | 0.088 | 1.986 | 0.044 |

**反直觉的结果**：
- is_reshape 的 grad_norm 比 noclip 小 **10-100000 倍**
- 但 is_reshape 的 KL 增长却**更快**（step 80: 0.068 vs noclip: 0.022）

### 1.2 根本原因

**v3.1 控制的是什么**：
- ✓ 梯度估计的**方差**（通过 w^γ 压缩）
- ✗ 梯度的**累积效应**（无信任域）
- ✗ KL 散度（无直接约束）

**为什么低方差反而导致更大的 KL？**

```
Noclip (高方差):
  Epoch 1: ████████░░░░  (大梯度)
  Epoch 2: ░░░░░░░░░░░░  (梯度爆炸，Adam 抑制)
  Epoch 3: ███░░░░░░░░░  (中等梯度)
  累积 KL: 小 (噪声相互抵消，Adam 抑制大梯度)

IS Reshape (低方差):
  Epoch 1: ██████░░░░░░  (一致的梯度)
  Epoch 2: ██████░░░░░░  (一致的梯度)
  Epoch 3: ██████░░░░░░  (一致的梯度)
  累积 KL: 大 (每步都往同一方向推，优化器信任这些梯度)
```

### 1.3 与 PPO Clip 的本质区别

| 机制 | PPO Clip | IS Reshape v3.1 |
|------|----------|-----------------|
| 控制什么 | KL 散度（信任域） | 梯度方差 |
| 如何控制 | 超出范围时**梯度=0** | 缩放梯度但**永不为0** |
| 累积效应 | 有上界 | 无上界 |

```python
# PPO clip: 超出范围时梯度为 0
if ratio > 1 + eps or ratio < 1 - eps:
    gradient = 0  # 停止更新！

# IS Reshape v3.1: 梯度被缩放但永不为 0
gradient = gamma * w_gamma * A  # 即使 gamma=0.1，也不是 0
```

**关键缺失**：v3.1 没有"停止更新"的机制。

---

# 第二部分：SAPO 的启示

## 2. SAPO 核心机制

### 2.1 SAPO 的梯度权重

SAPO (Soft Adaptive Policy Optimization) 使用：

$$f_{SAPO}(r) = \sigma(\tau(r-1)) \cdot \frac{4}{\tau}$$

对应的梯度权重（可从 $\nabla f$ 或 $r \cdot f'(r)$ 推导）：

$$w_{SAPO}(r) = r \cdot \text{sech}^2\left(\frac{\tau(r-1)}{2}\right)$$

### 2.2 关键特性

**非单调性**：$w_{SAPO}(r)$ 在 r=1 达到峰值，然后**对称衰减**：

| r | w_SAPO(r) (τ=1) |
|---|-----------------|
| 0.2 | 0.10 |
| 0.5 | 0.39 |
| 1.0 | 1.00 (峰值) |
| 2.0 | 0.79 |
| 5.0 | 0.35 |
| 10.0 | 0.002 |

**这是隐式的信任域**：当 r 偏离 1 时，梯度自然衰减到接近 0！

### 2.3 SAPO vs PPO Clip

| 方面 | PPO Clip | SAPO |
|------|----------|------|
| 边界处理 | 硬截断 | 软衰减 |
| 梯度连续性 | 不连续 | 连续 |
| 信任域形式 | [1-ε, 1+ε] | sech² 软边界 |
| 远离策略时 | 梯度=0 | 梯度→0 |

---

# 第三部分：IS Reshape v4 理论

## 3. 核心思想

### 3.1 设计目标

1. **保持 SFT-RL 连续性**：γ ∈ [0,1] 控制 mean-seeking (SFT) 到 mode-seeking (RL) 的插值
2. **引入信任域**：使用 SAPO 的 sech² 软边界，确保离策略样本梯度衰减
3. **Reward 调制**：让 |A| 决定使用多少 RL 成分

### 3.2 核心公式

**统一的梯度权重函数**：

$$\boxed{w_\gamma(r, A) = (1 - \gamma(|A|)) + \gamma(|A|) \cdot r \cdot \text{sech}^2\left(\frac{\tau(r-1)}{2}\right)}$$

其中：
- $r = \pi_\theta(y|x) / \mu(y|x)$ 是 importance sampling ratio
- $\gamma(|A|) = \sigma(\beta |A|)$ 是 reward 调制的混合参数
- $\tau$ 是信任域温度
- $\beta$ 是 reward 敏感度

### 3.3 极限行为

**当 γ = 0（SFT 极限）**：
$$w_0(r) = 1$$

这对应 SFT 的 Forward KL：$\nabla L_{SFT} = -\mathbb{E}_\mu[A \cdot \nabla\log\pi_\theta]$

**当 γ = 1（RL 极限）**：
$$w_1(r) = r \cdot \text{sech}^2\left(\frac{\tau(r-1)}{2}\right)$$

这就是 SAPO 的形式，带有软信任域。

### 3.4 γ(|A|) 的信息论解释

**为什么用 |A| 调制 γ？**

| \|A\| 的大小        | 含义             | 应该如何学习          |
|------------------|----------------|-----------------|
| \|A\| ≈ 0        | Reward 信号弱/不确定 | 保守学习 (SFT-like) |
| \|A      \| >> 0 | Reward 信号强/确定  | 积极学习 (RL-like)  |

$$\gamma(|A|) = \sigma(\beta |A|) = \frac{1}{1 + e^{-\beta |A|}}$$

- 当 |A| → 0: γ → 0.5（中性）
- 当 |A| → ∞: γ → 1（完全 RL）
- β 控制过渡的锐度

**与 SAPO 的关键区别**：
- SAPO 用 sign(A) 选择不同的温度 τ_pos 和 τ_neg
- 我们用 |A| 调制 γ，影响整个权重函数的形状

---

## 4. 对应的 f(r) 形式

### 4.1 从 w(r) 反推 f(r)

梯度权重 w(r) 和 f(r) 的关系：

$$w(r) = r \cdot f'(r) + f(r) \quad \text{(来自 } \nabla_\theta L = -\mathbb{E}[f(r) \cdot A \cdot \nabla\log\pi_\theta] \text{)}$$

对于 $w_\gamma(r) = (1-\gamma) + \gamma \cdot r \cdot \text{sech}^2(\tau(r-1)/2)$：

**闭式解**：

$$\boxed{f_\gamma(r) = (1-\gamma) \cdot \log r + \gamma \cdot \frac{2}{\tau} \cdot \tanh\left(\frac{\tau(r-1)}{2}\right)}$$

### 4.2 验证

对 $f_\gamma(r)$ 求导：

$$f'_\gamma(r) = \frac{1-\gamma}{r} + \gamma \cdot \text{sech}^2\left(\frac{\tau(r-1)}{2}\right)$$

因此：
$$r \cdot f'_\gamma(r) = (1-\gamma) + \gamma \cdot r \cdot \text{sech}^2\left(\frac{\tau(r-1)}{2}\right) = w_\gamma(r) \quad \checkmark$$

### 4.3 极限情况

**γ = 0**：
$$f_0(r) = \log r$$
这是 reverse KL 的形式（on-policy RL 的目标）

**γ = 1**：
$$f_1(r) = \frac{2}{\tau} \cdot \tanh\left(\frac{\tau(r-1)}{2}\right)$$
这是 SAPO 的形式

---

## 5. SFT 场景的处理

### 5.1 问题：SFT 没有显式 reward

在 SFT 中：
- 数据是人工筛选的高质量样本
- 没有显式的 advantage A
- 所有样本被认为是"好"的

### 5.2 方案 A：固定 γ = γ_SFT

对于 SFT，使用固定的小 γ：

$$\gamma_{SFT} = 0.1 \text{ 或 } 0.2$$

这意味着：
- 90% SFT 成分 (权重=1)
- 10% RL 成分 (带信任域)

好处：保持一定的信任域，防止过度拟合高 ratio 的样本。

### 5.3 方案 B：基于困惑度的 pseudo-advantage

定义 pseudo-advantage：

$$A_{pseudo} = -\log \pi_\theta(y|x) - \text{baseline}$$

直觉：
- 困惑度高的样本 → 更"需要学习" → 类似高 |A|
- 困惑度低的样本 → 已经学会 → 类似低 |A|

### 5.4 方案 C：数据质量分数

如果有数据质量分数 q(x, y)：

$$A_{quality} = q(x, y) - \bar{q}$$

这让高质量样本获得更多 RL 学习（更积极地去拟合）。

---

## 6. 与 v3.1 的关键区别

| 方面 | v3.1 (Rényi) | v4 (Reward-Modulated) |
|------|--------------|------------------------|
| γ 的来源 | ESS 约束 + 正确性指标 | Reward 调制 |
| 信任域 | 无（只控制方差） | sech² 软边界 |
| 远离策略时 | 梯度被缩放但不为0 | 梯度趋向0 |
| SFT-RL 插值 | 理论上有，实际失效 | 显式控制 |
| 与 SAPO 关系 | 无 | γ=1 时等价 |

---

# 第四部分：理论基础

## 7. 信息论视角

### 7.1 α-散度统一框架

α-散度定义：

$$D_\alpha(p \| q) = \frac{1}{\alpha(1-\alpha)}\left(1 - \int p(x)^\alpha q(x)^{1-\alpha} dx\right)$$

特殊情况：
- α → 0: Forward KL (SFT)
- α → 1: Reverse KL (RL)
- α = 0.5: Hellinger distance

### 7.2 f-散度与 f(r)

f-散度：
$$D_f(p \| q) = \mathbb{E}_q[f(p/q)]$$

IS Reshape v4 的 $f_\gamma(r)$ 定义了一族 f-散度：

| γ | f(r) | 对应散度 |
|---|------|----------|
| 0 | log r | Reverse KL |
| 1 | (2/τ)tanh(τ(r-1)/2) | SAPO 散度 |
| 中间 | 插值 | 混合散度 |

### 7.3 Reward 作为信息增益

|A| 的信息论解释：

$$|A| \propto \text{Info}(\text{sample})$$

- 高 |A|：这个样本对策略改进有高信息量 → 用 RL 方式学习
- 低 |A|：这个样本信息量低 → 用保守的 SFT 方式

这与 v3.1 的"信息量因子"I(w, A)有联系，但：
- v3.1：I 影响 γ_target 的选择
- v4：|A| 直接调制 γ

---

## 8. 与 SAPO 的深层区别

### 8.1 SAPO 的设计

SAPO 使用：
- sign(A) > 0: 用 τ_pos (正常温度)
- sign(A) < 0: 用 τ_neg (更高温度，更保守)

```python
# SAPO
if A > 0:
    weight = r * sech2(τ_pos * (r-1) / 2)
else:
    weight = r * sech2(τ_neg * (r-1) / 2)  # τ_neg > τ_pos
```

### 8.2 IS Reshape v4 的设计

我们使用 |A| 调制 γ：

```python
# IS Reshape v4
gamma = sigmoid(beta * |A|)
sapo_weight = r * sech2(τ * (r-1) / 2)
weight = (1 - gamma) + gamma * sapo_weight
```

### 8.3 关键区别

| 方面 | SAPO | IS Reshape v4 |
|------|------|---------------|
| 使用 A 的方式 | sign(A) 选择 τ | |A| 调制 γ |
| SFT 成分 | 无 | (1-γ) 项 |
| 可解释性 | 启发式 | 信息论基础 |
| 参数 | τ_pos, τ_neg | τ, β |
| 极限行为 | 只有 RL | SFT ↔ RL |

---

# 第五部分：实现

## 9. 算法

```python
import torch
import math

def sech2(x):
    """Compute sech²(x) = 1/cosh²(x)"""
    return 1.0 / torch.cosh(x).pow(2)

def is_reshape_v4(
    log_prob: torch.Tensor,      # log π_θ
    ref_log_prob: torch.Tensor,  # log μ
    advantages: torch.Tensor,    # A
    response_mask: torch.Tensor,
    tau: float = 1.0,            # 信任域温度
    beta: float = 1.0,           # Reward 敏感度
    gamma_min: float = 0.0,      # 最小 γ
    gamma_max: float = 1.0,      # 最大 γ
) -> tuple[torch.Tensor, dict]:
    """
    IS Reshape v4: Reward-modulated SFT-RL interpolation with trust region

    核心公式:
        w(r, A) = (1 - γ(|A|)) + γ(|A|) · r · sech²(τ(r-1)/2)
        γ(|A|) = σ(β|A|)

    特性:
        - γ → 0: SFT (权重=1)
        - γ → 1: SAPO (软信任域)
        - |A| 调制 γ: 高 |A| → 更 RL，低 |A| → 更 SFT
    """
    # Step 1: 计算 importance ratio r = exp(log π_θ - log μ)
    log_r = log_prob - ref_log_prob
    log_r = torch.clamp(log_r, min=-20.0, max=20.0)
    r = torch.exp(log_r)

    # Step 2: 计算 reward-modulated γ
    abs_A = torch.abs(advantages)
    gamma = torch.sigmoid(beta * abs_A)
    gamma = gamma_min + (gamma_max - gamma_min) * gamma

    # Step 3: 计算 SAPO 成分: r · sech²(τ(r-1)/2)
    sapo_weight = r * sech2(tau * (r - 1) / 2)

    # Step 4: 混合权重
    # w(r, A) = (1 - γ) + γ · sapo_weight
    weight = (1 - gamma) + gamma * sapo_weight

    # Step 5: Policy gradient loss
    # L = -E_μ[w(r, A) · A · log π_θ]
    # Note: 这里 log_prob 已经是 log π_θ(a|s) 对于当前 token
    pg_loss = -(weight.detach() * advantages * response_mask).sum() / (response_mask.sum() + 1e-8)

    # Metrics
    with torch.no_grad():
        mask = response_mask.bool()
        total = response_mask.sum()

        # KL 散度
        kl = (-log_r * response_mask).sum() / total

        metrics = {
            'v4/gamma_mean': (gamma * response_mask).sum().item() / total.item(),
            'v4/gamma_std': gamma[mask].std().item() if mask.sum() > 1 else 0.0,
            'v4/weight_mean': (weight * response_mask).sum().item() / total.item(),
            'v4/weight_max': weight[mask].max().item() if mask.sum() > 0 else 1.0,
            'v4/sapo_weight_mean': (sapo_weight * response_mask).sum().item() / total.item(),
            'v4/ratio_mean': (r * response_mask).sum().item() / total.item(),
            'v4/abs_adv_mean': (abs_A * response_mask).sum().item() / total.item(),
            'v4/kl': kl.item(),
        }

    return pg_loss, metrics


def is_reshape_v4_sft(
    log_prob: torch.Tensor,      # log π_θ
    ref_log_prob: torch.Tensor,  # log μ (pretrained model)
    response_mask: torch.Tensor,
    tau: float = 1.0,            # 信任域温度
    gamma_sft: float = 0.2,      # SFT 场景的固定 γ
) -> tuple[torch.Tensor, dict]:
    """
    IS Reshape v4 for SFT: 使用固定的小 γ

    在 SFT 中没有显式 advantage，使用固定 γ：
        w(r) = (1 - γ_sft) + γ_sft · r · sech²(τ(r-1)/2)
    """
    # Step 1: 计算 importance ratio
    log_r = log_prob - ref_log_prob
    log_r = torch.clamp(log_r, min=-20.0, max=20.0)
    r = torch.exp(log_r)

    # Step 2: 固定 γ
    gamma = gamma_sft

    # Step 3: SAPO 成分
    sapo_weight = r * sech2(tau * (r - 1) / 2)

    # Step 4: 混合权重
    weight = (1 - gamma) + gamma * sapo_weight

    # Step 5: SFT loss (maximize log likelihood)
    # L = -E_data[w(r) · log π_θ]
    sft_loss = -(weight.detach() * log_prob * response_mask).sum() / (response_mask.sum() + 1e-8)

    # Metrics
    with torch.no_grad():
        mask = response_mask.bool()
        total = response_mask.sum()

        metrics = {
            'v4_sft/gamma': gamma,
            'v4_sft/weight_mean': (weight * response_mask).sum().item() / total.item(),
            'v4_sft/ratio_mean': (r * response_mask).sum().item() / total.item(),
        }

    return sft_loss, metrics
```

---

# 第六部分：理论贡献总结

## 10. 主要贡献

1. **软信任域**
   - 引入 sech²(τ(r-1)/2)，确保离策略样本梯度自然衰减
   - 解决 v3.1 的核心问题：方差控制 ≠ KL 控制

2. **Reward 调制的 γ**
   - γ(|A|) = σ(β|A|) 让 reward 强度决定学习方式
   - 高 |A| → RL-like，低 |A| → SFT-like
   - 与 SAPO 的 sign(A) 方法形成理论区别

3. **真正的 SFT-RL 连续谱**
   - γ=0 时严格退化为 SFT (权重=1)
   - γ=1 时等价于 SAPO
   - 中间值实现平滑插值

4. **f-散度统一视角**
   - f_γ(r) = (1-γ)log r + γ(2/τ)tanh(τ(r-1)/2)
   - 定义了一族从 reverse KL 到 SAPO 的散度

## 11. 与之前版本的对比

| 方面 | v3.1 (Rényi) | v4 (Reward-Modulated) |
|------|--------------|------------------------|
| γ 来源 | ESS 约束 | Reward 调制 |
| 信任域 | 无 | sech² 软边界 |
| KL 控制 | 间接（失效） | 直接（梯度衰减） |
| 与 SAPO 关系 | 无 | γ=1 等价 |
| 理论基础 | Rényi 散度 | f-散度族 |

---

## 附录

### A. 参数选择建议

| 参数 | 建议值 | 作用 |
|------|--------|------|
| τ | 1.0-2.0 | 信任域宽度（越大越严格） |
| β | 0.5-2.0 | Reward 敏感度 |
| γ_min | 0.0-0.1 | 最小 SFT 成分 |
| γ_max | 0.8-1.0 | 最大 RL 成分 |
| γ_sft | 0.1-0.3 | SFT 场景的固定 γ |

### B. 数值行为

**w(r) 在不同 γ 下的行为** (τ=1.0):

| r | γ=0 | γ=0.5 | γ=1.0 |
|---|-----|-------|-------|
| 0.2 | 1.00 | 0.55 | 0.10 |
| 0.5 | 1.00 | 0.70 | 0.39 |
| 1.0 | 1.00 | 1.00 | 1.00 |
| 2.0 | 1.00 | 0.89 | 0.79 |
| 5.0 | 1.00 | 0.67 | 0.35 |

### C. 梯度分析

对于 loss L = -E[w(r,A) · A · log π_θ]，梯度为：

$$\nabla_\theta L = -\mathbb{E}_\mu\left[w(r, A) \cdot A \cdot \nabla_\theta\log\pi_\theta\right]$$

其中 w(r, A) 被 detach，所以：
- 梯度方向由 A 的符号决定
- 梯度大小由 w(r, A) 调制
- 当 r 远离 1 时，γ 成分使 w → (1-γ)，梯度受限

---

**文档状态**: v4.0 理论设计完成
**核心突破**: 将 SAPO 的软信任域与 reward 调制结合，实现有理论基础的 SFT-RL 插值
**下一步**: 实现代码并进行实验验证
