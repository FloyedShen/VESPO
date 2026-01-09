# IS Reshape v3.1 实验报告

**日期**: 2025-12-15
**版本**: v3.1 (is_reshape_renyi)
**状态**: 发现训练稳定性问题

---

## 1. 方法概述

### 1.1 核心理论

IS Reshape v3.1 基于 Rényi 散度和"正确性指标"构建统一框架：

**正确性指标**:
$$C(w, A) = A \cdot \log w$$

- $C > 0$: 策略在正确方向（好样本被偏好，或坏样本被避免）
- $C < 0$: 策略在错误方向（需要修正）

**Per-sample γ 公式**:
$$\gamma^*(w, A) = \gamma_{base} + (\gamma_{target} - \gamma_{base}) \cdot P_{correct}$$

其中：
- $\gamma_{base} = \min(1.0, \sqrt{-\log \rho_{min} / \sigma^2})$：全局 ESS 约束
- $\gamma_{target} = \sigma(-\log w \cdot T)$：目标 γ
- $P_{correct} = \sigma(C / \tau)$：策略正确的概率

**IS Reshape 权重**:
$$w^\gamma = \exp(\gamma \cdot \log w)$$

### 1.2 设计意图

| γ 值 | 行为 | 对应方法 |
|------|------|----------|
| γ = 0 | 完全忽略分布偏移 | SFT (Forward KL) |
| γ = 1 | 完整保留分布偏移 | RL (Reverse KL) |
| γ ∈ (0,1) | 平滑插值 | IS Reshape |

通过自适应 γ 在 SFT 和 RL 之间平滑过渡，期望获得：
- 比 SFT 更好的 reward 优化能力
- 比 RL 更好的训练稳定性

### 1.3 实现

代码位置: `recipe/is_shape/code/core_algos.py`

```python
@register_policy_loss("is_reshape_renyi")
def compute_policy_loss_is_reshape_renyi(...):
    # Step 1: log w = log π_θ - log μ
    log_w = log_prob - old_log_prob

    # Step 2: γ_base from ESS constraint
    sigma_sq = torch.var(log_w[mask], unbiased=True)
    gamma_base = min(1.0, sqrt(-log(rho_min) / sigma_sq))

    # Step 3: C = A · log w
    C = advantages * log_w

    # Step 4: P_correct = σ(C/τ)
    P_correct = torch.sigmoid(C / tau)

    # Step 5: γ_target = σ(-log w · T)
    gamma_target = torch.sigmoid(-log_w * T)

    # Step 6: γ* (per-sample)
    gamma = gamma_base + (gamma_target - gamma_base) * P_correct

    # Step 7: w^γ (detached)
    w_gamma = torch.exp(gamma.detach() * log_w)

    # Step 8: Loss
    pg_loss = -(w_gamma * advantages * response_mask).sum() / mask.sum()
```

---

## 2. 实验设置

### 2.1 基础配置

- **模型**: Qwen-3-4B-Base
- **任务**: 数学推理 (AMC23, AIME25)
- **PPO epochs**: 16 (每个 batch 更新 16 次)
- **Batch size**: 2048 samples

### 2.2 对比方法

| 方法 | loss_mode | 关键参数 |
|------|-----------|----------|
| is_reshape_renyi | is_reshape_renyi | rho_min=0.3, tau=1.0, T=5.0 |
| clip | clip | clip_ratio=0.2 |
| noclip | vanilla | - |

### 2.3 实验 wandb 路径

```
is_reshape_renyi: run-20251214_193551-qq5rhinn
clip:             run-20251214_193559-p8l9sdpw
noclip:           run-20251214_193604-cs7hx6ub
```

---

## 3. 实验结果

### 3.1 训练曲线概览

**is_reshape_renyi**:
- Step 1-80: KL 缓慢增长，从 0 到 0.068
- Step 80-96: KL 加速增长，从 0.068 到 0.59
- Step 97+: 采样开始异常，response_length 从 1100 跳到 4000
- Step 103+: KL 爆炸到 5+，出现 mode collapse
- Step 113: response_length = 13854 (完全崩溃)

**clip**:
- 整个训练过程 KL 保持在 0.001-0.02
- 训练稳定，无 mode collapse

**noclip**:
- KL 比 clip 略高，在 0.004-0.03
- 但远低于 is_reshape_renyi
- 训练基本稳定

### 3.2 关键指标对比 (Step 80)

| 指标 | is_reshape_renyi | noclip | clip |
|------|------------------|--------|------|
| **ppo_kl** | **0.0683** | 0.0217 | 0.0037 |
| is_ratio_orig | 1.013 | 1.047 | 0.999 |
| KL 相对 clip | **18.2x** | 5.8x | 1.0x |

Step 80 时 is_reshape_renyi 的 KL 已经比 clip 高 18 倍！

### 3.3 is_reshape_renyi 详细演化

```
Step    ppo_kl    sigma_sq    gamma_base    is_ratio_orig    is_ratio_used
────────────────────────────────────────────────────────────────────────────
  1     0.0000    0.0000      1.000         1.00             1.00
 30     0.0023    0.012       1.000         1.00             0.999
 50     0.0058    0.029       1.000         1.00             0.998
 70     0.0096    0.051       1.000         1.00             0.996
 80     0.0683    0.480       1.000         1.01             0.985    ← 问题前兆
 90     0.0617    0.415       0.998         1.00             0.985    ← gamma_base 首次下降
 95     0.2706    1.670       0.890         0.97             0.941
 96     0.5911    3.996       0.792         0.94             0.903
────────────────────────────────────────────────────────────────────────────
 97     0.0000    0.000       1.000         1.00             1.00     ← 新 batch (resp_len=4013!)
 98     0.0333    0.156       1.000         116              0.987    ← is_ratio 爆炸
103     0.8400    5.180       0.557         488              0.797    ← KL < 1 但 ratio=488
110     5.0813   27.015       0.219         488              0.384    ← KL=5, w^γ 仍受控
────────────────────────────────────────────────────────────────────────────
113     0.0000    0.000       1.000         1.00             1.00     ← 新 batch (resp_len=13854!)
117     6.6569   22.441       0.236         254              0.204    ← Mode collapse
```

### 3.4 关键观察

1. **w^γ 控制有效**: 即使 is_ratio_orig=488，is_ratio_used 仍被压到 0.38

2. **但 KL 失控**: w^γ 压缩没有阻止 KL 累积到 5+

3. **gamma_base 反应太慢**: 在 step 90 才首次下降，此时 KL 已经很高

4. **崩溃时 response_length 异常**: Step 97 和 113 的 response_length 显示策略已变质

---

## 4. 问题分析

### 4.1 根本原因：控制目标错配

**IS Reshape 的设计目标**：
- 控制 IS 权重的方差: $\text{Var}(w^\gamma) \leq V_{max}$
- 通过 ESS ratio 约束: $\text{ESS}/n \geq \rho_{min}$

**实际需要控制的**：
- KL 散度: $D_{KL}(\pi_\theta \| \mu) \leq \epsilon$
- 防止策略过度偏离采样分布

### 4.2 数学解释

**PPO clip 的效果**:
```
clip(w, 1-ε, 1+ε) 直接限制:
  log(1-ε) ≤ log π_θ - log μ ≤ log(1+ε)

即每个 token 的 KL 贡献被限制在 [-0.22, 0.18] (ε=0.2)
```

**IS Reshape 的效果**:
```
w^γ = exp(γ · log w)

当 γ=0.22, log w = 6.19 (对应 w=488):
  w^γ = exp(0.22 × 6.19) = exp(1.36) = 3.9

虽然 w^γ 被压缩了（从 488 到 3.9），但这只影响梯度大小，
不改变策略实际偏移量 log w = 6.19！
```

### 4.3 累积效应

```
每一步:
  Step N:   KL += 0.01 (w^γ 被压缩，但策略仍在更新)
  Step N+1: KL += 0.01
  ...
  Step N+50: 累积 KL = 0.5

IS Reshape 的 w^γ 只影响每步的"力度"，
不影响每步的"方向"和"是否执行"。
```

### 4.4 为什么 noclip 比 is_reshape_renyi 更稳定?

看 step 80 的 is_ratio:
- noclip: 1.047 (策略轻微偏移)
- is_reshape_renyi: 1.013 (表面上偏移更小)

但 KL:
- noclip: 0.022
- is_reshape_renyi: 0.068 (3x 更高!)

这说明 is_reshape_renyi 的 **有效学习率更高**。虽然 w^γ < w，但 loss 公式中没有 PPO 那样的保守更新机制。

### 4.5 深入分析：IS Reshape 真正控制的是什么？

#### 4.5.1 梯度公式推导

**IS Reshape v3.1 的梯度**:
```python
log_w = log_prob - old_log_prob          # log_prob 需要梯度
w_gamma = exp(gamma_detached * log_w)    # gamma 被 detach
loss = -(w_gamma * A).mean()
```

对 `log_prob` 求导：
$$\frac{\partial \text{loss}}{\partial \log \pi_\theta} = -A \cdot \frac{\partial w^\gamma}{\partial \log w} = -A \cdot \gamma \cdot w^\gamma$$

所以**实际梯度**是：
$$\nabla L = -\mathbb{E}[\gamma \cdot w^\gamma \cdot A \cdot \nabla \log \pi_\theta]$$

**对比 Vanilla (noclip) 的梯度**：
$$\nabla L = -\mathbb{E}[w \cdot A \cdot \nabla \log \pi_\theta]$$

#### 4.5.2 Grad Norm 对比

| Step | is_reshape grad_norm | noclip grad_norm | 比值 |
|------|---------------------|------------------|------|
| 70 | 0.074 | 0.402 | 0.18 |
| 71 | 0.075 | 8.724 | 0.009 |
| 74 | 0.067 | 4931.9 | 0.00001 |
| 75 | 0.071 | 4616.3 | 0.00002 |
| 80 | 0.088 | 1.986 | 0.044 |

**关键发现**：
- is_reshape 的 grad_norm 比 noclip 小 **10-100000 倍**
- noclip 有严重的梯度爆炸（step 74-75 达到 4000+）
- 但 is_reshape 的 KL 却更大！

#### 4.5.3 IS Reshape 真正控制的是梯度估计的方差

在 Importance Sampling 中，我们用 μ 的样本来估计 π_θ 的期望：
$$\hat{\mathbb{E}}[f] = \frac{1}{n} \sum_i w_i \cdot f(x_i), \quad w_i = \frac{\pi_\theta(x_i)}{\mu(x_i)}$$

这个估计器的**方差**是：
$$\text{Var}(\hat{\mathbb{E}}[f]) \approx \frac{1}{n} \cdot \mathbb{E}_\mu[(w \cdot f)^2]$$

当 w 变化很大时，方差会爆炸！

IS Reshape 用 $w^\gamma$ 替代 $w$：
$$\text{Var}(\hat{\mathbb{E}}_\gamma[f]) \approx \frac{1}{n} \cdot \mathbb{E}_\mu[(w^\gamma \cdot f)^2]$$

由于 $w^\gamma < w$ (当 γ < 1, w > 1)，**方差更小**。

数值验证（γ 对方差和均值的影响）：
| γ | Var(w^γ · f) | Mean(\|w^γ · f\|) |
|---|-------------|------------------|
| 1.00 | 2.07 | 0.93 |
| 0.75 | 1.41 | 0.87 |
| 0.50 | 1.16 | 0.83 |
| 0.25 | 1.04 | 0.81 |

**关键观察**：γ 降低了方差（从 2.07 到 1.04），但梯度的平均幅度没有大幅减少（从 0.93 到 0.81）！

#### 4.5.4 为什么低方差反而导致 KL 增长更快？

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                     低方差 ≠ 小梯度 ≠ 稳定训练                                 │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Noclip (高方差):                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐     │
│  │  Epoch 1:  ████████████░░░░░░░░░░░░░░░░░░░░░  (大梯度)               │     │
│  │  Epoch 2:  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  (梯度爆炸，Adam 抑制)   │     │
│  │  Epoch 3:  ███████░░░░░░░░░░░░░░░░░░░░░░░░░░  (中等梯度)             │     │
│  │  Epoch 4:  ░░████████░░░░░░░░░░░░░░░░░░░░░░░  (部分抵消)             │     │
│  │  ...                                                                 │     │
│  │  累积 KL: 小 (噪声相互抵消，Adam 抑制大梯度)                            │     │
│  └─────────────────────────────────────────────────────────────────────┘     │
│                                                                              │
│  IS Reshape (低方差):                                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐     │
│  │  Epoch 1:  ██████░░░░░░░░░░░░░░░░░░░░░░░░░░░  (一致的梯度)            │     │
│  │  Epoch 2:  ██████░░░░░░░░░░░░░░░░░░░░░░░░░░░  (一致的梯度)            │     │
│  │  Epoch 3:  ██████░░░░░░░░░░░░░░░░░░░░░░░░░░░  (一致的梯度)            │     │
│  │  Epoch 4:  ██████░░░░░░░░░░░░░░░░░░░░░░░░░░░  (一致的梯度)            │     │
│  │  ...                                                                 │     │
│  │  累积 KL: 大 (每步都往同一方向推，优化器信任这些梯度)                     │     │
│  └─────────────────────────────────────────────────────────────────────┘     │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

| 方法 | grad_norm | 方差 | 优化器行为 | 结果 |
|------|-----------|------|------------|------|
| **noclip** | 大 (有爆炸) | 高 | Adam 抑制大梯度 | KL 增长慢但不稳定 |
| **is_reshape** | 小且稳定 | 低 | Adam 信任这些梯度 | KL 稳定增长，累积更大 |

**核心矛盾**：
- IS Reshape 的设计目标：让优化器更信任梯度（低方差）
- 但这恰恰导致：优化器每步都自信地更新，累积漂移更大

#### 4.5.5 与 PPO Clip 的本质区别

**PPO clip 为什么有效？**
- 不是因为它控制方差
- 而是因为它**直接限制每步的 KL 变化**（信任域）
- 当 ratio 超出 [1-ε, 1+ε] 时，**梯度变成 0**

```python
# PPO clip: 超出范围时梯度为 0
if ratio > 1 + eps or ratio < 1 - eps:
    gradient = 0  # 停止更新！
```

**IS Reshape 缺少的**：
- 没有"停止更新"的机制
- 即使 γ 很小，梯度也不会变成 0（除非 γ=0）
- 每个 epoch 都会产生更新，累积起来就失控了

```python
# IS Reshape: 梯度被缩放但永不为 0
gradient = gamma * w_gamma * A  # 即使 gamma=0.1，也不是 0
```

---

## 5. 理论缺陷总结

### 5.1 v3.1 理论的问题

| 理论假设 | 实际情况 |
|----------|----------|
| w^γ 压缩可以控制训练稳定性 | 只控制了方差，没控制 KL |
| γ_base 从 ESS 约束推导 | ESS 约束 ≠ KL 约束 |
| Per-sample γ 可以精细控制 | 没有全局 KL 边界 |

### 5.2 缺失的机制

1. **没有 KL 上界**: 理论中没有 $D_{KL}(\pi_\theta \| \mu) \leq \epsilon$ 的约束

2. **没有信任域**: PPO 的 clip 本质上是信任域，IS Reshape 没有

3. **反应滞后**: gamma_base 依赖 sigma_sq，而 sigma_sq 是滞后指标

---

## 6. 可能的修复方向

### 6.1 方案 A: KL 早停

在每个 PPO epoch 检查 KL，超过阈值立即停止：

```python
for epoch in range(ppo_epochs):
    metrics = update_step(...)
    if metrics['ppo_kl'] > kl_threshold:  # e.g., 0.05
        break
```

优点：简单直接
缺点：可能过早停止，浪费计算

### 6.2 方案 B: 混合 ratio clipping

在 w^γ 基础上额外加 PPO 风格的 clip：

```python
w_gamma = torch.exp(gamma * log_w)
w_gamma_clipped = torch.clamp(w_gamma, 1 - eps, 1 + eps)
loss = -min(w_gamma * A, w_gamma_clipped * A)
```

优点：保持 IS Reshape 的方差控制 + PPO 的信任域
缺点：两套机制可能冲突

### 6.3 方案 C: KL 惩罚项

在 loss 中加入 KL 惩罚：

```python
pg_loss = -(w_gamma * A).mean()
kl_penalty = beta * kl_divergence
total_loss = pg_loss + kl_penalty
```

优点：平滑控制
缺点：需要调节 beta

### 6.4 方案 D: 重新设计 gamma

将 KL 约束直接融入 gamma 的计算：

```python
# 当前: gamma_base = f(sigma_sq, rho_min)
# 新增: gamma_kl = g(current_kl, kl_max)
# 最终: gamma = min(gamma_base, gamma_kl)
```

优点：理论上最优雅
缺点：需要重新推导理论

---

## 7. 结论

IS Reshape v3.1 在理论上提供了 SFT 和 RL 之间的平滑插值，但在实践中存在严重的训练稳定性问题。

### 7.1 核心发现

**IS Reshape 真正控制的是什么？**

| 控制的 | 不控制的 |
|--------|----------|
| 梯度估计的方差 (Variance) | 梯度的大小 (Magnitude) |
| IS 权重的分布 | KL 散度 |
| 单步更新的可靠性 | 累积漂移 |

### 7.2 反直觉的结果

实验数据显示：
- is_reshape 的 grad_norm 比 noclip 小 **10-100000 倍**
- 但 is_reshape 的 KL 增长却**更快**

原因：
1. **低方差 → 优化器更信任梯度**：Adam 对稳定的小梯度会持续执行更新
2. **高方差 → 优化器更谨慎**：Adam 会抑制梯度爆炸，部分更新会相互抵消
3. **结果**：IS Reshape 的一致性梯度累积起来，反而导致更大的策略漂移

### 7.3 与 PPO Clip 的本质区别

| 机制 | PPO Clip | IS Reshape |
|------|----------|------------|
| 控制什么 | KL 散度（信任域） | 梯度方差 |
| 如何控制 | 超出范围时梯度=0 | 缩放梯度但永不为0 |
| 累积效应 | 有上界 | 无上界 |

### 7.4 理论启示

IS Reshape 的理论假设：
> "低方差的梯度估计 → 更稳定的训练"

实际情况：
> "低方差的梯度估计 → 优化器更自信 → 每步都执行更新 → 累积漂移失控"

**关键缺失**：需要在框架中引入明确的 **KL 约束** 或 **信任域** 机制，而不仅仅是方差控制。

---

## 附录

### A. 实验配置文件

```yaml
actor:
  policy_loss:
    loss_mode: "is_reshape_renyi"
    is_reshape_renyi:
      rho_min: 0.3
      tau: 1.0
      T: 5.0
      gamma_min: 0.05
      clip_weight: false
      clip_threshold: 10.0
```

### B. 相关文件

- 理论文档: `recipe/is_shape/theory/is_reshape_v3_renyi.md`
- 实现代码: `recipe/is_shape/code/core_algos.py`
- Worker 代码: `recipe/is_shape/code/fsdp_workers.py`
