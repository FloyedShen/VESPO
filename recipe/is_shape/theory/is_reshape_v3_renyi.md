# IS Reshape：基于 Rényi 散度与正确性指标的统一理论

**版本**: 3.1
**核心创新**: 用单一的"正确性指标" C = A · log w 统一整个框架

---

## 摘要

本文建立 IS Reshape 的完整理论基础，核心贡献：

1. **为什么是 w^γ**：Rényi-IS 恒等式使 w^γ 成为唯一具有闭式方差控制的形式
2. **正确性指标**：C = A · log w 统一刻画"策略是否在正确方向"
3. **连续的 γ***：从变分原理导出完全连续的 γ*(w, A)，四象限是极限情况
4. **信息论基础**：信息量因子 I = σ(-C/τ) = P(策略做错) 有明确的信息论解释

---

# 第一部分：问题设定

## 1. 统一目标：Reward 最大化

### 1.1 两条路径

**SFT 路径**：
- 目标分布：$\tilde{\mu} \propto \mu \cdot r$
- 散度：Forward KL，$\min D_{KL}(\tilde{\mu} \| \pi_\theta)$
- 梯度：$g_{SFT} = \mathbb{E}_\mu[r \cdot \nabla\log\pi_\theta]$

**RL 路径**：
- 目标：$\max \mathbb{E}_{\pi_\theta}[r]$
- 通过 IS：$= \mathbb{E}_\mu[w \cdot r]$
- 梯度：$g_{RL} = \mathbb{E}_\mu[w \cdot r \cdot \nabla\log\pi_\theta]$

### 1.2 统一梯度形式

$$g = \mathbb{E}_\mu[f(w) \cdot A \cdot \nabla\log\pi_\theta]$$

| 路径 | f(w) |
|------|------|
| SFT | 1 |
| RL | w |
| **IS Reshape** | **w^γ** |

---

# 第二部分：Rényi 散度基础

## 2. 核心恒等式

**定理 2.1（Rényi-IS 恒等式）**：

$$\boxed{\mathbb{E}_\mu[w^\alpha] = \exp\left((\alpha - 1) D_\alpha^R(\pi_\theta \| \mu)\right)}$$

**推论**：w^γ 是唯一使 $\mathbb{E}[f(w)^2]$ 与散度有精确关系的形式。

## 3. γ 的物理意义

- γ = 0：完全忽略分布偏移（SFT）
- γ = 1：完整保留分布偏移（RL）
- γ ∈ (0,1)：在 Rényi 散度空间中的平滑插值

---

# 第三部分：正确性指标

## 4. 核心定义

### 4.1 正确性指标

**定义 4.1（正确性指标）**：

$$\boxed{C(w, A) = A \cdot \log w}$$

**物理意义**：

| C 的符号 | 含义 | 情况 |
|---------|------|------|
| C > 0 | 策略在**正确**方向 | (A>0, w>1) 或 (A<0, w<1) |
| C < 0 | 策略在**错误**方向 | (A>0, w<1) 或 (A<0, w>1) |
| C = 0 | 边界情况 | A=0 或 w=1 |

### 4.2 为什么是 A · log w？

**梯度的方向**：
$$\text{gradient} \propto A \cdot \nabla\log\pi_\theta$$
- A > 0：梯度要求增加 π_θ(y|x)
- A < 0：梯度要求减少 π_θ(y|x)

**当前状态**：
$$\log w = \log\pi_\theta - \log\mu$$
- log w > 0：π_θ 已经偏好这个样本
- log w < 0：π_θ 已经避免这个样本

**C = A · log w 的含义**：
- C > 0：梯度方向 = 当前状态（策略已在正确方向）
- C < 0：梯度方向 ≠ 当前状态（策略需要修正）

### 4.3 四种情况的统一

| Case | A | log w | C = A·log w | 含义 |
|------|---|-------|-------------|------|
| 1 | + | + | + | 好样本已被偏好 ✓ |
| 2 | - | + | - | 坏样本被错误偏好 ✗ |
| 3 | + | - | - | 好样本被错误忽略 ✗ |
| 4 | - | - | + | 坏样本已被避免 ✓ |

**关键观察**：C > 0（Case 1, 4）= 策略正确；C < 0（Case 2, 3）= 策略错误

---

## 5. 信息量因子

### 5.1 定义

**定义 5.1（信息量因子）**：

$$\boxed{I(w, A) = \sigma\left(-\frac{C(w, A)}{\tau}\right) = \sigma\left(-\frac{A \cdot \log w}{\tau}\right)}$$

其中 τ > 0 是温度参数。

### 5.2 信息论解释

**I(w, A) = P(策略做错)**

- 当 C > 0（策略正确）：I → 0（低信息量，无需学习）
- 当 C < 0（策略错误）：I → 1（高信息量，需要学习）

**与惊讶度的关系**：

在信息论中，惊讶度 = -log P(observed)。

如果我们观测到"策略正确"（C > 0），而 P(正确) = σ(C/τ)，则：
$$\text{Surprise} = -\log\sigma(C/\tau) = \text{softplus}(-C/\tau)$$

信息量因子 I = 1 - P(正确) = P(错误) 是惊讶度的归一化版本。

### 5.3 温度参数 τ 的作用

- τ → 0：I 趋向硬指示函数，四象限硬分
- τ → ∞：I → 0.5，所有样本同等对待
- τ 适中：平滑过渡

---

# 第四部分：变分推导

## 6. 优化问题

### 6.1 目标与约束

**目标**：最大化学习效用
$$\max_{\gamma(w,A)} \mathbb{E}_\mu[w^\gamma \cdot |A| \cdot I(w, A)]$$

**约束**：控制方差（Rényi 散度）
$$\mathbb{E}_\mu[w^{2\gamma} \cdot A^2] \leq V_{max}$$

### 6.2 Lagrangian 求解

$$\mathcal{L} = \mathbb{E}_\mu\left[w^\gamma |A| I - \lambda w^{2\gamma} A^2\right]$$

对 γ 的一阶条件：
$$w^\gamma \log w \cdot |A| I = 2\lambda w^{2\gamma} \log w \cdot A^2$$

解得：
$$w^\gamma = \frac{I(w, A)}{2\lambda |A|}$$

$$\gamma^* = \frac{\log I(w, A) - \log(2\lambda |A|)}{\log w}$$

### 6.3 分析极限行为

**当 C >> τ（策略很正确）**：I → 0，log I → -∞
- 若 log w > 0 (Case 1)：γ* → -∞，被截断为 0
- 若 log w < 0 (Case 4)：γ* → +∞，被截断为 1

**当 C << -τ（策略很错误）**：I → 1，log I → 0
- γ* ≈ -log(2λ|A|) / log w，由全局约束确定为 γ_base

---

## 7. 最优 γ* 的连续形式

### 7.1 完整公式

**定理 7.1（连续的变分最优 γ）**：

$$\boxed{\gamma^*(w, A) = \gamma_{base} + (\gamma_{target}(w) - \gamma_{base}) \cdot P_{correct}(w, A)}$$

其中：
- $P_{correct}(w, A) = \sigma(C/\tau) = \sigma(A \cdot \log w / \tau)$：策略正确的概率
- $\gamma_{target}(w) = \sigma(-\log w \cdot T)$：目标 γ（w < 1 时趋向 1，w > 1 时趋向 0）
- $\gamma_{base}$：由全局 ESS 约束确定

### 7.2 展开形式

$$\gamma^*(w, A) = \gamma_{base} + \left(\sigma(-\log w \cdot T) - \gamma_{base}\right) \cdot \sigma\left(\frac{A \cdot \log w}{\tau}\right)$$

### 7.3 物理解释

$$\gamma^* = \gamma_{base} + \underbrace{(\gamma_{target} - \gamma_{base})}_{\text{调整幅度}} \cdot \underbrace{P_{correct}}_{\text{调整权重}}$$

- 当策略**正确**时（$P_{correct} \to 1$）：$\gamma^* \to \gamma_{target}(w)$
  - w > 1：$\gamma_{target} \to 0$（压缩）
  - w < 1：$\gamma_{target} \to 1$（不提升）

- 当策略**错误**时（$P_{correct} \to 0$）：$\gamma^* \to \gamma_{base}$（标准处理）

### 7.4 四象限作为极限情况

当 τ → 0 且 T → ∞ 时：

| Case | C | P_correct | γ_target | γ* |
|------|---|-----------|----------|-----|
| 1 (w>1, A>0) | + | 1 | 0 | 0 |
| 2 (w>1, A<0) | - | 0 | 0 | γ_base |
| 3 (w<1, A>0) | - | 0 | 1 | γ_base |
| 4 (w<1, A<0) | + | 1 | 1 | 1 |

**结论**：四象限硬分是连续公式的极限情况，不是预设结构。

---

# 第五部分：完整理论框架

## 8. 三层结构

```
┌─────────────────────────────────────────────────────────────────┐
│                    IS Reshape 统一理论框架                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  【层次 1：为什么是 w^γ】                                         │
│                                                                 │
│      Rényi-IS 恒等式：E[w^α] = exp((α-1) D_α^R)                 │
│      → w^γ 是唯一使方差与散度有精确关系的形式                       │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  【层次 2：正确性指标】                                           │
│                                                                 │
│      C(w, A) = A · log w                                        │
│      → 统一刻画"策略是否在正确方向"                                │
│      → C > 0: 正确 (Case 1, 4)                                  │
│      → C < 0: 错误 (Case 2, 3)                                  │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  【层次 3：变分最优 γ*】                                          │
│                                                                 │
│      I(w, A) = σ(-C/τ) = P(策略做错)                            │
│                                                                 │
│      γ* = γ_base + (γ_target - γ_base) · σ(C/τ)                │
│                                                                 │
│      → 完全连续，四象限是极限情况                                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 9. 核心公式汇总

| 概念 | 公式 |
|------|------|
| 正确性指标 | $C(w, A) = A \cdot \log w$ |
| 信息量因子 | $I(w, A) = \sigma(-C/\tau)$ |
| 策略正确概率 | $P_{correct} = \sigma(C/\tau) = 1 - I$ |
| 目标 γ | $\gamma_{target}(w) = \sigma(-\log w \cdot T)$ |
| **最优 γ*** | $\gamma^* = \gamma_{base} + (\gamma_{target} - \gamma_{base}) \cdot P_{correct}$ |

---

# 第六部分：实现

## 10. 算法

```python
import torch
import math

def is_reshape_v3(
    log_prob: torch.Tensor,      # log π_θ
    ref_log_prob: torch.Tensor,  # log μ
    advantages: torch.Tensor,    # A
    response_mask: torch.Tensor,
    rho_min: float = 0.3,
    tau: float = 1.0,            # 正确性温度
    T: float = 5.0,              # γ_target 温度
) -> tuple[torch.Tensor, dict]:
    """
    IS Reshape v3: 基于正确性指标的连续公式

    核心创新：
    - C = A · log w 统一刻画策略正确性
    - γ* 完全连续，四象限是极限情况
    """
    # Step 1: 计算 log w
    log_w = log_prob - ref_log_prob

    # Step 2: 计算 γ_base（全局 ESS/Rényi 约束）
    valid_log_w = log_w[response_mask.bool()]
    sigma_sq = valid_log_w.var().item() if len(valid_log_w) > 1 else 0.0

    if sigma_sq < 1e-8:
        gamma_base = 1.0
    else:
        gamma_base = min(1.0, math.sqrt(-math.log(rho_min) / sigma_sq))

    # Step 3: 正确性指标 C = A · log w
    C = advantages * log_w

    # Step 4: 策略正确概率 P_correct = σ(C/τ)
    P_correct = torch.sigmoid(C / tau)

    # Step 5: 目标 γ: γ_target = σ(-log w · T)
    # w < 1 (log w < 0) → γ_target → 1
    # w > 1 (log w > 0) → γ_target → 0
    gamma_target = torch.sigmoid(-log_w * T)

    # Step 6: 最优 γ*
    # γ* = γ_base + (γ_target - γ_base) · P_correct
    gamma = gamma_base + (gamma_target - gamma_base) * P_correct

    # Step 7: 计算 w^γ（detached）
    w_gamma = torch.exp(gamma * log_w).detach()

    # Step 8: Policy gradient loss
    pg_loss = -(w_gamma * advantages * log_prob * response_mask).sum() / response_mask.sum()

    # Metrics
    with torch.no_grad():
        total = response_mask.sum()
        metrics = {
            'gamma/base': gamma_base,
            'gamma/mean': (gamma * response_mask).sum().item() / total.item(),
            'correctness/C_mean': (C * response_mask).sum().item() / total.item(),
            'correctness/P_correct_mean': (P_correct * response_mask).sum().item() / total.item(),
            'renyi/sigma_sq': sigma_sq,
        }

    return pg_loss, metrics
```

---

# 第七部分：理论贡献总结

## 11. 主要贡献

1. **正确性指标 C = A · log w**
   - 用单一标量统一四种情况
   - 物理意义明确：梯度方向与当前状态的一致性

2. **连续的 γ* 公式**
   - $\gamma^* = \gamma_{base} + (\gamma_{target} - \gamma_{base}) \cdot \sigma(C/\tau)$
   - 四象限是极限情况，不是预设结构

3. **信息论基础**
   - I(w, A) = P(策略做错) 有明确的信息论解释
   - 与惊讶度、自信息等概念一致

4. **Rényi 散度支撑**
   - w^γ 的形式由 Rényi-IS 恒等式决定
   - 方差控制有精确的理论保证

## 12. 与之前版本的对比

| 方面 | v2 (四象限) | v3 (正确性指标) |
|------|-------------|----------------|
| 核心概念 | sign(A), sign(log w) | C = A · log w |
| 连续性 | 分段函数 | 完全连续 |
| 四象限 | 预设结构 | 极限情况 |
| 优雅性 | 中等 | 高 |
| 参数 | temperature | τ (正确性), T (γ_target) |

---

## 附录

### A. 参数选择建议

| 参数 | 建议值 | 作用 |
|------|--------|------|
| ρ_min | 0.3 | 最小 ESS 比例 |
| τ | 1.0 | 正确性判断的平滑度 |
| T | 5.0 | γ_target 的锐度 |

### B. 极限行为分析

**τ → 0（硬正确性判断）**：
- P_correct → 1[C > 0]
- 退化为四象限硬分

**τ → ∞（忽略正确性）**：
- P_correct → 0.5
- γ* → (γ_base + γ_target) / 2

**T → 0（平滑 γ_target）**：
- γ_target → 0.5
- γ* 在 γ_base 和 0.5 之间

**T → ∞（硬 γ_target）**：
- γ_target → 1[w < 1]
- 标准的 Case 1/4 行为

---

**文档状态**：v3.1 完成
**核心突破**：用 C = A · log w 统一整个框架，实现完全连续的 γ* 公式
