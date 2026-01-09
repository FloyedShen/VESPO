# IS-Reshape v11: Cauchy/Arctan 框架 (MSE-Derived Importance Sampling)

**版本**: 11.0

---

## 摘要

本文从加权 MSE 最小化的角度推导出一种基于 **Arctan** 的重要性采样权重函数。核心发现是：**Arctan 不是一个设计选择，而是数学上的必然结果**——它直接源于 MSE 优化框架与 IS 方差特性（Var ∝ w²）的结合。

**核心贡献**：

1. **加权 MSE 框架**：统一偏差-方差权衡，引入学习紧迫度 α(A) 和风险厌恶 β(A)
2. **最优梯度权重**：$\phi^*(w, A) = \frac{w}{1 + \lambda(A) \cdot w^2}$
3. **Arctan 目标函数**：$f(w) = \frac{1}{\sqrt{\lambda}} \arctan(\sqrt{\lambda} \cdot w)$
4. **物理解释**：$\lambda(A) = \frac{\text{Risk Cost}}{\text{Opportunity Cost}}$
5. **指数效用**：$\lambda(A) = \frac{\beta_0}{\alpha_0} e^{-\text{scale} \cdot A}$

**核心公式**：

$$f(w) = \frac{1}{\sqrt{\lambda}} \arctan(\sqrt{\lambda} \cdot w)$$

$$\phi(w) = \frac{w}{1 + \lambda \cdot w^2}$$

---

# 第一部分：问题设定

## 1. Off-Policy 策略梯度估计

### 1.1 基本问题

**目标**：优化策略 $\pi_\theta$ 以最大化期望回报
$$\max_\theta J(\theta) = \mathbb{E}_{\pi_\theta}[A(s,a)]$$

**约束**：只能从行为策略 $\mu$ 采样

**真实梯度**：
$$g^* = \nabla_\theta \mathbb{E}_{\pi_\theta}[A] = \mathbb{E}_{\pi_\theta}[A \cdot \nabla_\theta \log \pi_\theta]$$

### 1.2 重要性采样估计

使用 IS 修正分布偏移：
$$\hat{g}_{IS} = \mathbb{E}_\mu\left[w \cdot A \cdot \nabla_\theta \log \pi_\theta\right]$$

其中 $w = \pi_\theta / \mu$ 是重要性权重。

**问题**：IS 方差 $\propto \mathbb{E}[w^2]$，当 $w$ 波动大时方差爆炸。

### 1.3 一般梯度权重函数

定义一般形式：
$$\hat{g} = \mathbb{E}_\mu[\phi(w, A) \cdot \nabla_\theta \log \pi_\theta]$$

**核心问题**：如何设计最优的 $\phi(w, A)$？

---

# 第二部分：加权 MSE 框架

## 2. 从 MSE 最小化出发

### 2.1 MSE 目标

考虑梯度权重 $\phi$ 的均方误差：
$$\text{MSE}[\phi] = \text{Bias}^2[\phi] + \text{Var}[\phi]$$

### 2.2 关键洞察：IS 方差特性

**定理 2.1**：IS 方差与权重平方成正比
$$\text{Var}[\hat{g}_{IS}] \propto \mathbb{E}_\mu[w^2]$$

**推论**：如果使用权重 $\phi$，则方差项包含 $(w \cdot \phi)^2$。

### 2.3 加权 MSE 目标函数

构造加权 MSE：
$$L(\phi) = \alpha(A) \cdot (w - \phi)^2 + \beta(A) \cdot (w \cdot \phi)^2$$

其中：
- **第一项** $(w - \phi)^2$：偏差项（$\phi$ 偏离真实 IS 权重 $w$ 的程度）
- **第二项** $(w \cdot \phi)^2$：方差项（IS 方差的代理）
- $\alpha(A)$：**学习紧迫度** (Learning Urgency / Opportunity Cost)
- $\beta(A)$：**风险厌恶** (Risk Aversion / Crash Cost)

### 2.4 物理解释

| 系数 | 含义 | A > 0 时 | A < 0 时 |
|------|------|----------|----------|
| $\alpha(A)$ | 学习紧迫度 | 高（想抓住好样本）| 低（不急于惩罚）|
| $\beta(A)$ | 风险厌恶 | 低（愿意冒险学习）| 高（怕方差爆炸）|

---

## 3. 最优梯度权重推导

### 3.1 求解最优 $\phi^*$

对 $L(\phi)$ 求导并令其为零：
$$\frac{\partial L}{\partial \phi} = -2\alpha(A)(w - \phi) + 2\beta(A) w^2 \phi = 0$$

展开：
$$\alpha(A) w - \alpha(A) \phi + \beta(A) w^2 \phi = 0$$

$$\alpha(A) w = \phi \left[\alpha(A) + \beta(A) w^2\right]$$

**定理 3.1（最优梯度权重）**：
$$\boxed{\phi^*(w, A) = \frac{\alpha(A) \cdot w}{\alpha(A) + \beta(A) \cdot w^2} = \frac{w}{1 + \lambda(A) \cdot w^2}}$$

其中 $\lambda(A) = \frac{\beta(A)}{\alpha(A)} = \frac{\text{Risk Cost}}{\text{Opportunity Cost}}$

### 3.2 $\lambda(A)$ 的物理意义

$$\lambda(A) = \frac{\beta(A)}{\alpha(A)} = \frac{\text{风险厌恶}}{\text{学习紧迫度}}$$

| 情况 | $\lambda(A)$ | $\phi^*(w)$ 行为 |
|------|-------------|-----------------|
| A > 0 (好样本) | 小 | $\phi \approx w$（接近 IS，激进学习）|
| A < 0 (坏样本) | 大 | $\phi$ 被截断（保守，控制方差）|

---

## 4. 目标函数推导：Arctan 的必然性

### 4.1 从 $\phi$ 反推 $f(w)$

**问题**：是否存在标量目标函数 $f(w)$ 使得通过 autograd 得到正确的 $\phi$？

**关键关系**：在策略梯度中，如果目标是 $J = \mathbb{E}[f(w) \cdot A]$，则：
$$\frac{\partial J}{\partial \theta} = \mathbb{E}\left[\frac{\partial f}{\partial w} \cdot \frac{\partial w}{\partial \theta} \cdot A\right] = \mathbb{E}\left[f'(w) \cdot w \cdot A \cdot \nabla \log \pi\right]$$

因此：$\phi(w) = f'(w) \cdot w$

### 4.2 求解 $f(w)$

从 $\phi = w \cdot f'(w)$ 出发：
$$f'(w) = \frac{\phi(w)}{w} = \frac{1}{1 + \lambda w^2}$$

积分：
$$f(w) = \int \frac{1}{1 + \lambda w^2} dw$$

**定理 4.1（Arctan 目标函数）**：
$$\boxed{f(w) = \frac{1}{\sqrt{\lambda}} \arctan(\sqrt{\lambda} \cdot w)}$$

### 4.3 验证

$$f'(w) = \frac{1}{\sqrt{\lambda}} \cdot \frac{\sqrt{\lambda}}{1 + \lambda w^2} = \frac{1}{1 + \lambda w^2} \quad \checkmark$$

$$\phi(w) = w \cdot f'(w) = \frac{w}{1 + \lambda w^2} \quad \checkmark$$

### 4.4 Arctan 的必然性

**定理 4.2**：Arctan 不是设计选择，而是数学必然。

**证明**：
1. IS 方差 $\propto w^2$ → MSE 中的方差项必须包含 $w^2$
2. MSE 最优解给出 $\phi^* = \frac{w}{1 + \lambda w^2}$
3. 从 $\phi = w \cdot f'(w)$ 反推 → $f'(w) = \frac{1}{1 + \lambda w^2}$
4. 这个积分的唯一解是 Arctan

$\blacksquare$

---

# 第三部分：指数效用与动态 $\lambda$

## 5. 指数效用假设

### 5.1 动机

希望 $\lambda(A)$ 能够：
- A > 0 时小（激进学习好样本）
- A < 0 时大（保守处理坏样本）

### 5.2 指数效用形式

假设学习紧迫度随 A 指数增长：
$$\alpha(A) = \alpha_0 \cdot e^{\text{scale} \cdot A}$$

风险厌恶保持恒定：
$$\beta(A) = \beta_0$$

则：
$$\boxed{\lambda(A) = \frac{\beta_0}{\alpha_0} \cdot e^{-\text{scale} \cdot A}}$$

### 5.3 行为分析

| A 值 | $e^{-\text{scale} \cdot A}$ | $\lambda(A)$ | 行为 |
|------|---------------------------|-------------|------|
| A >> 0 | → 0 | 小 | $\phi \approx w$（IS 模式）|
| A = 0 | = 1 | $\beta_0/\alpha_0$ | 均衡 |
| A << 0 | → ∞ | 大 | $\phi$ 截断（SFT 模式）|

---

## 6. 四象限分析

### 6.1 完整行为表

| 象限 | w | A | $\lambda$ | $\phi(w)$ | 行为 |
|-----|---|---|-----------|-----------|------|
| I | < 1 | > 0 | 小 | ≈ w | **IS 纠偏**：新好样本，放大权重 |
| II | > 1 | > 0 | 小 | ≈ w/(1+λw²) | **适度学习**：已知好样本，轻微抑制 |
| III | > 1 | < 0 | 大 | ≈ 1/(λw) | **强力惩罚**：未避免坏样本，截断高权重 |
| IV | < 1 | < 0 | 大 | ≈ w | **维持**：已避免坏样本，保持现状 |

### 6.2 可视化

```
                        w >> 1
                          │
              II          │          III
        A > 0, w >> 1     │     A < 0, w >> 1
        λ 小, φ ≈ w       │     λ 大, φ → 1/(λw)
        "适度学习"         │     "强力截断"
                          │
    ──────────────────────┼──────────────────────
                          │
        A > 0, w << 1     │     A < 0, w << 1
        λ 小, φ ≈ w       │     λ 大, φ ≈ w
        "IS 纠偏"          │     "维持现状"
              I           │          IV
                          │
                        w << 1
```

---

# 第四部分：与其他方法的联系

## 7. 方法对比

### 7.1 梯度权重函数对比

| 方法 | $\phi(w)$ | 有界？ | 理论基础 |
|------|-----------|--------|----------|
| IS (RL) | $w$ | ❌ | 无偏估计 |
| SFT | $1$ | ✓ | 忽略分布偏移 |
| PPO clip | $\text{clip}(w, 1\pm\epsilon)$ | ✓ | 信赖域 |
| IS-Reshape v1-v7 | $w^\gamma$ | ❌ | ESS 约束 |
| SAPO | $\sigma(\tau(w-1)) \cdot \frac{4}{\tau}$ | ✓ | 工程设计 |
| Harmonic (v10) | $\frac{w}{w+\lambda}$ | ✓ | Defensive IS / JSD |
| **Cauchy (v11)** | $\frac{w}{1+\lambda w^2}$ | ✓ | **MSE 最优** |

### 7.2 目标函数对比

| 方法 | $f(w)$ | 来源 |
|------|--------|------|
| IS | $\frac{1}{2}w^2$ | 无偏 |
| SFT | $w$ | 忽略偏移 |
| Harmonic (v10) | $\frac{1}{\lambda}\ln(\lambda w + 1)$ | 对数效用 |
| **Cauchy (v11)** | $\frac{1}{\sqrt{\lambda}}\arctan(\sqrt{\lambda} w)$ | **MSE 最优** |

### 7.3 v10 vs v11

| 特性 | v10 Harmonic | v11 Cauchy |
|------|--------------|------------|
| $\phi(w)$ | $\frac{w}{w+\lambda}$ | $\frac{w}{1+\lambda w^2}$ |
| $f(w)$ | $\frac{1}{\lambda}\ln(\lambda w+1)$ | $\frac{1}{\sqrt{\lambda}}\arctan(\sqrt{\lambda} w)$ |
| 大 w 行为 | $\phi \to 1$ | $\phi \to 1/(\lambda w)$ |
| 理论基础 | Defensive IS | MSE 最小化 |
| $\lambda$ | 固定 | 可动态依赖 A |

**关键区别**：
- v10：大 w 时 $\phi \to 1$（饱和）
- v11：大 w 时 $\phi \to 1/(\lambda w)$（衰减更快）

---

# 第五部分：实现

## 8. 核心算法

### 8.1 静态模式（推荐）

```python
# 固定 λ
lambda_ = 1.0  # 超参数

# 计算 IS ratio
log_w = log_prob - old_log_prob
w = torch.exp(torch.clamp(log_w, -10, 10))

# 目标函数 f(w) = (1/√λ) · arctan(√λ · w)
sqrt_lambda = math.sqrt(lambda_)
f_w = torch.atan(sqrt_lambda * w) / sqrt_lambda

# 损失
loss = -f_w * advantages
```

### 8.2 动态模式

```python
# 动态 λ(A) = (β₀/α₀) · exp(-scale · A)
alpha_A = base_urgency * torch.exp(urgency_scale * advantages)
lambda_A = risk_aversion / (alpha_A + 1e-8)
lambda_A = torch.clamp(lambda_A, lambda_min, lambda_max)

# 目标函数
sqrt_lambda = torch.sqrt(lambda_A)
f_w = torch.atan(sqrt_lambda * w) / (sqrt_lambda + 1e-8)

# 损失
loss = -f_w * advantages
```

### 8.3 梯度验证

通过 autograd：
$$\frac{\partial f}{\partial \theta} = f'(w) \cdot \frac{\partial w}{\partial \theta} = \frac{1}{1+\lambda w^2} \cdot w \cdot \nabla \log \pi = \phi(w) \cdot \nabla \log \pi \quad \checkmark$$

---

## 9. 超参数指南

### 9.1 静态模式（1 个超参数）

| λ 值 | 行为 | 适用场景 |
|------|------|----------|
| 0.5 | 激进 | 分布接近、需要快速学习 |
| **1.0** | **均衡** | **默认推荐** |
| 2.0 | 保守 | 分布差异大、早期训练 |

### 9.2 动态模式（5 个超参数）

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `risk_aversion` (β₀) | 1.0 | 风险厌恶程度 |
| `base_urgency` (α₀) | 1.0 | 基础学习紧迫度 |
| `urgency_scale` | 1.0 | A 对 λ 的影响强度 |
| `lambda_min` | 0.01 | λ 下限 |
| `lambda_max` | 100.0 | λ 上限 |

---

# 第六部分：总结

## 10. 核心贡献

```
┌─────────────────────────────────────────────────────────────────────────┐
│                   IS-Reshape v11: Cauchy/Arctan 框架                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  【核心公式】                                                             │
│                                                                         │
│      φ(w) = w / (1 + λ·w²)                                             │
│                                                                         │
│      f(w) = (1/√λ) · arctan(√λ · w)                                    │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  【核心洞察】                                                             │
│                                                                         │
│      Arctan 不是设计选择，是数学必然！                                    │
│                                                                         │
│      MSE 最小化 + IS 方差 ∝ w² → 唯一解是 Arctan                        │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  【推导路径】                                                             │
│                                                                         │
│      加权 MSE: L(φ) = α(A)·(w-φ)² + β(A)·(wφ)²                         │
│           ↓                                                             │
│      最优解: φ* = w / (1 + λw²),  λ = β/α                              │
│           ↓                                                             │
│      反推目标函数: φ = w·f'(w) → f'(w) = 1/(1+λw²)                      │
│           ↓                                                             │
│      积分: f(w) = (1/√λ)·arctan(√λ·w)                                  │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  【物理解释】                                                             │
│                                                                         │
│      λ(A) = Risk_Cost / Opportunity_Cost                               │
│                                                                         │
│      A > 0: 高紧迫度 → λ 小 → φ ≈ w (激进学习)                          │
│      A < 0: 高风险厌恶 → λ 大 → φ 截断 (保守)                            │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## 11. 与历史版本的关系

```
v1-v6: 理论探索期 (Box-Cox, ESS 约束)
  │
  ├─ v7: 显式 γ+/γ- 分组
  │
  ├─ v8: α-Divergence 视角
  │
  ├─ v9: Information Bottleneck, Softplus 截断
  │
  ├─ v10: 调和加权 Harmonic IS (Defensive IS / JSD)
  │       φ(w) = w/(w+λ), f(w) = (1/λ)ln(λw+1)
  │
  └─ v11: Cauchy/Arctan IS (MSE 最优)
          φ(w) = w/(1+λw²), f(w) = (1/√λ)arctan(√λw)

          关键突破：证明 Arctan 是数学必然，不是设计选择
```

---

## 附录 A：关键公式汇总

**梯度权重**：
$$\phi(w) = \frac{w}{1 + \lambda w^2}$$

**目标函数**：
$$f(w) = \frac{1}{\sqrt{\lambda}} \arctan(\sqrt{\lambda} \cdot w)$$

**一阶导**：
$$f'(w) = \frac{1}{1 + \lambda w^2}$$

**动态 λ（指数效用）**：
$$\lambda(A) = \frac{\beta_0}{\alpha_0} e^{-\text{scale} \cdot A}$$

**极限行为**：
$$\lim_{w \to 0} \phi(w) = w, \quad \lim_{w \to \infty} \phi(w) = \frac{1}{\lambda w}$$

---

## 附录 B：Cauchy 分布联系

$\phi(w) = \frac{w}{1 + \lambda w^2}$ 的形式与 **Cauchy 分布** 的 PDF 相关：
$$p(x) = \frac{1}{\pi \gamma \left[1 + \left(\frac{x}{\gamma}\right)^2\right]}$$

这也是为什么我们将此方法命名为 "Cauchy IS"。

---

## 附录 C：配置示例

**静态模式（推荐）**：
```yaml
actor_rollout_ref:
  actor:
    policy_loss:
      loss_mode: "cauchy_is"
      cauchy_is:
        lambda_: 1.0
```

**动态模式**：
```yaml
actor_rollout_ref:
  actor:
    policy_loss:
      loss_mode: "cauchy_is"
      cauchy_is:
        lambda_: null
        risk_aversion: 1.0
        base_urgency: 1.0
        urgency_scale: 1.0
        lambda_min: 0.01
        lambda_max: 100.0
```
