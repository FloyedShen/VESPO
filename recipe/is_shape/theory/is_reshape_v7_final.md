# IS-Reshape: 从策略梯度到 f-散度的统一框架

**版本**: 7.7 (VER Framework)

---

## 摘要

本文从策略梯度的基本形式出发，系统地推导出 IS-Reshape 框架。

**核心问题**：如何用 off-policy 数据估计 on-policy 策略梯度？

**主要贡献**：
1. 从 $\nabla_\theta \mathbb{E}_{\pi_\theta}[A]$ 出发，统一 RL（Reverse KL）和 Weighted SFT（Forward KL）
2. 证明 $(w^\gamma - 1)/\gamma$ 形式源自 f-散度的梯度结构（Box-Cox 变换）
3. 建立 γ 与幂散度族的精确对应：γ < 1 → Forward KL 行为，γ > 1 → 超 Reverse KL 行为
4. **严格证明 MSE 在 (0,1) 存在内点最优，在 (1,2) 单调递增**
5. **提出价值效率比（VER）准则**，统一正负样本的优化目标
6. **建立非对称优化框架**：正样本最大化 VER+，负样本最大化 VER-
7. 提供 γ+, γ- 的闭式 O(n) 计算方法
8. **多角度理论支撑**：梯度动力学、凸对偶、信息几何

**理论创新（v7.7）**：
- **价值效率比（VER）**：明确的优化目标——最大化"有价值梯度"占比
- **单调性定理**：VER+ 关于 γ 递减，VER- 关于 γ 递增
- **统一框架**：正负样本用同一准则（VER 最大化），约束同为 ESS
- **最优解在约束边界**：由 VER 单调性直接得出，无需 MSE 分析

---

# 第一部分：问题与两端

## 1. 核心问题

### 1.1 目标

策略优化的目标是最大化期望优势：
$$J(\theta) = \mathbb{E}_{\pi_\theta}[A(x, y)]$$

其梯度为：
$$\boxed{g^* = \nabla_\theta J = \mathbb{E}_{\pi_\theta}[A \cdot \nabla_\theta \log\pi_\theta]}$$

### 1.2 Off-Policy 约束

实践中，我们只有从行为策略 μ 采样的数据 $\{(x_i, y_i, A_i)\}_{i=1}^n$。

**问题**：如何用 μ 的数据估计 $g^*$？

### 1.3 两种极端方案

**方案 A：标准重要性采样（RL 端）**

$$g_{IS} = \mathbb{E}_\mu\left[\frac{\pi_\theta}{\mu} \cdot A \cdot \nabla_\theta \log\pi_\theta\right] = \mathbb{E}_\mu[w \cdot A \cdot \nabla_\theta \log\pi_\theta]$$

- ✅ 无偏：$\mathbb{E}[g_{IS}] = g^*$
- ❌ 高方差：当 $\pi_\theta$ 偏离 μ 时，w 的方差爆炸

**方案 B：忽略分布差异（SFT 端）**

$$g_{SFT} = \mathbb{E}_\mu[A \cdot \nabla_\theta \log\pi_\theta]$$

- ❌ 有偏：$\mathbb{E}[g_{SFT}] \neq g^*$
- ✅ 低方差：不依赖 w

---

## 2. 两端的目标函数

### 2.1 RL 端的目标

$g_{IS}$ 对应什么目标函数？

$$\nabla_\theta \mathbb{E}_\mu[w \cdot A] = \mathbb{E}_\mu[A \cdot \nabla_\theta w] = \mathbb{E}_\mu[w \cdot A \cdot \nabla_\theta \log\pi_\theta] = g_{IS}$$

所以 **RL 目标**是：
$$\boxed{L_{RL} = \mathbb{E}_\mu[w \cdot A]}$$

### 2.2 SFT 端的目标

$g_{SFT}$ 对应什么目标函数？

如果 $f(w) = \log w$，则：
$$\nabla_\theta \mathbb{E}_\mu[\log w \cdot A] = \mathbb{E}_\mu\left[A \cdot \frac{\nabla_\theta w}{w}\right] = \mathbb{E}_\mu[A \cdot \nabla_\theta \log\pi_\theta] = g_{SFT}$$

所以 **Advantage-Weighted SFT 目标**是：
$$\boxed{L_{SFT} = \mathbb{E}_\mu[\log w \cdot A] = \mathbb{E}_\mu[A \cdot \log\pi_\theta] - \text{const}}$$

### 2.3 统一视角

| 方法 | 目标函数 f(w) | 梯度权重 φ(w) = f'(w)·w | 性质 |
|------|-------------|------------------------|------|
| RL (IS) | w | w | 无偏，高方差 |
| Weighted SFT | log w | 1 | 有偏，低方差 |

**关键观察**：两端都可以写成 $L = \mathbb{E}_\mu[f(w) \cdot A]$ 的形式！

---

## 3. 一般形式：寻找最优的 f(w)

### 3.1 一般估计器

定义一般的梯度估计器：
$$\boxed{\hat{g} = \nabla_\theta \mathbb{E}_\mu[f(w) \cdot A]}$$

其中 f 是关于 w 的可微函数。

### 3.2 梯度的形式

$$\hat{g} = \mathbb{E}_\mu\left[A \cdot f'(w) \cdot \nabla_\theta w\right] = \mathbb{E}_\mu\left[A \cdot f'(w) \cdot w \cdot \nabla_\theta \log\pi_\theta\right]$$

**定义**：梯度权重函数
$$\boxed{\phi(w) = f'(w) \cdot w}$$

则：
$$\hat{g} = \mathbb{E}_\mu[\phi(w) \cdot A \cdot \nabla_\theta \log\pi_\theta]$$

### 3.3 无偏条件


**定理 3.1**：$\hat{g}$ 无偏当且仅当 $\phi(w) = w$，即 $f(w) = w$。

**证明**：
$$\hat{g} = g^* \iff \mathbb{E}_\mu[\phi(w) \cdot A \cdot \nabla\log\pi] = \mathbb{E}_\mu[w \cdot A \cdot \nabla\log\pi]$$

对所有 π, μ, A 成立，必须 $\phi(w) = w$。由 $\phi(w) = f'(w) \cdot w = w$，得 $f'(w) = 1$，即 $f(w) = w + c$。$\blacksquare$

### 3.4 Bias-Variance 权衡

**目标**：最小化均方误差
$$\boxed{\text{MSE}(\phi) = \|\text{Bias}(\phi)\|^2 + \frac{1}{n}\text{Var}(\phi)}$$

**偏差**：
$$\text{Bias}(\phi) = \mathbb{E}_\mu[(\phi(w) - w) \cdot A \cdot \nabla\log\pi]$$

当 $\phi(w) = w^\gamma$：
$$\text{Bias}(\gamma) = \mathbb{E}_\mu[(w^\gamma - w) \cdot A \cdot \nabla\log\pi]$$

**方差**：
$$\text{Var}(\phi) \propto \mathbb{E}_\mu[\phi(w)^2 \cdot A^2 \cdot \|\nabla\log\pi\|^2]$$

当 $\phi(w) = w^\gamma$：
$$\text{Var}(\gamma) \propto \mathbb{E}_\mu[w^{2\gamma} \cdot A^2]$$

**权衡**：
- γ = 1：Bias = 0，但 Var 最大
- γ → 0：Var 降低，但 Bias 增大
- **存在最优 γ* 使 MSE 最小**

### 3.5 核心问题

**在 MSE 意义下，什么是最优的 f(w)？**

---

# 第二部分：f-散度与最优 f(w) 的推导

## 4. f(w) 与 f-散度的对应

### 4.1 关键观察

我们的估计器梯度具有形式：
$$\hat{g} = \mathbb{E}_\mu[\phi(w) \cdot A \cdot \nabla\log\pi], \quad \phi(w) = f'(w) \cdot w$$

**这与 f-散度的梯度结构完全一致！**

### 4.2 f-散度定义

**定义 4.1**：给定凸函数 $f: \mathbb{R}^+ \to \mathbb{R}$ 且 $f(1) = 0$，f-散度为：
$$D_f(\pi \| \mu) = \mathbb{E}_\mu\left[f\left(\frac{\pi}{\mu}\right)\right] = \mathbb{E}_\mu[f(w)]$$

### 4.3 f-散度的梯度

**定理 4.2**：f-散度关于 π 的梯度为：
$$\nabla_\theta D_f(\pi_\theta \| \mu) = \mathbb{E}_\mu[f'(w) \cdot w \cdot \nabla_\theta \log\pi_\theta]$$

**证明**：
$$\nabla_\theta D_f = \nabla_\theta \mathbb{E}_\mu[f(w)] = \mathbb{E}_\mu[f'(w) \cdot \nabla_\theta w] = \mathbb{E}_\mu[f'(w) \cdot w \cdot \nabla_\theta \log\pi_\theta]$$
$\blacksquare$

### 4.4 核心对应关系

**定理 4.3**：选择目标函数 $L = \mathbb{E}_\mu[f(w) \cdot A]$ 等价于选择 f-散度的生成函数 f。

梯度权重 $\phi(w) = f'(w) \cdot w$ 完全由 f 决定。

**问题转化**：寻找最优 f(w) = 寻找最优的 f-散度生成函数。

---

## 5. 从端点条件推导 Box-Cox 形式

### 5.1 两端的约束

我们需要 f(w) 满足：
- **SFT 端**（γ → 0）：$\phi(w) = f'(w) \cdot w = 1$（忽略 IS）
- **RL 端**（γ = 1）：$\phi(w) = f'(w) \cdot w = w$（完整 IS）

### 5.2 求解 f(w)

**从 SFT 端**：$\phi(w) = 1$ 要求 $f'(w) = 1/w$，即 $f(w) = \log w$

**从 RL 端**：$\phi(w) = w$ 要求 $f'(w) = 1$，即 $f(w) = w$（+ 常数）

### 5.3 寻找插值

需要一族函数 $f_\gamma(w)$ 满足：
- $f_0(w) = \log w$
- $f_1(w) = w - 1$（减 1 保证 $f(1) = 0$）

**唯一自然的选择**：Box-Cox 变换

$$\boxed{f_\gamma(w) = \frac{w^\gamma - 1}{\gamma}, \quad \gamma \in (0, 1]}$$

### 5.4 验证

**极限 γ → 0**：
$$\lim_{\gamma \to 0} \frac{w^\gamma - 1}{\gamma} = \lim_{\gamma \to 0} \frac{e^{\gamma \log w} - 1}{\gamma} = \log w \quad \checkmark$$

**γ = 1**：
$$f_1(w) = w - 1 \quad \checkmark$$

**梯度权重**：
$$\phi_\gamma(w) = f'_\gamma(w) \cdot w = w^{\gamma-1} \cdot w = w^\gamma \quad \checkmark$$

### 5.5 对应的散度族

Box-Cox $f_\gamma(w) = \frac{w^\gamma - 1}{\gamma}$ 对应**幂散度族**（Power Divergence）：

$$D_\gamma^{pow}(\pi \| \mu) = \frac{1}{\gamma(\gamma-1)}\mathbb{E}_\mu[w^\gamma - \gamma w + \gamma - 1]$$

| γ | 散度名称 | 性质 |
|---|---------|------|
| γ → 0 | **Forward KL** $D_{KL}(\mu \| \pi)$ | Mean-seeking |
| γ = 0.5 | Hellinger 距离 | 对称 |
| γ → 1 | **Reverse KL** $D_{KL}(\pi \| \mu)$ | Mode-seeking |
| γ = 2 | χ² 散度 | 对尾部敏感 |

---

## 6. 与 γ-散度（Cressie-Read 族）的联系

### 6.1 Cressie-Read 散度族

**定义 6.1（Cressie-Read γ-散度）**：
$$D_\gamma^{CR}(\pi \| \mu) = \frac{1}{\gamma(\gamma+1)}\mathbb{E}_\mu\left[w^{\gamma+1} - (\gamma+1)w + \gamma\right]$$

**关键**：其生成函数为
$$f_{CR}(t) = \frac{t^{\gamma+1} - (\gamma+1)t + \gamma}{\gamma(\gamma+1)}$$

### 6.2 与 Box-Cox 的精确关系

**定理 6.2**：Box-Cox 变换 $f_\gamma(w) = \frac{w^\gamma - 1}{\gamma}$ 的梯度权重与 Cressie-Read 散度的梯度结构一致。

对于 Cressie-Read：
$$\frac{\partial f_{CR}}{\partial t} = \frac{t^\gamma - 1}{\gamma}$$

这正是我们的 Box-Cox 函数！因此：
$$\phi(w) = f'_{CR}(w) \cdot w = \frac{w^\gamma - 1}{\gamma} \cdot w \neq w^\gamma$$

**修正**：实际上，我们的目标函数 $L = \mathbb{E}_\mu[\frac{w^\gamma - 1}{\gamma} \cdot A]$ 直接给出 $\phi(w) = w^\gamma$，这对应于**幂散度族**：

$$D_\gamma^{pow}(\pi \| \mu) = \frac{1}{\gamma-1}\left(\mathbb{E}_\mu[w^\gamma] - 1\right)$$

### 6.3 幂散度的特殊情况

| γ | 散度名称 | 性质 |
|---|---------|------|
| γ → 0 | **Forward KL** $D_{KL}(\mu \| \pi)$ | Mean-seeking |
| γ = 0.5 | Hellinger 距离 | 对称 |
| γ → 1 | **Reverse KL** $D_{KL}(\pi \| \mu)$ | Mode-seeking |
| γ = 2 | χ² 散度 | 对尾部敏感 |

### 6.4 Forward KL vs Reverse KL 的几何意义

**Forward KL（γ → 0，SFT 端）**：
$$D_{KL}(\mu \| \pi) = \mathbb{E}_\mu[\log(\mu/\pi)]$$

- **Mean-seeking**：π 被迫覆盖 μ 的所有支撑集
- 惩罚 π(y) = 0 但 μ(y) > 0 的情况
- **倾向于过度泛化**

**Reverse KL（γ → 1，RL 端）**：
$$D_{KL}(\pi \| \mu) = \mathbb{E}_\pi[\log(\pi/\mu)]$$

- **Mode-seeking**：π 可以只覆盖 μ 的主要模式
- 惩罚 π(y) > 0 但 μ(y) = 0 的情况
- **倾向于模式坍缩**

### 6.5 γ < 1 vs γ > 1 的几何含义

**定理 6.3**：γ 控制估计器的"模式偏好"：

| γ 范围 | 散度行为 | 估计器性质 |
|--------|---------|-----------|
| γ < 1 | 更接近 Forward KL | 倾向于覆盖所有样本，容忍 w < 1 |
| γ = 1 | Reverse KL | 无偏但高方差 |
| γ > 1 | 超越 Reverse KL | **更激进的 mode-seeking**，放大 w > 1 |

**关键洞察**：
- γ < 1 对 w < 1 的样本更敏感（放大新发现的样本）
- γ > 1 对 w > 1 的样本更敏感（惩罚未改变的样本）

---

## 7. 与 Rényi 散度的联系

### 7.1 Rényi 散度定义

$$D_\alpha^R(\pi \| \mu) = \frac{1}{\alpha - 1} \log \mathbb{E}_\mu[w^\alpha]$$

### 7.2 核心恒等式

**定理 7.1（精确恒等式）**：
$$\boxed{\mathbb{E}_\mu[w^\gamma] = \exp\left((\gamma - 1) \cdot D_\gamma^R(\pi \| \mu)\right)}$$

### 7.3 方差的 Rényi 刻画

梯度方差的主导项：
$$\text{Var}(\hat{g}_\gamma) \propto \mathbb{E}_\mu[w^{2\gamma}] = \exp\left((2\gamma - 1) \cdot D_{2\gamma}^R(\pi \| \mu)\right)$$

**结论**：
- γ 小 → 低阶 Rényi → 低方差
- γ 大 → 高阶 Rényi → 高方差

---

# 第三部分：正负样本的不对称处理

## 8. 为什么需要 γ(A)

### 8.1 问题：单一 γ 的局限

对于 $\phi_\gamma(w) = w^\gamma$：

| γ 范围 | w < 1 时 | w > 1 时 | 函数形状 |
|--------|---------|---------|---------|
| γ < 1 | $w^\gamma > w$（放大） | $w^\gamma < w$（缩小） | 凹函数 |
| γ = 1 | $w^\gamma = w$ | $w^\gamma = w$ | 线性 |
| γ > 1 | $w^\gamma < w$（缩小） | $w^\gamma > w$（放大） | 凸函数 |

### 8.2 正负样本需要相反的行为

**正样本 (A > 0)** 的需求：
- w < 1（新发现的好样本）→ 应该**放大**权重来学习
- w > 1（已知的好样本）→ 可以**缩小**权重降低方差
- **需要凹函数（γ < 1）**

**负样本 (A < 0)** 的需求：
- w < 1（已避免的坏样本）→ 应该**缩小**权重（不需要额外惩罚）
- w > 1（未避免的坏样本）→ 应该**放大**权重来惩罚
- **需要凸函数（γ > 1）**

### 8.3 MSE 分析：为什么 (0,1) 有最小值而 (1,2) 没有

**假设**：$\log w \sim \mathcal{N}(-\sigma^2/2, \sigma^2)$（保证 $\mathbb{E}[w] = 1$）

**MSE 函数**：
$$\text{MSE}(\gamma) = c(1-\gamma)^2 + \frac{1}{n} e^{\sigma^2\gamma(2\gamma-1)}$$

其中：
- $B(\gamma) = c(1-\gamma)^2$：偏差项（在 γ=1 处最小）
- $V(\gamma) = \frac{1}{n} e^{\sigma^2\gamma(2\gamma-1)}$：方差项

#### 方差项的性质

设 $h(\gamma) = \sigma^2\gamma(2\gamma-1) = \sigma^2(2\gamma^2 - \gamma)$，则：
$$h'(\gamma) = \sigma^2(4\gamma - 1) = 0 \implies \gamma^* = \frac{1}{4}$$

**方差在 γ = 1/4 处取得最小值**，而非 γ = 1！

| γ 值 | h(γ) | V(γ) 相对大小 |
|------|------|--------------|
| 0 | 0 | 1 |
| 1/4 | -σ²/8 | **最小** |
| 1 | σ² | exp(σ²) |
| 2 | 6σ² | exp(6σ²) |

#### 定理 8.3（MSE 在 (0,1) 上存在内点最小值）

**命题**：MSE(γ) 在 $\gamma \in (0, 1)$ 上存在唯一的内点最小值 $\gamma^* \in (0, 1)$。

**证明**：计算一阶导数：
$$\frac{d\text{MSE}}{d\gamma} = 2c(\gamma-1) + \frac{1}{n} \cdot e^{h(\gamma)} \cdot \sigma^2(4\gamma - 1)$$

在边界检验：
- **γ = 0**：$\frac{d\text{MSE}}{d\gamma}\big|_{\gamma=0} = -2c + \frac{\sigma^2}{n} \cdot (-1) = -2c - \frac{\sigma^2}{n} < 0$
- **γ = 1**：$\frac{d\text{MSE}}{d\gamma}\big|_{\gamma=1} = 0 + \frac{1}{n} \cdot e^{\sigma^2} \cdot 3\sigma^2 > 0$

导数从负变正，由连续性，**存在 $\gamma^* \in (0, 1)$ 使得导数为零**。

结合 MSE 在整个定义域上严格凸（二阶导数恒正），该零点是**唯一的全局最小值**。$\blacksquare$

#### 定理 8.4（MSE 在 (1,2) 上单调递增——无内点最小值）

**命题**：MSE(γ) 在 $\gamma \in (1, 2)$ 上**严格单调递增**，最小值在边界 γ = 1 处取得。

**证明**：在 (1, 2) 上检验导数符号：
$$\frac{d\text{MSE}}{d\gamma} = \underbrace{2c(\gamma-1)}_{> 0 \text{ (因为 } \gamma > 1)} + \underbrace{\frac{1}{n} \cdot e^{h(\gamma)} \cdot \sigma^2(4\gamma - 1)}_{> 0 \text{ (因为 } \gamma > 1/4)}$$

两项均为正，故 $\frac{d\text{MSE}}{d\gamma} > 0$ 对所有 $\gamma \in (1, 2)$ 成立。

**结论**：MSE 在 (1, 2) 上严格递增，不存在内点最小值。$\blacksquare$

#### 关键结论

**定理 8.5（MSE 最小化的局限性）**：

从 MSE 最小化的角度：
1. **存在唯一的全局最优** $\gamma^* \in (0, 1)$
2. 在 (1, 2) 区间内，MSE 单调递增，**不存在内点最优**
3. **MSE 最小化无法为 γ > 1 提供理论支撑**

```
MSE
 │
 │                                    /
 │                                   /
 │                                  /
 │\                                /
 │ \                              /
 │  \                            /
 │   \         ·────────────────·
 │    *       ↑                ↑
 │            γ=1              γ=2
 └─────────────────────────────────────── γ
      γ^*     (1,2) 单调递增
     (0,1)
```

---

### 8.4 价值效率比（Value Efficiency Ratio, VER）准则

MSE 准则回答的问题是：如何准确估计真实梯度 $g^* = \mathbb{E}[w \cdot A \cdot \nabla\log\pi]$？

但对于强化学习，更重要的问题是：**如何高效地学习**？

#### 8.4.1 有价值样本与无价值样本

**核心洞察**：不是所有样本对学习都同等重要。

**对于正样本 (A > 0)**：
| 样本类型 | 含义 | 学习价值 |
|---------|------|---------|
| w < 1 | π(y) < μ(y)，新发现的好样本 | **高** — 需要学习 |
| w > 1 | π(y) > μ(y)，已学好的样本 | **低** — 继续强化是浪费 |

**对于负样本 (A < 0)**：
| 样本类型 | 含义 | 学习价值 |
|---------|------|---------|
| w < 1 | π(y) < μ(y)，已避免的坏样本 | **低** — 继续惩罚是浪费 |
| w > 1 | π(y) > μ(y)，未纠正的坏样本 | **高** — 需要惩罚 |

#### 8.4.2 梯度价值分解

定义梯度的**价值函数**：
$$V(\gamma; w, A) = w^\gamma \cdot |A|$$

**对于正样本**，分解为有价值部分和无价值部分：
$$V_+^{useful}(\gamma) = \mathbb{E}[w^\gamma \cdot A \cdot \mathbf{1}(w < 1) \mid A > 0]$$
$$V_+^{waste}(\gamma) = \mathbb{E}[w^\gamma \cdot A \cdot \mathbf{1}(w > 1) \mid A > 0]$$

**对于负样本**：
$$V_-^{useful}(\gamma) = \mathbb{E}[w^\gamma \cdot |A| \cdot \mathbf{1}(w > 1) \mid A < 0]$$
$$V_-^{waste}(\gamma) = \mathbb{E}[w^\gamma \cdot |A| \cdot \mathbf{1}(w < 1) \mid A < 0]$$

#### 8.4.3 价值效率比定义

**定义 8.9（价值效率比 VER）**：

$$\boxed{\text{VER}(\gamma) = \frac{V^{useful}(\gamma)}{V^{useful}(\gamma) + V^{waste}(\gamma)}}$$

**对于正样本**：
$$\text{VER}_+(\gamma) = \frac{\mathbb{E}[w^\gamma \cdot A \cdot \mathbf{1}(w < 1) \mid A > 0]}{\mathbb{E}[w^\gamma \cdot A \mid A > 0]}$$

**对于负样本**：
$$\text{VER}_-(\gamma) = \frac{\mathbb{E}[w^\gamma \cdot |A| \cdot \mathbf{1}(w > 1) \mid A < 0]}{\mathbb{E}[w^\gamma \cdot |A| \mid A < 0]}$$

**物理意义**：VER 衡量梯度信号中"有价值部分"所占的比例。VER 越高，学习效率越高。

#### 8.4.4 VER 的单调性定理

**定理 8.10（VER 单调性）**：

1. $\text{VER}_+(\gamma)$ 关于 γ **严格单调递减**
2. $\text{VER}_-(\gamma)$ 关于 γ **严格单调递增**

**证明**：

考虑权重函数 $\phi_\gamma(w) = w^\gamma$ 的行为：

**情况 1：γ 增大时（γ₁ < γ₂）**

对于 w < 1：
$$\frac{w^{\gamma_2}}{w^{\gamma_1}} = w^{\gamma_2 - \gamma_1} < 1 \quad (\text{因为 } w < 1, \gamma_2 - \gamma_1 > 0)$$

对于 w > 1：
$$\frac{w^{\gamma_2}}{w^{\gamma_1}} = w^{\gamma_2 - \gamma_1} > 1 \quad (\text{因为 } w > 1, \gamma_2 - \gamma_1 > 0)$$

因此，γ 增大时：
- w < 1 样本的权重**相对减小**
- w > 1 样本的权重**相对增大**

**对于 $\text{VER}_+(\gamma)$**（分子是 w < 1 部分）：
γ 增大 → w < 1 部分相对减小 → $\text{VER}_+(\gamma)$ **递减** ✓

**对于 $\text{VER}_-(\gamma)$**（分子是 w > 1 部分）：
γ 增大 → w > 1 部分相对增大 → $\text{VER}_-(\gamma)$ **递增** ✓

$\blacksquare$

#### 8.4.5 VER 最优化

**推论 8.11（VER 最优化原则）**：

- 最大化 $\text{VER}_+(\gamma)$：选择 γ **尽量小**
- 最大化 $\text{VER}_-(\gamma)$：选择 γ **尽量大**

**约束**：γ 不能任意小/大，需要满足 ESS 约束以控制方差：
$$\text{ESS}(\gamma) \geq \rho_{min} \cdot n$$

#### 8.4.6 统一优化框架

**定理 8.12（VER 统一框架）**：

正负样本的最优 γ 可以统一表述为：

$$\boxed{\max_{\gamma_+, \gamma_-} \text{VER}_+(\gamma_+) + \text{VER}_-(\gamma_-) \quad \text{s.t.} \quad \text{ESS}(\gamma) \geq \rho_{min} \cdot n}$$

由于 VER+ 关于 γ 递减、VER- 关于 γ 递增，最优解在约束边界：

$$\gamma_+^* = \min\{\gamma \in (0, 1) : \text{ESS}_+(\gamma) \geq \rho_{min} \cdot n_+\}$$
$$\gamma_-^* = \max\{\gamma \in (1, 2) : \text{ESS}_-(\gamma) \geq \rho_{min} \cdot n_-\}$$

**关键性质**：两种样本使用**同一优化准则**（VER 最大化），只是单调性方向相反，导致最优 γ 位于不同区间。

---

### 8.5 多角度理论支撑

#### 8.5.1 梯度动力学视角

考虑策略参数在梯度流下的演化：
$$\frac{d\theta}{dt} = \mathbb{E}[w^\gamma \cdot A \cdot \nabla\log\pi_\theta]$$

不同的 γ 定义了不同的**向量场**（flow field）。

**核心洞察**：正/负样本在"相空间"中应该有不同的动力学！

| 样本类型 | 期望动力学 | 对应 γ |
|---------|-----------|--------|
| 新发现的好样本 ($w<1, A>0$) | **加速**学习 | $\gamma < 1$（放大） |
| 已学好的样本 ($w>1, A>0$) | **减速**避免过拟合 | $\gamma < 1$（缩小） |
| 未纠正的坏样本 ($w>1, A<0$) | **加速**纠正 | $\gamma > 1$（放大） |
| 已避免的坏样本 ($w<1, A<0$) | **减速**节省梯度 | $\gamma > 1$（缩小） |

**结论**：凹函数（$\gamma < 1$）和凸函数（$\gamma > 1$）自然对应了这两种不同的动力学！

#### 8.5.2 凸对偶视角

原问题：
$$\min_\gamma \text{MSE}(\gamma) = \text{Bias}^2(\gamma) + \frac{1}{n}\text{Var}(\gamma)$$

等价的 Lagrangian 形式：
$$\mathcal{L}(\gamma, \lambda) = \text{Bias}^2(\gamma) + \lambda \cdot \text{Var}(\gamma)$$

其中 $\lambda = 1/n$ 是**方差惩罚系数**。

**关键观察**：
- $\lambda$ 大（小样本）→ 更重视低方差 → 最优 $\gamma$ 更小
- $\lambda$ 小（大样本）→ 更重视无偏 → 最优 $\gamma \to 1$

**对于正负样本**，可以定义不同的等效 $\lambda$：
- 正样本：标准 $\lambda_+ = 1/n_+$
- 负样本：负的等效 $\lambda_-$（因为我们想要更大的有效信号而非更小的方差）

这从对偶角度解释了为什么负样本需要 $\gamma > 1$——它们的优化目标本质上不同。

#### 8.5.3 信息几何视角

在信息几何中，参数空间有 Fisher 信息度量：
$$ds^2 = \mathbb{E}[(\nabla\log\pi)^T (\nabla\log\pi)] \, d\theta^2$$

IS 权重 $w^\gamma$ 修改了有效度量：
$$ds_\gamma^2 = \mathbb{E}[w^{2\gamma} \cdot (\nabla\log\pi)^T (\nabla\log\pi)] \, d\theta^2$$

**γ 改变了参数空间的几何**：
- $\gamma < 1$：压缩 $w > 1$ 方向的度量，更保守的更新
- $\gamma > 1$：拉伸 $w > 1$ 方向的度量，更激进的更新

**结论**：正负样本需要在**不同的几何**中进行优化。

#### 8.5.4 $w^\gamma$ 的行为对称性

$w^\gamma$ 对于 $\gamma < 1$ 和 $\gamma > 1$ 存在"行为对称性"：

| 条件 | $\gamma < 1$ (凹函数) | $\gamma > 1$ (凸函数) |
|------|---------------------|---------------------|
| $w < 1$ | $w^\gamma > w$（放大） | $w^\gamma < w$（缩小） |
| $w > 1$ | $w^\gamma < w$（缩小） | $w^\gamma > w$（放大） |

这说明 $\gamma < 1$ 和 $\gamma > 1$ 在处理 $w$ 时行为**完全相反**。

**重要说明**：虽然存在行为对称性，但**方差结构不对称**。在 Log-Normal 假设下：

$$\text{Var}(\gamma) \propto \exp(\sigma^2 \gamma(2\gamma - 1))$$

方差在 $\gamma > 1$ 时增长更快，这是固有的不对称性。因此：
- 行为对称性 → 解释了**为什么**正负样本需要不同范围的 γ
- 方差不对称性 → 需要 ESS 约束来**控制** γ 的具体取值

---

### 8.6 非对称优化框架（VER 视角）

#### 8.6.1 统一目标

**定义**：学习效率函数
$$\mathcal{E}(\gamma_+, \gamma_-) = \text{VER}_+(\gamma_+) + \text{VER}_-(\gamma_-)$$

**优化问题**：
$$\max_{\gamma_+, \gamma_-} \mathcal{E}(\gamma_+, \gamma_-) \quad \text{s.t.} \quad \text{ESS}_\pm(\gamma_\pm) \geq \rho_{min} \cdot n_\pm$$

#### 8.6.2 最优解

由 VER 单调性（定理 8.10），最优解在 ESS 约束边界：

**正样本**：VER+ 递减 → γ 越小越好 → 取下界
$$\gamma_+^* = \text{argmin}_{\gamma} \{\gamma : \text{ESS}_+(\gamma) \geq \rho_{min} \cdot n_+\}$$

**负样本**：VER- 递增 → γ 越大越好 → 取上界
$$\gamma_-^* = \text{argmax}_{\gamma} \{\gamma : \text{ESS}_-(\gamma) \geq \rho_{min} \cdot n_-\}$$

在 Log-Normal 假设下，$\text{ESS}/n \approx e^{-\sigma^2\gamma^2}$，约束等价于：
$$|\gamma| \leq \sqrt{\frac{-\log\rho_{min}}{\sigma^2}}$$

#### 8.6.3 闭式解

**正样本 γ+**（需要 γ < 1 且接近 0）：
$$\boxed{\gamma_+ = \min\left(1 - \epsilon, \sqrt{\frac{-\log\rho_{min}}{\sigma_+^2}}\right)}$$

**负样本 γ-**（需要 γ > 1 且尽量大）：
$$\boxed{\gamma_- = 1 + \min\left(1 - \epsilon, \sqrt{\frac{-\log\rho_{min}}{\sigma_-^2}}\right)}$$

**注**：γ- 从 1 开始增加，增量由 ESS 约束决定。

#### 8.6.4 理论完备性

**定理 8.13（VER 最优性）**：

设 $\gamma_+^{VER}$ 和 $\gamma_-^{VER}$ 为 VER 最优解。则：

1. $\gamma_+^{VER} \in (0, 1)$（ESS 约束下界）
2. $\gamma_-^{VER} \in (1, 2)$（ESS 约束上界）
3. 使用 $(\gamma_+^{VER}, \gamma_-^{VER})$ 的学习效率 > 任何单一 γ 值

**证明要点**：

对于 (1) 和 (2)：由 VER 单调性 + ESS 约束直接得出。

对于 (3)：
- 若对所有样本用 $\gamma < 1$：VER- 较低（w > 1 负样本的权重被抑制）
- 若对所有样本用 $\gamma > 1$：VER+ 较低（w < 1 正样本的权重被抑制）
- 分别优化能同时最大化两个 VER。$\blacksquare$

#### 8.6.5 理论总结表

| 维度 | 正样本 (A > 0) | 负样本 (A < 0) |
|------|---------------|---------------|
| **学习目标** | 学习好行为 | 纠正坏行为 |
| **有价值样本** | w < 1（新发现） | w > 1（未纠正） |
| **优化准则** | VER+ 最大化 | VER- 最大化 |
| **VER 单调性** | 递减于 γ | 递增于 γ |
| **最优 γ 方向** | 越小越好 | 越大越好 |
| **约束** | ESS ≥ ρ_min · n | ESS ≥ ρ_min · n |
| **最优 γ 位置** | (0, 1) 下界 | (1, 2) 上界 |
| **权重函数性质** | 凹函数 | 凸函数 |

---

## 9. 实际计算：按 A 符号分组

### 9.1 核心思想

**不需要连续调制**！直接按 Advantage 符号分组，分别最大化 VER：

$$\boxed{\gamma(A) = \begin{cases}
\gamma_+ \in (0, 1) & A > 0 \quad \text{(VER+ 最大化)} \\
\gamma_- \in (1, 2) & A < 0 \quad \text{(VER- 最大化)}
\end{cases}}$$

### 9.2 分组计算流程

```
输入: 样本 {(x_i, y_i, A_i, log_w_i)}

1. 按 A 符号分组:
   - 正样本组: S+ = {i : A_i > 0}
   - 负样本组: S- = {i : A_i < 0}

2. 分别计算最优 γ (统一 VER 准则):
   - γ+ = compute_ver_optimal_gamma(log_w[S+], 'min')  # VER+ 递减，取下界
   - γ- = compute_ver_optimal_gamma(log_w[S-], 'max')  # VER- 递增，取上界

3. 应用权重:
   - 对 i ∈ S+: weight_i = exp(γ+ · log_w_i)
   - 对 i ∈ S-: weight_i = exp(γ- · log_w_i)
```

### 9.3 为什么不需要连续调制

之前版本使用 $\gamma(A) = 1 - \delta \cdot \tanh(A/\tau)$ 进行连续调制。

**问题**：
1. 引入额外超参数 τ
2. 对 |A| 小的样本，γ 接近 1，可能不是最优
3. 物理意义不明确

**正确理解**：
- γ 的选择取决于**样本的学习目标**（学习 vs 纠正）
- γ 的选择取决于**样本在 w 空间的分布**（通过 σ² 刻画）
- **与 A 的具体数值无关**，只与 A 的符号相关

### 9.4 闭式 γ 计算

#### 基于 ESS 的推导

**有效样本量（ESS）**：
$$\text{ESS}_\gamma = \frac{(\sum_i w_i^\gamma)^2}{\sum_i w_i^{2\gamma}}$$

**定理 9.1（ESS 闭式近似）**：在 Log-Normal 假设下：
$$\frac{\text{ESS}_\gamma}{n} \approx e^{-\sigma^2 \gamma^2}$$

#### 正样本的 γ+ 计算（VER+ 最大化）

对于正样本，γ+ ∈ (0, 1)。设 $\sigma_+^2 = \text{Var}(\log w | A > 0)$。

**理论基础**：VER+ 关于 γ 递减（定理 8.10），故 γ 越小越好，在 ESS 约束下取下界。

$$\boxed{\gamma_+ = \min\left(1 - \epsilon, \sqrt{\frac{-\log\rho_{min}}{\sigma_+^2}}\right)}$$

其中 $\epsilon > 0$ 是小常数，确保 γ+ < 1。

#### 负样本的 γ- 计算（VER- 最大化）

对于负样本，γ- ∈ (1, 2)。设 $\sigma_-^2 = \text{Var}(\log w | A < 0)$。

**理论基础**：VER- 关于 γ 递增（定理 8.10），故 γ 越大越好，在 ESS 约束下取上界。

$$\boxed{\gamma_- = 1 + \min\left(1 - \epsilon, \sqrt{\frac{-\log\rho_{min}}{\sigma_-^2}}\right)}$$

**解释**：
- γ- 从 1 开始（标准 IS 作为基准）
- 增量 δ = γ- - 1 由 ESS 约束决定
- 这是**原理性最优**（VER 单调递增 → 取约束边界）

#### 计算复杂度

| 步骤 | 复杂度 |
|------|--------|
| 计算 log_w | O(n) |
| 分组 | O(n) |
| 计算 σ+², σ-² | O(n) |
| 计算 γ+, γ- | **O(1)** |
| **总计** | **O(n)** |

---

# 第四部分：完整框架

## 10. 最终形式

### 10.1 目标函数

$$\boxed{L(\theta) = \mathbb{E}_\mu\left[\frac{w^{\gamma(A)} - 1}{\gamma(A)} \cdot A\right]}$$

### 10.2 梯度

$$\boxed{\nabla_\theta L = \mathbb{E}_\mu\left[w^{\gamma(A)} \cdot A \cdot \nabla_\theta \log\pi_\theta\right]}$$

### 10.3 γ(A) 的形式（分组计算）

$$\boxed{\gamma(A) = \begin{cases}
\gamma_+ & A > 0, \quad \gamma_+ \in (0, 1) \\
\gamma_- & A < 0, \quad \gamma_- \in (1, 2)
\end{cases}}$$

**其中 γ+ 和 γ- 根据各组的 σ² 独立计算。**

### 10.4 参数计算

| 参数 | 计算公式 | 区间 | 优化准则 | VER 单调性 |
|------|---------|------|---------|-----------|
| $\gamma_+$ | $\min(1-\epsilon, \sqrt{-\log\rho_{min}/\sigma_+^2})$ | (0, 1) | VER+ 最大化 | 递减 → 取下界 |
| $\gamma_-$ | $1 + \min(1-\epsilon, \sqrt{-\log\rho_{min}/\sigma_-^2})$ | (1, 2) | VER- 最大化 | 递增 → 取上界 |

**理论基础**：
- γ+：VER+ 关于 γ 递减，在 ESS 约束下取最小值（定理 8.10）
- γ-：VER- 关于 γ 递增，在 ESS 约束下取最大值（定理 8.10）
- **统一准则**：两种样本都最大化 VER，只是单调性方向不同

---

## 11. 两端的极限

### 11.1 γ+ = γ- = 1（纯 RL）

$$\nabla L = \mathbb{E}_\mu[w \cdot A \cdot \nabla\log\pi]$$

**标准 RL with Importance Sampling**（无偏，高方差）

### 11.2 γ+ → 0, γ- → 2（极端调整）

**对于 A > 0**（γ+ → 0）：
$$\nabla L_+ = \mathbb{E}_\mu[A \cdot \nabla\log\pi]$$

这是 **Advantage-Weighted SFT**（忽略 IS，Forward KL 行为）

**对于 A < 0**（γ- → 2）：
$$\nabla L_- = \mathbb{E}_\mu[w^2 \cdot A \cdot \nabla\log\pi]$$

这是 **强化惩罚**（χ² 散度行为，对 w > 1 的负样本放大权重）

### 11.3 谱系图

```
γ+ (正样本)                           γ- (负样本)
    │                                     │
    │  γ+ → 0: Weighted SFT              │  γ- = 1: RL (IS)
    │  (Forward KL, mean-seeking)         │  (Reverse KL)
    │        │                            │      │
    │        ↓                            │      ↓
    │  γ+ ∈ (0,1): 凹函数                 │  γ- ∈ (1,2): 凸函数
    │  放大 w<1, 缩小 w>1                 │  缩小 w<1, 放大 w>1
    │        │                            │      │
    │        ↓                            │      ↓
    │  γ+ = 1: RL (IS)                   │  γ- → 2: χ² 散度
    │  (Reverse KL)                       │  (强惩罚)
    └────────┴────────────────────────────┴──────┘
```

---

## 12. 与现有方法的联系

| 方法 | 在本框架中的位置 | γ+ | γ- |
|------|-----------------|----|----|
| **RL with IS** | 两端都是 RL | 1 | 1 |
| **Advantage-Weighted SFT** | 只处理正样本 | 0 | N/A |
| **PPO (clipping)** | 隐式的 γ 截断 | ~0.8 | ~0.8 |
| **GRPO** | 组内归一化后的 IS | ~1 | ~1 |
| **IS-Reshape** | 正负样本分别最优 | $\gamma_+ \in (0,1)$ | $\gamma_- \in (1,2)$ |

---

# 第五部分：算法实现

## 13. 完整算法

```python
import torch
import math
from dataclasses import dataclass
from typing import Tuple, Dict

@dataclass
class ISReshapeConfig:
    """IS-Reshape 配置"""
    rho_min: float = 0.3        # 最小 ESS 比例
    eps: float = 1e-8           # 数值稳定性


class ISReshapeFinal:
    """
    IS-Reshape: 从 f-散度出发的统一框架

    核心公式：
        L(θ) = E_μ[(w^{γ(A)} - 1) / γ(A) · A]
        ∇L = E_μ[w^{γ(A)} · A · ∇log π]

    γ(A) 按 A 符号分组计算：
        - A > 0: γ+ ∈ (0, 1)，凹函数
        - A < 0: γ- ∈ (1, 2)，凸函数

    理论基础：
        - f(w) = (w^γ - 1)/γ 是 Box-Cox 变换
        - 梯度权重 φ(w) = w^γ 源自 f-散度结构
        - γ < 1: Forward KL 行为
        - γ > 1: 超 Reverse KL 行为
    """

    def __init__(self, config: ISReshapeConfig):
        self.config = config

    def compute_gamma_for_group(
        self,
        log_w: torch.Tensor,
        target_range: str  # 'pos' for (0,1), 'neg' for (1,2)
    ) -> float:
        """
        为一组样本计算最优 γ

        基于 ESS 约束：γ = √(-log ρ_min / σ²)
        """
        if len(log_w) < 2:
            return 0.5 if target_range == 'pos' else 1.5

        sigma_sq = torch.var(log_w).item()

        if sigma_sq < self.config.eps:
            # σ² 很小，可以用接近边界的值
            return 0.9 if target_range == 'pos' else 1.1

        log_rho = -math.log(self.config.rho_min)
        gamma_raw = math.sqrt(log_rho / sigma_sq)

        if target_range == 'pos':
            # γ+ ∈ (0, 1)
            gamma = min(0.99, gamma_raw)
            gamma = max(0.01, gamma)
        else:
            # γ- ∈ (1, 2)
            gamma = 1.0 + min(0.99, gamma_raw)
            gamma = min(1.99, gamma)

        return gamma

    def compute_loss(
        self,
        log_pi: torch.Tensor,
        log_mu: torch.Tensor,
        A: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        计算 IS-Reshape 损失

        L = -E[(w^γ - 1)/γ · A]  (负号因为要最大化)

        梯度通过 w^γ 自动传递，不需要 stop gradient
        """
        # log importance ratio
        log_w = log_pi - log_mu

        # 按 A 符号分组
        pos_mask = A > 0
        neg_mask = A < 0

        # 分别计算 γ+ 和 γ-
        gamma_pos = self.compute_gamma_for_group(
            log_w[pos_mask].detach(), 'pos'
        ) if pos_mask.any() else 0.5

        gamma_neg = self.compute_gamma_for_group(
            log_w[neg_mask].detach(), 'neg'
        ) if neg_mask.any() else 1.5

        # 构建 γ tensor
        gamma = torch.ones_like(A)
        gamma[pos_mask] = gamma_pos
        gamma[neg_mask] = gamma_neg

        # w^γ = exp(γ · log w)
        # 注意：log_w 不 detach，梯度通过 IS 传递
        w_gamma = torch.exp(gamma * log_w)

        # Box-Cox 目标：(w^γ - 1) / γ · A
        # 处理 γ 接近 0 的情况
        safe_gamma = torch.where(
            gamma.abs() < 0.01,
            torch.ones_like(gamma),
            gamma
        )

        box_cox = torch.where(
            gamma.abs() < 0.01,
            log_w,  # γ → 0 时的极限是 log w
            (w_gamma - 1) / safe_gamma
        )

        objective = box_cox * A

        # 最大化目标 = 最小化负目标
        loss = -objective.mean()

        # 诊断信息
        with torch.no_grad():
            metrics = {
                'gamma_pos': gamma_pos,
                'gamma_neg': gamma_neg,
                'n_pos': pos_mask.sum().item(),
                'n_neg': neg_mask.sum().item(),
                'sigma_sq_pos': torch.var(log_w[pos_mask]).item() if pos_mask.any() else 0,
                'sigma_sq_neg': torch.var(log_w[neg_mask]).item() if neg_mask.any() else 0,
                'w_gamma_mean': w_gamma.mean().item(),
                'w_gamma_max': w_gamma.max().item(),
                'objective_mean': objective.mean().item(),
            }

        return loss, metrics


# 便捷函数
def is_reshape_loss(
    log_pi: torch.Tensor,
    log_mu: torch.Tensor,
    advantages: torch.Tensor,
    rho_min: float = 0.3,
) -> torch.Tensor:
    """
    快速计算 IS-Reshape 损失（按 A 符号分组）

    Args:
        log_pi: 当前策略的 log 概率
        log_mu: 行为策略的 log 概率
        advantages: Advantage 值
        rho_min: 最小 ESS 比例

    Returns:
        loss: 标量损失（要最小化）
    """
    log_w = log_pi - log_mu
    log_rho = -math.log(rho_min)

    # 分组
    pos_mask = advantages > 0
    neg_mask = advantages < 0

    # 计算各组的 σ²
    sigma_sq_pos = torch.var(log_w[pos_mask]).item() if pos_mask.any() else 1.0
    sigma_sq_neg = torch.var(log_w[neg_mask]).item() if neg_mask.any() else 1.0

    # 计算 γ
    gamma_pos = min(0.99, math.sqrt(log_rho / max(sigma_sq_pos, 1e-8)))
    gamma_neg = 1.0 + min(0.99, math.sqrt(log_rho / max(sigma_sq_neg, 1e-8)))

    # 构建 γ tensor
    gamma = torch.ones_like(advantages)
    gamma[pos_mask] = gamma_pos
    gamma[neg_mask] = gamma_neg

    # w^γ
    w_gamma = torch.exp(gamma * log_w)

    # (w^γ - 1) / γ · A
    box_cox = torch.where(
        gamma.abs() < 0.01,
        log_w,
        (w_gamma - 1) / gamma
    )

    loss = -(box_cox * advantages).mean()

    return loss
```

---

## 14. 理论框架总图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    IS-Reshape: 从策略梯度到 f-散度                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  【起点】策略梯度                                                         │
│                                                                         │
│      g* = ∇_θ E_{π_θ}[A] = E_{π_θ}[A · ∇log π]                         │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  【两端】RL vs Weighted SFT                                              │
│                                                                         │
│      RL:  E_μ[w · A · ∇log π]     ←  f(w) = w      (无偏, 高方差)        │
│      SFT: E_μ[A · ∇log π]         ←  f(w) = log w  (有偏, 低方差)        │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  【f-散度联系】                                                           │
│                                                                         │
│      目标函数 L = E_μ[f(w) · A]                                          │
│      梯度权重 φ(w) = f'(w) · w                                           │
│                                                                         │
│      选择 f(w) = (w^γ - 1)/γ (Box-Cox)  →  φ(w) = w^γ                  │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  【核心创新】价值效率比（VER）统一准则                                      │
│                                                                         │
│      ┌──────────────────┬──────────────────┐                           │
│      │    正样本 A > 0   │    负样本 A < 0   │                           │
│      ├──────────────────┼──────────────────┤                           │
│      │ 目标: 学习好行为   │ 目标: 纠正坏行为   │                           │
│      │ 有价值: w < 1     │ 有价值: w > 1     │                           │
│      │ 准则: VER+ 最大化 │ 准则: VER- 最大化 │                           │
│      │ VER+: 递减于 γ   │ VER-: 递增于 γ   │                           │
│      │ 结果: γ ∈ (0,1)  │ 结果: γ ∈ (1,2)  │                           │
│      └──────────────────┴──────────────────┘                           │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  【价值效率比 VER】                                                       │
│                                                                         │
│      VER+(γ) = E[w^γ·A·𝟙(w<1)] / E[w^γ·A]  ← 递减于 γ → 取下界        │
│      VER-(γ) = E[w^γ·|A|·𝟙(w>1)] / E[w^γ·|A|]  ← 递增于 γ → 取上界    │
│                                                                         │
│      定理 8.10: VER 单调性 → 最优解在 ESS 约束边界                        │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  【最终形式】                                                             │
│                                                                         │
│      L(θ) = E_μ[(w^{γ(A)} - 1) / γ(A) · A]                              │
│      ∇L  = E_μ[w^{γ(A)} · A · ∇log π]                                   │
│                                                                         │
│      γ+ = min(1-ε, √(-log ρ_min / σ+²))     ← VER+ 最大化（取下界）      │
│      γ- = 1 + min(1-ε, √(-log ρ_min / σ-²)) ← VER- 最大化（取上界）      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 附录 A：关键定理证明

### A.1 Box-Cox 极限

$$\lim_{\gamma \to 0} \frac{w^\gamma - 1}{\gamma} = \log w$$

**证明**：$w^\gamma = e^{\gamma \log w} = 1 + \gamma \log w + O(\gamma^2)$，代入得结果。

### A.2 f-散度梯度

$$\nabla_\theta D_f(\pi_\theta \| \mu) = \mathbb{E}_\mu[f'(w) \cdot w \cdot \nabla_\theta \log\pi_\theta]$$

**证明**：链式法则 + $\nabla_\theta w = w \cdot \nabla_\theta \log\pi$。

### A.3 Rényi 散度恒等式

$$\mathbb{E}_\mu[w^\alpha] = \exp((\alpha - 1) D_\alpha^R(\pi \| \mu))$$

**证明**：由 Rényi 散度定义 $D_\alpha^R = \frac{1}{\alpha-1} \log \mathbb{E}_\mu[w^\alpha]$ 直接移项。

---

## 附录 B：符号表

| 符号 | 定义 |
|------|------|
| μ | 行为策略（数据分布） |
| π_θ | 学习策略 |
| w = π/μ | 重要性权重 |
| A | Advantage |
| γ(A) | A 依赖的 reshape 参数 |
| γ+ | 正样本的 γ 值（VER+ 最大化，∈ (0,1)） |
| γ- | 负样本的 γ 值（VER- 最大化，∈ (1,2)） |
| $\text{VER}_+(\gamma)$ | 正样本价值效率比（递减于 γ） |
| $\text{VER}_-(\gamma)$ | 负样本价值效率比（递增于 γ） |
| $D_\alpha^R$ | α 阶 Rényi 散度 |
| $D_f$ | f-散度 |
| $f_\gamma(w) = (w^\gamma-1)/\gamma$ | Box-Cox 变换 |
| $\phi_\gamma(w) = w^\gamma$ | 梯度权重函数 |
| σ² | log w 的方差 |
| ρ_min | 最小 ESS 比例约束 |
| MSE | 均方误差（Mean Squared Error） |
| VER | 价值效率比（Value Efficiency Ratio） |
| ESS | 有效样本量（Effective Sample Size） |
