# IS Reshape：从 f-Divergence 到局部化核的统一理论

**版本**: 6.0
**核心思想**: 钟形梯度权重源于 Importance Sampling 的**局部化原理**
**状态**: 理论推导

---

## 第一部分：从 f-Divergence 出发

### 1.1 f-Divergence 回顾

f-divergence 的一般形式：
$$D_f(p \| q) = \mathbb{E}_q\left[f\left(\frac{p}{q}\right)\right] = \mathbb{E}_q[f(r)]$$

其中 $r = p/q$ 是似然比，$f$ 是凸函数且 $f(1) = 0$。

**特殊情况**：
| f(r) | Divergence |
|------|------------|
| $r \log r$ | Forward KL |
| $-\log r$ | Reverse KL |
| $(r-1)^2$ | χ² divergence |
| $(\sqrt{r} - 1)^2$ | Hellinger |

### 1.2 策略优化的目标

我们的目标是：
$$\max_\theta \mathbb{E}_{\pi_\theta}[A] = \max_\theta \mathbb{E}_\mu[r \cdot A]$$

其中 $r = \pi_\theta / \mu$，$A$ 是 advantage。

**梯度**：
$$\nabla_\theta J = \mathbb{E}_\mu[r \cdot A \cdot \nabla\log\pi_\theta]$$

问题：当 $r$ 很大时，方差爆炸。

### 1.3 广义目标函数

考虑用 f(r) 替代 r：
$$L = -\mathbb{E}_\mu[f(r) \cdot A]$$

梯度：
$$\nabla L = -\mathbb{E}_\mu\left[\frac{\partial f(r)}{\partial \theta} \cdot A\right] = -\mathbb{E}_\mu[f'(r) \cdot r \cdot A \cdot \nabla\log\pi_\theta]$$

**定义梯度权重**：
$$\boxed{g(r) = f'(r) \cdot r}$$

| f(r) | g(r) = f'(r)·r | 方法 |
|------|----------------|------|
| $\log r$ | 1 | SFT |
| $r$ | $r$ | RL (IS) |
| **?** | **钟形** | **IS Reshape** |

**核心问题**：什么样的 f(r) 给出钟形的 g(r)？

---

## 第二部分：局部化原理

### 2.1 为什么需要钟形权重？

在 off-policy 学习中：
- **r ≈ 1**：样本来自当前策略，估计可靠
- **r >> 1**：样本在 π_θ 下过于常见，方差高
- **r << 1**：样本在 π_θ 下过于罕见，代表性差

**局部化原理**：给 r ≈ 1 的样本高权重，给 |r - 1| 大的样本低权重。

这自然导向**钟形**权重函数！

### 2.2 局部化核

**定义**：局部化核 $K: \mathbb{R} \to [0, 1]$ 满足：
1. $K(0) = 1$（中心最大）
2. $K(x) \to 0$ 当 $|x| \to \infty$（远处衰减）
3. $K(-x) = K(x)$（对称，可选）

**例子**：

| 核 | $K(x)$ | 特点 |
|----|--------|------|
| Gaussian | $e^{-x^2/2\sigma^2}$ | 平滑，快速衰减 |
| Laplace | $e^{-|x|/b}$ | 尖峰，指数衰减 |
| Cauchy | $1/(1 + x^2/\gamma^2)$ | 重尾 |
| Logistic (sech²) | $\text{sech}^2(x/\tau)$ | 平滑，中等衰减 |
| Epanechnikov | $(1 - x^2/h^2)_+$ | 有界支撑 |

### 2.3 从核到 f-Divergence

给定核 $K$，构造：
$$\boxed{f_K(r) = \int_1^r K(t - 1) \, dt}$$

则：
$$f'_K(r) = K(r - 1)$$
$$g_K(r) = r \cdot K(r - 1)$$

**验证**：
- $g_K(1) = 1 \cdot K(0) = 1$（on-policy 权重为 1）
- $g_K(r) \to 0$ 当 $r \to \infty$（off-policy 权重衰减）

---

## 第三部分：具体核函数分析

### 3.1 Gaussian 核

$$K_G(x) = e^{-x^2/2\sigma^2}$$

**f 函数**：
$$f_G(r) = \int_1^r e^{-(t-1)^2/2\sigma^2} dt = \sigma\sqrt{\frac{\pi}{2}} \cdot \text{erf}\left(\frac{r-1}{\sigma\sqrt{2}}\right)$$

**梯度权重**：
$$g_G(r) = r \cdot e^{-(r-1)^2/2\sigma^2}$$

**特点**：
- 在 r=1 处峰值 = 1
- 衰减速度由 σ 控制
- 快速衰减（Gaussian 尾）

### 3.2 Logistic (sech²) 核

$$K_L(x) = \text{sech}^2(x/\tau) = \frac{4}{(e^{x/\tau} + e^{-x/\tau})^2}$$

**f 函数**：
$$f_L(r) = \int_1^r \text{sech}^2\left(\frac{t-1}{\tau}\right) dt = \tau \cdot \tanh\left(\frac{r-1}{\tau}\right)$$

**梯度权重**：
$$g_L(r) = r \cdot \text{sech}^2\left(\frac{r-1}{\tau}\right)$$

**特点**：
- 这就是 SAPO 使用的形式！
- 中等衰减速度
- f(r) 有界：$|f_L(r)| \leq \tau$

### 3.3 Cauchy 核（重尾）

$$K_C(x) = \frac{1}{1 + x^2/\gamma^2}$$

**f 函数**：
$$f_C(r) = \gamma \cdot \arctan\left(\frac{r-1}{\gamma}\right)$$

**梯度权重**：
$$g_C(r) = \frac{r}{1 + (r-1)^2/\gamma^2}$$

**特点**：
- 重尾：衰减比 Gaussian 和 sech² 慢
- 对 off-policy 样本更宽容

### 3.4 Laplace 核

$$K_{Lap}(x) = e^{-|x|/b}$$

**f 函数**：
$$f_{Lap}(r) = \begin{cases}
b(1 - e^{-(r-1)/b}) & r \geq 1 \\
-b(1 - e^{(r-1)/b}) & r < 1
\end{cases}$$

**梯度权重**：
$$g_{Lap}(r) = r \cdot e^{-|r-1|/b}$$

**特点**：
- 在 r=1 处不可微（尖峰）
- 指数衰减

### 3.5 核函数对比

在 r = 3 时（中等 off-policy）：

| 核 | 参数 | K(r-1) | g(r) = r·K |
|----|------|--------|------------|
| Gaussian | σ=1 | 0.135 | 0.41 |
| sech² | τ=1 | 0.420 | 1.26 |
| Cauchy | γ=1 | 0.200 | 0.60 |
| Laplace | b=1 | 0.135 | 0.41 |

**观察**：sech² 核比其他核更"宽容"，给 off-policy 样本更高的权重。

---

## 第四部分：统一的 SFT-RL 插值

### 4.1 混合公式

将核方法与 SFT 混合：
$$\boxed{g_\gamma(r) = (1 - \gamma) + \gamma \cdot r \cdot K(r - 1)}$$

**极限行为**：
- $\gamma = 0$：$g(r) = 1$（SFT）
- $\gamma = 1$：$g(r) = r \cdot K(r-1)$（带核的 RL）

### 4.2 对应的 f 函数

$$f_\gamma(r) = (1 - \gamma) \log r + \gamma \cdot F_K(r)$$

其中 $F_K(r) = \int_1^r K(t-1) dt$。

**物理意义**：
- $(1-\gamma) \log r$：reverse KL 相关项（稳定性）
- $\gamma \cdot F_K(r)$：局部化 IS 项（学习）

### 4.3 f-Divergence 族的解释

定义混合 f-divergence：
$$D_\gamma(\pi \| \mu) = \mathbb{E}_\mu[f_\gamma(r)]$$

这是一个从 SFT 目标到带核 RL 目标的连续谱！

---

## 第五部分：核函数的选择原理

### 5.1 理论约束

一个"好"的核应满足：

**必要条件**：
1. $K(0) = 1$（on-policy 归一化）
2. $K(x) \geq 0$（非负权重）
3. $K(x) \to 0$ 当 $|x| \to \infty$（局部化）

**期望性质**：
4. $K$ 处处可微（梯度连续）
5. $K$ 单峰（无振荡）
6. 衰减速度可调（通过参数）

### 5.2 从 Bias-Variance 角度选择

IS 估计的 MSE = Bias² + Variance

**Variance** ∝ $\mathbb{E}_\mu[g(r)^2]$

**Bias** = $\mathbb{E}_\mu[(g(r) - r) \cdot A]$

最小化 MSE 在 variance 约束下：
$$\min_g \text{Bias}^2 \quad \text{s.t.} \quad \mathbb{E}[g(r)^2] \leq V_{max}$$

Lagrangian 方法给出：
$$g^*(r) \propto \frac{r}{1 + \lambda \cdot \text{Var}(A|r)}$$

当 $\text{Var}(A|r)$ 与 $|r-1|$ 正相关时，这自然给出钟形权重。

### 5.3 信息论角度

从信息论角度，off-policy 样本的"有效信息量"随 |r-1| 衰减。

定义有效信息：
$$I_{eff}(r) = I(r) \cdot \text{reliability}(r)$$

其中 reliability(r) 是关于 r 的递减函数（对于 |r-1|）。

核函数 K(r-1) 可以解释为 reliability 函数。

---

## 第六部分：一般化的 IS Reshape 框架

### 6.1 最一般形式

**梯度权重**：
$$g(r, A) = (1 - \gamma(r, A)) + \gamma(r, A) \cdot r \cdot K(r - 1)$$

**自由度**：
1. **核函数 K**：Gaussian, sech², Cauchy, etc.
2. **混合参数 γ**：可以是常数，也可以依赖 (r, A)
3. **核参数**：σ, τ, γ, b 等控制信任域宽度

### 6.2 γ(r, A) 的设计

**选项 1：固定 γ**
$$\gamma = \gamma_0 \in [0, 1]$$

简单，但不适应样本特性。

**选项 2：基于 Correctness**
$$\gamma(r, A) = \gamma_{base} + (\gamma_{max} - \gamma_{base}) \cdot \sigma\left(-\frac{A \cdot \log r}{\tau_c}\right)$$

当策略"错误"时更 RL-like。

**选项 3：基于 |A|**
$$\gamma(|A|) = \sigma(\beta |A|)$$

reward 信号强时更 RL-like。

**选项 4：结合**
$$\gamma(r, A) = h(|A|) \cdot P_{wrong}(r, A)$$

### 6.3 四象限验证（对任意核）

设 K 满足基本条件，则：

| Case | r | A | C=A·log r | γ (correctness) | g(r,A) | 行为 |
|------|---|---|-----------|-----------------|--------|------|
| 1 | >1 | >0 | + | low | ≈ 1 | 稳定 |
| 2 | >1 | <0 | - | high | ≈ r·K | 修正 |
| 3 | <1 | >0 | - | high | ≈ r·K | 学习 |
| 4 | <1 | <0 | + | low | ≈ 1 | 稳定 |

关键：无论选择哪个核 K，只要满足局部化原理，四象限行为都是合理的！

---

## 第七部分：与已有方法的统一

### 7.1 PPO Clip 作为极限情况

PPO clip 可以看作 Epanechnikov 核的极限：
$$K_{PPO}(x) = \mathbb{1}[|x| \leq \epsilon]$$

这是一个"硬"核，不满足可微性，但满足局部化原理。

### 7.2 SAPO 作为特例

SAPO = sech² 核 + 固定 γ=1：
$$g_{SAPO}(r) = r \cdot \text{sech}^2\left(\frac{\tau(r-1)}{2}\right)$$

### 7.3 v3.1 的问题

v3.1 使用 $g(r) = r^\gamma$，这不是局部化核的形式：
- 当 r → ∞：$r^\gamma \to \infty$（不衰减！）
- 违反了局部化原理

这就是 v3.1 失败的根本原因。

### 7.4 统一视角

| 方法 | 核 K | γ | 稳定性 |
|------|------|---|--------|
| SFT | - | 0 | ✓✓✓ |
| PPO clip | Hard indicator | 1 (implicit) | ✓✓ |
| SAPO | sech² | 1 | ✓✓ |
| v3.1 | **无（不局部化）** | adaptive | ✗ |
| **v6** | **任意局部化核** | **adaptive** | **✓✓** |

---

## 第八部分：实践建议

### 8.1 核函数选择

| 场景 | 推荐核 | 理由 |
|------|--------|------|
| 一般情况 | sech² | 平滑，中等衰减，SAPO 验证过 |
| 需要更宽容 | Cauchy | 重尾，对 off-policy 更友好 |
| 需要更严格 | Gaussian | 快速衰减，强信任域 |
| 理论分析 | Gaussian | 解析性质好 |

### 8.2 参数设置

**核宽度参数**（σ, τ, γ, b）：
- 越大 → 信任域越宽 → 更 RL-like
- 越小 → 信任域越窄 → 更 SFT-like
- 建议：τ ∈ [0.5, 2.0]

**混合参数 γ**：
- γ 固定：简单，但不够灵活
- γ 自适应：根据 (r, A) 调整，更灵活

### 8.3 数值稳定性

对于任意局部化核，当 r → ∞：
$$g(r) = (1-\gamma) + \gamma \cdot r \cdot K(r-1) \to (1-\gamma)$$

因为 $r \cdot K(r-1) \to 0$（K 衰减速度快于 r 增长速度，对于良好的核）。

这保证了梯度有界！

---

## 第九部分：总结

### 9.1 核心贡献

1. **局部化原理**：钟形梯度权重源于 IS 的局部化需求
2. **核函数框架**：任意满足局部化条件的核都可用
3. **f-divergence 连接**：$f_K(r) = \int_1^r K(t-1) dt$ 建立了核与 f-divergence 的联系
4. **统一视角**：PPO clip、SAPO、IS Reshape 都是该框架的特例

### 9.2 关键公式

$$\boxed{g(r) = (1 - \gamma) + \gamma \cdot r \cdot K(r - 1)}$$

$$\boxed{f(r) = (1 - \gamma) \log r + \gamma \cdot \int_1^r K(t-1) dt}$$

### 9.3 设计自由度

1. **核函数 K**：Gaussian, sech², Cauchy, Laplace, ...
2. **核参数**：控制信任域宽度
3. **混合参数 γ**：固定或自适应

### 9.4 理论保证

只要 K 满足：
- K(0) = 1
- K(x) → 0 (|x| → ∞)
- K(x) ≥ 0

则：
- 梯度有界 ✓
- 四象限行为正确 ✓
- SFT-RL 统一 ✓
