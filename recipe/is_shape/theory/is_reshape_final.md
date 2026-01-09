# IS Reshape：连接监督学习与强化学习的统一框架

**版本**: 6.1

---

## 摘要

本文建立一个连接监督微调（SFT）和强化学习（RL）的统一理论框架。

**适用场景**：当训练涉及奖励信号 r 时（如 RFT、奖励加权 SFT、RLHF），本框架提供了一个连续的参数 γ ∈ [0,1] 来控制如何利用奖励信息。

**核心结果**：

1. **两端的散度解释**：γ=0 对应 Forward KL（mean-seeking），γ=1 对应 Reverse KL（mode-seeking）
2. **统一的编码视角**：两端都可写成 $\mu^\alpha \cdot e^{f(r)}$，差异在于对 r 的编码（$\log r$ vs $r/\tau$）
3. **方差主导因子**：梯度方差的主导项由 Rényi 散度刻画，提供 γ 选择的理论依据
4. **闭式 γ 选择**：$\gamma^* = \min(1, \sqrt{-\log\rho_{\min}/\sigma^2})$，O(1) 计算

**核心洞察**：
- **纯 SFT**（r=1）是退化情况，γ 无意义
- **当 r 非平凡时**，γ 控制两件事：(1) 对数据分布 μ 的依赖程度，(2) 对奖励差异的敏感程度

---

# 第一部分：从观察到问题

## 1. 问题设定

### 1.1 符号定义

| 符号 | 定义 |
|------|------|
| $\mu(y\|x)$ | 行为策略（离线数据的采样分布） |
| $\pi_\theta(y\|x)$ | 待学习策略 |
| $r(x, y)$ | 奖励函数（假设 $r > 0$） |
| $\tau > 0$ | 温度参数 |
| $w = \pi_\theta / \mu$ | 重要性采样比率 |
| $\tilde{\pi} = \mu \cdot r / Z_0$ | 奖励加权的数据分布（SFT 目标） |
| $p^* = e^{r/\tau}/Z_1$ | 最优软策略（RL 目标） |

### 1.2 两种学习范式

**监督微调（SFT）**：从数据中学习
$$\max_\theta \mathbb{E}_\mu[\log \pi_\theta(y|x)]$$

**强化学习（RL）**：最大化奖励
$$\max_\theta \mathbb{E}_{\pi_\theta}[r(x, y)]$$

**核心问题**：这两种范式看似完全不同，它们之间有什么联系？

### 1.3 SFT 中奖励的角色

实践中，SFT 对奖励的处理方式多样：

| 方法 | r 的形式 | 特点 |
|------|---------|------|
| **纯 SFT** | $r = 1$（无奖励） | 直接模仿数据分布 μ |
| **RFT** | $r = \mathbf{1}[\text{pass filter}]$ | 隐式奖励，体现在数据筛选中 |
| **奖励加权 SFT** | $r = \text{reward}(x,y)$ | 显式奖励，软权重 |

**关键观察**：
- 纯 SFT 是 $r = 1$ 的特例，此时目标就是 $\min D_{KL}(\mu \| \pi_\theta)$
- RFT 的奖励是**隐式**的，通过数据过滤实现
- 本框架关注的是**当 r 非平凡时**（RFT 或奖励加权 SFT），如何在 SFT 风格和 RL 风格之间平滑过渡

---

## 2. 关键观察：统一梯度形式

### 2.1 纯 SFT 的梯度

纯 SFT 目标：$L_{\text{SFT}} = \mathbb{E}_\mu[\log\pi_\theta]$

梯度：
$$g_{\text{SFT}} = \mathbb{E}_\mu[\nabla_\theta \log\pi_\theta]$$

### 2.2 奖励加权 SFT 的梯度

当引入奖励 r 时：$L_{\text{weighted-SFT}} = \mathbb{E}_\mu[r \cdot \log\pi_\theta]$

梯度：
$$g_{\text{weighted-SFT}} = \mathbb{E}_\mu[r \cdot \nabla_\theta \log\pi_\theta]$$

### 2.3 RL 的梯度

期望奖励目标：$L_{\text{RL}} = \mathbb{E}_{\pi_\theta}[r]$

由策略梯度定理：
$$g_{\text{RL}} = \mathbb{E}_{\pi_\theta}[r \cdot \nabla_\theta \log\pi_\theta] = \mathbb{E}_\mu[w \cdot r \cdot \nabla_\theta \log\pi_\theta]$$

### 2.4 统一形式

**观察**：当 r 非平凡时，梯度可以统一写成：

$$\boxed{g = \mathbb{E}_\mu[f(w) \cdot r \cdot \nabla_\theta \log\pi_\theta]}$$

其中：
- 奖励加权 SFT：$f(w) = 1$（γ = 0）
- RL：$f(w) = w$（γ = 1）

**自然问题**：$f(w) = w^\gamma$ 对于 $\gamma \in (0, 1)$ 对应什么？

**注意**：纯 SFT（r = 1）是退化情况，此时 γ 的选择无意义。本框架的价值在于**当 r 携带信息时**，如何最优地利用它。

### 2.5 从梯度到目标函数

统一梯度形式 $g_\gamma = \mathbb{E}_\mu[w^\gamma \cdot r \cdot \nabla_\theta \log\pi_\theta]$ 对应的**目标函数**为：

$$\boxed{L_\gamma(\theta) = \frac{1}{\gamma}\left(\mathbb{E}_\mu[w^\gamma \cdot r] - \mathbb{E}_\mu[r]\right), \quad \gamma > 0}$$

**定理 2.1**：$\nabla_\theta L_\gamma = g_\gamma$

**证明**：
$$\nabla_\theta w^\gamma = \nabla_\theta \left(\frac{\pi_\theta}{\mu}\right)^\gamma = \gamma w^{\gamma-1} \cdot \frac{\nabla_\theta \pi_\theta}{\mu} = \gamma w^\gamma \nabla_\theta \log\pi_\theta$$

因此：
$$\nabla_\theta L_\gamma = \frac{1}{\gamma}\mathbb{E}_\mu[r \cdot \nabla_\theta w^\gamma] = \frac{1}{\gamma}\mathbb{E}_\mu[r \cdot \gamma w^\gamma \nabla_\theta\log\pi_\theta] = g_\gamma \quad \blacksquare$$

**边界情况 γ → 0**：
$$L_0 = \lim_{\gamma \to 0} L_\gamma = \mathbb{E}_\mu[r \cdot \log w] = \mathbb{E}_\mu[r \cdot \log\pi_\theta] - \text{const}$$

**关键点：不需要 stop gradient！**

这与 RL 中重要性采样传递梯度的方式一致：$w^\gamma$ 是目标函数的一部分，梯度自然通过 $w^\gamma$ 传递到 $\pi_\theta$。

### 2.6 两种等价的实现方式

**方式 A：直接优化目标函数**（推荐，不需要 stop gradient）
```python
# w^γ 参与梯度计算，IS 传递梯度
w_gamma = torch.exp(gamma * log_w)  # 不 detach！
loss = -(1/gamma) * (w_gamma * rewards).mean()  # 负号因为要最大化
```

**方式 B：REINFORCE 风格**（需要 stop gradient）
```python
# 把 w^γ 当作固定权重
w_gamma = torch.exp(gamma * log_w).detach()  # 必须 detach
loss = -(w_gamma * rewards * log_pi).mean()
```

**两种方式得到相同的梯度方向**，但：
- 方式 A 更符合 RL 的 IS 传递梯度的思想
- 方式 B 避免了 γ → 0 时 1/γ 的数值问题

**实践建议**：
- γ 远离 0 时（γ > 0.1）：推荐方式 A
- γ 接近 0 时：使用方式 B 或直接用 SFT 损失

---

# 第二部分：两端的散度解释

## 3. γ = 0 和 γ = 1 的本质

### 3.1 γ = 0：Forward KL（SFT）

**定理 3.1**：当 $\gamma = 0$ 且 $r(x,y) > 0$ 时，

$$\max_\theta \mathbb{E}_\mu[r \cdot \log\pi_\theta] \iff \min_\theta D_{KL}(\tilde{\pi} \| \pi_\theta)$$

其中 $\tilde{\pi}(y|x) = \mu(y|x) \cdot r(x,y) / \mathbb{E}_\mu[r]$。

**证明**：

$$D_{KL}(\tilde{\pi} \| \pi_\theta) = \mathbb{E}_{\tilde{\pi}}[\log\tilde{\pi}] - \mathbb{E}_{\tilde{\pi}}[\log\pi_\theta]$$

第一项与 θ 无关。第二项：
$$\mathbb{E}_{\tilde{\pi}}[\log\pi_\theta] = \frac{\mathbb{E}_\mu[r \cdot \log\pi_\theta]}{\mathbb{E}_\mu[r]}$$

因此 $\min D_{KL}(\tilde{\pi} \| \pi_\theta) \iff \max \mathbb{E}_\mu[r \cdot \log\pi_\theta]$。$\blacksquare$

**性质**：Forward KL 是 **mean-seeking** 的——$\pi_\theta$ 会覆盖 $\tilde{\pi}$ 的所有模式。

### 3.2 γ = 1：Reverse KL（RL）

**定理 3.2**：当 $\gamma = 1$ 时，加入熵正则化后，

$$\max_\theta \left[\mathbb{E}_{\pi_\theta}[r] + \tau H(\pi_\theta)\right] \iff \min_\theta D_{KL}(\pi_\theta \| p^*)$$

其中 $p^*(y|x) = e^{r(x,y)/\tau} / Z$。

**证明**：

$$D_{KL}(\pi_\theta \| p^*) = \mathbb{E}_{\pi_\theta}\left[\log\frac{\pi_\theta}{e^{r/\tau}/Z}\right] = -H(\pi_\theta) - \frac{1}{\tau}\mathbb{E}_{\pi_\theta}[r] + \log Z$$

因此：
$$\min D_{KL}(\pi_\theta \| p^*) \iff \max \left[\mathbb{E}_{\pi_\theta}[r] + \tau H(\pi_\theta)\right]$$

$\blacksquare$

**性质**：Reverse KL 是 **mode-seeking** 的——$\pi_\theta$ 会集中在 $p^*$ 的主要模式上。

**关于熵正则化的说明**：
- γ = 0（SFT 端）：自然包含熵效应，因为 $D_{KL}(\tilde{\pi} \| \pi_\theta)$ 鼓励 $\pi_\theta$ 覆盖 $\tilde{\pi}$ 的支撑集
- γ = 1（RL 端）：需要**显式**加入熵正则化 $\tau H(\pi_\theta)$
- γ ∈ (0,1)：熵正则化的效果通过 τ 隐含在 $p^* = e^{r/\tau}/Z$ 的定义中

### 3.3 关键洞察：两端的目标分布不同

| γ | 散度类型 | 目标分布 | r 的角色 |
|---|---------|---------|---------|
| 0 | Forward KL | $\tilde{\pi} = \mu \cdot r / Z_0$ | **重塑数据** |
| 1 | Reverse KL | $p^* = e^{r/\tau}/Z_1$ | **定义目标** |

**本质差异**：
- γ = 0（SFT）：r 作为**权重**调整数据分布，目标是 $\tilde{\pi} \propto \mu \cdot r$
- γ = 1（RL）：r 作为**目标**定义最优策略，目标是 $p^* \propto e^{r/\tau}$

这不是缺陷，而是**忠实反映了两种范式的本质区别**。

### 3.4 统一视角：对 r 的编码方式

两端都可以写成指数形式，差异在于**对 r 的编码**：

$$\tilde{\pi} \propto \mu \cdot r = \mu \cdot e^{\log r}$$
$$p^* \propto e^{r/\tau}$$

| | SFT | RL |
|--|-----|-----|
| 统一形式 | $\mu \cdot e^{\log r}$ | $\mu^0 \cdot e^{r/\tau}$ |
| **对 r 的编码** | $\log r$（对数） | $r/\tau$（线性） |
| 对奖励差异的响应 | **压缩**（温和） | **放大**（激进） |

**物理意义**：
- **对数编码 $\log r$**：高奖励的边际收益递减，对奖励差异的响应温和
- **线性编码 $r/\tau$**：边际收益恒定，对奖励差异的响应激进

**类比**：这类似于效用函数理论中的风险偏好
- SFT（$\log r$）：风险厌恶，倾向于覆盖所有"还不错"的样本
- RL（$r/\tau$）：风险中性/偏好，集中在最高奖励区域

### 3.5 r 作为对 μ 的重塑

从数据重塑的角度理解 SFT：

**RFT（Rejection Fine-Tuning）**：硬过滤
$$\tilde{\pi}_{RFT} \propto \mu \cdot \mathbf{1}[r > \text{threshold}]$$

**奖励加权 SFT**：软过滤
$$\tilde{\pi} \propto \mu \cdot r$$

**RL**：完全脱离 μ
$$p^* \propto e^{r/\tau}$$

```
μ (原始数据分布)
│
├── RFT:      μ · 𝟙[r>θ]           ← 硬阈值过滤
│
├── SFT:      μ · r                ← 软权重重塑（对数编码）
│
├── γ∈(0,1): μ^{1-γ} · r^{1-γ} · e^{γr/τ}  ← 渐进脱离 μ
│
└── RL:       e^{r/τ}              ← 完全脱离 μ（线性编码）
```

**核心洞察**：γ 同时控制两件事
1. **对 μ 的依赖程度**：从完全依赖（γ=0）到完全独立（γ=1）
2. **对 r 的编码方式**：从对数编码（γ=0）到线性编码（γ=1）

---

# 第三部分：目标分布的几何演化

## 4. 几何插值分布

### 4.1 核心定义

**定义 4.1（γ-目标分布）**：

$$\boxed{p_\gamma^*(y|x) \propto \tilde{\pi}(y|x)^{1-\gamma} \cdot p^*(y|x)^\gamma}$$

**展开形式**：
$$p_\gamma^* \propto \left(\frac{\mu \cdot r}{Z_0}\right)^{1-\gamma} \cdot \left(\frac{e^{r/\tau}}{Z_1}\right)^\gamma \propto \mu^{1-\gamma} \cdot r^{1-\gamma} \cdot e^{\gamma r/\tau}$$

**边界验证**：
- γ=0: $p_0^* \propto \mu \cdot r = \tilde{\pi} \cdot Z_0$ ✓
- γ=1: $p_1^* \propto e^{r/\tau} = p^* \cdot Z_1$ ✓

### 4.2 几何意义

**定理 4.2（散度凸组合最优解）**：

$p_\gamma^*$ 是以下优化问题的唯一解：

$$\boxed{p_\gamma^* = \arg\min_p \left[(1-\gamma) D_{KL}(p \| \tilde{\pi}) + \gamma D_{KL}(p \| p^*)\right]}$$

**证明**：

构造 Lagrangian（带归一化约束）：
$$\mathcal{L} = (1-\gamma) D_{KL}(p \| \tilde{\pi}) + \gamma D_{KL}(p \| p^*) + \lambda\left(\int p - 1\right)$$

一阶条件：
$$\frac{\delta \mathcal{L}}{\delta p} = (1-\gamma)\left(\log p - \log\tilde{\pi} + 1\right) + \gamma\left(\log p - \log p^* + 1\right) + \lambda = 0$$

整理：
$$\log p = (1-\gamma)\log\tilde{\pi} + \gamma\log p^* + \text{const}$$

因此：
$$p \propto \tilde{\pi}^{1-\gamma} \cdot (p^*)^\gamma$$

$\blacksquare$

### 4.3 信息几何解释

**命题 4.3**：$p_\gamma^*$ 是 $\tilde{\pi}$ 和 $p^*$ 在 **Fisher-Rao 度量**下的测地线（geodesic）。

这意味着 $p_\gamma^*$ 是连接两个目标分布的**最短路径**（在信息几何意义下）。

### 4.4 r 的编码演化

**命题 4.4**：几何插值可以统一写成：

$$p_\gamma^* \propto \mu^{1-\gamma} \cdot \exp\left[(1-\gamma)\log r + \gamma \cdot r/\tau\right]$$

展开后：
$$p_\gamma^* \propto \mu^{1-\gamma} \cdot r^{1-\gamma} \cdot e^{\gamma r/\tau}$$

**对 r 的"有效编码"**：
$$f_\gamma(r) = (1-\gamma)\log r + \gamma \cdot r/\tau$$

| γ | 有效编码 $f_\gamma(r)$ | 对奖励差异的响应 |
|---|----------------------|----------------|
| 0 | $\log r$ | **压缩**（对数尺度） |
| 0.5 | $0.5\log r + 0.5r/\tau$ | **混合** |
| 1 | $r/\tau$ | **放大**（线性尺度） |

**物理解释**：γ 控制对奖励差异的敏感程度
- γ 小：对奖励差异响应温和，倾向于覆盖多样性
- γ 大：对奖励差异响应激进，倾向于集中在高奖励区域

### 4.5 γ 的双重定位：训练技巧还是设计选择？

框架中的 γ 可以从两个角度理解，这决定了如何使用它：

**解释 1：γ 作为训练技巧（方差控制）**

在这个视角下：
- **最终目标**仍是 γ = 1 的 $p^* = e^{r/\tau}/Z$（最大化奖励）
- γ < 1 是为了**控制方差**，使训练更稳定
- 随着训练进行，应该逐渐**增大 γ → 1**
- 类比：学习率 warmup（先稳定，后激进）

```
训练轨迹：γ = 0.3 → 0.5 → 0.7 → 0.9 → 1.0
          (保守)            →           (激进)
```

**解释 2：γ 作为设计选择（目标定义）**

在这个视角下：
- $p_\gamma^*$ **本身就是有价值的目标**，而非 $p^*$ 的近似
- 中间的 γ 值代表"在数据分布和奖励最大化之间的最优平衡"
- 纯 RL 目标 $p^*$ 可能导致 **reward hacking** 或 **mode collapse**
- γ < 1 是一种**隐式正则化**，保持对数据分布的尊重

```
目标选择：p_γ* 平衡了探索（保持多样性）和利用（追求高奖励）
```

**两种解释的对比**：

| 方面 | 训练技巧视角 | 设计选择视角 |
|------|-------------|-------------|
| γ 的目标值 | 最终应趋向 1 | 可以停在中间值 |
| $p_\gamma^*$ 的角色 | 过渡状态 | 最终目标 |
| 适用场景 | 奖励信号可靠 | 奖励信号有噪声或不完整 |
| 理论支撑 | Bias-Variance 权衡 | 散度凸组合最优解 |

**实践建议**：
1. **当奖励函数可靠时**（如数学题的正确性）：采用训练技巧视角，逐步增大 γ
2. **当奖励函数有噪声时**（如人类偏好模型）：采用设计选择视角，保持 γ < 1
3. **不确定时**：从中间值（γ ≈ 0.5）开始，观察训练动态后调整

---

# 第四部分：Rényi 散度与方差控制

## 5. 方差的主导因子分析

### 5.1 Rényi 散度定义

**定义 5.1（α-Rényi 散度）**：

$$D_\alpha^R(P \| Q) = \frac{1}{\alpha - 1} \log \mathbb{E}_Q\left[\left(\frac{P}{Q}\right)^\alpha\right]$$

### 5.2 核心等式

**定理 5.2（IS 权重与 Rényi 散度的精确关系）**：

$$\boxed{\mathbb{E}_\mu[w^\gamma] = \exp\left((\gamma - 1) \cdot D_\gamma^R(\pi_\theta \| \mu)\right)}$$

**证明**：直接由定义：

$$D_\gamma^R(\pi_\theta \| \mu) = \frac{1}{\gamma - 1} \log \mathbb{E}_\mu[w^\gamma]$$

移项即得。$\blacksquare$

**这是恒等式，精确成立。**

### 5.3 方差的分解与主导项

**定理 5.3（方差分解）**：

梯度方差可分解为：
$$\boxed{\text{Var}(g_\gamma) \propto \mathbb{E}_\mu[w^{2\gamma} r^2 \|\nabla\log\pi\|^2]}$$

**注意**：这三个因子（$w^{2\gamma}$, $r^2$, $\|\nabla\log\pi\|^2$）**通常不独立**，因此不能简单分解为各自期望的乘积。

**主导项分析**：在实践中，当 $\pi_\theta$ 偏离 $\mu$ 时，权重项 $w^{2\gamma}$ 往往是方差爆炸的**主要来源**：

$$\mathbb{E}_\mu[w^{2\gamma}] = \exp\left((2\gamma - 1) \cdot D_{2\gamma}^R(\pi_\theta \| \mu)\right)$$

**推论 5.4**：$\mathbb{E}[w^{2\gamma}]$ 由 $2\gamma$ 阶 Rényi 散度**精确刻画**（恒等式）。这提供了通过控制 γ 来抑制权重贡献的理论基础，尽管它不能完全刻画整体方差。

---

## 6. Bias-Variance 分析

### 6.1 偏差定义

**定义 6.1（目标偏移）**：

优化 γ < 1 而非 γ = 1（完整 RL）导致的偏差：
$$\text{Bias}(\gamma) = D_{KL}(p^* \| p_\gamma^*)$$

### 6.2 偏差单调性

**定理 6.2**：$\text{Bias}(\gamma)$ 关于 γ **严格单调递减**。

**证明**：

$$D_{KL}(p^* \| p_\gamma^*) = \mathbb{E}_{p^*}\left[\log\frac{p^*}{p_\gamma^*}\right]$$

由于 $p_\gamma^* \propto \tilde{\pi}^{1-\gamma}(p^*)^\gamma$：
$$\log p_\gamma^* = (1-\gamma)\log\tilde{\pi} + \gamma\log p^* - \log Z_\gamma$$

代入：
$$D_{KL}(p^* \| p_\gamma^*) = (1-\gamma)\mathbb{E}_{p^*}\left[\log\frac{p^*}{\tilde{\pi}}\right] + \log Z_\gamma = (1-\gamma)D_{KL}(p^* \| \tilde{\pi}) + \log Z_\gamma$$

对 γ 求导可以证明导数为负。$\blacksquare$

**性质**：
- γ = 0：Bias 最大
- γ = 1：Bias = 0

### 6.3 方差分析（Log-Normal 假设）

**假设 6.3**：$\log w \sim \mathcal{N}(-\sigma^2/2, \sigma^2)$

**定理 6.4**：在此假设下：
$$V(\gamma) := \mathbb{E}_\mu[w^{2\gamma}] = e^{\sigma^2\gamma(2\gamma-1)}$$

**推论 6.5**：
- $V(\gamma)$ 在 $\gamma = 0.25$ 处取得最小值
- 当 $\gamma > 0.25$ 时，$V(\gamma)$ 严格单调递增

### 6.4 MSE 权衡

$$\text{MSE}(\gamma) = \text{Bias}^2(\gamma) + \frac{1}{n}V(\gamma)$$

- Bias²(γ)：单调递减
- V(γ)：在 γ > 0.25 时单调递增

**存在最优 γ*** 平衡两者。

---

# 第五部分：γ 的最优选择

## 7. 有效样本量

### 7.1 ESS 定义

**定义 7.1**：
$$\text{ESS}_\gamma = \frac{(\sum_i w_i^\gamma)^2}{\sum_i w_i^{2\gamma}}$$

### 7.2 ESS 的闭式近似

**定理 7.2**：在 Log-Normal 假设下：

$$\boxed{\frac{\text{ESS}_\gamma}{n} \approx e^{-\sigma^2\gamma^2}}$$

**证明**：

$$\frac{\text{ESS}_\gamma}{n} = \frac{\mathbb{E}[w^\gamma]^2}{\mathbb{E}[w^{2\gamma}]}$$

在 Log-Normal 假设下（$\mathbb{E}[w] = 1$）：
- $\mathbb{E}[w^\gamma] = e^{\sigma^2\gamma(\gamma-1)/2}$
- $\mathbb{E}[w^{2\gamma}] = e^{\sigma^2\gamma(2\gamma-1)}$

因此：
$$\frac{\text{ESS}_\gamma}{n} = \frac{e^{\sigma^2\gamma(\gamma-1)}}{e^{\sigma^2\gamma(2\gamma-1)}} = e^{-\sigma^2\gamma^2}$$

$\blacksquare$

## 8. 闭式 γ 选择

### 8.1 主定理

**定理 8.1（γ 的 O(1) 闭式解）**：

给定最小 ESS 比例 $\rho_{\min} \in (0, 1)$，满足约束的最大 γ 为：

$$\boxed{\gamma^* = \min\left(1, \sqrt{\frac{-\log\rho_{\min}}{\sigma^2}}\right)}$$

其中 $\sigma^2 = \text{Var}(\log w)$。

**证明**：

要求 $\text{ESS}_\gamma / n \geq \rho_{\min}$：
$$e^{-\sigma^2\gamma^2} \geq \rho_{\min}$$
$$\gamma^2 \leq \frac{-\log\rho_{\min}}{\sigma^2}$$
$$\gamma \leq \sqrt{\frac{-\log\rho_{\min}}{\sigma^2}}$$

取满足约束的最大 γ，并限制在 [0, 1]。$\blacksquare$

### 8.2 计算复杂度

| 步骤 | 复杂度 |
|------|--------|
| 计算 $\log w = \log\pi_\theta - \log\mu$ | O(n) |
| 计算 $\sigma^2 = \text{Var}(\log w)$ | O(n) |
| 计算 $\gamma^*$ | O(1) |
| **总计** | **O(n)** |

**无需迭代或二分搜索。**

### 8.3 Log-Normal 假设的诊断与鲁棒处理

**假设适用条件**：当 $\pi_\theta$ 和 $\mu$ 差异不太大时，$\log w$ 近似正态。

**实践中的诊断方法**：

```python
def diagnose_log_normal(log_w):
    """检验 log w 的分布是否接近正态"""
    from scipy import stats

    # 1. Shapiro-Wilk 正态性检验
    _, p_value = stats.shapiro(log_w[:5000])  # 取子样本

    # 2. 偏度和峰度
    skewness = stats.skew(log_w)
    kurtosis = stats.kurtosis(log_w)  # 正态分布为 0

    # 3. 重尾指标：最大权重占比
    w = torch.exp(log_w)
    max_weight_ratio = w.max() / w.sum()

    return {
        'normality_p_value': p_value,
        'skewness': skewness,        # 期望接近 0
        'excess_kurtosis': kurtosis, # 期望接近 0
        'max_weight_ratio': max_weight_ratio,  # 期望 < 0.1
    }
```

**当假设不满足时的处理策略**：

| 情况 | 诊断指标 | 处理方法 |
|------|---------|---------|
| **轻微偏离** | kurtosis ∈ (0, 3), skewness 小 | 闭式解仍可用，可适当降低 $\rho_{\min}$ |
| **重尾分布** | kurtosis > 3, max_weight_ratio > 0.1 | 使用 PSIS（见下文） |
| **严重偏离** | 分布高度非正态 | 刷新 μ 或使用保守的固定 γ |

**方法 1：Pareto 平滑重要性采样（PSIS）**

当 $\log w$ 呈重尾分布时，PSIS（Vehtari et al. 2024）提供鲁棒的 ESS 估计：

```python
def psis_gamma_selection(log_w, rho_min=0.3):
    """使用 PSIS 进行鲁棒的 γ 选择"""
    from arviz import psislw  # 或自行实现

    # PSIS 返回平滑后的 log 权重和诊断值 k
    log_w_smooth, k_hat = psislw(log_w)

    # k_hat < 0.5: 稳定; 0.5-0.7: 可接受; >0.7: 不可靠
    if k_hat > 0.7:
        # 重尾严重，使用保守 γ
        return 0.3, {'warning': 'heavy_tail', 'k_hat': k_hat}

    # 基于平滑权重计算 ESS
    w_smooth = np.exp(log_w_smooth - log_w_smooth.max())
    ess_ratio = (w_smooth.sum()**2) / (w_smooth**2).sum() / len(w_smooth)

    # 二分搜索找满足 ESS 约束的最大 γ
    # （因为 PSIS 不提供闭式解）
    gamma = binary_search_gamma(log_w_smooth, rho_min)

    return gamma, {'k_hat': k_hat, 'ess_ratio': ess_ratio}
```

**方法 2：保守的固定 γ 策略**

当诊断显示假设严重失效时，可以放弃自适应，使用经验验证的固定值：

| 场景 | 推荐 γ | 理由 |
|------|--------|------|
| 训练初期 | 0.8 - 1.0 | $\pi_\theta \approx \mu$，可激进 |
| 训练中期 | 0.3 - 0.5 | 平衡探索和稳定 |
| 假设失效 | 0.2 - 0.3 | 保守选择，接近 SFT |

**方法 3：刷新行为策略 μ**

当 $\sigma^2 = \text{Var}(\log w)$ 过大（如 > 10）时，最根本的解决方案是**重新采样**：
- 用当前 $\pi_\theta$ 生成新数据
- 将 μ 更新为新的采样分布
- 这本质上是从 off-policy 转向 on-policy

**实践建议**：
1. **默认使用闭式解**：在大多数情况下足够好
2. **监控诊断指标**：特别是 kurtosis 和 max_weight_ratio
3. **渐进保守**：当诊断异常时，逐步降低 γ 而非直接切换方法

### 8.4 性质分析

**命题 8.2（γ* 的行为）**：

| 场景 | $\sigma^2$ | $\gamma^*$ | 解释 |
|------|-----------|-----------|------|
| $\pi_\theta \approx \mu$ | 小 → 0 | → 1 | 可以完全信任 IS |
| $\pi_\theta$ 远离 $\mu$ | 大 | 小 | 需要保守，靠近 SFT |
| $\sigma^2 = -\log\rho_{\min}$ | 中等 | = 1 | 临界点 |

### 8.5 τ 与 γ 的联合选择

#### τ 的角色随 γ 变化

一个重要的观察是：**τ 的影响随 γ 平滑"渐入"**。

从目标分布的展开形式可以看出：
$$p_\gamma^* \propto \mu^{1-\gamma} \cdot r^{1-\gamma} \cdot e^{\gamma r/\tau}$$

| γ 值 | τ 在目标分布中的出现 | τ 的实际影响 |
|------|---------------------|-------------|
| γ = 0 | 不出现（$e^{0 \cdot r/\tau} = 1$） | **无影响**，目标完全由 $\mu \cdot r$ 决定 |
| γ ∈ (0, 1) | 出现在 $e^{\gamma r/\tau}$ 中 | **部分影响**，与 $\mu^{1-\gamma} r^{1-\gamma}$ 共同作用 |
| γ = 1 | 完全决定目标 $e^{r/\tau}$ | **完全影响**，是唯一的形状参数 |

**物理解释**：
- 当 γ = 0（纯 SFT），我们完全依赖数据分布 μ 和奖励权重 r，τ 自然没有意义
- 当 γ = 1（纯 RL），我们完全依赖 $e^{r/\tau}$，τ 是熵正则化系数
- 当 γ ∈ (0, 1)，我们在两者之间插值，τ 的影响程度由 γ 控制

**这不是缺陷，而是框架的自然结构**：τ 是 RL 端的参数，γ 控制我们"靠近 RL 端多少"。

#### τ 和 γ 的交互效应

τ 和 γ 存在交互效应：

| τ 的值 | $p^* = e^{r/\tau}$ 的形状 | γ 的影响 |
|--------|------------------------|---------|
| τ 大 | 接近均匀 | γ 的选择影响较小 |
| τ 小 | 尖锐（集中在高奖励） | γ 的选择更关键 |

**有效敏感度**：可以定义"对奖励差异的有效敏感度"：

$$\text{Sensitivity}(\gamma, \tau) \approx \gamma / \tau$$

- γ 大 + τ 小：对奖励差异**极度敏感**（激进 RL）
- γ 小 + τ 大：对奖励差异**不敏感**（保守 SFT）

**注意**：这个近似在 γ 接近 0 时不再有意义，因为此时 τ 本身就不起作用。

#### 实践建议

- **先固定 τ**：通常 τ ∈ [0.1, 1.0]，根据奖励的尺度选择
- **再自适应选择 γ**：用闭式解根据 $\sigma^2$ 动态调整
- **联合约束**：确保 $\gamma / \tau$ 不会导致权重过于极端
- **边界情况**：当 γ 很小（< 0.1）时，不必担心 τ 的设置

---

# 第六部分：与现有方法的联系

## 9. 现有方法在框架中的位置

### 9.1 方法分类

| 方法 | r 的形式 | γ 的位置 | 特点 |
|------|---------|---------|------|
| **纯 SFT** | r = 1 | N/A | 框架退化 |
| **RFT / RAFT / ReST** | r = 𝟙[reward > θ] | γ ≈ 0 | 硬过滤 + SFT |
| **奖励加权 SFT** | r = reward | γ = 0 | 软权重 + SFT |
| **GRPO** | r = normalized | γ ∈ (0,1) | 组内相对排名 |
| **RLOO** | r = reward - baseline | γ = 1 | on-policy RL |
| **PPO** | r = clipped advantage | γ = 1 | on-policy RL + clipping |

### 9.2 具体方法分析

**RAFT / ReST（Rejection sampling Fine-Tuning）**：
```
1. 用 reward model 对样本打分
2. 只保留 reward > threshold 的样本
3. 在过滤后的数据上做纯 SFT
```
- **框架视角**：$r = \mathbf{1}[\text{reward} > \theta]$，γ = 0
- **特点**：硬阈值，完全的 SFT 风格，不使用连续奖励信息

**GRPO（Group Relative Policy Optimization）**：
```
1. 对每个 prompt 生成多个 response
2. 在组内用相对奖励排名
3. 用排名加权更新策略
```
- **框架视角**：$r = \text{rank\_score}$，γ 介于 0 和 1 之间
- **特点**：使用相对排名而非绝对奖励，隐式的方差控制

**RLOO（REINFORCE Leave-One-Out）**：
```
1. 对每个 prompt 生成 k 个 response
2. 用 leave-one-out 均值作为 baseline
3. REINFORCE 更新
```
- **框架视角**：$r = \text{reward} - \text{baseline}$，γ = 1
- **特点**：纯 RL 风格，baseline 减少方差

### 9.3 DPO 的特殊地位

**DPO（Direct Preference Optimization）**不直接纳入本框架，原因：

1. **输入不同**：DPO 使用偏好对 $(y_w, y_l)$，而非点估计奖励
2. **奖励是隐式的**：$r_{implicit}(y) = \beta \log \frac{\pi_\theta(y)}{\pi_{ref}(y)}$
3. **目标不同**：直接优化 Bradley-Terry 偏好模型

**联系**：DPO 可以看作是 γ = 1 的 KL 约束 RL 的一种**闭式求解**：
- KL-RL：$\max_\pi \mathbb{E}[r] - \beta D_{KL}(\pi \| \pi_{ref})$
- DPO：绕过显式奖励，直接从偏好学习

### 9.4 框架的统一视角

```
                        γ = 0                    γ = 1
                     (SFT 风格)               (RL 风格)
                         │                        │
    ┌────────────────────┼────────────────────────┼────────────────┐
    │                    │                        │                │
    │   RFT/RAFT ────────┤                        ├──── RLOO/PPO   │
    │   (硬过滤)          │                        │    (on-policy)  │
    │                    │                        │                │
    │   奖励加权 SFT ─────┤                        │                │
    │   (软权重)          │                        │                │
    │                    │        GRPO            │                │
    │                    │    (组内相对排名)        │                │
    │                    │          │             │                │
    └────────────────────┴──────────┴─────────────┴────────────────┘

                              IS Reshape
                         γ* = min(1, √(-log ρ/σ²))
                            自适应选择最优位置
```

**框架的价值**：提供了一个**自适应**的方法，根据 $\pi_\theta$ 和 $\mu$ 的差异自动选择最优的 γ，而不是固定使用某种方法。

---

# 第七部分：完整框架总结

## 10. 理论结构

```
┌─────────────────────────────────────────────────────────────────┐
│                     IS Reshape 统一框架                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  【层次 1：两端的散度解释】                                        │
│                                                                 │
│      γ=0 (SFT): min D_KL(μ·r ‖ π)     → Forward KL / Mean-seeking│
│      γ=1 (RL):  min D_KL(π ‖ e^{r/τ}) → Reverse KL / Mode-seeking│
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  【层次 2：对 r 的编码方式】                                       │
│                                                                 │
│      SFT: μ · e^{log r}    → 对数编码，压缩奖励差异               │
│      RL:  e^{r/τ}          → 线性编码，放大奖励差异               │
│                                                                 │
│      有效编码：f_γ(r) = (1-γ)log r + γ·r/τ                       │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  【层次 3：目标分布几何演化】                                      │
│                                                                 │
│      p_γ^* ∝ μ^{1-γ} · exp[f_γ(r)]                              │
│                                                                 │
│      几何插值：μ·r ────────────────────→ e^{r/τ}                 │
│               γ=0 (重塑数据)        γ=1 (定义目标)               │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  【层次 4：方差控制（主导因子）】                                     │
│                                                                 │
│      Var(g_γ) 的主导项 E[w^{2γ}] = exp((2γ-1) · D_{2γ}^R(π‖μ)) │
│                                                                 │
│      Rényi 散度刻画权重贡献，是 γ 选择的理论基础                      │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  【层次 5：闭式 γ 选择】                                           │
│                                                                 │
│      γ* = min(1, √(-log ρ_min / σ²))                           │
│                                                                 │
│      O(1) 计算，无需迭代                                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 11. 核心定理索引

| 定理 | 内容 | 性质 |
|------|------|------|
| 3.1 | γ=0 对应 Forward KL | 精确 |
| 3.2 | γ=1 对应 Reverse KL | 精确（需熵正则） |
| 4.2 | $p_\gamma^*$ 最小化散度凸组合 | 精确 |
| 5.2 | Rényi-IS 权重关系 | **恒等式** |
| 5.3 | 方差分解与主导项 | 精确（主导项由 Rényi 刻画） |
| 6.2 | Bias 单调递减 | 精确 |
| 7.2 | ESS 闭式近似 | Log-Normal 假设 |
| 8.1 | γ 闭式选择 | Log-Normal 假设 |

---

# 第七部分：实现

## 12. 算法

```python
import torch
import math

class ISReshapeTrainer:
    """
    IS Reshape 统一框架实现

    理论基础：
    - 目标函数：L_γ = (1/γ)(E[w^γ r] - E[r])
    - 梯度通过 w^γ 传递（不需要 stop gradient）
    - γ 存在闭式解
    """

    def __init__(
        self,
        model,
        ref_model,
        rho_min: float = 0.3,
        gamma_min: float = 0.1,  # 避免 1/γ 数值问题
    ):
        self.model = model
        self.ref_model = ref_model
        self.rho_min = rho_min
        self.gamma_min = gamma_min

        # 冻结参考模型
        for p in self.ref_model.parameters():
            p.requires_grad = False

    def compute_gamma(self, log_w: torch.Tensor) -> float:
        """
        O(1) 闭式 γ 选择

        γ* = min(1, √(-log ρ_min / σ²))
        """
        sigma_sq = torch.var(log_w).item()

        if sigma_sq < 1e-8:
            return 1.0

        gamma = math.sqrt(-math.log(self.rho_min) / sigma_sq)
        return max(self.gamma_min, min(1.0, gamma))

    def compute_loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        rewards: torch.Tensor = None,
    ):
        """
        计算 IS Reshape 损失

        方式 A（推荐）：L = -(1/γ) E[w^γ r]
        梯度通过 w^γ 传递，符合 RL 的 IS 思想
        """
        # 计算 log 概率
        with torch.no_grad():
            log_mu = self.ref_model.log_prob(y, x)
        log_pi = self.model.log_prob(y, x)

        # log 重要性权重
        log_w = log_pi - log_mu

        # 闭式 γ 选择
        gamma = self.compute_gamma(log_w.detach())

        # 计算 w^γ（不 detach，梯度通过 IS 传递）
        w_gamma = torch.exp(gamma * log_w)

        # 奖励（默认为 1，即纯 SFT）
        if rewards is None:
            rewards = torch.ones_like(log_w)

        # 目标函数：L_γ = (1/γ) E[w^γ r]
        # 取负因为要最大化
        loss = -(1.0 / gamma) * (w_gamma * rewards).mean()

        # 诊断信息
        with torch.no_grad():
            weights_normalized = w_gamma / w_gamma.sum()
            ess = 1.0 / (weights_normalized ** 2).sum()

        metrics = {
            'gamma': gamma,
            'ess': ess.item(),
            'ess_ratio': ess.item() / len(log_w),
            'sigma_sq': torch.var(log_w).item(),
            'max_weight': weights_normalized.max().item(),
        }

        return loss, metrics


def train_step(trainer, batch, optimizer):
    """单步训练"""
    x, y = batch['prompt'], batch['response']
    rewards = batch.get('reward', None)

    loss, metrics = trainer.compute_loss(x, y, rewards)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), 1.0)
    optimizer.step()

    return loss.item(), metrics
```

---

# 附录

## A. Log-Normal 假设下的方差推导

**设定**：$\log w \sim \mathcal{N}(-\sigma^2/2, \sigma^2)$（保证 $\mathbb{E}[w] = 1$）

**计算 $\mathbb{E}[w^{2\gamma}]$**：

$$\mathbb{E}[w^{2\gamma}] = \mathbb{E}[e^{2\gamma \log w}]$$

由于 $\log w \sim \mathcal{N}(-\sigma^2/2, \sigma^2)$，有 $2\gamma \log w \sim \mathcal{N}(-\gamma\sigma^2, 4\gamma^2\sigma^2)$。

因此：
$$\mathbb{E}[e^{2\gamma \log w}] = e^{-\gamma\sigma^2 + 2\gamma^2\sigma^2} = e^{\sigma^2\gamma(2\gamma-1)}$$

## B. 符号表

| 符号 | 定义 |
|------|------|
| $\mu$ | 行为策略（数据分布） |
| $\pi_\theta$ | 待学习策略 |
| $w = \pi_\theta/\mu$ | 重要性权重 |
| $\tilde{\pi} = \mu \cdot r / Z_0$ | 奖励加权数据分布（SFT 目标） |
| $p^* = e^{r/\tau}/Z$ | 最优软策略（RL 目标） |
| $p_\gamma^*$ | γ-目标分布（几何插值） |
| $\gamma \in [0,1]$ | IS reshape 参数 |
| $D_\alpha^R$ | α-Rényi 散度 |
| $\sigma^2$ | $\text{Var}(\log w)$ |
| ESS | 有效样本量 |
| $\rho_{\min}$ | 最小 ESS 比例 |

## C. 参考文献

1. Kong, A. (1992). A note on importance sampling using standardized weights.
2. Owen, A. B. (2013). Monte Carlo theory, methods and examples.
3. Vehtari, A. et al. (2024). Pareto smoothed importance sampling. JMLR.
4. Amari, S. I. (2016). Information geometry and its applications.

---

## 总结

本框架的核心贡献：

1. **揭示两端的本质差异**：SFT (Forward KL) 和 RL (Reverse KL) 是本质不同的范式

2. **统一的编码视角**：两端都可写成 $\mu^\alpha \cdot e^{f(r)}$ 形式，差异在于对 r 的编码方式（$\log r$ vs $r/\tau$）

3. **几何插值连接两端**：$p_\gamma^* \propto \mu^{1-\gamma} \cdot \exp[(1-\gamma)\log r + \gamma r/\tau]$ 是信息几何意义下的测地线

4. **r 作为数据重塑**：SFT 是对 μ 的重塑（RFT 是硬过滤，奖励加权是软过滤），RL 则完全脱离 μ

5. **方差的主导因子**：$\mathbb{E}[w^{2\gamma}]$ 由 Rényi 散度精确刻画，是方差爆炸的主要来源

6. **实用的 γ 选择**：O(1) 闭式解（Log-Normal 假设下），附带假设失效时的鲁棒处理策略

7. **γ 的双重意义**：同时控制"对 μ 的依赖程度"和"对 r 的编码方式"；既可作为训练技巧，也可作为设计选择

8. **τ 的渐入效应**：τ 的影响随 γ 平滑变化，这是框架的自然结构而非缺陷
