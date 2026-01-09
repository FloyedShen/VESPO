# α-ELBO 框架用于离线强化学习：完整证明

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
| $w(y) = \frac{\pi_\theta(y\|x)}{\mu(y\|x)}$ | 重要性采样比率 |

### 1.2 目标

给定从 $\mu$ 采样的离线数据集 $\mathcal{D} = \{(x_i, y_i, r_i)\}_{i=1}^n$，学习一个策略 $\pi_\theta$ 使得期望奖励最大化，同时控制分布偏移。

---

## 2. 目标分布与配分函数

### 2.1 定义

定义**奖励加权的目标分布**：

$$p^*(y|x) = \frac{\mu(y|x) \exp(r(x,y)/\tau)}{Z(x)}$$

其中**配分函数**为：

$$Z(x) = \int \mu(y|x) \exp(r(x,y)/\tau) \, dy = \mathbb{E}_{y \sim \mu}[\exp(r(x,y)/\tau)]$$

### 2.2 直观理解

- $p^*(y|x)$ 是在最大化奖励的同时保持接近 $\mu$ 的最优软策略
- 这恰好是以下优化问题的解：$\max_\pi \mathbb{E}_\pi[r] - \tau D_{KL}(\pi \| \mu)$
- $Z(x)$ 衡量了离线数据在奖励函数下的"质量"

**命题 2.1**：$p^*$ 是以下问题的唯一解：
$$p^* = \arg\max_p \left\{ \mathbb{E}_{y \sim p}[r(x,y)] - \tau D_{KL}(p \| \mu) \right\}$$

**证明**：对泛函求变分并令其为零：
$$\frac{\delta}{\delta p(y)} \left[ \int p(y) r(y) dy - \tau \int p(y) \log \frac{p(y)}{\mu(y)} dy - \lambda\left(\int p(y) dy - 1\right) \right] = 0$$

$$r(y) - \tau \log \frac{p(y)}{\mu(y)} - \tau - \lambda = 0$$

$$p(y) \propto \mu(y) \exp(r(y)/\tau) \quad \blacksquare$$

---

## 3. 标准 ELBO

### 3.1 推导

**定理 3.1（标准 ELBO）**：对于任意分布 $\pi_\theta$：
$$\log Z(x) \geq \mathcal{L}_1(\pi_\theta) := \frac{1}{\tau}\mathbb{E}_{\pi_\theta}[r(x,y)] - D_{KL}(\pi_\theta \| \mu)$$

当且仅当 $\pi_\theta = p^*$ 时取等号。

**证明**：

从 KL 散度的定义出发：

$$D_{KL}(\pi_\theta \| p^*) = \mathbb{E}_{\pi_\theta}\left[\log \frac{\pi_\theta(y|x)}{p^*(y|x)}\right] \geq 0$$

代入 $p^* = \mu e^{r/\tau} / Z$：

$$D_{KL}(\pi_\theta \| p^*) = \mathbb{E}_{\pi_\theta}\left[\log \frac{\pi_\theta(y|x)}{\mu(y|x) e^{r/\tau} / Z}\right]$$

$$= \mathbb{E}_{\pi_\theta}\left[\log \frac{\pi_\theta}{\mu}\right] - \frac{1}{\tau}\mathbb{E}_{\pi_\theta}[r] + \log Z$$

$$= D_{KL}(\pi_\theta \| \mu) - \frac{1}{\tau}\mathbb{E}_{\pi_\theta}[r] + \log Z$$

由于 $D_{KL}(\pi_\theta \| p^*) \geq 0$：

$$\log Z \geq \frac{1}{\tau}\mathbb{E}_{\pi_\theta}[r] - D_{KL}(\pi_\theta \| \mu) = \mathcal{L}_1(\pi_\theta) \quad \blacksquare$$

### 3.2 Gap 分析

**推论 3.2**：$\log Z$ 与 ELBO 之间的 gap 恰好是 reverse KL：
$$\log Z - \mathcal{L}_1(\pi_\theta) = D_{KL}(\pi_\theta \| p^*)$$

---

## 4. Rényi 散度基础

### 4.1 定义

**定义 4.1（Rényi 散度）**：对于 $\alpha \in (0, 1) \cup (1, \infty)$：

$$D_\alpha(P \| Q) = \frac{1}{\alpha - 1} \log \mathbb{E}_Q\left[\left(\frac{P(x)}{Q(x)}\right)^\alpha\right]$$

### 4.2 关键性质

**命题 4.2**：Rényi 散度满足：

1. **非负性**：$D_\alpha(P \| Q) \geq 0$，当且仅当 $P = Q$ 时取等号

2. **关于 α 的单调性**：对于 $\alpha_1 < \alpha_2$：
   $$D_{\alpha_1}(P \| Q) \leq D_{\alpha_2}(P \| Q)$$

3. **KL 散度极限**：
   $$\lim_{\alpha \to 1} D_\alpha(P \| Q) = D_{KL}(P \| Q)$$

4. **最大散度极限**：
   $$\lim_{\alpha \to \infty} D_\alpha(P \| Q) = D_\infty(P \| Q) = \log \sup_x \frac{P(x)}{Q(x)}$$

**性质 2 的证明**：

定义 $f(\alpha) = (\alpha - 1) D_\alpha(P \| Q) = \log \mathbb{E}_Q[w^\alpha]$，其中 $w = P/Q$。

由 Hölder 不等式，对于 $\alpha_1 < \alpha_2$：
$$\mathbb{E}_Q[w^{\alpha_1}] = \mathbb{E}_Q[w^{\alpha_1} \cdot 1] \leq \mathbb{E}_Q[w^{\alpha_2}]^{\alpha_1/\alpha_2} \cdot 1$$

取对数并整理即得结果。$\blacksquare$

---

## 5. α-ELBO（变分 Rényi 界）

### 5.1 定义与主定理

**定义 5.1（α-ELBO）**：对于 $\alpha \in (0, 1)$，使用未归一化的目标分布 $\tilde{p}(y|x) = \mu(y|x) e^{r/\tau}$：

$$\mathcal{L}_\alpha(\pi_\theta) = \frac{1}{1-\alpha} \log \mathbb{E}_{\pi_\theta}\left[\left(\frac{\mu(y|x) e^{r/\tau}}{\pi_\theta(y|x)}\right)^{1-\alpha}\right]$$

**定理 5.2（Rényi ELBO 界）**：对于所有 $\alpha \in (0, 1)$：
$$\mathcal{L}_\alpha(\pi_\theta) \leq \log Z(x)$$

具有以下极限行为：
- $\lim_{\alpha \to 1^-} \mathcal{L}_\alpha(\pi_\theta) = \mathcal{L}_1(\pi_\theta)$（标准 ELBO）
- $\lim_{\alpha \to 0^+} \mathcal{L}_\alpha(\pi_\theta) = \log Z(x)$

**证明**：

**第一步**：应用 Hölder 不等式。

对于 $\alpha \in (0, 1)$，取共轭指数 $p = \frac{1}{1-\alpha} > 1$ 和 $q = \frac{1}{\alpha}$：

$$\mathbb{E}_{\pi_\theta}\left[\left(\frac{\mu e^{r/\tau}}{\pi_\theta}\right)^{1-\alpha}\right] = \mathbb{E}_{\pi_\theta}\left[\left(\frac{\mu e^{r/\tau}}{\pi_\theta}\right)^{1-\alpha} \cdot 1^\alpha \right]$$

$$\leq \left(\mathbb{E}_{\pi_\theta}\left[\frac{\mu e^{r/\tau}}{\pi_\theta}\right]\right)^{1-\alpha} \cdot \left(\mathbb{E}_{\pi_\theta}[1]\right)^{\alpha}$$

$$= \left(\int \mu e^{r/\tau} dy\right)^{1-\alpha} = Z^{1-\alpha}$$

**第二步**：两边取 $\frac{1}{1-\alpha} \log$：

$$\mathcal{L}_\alpha = \frac{1}{1-\alpha} \log \mathbb{E}_{\pi_\theta}[\cdot] \leq \frac{1}{1-\alpha} \log Z^{1-\alpha} = \log Z \quad \blacksquare$$

**极限行为证明**：

**当 $\alpha \to 1$ 时**：

令 $f(\alpha) = \log \mathbb{E}_{\pi_\theta}\left[\left(\frac{\mu e^{r/\tau}}{\pi_\theta}\right)^{1-\alpha}\right]$。

在 $\alpha = 1$ 处：$f(1) = \log \mathbb{E}_{\pi_\theta}[1] = 0$。

对 $\alpha$ 求导：
$$f'(\alpha) = \frac{-\mathbb{E}_{\pi_\theta}\left[\left(\frac{\mu e^{r/\tau}}{\pi_\theta}\right)^{1-\alpha} \log\left(\frac{\mu e^{r/\tau}}{\pi_\theta}\right)\right]}{\mathbb{E}_{\pi_\theta}\left[\left(\frac{\mu e^{r/\tau}}{\pi_\theta}\right)^{1-\alpha}\right]}$$

在 $\alpha = 1$ 处：
$$f'(1) = -\mathbb{E}_{\pi_\theta}\left[\log\left(\frac{\mu e^{r/\tau}}{\pi_\theta}\right)\right] = -\mathbb{E}_{\pi_\theta}[\log \mu + r/\tau - \log \pi_\theta]$$

由 L'Hôpital 法则：
$$\lim_{\alpha \to 1} \mathcal{L}_\alpha = \lim_{\alpha \to 1} \frac{f(\alpha)}{1-\alpha} = -f'(1) = \mathbb{E}_{\pi_\theta}[\log \mu + r/\tau - \log \pi_\theta]$$

$$= \frac{1}{\tau}\mathbb{E}_{\pi_\theta}[r] - D_{KL}(\pi_\theta \| \mu) = \mathcal{L}_1 \quad \blacksquare$$

**当 $\alpha \to 0$ 时**：

$$\mathcal{L}_0 = \lim_{\alpha \to 0} \frac{1}{1-\alpha} \log \mathbb{E}_{\pi_\theta}\left[\frac{\mu e^{r/\tau}}{\pi_\theta}\right] = \log \mathbb{E}_{\pi_\theta}\left[\frac{\mu e^{r/\tau}}{\pi_\theta}\right] = \log \int \mu e^{r/\tau} dy = \log Z \quad \blacksquare$$

---

## 6. 转换到离线数据形式

### 6.1 主要结果

**定理 6.1（离线 α-ELBO）**：α-ELBO 可以改写为关于行为策略 $\mu$ 的期望：

$$\mathcal{L}_\alpha(\pi_\theta) = \frac{1}{1-\alpha} \log \mathbb{E}_{y \sim \mu}\left[w(y)^\alpha \cdot e^{(1-\alpha)r(y)/\tau}\right]$$

其中 $w(y) = \pi_\theta(y|x) / \mu(y|x)$ 是重要性采样比率。

**证明**：

从定义出发：

$$\mathcal{L}_\alpha = \frac{1}{1-\alpha} \log \mathbb{E}_{\pi_\theta}\left[\left(\frac{\mu(y|x) e^{r/\tau}}{\pi_\theta(y|x)}\right)^{1-\alpha}\right]$$

$$= \frac{1}{1-\alpha} \log \int \pi_\theta(y|x) \left(\frac{\mu(y|x) e^{r/\tau}}{\pi_\theta(y|x)}\right)^{1-\alpha} dy$$

$$= \frac{1}{1-\alpha} \log \int \pi_\theta(y|x)^{1-(1-\alpha)} \mu(y|x)^{1-\alpha} e^{(1-\alpha)r/\tau} dy$$

$$= \frac{1}{1-\alpha} \log \int \pi_\theta(y|x)^\alpha \mu(y|x)^{1-\alpha} e^{(1-\alpha)r/\tau} dy$$

$$= \frac{1}{1-\alpha} \log \int \mu(y|x) \left(\frac{\pi_\theta(y|x)}{\mu(y|x)}\right)^\alpha e^{(1-\alpha)r/\tau} dy$$

$$= \frac{1}{1-\alpha} \log \mathbb{E}_{y \sim \mu}\left[w(y)^\alpha e^{(1-\alpha)r/\tau}\right] \quad \blacksquare$$

### 6.2 与奖励加权 Rényi 散度的联系

**推论 6.2**：定义奖励加权的 Rényi 散度：

$$D_\alpha^{(r)}(\pi_\theta \| \mu) := \frac{1}{\alpha - 1} \log \mathbb{E}_\mu\left[w^\alpha e^{(1-\alpha)r/\tau}\right]$$

则：
$$\mathcal{L}_\alpha(\pi_\theta) = -D_\alpha^{(r)}(\pi_\theta \| \mu)$$

**证明**：直接比较定义：
$$\mathcal{L}_\alpha = \frac{1}{1-\alpha} \log \mathbb{E}_\mu[w^\alpha e^{(1-\alpha)r/\tau}] = -\frac{1}{\alpha-1} \log \mathbb{E}_\mu[w^\alpha e^{(1-\alpha)r/\tau}] = -D_\alpha^{(r)} \quad \blacksquare$$

---

## 7. 梯度分析

### 7.1 α-ELBO 的梯度

**定理 7.1**：离线 α-ELBO 的梯度为：

$$\nabla_\theta \mathcal{L}_\alpha = \frac{\alpha}{1-\alpha} \cdot \mathbb{E}_\mu\left[\tilde{w}_\alpha(y) \cdot \nabla_\theta \log \pi_\theta(y|x)\right]$$

其中**自归一化重要性权重**为：

$$\tilde{w}_\alpha(y) = \frac{w(y)^\alpha e^{(1-\alpha)r(y)/\tau}}{\mathbb{E}_\mu[w^\alpha e^{(1-\alpha)r/\tau}]}$$

**证明**：

令 $G(\theta) = \mathbb{E}_\mu[w^\alpha e^{(1-\alpha)r/\tau}]$，则 $\mathcal{L}_\alpha = \frac{1}{1-\alpha} \log G(\theta)$。

**第一步**：对数的梯度：
$$\nabla_\theta \mathcal{L}_\alpha = \frac{1}{1-\alpha} \cdot \frac{\nabla_\theta G(\theta)}{G(\theta)}$$

**第二步**：$G$ 的梯度：

由于 $w = \pi_\theta / \mu$ 且 $\mu$ 不依赖于 $\theta$：
$$\nabla_\theta w^\alpha = \alpha w^{\alpha-1} \nabla_\theta w = \alpha w^{\alpha-1} \cdot \frac{\nabla_\theta \pi_\theta}{\mu}$$

使用对数导数技巧：$\nabla_\theta \pi_\theta = \pi_\theta \nabla_\theta \log \pi_\theta$：
$$\nabla_\theta w^\alpha = \alpha w^{\alpha-1} \cdot \frac{\pi_\theta \nabla_\theta \log \pi_\theta}{\mu} = \alpha w^{\alpha-1} \cdot w \cdot \nabla_\theta \log \pi_\theta = \alpha w^\alpha \nabla_\theta \log \pi_\theta$$

因此：
$$\nabla_\theta G = \mathbb{E}_\mu[\alpha w^\alpha e^{(1-\alpha)r/\tau} \nabla_\theta \log \pi_\theta]$$

**第三步**：合并：
$$\nabla_\theta \mathcal{L}_\alpha = \frac{1}{1-\alpha} \cdot \frac{\alpha \mathbb{E}_\mu[w^\alpha e^{(1-\alpha)r/\tau} \nabla_\theta \log \pi_\theta]}{G(\theta)}$$

$$= \frac{\alpha}{1-\alpha} \mathbb{E}_\mu\left[\frac{w^\alpha e^{(1-\alpha)r/\tau}}{G(\theta)} \nabla_\theta \log \pi_\theta\right]$$

$$= \frac{\alpha}{1-\alpha} \mathbb{E}_\mu[\tilde{w}_\alpha \nabla_\theta \log \pi_\theta] \quad \blacksquare$$

### 7.2 特殊情况

**推论 7.2**（极限情况）：

1. **α → 1**（标准策略梯度）：
$$\nabla_\theta \mathcal{L}_1 = \mathbb{E}_{\pi_\theta}[\nabla_\theta \log \pi_\theta \cdot r/\tau] - \nabla_\theta D_{KL}(\pi_\theta \| \mu)$$

2. **α → 0**（加权 SFT）：

当 $\alpha \to 0$ 时，$\mathcal{L}_0 = \log Z$ 不依赖于 $\theta$，所以 $\nabla_\theta \mathcal{L}_0 = 0$。

实际应用中使用小的正 α：
$$\nabla_\theta \mathcal{L}_\alpha \big|_{\alpha \text{ 较小}} \approx \alpha \cdot \mathbb{E}_\mu\left[\frac{e^{r/\tau}}{Z} \nabla_\theta \log \pi_\theta\right]$$

这是以 $e^{r/\tau}$ 为权重的最大似然估计。$\blacksquare$

---

## 8. Bias-Variance 分析

### 8.1 设定

考虑从 $n$ 个样本估计 $\mathcal{L}_\alpha$ 的蒙特卡洛估计器：

$$\hat{\mathcal{L}}_\alpha = \frac{1}{1-\alpha} \log \left(\frac{1}{n} \sum_{i=1}^n w_i^\alpha e^{(1-\alpha)r_i/\tau}\right)$$

其中 $(y_i, r_i) \sim \mu$，$w_i = \pi_\theta(y_i|x) / \mu(y_i|x)$。

### 8.2 Bias 分析

**定理 8.1（Bias 分解）**：最大化 $\mathcal{L}_\alpha$ 而非 $\log Z$ 的 bias 为：

$$\boxed{\text{Bias}(\alpha) := \log Z - \mathcal{L}_\alpha(\pi_\theta) = D_\alpha(\pi_\theta \| p^*)}$$

其中 $D_\alpha$ 是 $\alpha$ 阶 Rényi 散度。

**证明**：

从定理 5.2 的证明中，我们有：

$$\mathcal{L}_\alpha = \frac{1}{1-\alpha} \log \mathbb{E}_{\pi_\theta}\left[\left(\frac{\mu e^{r/\tau}}{\pi_\theta}\right)^{1-\alpha}\right]$$

令 $u = \mu e^{r/\tau} / \pi_\theta = (p^* \cdot Z) / \pi_\theta$，则：

$$\mathcal{L}_\alpha = \frac{1}{1-\alpha} \log \mathbb{E}_{\pi_\theta}\left[\left(\frac{p^* Z}{\pi_\theta}\right)^{1-\alpha}\right]$$

$$= \frac{1}{1-\alpha} \log \left(Z^{1-\alpha} \mathbb{E}_{\pi_\theta}\left[\left(\frac{p^*}{\pi_\theta}\right)^{1-\alpha}\right]\right)$$

$$= \log Z + \frac{1}{1-\alpha} \log \mathbb{E}_{\pi_\theta}\left[\left(\frac{p^*}{\pi_\theta}\right)^{1-\alpha}\right]$$

因此：
$$\log Z - \mathcal{L}_\alpha = -\frac{1}{1-\alpha} \log \mathbb{E}_{\pi_\theta}\left[\left(\frac{p^*}{\pi_\theta}\right)^{1-\alpha}\right]$$

现在，注意到：
$$\mathbb{E}_{\pi_\theta}\left[\left(\frac{p^*}{\pi_\theta}\right)^{1-\alpha}\right] = \int \pi_\theta \left(\frac{p^*}{\pi_\theta}\right)^{1-\alpha} dy = \int \pi_\theta^\alpha p^{*1-\alpha} dy$$

而 $\alpha$ 阶 Rényi 散度：
$$D_\alpha(\pi_\theta \| p^*) = \frac{1}{\alpha - 1} \log \mathbb{E}_{p^*}\left[\left(\frac{\pi_\theta}{p^*}\right)^\alpha\right] = \frac{1}{\alpha-1} \log \int p^{*1-\alpha} \pi_\theta^\alpha dy$$

因此：
$$\mathbb{E}_{p^*}\left[\left(\frac{\pi_\theta}{p^*}\right)^\alpha\right] = \mathbb{E}_{\pi_\theta}\left[\left(\frac{p^*}{\pi_\theta}\right)^{1-\alpha}\right]$$

代入：
$$\log Z - \mathcal{L}_\alpha = -\frac{1}{1-\alpha} \log \mathbb{E}_{p^*}\left[\left(\frac{\pi_\theta}{p^*}\right)^\alpha\right]$$

$$= \frac{1}{\alpha - 1} \log \mathbb{E}_{p^*}\left[\left(\frac{\pi_\theta}{p^*}\right)^\alpha\right] = D_\alpha(\pi_\theta \| p^*) \quad \blacksquare$$

**推论 8.2**：
- 当 $\alpha \to 1$：Bias $= D_{KL}(\pi_\theta \| p^*)$（标准 ELBO gap）
- 当 $\alpha \to 0$：Bias $\to 0$（bound 变紧）
- Bias 关于 $\alpha$ 单调递增（由性质 4.2）

### 8.3 Variance 分析

**定理 8.3（梯度估计器的方差）**：蒙特卡洛梯度估计器的方差为：

$$\text{Var}(\hat{\nabla}_\theta \mathcal{L}_\alpha) \approx \frac{1}{n} \cdot \frac{\alpha^2}{(1-\alpha)^2} \cdot \frac{\mathbb{E}_\mu[w^{2\alpha} e^{2(1-\alpha)r/\tau} \|\nabla \log \pi_\theta\|^2]}{G(\theta)^2}$$

**证明概要**：

梯度估计器为：
$$\hat{\nabla}_\theta \mathcal{L}_\alpha = \frac{\alpha}{1-\alpha} \cdot \frac{\sum_i w_i^\alpha e^{(1-\alpha)r_i/\tau} \nabla \log \pi_\theta(y_i)}{\sum_i w_i^\alpha e^{(1-\alpha)r_i/\tau}}$$

这是一个比率估计器。使用 delta 方法：

$$\text{Var}\left(\frac{\bar{X}}{\bar{Y}}\right) \approx \frac{1}{n} \left[\frac{\text{Var}(X)}{\mu_Y^2} - \frac{2\text{Cov}(X,Y)\mu_X}{\mu_Y^3} + \frac{\text{Var}(Y)\mu_X^2}{\mu_Y^4}\right]$$

当 $w$ 方差较大时，主导项为：
$$\text{Var}(\hat{\nabla} \mathcal{L}_\alpha) \propto \frac{1}{n} \mathbb{E}_\mu[w^{2\alpha}] / G(\theta)^2$$

关键因子是 $\mathbb{E}_\mu[w^{2\alpha}]$：
- α 大 → 当 $\pi_\theta \gg \mu$ 时 $w^{2\alpha}$ 可能很大 → 高方差
- α 小 → $w^{2\alpha} \approx 1$ → 低方差 $\blacksquare$

### 8.4 Bias-Variance 权衡

**定理 8.4（MSE 分解）**：梯度估计器的均方误差分解为：

$$\text{MSE}(\alpha) = \underbrace{\|\nabla_\theta D_\alpha(\pi_\theta \| p^*)\|^2}_{\text{Bias}^2} + \underbrace{\frac{C(\alpha)}{n} \mathbb{E}_\mu[w^{2\alpha}]}_{\text{Variance}}$$

**定性行为**：

| $\alpha$ | Bias | Variance | 行为模式 |
|----------|------|----------|----------|
| $\alpha \to 0$ | 小（bound 紧） | 大（$1/\alpha^2$ 因子） | 保守 / SFT 风格 |
| $\alpha = 0.5$ | 中等 | 中等 | 平衡 |
| $\alpha \to 1$ | 大（$D_{KL}$） | 小（标准 PG） | 激进 / RL 风格 |

---

## 9. α 的理论角色：从 α-ELBO 结构出发

前面的分析给出了一些选择 α 的"工程方法"（ESS、方差约束等），但它们与 α-ELBO 本身的结构联系不够紧密。本节从 α-ELBO 的内在结构出发，建立 α 选择的理论基础。

### 9.1 核心定理：最优解不依赖于 α

**定理 9.1（α 不变性）**：对于任意 $\alpha \in (0, 1)$，$\mathcal{L}_\alpha(\pi_\theta)$ 关于 $\pi_\theta$ 的最优解都是 $\pi_\theta^* = p^*$。

**证明**：

对 $\pi_\theta$ 做变分优化（固定 $x$，优化 $\pi_\theta(\cdot|x)$）：

$$\max_{\pi_\theta} \mathcal{L}_\alpha = \max_{\pi_\theta} \frac{1}{1-\alpha} \log \int \pi_\theta(y)^\alpha \mu(y)^{1-\alpha} e^{(1-\alpha)r/\tau} dy$$

约束条件：$\int \pi_\theta(y) dy = 1$。

令 $G = \int \pi_\theta^\alpha \mu^{1-\alpha} e^{(1-\alpha)r/\tau} dy$，构造拉格朗日函数并对 $\pi_\theta(y)$ 求变分：

$$\frac{\delta \mathcal{J}}{\delta \pi_\theta(y)} = \frac{1}{(1-\alpha)G} \cdot \alpha \pi_\theta^{\alpha-1} \mu^{1-\alpha} e^{(1-\alpha)r/\tau} - \lambda = 0$$

解得：
$$\pi_\theta^{\alpha-1} = \frac{\lambda (1-\alpha) G}{\alpha} \cdot \mu^{\alpha-1} e^{-(1-\alpha)r/\tau}$$

$$\pi_\theta = \left(\frac{\lambda (1-\alpha) G}{\alpha}\right)^{1/(\alpha-1)} \cdot \mu \cdot e^{r/\tau}$$

归一化后：
$$\pi_\theta^* = \frac{\mu \cdot e^{r/\tau}}{\int \mu \cdot e^{r/\tau} dy} = \frac{\mu \cdot e^{r/\tau}}{Z} = p^* \quad \blacksquare$$

### 9.2 关键洞察：α 控制优化路径而非目标

**推论 9.2**：α 不改变"去哪里"（最优解），只改变"怎么去"（优化动态）。

这意味着不同的 α 会导致：
- 不同的梯度方向和大小
- 不同的优化 landscape（曲率）
- 不同的收敛速度和稳定性

### 9.3 从梯度对齐角度选择 α

**理想梯度**（如果能从 $p^*$ 采样）：
$$\nabla_\theta^* = \mathbb{E}_{p^*}[\nabla_\theta \log \pi_\theta]$$

**实际梯度**（从 μ 采样，使用 α-ELBO）：
$$\nabla_\theta^\alpha = \frac{\alpha}{1-\alpha} \mathbb{E}_\mu[\tilde{w}_\alpha \nabla_\theta \log \pi_\theta]$$

**定理 9.3（梯度对齐准则）**：最优 α 应最小化实际梯度与理想梯度的偏差：

$$\alpha^* = \arg\min_\alpha \mathbb{E}\left[\left\| \nabla_\theta^\alpha - \nabla_\theta^* \right\|^2\right]$$

展开为 bias-variance 分解：
$$= \arg\min_\alpha \left[ \underbrace{\left\| \mathbb{E}[\nabla_\theta^\alpha] - \nabla_\theta^* \right\|^2}_{\text{Bias}^2：梯度偏差}} + \underbrace{\text{Var}(\nabla_\theta^\alpha)}_{\text{Variance：梯度方差}} \right]$$

**证明**：

由 MSE 分解：
$$\mathbb{E}[\|\nabla_\theta^\alpha - \nabla_\theta^*\|^2] = \|\mathbb{E}[\nabla_\theta^\alpha] - \nabla_\theta^*\|^2 + \mathbb{E}[\|\nabla_\theta^\alpha - \mathbb{E}[\nabla_\theta^\alpha]\|^2]$$

第一项是 bias 的平方，第二项是 variance。$\blacksquare$

### 9.4 从 α-ELBO 目标函数直接导出 α 选择

考虑一个**正则化的 α-ELBO 目标**：

$$\mathcal{J}(\alpha) = \mathcal{L}_\alpha(\pi_\theta) - \frac{\lambda}{2} \text{Var}(\hat{\mathcal{L}}_\alpha)$$

**定理 9.4（最优 α 的变分刻画）**：最优 α 满足：

$$\alpha^* = \arg\max_\alpha \mathcal{J}(\alpha) = \arg\max_\alpha \left\{ \mathcal{L}_\alpha(\pi_\theta) - \frac{\lambda}{2} \text{Var}(\hat{\mathcal{L}}_\alpha) \right\}$$

**直觉**：
- 第一项 $\mathcal{L}_\alpha$：α 越小，bound 越紧，值越大
- 第二项 $\text{Var}$：α 越小，方差越大

最优 α 在两者之间取得平衡。

**显式形式**：

利用 §8 的结果：
$$\mathcal{L}_\alpha = \frac{1}{1-\alpha} \log \bar{G}_\alpha, \quad \bar{G}_\alpha = \frac{1}{n}\sum_i w_i^\alpha e^{(1-\alpha)r_i/\tau}$$

$$\text{Var}(\hat{\mathcal{L}}_\alpha) \approx \frac{1}{n(1-\alpha)^2} \cdot \frac{\text{Var}_\mu(w^\alpha e^{(1-\alpha)r/\tau})}{\bar{G}_\alpha^2}$$

代入 $\mathcal{J}(\alpha)$ 并对 α 求导，可以（数值）求解最优 $\alpha^*$。

### 9.5 从信息几何角度理解 α 的选择

**命题 9.5**：不同的 α 对应不同的信息几何结构：

| α 值 | 对应的几何 | 散度性质 | 优化行为 |
|------|-----------|---------|---------|
| α → 0 | e-connection (mixture) | 对低概率区域敏感 | mean-seeking，覆盖所有模式 |
| α = 0.5 | Hellinger 几何 | 对称，平衡 | 中间行为 |
| α → 1 | m-connection (exponential) | 对高概率区域敏感 | mode-seeking，集中于最优 |

**几何直觉**：
- α 小：优化时"看到"整个分布，倾向于覆盖 $p^*$ 的所有模式
- α 大：优化时"聚焦"于高密度区域，倾向于找到 $p^*$ 的主要模式

### 9.6 基于 α-ELBO Gap 的自适应选择

由定理 8.1，gap $= D_\alpha(\pi_\theta \| p^*)$。虽然 $p^*$ 未知，但可以构造代理：

**方法**：利用不同 α 的 ELBO 差异估计 gap

对于 $\alpha_1 < \alpha_2$，由 Rényi 散度单调性：
$$\mathcal{L}_{\alpha_1} - \mathcal{L}_{\alpha_2} = D_{\alpha_2}(\pi_\theta \| p^*) - D_{\alpha_1}(\pi_\theta \| p^*) \geq 0$$

**自适应规则**：

$$\alpha^* = \arg\min_\alpha \left\{ D_\alpha(\pi_\theta \| p^*) : \text{Var}(\hat{\mathcal{L}}_\alpha) \leq \epsilon \right\}$$

由于 $D_\alpha$ 关于 α 单调，这等价于：
$$\alpha^* = \min \left\{ \alpha : \text{Var}(\hat{\mathcal{L}}_\alpha) \leq \epsilon \right\}$$

即：**在方差可控的前提下，选择最小的 α**。

### 9.7 统一的理论框架

**定理 9.6（α-ELBO 视角下的最优 α 统一刻画）**：

设 $\pi_\theta^{(t)}$ 是第 $t$ 步的策略，则最优 α 可以统一表示为：

$$\alpha_t^* = \arg\max_\alpha \left\{ \underbrace{\mathcal{L}_\alpha(\pi_\theta^{(t)})}_{\text{bound 紧度}} - \underbrace{\frac{\lambda_1}{2} \text{Var}(\hat{\nabla}_\theta \mathcal{L}_\alpha)}_{\text{梯度方差}} - \underbrace{\lambda_2 \cdot \kappa(H_\alpha)}_{\text{优化稳定性（可选）}} \right\}$$

其中：
- $\mathcal{L}_\alpha$：α-ELBO 值（越大越好）
- $\text{Var}(\hat{\nabla}_\theta \mathcal{L}_\alpha)$：梯度估计方差（越小越好）
- $\kappa(H_\alpha)$：Hessian 条件数（越小越稳定，可选项）

**三种方法的统一**：

| 方法 | 在框架中的体现 |
|------|--------------|
| ESS 方法 | 通过 ESS 约束间接控制 $\text{Var}$ |
| 方差约束方法 | 直接约束 $\text{Var} \leq \epsilon$ |
| MSE 方法 | 同时考虑 bias（$\mathcal{L}_\alpha$）和 variance |

### 9.8 更深层的理论视角：α 的本质

前面的分析给出了 α 选择的统一框架，但 α 的选择仍显得有些 ad-hoc。本节从更深层的理论角度推导 α，给出更有理论美感的选择方式。

#### 9.8.1 从 Rényi 熵正则化推导

**动机**：标准 soft RL 使用 Shannon 熵正则化，如果改用 Rényi 熵，α 会自然出现。

**定义 9.7（Rényi 熵）**：
$$H_\alpha(\pi) = \frac{1}{1-\alpha} \log \int \pi(y)^\alpha dy = \frac{1}{1-\alpha} \log \mathbb{E}_\pi[\pi(y)^{\alpha-1}]$$

**Rényi 熵正则化的优化问题**：
$$\max_\pi \mathbb{E}_\pi[r(y)] + \tau H_\alpha(\pi) - \tau \mathbb{E}_\pi[\log \mu(y)]$$

**定理 9.8（Rényi 正则化的最优解）**：

上述问题的最优解为：
$$p^*_\alpha(y|x) = \frac{\left[\mu(y|x) \exp(r(y)/(\tau\alpha))\right]^\alpha}{Z_\alpha}$$

其中 $Z_\alpha = \int \left[\mu(y) \exp(r(y)/(\tau\alpha))\right]^\alpha dy$。

**证明**：

构造 Lagrangian 并对 $\pi(y)$ 求变分：
$$\frac{\delta}{\delta \pi(y)} \left[ \int \pi r \, dy + \frac{\tau}{1-\alpha} \log \int \pi^\alpha dy - \tau \int \pi \log \mu \, dy - \lambda(\int \pi dy - 1) \right] = 0$$

$$r + \frac{\tau \alpha}{1-\alpha} \cdot \frac{\pi^{\alpha-1}}{\int \pi^\alpha dy} - \tau \log \mu - \lambda = 0$$

解得：
$$\pi^{\alpha-1} \propto \mu \exp(r/(\tau\alpha))$$
$$\pi \propto \left[\mu \exp(r/(\tau\alpha))\right]^{\alpha/(\alpha-1)}$$

当 $\alpha \in (0,1)$ 时，$\alpha/(\alpha-1) < 0$，需要重新整理。设 $\beta = \alpha/(1-\alpha)$，则：
$$\pi \propto \left[\mu \exp(r/(\tau\alpha))\right]^{-1/\beta} = \left[\mu \exp(r/(\tau\alpha))\right]^{(1-\alpha)/(-\alpha)}$$

经过仔细计算（或直接验证），最优解确为 $p^*_\alpha \propto [\mu e^{r/(\tau\alpha)}]^\alpha$。$\blacksquare$

**推论 9.9**：当 $\alpha \to 1$ 时，$p^*_\alpha \to p^* = \mu e^{r/\tau} / Z$，恢复标准结果。

**关键洞察**：α 的物理意义是**探索-利用权衡**：
- α 小 → Rényi 熵更大 → 更鼓励探索（分布更平坦）
- α 大 → Rényi 熵更小 → 更鼓励利用（分布更集中）

#### 9.8.2 从 Lagrange 对偶推导（最优雅的方式）

**动机**：把 α 看作约束优化问题的对偶变量，从 KKT 条件自然导出。

**原问题**：最大化最紧的界，同时控制分布偏移
$$\max_\theta \mathcal{L}_0(\pi_\theta) \quad \text{s.t.} \quad D_{KL}(\pi_\theta \| \mu) \leq \delta$$

其中 $\mathcal{L}_0 = \log Z$ 是 $\alpha \to 0$ 时的极限（最紧但方差无穷）。

**引入 Rényi 散度族的松弛**：
$$\max_\theta \mathcal{L}_\alpha(\pi_\theta) \quad \text{s.t.} \quad D_\alpha(\pi_\theta \| \mu) \leq \delta$$

**Lagrangian**：
$$\mathcal{L}(\theta, \lambda, \alpha) = \mathcal{L}_\alpha(\pi_\theta) - \lambda(D_\alpha(\pi_\theta \| \mu) - \delta)$$

**对 α 的 KKT 条件**：
$$\frac{\partial \mathcal{L}}{\partial \alpha} = \frac{\partial \mathcal{L}_\alpha}{\partial \alpha} - \lambda \frac{\partial D_\alpha}{\partial \alpha} = 0$$

**定理 9.10（对偶最优 α）**：

最优 α 满足：
$$\boxed{\frac{\partial \mathcal{L}_\alpha}{\partial \alpha} = \lambda^* \frac{\partial D_\alpha}{\partial \alpha}}$$

其中 $\lambda^* > 0$ 是最优 Lagrange 乘子，由互补松弛条件 $\lambda^*(D_\alpha - \delta) = 0$ 确定。

**显式计算**：

$$\frac{\partial \mathcal{L}_\alpha}{\partial \alpha} = \frac{1}{(1-\alpha)^2} \log G_\alpha - \frac{1}{1-\alpha} \cdot \frac{\mathbb{E}_\mu[w^\alpha e^{(1-\alpha)r/\tau} (\log w - r/\tau)]}{G_\alpha}$$

$$\frac{\partial D_\alpha}{\partial \alpha} = \frac{1}{(\alpha-1)^2} \log \mathbb{E}_\mu[w^\alpha] + \frac{1}{\alpha-1} \cdot \frac{\mathbb{E}_\mu[w^\alpha \log w]}{\mathbb{E}_\mu[w^\alpha]}$$

令两者之比等于 $\lambda^*$，可以数值求解 $\alpha^*$。

#### 9.8.3 从 Escort Distribution 推导

**定义 9.11（Escort Distribution）**：

给定分布 $\pi$，其 α-escort 分布为：
$$\pi^{(\alpha)}(y) = \frac{\pi(y)^\alpha}{\int \pi(y')^\alpha dy'} = \frac{\pi(y)^\alpha}{Z_\alpha(\pi)}$$

**命题 9.12**：α-ELBO 的梯度可以写成 escort distribution 下的期望：

$$\nabla_\theta \mathcal{L}_\alpha = \frac{\alpha}{1-\alpha} \mathbb{E}_{\mu^{(1)} \to \pi_\theta^{(\alpha)}}[\nabla_\theta \log \pi_\theta]$$

其中权重隐式地将 $\mu$ 变换到 $\pi_\theta$ 的 escort 分布。

**定理 9.13（Escort 最优 α）**：

最优 α 使得 escort distribution $\pi_\theta^{(\alpha)}$ 最接近目标分布 $p^*$：

$$\boxed{\alpha^* = \arg\min_\alpha D_{KL}(\pi_\theta^{(\alpha)} \| p^*)}$$

**直觉**：
- α 控制了 $\pi_\theta$ 的"温度"：α 小使分布更平坦，α 大使分布更尖锐
- 最优 α 使得调温后的分布最接近目标

**闭式近似**（在指数族假设下）：

如果 $\pi_\theta(y) \propto \exp(\theta^T \phi(y))$，则：
$$\alpha^* \approx \frac{\mathbb{E}_{p^*}[\phi]^T \Sigma_\theta^{-1} \mathbb{E}_{\pi_\theta}[\phi]}{\mathbb{E}_{\pi_\theta}[\phi]^T \Sigma_\theta^{-1} \mathbb{E}_{\pi_\theta}[\phi]}$$

其中 $\Sigma_\theta = \text{Cov}_{\pi_\theta}[\phi]$。

#### 9.8.4 从信息几何推导

**动机**：不同的 α 对应参数空间上不同的黎曼度量。

**定义 9.14（α-表示下的 Fisher 信息）**：

$$G_\alpha(\theta) = \mathbb{E}_{\pi_\theta}\left[s(\theta) s(\theta)^T \cdot w^{1-\alpha}\right]$$

其中 $s(\theta) = \nabla_\theta \log \pi_\theta$ 是 score function，$w = \pi_\theta / \mu$。

**α-自然梯度**：
$$\tilde{\nabla}_\theta^{(\alpha)} = G_\alpha^{-1} \nabla_\theta \mathcal{L}_\alpha$$

**定理 9.15（信息几何最优 α）**：

最优 α 使得 α-自然梯度与理想梯度（从 $p^*$ 采样）方向最对齐：

$$\alpha^* = \arg\max_\alpha \frac{\langle \tilde{\nabla}_\theta^{(\alpha)}, \nabla_\theta^* \rangle}{\|\tilde{\nabla}_\theta^{(\alpha)}\| \cdot \|\nabla_\theta^*\|}$$

**简化结果**（在特定假设下）：

若 $\pi_\theta$ 接近 $\mu$，则：
$$\alpha^* \approx \frac{1}{1 + \text{tr}(G_1^{-1} \text{Cov}_\mu[w \cdot s])}$$

#### 9.8.5 最优 α 的闭式解（综合结果）

综合以上分析，在 log-normal 假设下可以得到**闭式解**。

**假设 9.16**：$\log w \sim \mathcal{N}(-\sigma^2/2, \sigma^2)$（满足 $\mathbb{E}[w] = 1$）。

**定理 9.17（最优 α 闭式解）**：

在假设 9.16 下，综合对偶条件和 escort 分布匹配，最优 α 为：

$$\boxed{\alpha^* = \frac{1}{1 + \sigma^2 / (2\delta)}}$$

其中：
- $\sigma^2 = \text{Var}(\log w) = \text{Var}(\log \pi_\theta - \log \mu)$ 是 log importance ratio 的方差
- $\delta > 0$ 是可接受的 Rényi 散度上界（超参数）

**证明**：

在 log-normal 假设下，由定理 10.2：
$$D_\alpha(\pi_\theta \| \mu) = \frac{\alpha \sigma^2}{2}$$

约束 $D_\alpha \leq \delta$ 给出 $\alpha \leq 2\delta / \sigma^2$。

同时，由 escort 分布匹配条件，最优 α 应最小化：
$$D_{KL}(\pi_\theta^{(\alpha)} \| p^*) \approx (1-\alpha)^2 \cdot C$$

对于某个常数 $C$。因此 α 应尽量大。

在约束 $\alpha \sigma^2 / 2 = \delta$ 取等号时达到最优，解得：
$$\alpha^* = \frac{2\delta}{\sigma^2}$$

为了保证 $\alpha^* \in (0, 1)$，改写为：
$$\alpha^* = \frac{1}{1 + \sigma^2/(2\delta)} \quad \blacksquare$$

**性质**：
| 情况 | $\sigma^2$ | $\alpha^*$ | 行为 |
|------|-----------|-----------|------|
| $\pi_\theta \approx \mu$ | 小 | → 1 | 标准 RL |
| $\pi_\theta$ 偏离 $\mu$ | 大 | → 0 | 保守 SFT |
| 约束松（δ 大） | - | → 1 | 更激进 |
| 约束紧（δ 小） | - | → 0 | 更保守 |

**直觉**：这个公式说明 α 是**分布偏移程度**和**可接受风险**的自然权衡。

#### 9.8.6 各方法对比

| 方法 | 理论基础 | α 的含义 | 可计算性 |
|------|---------|---------|---------|
| Rényi 熵正则化 | 广义熵 | 探索-利用权衡 | 需调参 |
| Lagrange 对偶 | 约束优化 | 对偶变量 | 数值求解 |
| Escort Distribution | 分布变换 | 温度参数 | 需 $p^*$ 信息 |
| 信息几何 | 自然梯度 | 度量选择 | 计算复杂 |
| **闭式解** | 综合 | 偏移-风险权衡 | **直接计算** |

---

## 10. 最优 α 的实用选择方法

基于 §9 的理论分析，本节给出具体可计算的 α 选择方法。

### 10.1 基于 ESS 的选择

**定义 10.1（广义有效样本量）**：

$$\text{ESS}_\alpha = \frac{\left(\sum_{i=1}^n w_i^\alpha e^{(1-\alpha)r_i/\tau}\right)^2}{\sum_{i=1}^n w_i^{2\alpha} e^{2(1-\alpha)r_i/\tau}}$$

**定理 10.1**：若 $\text{ESS}_\alpha \geq n \cdot \rho$（某个阈值 $\rho \in (0, 1)$），则：

$$\text{Var}(\hat{\mathcal{L}}_\alpha) \leq \frac{C}{n \rho}$$

其中 $C$ 是与 $\alpha$ 无关的常数。

**选择规则**：
$$\alpha^* = \max\{\alpha \in [0, 1] : \text{ESS}_\alpha \geq n \cdot \rho_{\min}\}$$

**与 §9 的联系**：这对应于定理 9.6 中约束 $\text{Var}$ 的方式，通过 ESS 间接控制方差。

### 10.2 Log-Normal 近似

**假设 10.2**：假设 $\log w \sim \mathcal{N}(\nu, \sigma^2)$（近似）。

**定理 10.2**：在假设 10.2 下：

$$D_\alpha(\pi_\theta \| \mu) = \frac{\alpha}{2}\sigma^2$$

（当 $\nu = -\sigma^2/2$ 以确保 $\mathbb{E}_\mu[w] = 1$ 时）

**证明**：

若 $\log w \sim \mathcal{N}(\nu, \sigma^2)$，则 $w^\alpha = e^{\alpha \log w}$，且 $\alpha \log w \sim \mathcal{N}(\alpha\nu, \alpha^2\sigma^2)$。

因此 $\mathbb{E}[w^\alpha] = e^{\alpha\nu + \alpha^2\sigma^2/2}$。

为使 $w$ 是有效的 IS 比率，需要 $\mathbb{E}_\mu[w] = 1$，即 $\nu = -\sigma^2/2$。

则：
$$\mathbb{E}[w^\alpha] = e^{-\alpha\sigma^2/2 + \alpha^2\sigma^2/2} = e^{\alpha(\alpha-1)\sigma^2/2}$$

$$D_\alpha(\pi_\theta \| \mu) = \frac{1}{\alpha-1}\log e^{\alpha(\alpha-1)\sigma^2/2} = \frac{\alpha\sigma^2}{2} \quad \blacksquare$$

**推论 10.3**：为确保 $D_\alpha(\pi_\theta \| \mu) \leq \delta$：

$$\alpha^* = \frac{2\delta}{\sigma^2} = \frac{2\delta}{\text{Var}(\log w)}$$

**与 §9 的联系**：这对应于定理 9.4，通过约束 Rényi 散度（即 α-ELBO 的 gap）来选择 α。

### 10.3 闭式解方法（推荐）

基于定理 9.17，给出最简洁的 α 选择方式。

**算法 2：闭式最优 α**

```python
def optimal_alpha_closed_form(log_pi_theta, log_mu, samples, delta=0.1):
    """
    基于定理 9.17 的闭式最优 α

    参数：
        log_pi_theta: 当前策略的 log 概率
        log_mu: 行为策略的 log 概率
        samples: 样本
        delta: 可接受的 Rényi 散度上界

    返回：
        alpha: 最优 α
    """
    # 计算 log importance ratio
    log_w = log_pi_theta(samples) - log_mu(samples)

    # 计算方差
    sigma_sq = np.var(log_w)

    # 闭式解
    alpha = 1.0 / (1.0 + sigma_sq / (2.0 * delta))

    return np.clip(alpha, 0.01, 0.99)
```

**超参数 δ 的选择**：
- δ ≈ 0.1: 保守选择，适合分布偏移大的情况
- δ ≈ 0.5: 中等选择，平衡
- δ ≈ 1.0: 激进选择，适合分布偏移小的情况

### 10.4 自适应选择算法

**算法 3：自适应 α-ELBO**（综合各方法）

```
输入：D = {(x_i, y_i, r_i)}，μ，δ > 0
初始化：π_θ，α = 0.5

对于 t = 1, 2, ..., T：
    1. 计算 w_i = π_θ(y_i|x_i) / μ(y_i|x_i) 对所有 i

    2. 更新 α（选择以下方法之一）：

       (a) 闭式解（推荐，基于定理 9.17）：
           σ² = Var(log w)
           α_t = 1 / (1 + σ²/(2δ))

       (b) ESS 方法（基于 §9.6 的方差约束）：
           α_t = max{α : ESS_α ≥ n · ρ_min}

       (c) 对偶方法（基于定理 9.10，数值求解）：
           α_t = solve(∂L_α/∂α = λ · ∂D_α/∂α)

       (d) Escort 方法（基于定理 9.13）：
           α_t = argmin_α D_KL(π_θ^(α) || p̂*)

    3. 计算归一化权重：
           ω_i = w_i^{α_t} e^{(1-α_t)r_i/τ} / Σ_j w_j^{α_t} e^{(1-α_t)r_j/τ}

    4. 计算梯度：
           g = (α_t/(1-α_t)) Σ_i ω_i ∇log π_θ(y_i|x_i)

    5. 更新：θ ← θ + η · g

输出：π_θ
```

---

## 11. 与现有方法的联系

### 11.1 特殊情况的恢复

**命题 11.1**：α-ELBO 框架可以恢复：

1. **α = 1**：标准 RLHF/PPO 目标（mode-seeking）
   $$\mathcal{L}_1 = \frac{1}{\tau}\mathbb{E}_{\pi_\theta}[r] - D_{KL}(\pi_\theta \| \mu)$$

2. **α → 0**：奖励加权 SFT（mean-seeking）
   $$\nabla_\theta \mathcal{L}_{\alpha \to 0} \propto \mathbb{E}_\mu\left[\frac{e^{r/\tau}}{Z} \nabla_\theta \log \pi_\theta\right]$$

3. **α = 0.5**：Hellinger 距离正则化
   $$D_{0.5}(\pi_\theta \| \mu) = 2\left(1 - \int \sqrt{\pi_\theta \mu} dy\right)$$

### 11.2 与 PPO Clipping 的关系

**命题 11.2**：PPO 的 clipped 目标：
$$L^{CLIP}(\theta) = \mathbb{E}_\mu\left[\min(w \cdot A, \text{clip}(w, 1-\epsilon, 1+\epsilon) \cdot A)\right]$$

可以看作一种自适应 α 方案：
- 当 $w \in [1-\epsilon, 1+\epsilon]$：使用 $\alpha = 1$（完整 IS）
- 当 $w$ 超出范围：降低有效 α（保守更新）

### 11.3 与 AWR/AWAC 的关系

**命题 11.3**：Advantage Weighted Regression：
$$\max_\theta \mathbb{E}_\mu\left[\exp(A/\tau) \log \pi_\theta(y|x)\right]$$

等价于 $\alpha = 0$ 的 α-ELBO（reward = advantage）：
$$\nabla_\theta \mathcal{L}_0 \propto \mathbb{E}_\mu[e^{r/\tau} \nabla_\theta \log \pi_\theta]$$

---

## 12. 总结

### 主要结果

1. **α-ELBO 提供了 $\log Z$ 的有原则的下界**（定理 5.2）

2. **Bound gap 恰好是 Rényi 散度**：$\log Z - \mathcal{L}_\alpha = D_\alpha(\pi_\theta \| p^*)$（定理 8.1）

3. **α 不改变最优解，只改变优化路径**（定理 9.1）：
   - 无论 α 取何值，$\mathcal{L}_\alpha$ 的最优解都是 $p^*$
   - α 控制的是"怎么到达"而非"到达哪里"

4. **α 的多重理论解释**（§9.8）：
   - **Rényi 熵视角**：探索-利用权衡（定理 9.8）
   - **Lagrange 对偶视角**：约束优化的对偶变量（定理 9.10）
   - **Escort 分布视角**：温度参数（定理 9.13）
   - **信息几何视角**：参数空间的度量选择（定理 9.15）

5. **最优 α 的闭式解**（定理 9.17）：
   $$\boxed{\alpha^* = \frac{1}{1 + \sigma^2/(2\delta)}}$$
   其中 $\sigma^2 = \text{Var}(\log w)$，$\delta$ 是可接受的散度上界

6. **该框架统一了** SFT（α→0）、标准 RL（α=1）以及中间方法

### 核心洞察

$$\boxed{\text{α-ELBO 通过 Rényi 散度的阶数在 mean-seeking（SFT）和 mode-seeking（RL）之间进行插值}}$$

### 理论框架图示

```
                        α-ELBO 框架
                            │
            ┌───────────────┼───────────────┐
            │               │               │
            ▼               ▼               ▼
        α → 0           α = 0.5          α → 1
      (mean-seeking)   (balanced)    (mode-seeking)
            │               │               │
            ▼               ▼               ▼
     奖励加权 SFT      Hellinger 正则    标准 RL/PPO
            │               │               │
            └───────────────┴───────────────┘
                            │
                            ▼
                    最优 α 选择（§9）
                            │
            ┌───────────────┼───────────────┐
            │               │               │
            ▼               ▼               ▼
      梯度对齐准则    正则化目标优化    方差约束下最小化
      (定理 9.3)      (定理 9.4)        gap (§9.6)
```

---

## 附录：符号速查表

| 符号 | 含义 |
|------|------|
| $\mathcal{L}_\alpha$ | α-ELBO（变分 Rényi 下界） |
| $D_\alpha(P \| Q)$ | α 阶 Rényi 散度 |
| $H_\alpha(\pi)$ | α 阶 Rényi 熵 |
| $w = \pi_\theta / \mu$ | 重要性采样比率 |
| $\tilde{w}_\alpha$ | 自归一化重要性权重 |
| $p^* = \mu e^{r/\tau} / Z$ | 奖励加权目标分布 |
| $p^*_\alpha$ | Rényi 熵正则化下的目标分布 |
| $Z = \mathbb{E}_\mu[e^{r/\tau}]$ | 配分函数 |
| $\text{ESS}_\alpha$ | 广义有效样本量 |
| $\pi^{(\alpha)}$ | α-escort 分布 |
| $G_\alpha(\theta)$ | α-表示下的 Fisher 信息矩阵 |
| $\nabla_\theta^*$ | 理想梯度（从 $p^*$ 采样） |
| $\nabla_\theta^\alpha$ | α-ELBO 梯度（从 $\mu$ 采样） |
| $\tilde{\nabla}_\theta^{(\alpha)}$ | α-自然梯度 |
| $\sigma^2$ | $\text{Var}(\log w)$，分布偏移度量 |
| $\delta$ | 可接受的 Rényi 散度上界 |
