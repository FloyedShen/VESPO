# 从经典 IS 文献补充到 IS Reshape 框架的理论改进

本文档整理可以直接融入 IS Reshape 框架的关键理论结果。

---

## 1. PSIS 的 k̂ 诊断：更好的 γ 选择方法

### 1.1 核心思想

Vehtari et al. (2024) 的关键洞察：**Pareto k 参数直接刻画了 IS 的可靠性**。

$$\text{upper tail of } w \sim \text{GPD}(\mu, \sigma, k)$$

| k̂ 值 | 可靠性 | 有限矩数 | 实践建议 |
|------|-------|---------|---------|
| < 0.5 | 非常可靠 | > 2 | 放心使用 |
| 0.5 - 0.7 | 可靠 | 1.4 - 2 | 可以使用 |
| 0.7 - 1 | 不太可靠 | 1 - 1.4 | 需要调整 |
| ≥ 1 | 不可靠 | < 1 | 必须处理 |

### 1.2 融入 IS Reshape：k̂-based γ 选择

**核心定理（新）**：对于 IS Reshape 权重 $w^\gamma$，其 Pareto k 参数满足：

$$\hat{k}(w^\gamma) \approx \gamma \cdot \hat{k}(w)$$

**证明思路**：
- 若 $w$ 的尾部服从 Pareto(k)，即 $P(w > t) \sim t^{-1/k}$
- 则 $w^\gamma$ 的尾部：$P(w^\gamma > t) = P(w > t^{1/\gamma}) \sim t^{-1/(\gamma k)}$
- 因此 $k(w^\gamma) = \gamma \cdot k(w)$

**实践意义**：选择 γ 使得 $\gamma \cdot \hat{k}(w) < 0.7$

$$\boxed{\gamma^*_{\text{PSIS}} = \min\left(1, \frac{0.7}{\hat{k}(w)}\right)}$$

### 1.3 与 ESS 方法的对比

| 方法 | 基于 | 优点 | 缺点 |
|-----|------|------|------|
| ESS 约束 | 二阶矩 | 简单，无分布假设 | 对尾部不敏感 |
| PSIS k̂ | 尾部分布 | 直接刻画尾部风险 | 需要拟合 Pareto |

**推荐**：结合使用
$$\gamma^* = \min(\gamma^*_{\text{ESS}}, \gamma^*_{\text{PSIS}})$$

---

## 2. SNIS 偏差公式：精确的 Bias 刻画

### 2.1 Kong-Owen 偏差公式

对于自归一化 IS 估计量 $\hat{\mu} = \sum_i \bar{w}_i h_i$，偏差满足（Kong 1992, Owen 2013）：

$$\text{Bias}(\hat{\mu}) = -\frac{1}{n} \cdot \frac{\text{Cov}_g(w, h)}{\mathbb{E}_g[w]} + O(n^{-2})$$

### 2.2 应用到 IS Reshape

设 $h = r \cdot \nabla\log\pi_\theta$，对于 $f(w) = w^\gamma$：

$$\boxed{\text{Bias}(\gamma) \approx -\frac{1}{n} \cdot \frac{\text{Cov}_\mu(w^\gamma, r \cdot \nabla\log\pi)}{\mathbb{E}_\mu[w^\gamma]}}$$

**关键洞察**：

1. **Cov 项的符号**：通常 $\text{Cov}(w^\gamma, r \cdot \nabla\log\pi) > 0$（高权重样本有高梯度）
   - 因此偏差通常为**负**
   - 意味着梯度估计**低估**了真实值

2. **γ 的影响**：
   - γ 减小 → $w^\gamma$ 更均匀 → Cov 减小 → 偏差减小
   - 但 γ = 0 时，Cov = 0，偏差为零

3. **精确公式**（Log-Normal 假设下）：

$$\text{Bias}(\gamma) \approx -\frac{1}{n} \cdot \left(e^{\sigma^2 \gamma} - 1\right) \cdot \rho_{r,\nabla}$$

其中 $\rho_{r,\nabla}$ 是奖励和梯度的相关系数。

### 2.3 Bias-Variance 的精确权衡

结合 §8 的方差分析，我们有完整的 MSE 分解：

$$\text{MSE}(\gamma) = \underbrace{\left(\frac{e^{\sigma^2\gamma} - 1}{n}\right)^2 \rho^2}_{\text{Bias}^2} + \underbrace{\frac{e^{\sigma^2\gamma(2\gamma-1)}}{n} \cdot V_0}_{\text{Variance}}$$

**最优 γ** 可以通过求导得到（数值解）。

---

## 3. Defensive IS 的混合思想：稳定性改进

### 3.1 Hesterberg 的核心思想

使用混合提议分布：
$$g_{\text{def}}(x) = \alpha \cdot f(x) + (1-\alpha) \cdot g_0(x)$$

权重变为：
$$w_{\text{def}} = \frac{f}{g_{\text{def}}} = \frac{w}{\alpha w + (1-\alpha)} \leq \frac{1}{\alpha}$$

### 3.2 融入 IS Reshape：Defensive Power Weighting

**定义（新）**：Defensive IS Reshape

$$f_{\text{def}}(w; \gamma, \alpha) = \frac{w^\gamma}{\alpha w^\gamma + (1-\alpha)}$$

**性质**：

| 参数设置 | 行为 |
|---------|------|
| α = 0 | 标准 IS Reshape: $f = w^\gamma$ |
| α = 1 | Unweighted: $f = 1$ |
| α ∈ (0,1) | 混合，权重有界 $\leq 1/\alpha$ |

**关键优势**：即使 γ = 1（完整 IS），权重也是有界的！

$$f_{\text{def}}(w; 1, \alpha) = \frac{w}{\alpha w + (1-\alpha)} \leq \frac{1}{\alpha}$$

### 3.3 实现

```python
def defensive_is_reshape(log_w, gamma, alpha=0.1):
    """
    Defensive IS Reshape: 结合 power tempering 和 defensive mixing

    f(w) = w^γ / (α·w^γ + (1-α))

    优点：即使 γ=1 也有有界权重
    """
    w_gamma = torch.exp(gamma * log_w)
    f_def = w_gamma / (alpha * w_gamma + (1 - alpha))

    # 归一化
    return f_def / f_def.sum()
```

### 3.4 理论分析

**定理（Defensive IS Reshape 的偏差）**：

$$\mathbb{E}_\mu[f_{\text{def}}(w)] = \mathbb{E}_\mu\left[\frac{w^\gamma}{\alpha w^\gamma + (1-\alpha)}\right] \neq 1$$

偏差大小取决于 α 和 w 的分布。当 α 较小时，偏差较小。

**Bias-Variance 权衡**：α 增大 → 方差减小，偏差增大

---

## 4. AIS 启发的 γ-Annealing 策略

### 4.1 理论基础

Neal (2001) 的 Annealed Importance Sampling：

$$p_t = p_0^{1-\beta_t} \cdot p_T^{\beta_t}, \quad \beta: 0 \to 1$$

### 4.2 IS Reshape 的几何插值解释

我们的目标分布族：
$$p_\gamma \propto \mu \cdot \left(\frac{p^*}{\mu}\right)^\gamma = \mu^{1-\gamma} \cdot (p^*)^\gamma$$

这正是 $p_0 = \mu$（SFT目标）和 $p_1 = p^*$（RL目标）的**几何插值**！

### 4.3 γ-Annealing 训练策略

**动机**：直接从 γ=0 跳到 γ=1 可能导致训练不稳定

**策略 1：线性 Annealing**
```python
def linear_gamma_schedule(step, total_steps, gamma_init=0.0, gamma_final=1.0):
    return gamma_init + (gamma_final - gamma_init) * step / total_steps
```

**策略 2：Cosine Annealing**
```python
def cosine_gamma_schedule(step, total_steps, gamma_init=0.0, gamma_final=1.0):
    progress = step / total_steps
    return gamma_init + (gamma_final - gamma_init) * (1 - np.cos(np.pi * progress)) / 2
```

**策略 3：ESS-Adaptive Annealing**
```python
def adaptive_gamma_schedule(current_gamma, log_w, target_ess_ratio=0.3):
    """
    根据当前 ESS 自适应增加 γ
    只有当 ESS 足够高时才增加 γ
    """
    ess_ratio = compute_ess_ratio(current_gamma * log_w)

    if ess_ratio > target_ess_ratio * 1.5:
        # ESS 很充足，可以增加 γ
        return min(current_gamma + 0.1, 1.0)
    elif ess_ratio < target_ess_ratio * 0.8:
        # ESS 不足，需要减小 γ
        return max(current_gamma - 0.05, 0.0)
    else:
        return current_gamma
```

### 4.4 与 PPO Clip 的对比

| 方面 | PPO Clip | γ-Annealing |
|-----|----------|-------------|
| 约束方式 | 硬截断 w ∈ [1-ε, 1+ε] | 软压缩 w^γ |
| 随训练变化 | ε 通常固定 | γ 逐渐增加 |
| 理论解释 | Trust region | 分布 annealing |
| 极限行为 | 总是截断 | γ→1 时恢复完整 IS |

---

## 5. Rényi Divergence 视角：方差的散度解释

### 5.1 核心联系

α-Rényi 散度定义：
$$D_\alpha(p \| q) = \frac{1}{\alpha - 1} \log \mathbb{E}_q\left[\left(\frac{p}{q}\right)^\alpha\right]$$

**关键等式**：
$$\mathbb{E}_\mu[w^\gamma] = \exp\left((\gamma - 1) \cdot D_\gamma(\pi_\theta \| \mu)\right)$$

当 $\gamma > 1$ 时，$D_\gamma > 0$，所以 $\mathbb{E}[w^\gamma] > 1$。

### 5.2 方差的散度表示

回顾方差：$\text{Var}(\gamma) \propto \mathbb{E}_\mu[w^{2\gamma}]$

$$\boxed{\mathbb{E}_\mu[w^{2\gamma}] = \exp\left((2\gamma - 1) \cdot D_{2\gamma}(\pi_\theta \| \mu)\right)}$$

**解释**：
- 方差由 **2γ-Rényi 散度** 决定
- γ = 0.5 时：$D_1 = D_{KL}$，方差与 KL 散度相关
- γ = 1 时：$D_2 = \chi^2$-divergence 相关

### 5.3 最优 γ 的散度视角

**定理（新）**：在 Rényi 散度框架下，最优 γ 满足：

$$\gamma^* = \arg\min_\gamma \left[\text{Bias}^2(\gamma) + \text{Var}(\gamma)\right]$$

$$= \arg\min_\gamma \left[c_1 \cdot e^{2(\gamma-1)D_\gamma} + c_2 \cdot e^{(2\gamma-1)D_{2\gamma}}\right]$$

其中 $c_1, c_2$ 是常数。

---

## 6. Control Variates：进一步降低方差

### 6.1 Owen & Zhou (2000) 的方法

使用 $\mathbb{E}[w^\gamma]$ 的可估计性构造控制变量：

$$\hat{\mu}_{CV} = \frac{1}{n}\sum_i w_i^\gamma r_i - c \cdot \left(\frac{1}{n}\sum_i w_i^\gamma - \mathbb{E}[w^\gamma]\right)$$

### 6.2 应用到 IS Reshape

**问题**：$\mathbb{E}_\mu[w^\gamma]$ 通常未知。

**解决方案 1**：使用样本估计

对于 self-normalized 版本，我们实际使用：
$$\bar{w}_i = \frac{w_i^\gamma}{\sum_j w_j^\gamma}$$

此时 $\sum_i \bar{w}_i = 1$，自动满足约束。

**解决方案 2**：双样本估计

```python
def cv_is_reshape(log_w, rewards, gamma, cv_coef='optimal'):
    """
    带 Control Variate 的 IS Reshape
    """
    n = len(log_w)
    w_gamma = torch.exp(gamma * log_w)

    # 分成两半：一半估计 E[w^γ]，一半做主估计
    n_half = n // 2
    E_w_gamma = w_gamma[:n_half].mean()

    # 主估计
    main_est = (w_gamma[n_half:] * rewards[n_half:]).mean()

    # Control variate 修正
    cv_term = w_gamma[n_half:].mean() - E_w_gamma

    if cv_coef == 'optimal':
        # 最优系数
        cov = torch.cov(torch.stack([w_gamma[n_half:], rewards[n_half:]]))
        c = cov[0, 1] / (cov[0, 0] + 1e-8)
    else:
        c = cv_coef

    return main_est - c * cv_term
```

---

## 7. 权重平滑：PSIS 风格的处理

### 7.1 问题

极端权重（outliers）会主导估计，导致高方差。

### 7.2 PSIS 的平滑方法

用 Pareto 拟合替换最大的 M 个权重：

$$w_{n-M+z} \leftarrow F^{-1}\left(\frac{z - 0.5}{M}\right)$$

### 7.3 融入 IS Reshape

```python
def psis_smoothed_is_reshape(log_w, gamma, M=None):
    """
    PSIS 平滑的 IS Reshape

    1. 计算 w^γ
    2. 拟合尾部 Pareto 分布
    3. 平滑最大的 M 个权重
    """
    w_gamma = torch.exp(gamma * log_w)
    n = len(w_gamma)

    if M is None:
        M = max(int(n * 0.2), 10)  # 取 20% 或至少 10 个

    # 排序
    sorted_w, indices = torch.sort(w_gamma)

    # 拟合 Pareto 到尾部
    tail = sorted_w[-M:]
    threshold = sorted_w[-M-1]

    # 简化的 Pareto 平滑（实际应该用 MLE 拟合）
    k_hat = estimate_pareto_k(tail - threshold)

    if k_hat > 0.7:
        # 需要平滑
        # 用 expected order statistics 替换
        for z in range(M):
            # F^{-1}((z + 0.5) / M) for Pareto
            quantile = threshold * ((M / (z + 0.5)) ** k_hat)
            sorted_w[n - M + z] = min(sorted_w[n - M + z], quantile)

    # 恢复顺序
    smoothed_w = torch.zeros_like(w_gamma)
    smoothed_w[indices] = sorted_w

    return smoothed_w / smoothed_w.sum()
```

---

## 8. 统一：完整的改进框架

### 8.1 增强版 γ 选择算法

```python
def enhanced_gamma_selection(log_w, method='combined',
                              ess_min=0.3, psis_k_max=0.7):
    """
    结合多种方法的 γ 选择

    方法:
    - 'ess': 基于 ESS 约束
    - 'psis': 基于 Pareto k̂
    - 'combined': 取两者的最小值
    """
    w = torch.exp(log_w)

    # 1. ESS-based γ
    gamma_ess = binary_search_gamma_ess(log_w, ess_min)

    # 2. PSIS-based γ
    k_hat = estimate_pareto_k(w)
    gamma_psis = min(1.0, psis_k_max / (k_hat + 1e-8))

    if method == 'ess':
        return gamma_ess
    elif method == 'psis':
        return gamma_psis
    else:  # combined
        return min(gamma_ess, gamma_psis)
```

### 8.2 训练流程整合

```python
class EnhancedISReshapeTrainer:
    """
    增强版 IS Reshape 训练器

    融合了:
    1. PSIS k̂ 诊断
    2. Defensive mixing (可选)
    3. γ-annealing
    4. Control variates (可选)
    5. 权重平滑
    """

    def __init__(self,
                 gamma_init=0.3,
                 gamma_schedule='adaptive',  # 'fixed', 'linear', 'cosine', 'adaptive'
                 use_defensive=False,
                 defensive_alpha=0.1,
                 use_psis_smoothing=True,
                 use_control_variate=False):
        self.gamma = gamma_init
        self.gamma_schedule = gamma_schedule
        self.use_defensive = use_defensive
        self.alpha = defensive_alpha
        self.use_psis = use_psis_smoothing
        self.use_cv = use_control_variate

        # 诊断记录
        self.history = {
            'gamma': [], 'ess_ratio': [], 'pareto_k': []
        }

    def compute_weights(self, log_w, rewards):
        """计算最终权重"""

        # 1. 基础 power weighting
        w_gamma = torch.exp(self.gamma * log_w)

        # 2. 可选：Defensive mixing
        if self.use_defensive:
            w_gamma = w_gamma / (self.alpha * w_gamma + (1 - self.alpha))

        # 3. 可选：PSIS 平滑
        if self.use_psis:
            w_gamma = psis_smooth(w_gamma)

        # 4. 归一化
        weights = w_gamma / w_gamma.sum()

        # 5. 可选：Control variate
        if self.use_cv:
            # 实现略
            pass

        return weights

    def update_gamma(self, log_w, step, total_steps):
        """更新 γ"""

        if self.gamma_schedule == 'fixed':
            pass

        elif self.gamma_schedule == 'linear':
            self.gamma = 0.0 + 1.0 * step / total_steps

        elif self.gamma_schedule == 'cosine':
            progress = step / total_steps
            self.gamma = (1 - np.cos(np.pi * progress)) / 2

        elif self.gamma_schedule == 'adaptive':
            # 基于 ESS 和 PSIS k̂
            gamma_new = enhanced_gamma_selection(log_w)
            # 平滑更新
            self.gamma = 0.9 * self.gamma + 0.1 * gamma_new

        # 记录诊断
        self.history['gamma'].append(self.gamma)
        self.history['ess_ratio'].append(compute_ess_ratio(self.gamma * log_w))
        self.history['pareto_k'].append(estimate_pareto_k(torch.exp(log_w)))
```

---

## 9. 总结：融入框架的关键改进

| 来源 | 融入的理论/方法 | 具体改进 |
|-----|----------------|---------|
| **PSIS** | k̂ 诊断 | $\gamma^* \leq 0.7/\hat{k}(w)$ |
| **SNIS** | Kong-Owen 偏差公式 | 精确的 Bias(γ) 表达式 |
| **Defensive IS** | 混合权重 | $f_{\text{def}} = w^\gamma/(\alpha w^\gamma + 1-\alpha)$ |
| **AIS** | 几何插值 | γ-annealing 训练策略 |
| **Rényi** | 散度视角 | Var(γ) ∝ exp((2γ-1)·D_{2γ}) |
| **Control Var** | 方差降低 | CV 修正项 |

### 最重要的三个改进

1. **PSIS k̂ + ESS 联合 γ 选择**：比单独使用任一方法更稳健
2. **γ-Annealing**：从 SFT 平滑过渡到 RL
3. **Defensive IS Reshape**：保证权重有界，即使 γ=1

这些改进让 IS Reshape 框架更加**理论完备**和**实践稳健**。
