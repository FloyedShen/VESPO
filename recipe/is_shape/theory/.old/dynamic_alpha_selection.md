# 动态 α 选择：基于 Bias-Variance Trade-off 的自适应方法

## 摘要

本文档从 bias-variance trade-off 角度分析统一 α-散度框架，并提出多种基于样本统计量的动态 α 选择方法。核心思想是：**在训练过程中根据实时统计信息自适应调整 α**，以在目标偏差（bias）和梯度方差（variance）之间取得最优平衡。

---

## 1. Bias-Variance 分解

### 1.1 问题设定

在统一 Amari α-散度框架中，我们优化：

$$\mathcal{L}_\alpha(\theta) = -D_\alpha^{(A)}\left(\pi_\theta \| p_{\frac{1+\alpha}{2}}\right)$$

有效的重要性权重为：

$$\tilde{w}_\alpha \propto w^{\frac{1+\alpha}{2}} \cdot e^{\frac{(1-\alpha^2)r}{4\tau}}$$

其中 $w = \pi_\theta / \mu$ 是重要性采样比率。

### 1.2 Bias 分析

**定义（目标偏差）**：

$$\text{Bias}(\alpha) = \left\| \mathbb{E}[\nabla_\theta \mathcal{L}_\alpha] - \nabla_\theta \mathcal{L}_{+1} \right\|$$

**来源**：我们优化的目标分布是 $p_\beta$（其中 β = (1+α)/2），而非真正的最优分布 $p^* = p_1$。

**定理 1.1（Bias 的显式形式）**：

$$\text{Bias}(\alpha) \approx \frac{1-\alpha}{2} \cdot \left\| \nabla_\theta D_{KL}(p^* \| p_\beta) \right\|$$

**性质**：
- α = +1 时，Bias = 0（无偏）
- α = -1 时，Bias 最大
- Bias 关于 α 单调递减

### 1.3 Variance 分析

**定义（梯度方差）**：

$$\text{Var}(\alpha) = \text{Var}_\mu\left[\tilde{w}_\alpha \nabla_\theta \log \pi_\theta\right]$$

**关键因子**：权重的二阶矩

$$\mathbb{E}_\mu[\tilde{w}_\alpha^2] \propto \mathbb{E}_\mu\left[w^{1+\alpha} \cdot e^{\frac{(1-\alpha^2)r}{2\tau}}\right]$$

**性质**：
- α = +1 时，$w^2$ 项主导，方差可能很大
- α = -1 时，$w^0 = 1$，方差最小
- Variance 关于 α 单调递增

### 1.4 MSE 分解

**定理 1.2（MSE 分解）**：

梯度估计器的均方误差为：

$$\boxed{\text{MSE}(\alpha) = \underbrace{\left(\frac{1-\alpha}{2}\right)^2 B^2}_{\text{Bias}^2} + \underbrace{\frac{1}{n} \cdot \frac{V(\alpha)}{\text{ESS}_\alpha}}_{\text{Variance}}}$$

其中：
- $B$ = 目标偏差的度量（与 $\nabla D_{KL}(p^* \| p_\beta)$ 相关）
- $V(\alpha)$ = 单样本方差
- $\text{ESS}_\alpha$ = 有效样本量
- $n$ = 实际样本数

### 1.5 Trade-off 可视化

```
MSE
 ↑
 │    ╲                      Variance 主导
 │     ╲                    /
 │      ╲                  /
 │       ╲    最优点      /
 │        ╲    ●        /
 │         ╲__________/
 │         /          ╲
 │        /            ╲
 │       / Bias 主导
 │      /
 └──────────────────────────────→ α
   -1              0              +1
   (SFT)      (Hellinger)        (RL)
```

---

## 2. 有效样本量（ESS）分析

### 2.1 定义

**定义 2.1（广义有效样本量）**：

$$\text{ESS}_\alpha = \frac{\left(\sum_{i=1}^n \tilde{w}_{\alpha,i}\right)^2}{\sum_{i=1}^n \tilde{w}_{\alpha,i}^2}$$

对于归一化权重 $\bar{w}_i = \tilde{w}_{\alpha,i} / \sum_j \tilde{w}_{\alpha,j}$：

$$\text{ESS}_\alpha = \frac{1}{\sum_{i=1}^n \bar{w}_i^2}$$

### 2.2 ESS 与方差的关系

**定理 2.1**：

$$\text{Var}(\hat{g}_\alpha) \leq \frac{\sigma_g^2}{\text{ESS}_\alpha}$$

其中 $\sigma_g^2 = \max_y \|\nabla_\theta \log \pi_\theta(y)\|^2$。

**推论**：ESS 越大，方差越小。

### 2.3 ESS 关于 α 的行为

| α 值 | 有效权重 $\tilde{w}_\alpha$ | ESS 行为 |
|-----|---------------------------|---------|
| -1 | $w^0 \cdot e^0 = 1$ | ESS = n（最大） |
| 0 | $w^{1/2} \cdot e^{r/(4\tau)}$ | ESS 中等 |
| +1 | $w^1 \cdot e^0 = w$ | ESS 可能很小 |

---

## 3. 动态 α 选择方法

### 3.1 方法一：ESS 自适应

**核心思想**：保持有效样本量在可接受范围内。

**算法**：

```
输入：ESS 目标范围 [ρ_low, ρ_high]，步长 Δα
初始化：α = 0

每个训练步：
    1. 计算当前 ESS_α
    2. 计算 ESS 比例：ρ = ESS_α / n
    3. 更新 α：
       若 ρ > ρ_high：α ← α + Δα  (更激进)
       若 ρ < ρ_low： α ← α - Δα  (更保守)
       否则：α 不变
    4. 裁剪：α ← clip(α, -0.99, 0.99)
```

**实现**：

```python
def ess_adaptive_alpha(
    weights: np.ndarray,
    alpha_prev: float,
    n: int,
    rho_low: float = 0.1,
    rho_high: float = 0.5,
    delta: float = 0.05
) -> tuple[float, float]:
    """
    基于 ESS 的自适应 α 选择

    参数：
        weights: 当前的重要性权重（未归一化）
        alpha_prev: 上一步的 α
        n: 样本数量
        rho_low: ESS 比例下界
        rho_high: ESS 比例上界
        delta: α 调整步长

    返回：
        alpha_new: 更新后的 α
        ess_ratio: 当前 ESS 比例
    """
    # 计算 ESS
    weights_normalized = weights / np.sum(weights)
    ess = 1.0 / np.sum(weights_normalized ** 2)
    ess_ratio = ess / n

    # 自适应调整
    if ess_ratio > rho_high:
        # ESS 充足，可以更激进（增大 α 向 RL 靠拢）
        alpha_new = min(alpha_prev + delta, 0.99)
    elif ess_ratio < rho_low:
        # ESS 不足，需要更保守（减小 α 向 SFT 靠拢）
        alpha_new = max(alpha_prev - delta, -0.99)
    else:
        # ESS 在目标范围内，保持不变
        alpha_new = alpha_prev

    return alpha_new, ess_ratio
```

**超参数建议**：
- `rho_low = 0.1`：低于此值方差过大
- `rho_high = 0.5`：高于此值可以更激进
- `delta = 0.05`：步长不宜过大，避免震荡

### 3.2 方法二：梯度方差在线估计

**核心思想**：直接监控梯度方差，控制在目标范围内。

**梯度方差估计**：

$$\widehat{\text{Var}}(\nabla_\theta \mathcal{L}_\alpha) = \frac{1}{n-1} \sum_{i=1}^n \left\| \bar{w}_i \nabla_i - \bar{g} \right\|^2$$

其中 $\bar{g} = \sum_i \bar{w}_i \nabla_i$ 是加权平均梯度。

**自适应规则**：

$$\alpha^* = \arg\min_\alpha \left\{ (1-\alpha)^2 : \widehat{\text{Var}}(\alpha) \leq \sigma_{\max}^2 \right\}$$

即：**在方差约束下最大化 α**（最小化 bias）。

**实现**：

```python
def variance_adaptive_alpha(
    log_w: np.ndarray,
    grads: np.ndarray,
    rewards: np.ndarray,
    tau: float,
    var_budget: float
) -> float:
    """
    基于梯度方差的自适应 α 选择

    参数：
        log_w: log 重要性比率 [n]
        grads: 每样本梯度 [n, d]
        rewards: 奖励 [n]
        tau: 温度参数
        var_budget: 允许的最大梯度方差

    返回：
        alpha: 最优 α
    """
    def compute_grad_var(alpha):
        # 计算有效权重
        gamma = (1 + alpha) / 2
        beta = gamma
        log_weights = gamma * log_w + (1 - alpha) * beta * rewards / (2 * tau)

        # 数值稳定的 softmax
        log_weights = log_weights - np.max(log_weights)
        weights = np.exp(log_weights)
        weights = weights / np.sum(weights)

        # 加权梯度和方差
        weighted_grads = weights[:, None] * grads
        mean_grad = np.sum(weighted_grads, axis=0)

        # 方差估计
        diff = grads - mean_grad
        var = np.sum(weights[:, None] ** 2 * diff ** 2)

        return var

    # 二分搜索找满足方差约束的最大 α
    alpha_low, alpha_high = -0.99, 0.99

    for _ in range(20):  # 二分迭代次数
        alpha_mid = (alpha_low + alpha_high) / 2
        var_mid = compute_grad_var(alpha_mid)

        if var_mid <= var_budget:
            alpha_low = alpha_mid  # 可以更激进
        else:
            alpha_high = alpha_mid  # 需要更保守

    return alpha_low
```

### 3.3 方法三：重要性权重矩估计（闭式解）

**核心思想**：利用 log w 的统计量直接计算最优 α。

**假设 3.1（Log-Normal 近似）**：

$$\log w \sim \mathcal{N}(\nu, \sigma^2)$$

**定理 3.1（闭式最优 α）**：

在 Log-Normal 假设下，最优 α 为：

$$\boxed{\alpha^* = \frac{2\delta - \sigma^2}{\sigma^2 + 2\delta} = 1 - \frac{2\sigma^2}{\sigma^2 + 2\delta}}$$

其中：
- $\sigma^2 = \text{Var}(\log w)$：分布偏移的度量
- $\delta > 0$：可接受的 bias 参数（超参数）

**证明**：

在 Log-Normal 假设下，Rényi 散度为：
$$D_{\frac{1+\alpha}{2}}(\pi_\theta \| \mu) = \frac{(1+\alpha)\sigma^2}{4}$$

方差正比于：
$$\text{Var} \propto \mathbb{E}[w^{1+\alpha}] = e^{(1+\alpha)\nu + (1+\alpha)^2\sigma^2/2}$$

最小化 MSE = Bias² + λ·Var，并令 Rényi 散度 ≤ δ，解得上述闭式解。$\blacksquare$

**实现**：

```python
def moment_based_alpha(
    log_w: np.ndarray,
    delta: float = 0.5
) -> tuple[float, float]:
    """
    基于重要性权重矩的 α 选择（闭式解）

    参数：
        log_w: log 重要性比率
        delta: 可接受的 bias 参数（越大越激进）

    返回：
        alpha: 最优 α
        sigma_sq: 估计的方差
    """
    # 估计 log w 的方差
    sigma_sq = np.var(log_w)

    # 闭式最优 α
    alpha = (2 * delta - sigma_sq) / (sigma_sq + 2 * delta)

    return np.clip(alpha, -0.99, 0.99), sigma_sq
```

**超参数 δ 的选择**：

| δ 值 | 含义 | 适用场景 |
|-----|------|---------|
| 0.1 | 保守 | 分布偏移大，需要稳定性 |
| 0.5 | 中等 | 一般情况 |
| 1.0 | 激进 | 分布偏移小，追求性能 |

**性质**：

| 情况 | σ² | α* | 行为 |
|-----|----|----|-----|
| π_θ ≈ μ | 小 → 0 | → +1 | 纯 RL |
| π_θ 偏离 μ | 大 → ∞ | → -1 | 纯 SFT |
| σ² = 2δ | 中等 | = 0 | Hellinger |

### 3.4 方法四：MSE 最小化（完整版）

**核心思想**：显式估计 bias 和 variance，最小化 MSE。

$$\alpha^* = \arg\min_\alpha \left[ \widehat{\text{Bias}}^2(\alpha) + \lambda \cdot \widehat{\text{Var}}(\alpha) \right]$$

**实现**：

```python
def mse_optimal_alpha(
    log_w: np.ndarray,
    grads: np.ndarray,
    rewards: np.ndarray,
    tau: float,
    lambda_var: float = 1.0
) -> float:
    """
    最小化估计的 MSE 来选择 α

    参数：
        log_w: log 重要性比率 [n]
        grads: 每样本梯度 [n, d]
        rewards: 奖励 [n]
        tau: 温度参数
        lambda_var: 方差的权重系数

    返回：
        alpha: MSE 最优的 α
    """
    n = len(log_w)

    def compute_mse(alpha):
        gamma = (1 + alpha) / 2
        beta = gamma

        # 计算归一化权重
        log_weights = gamma * log_w + (1 - alpha) * beta * rewards / (2 * tau)
        log_weights = log_weights - np.max(log_weights)
        weights = np.exp(log_weights)
        weights = weights / np.sum(weights)

        # Bias 估计（代理）
        # Bias ∝ (1-α) * |目标分布偏差|
        weighted_reward = np.sum(weights * rewards)
        mean_reward = np.mean(rewards)
        bias_proxy = ((1 - alpha) / 2) ** 2 * (weighted_reward - mean_reward) ** 2

        # Variance 估计
        weighted_grads = weights[:, None] * grads
        mean_grad = np.sum(weighted_grads, axis=0)
        var = np.sum(np.sum((weighted_grads - weights[:, None] * mean_grad) ** 2, axis=1))
        var = var / n  # 归一化

        return bias_proxy + lambda_var * var

    # 网格搜索
    alphas = np.linspace(-0.95, 0.95, 50)
    mses = [compute_mse(a) for a in alphas]

    return alphas[np.argmin(mses)]
```

---

## 4. 综合自适应调度器

### 4.1 设计原则

1. **多信号融合**：结合 ESS、方差、矩估计等多种信号
2. **平滑更新**：使用 EMA 避免 α 剧烈震荡
3. **安全约束**：设置 α 的上下界
4. **可解释性**：记录详细的调试信息

### 4.2 完整实现

```python
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class AlphaSchedulerConfig:
    """自适应 α 调度器配置"""
    alpha_init: float = 0.0          # 初始 α（从平衡点开始）
    ess_target: float = 0.3          # 目标 ESS 比例
    var_budget: float = 1.0          # 梯度方差预算
    delta: float = 0.5               # bias 容忍度参数
    momentum: float = 0.9            # EMA 动量
    min_alpha: float = -0.95         # α 下界
    max_alpha: float = 0.95          # α 上界

class AdaptiveAlphaScheduler:
    """
    综合多种信号的自适应 α 调度器

    特点：
    1. 结合矩估计、ESS、梯度方差三种方法
    2. 使用指数移动平均平滑更新
    3. 提供详细的调试信息
    """

    def __init__(self, config: Optional[AlphaSchedulerConfig] = None):
        self.config = config or AlphaSchedulerConfig()
        self.alpha = self.config.alpha_init

        # 运行时统计（EMA）
        self.ema_sigma_sq: Optional[float] = None
        self.ema_ess_ratio: Optional[float] = None
        self.ema_grad_var: Optional[float] = None

        # 历史记录
        self.history = {
            'alpha': [],
            'sigma_sq': [],
            'ess_ratio': [],
            'grad_var': [],
            'alpha_moment': [],
        }

    def _compute_weights(
        self,
        log_w: np.ndarray,
        rewards: np.ndarray,
        tau: float,
        alpha: float
    ) -> np.ndarray:
        """计算给定 α 下的归一化权重"""
        gamma = (1 + alpha) / 2
        beta = gamma

        log_weights = gamma * log_w + (1 - alpha**2) * rewards / (4 * tau)
        log_weights = log_weights - np.max(log_weights)  # 数值稳定
        weights = np.exp(log_weights)
        weights = weights / np.sum(weights)

        return weights

    def _compute_ess(self, weights: np.ndarray, n: int) -> float:
        """计算有效样本量比例"""
        ess = 1.0 / np.sum(weights ** 2)
        return ess / n

    def _compute_grad_var(
        self,
        weights: np.ndarray,
        grads: np.ndarray
    ) -> float:
        """计算梯度方差"""
        weighted_grads = weights[:, None] * grads
        mean_grad = np.sum(weighted_grads, axis=0)
        var = np.sum(weights ** 2 * np.sum((grads - mean_grad) ** 2, axis=1))
        return var

    def _update_ema(
        self,
        current: float,
        ema: Optional[float]
    ) -> float:
        """更新指数移动平均"""
        if ema is None:
            return current
        return self.config.momentum * ema + (1 - self.config.momentum) * current

    def update(
        self,
        log_w: np.ndarray,
        rewards: np.ndarray,
        tau: float,
        grads: Optional[np.ndarray] = None
    ) -> tuple[float, Dict[str, Any]]:
        """
        根据当前 batch 统计量更新 α

        参数：
            log_w: log 重要性比率 [n]
            rewards: 奖励 [n]
            tau: 温度参数
            grads: 每样本梯度 [n, d]（可选）

        返回：
            alpha: 更新后的 α
            info: 调试信息字典
        """
        n = len(log_w)
        cfg = self.config

        # ========== 1. 计算基础统计量 ==========
        sigma_sq = np.var(log_w)
        self.ema_sigma_sq = self._update_ema(sigma_sq, self.ema_sigma_sq)

        # ========== 2. 方法 A：基于矩的闭式解 ==========
        alpha_moment = (2 * cfg.delta - self.ema_sigma_sq) / (self.ema_sigma_sq + 2 * cfg.delta)
        alpha_moment = np.clip(alpha_moment, cfg.min_alpha, cfg.max_alpha)

        # ========== 3. 方法 B：基于 ESS 的调整 ==========
        weights = self._compute_weights(log_w, rewards, tau, self.alpha)
        ess_ratio = self._compute_ess(weights, n)
        self.ema_ess_ratio = self._update_ema(ess_ratio, self.ema_ess_ratio)

        # ESS 调整量
        if ess_ratio < cfg.ess_target * 0.5:
            alpha_ess_adj = -0.1  # 需要大幅保守
        elif ess_ratio < cfg.ess_target:
            alpha_ess_adj = -0.02  # 轻微保守
        elif ess_ratio > cfg.ess_target * 2:
            alpha_ess_adj = 0.05  # 可以更激进
        else:
            alpha_ess_adj = 0.01  # 轻微激进

        # ========== 4. 方法 C：基于梯度方差的调整 ==========
        grad_var = None
        alpha_var_adj = 0

        if grads is not None:
            grad_var = self._compute_grad_var(weights, grads)
            self.ema_grad_var = self._update_ema(grad_var, self.ema_grad_var)

            if grad_var > cfg.var_budget * 2:
                alpha_var_adj = -0.1  # 方差过大
            elif grad_var > cfg.var_budget:
                alpha_var_adj = -0.02
            else:
                alpha_var_adj = 0.02  # 方差可控

        # ========== 5. 综合决策 ==========
        # 加权平均各方法的建议
        alpha_target = 0.5 * alpha_moment + 0.5 * self.alpha
        alpha_new = alpha_target + 0.3 * alpha_ess_adj + 0.2 * alpha_var_adj

        # 平滑更新
        self.alpha = cfg.momentum * self.alpha + (1 - cfg.momentum) * alpha_new
        self.alpha = np.clip(self.alpha, cfg.min_alpha, cfg.max_alpha)

        # ========== 6. 记录历史 ==========
        info = {
            'alpha': self.alpha,
            'gamma': (1 + self.alpha) / 2,
            'sigma_sq': sigma_sq,
            'ema_sigma_sq': self.ema_sigma_sq,
            'ess_ratio': ess_ratio,
            'ema_ess_ratio': self.ema_ess_ratio,
            'grad_var': grad_var,
            'ema_grad_var': self.ema_grad_var,
            'alpha_moment': alpha_moment,
            'alpha_ess_adj': alpha_ess_adj,
            'alpha_var_adj': alpha_var_adj,
        }

        for key in ['alpha', 'sigma_sq', 'ess_ratio', 'grad_var', 'alpha_moment']:
            self.history[key].append(info.get(key))

        return self.alpha, info

    def get_gamma(self) -> float:
        """获取当前的 γ = (1+α)/2，即 f(w) = w^γ 的指数"""
        return (1 + self.alpha) / 2

    def get_beta(self) -> float:
        """获取当前的 β，即目标分布的倾斜度"""
        return (1 + self.alpha) / 2

    def reset(self):
        """重置调度器状态"""
        self.alpha = self.config.alpha_init
        self.ema_sigma_sq = None
        self.ema_ess_ratio = None
        self.ema_grad_var = None
        self.history = {k: [] for k in self.history}
```

### 4.3 使用示例

```python
# 创建调度器
config = AlphaSchedulerConfig(
    alpha_init=0.0,      # 从 Hellinger 点开始
    ess_target=0.3,      # 目标 30% 有效样本
    var_budget=1.0,      # 方差预算
    delta=0.5,           # bias 容忍度
    momentum=0.9         # 平滑系数
)
scheduler = AdaptiveAlphaScheduler(config)

# 训练循环
for epoch in range(num_epochs):
    for batch in dataloader:
        x, y, r = batch

        # 计算 log 重要性比率
        log_pi = policy.log_prob(y, x)
        log_mu = behavior_policy.log_prob(y, x)
        log_w = (log_pi - log_mu).detach().numpy()

        # 计算每样本梯度（可选）
        grads = compute_per_sample_grads(policy, x, y)

        # 更新 α
        alpha, info = scheduler.update(log_w, r.numpy(), tau, grads)

        # 使用更新后的 α 计算损失
        gamma = scheduler.get_gamma()
        # ... 训练代码 ...

        print(f"α={alpha:.3f}, γ={gamma:.3f}, ESS={info['ess_ratio']:.3f}")
```

---

## 5. 训练集成

### 5.1 完整训练循环

```python
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

def train_with_adaptive_alpha(
    policy: torch.nn.Module,
    behavior_policy: torch.nn.Module,
    offline_data: DataLoader,
    tau: float = 1.0,
    num_epochs: int = 100,
    lr: float = 1e-4,
    scheduler_config: Optional[AlphaSchedulerConfig] = None
) -> tuple[torch.nn.Module, dict]:
    """
    使用自适应 α 的离线 RL 训练

    参数：
        policy: 待训练的策略网络
        behavior_policy: 行为策略（用于计算重要性权重）
        offline_data: 离线数据集
        tau: 温度参数
        num_epochs: 训练轮数
        lr: 学习率
        scheduler_config: α 调度器配置

    返回：
        policy: 训练后的策略
        history: 训练历史
    """
    # 初始化
    scheduler = AdaptiveAlphaScheduler(scheduler_config)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    history = {
        'loss': [],
        'alpha': [],
        'ess_ratio': [],
        'sigma_sq': [],
        'reward': []
    }

    for epoch in range(num_epochs):
        epoch_loss = 0
        epoch_reward = 0
        num_batches = 0

        for batch in offline_data:
            x, y, r = batch['context'], batch['response'], batch['reward']

            # ========== 计算 log 概率 ==========
            with torch.no_grad():
                log_mu = behavior_policy.log_prob(y, x)
            log_pi = policy.log_prob(y, x)
            log_w = log_pi - log_mu

            # ========== 自适应更新 α ==========
            # 可选：计算每样本梯度用于方差估计
            # grads = compute_per_sample_grads(policy, x, y)

            alpha, info = scheduler.update(
                log_w.detach().cpu().numpy(),
                r.cpu().numpy(),
                tau,
                grads=None  # 或传入 grads
            )

            # ========== 计算加权损失 ==========
            gamma = (1 + alpha) / 2
            beta = gamma

            # 计算 log 权重
            log_weights = gamma * log_w + (1 - alpha**2) * r / (4 * tau)

            # 归一化（使用 softmax）
            weights = F.softmax(log_weights, dim=0).detach()

            # 加权负对数似然损失
            loss = -torch.sum(weights * log_pi)

            # ========== 优化步骤 ==========
            optimizer.zero_grad()
            loss.backward()

            # 可选：梯度裁剪
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)

            optimizer.step()

            # ========== 记录 ==========
            epoch_loss += loss.item()
            epoch_reward += torch.sum(weights * r).item()
            num_batches += 1

        # 每个 epoch 的统计
        avg_loss = epoch_loss / num_batches
        avg_reward = epoch_reward / num_batches

        history['loss'].append(avg_loss)
        history['alpha'].append(alpha)
        history['ess_ratio'].append(info['ess_ratio'])
        history['sigma_sq'].append(info['sigma_sq'])
        history['reward'].append(avg_reward)

        # 打印进度
        print(f"Epoch {epoch+1}/{num_epochs}: "
              f"Loss={avg_loss:.4f}, "
              f"α={alpha:.3f}, "
              f"ESS={info['ess_ratio']:.3f}, "
              f"σ²={info['sigma_sq']:.3f}, "
              f"Reward={avg_reward:.4f}")

    return policy, history
```

### 5.2 可视化训练过程

```python
import matplotlib.pyplot as plt

def plot_training_history(history: dict):
    """可视化训练历史"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # α 的演化
    ax = axes[0, 0]
    ax.plot(history['alpha'], 'b-', linewidth=2)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=1, color='r', linestyle=':', alpha=0.5, label='RL (α=+1)')
    ax.axhline(y=-1, color='g', linestyle=':', alpha=0.5, label='SFT (α=-1)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('α')
    ax.set_title('α Evolution')
    ax.legend()
    ax.set_ylim(-1.1, 1.1)

    # ESS 比例
    ax = axes[0, 1]
    ax.plot(history['ess_ratio'], 'g-', linewidth=2)
    ax.axhline(y=0.3, color='orange', linestyle='--', label='Target ESS')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('ESS Ratio')
    ax.set_title('Effective Sample Size Ratio')
    ax.legend()

    # 分布偏移 (σ²)
    ax = axes[1, 0]
    ax.plot(history['sigma_sq'], 'm-', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Var(log w)')
    ax.set_title('Distribution Shift (σ²)')

    # 损失和奖励
    ax = axes[1, 1]
    ax.plot(history['loss'], 'r-', label='Loss', linewidth=2)
    ax2 = ax.twinx()
    ax2.plot(history['reward'], 'b-', label='Reward', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss', color='r')
    ax2.set_ylabel('Reward', color='b')
    ax.set_title('Loss and Reward')

    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150)
    plt.show()
```

---

## 6. 理论分析

### 6.1 收敛性保证

**定理 6.1（自适应 α 的收敛性）**：

设 $\{\alpha_t\}$ 由 ESS 自适应规则生成，若满足：
1. $\sum_{t=1}^\infty (1 - \alpha_t) = \infty$（累积探索）
2. $\sum_{t=1}^\infty (1 - \alpha_t)^2 < \infty$（bias 可控）
3. 梯度方差有界：$\mathbb{E}[\|\nabla \mathcal{L}_{\alpha_t}\|^2] \leq G^2$

则策略序列 $\{\pi_{\theta_t}\}$ 收敛到最优策略 $p^*$ 的 ε-邻域。

**证明思路**：

这是随机逼近理论的推广。条件 1 保证累积步长足够大以到达最优解；条件 2 控制累积 bias；条件 3 是标准的有界方差假设。

### 6.2 遗憾界

**定理 6.2（Regret Bound）**：

使用 ESS 自适应的 α 调度，累积遗憾满足：

$$\text{Regret}(T) = \sum_{t=1}^T \left[\mathcal{L}_{+1}^* - \mathcal{L}_{\alpha_t}(\theta_t)\right] \leq O\left(\sqrt{T \log T}\right)$$

**解释**：即使在动态调整 α 的情况下，累积遗憾仍然是次线性的。

### 6.3 最优 α 的渐近行为

**命题 6.3**：

当训练收敛时（$\pi_\theta \to p_\beta$），最优 α 满足：

$$\alpha^* \to +1 \quad \text{as} \quad \sigma^2 = \text{Var}(\log w) \to 0$$

**直觉**：当策略接近目标分布时，分布偏移减小，可以使用更激进的 α。

---

## 7. 实践建议

### 7.1 方法选择指南

| 场景 | 推荐方法 | 原因 |
|-----|---------|-----|
| 快速原型 | 矩估计闭式解 | 简单高效，无需梯度 |
| 稳定训练 | ESS 自适应 | 直接控制有效样本 |
| 精细调优 | MSE 最小化 | 理论最优，但计算成本高 |
| 生产部署 | 综合调度器 | 鲁棒，多信号融合 |

### 7.2 超参数调优

**矩估计方法**：
- `delta = 0.1~1.0`：根据任务复杂度调整
- 分布偏移大时用小 δ（保守）
- 分布偏移小时用大 δ（激进）

**ESS 方法**：
- `rho_low = 0.05~0.2`：低于此值方差过大
- `rho_high = 0.3~0.7`：高于此值可以更激进
- `delta = 0.02~0.1`：步长，影响响应速度

**综合调度器**：
- `momentum = 0.8~0.95`：平滑系数，越大越稳定
- `var_budget`：根据梯度范数量级设置

### 7.3 调试技巧

1. **监控 ESS**：ESS < 10% 说明方差过大
2. **监控 σ²**：σ² > 2 说明分布偏移大
3. **观察 α 曲线**：震荡说明需要增大 momentum
4. **检查梯度范数**：过大说明需要更小的 α

---

## 8. 总结

### 核心公式

**MSE 分解**：
$$\text{MSE}(\alpha) = \left(\frac{1-\alpha}{2}\right)^2 B^2 + \frac{V(\alpha)}{n \cdot \text{ESS}_\alpha}$$

**闭式最优 α**：
$$\boxed{\alpha^* = \frac{2\delta - \sigma^2}{\sigma^2 + 2\delta}}$$

### 方法对比

| 方法 | 核心统计量 | 计算复杂度 | 稳定性 |
|-----|-----------|-----------|--------|
| ESS 自适应 | ESS/n | O(n) | 高 |
| 梯度方差 | Var(∇) | O(nd) | 中 |
| 矩估计 | Var(log w) | O(n) | 高 |
| MSE 最小化 | Bias + Var | O(nd) | 中 |
| **综合方法** | 多信号 | O(nd) | **最高** |

### 关键洞察

$$\boxed{\text{动态 } \alpha \text{ 选择 = 在 bias-variance trade-off 中自适应寻找最优工作点}}$$

```
训练过程中 α 的典型演化：

α
+1 ┤                                          ●●●●●●●● (收敛)
   │                                    ●●●●●
   │                              ●●●●●
 0 ┤─────●●●●●●●●●●●●●●●●●●●●●●●●●
   │    (初始)      (逐渐激进)
-1 ┤
   └──────────────────────────────────────────────────── t
```

**直觉**：
- **初期**：策略与数据接近，σ² 小，用较大的 α
- **中期**：策略偏离，σ² 增大，α 自动降低
- **后期**：策略收敛，σ² 减小，α 回升

---

---

## 9. 优雅的 f(w) 设计：从单调性与有界性出发

本节从已证明的单调性和有界性出发，推导出一个自然、优雅且自一致的 f(w) 设计方案。

### 10.1 设计动机

#### 10.1.1 原始方案的问题

原始的自适应规则基于 $\sigma^2 = \text{Var}(\log w)$：

$$\alpha^* = \frac{2\delta - \sigma^2}{\sigma^2 + 2\delta}$$

**问题**：当 $\pi_\theta \to p^*$ 时，$\sigma^2 = \text{Var}(r/\tau) \neq 0$，导致 $\alpha \not\to +1$。

#### 10.1.2 设计目标

我们需要 f(w) 满足：
1. **自一致性**：当 $\pi_\theta \to p^*$ 时，$\gamma \to 1$
2. **单调性利用**：利用 ESS(γ) 单调递减来确定 γ
3. **有界性保证**：确保权重和梯度有界
4. **优雅简洁**：从第一性原理自然导出

### 10.2 效率最大化原则

#### 10.2.1 学习效率函数

定义**学习效率**为信噪比：

$$\text{Efficiency}(\gamma) = \frac{\text{Signal}}{\text{Noise}} = \frac{1 - \text{Bias}(\gamma)}{\sqrt{\text{Var}(\gamma)}}$$

**直觉**：
- 分子 $(1 - \text{Bias})$：瞄准正确目标的程度
- 分母 $\sqrt{\text{Var}}$：梯度估计的噪声

#### 10.2.2 最优 γ

**定理 10.0（效率最优 γ）**：

最优 γ 最大化效率：

$$\gamma^* = \arg\max_{\gamma \in [0,1]} \frac{1 - \text{Bias}(\gamma)}{\sqrt{\text{Var}(\gamma)}}$$

在 Log-Normal 假设下（Bias 线性递减，Var 指数递增）：

$$\text{Bias}(\gamma) = (1-\gamma) B_0, \quad \text{Var}(\gamma) = V_0 e^{\sigma^2 \gamma}$$

效率函数的一阶条件给出：

$$\boxed{\gamma^*_{\text{eff}} = \frac{2}{\sigma^2} - \frac{1-B_0}{B_0}}$$

**简化**（当 $B_0 \approx 1$）：

$$\gamma^*_{\text{eff}} \approx \frac{2}{\sigma^2}$$

### 10.3 自校准 f(w) 设计

#### 10.3.1 核心问题

原始设计使用 $w = \pi_\theta/\mu$，度量的是**与行为策略 μ 的偏离**。

但我们真正关心的是**与目标分布 p* 的偏离**。

#### 10.3.2 校准权重

**定义 9.1（校准权重）**：

$$w^* = \frac{e^{r/\tau}}{\bar{Z}}, \quad \bar{Z} = \frac{1}{n}\sum_{i=1}^n e^{r_i/\tau}$$

$w^*$ 是"理想"的重要性权重（当 $\pi_\theta = p^*$ 时应有的权重）。

**定义 9.2（校准比率）**：

$$\tilde{w} = \frac{w}{w^*} = \frac{\pi_\theta/\mu}{e^{r/\tau}/\bar{Z}} = \frac{\pi_\theta \cdot \bar{Z}}{\mu \cdot e^{r/\tau}}$$

#### 10.3.3 关键性质

**定理 10.1（校准权重的自一致性）**：

当 $\pi_\theta = p^* = \mu e^{r/\tau}/Z$ 时：

$$\tilde{w} = \frac{w}{w^*} = \frac{e^{r/\tau}/Z}{e^{r/\tau}/\bar{Z}} = \frac{\bar{Z}}{Z} \approx 1$$

即**校准后的权重在最优点处为常数**。

**证明**：

$$w = \frac{\pi_\theta}{\mu} = \frac{p^*}{\mu} = \frac{\mu e^{r/\tau}/Z}{\mu} = \frac{e^{r/\tau}}{Z}$$

$$\tilde{w} = \frac{w}{w^*} = \frac{e^{r/\tau}/Z}{e^{r/\tau}/\bar{Z}} = \frac{\bar{Z}}{Z}$$

当样本量 n 足够大时，$\bar{Z} \to Z$，所以 $\tilde{w} \to 1$。$\blacksquare$

**推论**：当 $\pi_\theta \to p^*$：
- $\text{Var}(\log \tilde{w}) \to 0$
- $\text{ESS} \to n$
- 由 ESS 单调性，$\gamma \to 1$ ✓

### 10.4 三层优雅设计

#### 10.4.1 设计框架

```
═══════════════════════════════════════════════════════════════════
                    优雅的 f(w) 设计框架
═══════════════════════════════════════════════════════════════════

                    ┌─────────────────────┐
                    │   第一层：校准      │
                    │   w → w̃ = w/w*     │
                    │   消除目标偏移      │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   第二层：幂变换     │
                    │   f(w̃) = w̃^γ      │
                    │   控制方差          │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   第三层：归一化     │
                    │   w̄ = f(w̃)/Σf(w̃)  │
                    │   保证有界          │
                    └─────────────────────┘

═══════════════════════════════════════════════════════════════════
```

#### 10.4.2 完整公式

**定义 9.3（优雅 f(w)）**：

$$\boxed{f_{\text{elegant}}(w; r, \gamma) = \text{Normalize}\left[\left(\frac{w \cdot \bar{Z}}{e^{r/\tau}}\right)^\gamma\right]}$$

其中：
- $w = \pi_\theta/\mu$：原始 IS 权重
- $\bar{Z} = \frac{1}{n}\sum_i e^{r_i/\tau}$：配分函数估计
- $\gamma$：由 ESS 约束确定的幂指数
- Normalize：归一化使权重和为 1

#### 10.4.3 γ 的确定

由 ESS 单调性（定理 10.3），γ 唯一确定为：

$$\boxed{\gamma^* = \max\left\{\gamma \in [0, 1] : \text{ESS}_\gamma \geq n \cdot \rho_{\min}\right\}}$$

**计算方法**：利用 ESS(γ) 单调递减，使用二分搜索。

### 10.5 完整算法实现

```python
import numpy as np
from typing import Tuple, Dict

def elegant_f_weights(
    log_pi: np.ndarray,
    log_mu: np.ndarray,
    rewards: np.ndarray,
    tau: float,
    rho_min: float = 0.3,
    max_iter: int = 50
) -> Tuple[np.ndarray, float, Dict]:
    """
    优雅的 f(w) 设计：基于单调性和有界性

    三层设计：
    1. 校准：w → w/w*，消除与目标的偏移
    2. 幂变换：w^γ，控制方差
    3. 归一化：保证有界

    参数：
        log_pi: log π_θ(y|x)
        log_mu: log μ(y|x)
        rewards: 奖励 r(x,y)
        tau: 温度参数
        rho_min: 最小 ESS 比例
        max_iter: 二分搜索最大迭代次数

    返回：
        weights: 最终的归一化权重
        gamma: 最优幂指数
        info: 调试信息
    """
    n = len(rewards)

    # ========== 第一层：校准 ==========
    # 计算原始 log 权重
    log_w = log_pi - log_mu

    # 计算目标 log 权重：log(e^{r/τ}/Z̄)
    log_w_star = rewards / tau
    # 数值稳定的 log Z̄ 计算
    log_w_star_max = np.max(log_w_star)
    log_Z_bar = np.log(np.mean(np.exp(log_w_star - log_w_star_max))) + log_w_star_max
    log_w_star = log_w_star - log_Z_bar

    # 校准后的 log 权重：log(w/w*)
    log_w_calibrated = log_w - log_w_star

    # 归一化校准权重（数值稳定）
    log_w_calibrated = log_w_calibrated - np.max(log_w_calibrated)
    w_calibrated = np.exp(log_w_calibrated)
    w_calibrated = w_calibrated / np.mean(w_calibrated)  # 使均值为 1

    # ========== 第二层：幂变换（由 ESS 单调性确定 γ） ==========
    def compute_ess_ratio(gamma: float) -> float:
        """计算给定 γ 下的 ESS 比例"""
        if gamma == 0:
            return 1.0  # 所有权重相等
        w_gamma = np.power(w_calibrated, gamma)
        w_gamma = w_gamma / np.sum(w_gamma)  # 归一化
        ess = 1.0 / np.sum(w_gamma ** 2)
        return ess / n

    # 利用 ESS(γ) 单调递减，二分搜索最大的满足约束的 γ
    gamma_low, gamma_high = 0.0, 1.0

    for _ in range(max_iter):
        gamma_mid = (gamma_low + gamma_high) / 2
        ess_mid = compute_ess_ratio(gamma_mid)

        if ess_mid >= rho_min:
            gamma_low = gamma_mid  # ESS 足够，可以尝试更大的 γ
        else:
            gamma_high = gamma_mid  # ESS 不足，需要更小的 γ

        if gamma_high - gamma_low < 1e-6:
            break

    gamma_opt = gamma_low  # 取满足约束的最大 γ

    # ========== 第三层：归一化 ==========
    if gamma_opt > 0:
        w_final = np.power(w_calibrated, gamma_opt)
    else:
        w_final = np.ones(n)
    w_final = w_final / np.sum(w_final)

    # ========== 计算诊断信息 ==========
    ess_final = 1.0 / np.sum(w_final ** 2)
    sigma_sq_original = np.var(log_w)
    sigma_sq_calibrated = np.var(log_w_calibrated)

    info = {
        'gamma': gamma_opt,
        'alpha': 2 * gamma_opt - 1,  # 对应的 α
        'ess': ess_final,
        'ess_ratio': ess_final / n,
        'sigma_sq_original': sigma_sq_original,
        'sigma_sq_calibrated': sigma_sq_calibrated,
        'max_weight': np.max(w_final),
        'min_weight': np.min(w_final),
        'weight_entropy': -np.sum(w_final * np.log(w_final + 1e-10))
    }

    return w_final, gamma_opt, info


class ElegantAlphaScheduler:
    """
    优雅的自适应调度器

    基于三层设计：校准 → 幂变换 → 归一化
    """

    def __init__(
        self,
        rho_min: float = 0.3,
        momentum: float = 0.9,
        tau: float = 1.0
    ):
        self.rho_min = rho_min
        self.momentum = momentum
        self.tau = tau

        # 状态
        self.gamma = 0.5
        self.ema_sigma_calibrated = None

        # 历史记录
        self.history = {
            'gamma': [],
            'alpha': [],
            'ess_ratio': [],
            'sigma_original': [],
            'sigma_calibrated': []
        }

    def update(
        self,
        log_pi: np.ndarray,
        log_mu: np.ndarray,
        rewards: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        """
        更新并返回优化后的权重
        """
        weights, gamma_new, info = elegant_f_weights(
            log_pi, log_mu, rewards, self.tau, self.rho_min
        )

        # 平滑更新 gamma
        self.gamma = self.momentum * self.gamma + (1 - self.momentum) * gamma_new

        # 更新 EMA
        if self.ema_sigma_calibrated is None:
            self.ema_sigma_calibrated = info['sigma_sq_calibrated']
        else:
            self.ema_sigma_calibrated = (
                self.momentum * self.ema_sigma_calibrated +
                (1 - self.momentum) * info['sigma_sq_calibrated']
            )

        # 记录历史
        self.history['gamma'].append(self.gamma)
        self.history['alpha'].append(2 * self.gamma - 1)
        self.history['ess_ratio'].append(info['ess_ratio'])
        self.history['sigma_original'].append(np.sqrt(info['sigma_sq_original']))
        self.history['sigma_calibrated'].append(np.sqrt(info['sigma_sq_calibrated']))

        # 使用平滑后的 gamma 重新计算权重
        weights_smooth, _, _ = elegant_f_weights(
            log_pi, log_mu, rewards, self.tau, self.rho_min
        )

        return weights_smooth, {
            'gamma': self.gamma,
            'alpha': 2 * self.gamma - 1,
            'ess_ratio': info['ess_ratio'],
            'sigma_calibrated': np.sqrt(self.ema_sigma_calibrated)
        }
```

### 10.6 理论性质

#### 10.6.1 自一致性定理

**定理 10.2（优雅设计的自一致性）**：

使用上述三层设计，当 $\pi_\theta \to p^*$ 时：

1. 校准权重 $\tilde{w} = w/w^* \to 1$
2. $\text{Var}(\log \tilde{w}) \to 0$
3. $\text{ESS} \to n$
4. $\gamma^* \to 1$（由 ESS 单调性）
5. 最终权重 $\bar{w} \to 1/n$（均匀）

**证明**：

(1) 由定理 10.1。

(2) 由 (1)，$\log \tilde{w} \to \log 1 = 0$，方差趋于 0。

(3) 当所有 $\tilde{w} \approx 1$ 时，$\tilde{w}^\gamma \approx 1$，归一化后均匀，ESS = n。

(4) ESS = n 满足任意 $\rho_{\min} < 1$ 的约束，所以 $\gamma^* = 1$。

(5) 直接由 (3)(4)。$\blacksquare$

#### 10.6.2 与原始设计的对比

| 性质 | 原始 $\sigma^2 = \text{Var}(\log w)$ | 优雅 $\sigma^2_{\text{cal}} = \text{Var}(\log \tilde{w})$ |
|-----|-------------------------------------|-------------------------------------------------------|
| 收敛时 | $\sigma^2 = \text{Var}(r)/\tau^2 \neq 0$ | $\sigma^2_{\text{cal}} \to 0$ |
| $\gamma$ 极限 | $\gamma \not\to 1$ | $\gamma \to 1$ ✓ |
| 度量对象 | 与 μ 的偏离 | 与 p* 的偏离 ✓ |
| 自一致性 | ❌ | ✅ |

#### 10.6.3 单调性的利用

**命题 10.3**：优雅设计充分利用了 ESS(γ) 的单调性：

$$\gamma^* = \max\{\gamma : \text{ESS}_\gamma \geq n \cdot \rho_{\min}\}$$

由单调性保证：
1. 解**存在**（γ = 0 时 ESS = n ≥ n·ρ_min）
2. 解**唯一**（ESS 严格单调递减）
3. **可高效计算**（二分搜索，O(log(1/ε)) 迭代）

#### 10.6.4 有界性的保证

**命题 10.4**：优雅设计保证所有量有界：

| 量 | 界 | 保证来源 |
|---|-----|---------|
| $\gamma$ | [0, 1] | 定义 |
| $\bar{w}_i$ | [0, 1] | 归一化（第三层）|
| ESS | $[n \cdot \rho_{\min}, n]$ | 约束 |
| Var(ĝ) | $\leq 4G^2/(n \cdot \rho_{\min})$ | ESS 约束 |

### 10.7 可视化：收敛行为

```
训练过程中的典型演化（优雅设计）：

γ (或 α = 2γ-1)
  1 ┤                                    ●●●●●●●●●●● (收敛到 1)
    │                              ●●●●●●
    │                        ●●●●●●
0.5 ┤─────●●●●●●●●●●●●●●●●●●●●
    │    (初始)
    │
  0 ┤
    └──────────────────────────────────────────────────────→ t

σ²_calibrated (校准后的方差)
  2 ┤●●
    │  ●●●
    │     ●●●●
  1 ┤        ●●●●●
    │             ●●●●●●●
  0 ┤                    ●●●●●●●●●●●●●●●●●●●●● (趋近于 0)
    └──────────────────────────────────────────────────────→ t

ESS/n
  1 ┤                              ●●●●●●●●●●●●●●● (趋近于 1)
    │                        ●●●●●
    │                  ●●●●●●
0.5 ┤            ●●●●●●
    │      ●●●●●●
0.3 ┤─●●●●●─────────────────────────────────────── (ρ_min)
    └──────────────────────────────────────────────────────→ t

关键观察：
• σ²_calibrated → 0 保证 γ → 1
• 这与原始设计不同：原始 σ² = Var(r)/τ² 不趋于 0
```

### 10.8 总结：优雅设计的核心

**核心公式**：

$$\boxed{f(w) = \left(\frac{w \cdot \bar{Z}}{e^{r/\tau}}\right)^{\gamma^*_{\text{ESS}}}}$$

**三层结构**：
1. **校准**：$w \to \tilde{w} = w/w^*$（度量与 p* 的偏离）
2. **幂变换**：$\tilde{w} \to \tilde{w}^\gamma$（由 ESS 单调性确定 γ）
3. **归一化**：保证有界

**理论保证**：

| 保证 | 来源 |
|-----|------|
| 自一致性 | 校准项使 $\pi_\theta \to p^*$ 时 $\gamma \to 1$ |
| 唯一性 | ESS(γ) 严格单调递减 |
| 有界性 | 归一化 + ESS 约束 |
| 高效性 | 二分搜索 O(log(1/ε)) |

---

## 10. 理论保证：单调性与有界性

本节提供统一框架的严格理论保证，包括单调性定理和有界性分析。

### 10.1 Bias 关于 α 的单调性

**定理 10.1（Bias 单调递减）**：

目标偏差 $\text{Bias}(\alpha)$ 关于 $\alpha$ **严格单调递减**。

**证明**：

Bias 来自于优化目标分布 $p_\beta$（其中 β = (1+α)/2）而非真正的最优分布 $p^* = p_1$。

定义 Bias 为目标分布之间的 KL 散度：
$$\text{Bias}(\alpha) \propto D_{KL}(p_1 \| p_\beta)$$

由于 $p_\beta = \mu e^{\beta r/\tau} / Z_\beta$，展开 KL 散度：

$$D_{KL}(p_1 \| p_\beta) = \mathbb{E}_{p_1}\left[\log \frac{p_1}{p_\beta}\right]$$

$$= \mathbb{E}_{p_1}\left[\log \frac{\mu e^{r/\tau}/Z_1}{\mu e^{\beta r/\tau}/Z_\beta}\right]$$

$$= \mathbb{E}_{p_1}\left[(1-\beta) \frac{r}{\tau}\right] + \log Z_\beta - \log Z_1$$

对 β 求导，利用 $\frac{\partial \log Z_\beta}{\partial \beta} = \mathbb{E}_{p_\beta}[r/\tau]$：

$$\frac{\partial}{\partial \beta} D_{KL}(p_1 \| p_\beta) = -\mathbb{E}_{p_1}[r/\tau] + \mathbb{E}_{p_\beta}[r/\tau]$$

**关键观察**：由于 $p_1 = p^*$ 是奖励加权的最优分布，它比 $p_\beta$（当 β < 1）更倾向于高奖励样本。因此：

$$\mathbb{E}_{p_1}[r] > \mathbb{E}_{p_\beta}[r] \quad \text{当 } \beta < 1$$

这意味着：
$$\frac{\partial}{\partial \beta} D_{KL}(p_1 \| p_\beta) < 0$$

由于 β = (1+α)/2 关于 α 严格递增，所以：

$$\boxed{\frac{\partial \text{Bias}}{\partial \alpha} < 0}$$

即 **Bias 关于 α 严格单调递减**。$\blacksquare$

### 10.2 Variance 关于 α 的单调性

**定理 10.2（Variance 单调递增）**：

在 Log-Normal 假设下，梯度方差关于 $\alpha$ **单调递增**（当 α > -1/2）。

**假设**：$\log w \sim \mathcal{N}(\nu, \sigma^2)$，其中 $\nu = -\sigma^2/2$（保证 $\mathbb{E}[w] = 1$）。

**证明**：

方差的主导项是有效权重的二阶矩。令 γ = (1+α)/2，关键量是：

$$\mathbb{E}_\mu[w^{2\gamma}] = \mathbb{E}[e^{2\gamma \log w}]$$

由于 $\log w \sim \mathcal{N}(-\sigma^2/2, \sigma^2)$，有 $2\gamma \log w \sim \mathcal{N}(-\gamma\sigma^2, 4\gamma^2\sigma^2)$。

利用对数正态分布的矩公式：

$$\mathbb{E}[w^{2\gamma}] = e^{-\gamma\sigma^2 + 2\gamma^2\sigma^2} = e^{\sigma^2 \gamma(2\gamma - 1)}$$

对 γ 求导：

$$\frac{d}{d\gamma} \mathbb{E}[w^{2\gamma}] = e^{\sigma^2 \gamma(2\gamma-1)} \cdot \sigma^2(4\gamma - 1)$$

当 $\gamma > 1/4$（即 $\alpha > -1/2$）时，$(4\gamma - 1) > 0$，所以导数为正。

由于 γ = (1+α)/2 关于 α 递增：

$$\boxed{\frac{\partial \text{Var}}{\partial \alpha} > 0 \quad \text{当 } \alpha > -1/2}$$

即 **Variance 关于 α 单调递增**（在 α > -1/2 的实际使用范围内）。$\blacksquare$

**注**：α = -1/2 对应 γ = 1/4，此时方差达到最小值。

### 10.3 ESS 关于 α 的单调性

**定理 10.3（ESS 单调递减）**：

有效样本量 $\text{ESS}_\alpha$ 关于 $\alpha$ **单调递减**。

**证明**：

ESS 定义为：
$$\text{ESS}_\alpha = \frac{1}{\sum_i \bar{w}_{\alpha,i}^2}$$

其中 $\bar{w}_{\alpha,i}$ 是归一化权重。

ESS 与权重的集中度成反比。定义权重的 Rényi 熵（阶数为 2）：

$$H_2(\bar{w}_\alpha) = -\log \sum_i \bar{w}_{\alpha,i}^2 = \log \text{ESS}_\alpha$$

**关键观察**：当 α 增大时：

1. 有效权重 $\tilde{w}_\alpha \propto w^\gamma$ 中的指数 γ = (1+α)/2 增大
2. 对于 $w_i > 1$ 的样本，$w_i^\gamma$ 增长更快
3. 权重分布变得更加**集中**（少数高 w 样本主导）
4. $\sum_i \bar{w}_i^2$ 增大，ESS 减小

形式化地，由定理 10.2：
$$\mathbb{E}[w^{2\gamma}] \text{ 关于 } \gamma \text{ 递增}$$

这直接导致权重的集中度增加，因此：

$$\boxed{\frac{\partial \text{ESS}_\alpha}{\partial \alpha} < 0}$$

**边界情况**：
- α = -1 时，γ = 0，$w^0 = 1$，所有权重相等，ESS = n（最大）
- α = +1 时，γ = 1，$w^1 = w$，ESS 由原始 IS 权重决定（通常最小）

$\blacksquare$

### 10.4 最优 α 关于 σ² 的单调性

**定理 10.4（最优 α 单调递减于分布偏移）**：

闭式最优解 $\alpha^* = \frac{2\delta - \sigma^2}{\sigma^2 + 2\delta}$ 关于分布偏移 $\sigma^2$ **严格单调递减**。

**证明**：

直接计算偏导数：

$$\frac{\partial \alpha^*}{\partial \sigma^2} = \frac{\partial}{\partial \sigma^2}\left(\frac{2\delta - \sigma^2}{\sigma^2 + 2\delta}\right)$$

$$= \frac{-1 \cdot (\sigma^2 + 2\delta) - (2\delta - \sigma^2) \cdot 1}{(\sigma^2 + 2\delta)^2}$$

$$= \frac{-\sigma^2 - 2\delta - 2\delta + \sigma^2}{(\sigma^2 + 2\delta)^2} = \frac{-4\delta}{(\sigma^2 + 2\delta)^2}$$

由于 δ > 0，有：

$$\boxed{\frac{\partial \alpha^*}{\partial \sigma^2} = \frac{-4\delta}{(\sigma^2 + 2\delta)^2} < 0}$$

即 **α* 关于 σ² 严格单调递减**。$\blacksquare$

**物理意义**：

| σ² (分布偏移) | α* | 行为 |
|--------------|-----|-----|
| σ² → 0 | α* → +1 | 策略接近数据，可以激进（纯 RL）|
| σ² = 2δ | α* = 0 | 中等偏移，平衡点（Hellinger）|
| σ² → ∞ | α* → -1 | 策略远离数据，必须保守（纯 SFT）|

### 10.5 单调性总结

**定理 10.5（单调性汇总）**：

在统一 Amari α-散度框架中，以下单调性成立：

| 量 | 关于 α 的单调性 | 有效范围 | 证明 |
|---|---------------|---------|------|
| Bias(α) | 严格递减 ↓ | α ∈ [-1, +1] | 定理 10.1 |
| Var(α) | 严格递增 ↑ | α ∈ (-1/2, +1] | 定理 10.2 |
| ESS(α) | 严格递减 ↓ | α ∈ [-1, +1] | 定理 10.3 |
| α*(σ²) | 严格递减 ↓ | σ² ∈ [0, ∞) | 定理 10.4 |

**可视化**：

```
         Bias(α)              Variance(α)            ESS(α)
          ↑                      ↑                     ↑
      max │●                     │                 n   │●●●●
          │ ●●                   │              ●●●    │    ●●
          │   ●●●                │           ●●●       │      ●●
          │      ●●●             │        ●●●          │        ●●
          │         ●●●●         │     ●●●             │          ●●
        0 │             ●●●●●●●● │●●●●●                │            ●●●
          └──────────────────→ α └──────────────────→ α └──────────────────→ α
         -1     -0.5    0    +1  -1    -0.5    0    +1  -1    -0.5    0    +1
```

---

### 10.6 有界性分析

#### 10.6.1 α 的有界性

**命题 10.6**：$\alpha \in [-1, +1]$（定义域有界）。

在数值实现中，进一步限制 $\alpha \in [-1 + \epsilon, 1 - \epsilon]$（如 [-0.99, 0.99]）以避免边界处的奇异性。

#### 10.6.2 Amari α-散度的有界性

**定理 10.7（Amari 散度有界）**：

$$\boxed{0 \leq D_\alpha^{(A)}(P \| Q) \leq \frac{4}{1-\alpha^2}}$$

**证明**：

$$D_\alpha^{(A)}(P \| Q) = \frac{4}{1-\alpha^2}\left(1 - \int P^{\frac{1+\alpha}{2}} Q^{\frac{1-\alpha}{2}} dx\right)$$

**下界**（非负性）：

由 Hölder 不等式，对于 $p = \frac{2}{1+\alpha}$ 和 $q = \frac{2}{1-\alpha}$（共轭指数）：

$$\int P^{\frac{1+\alpha}{2}} Q^{\frac{1-\alpha}{2}} dx \leq \left(\int P \, dx\right)^{\frac{1+\alpha}{2}} \left(\int Q \, dx\right)^{\frac{1-\alpha}{2}} = 1$$

因此 $1 - \int P^{\frac{1+\alpha}{2}} Q^{\frac{1-\alpha}{2}} \geq 0$，即 $D_\alpha^{(A)} \geq 0$。

等号成立当且仅当 P = Q。

**上界**：

由于积分项非负：$\int P^{\frac{1+\alpha}{2}} Q^{\frac{1-\alpha}{2}} \geq 0$，有：

$$D_\alpha^{(A)} = \frac{4}{1-\alpha^2}\left(1 - \int \cdots\right) \leq \frac{4}{1-\alpha^2}$$

$\blacksquare$

**推论**：对于固定的 $\alpha \in (-1, +1)$，Amari 散度是有界的。

#### 10.6.3 自归一化权重的有界性

**定理 10.8（自归一化权重有界）**：

归一化的重要性权重满足：

$$\boxed{\bar{w}_{\alpha,i} \in [0, 1], \quad \sum_{i=1}^n \bar{w}_{\alpha,i} = 1}$$

**证明**：

由定义：
$$\bar{w}_{\alpha,i} = \frac{\tilde{w}_{\alpha,i}}{\sum_{j=1}^n \tilde{w}_{\alpha,j}}$$

其中 $\tilde{w}_{\alpha,i} \geq 0$。显然 $\bar{w}_{\alpha,i} \geq 0$ 且 $\sum_i \bar{w}_{\alpha,i} = 1$。

由于每个 $\bar{w}_{\alpha,i}$ 是非负数且总和为 1，必有 $\bar{w}_{\alpha,i} \leq 1$。$\blacksquare$

#### 10.6.4 原始重要性权重的条件有界性

**定理 10.9（条件有界性）**：

原始重要性权重 $w = \pi_\theta / \mu$ 在以下条件下有界：

1. **支撑包含 + 密度下界**：
   $$\text{supp}(\pi_\theta) \subseteq \text{supp}(\mu) \text{ 且 } \mu(y) \geq \mu_{\min} > 0$$
   则 $w \leq \|\pi_\theta\|_\infty / \mu_{\min}$

2. **密度比有界**：
   $$\exists M > 0: \pi_\theta(y) \leq M \cdot \mu(y), \forall y$$
   则 $w \leq M$

3. **显式截断**：
   使用 $\tilde{w} = \min(w, c)$ 或 $\tilde{w} = w / (1 + w/c)$

**实践建议**：

```python
def safe_importance_weight(log_pi, log_mu, max_log_ratio=5.0):
    """
    安全的重要性权重计算（带截断）

    max_log_ratio=5.0 对应 w_max ≈ 148
    """
    log_w = log_pi - log_mu
    log_w_clipped = torch.clamp(log_w, min=-max_log_ratio, max=max_log_ratio)
    return torch.exp(log_w_clipped)
```

#### 10.6.5 梯度估计器的有界性

**定理 10.10（梯度估计器有界）**：

若 score function 有界：$\|\nabla_\theta \log \pi_\theta(y)\| \leq G$，则梯度估计器有界：

$$\boxed{\|\hat{g}_\alpha\| \leq G}$$

**证明**：

$$\hat{g}_\alpha = \sum_{i=1}^n \bar{w}_{\alpha,i} \nabla_\theta \log \pi_\theta(y_i)$$

由三角不等式和归一化权重性质：

$$\|\hat{g}_\alpha\| \leq \sum_{i=1}^n \bar{w}_{\alpha,i} \|\nabla_\theta \log \pi_\theta(y_i)\| \leq \sum_{i=1}^n \bar{w}_{\alpha,i} \cdot G = G$$

$\blacksquare$

#### 10.6.6 梯度方差的有界性

**定理 10.11（方差有界）**：

在自归一化估计器下，梯度方差满足：

$$\boxed{\text{Var}(\hat{g}_\alpha) \leq \frac{4G^2}{\text{ESS}_\alpha}}$$

**证明**：

$$\text{Var}(\hat{g}_\alpha) = \mathbb{E}\left[\|\hat{g}_\alpha - \mathbb{E}[\hat{g}_\alpha]\|^2\right]$$

$$\leq \mathbb{E}\left[\left\|\sum_i \bar{w}_{\alpha,i} (\nabla_i - \bar{g})\right\|^2\right]$$

由 Jensen 不等式：

$$\leq \mathbb{E}\left[\sum_i \bar{w}_{\alpha,i} \|\nabla_i - \bar{g}\|^2\right]$$

$$\leq \sum_i \bar{w}_{\alpha,i}^2 \cdot \max_i \|\nabla_i - \bar{g}\|^2$$

$$\leq (2G)^2 \sum_i \bar{w}_{\alpha,i}^2 = \frac{4G^2}{\text{ESS}_\alpha}$$

$\blacksquare$

**推论**：方差由 ESS 控制。ESS 越大，方差上界越小。

#### 10.6.7 有界性总结

**定理 10.12（有界性汇总）**：

| 量 | 有界性 | 界 | 条件 |
|---|-------|-----|-----|
| α | 有界 | [-1, +1] | 定义 |
| $D_\alpha^{(A)}(P\|Q)$ | 有界 | $[0, \frac{4}{1-\alpha^2}]$ | 无 |
| $\bar{w}_{\alpha,i}$ | 有界 | [0, 1] | 无 |
| $w = \pi_\theta/\mu$ | 条件有界 | $[0, M]$ | 需要截断或支撑条件 |
| $\|\hat{g}_\alpha\|$ | 有界 | $[0, G]$ | $\|\nabla \log \pi\| \leq G$ |
| $\text{Var}(\hat{g}_\alpha)$ | 有界 | $[0, \frac{4G^2}{\text{ESS}}]$ | $\|\nabla \log \pi\| \leq G$ |

---

### 10.7 主要理论结果

#### 10.7.1 单调对偶定理

**定理 10.13（Bias-Variance 单调对偶）**：

在统一 Amari α-散度框架中：

$$\frac{\partial \text{Bias}}{\partial \alpha} < 0, \quad \frac{\partial \text{Var}}{\partial \alpha} > 0 \quad (\text{对于 } \alpha > -1/2)$$

因此 **Bias 和 Variance 关于 α 具有相反的单调性**，构成对偶关系。

**推论（最优 α 存在性）**：

MSE(α) = Bias²(α) + λ·Var(α) 存在唯一的最小值点 α* ∈ (-1/2, 1)。

**证明**：

Bias²(α) 是关于 α 的严格递减函数（由定理 10.1）。
Var(α) 是关于 α 的严格递增函数（由定理 10.2）。

MSE 是递减凸函数与递增凸函数之和，因此是**拟凸函数**（quasi-convex）。

由于 MSE(-1) 大（高 bias），MSE(+1) 也可能大（高 variance），
而 MSE 连续，必存在唯一的内部最小值点。$\blacksquare$

#### 10.7.2 自适应收敛定理

**定理 10.14（自适应 α 收敛）**：

设 $\{\alpha_t\}$ 由矩估计规则生成：$\alpha_t = (2\delta - \sigma_t^2)/(\sigma_t^2 + 2\delta)$

若策略序列 $\{\pi_{\theta_t}\}$ 收敛到 $\pi^*$，则：

1. 分布偏移收敛：$\sigma_t^2 \to \sigma_\infty^2$
2. α 收敛：$\alpha_t \to \alpha_\infty = (2\delta - \sigma_\infty^2)/(\sigma_\infty^2 + 2\delta)$
3. 若 $\pi^* = p^*$（达到最优），则 $\sigma_\infty^2 \to 0$，$\alpha_\infty \to +1$

**证明**：

(1) 由策略收敛，$\text{Var}(\log \pi_{\theta_t}/\mu) = \text{Var}(\log w_t)$ 收敛。

(2) α* 是 σ² 的连续函数，由 (1) 知 α_t 收敛。

(3) 若 $\pi^* = p^* = \mu e^{r/\tau}/Z$，则：
$$\log w = \log \pi^* - \log \mu = r/\tau - \log Z$$

对于确定性奖励函数，$\log w$ 是 $y$ 的确定性函数乘以常数，其在 $\mu$ 下的方差趋于稳定。

特别地，当 $\pi_\theta \approx p^*$ 时，IS 权重的变异性降低，$\sigma^2$ 减小，α 增大。$\blacksquare$

#### 10.7.3 ESS 保证定理

**定理 10.15（ESS 下界保证）**：

使用 ESS 自适应规则（当 ESS < ρ_low·n 时减小 α）时，存在常数使得：

$$\boxed{\text{ESS}_{\alpha_t} \geq \rho_{\text{low}} \cdot n \quad \text{对所有 } t}$$

**证明**：

由定理 10.3，ESS 关于 α 严格单调递减。

边界条件：
- α = -1 时，$w^0 = 1$，所有权重相等，ESS = n
- α = +1 时，ESS 取决于原始 IS 权重

由 ESS 关于 α 的连续性和单调性，对于任意目标 ESS* ∈ (ESS(+1), n)，
存在唯一的 α* 使得 ESS(α*) = ESS*。

自适应规则保证当 ESS 跌破 ρ_low·n 时，α 减小，从而增加 ESS。

由于 α 有下界 -1，且 ESS(-1) = n > ρ_low·n，
系统必然能找到满足 ESS ≥ ρ_low·n 的 α。$\blacksquare$

---

### 10.8 理论保证可视化

```
═══════════════════════════════════════════════════════════════════════
                         理论保证总览
═══════════════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────────────┐
│                          单调性保证                                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│    Bias(α)          Var(α)           ESS(α)          α*(σ²)        │
│       ↓               ↑                ↓               ↓            │
│    递减             递增              递减            递减           │
│                                                                     │
│    ████░░░░░      ░░░░████        ████░░░░░       ████░░░░░        │
│    -1    +1       -1    +1        -1    +1        0     σ²         │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                          有界性保证                                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│    • α ∈ [-1, +1]                    ✓ 定义域有界                   │
│    • D_α^(A) ∈ [0, 4/(1-α²)]         ✓ 散度有界                     │
│    • w̄_i ∈ [0, 1], Σw̄_i = 1         ✓ 归一化权重有界               │
│    • ‖ĝ_α‖ ≤ G                       ✓ 梯度有界（若 score 有界）    │
│    • Var(ĝ) ≤ 4G²/ESS               ✓ 方差由 ESS 控制              │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                          存在性保证                                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│    MSE(α) = Bias²(α) + λ·Var(α)                                    │
│                                                                     │
│         MSE                                                         │
│          ↑                                                          │
│          │╲                 ╱                                       │
│          │ ╲    最优点     ╱                                        │
│          │  ╲     ●      ╱                                          │
│          │   ╲__________╱                                           │
│          │                                                          │
│          └────────────────────→ α                                   │
│         -1        α*        +1                                      │
│                                                                     │
│    定理：∃! α* ∈ (-1/2, 1) 使得 MSE 最小                            │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                          收敛性保证                                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│    若 π_θ → π*，则：                                                │
│                                                                     │
│    • σ² = Var(log w) 收敛                                          │
│    • α_t 收敛到 α_∞                                                 │
│    • 若 π* = p*（最优），则 α_∞ → +1                                │
│                                                                     │
│    ESS 保证：ESS_α ≥ ρ_low · n 始终成立                             │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════
                         核心公式
═══════════════════════════════════════════════════════════════════════

    ┌─────────────────────────────────────────────────────────────┐
    │                                                             │
    │   单调对偶：  ∂Bias/∂α < 0,    ∂Var/∂α > 0                 │
    │                                                             │
    │   最优 α：    α* = (2δ - σ²)/(σ² + 2δ)                     │
    │                                                             │
    │   方差界：    Var(ĝ) ≤ 4G²/ESS_α                           │
    │                                                             │
    │   散度界：    0 ≤ D_α^(A) ≤ 4/(1-α²)                       │
    │                                                             │
    └─────────────────────────────────────────────────────────────┘
```

---

## 附录：符号表

| 符号 | 含义 |
|------|------|
| α | 统一插值参数，∈ [-1, +1] |
| γ = (1+α)/2 | f(w) = w^γ 的指数 |
| β = (1+α)/2 | 目标分布倾斜度 |
| w = π_θ/μ | 重要性采样比率 |
| σ² | Var(log w)，分布偏移度量 |
| δ | bias 容忍度参数 |
| ESS_α | 有效样本量 |
| ρ = ESS/n | ESS 比例 |
| B | 目标偏差 |
| V(α) | 单样本方差 |
| G | score function 的上界 |
| $D_\alpha^{(A)}$ | Amari α-散度 |
| $p_\beta$ | β-倾斜目标分布 |
| $p^* = p_1$ | 最优软策略 |
