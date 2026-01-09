# IS-Reshape v10: 调和加权框架 (Harmonic Importance Sampling)

**版本**: 10.0

---

## 摘要

本文从 MSE 最小化的角度重新审视 off-policy 策略梯度估计问题，推导出一种**调和加权 (Harmonic Weighting)** 形式的梯度权重函数。

**核心贡献**：

1. **MSE 最小化框架**：统一偏差-方差权衡，推导最优混合策略
2. **调和加权公式**：$\phi(w) = \frac{w}{w + \beta}$，自然实现 SFT-RL 插值
3. **闭式目标函数**：$f(w) = \frac{1}{\beta} \ln(\beta w + 1)$，对应对数效用
4. **JSD 联系**：调和权重 = GAN 判别器形式，对应 Jensen-Shannon Divergence
5. **风险厌恶解释**：对数效用 = 典型的风险厌恶效用函数
6. **四象限自适应**：无需显式区分 A 的符号，单一公式覆盖所有情况

**核心公式**：

$$\phi(w, A) = \frac{w}{w + \beta} \cdot A$$

其中 $w = \pi_\theta / \mu$ 是重要性权重，$\beta > 0$ 是混合参数。

---

# 第一部分：问题设定

## 1. Off-Policy 策略梯度估计

### 1.1 基本问题

**目标**：优化策略 $\pi_\theta$ 以最大化期望回报
$$\max_\theta J(\theta) = \mathbb{E}_{\pi_\theta}[A(s,a)]$$

**约束**：只能从行为策略 $\mu$ 采样

**真实梯度**：
$$g^* = \nabla_\theta \mathbb{E}_{\pi_\theta}[A] = \mathbb{E}_{\pi_\theta}[A \cdot \nabla_\theta \log \pi_\theta]$$

### 1.2 两种极端方案

**方案 A：标准重要性采样 (RL)**
$$\hat{g}_{IS} = \mathbb{E}_\mu\left[\frac{\pi_\theta}{\mu} \cdot A \cdot \nabla_\theta \log \pi_\theta\right] = \mathbb{E}_\mu[w \cdot A \cdot \nabla \log \pi]$$

- **偏差**：0（无偏）
- **方差**：$\propto \mathbb{E}_\mu[w^2]$，可能很大

**方案 B：忽略分布偏移 (SFT-style)**
$$\hat{g}_{SFT} = \mathbb{E}_\mu[A \cdot \nabla_\theta \log \pi_\theta]$$

- **偏差**：$\mathbb{E}_\mu[(1-w) \cdot A \cdot \nabla \log \pi] \neq 0$
- **方差**：较小

### 1.3 一般梯度估计器

定义一般形式：
$$\hat{g} = \mathbb{E}_\mu[\phi(w, A) \cdot \nabla_\theta \log \pi_\theta]$$

其中 $\phi: \mathbb{R}^+ \times \mathbb{R} \to \mathbb{R}$ 是**梯度权重函数**。

**问题**：如何设计最优的 $\phi(w, A)$？

---

## 2. MSE 最小化框架

### 2.1 MSE 分解

均方误差 = 偏差² + 方差：
$$\text{MSE}[\hat{g}] = \|\text{Bias}[\hat{g}]\|^2 + \text{Var}[\hat{g}]$$

**偏差**：
$$\text{Bias}[\phi] = \mathbb{E}_\mu[(\phi(w, A) - w \cdot A) \cdot \nabla \log \pi]$$

**方差**：
$$\text{Var}[\phi] \propto \mathbb{E}_\mu[\phi(w, A)^2]$$

### 2.2 偏差-方差权衡

$$\min_\phi \text{MSE} = \min_\phi \left\{ \lambda \cdot \|\text{Bias}[\phi]\|^2 + \text{Var}[\phi] \right\}$$

- $\lambda \to \infty$：偏好低偏差 → $\phi = wA$（标准 IS）
- $\lambda \to 0$：偏好低方差 → $\phi = 0$（无学习）

### 2.3 关键洞察：从混合分布出发

考虑**混合采样分布**：
$$\rho = \alpha \pi + (1-\alpha) \mu, \quad \alpha \in [0, 1]$$

从 $\rho$ 采样时，IS 权重为：
$$w_\rho = \frac{\pi}{\rho} = \frac{\pi}{\alpha \pi + (1-\alpha) \mu}$$

**关键观察**：当我们只有 $\mu$ 的样本但想估计 $\rho$ 下的期望时...

---

# 第二部分：调和加权的推导

## 3. Defensive Importance Sampling

### 3.1 混合分布思想

**Defensive IS (DIS)** 的核心思想：不直接用 $\mu$ 估计 $\pi$ 下的期望，而是用 $\mu$ 估计混合分布 $\rho$ 下的期望。

设 $\rho = \alpha \pi + (1-\alpha) \mu$，则：
$$\mathbb{E}_\pi[f] \approx \mathbb{E}_\rho[f] = \mathbb{E}_\mu\left[\frac{\rho}{\mu} \cdot f\right]$$

### 3.2 关键推导

从 $\mu$ 估计 $\rho$ 下期望：
$$\frac{\rho(a)}{\mu(a)} = \frac{\alpha \pi(a) + (1-\alpha) \mu(a)}{\mu(a)} = \alpha w + (1-\alpha)$$

其中 $w = \pi/\mu$。

**权重变换**：
$$w_{\text{DIS}} = \alpha w + (1 - \alpha) = 1 + \alpha(w - 1)$$

### 3.3 从 $\rho$ 视角的 IS 权重

如果我们想从 $\mu$ 估计 $\pi$ 下的期望，但通过 $\rho$ 作为中介：

$$\mathbb{E}_\pi[f] = \mathbb{E}_\rho\left[\frac{\pi}{\rho} \cdot f\right] = \mathbb{E}_\mu\left[\frac{\rho}{\mu} \cdot \frac{\pi}{\rho} \cdot f\right]$$

其中：
$$\frac{\pi}{\rho} = \frac{\pi}{\alpha \pi + (1-\alpha)\mu} = \frac{w}{\alpha w + (1-\alpha)}$$

### 3.4 调和形式的涌现

设 $\alpha = \frac{\beta}{1+\beta}$，则 $1 - \alpha = \frac{1}{1+\beta}$：

$$\frac{\pi}{\rho} = \frac{w}{\frac{\beta}{1+\beta} w + \frac{1}{1+\beta}} = \frac{w(1+\beta)}{\beta w + 1} = \frac{(1+\beta) w}{\beta w + 1}$$

归一化使得 $w=1$ 时权重为 1：
$$\boxed{\phi(w) = \frac{w}{\beta w + 1} \cdot (1 + \beta) = \frac{(1+\beta)w}{\beta w + 1}}$$

或者更简洁的形式（归一化常数吸收到 learning rate）：
$$\boxed{\phi(w) = \frac{w}{w + \beta}}$$

---

## 4. 调和加权的性质

### 4.1 基本性质

**定理 4.1**：调和权重函数 $\phi(w) = \frac{w}{w + \beta}$ 满足：

1. **有界性**：$\phi(w) \in (0, 1)$ 对所有 $w > 0$
2. **单调增**：$\phi'(w) = \frac{\beta}{(w+\beta)^2} > 0$
3. **凹性**：$\phi''(w) = -\frac{2\beta}{(w+\beta)^3} < 0$
4. **归一化**：$\phi(1) = \frac{1}{1+\beta}$

### 4.2 极限行为

**小 w 极限** ($w \to 0$)：
$$\phi(w) = \frac{w}{w + \beta} \approx \frac{w}{\beta} \propto w$$

→ **线性放大**：新样本（$w \ll 1$）得到与 IS 成比例的权重

**大 w 极限** ($w \to \infty$)：
$$\phi(w) = \frac{w}{w + \beta} \approx 1 - \frac{\beta}{w} \to 1$$

→ **饱和**：已知样本（$w \gg 1$）权重饱和，防止方差爆炸

### 4.3 与 GAN 判别器的联系

**定理 4.2**：调和权重 $\phi(w) = \frac{w}{w + \beta}$ 与 GAN 判别器同构：

$$D^*(x) = \frac{p_{data}(x)}{p_{data}(x) + p_{model}(x)} = \frac{1}{1 + 1/w}$$

当 $\beta = 1$ 时：
$$\phi(w) = \frac{w}{w + 1} = \frac{1}{1 + 1/w} = D^*(x)$$

**解释**：调和权重是"$\pi$ vs $\mu$"分类问题的最优判别器概率！

---

## 5. 目标函数推导

### 5.1 从梯度权重反推目标函数

**问题**：是否存在标量目标函数 $f(w)$ 使得 $\phi(w) \propto \frac{\partial f}{\partial w} \cdot w$？

**定理 5.1**：对于 $\phi(w) = \frac{w}{w + \beta}$，对应的目标函数为：

$$\boxed{f(w) = \frac{1}{\beta} \ln(\beta w + 1)}$$

**证明**：

设 $\phi(w) = \frac{\partial f}{\partial w} \cdot w$，则：
$$\frac{\partial f}{\partial w} = \frac{\phi(w)}{w} = \frac{1}{w + \beta}$$

积分：
$$f(w) = \int \frac{1}{w + \beta} dw = \ln(w + \beta) + C$$

取 $C = -\ln(\beta)$ 使得 $f(0) = 0$：
$$f(w) = \ln(w + \beta) - \ln(\beta) = \ln\left(\frac{w + \beta}{\beta}\right) = \ln\left(1 + \frac{w}{\beta}\right)$$

或等价地：
$$f(w) = \frac{1}{\beta} \ln(\beta w + 1)$$

$\blacksquare$

### 5.2 与对数效用的联系

**定理 5.2**：目标函数 $f(w) = \frac{1}{\beta} \ln(\beta w + 1)$ 是典型的**风险厌恶效用函数**。

**对数效用**的标准形式：$u(x) = \ln(x + c)$

**性质**：
- **单调增**：$u'(x) > 0$（更多财富更好）
- **凹性**：$u''(x) < 0$（风险厌恶）
- **边际效用递减**：$u'(x)$ 随 $x$ 增加而减小

**经济学解释**：
- $w$ 是"财富"（信息量）
- $f(w)$ 是"效用"
- 高 $w$ 时效用增长变慢 → 风险厌恶 → 方差控制

### 5.3 与 Jensen-Shannon Divergence 的联系

**定理 5.3**：调和加权对应 JSD 的梯度结构。

设 $M = \frac{1}{2}(\pi + \mu)$（中点分布），则 JSD 定义为：
$$\text{JSD}(\pi \| \mu) = \frac{1}{2} KL(\pi \| M) + \frac{1}{2} KL(\mu \| M)$$

其梯度结构涉及：
$$\frac{\pi}{M} = \frac{\pi}{\frac{1}{2}(\pi + \mu)} = \frac{2w}{w + 1}$$

当 $\beta = 1$ 时，$\phi(w) = \frac{w}{w+1}$ 正是 JSD 梯度的核心项！

---

# 第三部分：四象限分析

## 6. 统一的四象限行为

### 6.1 四种情况分析

考虑梯度估计器 $\hat{g} = \mathbb{E}_\mu[\phi(w) \cdot A \cdot \nabla \log \pi]$：

| 象限 | w | A | φ(w)·A | 行为 |
|-----|---|---|--------|------|
| I | $\ll 1$ | $> 0$ | $\approx (w/\beta) \cdot A$ | **IS 纠偏**：新好样本，线性放大 |
| II | $\gg 1$ | $> 0$ | $\approx A$ | **SFT 稳定**：已知好样本，饱和防爆 |
| III | $\gg 1$ | $< 0$ | $\approx A$ | **稳定惩罚**：未避免坏样本，稳定梯度 |
| IV | $\ll 1$ | $< 0$ | $\approx (w/\beta) \cdot A \to 0$ | **自然消失**：已避免坏样本，梯度趋零 |

### 6.2 可视化

```
                        w >> 1
                          │
              II          │          III
        A > 0, w >> 1     │     A < 0, w >> 1
        φ ≈ 1 (SFT)       │     φ ≈ 1 (稳定惩罚)
        "已知好样本"       │     "未避免坏样本"
                          │
    ──────────────────────┼──────────────────────
                          │
        A > 0, w << 1     │     A < 0, w << 1
        φ ≈ w/β (IS)      │     φ ≈ w/β → 0
        "新好样本"         │     "已避免坏样本"
              I           │          IV
                          │
                        w << 1
```

### 6.3 与显式分组方法的对比

**v7/v9 的做法**：显式根据 A 的符号选择不同的 $\gamma$ 或截断方向

**v10 调和方法**：**单一公式**，四象限行为自动涌现！

| 方法 | A > 0 | A < 0 | 公式 |
|------|-------|-------|------|
| v7 (Box-Cox) | $w^{\gamma_+}$, $\gamma_+ < 1$ | $w^{\gamma_-}$, $\gamma_- > 1$ | 两个公式 |
| v9 (IB) | 上界 Softplus 截断 | 下界 Softplus 截断 | 两个公式 |
| **v10 (Harmonic)** | $\frac{w}{w+\beta}$ | $\frac{w}{w+\beta}$ | **一个公式** |

---

## 7. 自适应 β

### 7.1 β 的含义

$\beta$ 控制 SFT-RL 的混合程度：

| β 值 | 行为 | 含义 |
|------|------|------|
| $\beta \to 0$ | $\phi(w) \to 1$ | 纯 SFT（忽略分布偏移）|
| $\beta = 1$ | $\phi(w) = \frac{w}{w+1}$ | 均衡混合（JSD 形式）|
| $\beta \to \infty$ | $\phi(w) \to w/\beta \propto w$ | 纯 IS（完全纠偏）|

### 7.2 基于 KL 的自适应

可以根据分布偏移程度自适应调整 $\beta$：

$$\beta = \beta_0 \cdot \exp\left(\frac{KL(\pi \| \mu)}{\tau}\right)$$

- KL 大（分布差异大）→ $\beta$ 大 → 更多 IS 纠偏
- KL 小（分布接近）→ $\beta$ 小 → 更多 SFT 稳定

### 7.3 训练阶段调度

**早期训练**（分布差异大）：
$$\beta_{\text{early}} \sim 0.1 - 0.5$$

**后期训练**（分布接近）：
$$\beta_{\text{late}} \sim 1.0 - 5.0$$

---

# 第四部分：与现有方法的联系

## 8. 方法对比

### 8.1 统一视角

所有方法都可以看作不同的梯度权重函数 $\phi(w)$：

| 方法 | $\phi(w)$ | 有界？ | 凹？ | 单调？ | 闭式 f(w)？ |
|------|-----------|--------|------|--------|-------------|
| IS (RL) | $w$ | ❌ | ❌ | ✓ | $\frac{1}{2}w^2$ |
| SFT | $1$ | ✓ | - | - | $w$ |
| PPO clip | $\text{clip}(w, 1-\epsilon, 1+\epsilon)$ | ✓ | ❌ | ✓ | 分段 |
| IS-Reshape ($\gamma$) | $w^\gamma$ | ❌ | $\gamma<1$ | ✓ | $\frac{w^\gamma}{\gamma}$ |
| SAPO | $\sigma(\tau(w-1)) \cdot \frac{4}{\tau}$ | ✓ | 近似 | ❌ | 无标准形式 |
| IB-IS (v9) | Softplus 截断 | ✓ | 近似 | ✓ | 复杂 |
| **Harmonic (v10)** | $\frac{w}{w+\beta}$ | ✓ | ✓ | ✓ | $\frac{1}{\beta}\ln(\beta w + 1)$ |

### 8.2 v10 的独特优势

1. **数学优美**：单一简洁公式
2. **有界且凹且单调**：同时满足三个关键性质
3. **闭式目标函数**：有明确的优化目标
4. **多重理论联系**：JSD、对数效用、风险厌恶、GAN 判别器
5. **无需显式分组**：A > 0 和 A < 0 统一处理

### 8.3 与 SAPO 的比较

**SAPO**：
$$\phi_{\text{SAPO}}(w) = \sigma(\tau(w-1)) \cdot \frac{4}{\tau}$$

**Harmonic**：
$$\phi_{\text{Harmonic}}(w) = \frac{w}{w + \beta}$$

| 性质 | SAPO | Harmonic |
|------|------|----------|
| 形式 | Sigmoid | 有理函数 |
| 单调性 | ✓ | ✓ |
| 凹性 | 仅部分区域 | 全局 |
| 理论基础 | 工程设计 | MSE 最优/JSD |
| 参数 | $\tau$ | $\beta$ |

---

# 第五部分：实现

## 9. 算法实现

### 9.1 核心代码

```python
import torch
import torch.nn.functional as F
from typing import Tuple, Dict, Optional

class HarmonicIS:
    """
    调和重要性采样 (Harmonic Importance Sampling)

    理论基础：
    - MSE 最小化 → Defensive IS → 调和加权
    - 目标函数：f(w) = (1/β) ln(βw + 1)
    - JSD 联系：φ(w) = w/(w+1) 对应 GAN 判别器

    核心公式：
        φ(w) = w / (w + β)

    性质：
    - 有界：φ ∈ (0, 1)
    - 凹：φ'' < 0
    - 单调增：φ' > 0
    """

    def __init__(
        self,
        beta: float = 1.0,
        normalize: bool = True,
    ):
        """
        Args:
            beta: 混合参数，控制 SFT-RL 权衡
                  β → 0: 更像 SFT
                  β → ∞: 更像 IS
            normalize: 是否归一化使得 φ(1) = 0.5 (便于比较)
        """
        self.beta = beta
        self.normalize = normalize

    def compute_phi(self, w: torch.Tensor) -> torch.Tensor:
        """
        计算调和权重 φ(w) = w / (w + β)
        """
        phi = w / (w + self.beta)

        if self.normalize:
            # 归一化使得 φ(1) 对齐到 0.5
            # 原始 φ(1) = 1 / (1 + β)
            # 乘以 (1 + β) / 2 使得 φ(1) = 0.5
            phi = phi * (1 + self.beta) / 2

        return phi

    def compute_f(self, w: torch.Tensor) -> torch.Tensor:
        """
        计算目标函数 f(w) = (1/β) ln(βw + 1)
        """
        return torch.log(self.beta * w + 1) / self.beta

    def __call__(
        self,
        log_pi: torch.Tensor,
        log_mu: torch.Tensor,
        advantages: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        计算调和 IS 策略梯度损失

        Args:
            log_pi: 当前策略的 log 概率
            log_mu: 行为策略的 log 概率
            advantages: 优势值
            mask: 可选的掩码

        Returns:
            loss: 策略梯度损失
            metrics: 诊断指标
        """
        # 1. 计算 log IS ratio
        log_w = log_pi - log_mu
        log_w = torch.clamp(log_w, -20.0, 20.0)  # 数值稳定性
        w = torch.exp(log_w)

        # 2. 计算调和权重
        phi = self.compute_phi(w)

        # 3. 计算损失
        pg_obj = phi * advantages

        if mask is not None:
            loss = -(pg_obj * mask).sum() / (mask.sum() + 1e-8)
        else:
            loss = -pg_obj.mean()

        # 4. 诊断指标
        with torch.no_grad():
            valid_mask = mask > 0 if mask is not None else torch.ones_like(w, dtype=torch.bool)

            metrics = {
                "harmonic/w_mean": w[valid_mask].mean().item() if valid_mask.any() else 1.0,
                "harmonic/w_max": w[valid_mask].max().item() if valid_mask.any() else 1.0,
                "harmonic/w_min": w[valid_mask].min().item() if valid_mask.any() else 1.0,
                "harmonic/phi_mean": phi[valid_mask].mean().item() if valid_mask.any() else 0.5,
                "harmonic/phi_max": phi[valid_mask].max().item() if valid_mask.any() else 1.0,
                "harmonic/phi_min": phi[valid_mask].min().item() if valid_mask.any() else 0.0,
                "harmonic/beta": self.beta,
            }

            # 分组统计
            pos_mask = (advantages > 0) & valid_mask
            neg_mask = (advantages < 0) & valid_mask

            if pos_mask.any():
                metrics["harmonic/phi_pos_mean"] = phi[pos_mask].mean().item()
                metrics["harmonic/w_pos_mean"] = w[pos_mask].mean().item()
                metrics["harmonic/n_pos"] = pos_mask.sum().item()

            if neg_mask.any():
                metrics["harmonic/phi_neg_mean"] = phi[neg_mask].mean().item()
                metrics["harmonic/w_neg_mean"] = w[neg_mask].mean().item()
                metrics["harmonic/n_neg"] = neg_mask.sum().item()

            # KL 散度
            kl = -log_w[valid_mask].mean().item() if valid_mask.any() else 0.0
            metrics["harmonic/kl"] = kl

        return loss, metrics
```

### 9.2 注册到 veRL

```python
# 在 core_algos.py 中添加
from verl.trainer.ppo.core_algos import register_policy_loss, agg_loss
import verl.utils.torch_functional as verl_F

@register_policy_loss("harmonic_is")
def compute_policy_loss_harmonic_is(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, Dict]:
    """
    调和重要性采样策略损失 (Harmonic IS)

    核心公式：
        φ(w) = w / (w + β)
        L = -E[φ(w) · A]

    Config example:
        actor:
          policy_loss:
            loss_mode: "harmonic_is"
            harmonic_is:
              beta: 1.0        # 混合参数 (β → 0: SFT, β → ∞: IS)
              normalize: true  # 是否归一化
    """
    # 提取配置
    harmonic_config = config.policy_loss.get("harmonic_is", {}) if config else {}
    beta = harmonic_config.get("beta", 1.0)
    normalize = harmonic_config.get("normalize", True)

    # 确保 β > 0
    beta = max(0.01, beta)

    # 计算 log IS ratio
    log_w = log_prob - old_log_prob
    log_w = torch.clamp(log_w, -20.0, 20.0)
    w = torch.exp(log_w)

    # 计算调和权重 φ(w) = w / (w + β)
    phi = w / (w + beta)

    if normalize:
        # 归一化使得 φ(1) = 0.5
        phi = phi * (1 + beta) / 2

    # 计算损失
    loss_mat = -phi * advantages
    pg_loss = agg_loss(loss_mat, response_mask, loss_agg_mode)

    # 诊断指标
    with torch.no_grad():
        mask = response_mask > 0

        ppo_kl = verl_F.masked_mean(-log_w, response_mask)

        phi_mean = verl_F.masked_mean(phi, response_mask)
        w_mean = verl_F.masked_mean(w, response_mask)

        phi_flat = phi[mask]
        w_flat = w[mask]

        if phi_flat.numel() > 0:
            phi_max = phi_flat.max().item()
            phi_min = phi_flat.min().item()
            w_max = w_flat.max().item()
            w_min = w_flat.min().item()
        else:
            phi_max = phi_min = 0.5
            w_max = w_min = 1.0

        # 分组统计
        pos_mask = (advantages > 0) & mask
        neg_mask = (advantages < 0) & mask

        n_pos = pos_mask.sum().item()
        n_neg = neg_mask.sum().item()

        if pos_mask.any():
            phi_pos_mean = phi[pos_mask].mean().item()
            w_pos_mean = w[pos_mask].mean().item()
        else:
            phi_pos_mean = 0.5
            w_pos_mean = 1.0

        if neg_mask.any():
            phi_neg_mean = phi[neg_mask].mean().item()
            w_neg_mean = w[neg_mask].mean().item()
        else:
            phi_neg_mean = 0.5
            w_neg_mean = 1.0

    # 收集指标
    pg_metrics = {
        "actor/ppo_kl": ppo_kl.item(),
        "actor/w_mean": w_mean.item(),
        "actor/w_max": w_max,
        "actor/w_min": w_min,
        "harmonic_is/phi_mean": phi_mean.item(),
        "harmonic_is/phi_max": phi_max,
        "harmonic_is/phi_min": phi_min,
        "harmonic_is/phi_pos_mean": phi_pos_mean,
        "harmonic_is/phi_neg_mean": phi_neg_mean,
        "harmonic_is/w_pos_mean": w_pos_mean,
        "harmonic_is/w_neg_mean": w_neg_mean,
        "harmonic_is/beta": beta,
        "harmonic_is/n_pos": n_pos,
        "harmonic_is/n_neg": n_neg,
        # 兼容性
        "actor/pg_clipfrac": 0.0,
        "actor/pg_clipfrac_lower": 0.0,
    }

    return pg_loss, pg_metrics
```

### 9.3 配置示例

```yaml
# config/harmonic_is_example.yaml
actor_rollout_ref:
  actor:
    policy_loss:
      loss_mode: "harmonic_is"
      harmonic_is:
        beta: 1.0          # 混合参数
                           # β = 0.1: 偏 SFT (保守)
                           # β = 1.0: 均衡 (JSD 形式，推荐)
                           # β = 5.0: 偏 IS (激进)
        normalize: true    # 归一化 φ(1) = 0.5
```

---

## 10. 超参数指南

### 10.1 β 的选择

| β 值 | 行为 | 适用场景 |
|------|------|----------|
| 0.1 - 0.3 | 偏 SFT，保守 | 早期训练、分布差异大 |
| 0.5 - 1.0 | 均衡 | 默认推荐 |
| 2.0 - 5.0 | 偏 IS，激进 | 后期训练、分布接近 |

### 10.2 与训练阶段的关系

```
训练进度  ────────────────────────────────────→
          |                                   |
          |  分布差异大        分布差异小      |
          |  使用小 β         使用大 β        |
          |  (更保守)         (更激进)        |
          |                                   |
β 值      0.2              1.0              3.0
```

### 10.3 动态调整策略

可以根据 KL 散度动态调整 β：

```python
def adaptive_beta(kl_divergence, beta_min=0.1, beta_max=5.0, kl_target=0.1):
    """
    基于 KL 散度自适应调整 β

    KL 大 → β 小 → 更保守
    KL 小 → β 大 → 更激进
    """
    ratio = kl_divergence / kl_target
    # 使用 sigmoid 映射到 [beta_min, beta_max]
    t = 1 / (1 + ratio)  # KL 大 → t 小
    beta = beta_min + (beta_max - beta_min) * t
    return beta
```

---

# 第六部分：总结

## 11. 核心贡献总结

```
┌─────────────────────────────────────────────────────────────────────────┐
│                   IS-Reshape v10: 调和加权框架                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  【核心公式】                                                             │
│                                                                         │
│      φ(w) = w / (w + β)                                                │
│                                                                         │
│      f(w) = (1/β) ln(βw + 1)                                           │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  【理论贡献】                                                             │
│                                                                         │
│  1. MSE 最小化框架：统一偏差-方差权衡，推导最优混合策略                     │
│                                                                         │
│  2. 多重理论联系：                                                        │
│     - Defensive IS → 调和加权                                           │
│     - JSD → φ(w) = w/(w+1) = GAN 判别器                                 │
│     - 对数效用 → 风险厌恶                                                │
│                                                                         │
│  3. 四象限自适应：单一公式，行为自动分化                                   │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  【性质优势】                                                             │
│                                                                         │
│  ┌─────────────┬──────────────────────────────────────────────────┐    │
│  │   性质       │              调和加权 φ(w) = w/(w+β)             │    │
│  ├─────────────┼──────────────────────────────────────────────────┤    │
│  │ 有界性      │ ✓  φ ∈ (0, 1)，防止梯度爆炸                       │    │
│  │ 凹性        │ ✓  φ'' < 0，自动方差控制                          │    │
│  │ 单调性      │ ✓  φ' > 0，保持 IS 基本原则                       │    │
│  │ 闭式目标    │ ✓  f(w) = (1/β)ln(βw+1)                          │    │
│  │ 理论基础    │ ✓  MSE 最优、JSD、风险厌恶                        │    │
│  │ 简洁性      │ ✓  单一公式，一个超参数                           │    │
│  └─────────────┴──────────────────────────────────────────────────┘    │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  【四象限行为】                                                           │
│                                                                         │
│    A > 0, w << 1: φ ≈ w/β (IS 纠偏) │ A > 0, w >> 1: φ ≈ 1 (SFT 稳定)  │
│    ──────────────────────────────────┼──────────────────────────────────│
│    A < 0, w << 1: φ → 0 (自然消失)   │ A < 0, w >> 1: φ ≈ 1 (稳定惩罚)  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## 12. 与历史版本的关系

```
v1-v6: 理论探索期
  │
  ├─ v7: Box-Cox 形式，显式 γ+/γ- 分组
  │
  ├─ v8: α-Divergence 视角，梯度权重 φ 设计
  │
  ├─ v9: Information Bottleneck，Softplus 截断
  │
  └─ v10: 调和加权 (Harmonic IS)
          - 统一 MSE 最小化框架
          - 单一公式覆盖所有情况
          - 最优的理论-工程平衡
```

---

## 附录 A：关键公式汇总

**梯度权重**：
$$\phi(w) = \frac{w}{w + \beta}$$

**目标函数**：
$$f(w) = \frac{1}{\beta} \ln(\beta w + 1)$$

**梯度**：
$$\frac{\partial f}{\partial w} = \frac{1}{w + \beta}$$

**二阶导**（验证凹性）：
$$\frac{\partial^2 f}{\partial w^2} = -\frac{1}{(w + \beta)^2} < 0$$

**极限行为**：
$$\lim_{w \to 0} \phi(w) = 0, \quad \lim_{w \to \infty} \phi(w) = 1$$

**JSD 联系**（β = 1）：
$$\phi(w) = \frac{w}{w + 1} = D^*(x) = P(\text{sample from } \pi | x)$$

---

## 附录 B：参考文献

1. Owen, A. & Zhou, Y. (2000). Safe and effective importance sampling.
2. Hesterberg, T. (1995). Weighted average importance sampling.
3. Goodfellow, I. et al. (2014). Generative adversarial nets. (JSD 联系)
4. Schulman, J. et al. (2017). Proximal policy optimization algorithms.
5. SAPO paper: arXiv:2511.20347
6. IS-Reshape theory: v1-v9 documents in this repository.
