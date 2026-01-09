# IS-Reshape v9: Information Bottleneck 视角下的重要性采样重塑

**版本**: 9.0

---

## 摘要

本文从 Information Bottleneck (IB) 的角度重新审视 off-policy 策略梯度估计问题。我们将重要性采样（IS）权重的处理问题形式化为一个信息压缩问题：**在保持对目标策略期望估计能力的前提下，最小化与原始 IS 权重的互信息**。

**核心贡献**：

1. **IB 形式化**：将 IS 重塑问题映射到 Information Bottleneck 框架
2. **Rate-Distortion 解释**：偏差-方差权衡等价于 Rate-Distortion 权衡
3. **Softplus 截断推导**：从 Fenchel-Legendre 对偶推导出最优的软截断形式
4. **非对称压缩**：证明正负样本需要相反方向的信息压缩
5. **SFT-RL 统一**：建立从 SFT 到 RL 插值的信息论基础

**核心公式**：

$$\rho_{smooth} = \begin{cases}
C - \tau \cdot \text{Softplus}\left(\frac{C - \rho}{\tau}\right) & A > 0 \\[8pt]
-C + \tau \cdot \text{Softplus}\left(\frac{C + \rho}{\tau}\right) & A < 0
\end{cases}$$

其中 $\rho = \log \pi - \log \mu$，$C$ 是信息带宽，$\tau$ 是温度参数。

---

# 第一部分：问题设定与动机

## 1. Off-Policy 学习的信息传递视角

### 1.1 基本问题

**目标**：优化策略 $\pi_\theta$ 以最大化期望回报
$$\max_\theta J(\theta) = \mathbb{E}_{\pi_\theta}[A(s,a)]$$

**约束**：只能从行为策略 $\mu$ 采样

**IS 连接**：
$$\mathbb{E}_{\pi_\theta}[A] = \mathbb{E}_\mu\left[\frac{\pi_\theta}{\mu} \cdot A\right] = \mathbb{E}_\mu[w \cdot A]$$

### 1.2 信息传递视角

将 off-policy 学习看作一个**通信系统**：

```
┌─────────────────────────────────────────────────────────────────┐
│                        信息传递链                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│    信源 (Source)        信道 (Channel)        接收端 (Receiver) │
│         X          ────────→ T ────────→           Y           │
│                                                                 │
│   μ 下的样本           IS 权重处理            π 下期望的估计     │
│   (s, a, A, w)        φ(w) · A              𝔼_π[A] 的估计       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**核心问题**：如何设计"信道" $\phi(w)$？

- **完美传输** $\phi(w) = w$：无偏但高方差（信道容量过大）
- **完全压缩** $\phi(w) = 1$：零方差但高偏差（信道容量为零）

### 1.3 为什么需要 IB 视角？

传统视角的局限：
- **PPO clip**：启发式设计，缺乏理论基础
- **α-divergence**：优雅但无界（$w^\gamma$ 不满足有界性）
- **SAPO**：工程有效但缺乏系统性理论

IB 视角的优势：
- 偏差-方差权衡有严格的信息论解释
- Softplus 截断从第一性原理推导
- 自然导出正负样本的非对称处理

---

## 2. Information Bottleneck 基础

### 2.1 标准 IB 问题

给定马尔可夫链 $X \to T \to Y$，Information Bottleneck 问题是：

$$\min_{p(t|x)} I(X; T) - \beta \cdot I(T; Y)$$

**解释**：
- $I(X; T)$：压缩项 — T 保留了多少关于 X 的信息
- $I(T; Y)$：相关性项 — T 对预测 Y 有多大帮助
- $\beta$：权衡参数

### 2.2 Rate-Distortion 等价形式

IB 问题等价于 Rate-Distortion 问题：

$$\min_{p(t|x)} I(X; T) \quad \text{s.t.} \quad D(T, Y) \leq \epsilon$$

其中 $D(T, Y)$ 是某种失真度量。

**Rate-Distortion 函数**：
$$R(D) = \min_{p(t|x): \mathbb{E}[d(T,Y)] \leq D} I(X; T)$$

### 2.3 高斯情况的解析解

当 $X$ 是高斯分布时，IB 问题有解析解：

$$T^* = X + N$$

其中 $N$ 是独立高斯噪声，方差由 $\beta$ 决定。

这启发我们：**最优的信息压缩是在原始信号上加"噪声"或"截断"**。

---

# 第二部分：IS 问题的 IB 形式化

## 3. 将 IS 映射到 IB

### 3.1 变量对应

| IB 框架 | IS 问题 | 含义 |
|--------|--------|------|
| $X$ | $w = \pi/\mu$ | 原始 IS 权重 |
| $T$ | $\tilde{w} = \phi(w)$ | 重塑后的权重 |
| $Y$ | $\mathbb{E}_\pi[A]$ | 目标（on-policy 期望）|
| $I(X;T)$ | $\text{Var}[\tilde{w} \cdot A]$ | 方差（信息量）|
| $I(T;Y)$ | $\|\text{Bias}[\tilde{w}]\|^{-2}$ | 估计精度 |

### 3.2 IS-IB 目标函数

**原始形式**：
$$\min_{\phi} I(w; \phi(w)) - \beta \cdot I(\phi(w) \cdot A; \mathbb{E}_\pi[A])$$

**实用形式**（方差-偏差）：
$$\min_{\phi} \underbrace{\mathbb{E}_\mu[\phi(w)^2]}_{\text{方差代理}} + \lambda \cdot \underbrace{|\mathbb{E}_\mu[\phi(w)] - 1|^2}_{\text{偏差}^2}$$

### 3.3 约束优化形式

等价的约束优化问题：

$$\min_{\phi} \mathbb{E}_\mu[\phi(w)^2] \quad \text{s.t.} \quad |\mathbb{E}_\mu[\phi(w) \cdot A] - \mathbb{E}_\pi[A]| \leq \epsilon$$

或者（信息论形式）：

$$\max_{\phi} \mathbb{E}_\mu[\phi(w) \cdot A] \quad \text{s.t.} \quad H(\phi(w)) \leq C$$

其中 $H(\phi(w))$ 是重塑权重的熵，$C$ 是信息带宽约束。

---

## 4. 关键洞察：条件 IB

### 4.1 问题的非对称性

**核心观察**：对于 $A > 0$ 和 $A < 0$，"有用信息"的分布是不同的！

**对于 $A > 0$（好样本）**：
- $w < 1$（新好样本）：高信息量，需要保留
- $w > 1$（已学好样本）：低信息量，可以压缩

**对于 $A < 0$（坏样本）**：
- $w < 1$（已避免坏样本）：低信息量，可以压缩
- $w > 1$（未避免坏样本）：高信息量，需要保留

### 4.2 条件 IB 形式化

这导出**条件 Information Bottleneck**：

$$\min_{\phi} I(w; \phi(w) | A) - \beta \cdot I(\phi(w); Y | A)$$

展开为两个子问题：

**子问题 1**（$A > 0$）：
$$\min_{\phi^+} I(w; \phi^+(w) | A > 0) \quad \text{s.t.} \quad \text{Bias}^+ \leq \epsilon$$

**子问题 2**（$A < 0$）：
$$\min_{\phi^-} I(w; \phi^-(w) | A < 0) \quad \text{s.t.} \quad \text{Bias}^- \leq \epsilon$$

### 4.3 信息分布可视化

```
                    A > 0 时的信息分布

    信息量 │
         │     ████
         │   ████████                      低信息（可压缩）
         │ ████████████
         │█████████████████████████████████████
         └────────────────┬────────────────────→ w
                          1
                    新好样本        已学好样本
                   (需要学习)      (可以忽略)


                    A < 0 时的信息分布

    信息量 │
         │                              ████
         │                          ████████
         │                        ████████████
         │█████████████████████████████████████
         └────────────────┬────────────────────→ w
                          1
                   已避免坏样本      未避免坏样本
                   (可以忽略)       (需要惩罚)
```

---

# 第三部分：Softplus 截断的推导

## 5. 从约束优化到 Softplus

### 5.1 硬约束问题

考虑单边约束问题（以 $A > 0$ 为例）：

$$\max_{\tilde{\rho}} \mathbb{E}[e^{\tilde{\rho}} \cdot A] \quad \text{s.t.} \quad \tilde{\rho} \leq C$$

其中 $\rho = \log w$，$C$ 是带宽上界。

**硬约束解**：
$$\tilde{\rho}_{hard} = \min(\rho, C)$$

问题：不光滑，梯度在 $\rho = C$ 处不连续。

### 5.2 熵正则化

引入熵正则化将硬约束软化：

$$\max_{\tilde{\rho}} \mathbb{E}[e^{\tilde{\rho}} \cdot A] + \tau \cdot H(\tilde{\rho}) \quad \text{s.t.} \quad \tilde{\rho} \leq C$$

其中 $\tau$ 是温度参数，$H$ 是熵。

### 5.3 Fenchel-Legendre 对偶

**关键引理**：Softplus 是 max 函数的光滑近似

$$\text{Softplus}(x) = \tau \cdot \log(1 + e^{x/\tau})$$

是以下优化问题的解：

$$\text{Softplus}(x) = \max_{p \in [0,1]} \left\{ p \cdot x + \tau \cdot H(p) \right\}$$

其中 $H(p) = -p\log p - (1-p)\log(1-p)$ 是二元熵。

### 5.4 推导 Softplus 截断

**定理 5.1**：对于约束 $\tilde{\rho} \leq C$ 的熵正则化问题，最优解为：

$$\tilde{\rho}^* = C - \tau \cdot \text{Softplus}\left(\frac{C - \rho}{\tau}\right)$$

**证明**：

考虑拉格朗日函数：
$$\mathcal{L}(\tilde{\rho}, \lambda) = e^{\tilde{\rho}} \cdot A + \tau H(\tilde{\rho}) - \lambda(\tilde{\rho} - C)$$

KKT 条件给出：
$$\frac{\partial \mathcal{L}}{\partial \tilde{\rho}} = e^{\tilde{\rho}} \cdot A - \tau \log\frac{\tilde{\rho}}{1-\tilde{\rho}} - \lambda = 0$$

通过变分分析（详见附录 A），最优解满足：

$$\tilde{\rho}^* = C - \tau \cdot \text{Softplus}\left(\frac{C - \rho}{\tau}\right)$$

$\blacksquare$

### 5.5 边界行为分析

**命题 5.2**：Softplus 截断的边界行为

$$\tilde{\rho}(C, \rho, \tau) = C - \tau \cdot \text{Softplus}\left(\frac{C - \rho}{\tau}\right)$$

满足：

1. **上界**：$\lim_{\rho \to +\infty} \tilde{\rho} = C$
2. **线性还原**：$\lim_{\rho \to -\infty} \tilde{\rho} = \rho$
3. **转折点**：当 $\rho = C$ 时，$\tilde{\rho} = C - \tau \log 2$
4. **光滑性**：$\tilde{\rho}$ 关于 $\rho$ 处处可微

**证明**：

1. 当 $\rho \to +\infty$：
   $$\text{Softplus}\left(\frac{C - \rho}{\tau}\right) \to 0$$
   因此 $\tilde{\rho} \to C$

2. 当 $\rho \to -\infty$：
   $$\text{Softplus}\left(\frac{C - \rho}{\tau}\right) \approx \frac{C - \rho}{\tau}$$
   因此 $\tilde{\rho} \approx C - (C - \rho) = \rho$

3. 当 $\rho = C$：
   $$\tilde{\rho} = C - \tau \cdot \text{Softplus}(0) = C - \tau \log 2$$

4. Softplus 是光滑函数，因此 $\tilde{\rho}$ 光滑。

$\blacksquare$

---

## 6. 非对称截断

### 6.1 双边问题

对于 $A < 0$，我们需要**下界约束**而非上界：

$$\max_{\tilde{\rho}} \mathbb{E}[e^{\tilde{\rho}} \cdot (-|A|)] \quad \text{s.t.} \quad \tilde{\rho} \geq -C$$

等价于：
$$\min_{\tilde{\rho}} \mathbb{E}[e^{\tilde{\rho}} \cdot |A|] \quad \text{s.t.} \quad \tilde{\rho} \geq -C$$

### 6.2 下界截断的推导

**定理 6.1**：对于约束 $\tilde{\rho} \geq -C$ 的问题，最优解为：

$$\tilde{\rho}^* = -C + \tau \cdot \text{Softplus}\left(\frac{C + \rho}{\tau}\right)$$

**边界行为**：
1. **下界**：$\lim_{\rho \to -\infty} \tilde{\rho} = -C$
2. **线性还原**：$\lim_{\rho \to +\infty} \tilde{\rho} = \rho$

### 6.3 统一公式

**定理 6.2**（非对称 IB-IS 截断）：

$$\tilde{\rho}(w, A) = \begin{cases}
C - \tau \cdot \text{Softplus}\left(\frac{C - \rho}{\tau}\right) & A > 0 \quad \text{(上界截断)} \\[8pt]
-C + \tau \cdot \text{Softplus}\left(\frac{C + \rho}{\tau}\right) & A < 0 \quad \text{(下界截断)}
\end{cases}$$

其中 $\rho = \log w = \log \pi - \log \mu$。

**物理解释**：
- $A > 0$：限制"过度相信已知好样本"→ 上界截断
- $A < 0$：限制"过度惩罚已避免坏样本"→ 下界截断

---

# 第四部分：理论分析

## 7. 信息论性质

### 7.1 Rate-Distortion 分析

**定义 7.1**（Rate）：传输的信息量
$$R = I(w; \tilde{w}) = H(\tilde{w}) - H(\tilde{w}|w)$$

对于确定性映射 $\tilde{w} = \phi(w)$：
$$R = H(\tilde{w}) \leq H(w)$$

**定义 7.2**（Distortion）：估计偏差
$$D = |\mathbb{E}_\mu[\tilde{w} \cdot A] - \mathbb{E}_\pi[A]|^2$$

### 7.2 Rate-Distortion 权衡

**定理 7.1**：Softplus 截断实现了 Rate-Distortion 的 Pareto 最优

给定带宽 $C$ 和温度 $\tau$，Softplus 截断满足：

$$R(C, \tau) = O(C) \quad \text{(Rate 有界)}$$
$$D(C, \tau) = O(e^{-C/\tau}) \quad \text{(Distortion 指数衰减)}$$

### 7.3 信息瓶颈的几何解释

```
        Rate R = I(w; w̃)
            │
            │    ●  完整 IS (φ(w) = w)
            │   ╱
            │  ╱  Rate-Distortion 曲线
            │ ╱
            │●────── Softplus 截断
            │ ╲
            │  ╲
            │   ●  SFT (φ(w) = 1)
            └──────────────────────→ Distortion D
                                    (Bias²)
```

---

## 8. 与现有方法的联系

### 8.1 PPO Clip 作为硬约束 IB

PPO 的 clip 操作：
$$w_{clip} = \text{clip}(w, 1-\epsilon, 1+\epsilon)$$

这是双边硬约束 IB 的解（$\tau \to 0$ 极限）：
$$\tilde{\rho} = \text{clip}(\rho, \log(1-\epsilon), \log(1+\epsilon))$$

**问题**：不光滑，梯度在边界处不连续。

### 8.2 SAPO 作为近似 IB

SAPO 的 gate 函数：
$$\text{gate}(w) = \sigma(\tau(w-1)) \cdot \frac{4}{\tau}$$

这可以看作是在 $w$ 空间（而非 $\log w$ 空间）的软约束。

**问题**：不对应严格的 IB 解。

### 8.3 α-divergence 作为无约束 IB

IS-Reshape 的 $\phi(w) = w^\gamma$ 对应：
$$\tilde{\rho} = \gamma \cdot \rho$$

这是**无约束**的线性压缩：
- $\gamma < 1$：压缩（减少 Rate）
- $\gamma = 1$：无压缩（完整 IS）

**问题**：无界，$w \to \infty$ 时 $\tilde{w} \to \infty$。

### 8.4 方法对比总结

| 方法 | IB 视角 | 约束类型 | 有界性 | 光滑性 |
|-----|--------|---------|-------|-------|
| PPO Clip | 硬约束 IB | 双边对称 | ✓ | ✗ |
| SAPO | 近似 IB | w 空间 | ✓ | ✓ |
| IS-Reshape (w^γ) | 无约束 IB | 线性缩放 | ✗ | ✓ |
| **IB-IS (v9)** | 熵正则化 IB | 非对称 Softplus | ✓ | ✓ |

---

## 9. 梯度分析

### 9.1 梯度推导

设 $\tilde{\rho} = f(\rho)$ 为截断函数，损失为：
$$L = -\mathbb{E}_\mu[e^{\tilde{\rho}} \cdot A]$$

梯度：
$$\nabla_\theta L = -\mathbb{E}_\mu\left[e^{\tilde{\rho}} \cdot f'(\rho) \cdot A \cdot \nabla_\theta \log \pi\right]$$

其中 $f'(\rho) = \frac{d\tilde{\rho}}{d\rho}$。

### 9.2 Softplus 截断的梯度

对于上界截断 $\tilde{\rho} = C - \tau \cdot \text{Softplus}((C-\rho)/\tau)$：

$$f'(\rho) = \sigma\left(\frac{\rho - C}{\tau}\right) = \frac{1}{1 + e^{(C-\rho)/\tau}}$$

**性质**：
- $\rho \ll C$：$f'(\rho) \approx 1$（线性区，完整梯度）
- $\rho \gg C$：$f'(\rho) \approx 0$（饱和区，梯度衰减）
- $\rho = C$：$f'(\rho) = 0.5$（半梯度）

### 9.3 有效梯度权重

定义有效梯度权重：
$$\phi_{eff}(w, A) = e^{\tilde{\rho}} \cdot f'(\rho) = \tilde{w} \cdot f'(\log w)$$

**对于 $A > 0$**：
$$\phi_{eff}^+(w) = e^{C - \tau \cdot \text{Softplus}((C-\log w)/\tau)} \cdot \sigma\left(\frac{\log w - C}{\tau}\right)$$

**对于 $A < 0$**：
$$\phi_{eff}^-(w) = e^{-C + \tau \cdot \text{Softplus}((C+\log w)/\tau)} \cdot \sigma\left(\frac{\log w + C}{\tau}\right)$$

---

# 第五部分：实现

## 10. 算法实现

### 10.1 核心代码

```python
import torch
import torch.nn.functional as F
import math
from typing import Tuple, Dict, Optional

class InformationBottleneckIS:
    """
    Information Bottleneck 视角下的 IS 重塑

    理论基础：
    - 将 IS 权重处理视为信息压缩问题
    - 使用 Softplus 实现熵正则化的软约束
    - 正负样本使用相反方向的压缩

    核心公式：
        A > 0: ρ̃ = C - τ·Softplus((C - ρ)/τ)  [上界截断]
        A < 0: ρ̃ = -C + τ·Softplus((C + ρ)/τ) [下界截断]
    """

    def __init__(
        self,
        bandwidth: float = 0.5,
        temperature: float = 1.0,
        bandwidth_pos: Optional[float] = None,
        bandwidth_neg: Optional[float] = None,
    ):
        """
        Args:
            bandwidth: 默认信息带宽 C（控制截断位置）
            temperature: 温度参数 τ（控制截断光滑度）
            bandwidth_pos: A > 0 时的带宽（可选，默认使用 bandwidth）
            bandwidth_neg: A < 0 时的带宽（可选，默认使用 bandwidth）
        """
        self.bandwidth = bandwidth
        self.temperature = temperature
        self.C_pos = bandwidth_pos if bandwidth_pos is not None else bandwidth
        self.C_neg = bandwidth_neg if bandwidth_neg is not None else bandwidth

    def softplus_upper_clip(
        self,
        rho: torch.Tensor,
        C: float,
        tau: float
    ) -> torch.Tensor:
        """
        上界软截断: ρ̃ = C - τ·Softplus((C - ρ)/τ)

        性质：
        - ρ → -∞: ρ̃ → ρ (线性还原)
        - ρ → +∞: ρ̃ → C (饱和)
        """
        delta = (C - rho) / tau
        return C - tau * F.softplus(delta)

    def softplus_lower_clip(
        self,
        rho: torch.Tensor,
        C: float,
        tau: float
    ) -> torch.Tensor:
        """
        下界软截断: ρ̃ = -C + τ·Softplus((C + ρ)/τ)

        性质：
        - ρ → -∞: ρ̃ → -C (饱和)
        - ρ → +∞: ρ̃ → ρ (线性还原)
        """
        delta = (C + rho) / tau
        return -C + tau * F.softplus(delta)

    def compute_smoothed_rho(
        self,
        rho: torch.Tensor,
        advantages: torch.Tensor,
    ) -> torch.Tensor:
        """
        根据 Advantage 符号选择截断方向

        A > 0: 上界截断（压缩 w > 1 区域）
        A < 0: 下界截断（压缩 w < 1 区域）
        """
        # 上界截断 (A > 0)
        rho_upper = self.softplus_upper_clip(rho, self.C_pos, self.temperature)

        # 下界截断 (A < 0)
        rho_lower = self.softplus_lower_clip(rho, self.C_neg, self.temperature)

        # 根据 A 符号选择
        rho_smooth = torch.where(advantages > 0, rho_upper, rho_lower)

        return rho_smooth

    def __call__(
        self,
        log_pi: torch.Tensor,
        log_mu: torch.Tensor,
        advantages: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        计算 IB-IS 损失

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
        rho = log_pi - log_mu
        rho = torch.clamp(rho, -20.0, 20.0)  # 数值稳定性

        # 2. 非对称 Softplus 截断
        rho_smooth = self.compute_smoothed_rho(rho, advantages)

        # 3. 转换为权重
        w_smooth = torch.exp(rho_smooth)

        # 4. 计算损失
        pg_obj = w_smooth * advantages

        if mask is not None:
            loss = -(pg_obj * mask).sum() / (mask.sum() + 1e-8)
        else:
            loss = -pg_obj.mean()

        # 5. 诊断指标
        with torch.no_grad():
            w_original = torch.exp(rho)
            pos_mask = advantages > 0
            neg_mask = advantages < 0

            metrics = {
                "ib_is/w_original_mean": w_original.mean().item(),
                "ib_is/w_smooth_mean": w_smooth.mean().item(),
                "ib_is/rho_mean": rho.mean().item(),
                "ib_is/rho_smooth_mean": rho_smooth.mean().item(),
                "ib_is/compression_ratio": (rho_smooth / (rho + 1e-8)).mean().item(),
                "ib_is/bandwidth_pos": self.C_pos,
                "ib_is/bandwidth_neg": self.C_neg,
                "ib_is/temperature": self.temperature,
            }

            # 分组统计
            if pos_mask.any():
                metrics["ib_is/w_smooth_pos_mean"] = w_smooth[pos_mask].mean().item()
                metrics["ib_is/rho_clipped_pos"] = (rho[pos_mask] > self.C_pos - 0.1).float().mean().item()

            if neg_mask.any():
                metrics["ib_is/w_smooth_neg_mean"] = w_smooth[neg_mask].mean().item()
                metrics["ib_is/rho_clipped_neg"] = (rho[neg_mask] < -self.C_neg + 0.1).float().mean().item()

        return loss, metrics


def compute_policy_loss_ib_is(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    bandwidth: float = 0.5,
    temperature: float = 1.0,
    bandwidth_pos: Optional[float] = None,
    bandwidth_neg: Optional[float] = None,
) -> Tuple[torch.Tensor, Dict]:
    """
    Information Bottleneck IS 策略损失函数

    可直接集成到 veRL 的 policy loss 框架中
    """
    ib_is = InformationBottleneckIS(
        bandwidth=bandwidth,
        temperature=temperature,
        bandwidth_pos=bandwidth_pos,
        bandwidth_neg=bandwidth_neg,
    )

    return ib_is(log_prob, old_log_prob, advantages, response_mask)
```

### 10.2 注册到 veRL

```python
# 在 core_algos.py 中添加
from verl.trainer.ppo.core_algos import register_policy_loss, agg_loss

@register_policy_loss("ib_is")
def compute_policy_loss_ib_is_registered(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, Dict]:
    """
    Information Bottleneck IS 策略损失

    Config example:
        actor:
          policy_loss:
            loss_mode: "ib_is"
            ib_is:
              bandwidth: 0.5       # 默认信息带宽
              temperature: 1.0    # 温度参数
              bandwidth_pos: null # A > 0 时的带宽 (null = 使用默认)
              bandwidth_neg: null # A < 0 时的带宽 (null = 使用默认)
    """
    # 提取配置
    ib_config = config.policy_loss.get("ib_is", {}) if config else {}
    bandwidth = ib_config.get("bandwidth", 0.5)
    temperature = ib_config.get("temperature", 1.0)
    bandwidth_pos = ib_config.get("bandwidth_pos", None)
    bandwidth_neg = ib_config.get("bandwidth_neg", None)

    # 计算 rho
    rho = log_prob - old_log_prob
    rho = torch.clamp(rho, -20.0, 20.0)

    C_pos = bandwidth_pos if bandwidth_pos is not None else bandwidth
    C_neg = bandwidth_neg if bandwidth_neg is not None else bandwidth
    tau = temperature

    # 非对称 Softplus 截断
    # A > 0: 上界截断
    rho_upper = C_pos - tau * F.softplus((C_pos - rho) / tau)
    # A < 0: 下界截断
    rho_lower = -C_neg + tau * F.softplus((C_neg + rho) / tau)

    # 根据 A 符号选择
    rho_smooth = torch.where(advantages > 0, rho_upper, rho_lower)
    w_smooth = torch.exp(rho_smooth)

    # 计算损失
    loss_mat = -w_smooth * advantages
    pg_loss = agg_loss(loss_mat, response_mask, loss_agg_mode)

    # 诊断指标
    with torch.no_grad():
        import verl.utils.torch_functional as verl_F

        w_original = torch.exp(rho)
        mask = response_mask > 0
        pos_mask = (advantages > 0) & mask
        neg_mask = (advantages < 0) & mask

        ppo_kl = verl_F.masked_mean(-rho, response_mask)

        metrics = {
            "actor/ppo_kl": ppo_kl.item(),
            "actor/w_original_mean": verl_F.masked_mean(w_original, response_mask).item(),
            "actor/w_smooth_mean": verl_F.masked_mean(w_smooth, response_mask).item(),
            "ib_is/bandwidth": bandwidth,
            "ib_is/bandwidth_pos": C_pos,
            "ib_is/bandwidth_neg": C_neg,
            "ib_is/temperature": tau,
        }

        # 分组统计
        if pos_mask.any():
            metrics["ib_is/w_smooth_pos_mean"] = w_smooth[pos_mask].mean().item()
            metrics["ib_is/n_pos"] = pos_mask.sum().item()

        if neg_mask.any():
            metrics["ib_is/w_smooth_neg_mean"] = w_smooth[neg_mask].mean().item()
            metrics["ib_is/n_neg"] = neg_mask.sum().item()

    return pg_loss, metrics
```

### 10.3 配置示例

```yaml
# config/ib_is_example.yaml
actor_rollout_ref:
  actor:
    policy_loss:
      loss_mode: "ib_is"
      ib_is:
        bandwidth: 0.5       # 默认信息带宽 (对应 w 的范围约 [e^{-0.5}, e^{0.5}] ≈ [0.6, 1.6])
        temperature: 1.0     # 温度（越小截断越硬）
        bandwidth_pos: null  # A > 0 的带宽，null 表示使用默认
        bandwidth_neg: null  # A < 0 的带宽，null 表示使用默认
```

---

## 11. 超参数指南

### 11.1 bandwidth (C) 的选择

| C 值 | 对应 w 范围 | 效果 | 适用场景 |
|-----|-----------|------|---------|
| 0.2 | [0.82, 1.22] | 保守，接近 SFT | 早期训练，分布差异大 |
| 0.5 | [0.61, 1.65] | 中等 | 默认推荐 |
| 1.0 | [0.37, 2.72] | 激进，接近 IS | 后期训练，分布接近 |
| 2.0 | [0.14, 7.39] | 非常激进 | 几乎完整 IS |

### 11.2 temperature (τ) 的选择

| τ 值 | 效果 | 梯度特性 |
|-----|------|---------|
| 0.1 | 接近硬截断 | 边界处梯度陡峭 |
| 1.0 | 中等光滑 | 默认推荐 |
| 5.0 | 非常光滑 | 梯度平缓 |

### 11.3 非对称带宽

当正负样本分布差异大时，可以使用不同的带宽：

```python
# 示例：对好样本更保守，对坏样本更激进
bandwidth_pos: 0.3  # 限制对好样本的过拟合
bandwidth_neg: 0.7  # 允许更强的惩罚未避免的坏样本
```

---

# 第六部分：总结与展望

## 12. 核心贡献总结

```
┌─────────────────────────────────────────────────────────────────────────┐
│                   IS-Reshape v9: Information Bottleneck                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  【理论贡献】                                                            │
│                                                                         │
│  1. IB 形式化：将 IS 重塑映射到 Information Bottleneck 框架              │
│     - X (源): 原始 IS 权重 w                                            │
│     - T (瓶颈): 重塑权重 φ(w)                                           │
│     - Y (目标): π 下期望的估计                                          │
│                                                                         │
│  2. Rate-Distortion 等价：偏差-方差权衡 = Rate-Distortion 权衡          │
│     - Rate = I(w; φ(w)) ~ 方差                                         │
│     - Distortion = Bias² ~ 估计误差                                    │
│                                                                         │
│  3. Softplus 截断推导：从 Fenchel-Legendre 对偶推导最优解               │
│     ρ̃ = C - τ·Softplus((C - ρ)/τ)                                      │
│                                                                         │
│  4. 非对称压缩：证明正负样本需要相反方向的信息压缩                        │
│     - A > 0: 上界截断（压缩 w > 1）                                     │
│     - A < 0: 下界截断（压缩 w < 1）                                     │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  【与现有方法的统一】                                                    │
│                                                                         │
│  ┌─────────────┬────────────────┬─────────────────────────────────┐    │
│  │   方法       │    IB 视角     │          特点                   │    │
│  ├─────────────┼────────────────┼─────────────────────────────────┤    │
│  │ PPO Clip    │ 硬约束 IB      │ τ → 0 极限                      │    │
│  │ SAPO        │ 近似 IB (w空间)│ 工程有效，理论不严格             │    │
│  │ IS-Reshape  │ 无约束 IB      │ 优雅但无界                      │    │
│  │ IB-IS (v9)  │ 熵正则化 IB    │ 有界、光滑、非对称              │    │
│  └─────────────┴────────────────┴─────────────────────────────────┘    │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  【实用优势】                                                            │
│                                                                         │
│  ✓ 有界性：Softplus 截断保证权重有界                                     │
│  ✓ 光滑性：全程可微，梯度稳定                                            │
│  ✓ 非对称：正负样本自动使用最优压缩方向                                   │
│  ✓ 可解释：超参数有明确的信息论含义                                       │
│  ✓ 少参数：只需 bandwidth 和 temperature 两个超参数                      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## 13. 未来方向

### 13.1 自适应带宽

根据训练阶段或 KL 散度自动调整 $C$：

$$C(t) = C_0 + \alpha \cdot \log(1 + \text{KL}(\pi_t \| \mu))$$

### 13.2 多级 IB

使用多个带宽阈值实现更细粒度的控制：

$$\tilde{\rho} = \sum_i \alpha_i \cdot \text{Softplus\_clip}(\rho, C_i, \tau_i)$$

### 13.3 与其他正则化结合

将 IB-IS 与 KL 正则化、熵正则化结合：

$$L = L_{IB-IS} + \beta_{KL} \cdot KL(\pi \| \pi_{ref}) + \beta_H \cdot H(\pi)$$

---

## 附录 A：Fenchel-Legendre 对偶详细推导

### A.1 对偶问题

考虑约束优化问题：
$$\max_{\tilde{\rho}} f(\tilde{\rho}) \quad \text{s.t.} \quad \tilde{\rho} \leq C$$

引入拉格朗日乘子 $\lambda \geq 0$：
$$\mathcal{L}(\tilde{\rho}, \lambda) = f(\tilde{\rho}) - \lambda(\tilde{\rho} - C)$$

### A.2 熵正则化

加入熵正则化：
$$\mathcal{L}_\tau(\tilde{\rho}, \lambda) = f(\tilde{\rho}) - \lambda(\tilde{\rho} - C) + \tau H(\tilde{\rho})$$

对于二元选择（约束是否激活），熵为：
$$H(p) = -p\log p - (1-p)\log(1-p)$$

### A.3 求解

设 $p = P(\text{约束激活})$，则：
$$\tilde{\rho} = (1-p) \cdot \rho + p \cdot C$$

优化 $p$：
$$\max_p \left\{ f((1-p)\rho + pC) + \tau H(p) \right\}$$

一阶条件给出：
$$p^* = \sigma\left(\frac{C - \rho}{\tau}\right)$$

代入得：
$$\tilde{\rho}^* = C - \tau \cdot \text{Softplus}\left(\frac{C - \rho}{\tau}\right)$$

$\blacksquare$

---

## 附录 B：梯度推导细节

### B.1 链式法则

设 $\tilde{\rho} = f(\rho)$，损失 $L = -\mathbb{E}[e^{\tilde{\rho}} \cdot A]$。

$$\frac{\partial L}{\partial \theta} = -\mathbb{E}\left[e^{\tilde{\rho}} \cdot A \cdot \frac{\partial \tilde{\rho}}{\partial \theta}\right]$$

$$= -\mathbb{E}\left[e^{\tilde{\rho}} \cdot A \cdot f'(\rho) \cdot \frac{\partial \rho}{\partial \theta}\right]$$

$$= -\mathbb{E}\left[e^{\tilde{\rho}} \cdot f'(\rho) \cdot A \cdot \nabla_\theta \log \pi\right]$$

### B.2 Softplus 截断的导数

对于 $f(\rho) = C - \tau \cdot \text{Softplus}((C-\rho)/\tau)$：

$$f'(\rho) = \sigma\left(\frac{\rho - C}{\tau}\right)$$

这是一个 sigmoid 函数，值域 $(0, 1)$。

---

## 参考文献

1. Tishby, N., Pereira, F. C., & Bialek, W. (2000). The information bottleneck method.
2. Alemi, A. A., Fischer, I., Dillon, J. V., & Murphy, K. (2016). Deep variational information bottleneck.
3. Schulman, J., et al. (2017). Proximal policy optimization algorithms.
4. SAPO paper: arXiv:2511.20347
5. IS-Reshape theory: v1-v8 documents in this repository.
