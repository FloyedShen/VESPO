Efficient Learning from Offline Samples
1. Motivation
对于 LLM 的 RL 训练, 开销主要在于 auto regressive 过程, 而这一过程和获取 on policy 数据是密不可分的. 因此如何从 offline 数据中学习, 能够避免, 或者减少采样的过程, 对于高效优化至关重要. 
目前对于 offline 数据的利用, 主要有两个策略:
SFT, 就是直接优化 $\argmin_{\theta} \mathbf{KL}(\pi_{\mathcal{D}_{data}}||\pi_{\theta})$
Distillation, 就是在优化 $\mathbb{E}_{(x, y) \sim \mathcal{D}_{data}}[\mathbf{D}_{JSD(\beta)}(\pi_t||\pi_\theta)]$, 而当模型足够强大, 而无法找到一个合理的 $\pi_t$ 的时候, 方法就会失效
所以如何不引入, 或者在尽量少的 采样 的限制下, 高效利用 offline sample 仍是一个挑战. 

2. The difference between SFT and RL - from an information theory perspective
从信息论的视角, SFT / RL 在优化的时候, 到底在如何修改这个 policy 的分布呢?
对于 SFT, 优化目标是 $\min \mathbb{E}_{(x, y) \sim D}[-\log \pi_{\theta}(y | x)] = \min \mathbb{E}_{(x, y) \sim D}[\log \pi_{D}(y|x)-\log \pi_{\theta}(y | x)] = \min \text{KL}(\pi_{D} || \pi_{\theta})$, 
也就是在迫使 $\pi_\theta$做朝向 $\pi_D$ 的 mean-seeking
 而对于 RL 来说, 优化的目标是 $\max \mathbb{E}_{x \sim \pi_\theta}[r(x)] = \max \mathbb{E}_{x \sim \pi_\theta}[\log e^{r(x)}]$
如果定义最优分布 $\pi^* = e^{r(x) / \tau} / Z$, 那么可以反解 $r(x) = \tau \log \pi^*(x) + \tau \log Z$
那么就有 $\max \mathbb{E}_{x \sim \pi_\theta}[r(x)] = \max \tau  \mathbb{E}_{x \sim \pi_\theta}[\log \pi^*(x)] + \tau \log Z = \min [\tau \text{KL}(\pi_\theta || \pi^*) + \tau H(\pi_\theta)]$
也就是说, 让 $\pi_\theta $ 朝着 $\pi^*$ 做 mode-seeking

Forward / Reverse KL 


One More Thing
在使用 正确的答案做 SFT 的时候, 相当于优化 $\text{KL}(\pi_{off}(\cdot | x, r(y)=1) || \pi_\theta(\cdot|x))$. 相当于在向着 off policy 的正确的子空间做 mean-seeking. 如果有多种正确解法,模型仍会试图覆盖所有正确的模式.
而如果加上 IS 之后, 模型的行为就会变成 $\max_\theta \mathbb{E}_{x, y \sim \mu} \left[\mathbb{1}[r(y)=1] \cdot \frac{\pi_\theta(y|x)}{\mu(y|x)} \cdot \log \pi_\theta(y|x)\right] $
$\max_\theta \mathbb{E}_{x, y \sim \pi_\theta} \left[\mathbb{1}[r(y)=1] \cdot \log \pi_\theta(y|x)\right]$
也就是相当于 REINFORCE 算法的特殊形式: reward是0/1. 也相当于是在优化 $\text{KL}(\pi_\theta || \pi^*)$.  但是 IS ratio 可能不稳定, 导致优化的时候崩掉. 

那如果我们使用一些非线性函数, 在优化的时候, 使用 $f(\frac{\pi_\theta(y|x)}{\mu(y|x)})$ 进行 IS shaping, 从 KL 的角度 应该如何理解呢? 
不失一般性地, 可以将 $\mathbb{1}[r(y)=1] $ 这个条件推广为 $r(x, y).$ 
那么我们的目标分布就是 $\pi^*(y|x) = \frac{\exp(r(x,y)/\tau)}{\int \exp(r(x,y')/\tau) dy'}$
优化目标就是:
$\max_\theta \mathbb{E}_{y \sim \mu} \left[f(\frac{\pi_\theta(y|x)}{\mu(y|x)}) \cdot r(x,y)\right]$
对于 $f(w) = w$, 也就是相当于做正统 IS, 那么则有 $\max_\theta \mathbb{E}_{y \sim \pi_\theta} [r(x,y) ]$, 也就是, 在使用 IS 的时候, 优化目标就是
$\boxed{\min_\theta D_{\text{KL}}(\pi_\theta \| \pi^*)}$, 也就是 reverse KL, 对 $\pi^*$ 进行 mode seeking
对于 $f(w) = 1$, 也就是不进行 IS, 那么实际在优化 $L(\theta) = -\mathbb{E}_{y \sim \mu}[r(y) \log \pi_\theta(y|x)]$.
定义个新的分布: $\tilde{\pi}(y|x) = \frac{\mu(y|x) \exp(r(y)/\tau)}{\int \mu(y'|x) \exp(r(y')/\tau) dy'}$
那么实际上, 我们在优化的是 $\min D_{\text{KL}}(\tilde{\pi} \| \pi_\theta)$. 
证明:
$D_{\text{KL}}(\tilde{\pi} \| \pi_\theta) = \mathbb{E}_{\tilde{\pi}}\left[\log \frac{\tilde{\pi}}{\pi_\theta}\right] \\
= \mathbb{E}_{\tilde{\pi}}[\log \tilde{\pi}] - \mathbb{E}_{\tilde{\pi}}[\log \pi_\theta]$
其中第二项
$\mathbb{E}_{\tilde{\pi}}[\log \pi_\theta] = \int \tilde{\pi}(y|x) \log \pi_\theta(y|x) dy \\
= \int \frac{\mu(y|x) e^{r/\tau}}{Z} \log \pi_\theta(y|x) dy 
\propto \mathbb{E}_{\mu}[e^{r/\tau} \log \pi_\theta]$
如果有 r 较小, 也就是 $e^{r/\tau} \approx 1 + r/\tau$, 那么则有
$\mathbb{E}_{\mu}[e^{r/\tau} \log \pi_\theta] \approx \mathbb{E}_{\mu}[(1 + r/\tau) \log \pi_\theta] \propto \mathbb{E}_{\mu}[r \log \pi_\theta]$.
也就是有 $\boxed{\min_\theta L(\theta) \iff \min_\theta D_{\text{KL}}(\tilde{\pi} \| \pi_\theta)}$. 也就是 Forward KL, 在对 $\tilde{\pi} $ 进行 mean-seeking. 也就是试图覆盖所有的, $\mu(y|x) \exp(r(y)/\tau)$的模式, 即 $\mu(y|x)$ 中所有的高 reward 样本的模式. 
Intution: 对于 IS 进行 shaping, 是在 mean / mode seeking 之间进行转换. 所以很难写成一般的 KL 形式. 那么这是否能写成一种广义的 divergence? 
假设 #1: $f(w) = w^\alpha$;
定理: 在上述假设下, 我们在最大化期望奖励等价于优化奖励加权的 Rényi Divergence.
$D_\alpha^{(r)}(\pi_\theta \| \mu) := \frac{1}{\alpha-1} \log \mathbb{E}_\mu\left[\left(\frac{\pi_\theta}{\mu}\right)^\alpha e^{r/\tau}\right]$
证明:
奖励加权的 Rényi Divergence 的梯度为:
$\nabla_\theta D_\alpha^{(r)} = \frac{\alpha}{\alpha-1} \cdot \frac{\mathbb{E}_\mu[w^\alpha e^{r/\tau} \nabla \log \pi_\theta]}{\mathbb{E}_\mu[w^\alpha e^{r/\tau}]} \\
= \frac{\alpha}{\alpha-1} \cdot \frac{\mathbb{E}_{\pi_\theta}[w^{\alpha-1} e^{r/\tau} \nabla \log \pi_\theta]}{\mathbb{E}_{\pi_\theta}[w^{\alpha-1} e^{r/\tau}]}$
线性近似：如果 r 足够小, $e^{r/\tau} \approx 1 + r/\tau$：
$\nabla_\theta D_\alpha^{(r)} \approx \frac{\alpha}{\alpha-1} \cdot \mathbb{E}_{\pi_\theta}\left[w^{\alpha-1} r \nabla \log \pi_\theta\right]$
这正是策略梯度. (也可以从 $D_\alpha^{(r)}(\pi_\theta \| \mu) $对于其中的 log 项 展开来证明, 略.)

alpha-ELBO

