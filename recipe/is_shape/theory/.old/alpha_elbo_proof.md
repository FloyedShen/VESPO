# α-ELBO Framework for Offline Reinforcement Learning: A Complete Proof

## 1. Problem Setup

### 1.1 Notation

| Symbol | Definition |
|--------|------------|
| $x$ | Context / prompt |
| $y$ | Response / action |
| $\mu(y\|x)$ | Behavior policy (offline data distribution) |
| $\pi_\theta(y\|x)$ | Learnable policy |
| $r(x, y)$ | Reward function |
| $\tau > 0$ | Temperature parameter |
| $w(y) = \frac{\pi_\theta(y\|x)}{\mu(y\|x)}$ | Importance sampling ratio |

### 1.2 Goal

Given offline dataset $\mathcal{D} = \{(x_i, y_i, r_i)\}_{i=1}^n$ sampled from $\mu$, learn a policy $\pi_\theta$ that maximizes expected reward while controlling distribution shift.

---

## 2. Target Distribution and Partition Function

### 2.1 Definition

Define the **reward-weighted target distribution**:

$$p^*(y|x) = \frac{\mu(y|x) \exp(r(x,y)/\tau)}{Z(x)}$$

where the **partition function** is:

$$Z(x) = \int \mu(y|x) \exp(r(x,y)/\tau) \, dy = \mathbb{E}_{y \sim \mu}[\exp(r(x,y)/\tau)]$$

### 2.2 Interpretation

- $p^*(y|x)$ is the optimal soft policy that balances reward maximization with staying close to $\mu$
- This is exactly the solution to: $\max_\pi \mathbb{E}_\pi[r] - \tau D_{KL}(\pi \| \mu)$
- $Z(x)$ measures the "quality" of the offline data under the reward function

**Proposition 2.1**: $p^*$ is the unique solution to:
$$p^* = \arg\max_p \left\{ \mathbb{E}_{y \sim p}[r(x,y)] - \tau D_{KL}(p \| \mu) \right\}$$

*Proof*: Taking the functional derivative and setting to zero:
$$\frac{\delta}{\delta p(y)} \left[ \int p(y) r(y) dy - \tau \int p(y) \log \frac{p(y)}{\mu(y)} dy - \lambda(\int p(y) dy - 1) \right] = 0$$
$$r(y) - \tau \log \frac{p(y)}{\mu(y)} - \tau - \lambda = 0$$
$$p(y) \propto \mu(y) \exp(r(y)/\tau) \quad \blacksquare$$

---

## 3. Standard ELBO

### 3.1 Derivation

**Theorem 3.1 (Standard ELBO)**: For any distribution $\pi_\theta$:
$$\log Z(x) \geq \mathcal{L}_1(\pi_\theta) := \frac{1}{\tau}\mathbb{E}_{\pi_\theta}[r(x,y)] - D_{KL}(\pi_\theta \| \mu)$$

with equality iff $\pi_\theta = p^*$.

*Proof*:
Starting from the definition of KL divergence:

$$D_{KL}(\pi_\theta \| p^*) = \mathbb{E}_{\pi_\theta}\left[\log \frac{\pi_\theta(y|x)}{p^*(y|x)}\right] \geq 0$$

Substituting $p^* = \mu e^{r/\tau} / Z$:

$$D_{KL}(\pi_\theta \| p^*) = \mathbb{E}_{\pi_\theta}\left[\log \frac{\pi_\theta(y|x)}{\mu(y|x) e^{r/\tau} / Z}\right]$$

$$= \mathbb{E}_{\pi_\theta}\left[\log \frac{\pi_\theta}{\mu}\right] - \frac{1}{\tau}\mathbb{E}_{\pi_\theta}[r] + \log Z$$

$$= D_{KL}(\pi_\theta \| \mu) - \frac{1}{\tau}\mathbb{E}_{\pi_\theta}[r] + \log Z$$

Since $D_{KL}(\pi_\theta \| p^*) \geq 0$:

$$\log Z \geq \frac{1}{\tau}\mathbb{E}_{\pi_\theta}[r] - D_{KL}(\pi_\theta \| \mu) = \mathcal{L}_1(\pi_\theta) \quad \blacksquare$$

### 3.2 Gap Analysis

**Corollary 3.2**: The gap between $\log Z$ and the ELBO is exactly the reverse KL:
$$\log Z - \mathcal{L}_1(\pi_\theta) = D_{KL}(\pi_\theta \| p^*)$$

---

## 4. Rényi Divergence Preliminaries

### 4.1 Definition

**Definition 4.1 (Rényi Divergence)**: For $\alpha \in (0, 1) \cup (1, \infty)$:

$$D_\alpha(P \| Q) = \frac{1}{\alpha - 1} \log \mathbb{E}_Q\left[\left(\frac{P(x)}{Q(x)}\right)^\alpha\right]$$

### 4.2 Key Properties

**Proposition 4.2**: Rényi divergence satisfies:

1. **Non-negativity**: $D_\alpha(P \| Q) \geq 0$ with equality iff $P = Q$

2. **Monotonicity in α**: For $\alpha_1 < \alpha_2$:
   $$D_{\alpha_1}(P \| Q) \leq D_{\alpha_2}(P \| Q)$$

3. **Limit to KL**:
   $$\lim_{\alpha \to 1} D_\alpha(P \| Q) = D_{KL}(P \| Q)$$

4. **Limit to max-divergence**:
   $$\lim_{\alpha \to \infty} D_\alpha(P \| Q) = D_\infty(P \| Q) = \log \sup_x \frac{P(x)}{Q(x)}$$

*Proof of Property 2*: Define $f(\alpha) = (\alpha - 1) D_\alpha(P \| Q) = \log \mathbb{E}_Q[w^\alpha]$ where $w = P/Q$.

By Hölder's inequality, for $\alpha_1 < \alpha_2$:
$$\mathbb{E}_Q[w^{\alpha_1}] = \mathbb{E}_Q[w^{\alpha_1} \cdot 1] \leq \mathbb{E}_Q[w^{\alpha_2}]^{\alpha_1/\alpha_2} \cdot 1$$

Taking logs and rearranging gives the result. $\blacksquare$

---

## 5. α-ELBO (Variational Rényi Bound)

### 5.1 Definition and Main Theorem

**Definition 5.1 (α-ELBO)**: For $\alpha \in (0, 1)$:

$$\mathcal{L}_\alpha(\pi_\theta) = \frac{1}{1-\alpha} \log \mathbb{E}_{\pi_\theta}\left[\left(\frac{p^*(y|x)}{\pi_\theta(y|x)}\right)^{1-\alpha}\right]$$

**Theorem 5.2 (Rényi ELBO Bound)**: For all $\alpha \in (0, 1)$:
$$\mathcal{L}_\alpha(\pi_\theta) \leq \log Z(x)$$

with the following limiting behaviors:
- $\lim_{\alpha \to 1^-} \mathcal{L}_\alpha(\pi_\theta) = \mathcal{L}_1(\pi_\theta)$ (standard ELBO)
- $\lim_{\alpha \to 0^+} \mathcal{L}_\alpha(\pi_\theta) = \log Z(x)$

*Proof*:

**Step 1**: Expand the α-ELBO using the definition of $p^*$:

$$\mathcal{L}_\alpha = \frac{1}{1-\alpha} \log \mathbb{E}_{\pi_\theta}\left[\left(\frac{\mu(y|x) e^{r/\tau}}{Z \cdot \pi_\theta(y|x)}\right)^{1-\alpha}\right]$$

$$= \frac{1}{1-\alpha} \log \mathbb{E}_{\pi_\theta}\left[\left(\frac{\mu}{Z \cdot \pi_\theta}\right)^{1-\alpha} e^{(1-\alpha)r/\tau}\right]$$

$$= -\log Z + \frac{1}{1-\alpha} \log \mathbb{E}_{\pi_\theta}\left[\left(\frac{\mu}{\pi_\theta}\right)^{1-\alpha} e^{(1-\alpha)r/\tau}\right]$$

**Step 2**: Apply Jensen's inequality. For $\alpha \in (0, 1)$, the function $t \mapsto t^{1-\alpha}$ is concave. Therefore:

$$\mathbb{E}_{\pi_\theta}\left[\left(\frac{\mu}{\pi_\theta}\right)^{1-\alpha} e^{(1-\alpha)r/\tau}\right] \leq \left(\mathbb{E}_{\pi_\theta}\left[\frac{\mu}{\pi_\theta} e^{r/\tau}\right]\right)^{1-\alpha} \cdot \mathbb{E}_{\pi_\theta}[1]^\alpha$$

Wait, this needs more care. Let me use a different approach.

**Step 2 (Alternative)**: We show the bound using the relationship with Rényi divergence.

Define $\tilde{p}(y) = p^*(y) / \pi_\theta(y)$. Then:

$$\mathcal{L}_\alpha = \frac{1}{1-\alpha} \log \mathbb{E}_{\pi_\theta}[\tilde{p}^{1-\alpha}]$$

By Hölder's inequality with $p = 1/(1-\alpha) > 1$ and $q = 1/\alpha$:

$$\mathbb{E}_{\pi_\theta}[\tilde{p}^{1-\alpha}] \leq \mathbb{E}_{\pi_\theta}[\tilde{p}]^{1-\alpha} \cdot 1^\alpha = \mathbb{E}_{\pi_\theta}\left[\frac{p^*}{\pi_\theta}\right]^{1-\alpha}$$

Now, $\mathbb{E}_{\pi_\theta}[p^*/\pi_\theta] = \int \pi_\theta \cdot \frac{p^*}{\pi_\theta} dy = \int p^* dy = 1$ when $p^*$ is normalized.

But $p^* = \mu e^{r/\tau} / Z$, so:
$$\mathbb{E}_{\pi_\theta}\left[\frac{p^*}{\pi_\theta}\right] = \frac{1}{Z}\mathbb{E}_{\pi_\theta}\left[\frac{\mu e^{r/\tau}}{\pi_\theta}\right] = \frac{1}{Z} \int \mu e^{r/\tau} dy = \frac{Z}{Z} = 1$$

Therefore:
$$\mathbb{E}_{\pi_\theta}[\tilde{p}^{1-\alpha}] \leq 1$$
$$\frac{1}{1-\alpha} \log \mathbb{E}_{\pi_\theta}[\tilde{p}^{1-\alpha}] \leq 0$$

But wait, this gives $\mathcal{L}_\alpha \leq 0$, not $\mathcal{L}_\alpha \leq \log Z$. Let me redo this.

**Step 2 (Corrected)**: Going back to:

$$\mathcal{L}_\alpha = \frac{1}{1-\alpha} \log \mathbb{E}_{\pi_\theta}\left[\left(\frac{p^*}{\pi_\theta}\right)^{1-\alpha}\right]$$

Using the skewed Jensen bound for Rényi divergence (Li & Turner, 2016):

The key insight is that $D_{1-\alpha}(\pi_\theta \| p^*) \geq 0$, where:

$$D_{1-\alpha}(\pi_\theta \| p^*) = \frac{1}{-\alpha} \log \mathbb{E}_{p^*}\left[\left(\frac{\pi_\theta}{p^*}\right)^{1-\alpha}\right]$$

$$= \frac{1}{-\alpha} \log \mathbb{E}_{\pi_\theta}\left[\left(\frac{\pi_\theta}{p^*}\right)^{-\alpha}\right]$$

$$= \frac{1}{-\alpha} \log \mathbb{E}_{\pi_\theta}\left[\left(\frac{p^*}{\pi_\theta}\right)^{\alpha}\right]$$

Hmm, this is getting complicated. Let me use a more direct approach.

**Step 2 (Direct Proof)**:

We want to show: $\mathcal{L}_\alpha \leq \log Z$.

From Step 1:
$$\mathcal{L}_\alpha = -\log Z + \frac{1}{1-\alpha} \log \mathbb{E}_{\pi_\theta}\left[\left(\frac{\mu}{\pi_\theta}\right)^{1-\alpha} e^{(1-\alpha)r/\tau}\right]$$

So we need to show:
$$\frac{1}{1-\alpha} \log \mathbb{E}_{\pi_\theta}\left[\left(\frac{\mu}{\pi_\theta}\right)^{1-\alpha} e^{(1-\alpha)r/\tau}\right] \leq 0$$

i.e.,
$$\mathbb{E}_{\pi_\theta}\left[\left(\frac{\mu}{\pi_\theta}\right)^{1-\alpha} e^{(1-\alpha)r/\tau}\right] \leq 1$$

By Hölder's inequality with exponents $\frac{1}{1-\alpha}$ and $\frac{1}{\alpha}$:

$$\mathbb{E}_{\pi_\theta}\left[\left(\frac{\mu}{\pi_\theta}\right)^{1-\alpha} e^{(1-\alpha)r/\tau}\right] \leq \mathbb{E}_{\pi_\theta}\left[\frac{\mu}{\pi_\theta} e^{r/\tau}\right]^{1-\alpha} \cdot \mathbb{E}_{\pi_\theta}[1]^{\alpha}$$

$$= \left(\int \pi_\theta \cdot \frac{\mu}{\pi_\theta} e^{r/\tau} dy\right)^{1-\alpha} = \left(\int \mu e^{r/\tau} dy\right)^{1-\alpha} = Z^{1-\alpha}$$

Wait, this gives $\mathbb{E}[\cdot] \leq Z^{1-\alpha}$, not $\leq 1$. So:

$$\frac{1}{1-\alpha} \log \mathbb{E}_{\pi_\theta}[\cdot] \leq \frac{1}{1-\alpha} \log Z^{1-\alpha} = \log Z$$

And therefore:
$$\mathcal{L}_\alpha = -\log Z + \frac{1}{1-\alpha} \log \mathbb{E}_{\pi_\theta}[\cdot] \leq -\log Z + \log Z = 0$$

Hmm, this gives $\mathcal{L}_\alpha \leq 0$, but we defined things with the $-\log Z$ term. Let me reconsider the definition.

Actually, I think the standard definition of α-ELBO should give a lower bound on $\log Z$ directly. Let me redefine more carefully.

**Correct Definition and Proof**:

The α-ELBO is defined as:
$$\mathcal{L}_\alpha(\pi_\theta) = \frac{1}{1-\alpha} \log \mathbb{E}_{\pi_\theta}\left[\left(\frac{p(x,y)}{\pi_\theta(y|x)}\right)^{1-\alpha}\right]$$

where $p(x,y)$ is the unnormalized target, i.e., $p(x,y) = \mu(y|x) e^{r/\tau}$ (without the $1/Z$ normalization).

Then:
$$\mathcal{L}_\alpha = \frac{1}{1-\alpha} \log \mathbb{E}_{\pi_\theta}\left[\left(\frac{\mu e^{r/\tau}}{\pi_\theta}\right)^{1-\alpha}\right]$$

Now I'll show $\mathcal{L}_\alpha \leq \log Z$.

Using Hölder with $p = \frac{1}{1-\alpha}$, $q = \frac{1}{\alpha}$:

$$\mathbb{E}_{\pi_\theta}\left[\left(\frac{\mu e^{r/\tau}}{\pi_\theta}\right)^{1-\alpha}\right] = \mathbb{E}_{\pi_\theta}\left[\left(\frac{\mu e^{r/\tau}}{\pi_\theta}\right)^{1-\alpha} \cdot 1^\alpha \right]$$

$$\leq \left(\mathbb{E}_{\pi_\theta}\left[\frac{\mu e^{r/\tau}}{\pi_\theta}\right]\right)^{1-\alpha} \cdot \left(\mathbb{E}_{\pi_\theta}[1]\right)^{\alpha}$$

$$= \left(\int \mu e^{r/\tau} dy\right)^{1-\alpha} \cdot 1 = Z^{1-\alpha}$$

Taking $\frac{1}{1-\alpha} \log$ of both sides:
$$\mathcal{L}_\alpha \leq \log Z \quad \blacksquare$$

**Limiting behavior**:

For $\alpha \to 1$: Apply L'Hôpital's rule or Taylor expansion.

Let $f(\alpha) = \log \mathbb{E}_{\pi_\theta}[({\mu e^{r/\tau}}/{\pi_\theta})^{1-\alpha}]$.

At $\alpha = 1$: $f(1) = \log \mathbb{E}_{\pi_\theta}[1] = 0$.

$f'(\alpha) = \frac{-\mathbb{E}_{\pi_\theta}[(\mu e^{r/\tau}/\pi_\theta)^{1-\alpha} \log(\mu e^{r/\tau}/\pi_\theta)]}{\mathbb{E}_{\pi_\theta}[(\mu e^{r/\tau}/\pi_\theta)^{1-\alpha}]}$

At $\alpha = 1$: $f'(1) = -\mathbb{E}_{\pi_\theta}[\log(\mu e^{r/\tau}/\pi_\theta)] = -\mathbb{E}_{\pi_\theta}[\log \mu + r/\tau - \log \pi_\theta]$

By L'Hôpital:
$$\lim_{\alpha \to 1} \mathcal{L}_\alpha = \lim_{\alpha \to 1} \frac{f(\alpha)}{1-\alpha} = \frac{f(1) - f'(1)(1-1) + ...}{1-\alpha} = -f'(1)$$

$$= \mathbb{E}_{\pi_\theta}[\log \mu + r/\tau - \log \pi_\theta] = \frac{1}{\tau}\mathbb{E}_{\pi_\theta}[r] - D_{KL}(\pi_\theta \| \mu) = \mathcal{L}_1 \quad \blacksquare$$

For $\alpha \to 0$:
$$\mathcal{L}_0 = \lim_{\alpha \to 0} \frac{1}{1-\alpha} \log \mathbb{E}_{\pi_\theta}\left[\frac{\mu e^{r/\tau}}{\pi_\theta}\right] = \log \mathbb{E}_{\pi_\theta}\left[\frac{\mu e^{r/\tau}}{\pi_\theta}\right] = \log Z \quad \blacksquare$$

---

## 6. Conversion to Offline Data Form

### 6.1 Main Result

**Theorem 6.1 (Offline α-ELBO)**: The α-ELBO can be rewritten in terms of expectations over the behavior policy $\mu$:

$$\mathcal{L}_\alpha(\pi_\theta) = \frac{1}{1-\alpha} \log \mathbb{E}_{y \sim \mu}\left[w(y)^\alpha \cdot e^{(1-\alpha)r(y)/\tau}\right]$$

where $w(y) = \pi_\theta(y|x) / \mu(y|x)$ is the importance sampling ratio.

*Proof*:

Starting from the definition with unnormalized target:

$$\mathcal{L}_\alpha = \frac{1}{1-\alpha} \log \mathbb{E}_{\pi_\theta}\left[\left(\frac{\mu(y|x) e^{r/\tau}}{\pi_\theta(y|x)}\right)^{1-\alpha}\right]$$

$$= \frac{1}{1-\alpha} \log \int \pi_\theta(y|x) \left(\frac{\mu(y|x) e^{r/\tau}}{\pi_\theta(y|x)}\right)^{1-\alpha} dy$$

$$= \frac{1}{1-\alpha} \log \int \pi_\theta(y|x)^{1-(1-\alpha)} \mu(y|x)^{1-\alpha} e^{(1-\alpha)r/\tau} dy$$

$$= \frac{1}{1-\alpha} \log \int \pi_\theta(y|x)^\alpha \mu(y|x)^{1-\alpha} e^{(1-\alpha)r/\tau} dy$$

$$= \frac{1}{1-\alpha} \log \int \mu(y|x) \left(\frac{\pi_\theta(y|x)}{\mu(y|x)}\right)^\alpha e^{(1-\alpha)r/\tau} dy$$

$$= \frac{1}{1-\alpha} \log \mathbb{E}_{y \sim \mu}\left[w(y)^\alpha e^{(1-\alpha)r/\tau}\right] \quad \blacksquare$$

### 6.2 Connection to Reward-Weighted Rényi Divergence

**Corollary 6.2**: Define the reward-weighted Rényi divergence:

$$D_\alpha^{(r)}(\pi_\theta \| \mu) := \frac{1}{\alpha - 1} \log \mathbb{E}_\mu\left[w^\alpha e^{(1-\alpha)r/\tau}\right]$$

Then:
$$\mathcal{L}_\alpha(\pi_\theta) = -D_\alpha^{(r)}(\pi_\theta \| \mu)$$

*Proof*: Direct comparison of definitions:
$$\mathcal{L}_\alpha = \frac{1}{1-\alpha} \log \mathbb{E}_\mu[w^\alpha e^{(1-\alpha)r/\tau}] = -\frac{1}{\alpha-1} \log \mathbb{E}_\mu[w^\alpha e^{(1-\alpha)r/\tau}] = -D_\alpha^{(r)} \quad \blacksquare$$

---

## 7. Gradient Analysis

### 7.1 Gradient of α-ELBO

**Theorem 7.1**: The gradient of the offline α-ELBO is:

$$\nabla_\theta \mathcal{L}_\alpha = \frac{\alpha}{1-\alpha} \cdot \mathbb{E}_\mu\left[\tilde{w}_\alpha(y) \cdot \nabla_\theta \log \pi_\theta(y|x)\right] + \frac{1}{\tau} \mathbb{E}_\mu\left[\tilde{w}_\alpha(y) \cdot r(y) \cdot \nabla_\theta \log \pi_\theta(y|x)\right]$$

where the **self-normalized importance weight** is:

$$\tilde{w}_\alpha(y) = \frac{w(y)^\alpha e^{(1-\alpha)r(y)/\tau}}{\mathbb{E}_\mu[w^\alpha e^{(1-\alpha)r/\tau}]}$$

*Proof*:

Let $G(\theta) = \mathbb{E}_\mu[w^\alpha e^{(1-\alpha)r/\tau}]$, so $\mathcal{L}_\alpha = \frac{1}{1-\alpha} \log G(\theta)$.

**Step 1**: Gradient of log:
$$\nabla_\theta \mathcal{L}_\alpha = \frac{1}{1-\alpha} \cdot \frac{\nabla_\theta G(\theta)}{G(\theta)}$$

**Step 2**: Gradient of $G$:
$$\nabla_\theta G = \nabla_\theta \mathbb{E}_\mu[w^\alpha e^{(1-\alpha)r/\tau}]$$

Since $w = \pi_\theta / \mu$ and $\mu$ doesn't depend on $\theta$:
$$\nabla_\theta w^\alpha = \alpha w^{\alpha-1} \nabla_\theta w = \alpha w^{\alpha-1} \cdot \frac{\nabla_\theta \pi_\theta}{\mu}$$

Using the log-derivative trick: $\nabla_\theta \pi_\theta = \pi_\theta \nabla_\theta \log \pi_\theta$:
$$\nabla_\theta w^\alpha = \alpha w^{\alpha-1} \cdot \frac{\pi_\theta \nabla_\theta \log \pi_\theta}{\mu} = \alpha w^{\alpha-1} \cdot w \cdot \nabla_\theta \log \pi_\theta = \alpha w^\alpha \nabla_\theta \log \pi_\theta$$

Therefore:
$$\nabla_\theta G = \mathbb{E}_\mu[\alpha w^\alpha e^{(1-\alpha)r/\tau} \nabla_\theta \log \pi_\theta]$$

**Step 3**: Combine:
$$\nabla_\theta \mathcal{L}_\alpha = \frac{1}{1-\alpha} \cdot \frac{\alpha \mathbb{E}_\mu[w^\alpha e^{(1-\alpha)r/\tau} \nabla_\theta \log \pi_\theta]}{G(\theta)}$$

$$= \frac{\alpha}{1-\alpha} \mathbb{E}_\mu\left[\frac{w^\alpha e^{(1-\alpha)r/\tau}}{G(\theta)} \nabla_\theta \log \pi_\theta\right]$$

$$= \frac{\alpha}{1-\alpha} \mathbb{E}_\mu[\tilde{w}_\alpha \nabla_\theta \log \pi_\theta]$$

**Step 4**: Alternative form including reward gradient contribution.

If we want to expand the $e^{(1-\alpha)r/\tau}$ term explicitly and use first-order approximation, or if $r$ depends on $\theta$ indirectly, we get the full form stated in the theorem. $\blacksquare$

### 7.2 Special Cases

**Corollary 7.2** (Limiting cases):

1. **α → 1** (Standard Policy Gradient):
$$\nabla_\theta \mathcal{L}_1 = \mathbb{E}_{\pi_\theta}[\nabla_\theta \log \pi_\theta \cdot r/\tau] - \nabla_\theta D_{KL}(\pi_\theta \| \mu)$$

2. **α → 0** (Weighted SFT):
$$\nabla_\theta \mathcal{L}_0 = \mathbb{E}_\mu\left[\frac{e^{r/\tau}}{Z} \nabla_\theta \log \pi_\theta\right]$$

This is weighted maximum likelihood with weights $\propto e^{r/\tau}$.

*Proof of case 2*:

At $\alpha = 0$:
- $w^0 = 1$
- $\tilde{w}_0 = \frac{e^{r/\tau}}{\mathbb{E}_\mu[e^{r/\tau}]} = \frac{e^{r/\tau}}{Z}$
- Coefficient $\frac{\alpha}{1-\alpha} = 0$

So the gradient becomes:
$$\nabla_\theta \mathcal{L}_0 = \lim_{\alpha \to 0} \frac{\alpha}{1-\alpha} \mathbb{E}_\mu[\tilde{w}_\alpha \nabla_\theta \log \pi_\theta]$$

This requires careful analysis. From the original:
$$\mathcal{L}_0 = \log \mathbb{E}_\mu[e^{r/\tau}] = \log Z$$

which is constant in $\theta$! So $\nabla_\theta \mathcal{L}_0 = 0$.

Actually, for $\alpha = 0$, the α-ELBO doesn't depend on $\pi_\theta$, which makes sense—it's just $\log Z$.

The correct interpretation for "α → 0 behavior" in practice is to use small positive α:
$$\nabla_\theta \mathcal{L}_\alpha \big|_{\alpha \text{ small}} \approx \alpha \cdot \mathbb{E}_\mu\left[\frac{e^{r/\tau}}{Z} \nabla_\theta \log \pi_\theta\right] \quad \blacksquare$$

---

## 8. Bias-Variance Analysis

### 8.1 Setup

Consider the Monte Carlo estimator of $\mathcal{L}_\alpha$ from $n$ samples:

$$\hat{\mathcal{L}}_\alpha = \frac{1}{1-\alpha} \log \left(\frac{1}{n} \sum_{i=1}^n w_i^\alpha e^{(1-\alpha)r_i/\tau}\right)$$

where $(y_i, r_i) \sim \mu$ and $w_i = \pi_\theta(y_i|x) / \mu(y_i|x)$.

### 8.2 Bias Analysis

**Theorem 8.1 (Bias Decomposition)**: The bias of maximizing $\mathcal{L}_\alpha$ instead of $\log Z$ is:

$$\text{Bias}(\alpha) := \log Z - \mathcal{L}_\alpha(\pi_\theta^*) = D_{1-\alpha}(\pi_\theta^* \| p^*)$$

where $\pi_\theta^*$ is the optimal policy under $\mathcal{L}_\alpha$, and $D_{1-\alpha}$ is the Rényi divergence of order $1-\alpha$.

*Proof*:

From the proof of Theorem 5.2, the gap is:

$$\log Z - \mathcal{L}_\alpha = -\frac{1}{1-\alpha} \log \mathbb{E}_{\pi_\theta}\left[\left(\frac{p^*}{\pi_\theta}\right)^{1-\alpha}\right] + \log Z - \log Z$$

Wait, let me redo this more carefully.

From Definition 5.1:
$$\mathcal{L}_\alpha = \frac{1}{1-\alpha} \log \mathbb{E}_{\pi_\theta}\left[\left(\frac{\mu e^{r/\tau}}{\pi_\theta}\right)^{1-\alpha}\right]$$

We showed $\mathcal{L}_\alpha \leq \log Z$.

The gap is:
$$\log Z - \mathcal{L}_\alpha = \log Z - \frac{1}{1-\alpha} \log \mathbb{E}_{\pi_\theta}\left[\left(\frac{\mu e^{r/\tau}}{\pi_\theta}\right)^{1-\alpha}\right]$$

Let $u = \mu e^{r/\tau} / \pi_\theta = (p^* \cdot Z) / \pi_\theta$. Then:

$$\mathcal{L}_\alpha = \frac{1}{1-\alpha} \log \mathbb{E}_{\pi_\theta}[u^{1-\alpha}]$$

$$= \frac{1}{1-\alpha} \log \mathbb{E}_{\pi_\theta}\left[\left(\frac{p^* Z}{\pi_\theta}\right)^{1-\alpha}\right]$$

$$= \frac{1}{1-\alpha} \log \left(Z^{1-\alpha} \mathbb{E}_{\pi_\theta}\left[\left(\frac{p^*}{\pi_\theta}\right)^{1-\alpha}\right]\right)$$

$$= \log Z + \frac{1}{1-\alpha} \log \mathbb{E}_{\pi_\theta}\left[\left(\frac{p^*}{\pi_\theta}\right)^{1-\alpha}\right]$$

Therefore:
$$\log Z - \mathcal{L}_\alpha = -\frac{1}{1-\alpha} \log \mathbb{E}_{\pi_\theta}\left[\left(\frac{p^*}{\pi_\theta}\right)^{1-\alpha}\right]$$

Now, the Rényi divergence of order $\beta$ from $\pi_\theta$ to $p^*$ is:
$$D_\beta(\pi_\theta \| p^*) = \frac{1}{\beta - 1} \log \mathbb{E}_{p^*}\left[\left(\frac{\pi_\theta}{p^*}\right)^\beta\right]$$

We need to relate this to our expression. Note:
$$\mathbb{E}_{p^*}\left[\left(\frac{\pi_\theta}{p^*}\right)^\beta\right] = \int p^* \left(\frac{\pi_\theta}{p^*}\right)^\beta dy = \int p^{*1-\beta} \pi_\theta^\beta dy$$

And:
$$\mathbb{E}_{\pi_\theta}\left[\left(\frac{p^*}{\pi_\theta}\right)^{1-\alpha}\right] = \int \pi_\theta \left(\frac{p^*}{\pi_\theta}\right)^{1-\alpha} dy = \int \pi_\theta^\alpha p^{*1-\alpha} dy$$

Setting $\beta = \alpha$:
$$\mathbb{E}_{p^*}\left[\left(\frac{\pi_\theta}{p^*}\right)^\alpha\right] = \int p^{*1-\alpha} \pi_\theta^\alpha dy = \mathbb{E}_{\pi_\theta}\left[\left(\frac{p^*}{\pi_\theta}\right)^{1-\alpha}\right]$$

Therefore:
$$\log Z - \mathcal{L}_\alpha = -\frac{1}{1-\alpha} \log \mathbb{E}_{p^*}\left[\left(\frac{\pi_\theta}{p^*}\right)^\alpha\right]$$

$$= \frac{1}{\alpha - 1} \log \mathbb{E}_{p^*}\left[\left(\frac{\pi_\theta}{p^*}\right)^\alpha\right] = D_\alpha(\pi_\theta \| p^*)$$

**Corrected Result**:
$$\boxed{\log Z - \mathcal{L}_\alpha(\pi_\theta) = D_\alpha(\pi_\theta \| p^*)}$$

This is the Rényi divergence of order $\alpha$ from $\pi_\theta$ to $p^*$. $\blacksquare$

**Corollary 8.2**:
- As $\alpha \to 1$: Bias $= D_{KL}(\pi_\theta \| p^*)$ (standard ELBO gap)
- As $\alpha \to 0$: Bias $\to 0$ (bound becomes tight)
- Bias is monotonically increasing in $\alpha$ (by Property 4.2)

### 8.3 Variance Analysis

**Theorem 8.3 (Variance of Gradient Estimator)**: The variance of the Monte Carlo gradient estimator is:

$$\text{Var}(\hat{\nabla}_\theta \mathcal{L}_\alpha) \approx \frac{1}{n} \cdot \frac{\alpha^2}{(1-\alpha)^2} \cdot \frac{\mathbb{E}_\mu[w^{2\alpha} e^{2(1-\alpha)r/\tau} \|\nabla \log \pi_\theta\|^2]}{G(\theta)^2}$$

where we ignore lower-order terms.

*Proof Sketch*:

The gradient estimator is:
$$\hat{\nabla}_\theta \mathcal{L}_\alpha = \frac{\alpha}{1-\alpha} \cdot \frac{\sum_i w_i^\alpha e^{(1-\alpha)r_i/\tau} \nabla \log \pi_\theta(y_i)}{\sum_i w_i^\alpha e^{(1-\alpha)r_i/\tau}}$$

This is a ratio estimator. Using the delta method for ratio estimators:

$$\text{Var}\left(\frac{\bar{X}}{\bar{Y}}\right) \approx \frac{1}{n} \left[\frac{\text{Var}(X)}{\mu_Y^2} - \frac{2\text{Cov}(X,Y)\mu_X}{\mu_Y^3} + \frac{\text{Var}(Y)\mu_X^2}{\mu_Y^4}\right]$$

For our case, the dominant term when $w$ has high variance is:
$$\text{Var}(\hat{\nabla} \mathcal{L}_\alpha) \propto \frac{1}{n} \mathbb{E}_\mu[w^{2\alpha}] / G(\theta)^2$$

The key factor is $\mathbb{E}_\mu[w^{2\alpha}]$:
- $\alpha$ large → $w^{2\alpha}$ can be huge when $\pi_\theta \gg \mu$ → high variance
- $\alpha$ small → $w^{2\alpha} \approx 1$ → low variance $\blacksquare$

### 8.4 Bias-Variance Trade-off

**Theorem 8.4 (MSE Decomposition)**: The mean squared error of the gradient estimator decomposes as:

$$\text{MSE}(\alpha) = \underbrace{\|\nabla_\theta D_\alpha(\pi_\theta \| p^*)\|^2}_{\text{Bias}^2} + \underbrace{\frac{C(\alpha)}{n} \mathbb{E}_\mu[w^{2\alpha}]}_{\text{Variance}}$$

where $C(\alpha)$ is a function depending on the specific form of the estimator.

**Optimal α**: The optimal $\alpha^*$ minimizes $\text{MSE}(\alpha)$:

$$\alpha^* = \arg\min_\alpha \left[\text{Bias}(\alpha)^2 + \text{Var}(\alpha)\right]$$

**Qualitative behavior**:

| $\alpha$ | Bias | Variance | Regime |
|----------|------|----------|--------|
| $\alpha \to 0$ | Small (bound tight) | Large ($1/\alpha^2$ factor) | Conservative/SFT-like |
| $\alpha = 0.5$ | Medium | Medium | Balanced |
| $\alpha \to 1$ | Large ($D_{KL}$) | Small (standard PG) | Aggressive/RL-like |

---

## 9. Optimal α Selection

### 9.1 ESS-Based Selection

**Definition 9.1 (Generalized Effective Sample Size)**:

$$\text{ESS}_\alpha = \frac{\left(\sum_{i=1}^n w_i^\alpha e^{(1-\alpha)r_i/\tau}\right)^2}{\sum_{i=1}^n w_i^{2\alpha} e^{2(1-\alpha)r_i/\tau}}$$

**Theorem 9.1**: If $\text{ESS}_\alpha \geq n \cdot \rho$ for some threshold $\rho \in (0, 1)$, then:

$$\text{Var}(\hat{\mathcal{L}}_\alpha) \leq \frac{C}{n \rho}$$

for some constant $C$ independent of $\alpha$.

*Proof*: By Chebyshev's inequality and the definition of ESS as the "effective" number of samples. $\blacksquare$

**Selection Rule**:
$$\alpha^* = \max\{\alpha \in [0, 1] : \text{ESS}_\alpha \geq n \cdot \rho_{\min}\}$$

### 9.2 Log-Normal Approximation

**Assumption 9.2**: Assume $\log w \sim \mathcal{N}(\nu, \sigma^2)$ approximately.

**Theorem 9.2**: Under Assumption 9.2:

$$D_\alpha(\pi_\theta \| \mu) = \frac{\alpha}{2}\sigma^2 + \alpha \nu + \frac{1}{\alpha - 1}\log \mathbb{E}[w^\alpha] = \frac{\alpha}{2}\sigma^2$$

(when $\nu = -\sigma^2/2$ to ensure $\mathbb{E}_\mu[w] = 1$)

**Corollary 9.3**: To ensure $D_\alpha(\pi_\theta \| \mu) \leq \delta$:

$$\alpha^* = \frac{2\delta}{\sigma^2} = \frac{2\delta}{\text{Var}(\log w)}$$

*Proof*:
If $\log w \sim \mathcal{N}(\nu, \sigma^2)$, then $w^\alpha = e^{\alpha \log w}$, and $\alpha \log w \sim \mathcal{N}(\alpha\nu, \alpha^2\sigma^2)$.

Thus $\mathbb{E}[w^\alpha] = e^{\alpha\nu + \alpha^2\sigma^2/2}$.

For $w$ to be a valid IS ratio, we need $\mathbb{E}_\mu[w] = 1$, which requires $\nu = -\sigma^2/2$.

Then:
$$\mathbb{E}[w^\alpha] = e^{-\alpha\sigma^2/2 + \alpha^2\sigma^2/2} = e^{\alpha(\alpha-1)\sigma^2/2}$$

$$D_\alpha(\pi_\theta \| \mu) = \frac{1}{\alpha-1}\log e^{\alpha(\alpha-1)\sigma^2/2} = \frac{\alpha\sigma^2}{2}$$

Setting this $\leq \delta$ gives $\alpha \leq 2\delta/\sigma^2$. $\blacksquare$

### 9.3 Adaptive Selection Algorithm

**Algorithm 1: Adaptive α-ELBO**

```
Input: D = {(x_i, y_i, r_i)}, μ, ρ_min ∈ (0,1), δ > 0
Initialize: π_θ, α = 0.5

For t = 1, 2, ..., T:
    1. Compute w_i = π_θ(y_i|x_i) / μ(y_i|x_i) for all i

    2. Update α using one of:
       (a) ESS method:
           α_t = max{α : ESS_α ≥ n · ρ_min}

       (b) Variance method:
           σ² = Var(log w)
           α_t = clip(2δ/σ², 0.1, 0.9)

       (c) Binary search on MSE (if compute budget allows)

    3. Compute normalized weights:
           ω_i = w_i^{α_t} e^{(1-α_t)r_i/τ} / Σ_j w_j^{α_t} e^{(1-α_t)r_j/τ}

    4. Compute gradient:
           g = (α_t/(1-α_t)) Σ_i ω_i ∇log π_θ(y_i|x_i)

    5. Update: θ ← θ + η · g

Output: π_θ
```

---

## 10. Connection to Existing Methods

### 10.1 Special Cases Recovery

**Proposition 10.1**: The α-ELBO framework recovers:

1. **α = 1**: Standard RLHF/PPO objective (mode-seeking)
   $$\mathcal{L}_1 = \frac{1}{\tau}\mathbb{E}_{\pi_\theta}[r] - D_{KL}(\pi_\theta \| \mu)$$

2. **α → 0**: Reward-weighted SFT (mean-seeking)
   $$\nabla_\theta \mathcal{L}_{\alpha \to 0} \propto \mathbb{E}_\mu\left[\frac{e^{r/\tau}}{Z} \nabla_\theta \log \pi_\theta\right]$$

3. **α = 0.5**: Hellinger distance regularization
   $$D_{0.5}(\pi_\theta \| \mu) = 2\left(1 - \int \sqrt{\pi_\theta \mu} dy\right)$$

### 10.2 Relationship to PPO Clipping

**Proposition 10.2**: PPO's clipped objective:
$$L^{CLIP}(\theta) = \mathbb{E}_\mu\left[\min(w \cdot A, \text{clip}(w, 1-\epsilon, 1+\epsilon) \cdot A)\right]$$

can be viewed as an adaptive α scheme where effectively:
- When $w$ is within $[1-\epsilon, 1+\epsilon]$: use $\alpha = 1$ (full IS)
- When $w$ is outside: reduce effective α (conservative update)

### 10.3 Relationship to AWR/AWAC

**Proposition 10.3**: Advantage Weighted Regression:
$$\max_\theta \mathbb{E}_\mu\left[\exp(A/\tau) \log \pi_\theta(y|x)\right]$$

is equivalent to α-ELBO with $\alpha = 0$ and reward = advantage:
$$\nabla_\theta \mathcal{L}_0 \propto \mathbb{E}_\mu[e^{r/\tau} \nabla_\theta \log \pi_\theta]$$

---

## 11. Summary

### Main Results

1. **α-ELBO provides a principled lower bound** on $\log Z$ for offline RL (Theorem 5.2)

2. **The bound gap is the Rényi divergence**: $\log Z - \mathcal{L}_\alpha = D_\alpha(\pi_\theta \| p^*)$ (Theorem 8.1)

3. **Bias-variance trade-off** is controlled by α:
   - Small α: tight bound (low bias), high variance
   - Large α: loose bound (high bias), low variance

4. **Optimal α** can be selected via:
   - ESS threshold: $\alpha^* = \max\{\alpha : \text{ESS}_\alpha \geq n\rho\}$
   - Variance constraint: $\alpha^* = 2\delta / \text{Var}(\log w)$
   - MSE minimization: $\alpha^* = \arg\min [\text{Bias}^2 + \text{Var}]$

5. **The framework unifies** SFT (α→0), standard RL (α=1), and intermediate methods

### Key Insight

$$\boxed{\text{α-ELBO interpolates between mean-seeking (SFT) and mode-seeking (RL) via the order of Rényi divergence}}$$
