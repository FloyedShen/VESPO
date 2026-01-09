"""
Alpha Computer for SGLang Implementation

Simplified version that works with SGLang's control flow.
Reuses logic from vLLM implementation.
"""

import torch
import torch.nn.functional as F
from typing import Literal


class AlphaComputer:
    """
    Compute alpha for optimal sampling distribution

    Methods:
    - fixed: Constant alpha
    - kl_symmetry: KL(q||θ) = KL(q||t)
    - ess_balance: ESS_θ(q) = ESS_t(q)
    - entropy: H(θ) / (H(θ) + H(t))
    """

    def __init__(
        self,
        method: Literal["fixed", "kl_symmetry", "ess_balance", "entropy"] = "kl_symmetry",
        fixed_alpha: float = 0.5,
        alpha_min: float = 0.5,
        alpha_max: float = 1.0,
        tol: float = 1e-6,
        max_iter: int = 12,
        eps: float = 1e-10
    ):
        self.method = method
        self.fixed_alpha = fixed_alpha
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.tol = tol
        self.max_iter = max_iter
        self.eps = eps

    def compute(
        self,
        probs_theta: torch.Tensor,
        probs_t: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute alpha for a batch

        Args:
            probs_theta: [batch_size, vocab_size] Base model probabilities
            probs_t: [batch_size, vocab_size] Teacher model probabilities

        Returns:
            alpha: [batch_size] Alpha values
        """
        if self.method == "fixed":
            return self._fixed(probs_theta)
        elif self.method == "kl_symmetry":
            return self._kl_symmetry(probs_theta, probs_t)
        elif self.method == "ess_balance":
            return self._ess_balance(probs_theta, probs_t)
        elif self.method == "entropy":
            return self._entropy(probs_theta, probs_t)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _fixed(self, probs_theta: torch.Tensor) -> torch.Tensor:
        batch_size = probs_theta.shape[0]
        return torch.full(
            (batch_size,),
            self.fixed_alpha,
            device=probs_theta.device,
            dtype=probs_theta.dtype
        )

    def _kl_symmetry(
        self,
        probs_theta: torch.Tensor,
        probs_t: torch.Tensor
    ) -> torch.Tensor:
        """KL symmetry: D_KL(q||θ) = D_KL(q||t)"""
        batch_size = probs_theta.shape[0]
        device = probs_theta.device

        # Clamp for stability
        probs_theta = torch.clamp(probs_theta, min=self.eps, max=1.0)
        probs_t = torch.clamp(probs_t, min=self.eps, max=1.0)

        # Check if distributions are identical
        if (probs_theta - probs_t).abs().max() < 1e-6:
            mid = (self.alpha_min + self.alpha_max) / 2
            return torch.full((batch_size,), mid, device=device, dtype=probs_theta.dtype)

        # Precompute log ratio
        log_ratio = torch.log(probs_t) - torch.log(probs_theta)

        # Bisection search
        alpha_low = torch.full((batch_size,), self.alpha_min, device=device)
        alpha_high = torch.full((batch_size,), self.alpha_max, device=device)

        for _ in range(self.max_iter):
            alpha_mid = (alpha_low + alpha_high) / 2
            alpha_expanded = alpha_mid.unsqueeze(-1)

            # Compute q in log space
            log_q_unnorm = (1 - alpha_expanded) * torch.log(probs_theta) + \
                          alpha_expanded * torch.log(probs_t)
            log_q = log_q_unnorm - torch.logsumexp(log_q_unnorm, dim=-1, keepdim=True)
            q = torch.exp(log_q)

            # Delta = E_q[log(t/θ)]
            delta = (q * log_ratio).sum(dim=-1)

            # Update bounds
            mask = delta > 0
            alpha_high = torch.where(mask, alpha_mid, alpha_high)
            alpha_low = torch.where(mask, alpha_low, alpha_mid)

            if (alpha_high - alpha_low).max() < self.tol:
                break

        alpha = (alpha_low + alpha_high) / 2
        return torch.clamp(alpha, min=self.alpha_min, max=self.alpha_max)

    def _ess_balance(
        self,
        probs_theta: torch.Tensor,
        probs_t: torch.Tensor
    ) -> torch.Tensor:
        """ESS balance: ESS_θ(q) = ESS_t(q)"""
        batch_size = probs_theta.shape[0]
        device = probs_theta.device

        probs_theta = torch.clamp(probs_theta, min=self.eps, max=1.0)
        probs_t = torch.clamp(probs_t, min=self.eps, max=1.0)

        alpha_low = torch.full((batch_size,), self.alpha_min, device=device)
        alpha_high = torch.full((batch_size,), self.alpha_max, device=device)

        for _ in range(self.max_iter):
            alpha_mid = (alpha_low + alpha_high) / 2
            q = self._geometric_mean(probs_theta, probs_t, alpha_mid)

            sum_theta_sq = ((probs_theta ** 2) / (q + self.eps)).sum(dim=-1)
            sum_t_sq = ((probs_t ** 2) / (q + self.eps)).sum(dim=-1)
            delta = sum_t_sq - sum_theta_sq

            mask = delta < 0
            alpha_low = torch.where(mask, alpha_mid, alpha_low)
            alpha_high = torch.where(mask, alpha_high, alpha_mid)

            if (alpha_high - alpha_low).max() < self.tol:
                break

        alpha = (alpha_low + alpha_high) / 2
        return torch.clamp(alpha, min=self.alpha_min, max=self.alpha_max)

    def _entropy(
        self,
        probs_theta: torch.Tensor,
        probs_t: torch.Tensor
    ) -> torch.Tensor:
        """Entropy heuristic: α = H(θ) / (H(θ) + H(t))"""
        h_theta = -(probs_theta * torch.log(probs_theta + self.eps)).sum(dim=-1)
        h_t = -(probs_t * torch.log(probs_t + self.eps)).sum(dim=-1)
        alpha = h_theta / (h_theta + h_t + self.eps)
        return torch.clamp(alpha, min=self.alpha_min, max=self.alpha_max)

    def _geometric_mean(
        self,
        p1: torch.Tensor,
        p2: torch.Tensor,
        alpha: torch.Tensor
    ) -> torch.Tensor:
        """Geometric mean: q = p1^(1-α) × p2^α"""
        if alpha.dim() == 1:
            alpha = alpha.unsqueeze(-1)
        log_q = (1 - alpha) * torch.log(p1 + self.eps) + alpha * torch.log(p2 + self.eps)
        return F.softmax(log_q, dim=-1)

    def compute_q_star(
        self,
        probs_theta: torch.Tensor,
        probs_t: torch.Tensor,
        alpha: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute optimal sampling distribution q*

        Args:
            probs_theta: [batch_size, vocab_size]
            probs_t: [batch_size, vocab_size]
            alpha: [batch_size] or float

        Returns:
            q_star: [batch_size, vocab_size]
        """
        if isinstance(alpha, float):
            alpha = torch.full(
                (probs_theta.shape[0],),
                alpha,
                device=probs_theta.device
            )

        return self._geometric_mean(probs_theta, probs_t, alpha)
