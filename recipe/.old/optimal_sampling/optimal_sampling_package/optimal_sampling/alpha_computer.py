"""
Alpha Computer for vLLM Implementation

Simplified and optimized version for computing alpha in real-time during generation.
Reuses core logic from the transformers implementation.
"""

import torch
import torch.nn.functional as F
from typing import Literal


class AlphaComputer:
    """
    Lightweight alpha computer for vLLM implementation

    Optimized for:
    - Low latency (called every generation step)
    - Numerical stability
    - Batch processing

    Supports the same alpha methods as the transformers implementation:
    - fixed: Constant alpha
    - kl_symmetry: KL(q||θ) = KL(q||t)
    - ess_balance: ESS_θ(q) = ESS_t(q) (exact condition)
    - entropy: Heuristic based on entropy ratio
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
        """
        Args:
            method: Alpha computation method
            fixed_alpha: Fixed alpha value (when method="fixed")
            alpha_min: Minimum alpha (Teacher weight lower bound)
            alpha_max: Maximum alpha (Teacher weight upper bound)
            tol: Convergence tolerance for iterative methods
            max_iter: Maximum iterations for bisection
            eps: Small constant for numerical stability
        """
        self.method = method
        self.fixed_alpha = fixed_alpha
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.tol = tol
        self.max_iter = max_iter
        self.eps = eps

        print(f"   - Alpha method: {method}")
        print(f"   - Alpha range: [{alpha_min:.2f}, {alpha_max:.2f}]")

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
            alpha: [batch_size] Alpha values (Teacher weight)
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
            raise ValueError(f"Unknown alpha method: {self.method}")

    def _fixed(self, probs_theta: torch.Tensor) -> torch.Tensor:
        """Fixed alpha"""
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
        """
        KL symmetry: D_KL(q||θ) = D_KL(q||t)
        Equivalent to: E_q[log(t/θ)] = 0

        Solves via bisection in [alpha_min, alpha_max]
        """
        batch_size = probs_theta.shape[0]
        device = probs_theta.device

        # Clamp inputs for numerical stability
        probs_theta = torch.clamp(probs_theta, min=self.eps, max=1.0)
        probs_t = torch.clamp(probs_t, min=self.eps, max=1.0)

        # Check if distributions are nearly identical
        max_diff = (probs_theta - probs_t).abs().max()
        if max_diff < 1e-6:
            mid_alpha = (self.alpha_min + self.alpha_max) / 2
            return torch.full((batch_size,), mid_alpha, device=device, dtype=probs_theta.dtype)

        # Precompute log ratio
        log_probs_theta = torch.log(probs_theta)
        log_probs_t = torch.log(probs_t)
        log_ratio = log_probs_t - log_probs_theta  # log(t/θ)

        # Bisection search
        alpha_low = torch.full((batch_size,), self.alpha_min, device=device)
        alpha_high = torch.full((batch_size,), self.alpha_max, device=device)

        for _ in range(self.max_iter):
            alpha_mid = (alpha_low + alpha_high) / 2
            alpha_expanded = alpha_mid.unsqueeze(-1)  # [batch, 1]

            # Compute q_alpha in log space
            log_q_unnorm = (1 - alpha_expanded) * log_probs_theta + alpha_expanded * log_probs_t
            log_q = log_q_unnorm - torch.logsumexp(log_q_unnorm, dim=-1, keepdim=True)
            q_alpha = torch.exp(log_q)

            # Compute delta = E_q[log(t/θ)]
            delta = (q_alpha * log_ratio).sum(dim=-1)

            # Update bounds (note: delta > 0 means q is closer to t, need to reduce alpha)
            mask = delta > 0
            alpha_high = torch.where(mask, alpha_mid, alpha_high)
            alpha_low = torch.where(mask, alpha_low, alpha_mid)

            # Check convergence
            if (alpha_high - alpha_low).max() < self.tol:
                break

        alpha_result = (alpha_low + alpha_high) / 2
        return torch.clamp(alpha_result, min=self.alpha_min, max=self.alpha_max)

    def _ess_balance(
        self,
        probs_theta: torch.Tensor,
        probs_t: torch.Tensor
    ) -> torch.Tensor:
        """
        ESS balance: ESS_θ(q) = ESS_t(q)
        Equivalent to: Σ(θ²/q) = Σ(t²/q)

        This is the exact theoretical condition (KL symmetry is an approximation)
        """
        batch_size = probs_theta.shape[0]
        device = probs_theta.device

        # Clamp inputs
        probs_theta = torch.clamp(probs_theta, min=self.eps, max=1.0)
        probs_t = torch.clamp(probs_t, min=self.eps, max=1.0)

        # Bisection search
        alpha_low = torch.full((batch_size,), self.alpha_min, device=device)
        alpha_high = torch.full((batch_size,), self.alpha_max, device=device)

        for _ in range(self.max_iter):
            alpha_mid = (alpha_low + alpha_high) / 2

            # Compute q_alpha
            q_alpha = self._geometric_mean(probs_theta, probs_t, alpha_mid)

            # Compute ESS difference
            sum_theta_sq = ((probs_theta ** 2) / (q_alpha + self.eps)).sum(dim=-1)
            sum_t_sq = ((probs_t ** 2) / (q_alpha + self.eps)).sum(dim=-1)
            delta = sum_t_sq - sum_theta_sq

            # Update bounds
            mask = delta < 0
            alpha_low = torch.where(mask, alpha_mid, alpha_low)
            alpha_high = torch.where(mask, alpha_high, alpha_mid)

            if (alpha_high - alpha_low).max() < self.tol:
                break

        alpha_result = (alpha_low + alpha_high) / 2
        return torch.clamp(alpha_result, min=self.alpha_min, max=self.alpha_max)

    def _entropy(
        self,
        probs_theta: torch.Tensor,
        probs_t: torch.Tensor
    ) -> torch.Tensor:
        """
        Entropy-based heuristic: alpha = H(θ) / (H(θ) + H(t))

        Intuition:
        - High H(θ) → Base is uncertain → rely more on Teacher → high alpha
        - High H(t) → Teacher is uncertain → rely more on Base → low alpha
        """
        h_theta = -(probs_theta * torch.log(probs_theta + self.eps)).sum(dim=-1)
        h_t = -(probs_t * torch.log(probs_t + self.eps)).sum(dim=-1)
        alpha = h_theta / (h_theta + h_t + self.eps)
        return torch.clamp(alpha, min=self.alpha_min, max=self.alpha_max)

    def compute_q_star(
        self,
        probs_theta: torch.Tensor,
        probs_t: torch.Tensor,
        alpha: float
    ) -> torch.Tensor:
        """
        Compute the optimal distribution q* given alpha

        q* = (π_θ)^(1-α) × (π_t)^α / Z

        Args:
            probs_theta: [batch_size, vocab_size] Base model probabilities
            probs_t: [batch_size, vocab_size] Teacher model probabilities
            alpha: Teacher weight (can be scalar or tensor)

        Returns:
            q_star: [batch_size, vocab_size] Optimal distribution
        """
        # Convert alpha to tensor if needed
        if isinstance(alpha, (int, float)):
            batch_size = probs_theta.shape[0]
            alpha = torch.full(
                (batch_size,),
                alpha,
                device=probs_theta.device,
                dtype=probs_theta.dtype
            )

        return self._geometric_mean(probs_theta, probs_t, alpha)

    def _geometric_mean(
        self,
        p1: torch.Tensor,
        p2: torch.Tensor,
        alpha: torch.Tensor
    ) -> torch.Tensor:
        """
        Geometric mean: q = p1^(1-α) × p2^α

        Where:
        - p1 = π_θ (Base)
        - p2 = π_t (Teacher)
        - α = Teacher weight
        """
        if alpha.dim() == 1:
            alpha = alpha.unsqueeze(-1)
        log_q = (1 - alpha) * torch.log(p1 + self.eps) + alpha * torch.log(p2 + self.eps)
        return F.softmax(log_q, dim=-1)
