# Copyright 2024 IS Reshape Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
IS Reshape Core Algorithms

This module contains the IS Reshape policy loss implementation.
Import this module to register the "is_reshape" policy loss with veRL.

Usage:
    # Import to register the policy loss
    import recipe.is_shape.code.core_algos

    # Then use loss_mode="is_reshape" in config
"""

import math
from typing import Any, Optional

import torch

import verl.utils.torch_functional as verl_F
from verl.trainer.ppo.core_algos import agg_loss, register_policy_loss
from verl.workers.config import ActorConfig

__all__ = [
    "compute_is_reshape_gamma",
    "compute_is_reshape_gamma_per_sample",
    "compute_is_reshape_gamma_v7",
    "compute_policy_loss_is_reshape",
    "compute_policy_loss_is_reshape_pos",
    "compute_policy_loss_is_reshape_per_sample",
    "compute_policy_loss_is_reshape_renyi",
    "compute_policy_loss_is_reshape_v4",
    "compute_policy_loss_is_reshape_v5",
    "compute_policy_loss_is_reshape_v7",
    "compute_policy_loss_is_reshape_sym",
    "compute_policy_loss_is_reshape_static",
    "compute_policy_loss_is_reshape_harmonic",
    "compute_policy_loss_grpo_clip_asymmetric",
    "compute_policy_loss_sapo",
    "compute_policy_loss_sapo_is",
    "compute_policy_loss_sapo_is_mono",
    "compute_policy_loss_ib_is",
    "compute_policy_loss_harmonic_is",
    "compute_policy_loss_cauchy_is",
    "compute_policy_loss_welsch_is",
    "compute_policy_loss_gamma_is",
    "compute_policy_loss_gamma_is_adaptive",
    "compute_policy_loss_gamma_is_amplitude"
]


def compute_is_reshape_gamma(
    log_w: torch.Tensor,
    response_mask: torch.Tensor,
    rho_min: float = 0.3,
    gamma_min: float = 0.05,
    gamma_max: float = 1.0,
    gamma_fixed: Optional[float] = None,
) -> tuple[float, dict[str, Any]]:
    """Compute optimal gamma for IS Reshape using closed-form solution.

    Based on theory: γ* = min(γ_max, √(-log ρ_min / σ²))
    where σ² = Var(log w) and ρ_min is the minimum ESS ratio.

    Args:
        log_w: Log importance weights (log π_θ - log μ), shape (batch_size, seq_length)
            where μ is the behavior/reference policy (old_log_prob in the loss function)
        response_mask: Valid token mask, shape (batch_size, seq_length)
        rho_min: Minimum ESS ratio threshold (default: 0.3)
        gamma_min: Minimum γ value to avoid 1/γ explosion (default: 0.05)
        gamma_max: Maximum γ value (default: 1.0)
        gamma_fixed: If provided, use this fixed gamma instead of computing

    Returns:
        gamma: Optimal gamma value in [gamma_min, gamma_max]
        metrics: Dictionary containing diagnostic metrics

    Note:
        - γ=0 corresponds to SFT (Forward KL, mean-seeking)
        - γ=1 corresponds to RL (Reverse KL, mode-seeking)
        - gamma_min > 0 is required because the objective L_γ = (1/γ)E[w^γ r]
          has a 1/γ factor that explodes as γ → 0
    """
    with torch.no_grad():
        # If fixed gamma is provided, use it directly
        if gamma_fixed is not None:
            gamma = max(gamma_min, min(gamma_max, gamma_fixed))
            return gamma, {
                "is_reshape/gamma": gamma,
            }

        # Flatten and mask log_w
        log_w_flat = log_w[response_mask > 0]

        if log_w_flat.numel() == 0:
            return gamma_max, {
                "is_reshape/gamma": gamma_max,
                "is_reshape/sigma_sq": 0.0,
            }

        # Compute variance of log weights: σ² = Var(log w)
        sigma_sq = torch.var(log_w_flat, unbiased=True).item()

        # Handle edge cases
        if sigma_sq < 1e-8:
            # If variance is nearly zero, π_θ ≈ μ (on-policy), so we can use γ = γ_max (full IS)
            return gamma_max, {
                "is_reshape/gamma": gamma_max,
                "is_reshape/sigma_sq": sigma_sq,
                "is_reshape/ess_ratio": 1.0,
            }

        # Closed-form solution: γ* = min(γ_max, √(-log ρ_min / σ²))
        gamma = min(gamma_max, math.sqrt(-math.log(rho_min) / sigma_sq))
        gamma = max(gamma_min, gamma)  # Ensure γ >= γ_min to avoid 1/γ explosion

        # Compute effective sample size ratio for diagnostics
        # ESS/n ≈ exp(-σ² γ²)
        ess_ratio = math.exp(-sigma_sq * gamma * gamma)

        metrics = {
            "is_reshape/gamma": gamma,
            "is_reshape/sigma_sq": sigma_sq,
            "is_reshape/ess_ratio": ess_ratio,
        }

        return gamma, metrics


def compute_is_reshape_gamma_per_sample(
    log_w: torch.Tensor,
    response_mask: torch.Tensor,
    rho_min: float = 0.3,
    gamma_min: float = 0.05,
    gamma_max: float = 1.0,
    gamma_fixed: Optional[float] = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute optimal gamma for IS Reshape per sample (sequence).

    Unlike compute_is_reshape_gamma which computes a single batch-level gamma,
    this function computes a separate gamma for each sample based on the
    variance of log_w within that sample.

    Theory:
        Per-sample ESS constraint: ESS_i / n_i >= rho_min
        where ESS_i and n_i are for sample i.
        γ*_i = min(γ_max, √(-log ρ_min / σ²_i))
        where σ²_i = Var(log w) within sample i.

    Args:
        log_w: Log importance weights, shape (batch_size, seq_length)
        response_mask: Valid token mask, shape (batch_size, seq_length)
        rho_min: Minimum ESS ratio threshold (default: 0.3)
        gamma_min: Minimum γ value to avoid 1/γ explosion (default: 0.05)
        gamma_max: Maximum γ value (default: 1.0)
        gamma_fixed: If provided, use this fixed gamma for all samples

    Returns:
        gamma: Per-sample gamma values, shape (batch_size,)
        metrics: Dictionary containing diagnostic metrics
    """
    batch_size = log_w.shape[0]
    device = log_w.device
    dtype = log_w.dtype

    with torch.no_grad():
        # If fixed gamma is provided, use it for all samples
        if gamma_fixed is not None:
            gamma_val = max(gamma_min, min(gamma_max, gamma_fixed))
            gamma = torch.full((batch_size,), gamma_val, device=device, dtype=dtype)
            return gamma, {
                "is_reshape/gamma_mean": gamma_val,
                "is_reshape/gamma_std": 0.0,
            }

        # Compute per-sample variance of log_w
        gamma_list = []
        sigma_sq_list = []

        for i in range(batch_size):
            mask_i = response_mask[i] > 0
            log_w_i = log_w[i][mask_i]

            if log_w_i.numel() <= 1:
                # Not enough tokens, use gamma_max
                gamma_list.append(gamma_max)
                sigma_sq_list.append(0.0)
                continue

            # Compute variance within this sample
            sigma_sq_i = torch.var(log_w_i, unbiased=True).item()

            if sigma_sq_i < 1e-8:
                # Near-zero variance, use gamma_max
                gamma_list.append(gamma_max)
                sigma_sq_list.append(sigma_sq_i)
                continue

            # Closed-form solution: γ*_i = min(γ_max, √(-log ρ_min / σ²_i))
            gamma_i = min(gamma_max, math.sqrt(-math.log(rho_min) / sigma_sq_i))
            gamma_i = max(gamma_min, gamma_i)

            gamma_list.append(gamma_i)
            sigma_sq_list.append(sigma_sq_i)

        gamma = torch.tensor(gamma_list, device=device, dtype=dtype)
        sigma_sq_arr = torch.tensor(sigma_sq_list, device=device, dtype=dtype)

        # Compute metrics
        metrics = {
            "is_reshape/gamma_mean": gamma.mean().item(),
            "is_reshape/gamma_std": gamma.std().item() if batch_size > 1 else 0.0,
            "is_reshape/gamma_min": gamma.min().item(),
            "is_reshape/gamma_max": gamma.max().item(),
            "is_reshape/sigma_sq_mean": sigma_sq_arr.mean().item(),
        }

        return gamma, metrics


def compute_is_reshape_gamma_v7(
    log_w: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    rho_min: float = 0.3,
    gamma_pos_min: float = 0.05,
    gamma_pos_max: float = 0.99,
    gamma_neg_min: float = 1.01,
    gamma_neg_max: float = 1.99,
    gamma_pos_fixed: Optional[float] = None,
    gamma_neg_fixed: Optional[float] = None,
) -> tuple[tuple[float, float], dict[str, Any]]:
    """Compute optimal gamma for IS Reshape v7 with separate γ+ and γ- for positive/negative advantages.

    This implements the v7 theory which proves that:
    1. MSE is convex on both (0,1) and (1,2) intervals, so minima exist on both
    2. Positive samples (A > 0) should use γ+ ∈ (0, 1) - concave function
    3. Negative samples (A < 0) should use γ- ∈ (1, 2) - convex function

    Theory:
        For positive samples (A > 0):
            - w < 1 (new good samples) → amplify weight to learn
            - w > 1 (known good samples) → reduce weight to lower variance
            - Need concave function (γ < 1)

        For negative samples (A < 0):
            - w < 1 (already avoided bad samples) → reduce weight (no need to punish)
            - w > 1 (not yet avoided bad samples) → amplify weight to punish
            - Need convex function (γ > 1)

    Closed-form solution:
        γ+ = min(γ_pos_max, √(-log ρ_min / σ_+²))
        γ- = 1 + min(γ_neg_max - 1, √(-log ρ_min / σ_-²))

    Args:
        log_w: Log importance weights, shape (batch_size, seq_length)
        advantages: Advantage values, shape (batch_size, seq_length)
        response_mask: Valid token mask, shape (batch_size, seq_length)
        rho_min: Minimum ESS ratio threshold (default: 0.3)
        gamma_pos_min: Minimum γ for positive samples (default: 0.05)
        gamma_pos_max: Maximum γ for positive samples (default: 0.99)
        gamma_neg_min: Minimum γ for negative samples (default: 1.01)
        gamma_neg_max: Maximum γ for negative samples (default: 1.99)
        gamma_pos_fixed: If provided, use this fixed gamma for positive samples
        gamma_neg_fixed: If provided, use this fixed gamma for negative samples

    Returns:
        (gamma_pos, gamma_neg): Tuple of gamma values for positive/negative samples
        metrics: Dictionary containing diagnostic metrics
    """
    with torch.no_grad():
        # Create masks for positive and negative advantages
        mask = response_mask > 0
        pos_mask = mask & (advantages > 0)
        neg_mask = mask & (advantages < 0)

        # Extract log_w for each group
        log_w_pos = log_w[pos_mask]
        log_w_neg = log_w[neg_mask]

        # Compute γ+ for positive samples
        if gamma_pos_fixed is not None:
            gamma_pos = max(gamma_pos_min, min(gamma_pos_max, gamma_pos_fixed))
            sigma_sq_pos = 0.0
        elif log_w_pos.numel() < 2:
            gamma_pos = (gamma_pos_min + gamma_pos_max) / 2  # Default to middle
            sigma_sq_pos = 0.0
        else:
            sigma_sq_pos = torch.var(log_w_pos, unbiased=True).item()
            if sigma_sq_pos < 1e-8:
                gamma_pos = gamma_pos_max  # Near on-policy, can use high gamma
            else:
                # γ+ = min(γ_max, √(-log ρ_min / σ²))
                gamma_pos = min(gamma_pos_max, math.sqrt(-math.log(rho_min) / sigma_sq_pos))
                gamma_pos = max(gamma_pos_min, gamma_pos)

        # Compute γ- for negative samples
        if gamma_neg_fixed is not None:
            gamma_neg = max(gamma_neg_min, min(gamma_neg_max, gamma_neg_fixed))
            sigma_sq_neg = 0.0
        elif log_w_neg.numel() < 2:
            gamma_neg = (gamma_neg_min + gamma_neg_max) / 2  # Default to middle
            sigma_sq_neg = 0.0
        else:
            sigma_sq_neg = torch.var(log_w_neg, unbiased=True).item()
            if sigma_sq_neg < 1e-8:
                gamma_neg = gamma_neg_min  # Near on-policy, conservative
            else:
                # γ- = 1 + min(γ_max - 1, √(-log ρ_min / σ²))
                delta_neg = min(gamma_neg_max - 1.0, math.sqrt(-math.log(rho_min) / sigma_sq_neg))
                gamma_neg = 1.0 + delta_neg
                gamma_neg = max(gamma_neg_min, gamma_neg)

        # Compute ESS ratios for diagnostics
        ess_ratio_pos = math.exp(-sigma_sq_pos * gamma_pos * gamma_pos) if sigma_sq_pos > 0 else 1.0
        ess_ratio_neg = math.exp(-sigma_sq_neg * (gamma_neg - 1) * (gamma_neg - 1)) if sigma_sq_neg > 0 else 1.0

        metrics = {
            "is_reshape_v7/gamma_pos": gamma_pos,
            "is_reshape_v7/gamma_neg": gamma_neg,
            "is_reshape_v7/sigma_sq_pos": sigma_sq_pos,
            "is_reshape_v7/sigma_sq_neg": sigma_sq_neg,
            "is_reshape_v7/n_pos": pos_mask.sum().item(),
            "is_reshape_v7/n_neg": neg_mask.sum().item(),
            "is_reshape_v7/ess_ratio_pos": ess_ratio_pos,
            "is_reshape_v7/ess_ratio_neg": ess_ratio_neg,
        }

        return (gamma_pos, gamma_neg), metrics


@register_policy_loss("is_reshape")
def compute_policy_loss_is_reshape(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute policy loss using IS Reshape framework.

    This implements the IS Reshape unified framework that bridges SFT (γ=0) and RL (γ=1).

    Theory:
        - Unified objective: L_γ(θ) = (1/γ)(𝔼_μ[w^γ r] - 𝔼_μ[r])
        - Gradient: g_γ = 𝔼_μ[w^γ · r · ∇log π_θ]
        - Target distribution evolves: p_γ* ∝ (μ·r)^(1-γ) · e^(γr/τ)
        - Optimal γ: γ* = min(γ_max, √(-log ρ_min / σ²)) where σ² = Var(log w)

    Key differences from vanilla PPO:
        1. old_log_prob is treated as the behavior/reference policy μ (sampling distribution)
        2. Computes IS weights: w = π_θ / μ
        3. Applies γ-reshape: w^γ where γ is auto-computed or fixed
        4. No PPO clipping (variance controlled by γ instead)
        5. Gradients flow through w^γ (no stop gradient needed)

    Args:
        old_log_prob: Log probabilities from behavior/reference policy μ (sampling distribution)
            Shape: (batch_size, response_length)
        log_prob: Log probabilities from current policy π_θ
            Shape: (batch_size, response_length)
        advantages: Advantage estimates (serves as reward signal)
            Shape: (batch_size, response_length)
        response_mask: Valid token mask
            Shape: (batch_size, response_length)
        loss_agg_mode: Loss aggregation strategy (see agg_loss)
        config: Actor config, should contain:
            - is_reshape.rho_min: Minimum ESS ratio (default: 0.3)
            - is_reshape.gamma_min: Minimum gamma to avoid 1/γ explosion (default: 0.05)
            - is_reshape.gamma_max: Maximum gamma value (default: 1.0)
            - is_reshape.gamma: Fixed gamma value (optional, default: auto-compute)
            - is_reshape.clip_weight: Whether to clip weights (default: False)
            - is_reshape.clip_threshold: Weight clipping threshold (default: 10.0)
        rollout_is_weights: Ignored (for interface compatibility)

    Returns:
        pg_loss: Policy gradient loss
        pg_metrics: Dictionary of metrics including gamma, ESS, variance, etc.

    Config example:
        actor:
          policy_loss:
            loss_mode: "is_reshape"
            is_reshape:
              rho_min: 0.3          # Min ESS ratio for auto gamma
              gamma_min: 0.05       # Min γ to avoid 1/γ explosion
              gamma_max: 1.0        # Max γ value
              gamma: null           # null for auto, or float for fixed
              clip_weight: false    # Whether to clip IS weights
              clip_threshold: 10.0  # Clipping threshold if enabled

    References:
        IS Reshape theory paper (theory/is_reshape_final.md)
    """
    assert config is not None, "ActorConfig must be provided for IS Reshape"

    # Extract IS Reshape config
    is_reshape_config = config.policy_loss.get("is_reshape", {}) if hasattr(config, "policy_loss") else {}
    rho_min = is_reshape_config.get("rho_min", 0.3)
    gamma_min = is_reshape_config.get("gamma_min", 0.05)
    gamma_max = is_reshape_config.get("gamma_max", 1.0)
    gamma_fixed = is_reshape_config.get("gamma", None)
    clip_weight = is_reshape_config.get("clip_weight", False)
    clip_threshold = is_reshape_config.get("clip_threshold", 10.0)

    # Compute log importance weights: log w = log π_θ - log μ
    # where μ is the behavior/reference policy (old_log_prob)
    log_w = log_prob - old_log_prob

    # Clamp for numerical stability
    log_w = torch.clamp(log_w, min=-20.0, max=20.0)

    # Compute optimal gamma (or use fixed gamma)
    gamma, gamma_metrics = compute_is_reshape_gamma(
        log_w=log_w,
        response_mask=response_mask,
        rho_min=rho_min,
        gamma_min=gamma_min,
        gamma_max=gamma_max,
        gamma_fixed=gamma_fixed,
    )

    # Compute IS Reshape weights: w^γ = exp(γ · log w)
    # IMPORTANT: Don't detach! We need gradients to flow through w^γ
    # Theory (Theorem 2.1): ∇L_γ = E_μ[w^γ · r · ∇log π_θ]
    # This requires computing ∇w^γ = γ w^γ ∇log π_θ
    w_gamma = torch.exp(gamma * log_w)

    # Optional: clip weights for stability (note: this truncates gradients)
    if clip_weight:
        w_gamma = torch.clamp(w_gamma, max=clip_threshold)

    # IS Reshape objective: J_γ = (1/γ) E_μ[w^γ · r]
    # We compute the negative for minimization: L = -(1/γ) E_μ[w^γ · r]
    # PyTorch auto-diff will give: ∇L = -(1/γ) E[(∇w^γ) · r] = -E[w^γ · r · ∇log π_θ]
    #
    # Construct per-token loss matrix for agg_loss
    # loss_mat[i, j] = -w_gamma[i, j] * A[i, j] / gamma
    loss_mat = -(1.0 / gamma) * w_gamma * advantages

    # Use agg_loss to properly aggregate based on loss_agg_mode
    pg_loss = agg_loss(loss_mat, response_mask, loss_agg_mode)

    # Compute diagnostics
    with torch.no_grad():
        # KL divergence between current and reference policy
        negative_approx_kl = log_prob - old_log_prob
        ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

        # IS ratio metrics for comparison with vanilla PPO
        # Original IS ratio: w = π_θ / μ (before reshape)
        w_original = torch.exp(log_w)
        is_ratio_original = verl_F.masked_mean(w_original, response_mask)

        # Reshaped IS ratio: w^γ (actually used in loss)
        is_ratio_used = verl_F.masked_mean(w_gamma, response_mask)

        # Weight statistics
        w_gamma_flat = w_gamma[response_mask > 0]
        if w_gamma_flat.numel() > 0:
            max_weight = w_gamma_flat.max()
            mean_weight = verl_F.masked_mean(w_gamma, response_mask)

            # Actual ESS computation (more accurate than approximation)
            ess = (w_gamma_flat.sum() ** 2) / (w_gamma_flat**2).sum()
            ess_ratio = ess / w_gamma_flat.numel()
        else:
            max_weight = torch.tensor(1.0, device=w_gamma.device)
            mean_weight = torch.tensor(1.0, device=w_gamma.device)
            ess_ratio = 1.0

    # Collect metrics
    pg_metrics = {
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/is_ratio_original": is_ratio_original.detach().item(),
        "actor/is_ratio_used": is_ratio_used.detach().item(),
        "actor/is_reshape_max_weight": max_weight.detach().item(),
        "actor/is_reshape_mean_weight": mean_weight.detach().item(),
        "actor/is_reshape_ess_ratio_actual": ess_ratio.item() if isinstance(ess_ratio, torch.Tensor) else ess_ratio,
    }
    pg_metrics.update(gamma_metrics)

    return pg_loss, pg_metrics


@register_policy_loss("is_reshape_pos")
def compute_policy_loss_is_reshape_pos(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute policy loss using IS Reshape only for positive advantages.

    This variant applies IS reshape (w^γ) only to positive advantage samples,
    while negative advantage samples use standard IS (w^1 = w).

    Motivation:
        - For A > 0 (good samples): Use w^γ with adaptive γ < 1 to control variance
          while still learning from new good samples (w < 1 gets amplified by concave w^γ)
        - For A < 0 (bad samples): Use standard IS (w) to maintain full gradient signal
          for avoiding bad actions, especially when w > 1 (policy hasn't avoided them yet)

    Theory:
        Loss = -E_μ[(1/γ) * w^γ * A * 𝟙(A>0)] - E_μ[w * A * 𝟙(A<0)]

        Gradient:
        - A > 0: ∇L = -E_μ[w^γ * A * ∇log π]  (reshaped IS)
        - A < 0: ∇L = -E_μ[w * A * ∇log π]    (standard IS)

    Args:
        old_log_prob: Log probabilities from behavior/reference policy μ
            Shape: (batch_size, response_length)
        log_prob: Log probabilities from current policy π_θ
            Shape: (batch_size, response_length)
        advantages: Advantage estimates
            Shape: (batch_size, response_length)
        response_mask: Valid token mask
            Shape: (batch_size, response_length)
        loss_agg_mode: Loss aggregation strategy
        config: Actor config, should contain:
            - is_reshape_pos.rho_min: Min ESS ratio for γ computation (default: 0.3)
            - is_reshape_pos.gamma_min: Minimum γ value (default: 0.05)
            - is_reshape_pos.gamma_max: Maximum γ value (default: 1.0)
            - is_reshape_pos.gamma: Fixed γ value (optional, default: auto-compute)
            - is_reshape_pos.clip_weight: Whether to clip weights (default: False)
            - is_reshape_pos.clip_threshold: Weight clipping threshold (default: 10.0)
        rollout_is_weights: Ignored (for interface compatibility)

    Returns:
        pg_loss: Policy gradient loss
        pg_metrics: Dictionary of metrics

    Config example:
        actor:
          policy_loss:
            loss_mode: "is_reshape_pos"
            is_reshape_pos:
              rho_min: 0.3
              gamma_min: 0.05
              gamma_max: 1.0
              gamma: null
              clip_weight: false
              clip_threshold: 10.0
    """
    assert config is not None, "ActorConfig must be provided for IS Reshape Pos"

    # Extract config
    is_config = config.policy_loss.get("is_reshape_pos", {}) if hasattr(config, "policy_loss") else {}
    rho_min = is_config.get("rho_min", 0.3)
    gamma_min = is_config.get("gamma_min", 0.05)
    gamma_max = is_config.get("gamma_max", 1.0)
    gamma_fixed = is_config.get("gamma", None)
    clip_weight = is_config.get("clip_weight", False)
    clip_threshold = is_config.get("clip_threshold", 10.0)

    # Compute log importance weights: log w = log π_θ - log μ
    log_w = log_prob - old_log_prob
    log_w = torch.clamp(log_w, min=-20.0, max=20.0)

    # Create masks for positive and negative advantages
    pos_mask = (advantages > 0) & (response_mask > 0)
    neg_mask = (advantages < 0) & (response_mask > 0)

    # Compute gamma only based on positive samples (since we only reshape those)
    gamma, gamma_metrics = compute_is_reshape_gamma(
        log_w=log_w,
        response_mask=pos_mask.float(),  # Only use positive samples for variance estimation
        rho_min=rho_min,
        gamma_min=gamma_min,
        gamma_max=gamma_max,
        gamma_fixed=gamma_fixed,
    )

    # Compute weights
    w = torch.exp(log_w)  # Original IS weight
    w_gamma = torch.exp(gamma * log_w)  # Reshaped IS weight

    # Optional: clip weights for stability
    if clip_weight:
        w = torch.clamp(w, max=clip_threshold)
        w_gamma = torch.clamp(w_gamma, max=clip_threshold)

    # Compute loss differently for positive and negative advantages
    # A > 0: use reshaped IS: -(1/γ) * w^γ * A
    # A < 0: use standard IS: -w * A
    loss_mat_pos = -(1.0 / gamma) * w_gamma * advantages  # For A > 0
    loss_mat_neg = -w * advantages  # For A < 0

    # Combine: use reshaped for positive, standard for negative
    loss_mat = torch.where(pos_mask, loss_mat_pos, torch.where(neg_mask, loss_mat_neg, torch.zeros_like(advantages)))

    # Use agg_loss to properly aggregate
    pg_loss = agg_loss(loss_mat, response_mask, loss_agg_mode)

    # Compute diagnostics
    with torch.no_grad():
        # KL divergence
        ppo_kl = verl_F.masked_mean(-log_w, response_mask)

        # IS ratio metrics
        is_ratio_original = verl_F.masked_mean(w, response_mask)

        # Weight statistics for positive samples (reshaped)
        w_gamma_pos = w_gamma[pos_mask]
        if w_gamma_pos.numel() > 0:
            max_weight_pos = w_gamma_pos.max().item()
            mean_weight_pos = w_gamma_pos.mean().item()
            ess_pos = (w_gamma_pos.sum() ** 2) / (w_gamma_pos**2).sum()
            ess_ratio_pos = (ess_pos / w_gamma_pos.numel()).item()
        else:
            max_weight_pos = 1.0
            mean_weight_pos = 1.0
            ess_ratio_pos = 1.0

        # Weight statistics for negative samples (standard IS)
        w_neg = w[neg_mask]
        if w_neg.numel() > 0:
            max_weight_neg = w_neg.max().item()
            mean_weight_neg = w_neg.mean().item()
            ess_neg = (w_neg.sum() ** 2) / (w_neg**2).sum()
            ess_ratio_neg = (ess_neg / w_neg.numel()).item()
        else:
            max_weight_neg = 1.0
            mean_weight_neg = 1.0
            ess_ratio_neg = 1.0

        # Count samples
        n_pos = pos_mask.sum().item()
        n_neg = neg_mask.sum().item()

    # Collect metrics
    pg_metrics = {
        "actor/ppo_kl": ppo_kl.item(),
        "actor/is_ratio_original": is_ratio_original.item(),
        # Positive samples (reshaped)
        "actor/pos_max_weight": max_weight_pos,
        "actor/pos_mean_weight": mean_weight_pos,
        "actor/pos_ess_ratio": ess_ratio_pos,
        "actor/pos_n_samples": n_pos,
        # Negative samples (standard IS)
        "actor/neg_max_weight": max_weight_neg,
        "actor/neg_mean_weight": mean_weight_neg,
        "actor/neg_ess_ratio": ess_ratio_neg,
        "actor/neg_n_samples": n_neg,
    }
    pg_metrics.update(gamma_metrics)

    return pg_loss, pg_metrics


@register_policy_loss("is_reshape_per_sample")
def compute_policy_loss_is_reshape_per_sample(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute policy loss using IS Reshape with per-sample gamma.

    This is an enhanced version of IS Reshape that computes gamma per-sample
    instead of per-batch. Each sample (sequence) gets its own optimal gamma
    based on the variance of log_w within that sample.

    Key difference from is_reshape:
        - is_reshape: Single gamma for entire batch (batch-level ESS constraint)
        - is_reshape_per_sample: Per-sample gamma (sample-level ESS constraint)

    Per-sample gamma is ideal because:
        1. ESS constraint is meaningful at sample level
        2. Reward/Advantage is naturally per-sample
        3. Better balance between batch-level (coarse) and token-level (too fine)

    Theory:
        For each sample i:
        - σ²_i = Var(log w) within sample i
        - γ*_i = min(γ_max, √(-log ρ_min / σ²_i))
        - w^γ_i = exp(γ_i · log w) for tokens in sample i

    Args:
        old_log_prob: Log probabilities from behavior policy μ
            Shape: (batch_size, response_length)
        log_prob: Log probabilities from current policy π_θ
            Shape: (batch_size, response_length)
        advantages: Advantage estimates
            Shape: (batch_size, response_length)
        response_mask: Valid token mask
            Shape: (batch_size, response_length)
        loss_agg_mode: Loss aggregation strategy:
            - "token-mean": mean over all tokens
            - "seq-mean-token-sum": sum tokens per seq, mean over seqs
            - "seq-mean-token-mean": mean tokens per seq, mean over seqs
        config: Actor config, should contain:
            - is_reshape_per_sample.rho_min: Min ESS ratio (default: 0.3)
            - is_reshape_per_sample.gamma_min: Min gamma (default: 0.05)
            - is_reshape_per_sample.gamma_max: Max gamma (default: 1.0)
            - is_reshape_per_sample.gamma: Fixed gamma (optional)
            - is_reshape_per_sample.clip_weight: Clip weights (default: False)
            - is_reshape_per_sample.clip_threshold: Clip threshold (default: 10.0)
        rollout_is_weights: Ignored (for interface compatibility)

    Returns:
        pg_loss: Policy gradient loss
        pg_metrics: Dictionary of metrics

    Config example:
        actor:
          policy_loss:
            loss_mode: "is_reshape_per_sample"
            is_reshape_per_sample:
              rho_min: 0.3
              gamma_min: 0.05
              gamma_max: 1.0
              gamma: null  # null for auto, or float for fixed
              clip_weight: false
              clip_threshold: 10.0
    """
    assert config is not None, "ActorConfig must be provided for IS Reshape"

    # Extract config
    is_config = config.policy_loss.get("is_reshape_per_sample", {}) if hasattr(config, "policy_loss") else {}
    rho_min = is_config.get("rho_min", 0.3)
    gamma_min = is_config.get("gamma_min", 0.05)
    gamma_max = is_config.get("gamma_max", 1.0)
    gamma_fixed = is_config.get("gamma", None)
    clip_weight = is_config.get("clip_weight", False)
    clip_threshold = is_config.get("clip_threshold", 10.0)

    # Compute log importance weights: log w = log π_θ - log μ
    log_w = log_prob - old_log_prob
    log_w = torch.clamp(log_w, min=-20.0, max=20.0)

    # Compute per-sample gamma
    # gamma shape: (batch_size,)
    gamma, gamma_metrics = compute_is_reshape_gamma_per_sample(
        log_w=log_w,
        response_mask=response_mask,
        rho_min=rho_min,
        gamma_min=gamma_min,
        gamma_max=gamma_max,
        gamma_fixed=gamma_fixed,
    )

    # Expand gamma to match log_w shape for broadcasting
    # gamma: (batch_size,) -> (batch_size, 1) for broadcasting with (batch_size, seq_len)
    gamma_expanded = gamma.unsqueeze(-1)

    # Compute IS Reshape weights: w^γ = exp(γ · log w)
    # Now each sample has its own gamma
    w_gamma = torch.exp(gamma_expanded * log_w)

    # Optional: clip weights for stability
    if clip_weight:
        w_gamma = torch.clamp(w_gamma, max=clip_threshold)

    # Per-token loss: -w^γ · A / γ
    # Since gamma is per-sample, we need to be careful with the 1/gamma factor
    # For sample i, the loss contribution is: -w^γ_i · A / γ_i
    loss_mat = -(1.0 / gamma_expanded) * w_gamma * advantages

    # Use agg_loss to properly aggregate based on loss_agg_mode
    pg_loss = agg_loss(loss_mat, response_mask, loss_agg_mode)

    # Compute diagnostics
    with torch.no_grad():
        # KL divergence
        negative_approx_kl = log_prob - old_log_prob
        ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

        # IS ratio metrics
        w_original = torch.exp(log_w)
        is_ratio_original = verl_F.masked_mean(w_original, response_mask)
        is_ratio_used = verl_F.masked_mean(w_gamma, response_mask)

        # Weight statistics
        w_gamma_flat = w_gamma[response_mask > 0]
        if w_gamma_flat.numel() > 0:
            max_weight = w_gamma_flat.max()
            mean_weight = verl_F.masked_mean(w_gamma, response_mask)

            # Actual ESS computation
            ess = (w_gamma_flat.sum() ** 2) / (w_gamma_flat**2).sum()
            ess_ratio = ess / w_gamma_flat.numel()
        else:
            max_weight = torch.tensor(1.0, device=w_gamma.device)
            mean_weight = torch.tensor(1.0, device=w_gamma.device)
            ess_ratio = 1.0

    # Collect metrics
    pg_metrics = {
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/is_ratio_original": is_ratio_original.detach().item(),
        "actor/is_ratio_used": is_ratio_used.detach().item(),
        "actor/is_reshape_max_weight": max_weight.detach().item(),
        "actor/is_reshape_mean_weight": mean_weight.detach().item(),
        "actor/is_reshape_ess_ratio_actual": ess_ratio.item() if isinstance(ess_ratio, torch.Tensor) else ess_ratio,
    }
    pg_metrics.update(gamma_metrics)

    return pg_loss, pg_metrics


@register_policy_loss("is_reshape_renyi")
def compute_policy_loss_is_reshape_renyi(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute policy loss using IS Reshape Rényi framework (v3).

    This implements the v3 theory based on correctness metric C = A · log w.

    Theory (v3):
        - Correctness metric: C(w, A) = A · log w
          * C > 0: Policy is in the correct direction
          * C < 0: Policy is in the wrong direction

        - Information factor: I(w, A) = σ(-C/τ) = P(policy is wrong)

        - Optimal gamma (per-sample):
          γ*(w, A) = γ_base + (γ_target - γ_base) · P_correct(w, A)

          where:
          * γ_base: Global baseline from ESS constraint (same as v2)
          * γ_target(w) = σ(-log w · T): Target gamma based on w
          * P_correct = σ(C/τ): Probability that policy is correct

    Key innovations of v3:
        1. Per-sample gamma: Each token gets its own γ based on (w, A)
        2. Continuous formula: Smooth interpolation, no hard quadrants
        3. Unified metric C: Single scalar captures policy correctness

    Args:
        old_log_prob: Log probabilities from behavior/reference policy μ
            Shape: (batch_size, response_length)
        log_prob: Log probabilities from current policy π_θ
            Shape: (batch_size, response_length)
        advantages: Advantage estimates
            Shape: (batch_size, response_length)
        response_mask: Valid token mask
            Shape: (batch_size, response_length)
        loss_agg_mode: Loss aggregation strategy
        config: Actor config, should contain:
            - is_reshape_renyi.rho_min: Min ESS ratio for γ_base (default: 0.3)
            - is_reshape_renyi.tau: Temperature for correctness (default: 1.0)
            - is_reshape_renyi.T: Temperature for γ_target (default: 5.0)
            - is_reshape_renyi.gamma_min: Minimum gamma value (default: 0.05)
            - is_reshape_renyi.clip_weight: Whether to clip weights (default: False)
            - is_reshape_renyi.clip_threshold: Weight clipping threshold (default: 10.0)
        rollout_is_weights: Ignored (for interface compatibility)

    Returns:
        pg_loss: Policy gradient loss
        pg_metrics: Dictionary of metrics

    Config example:
        actor:
          policy_loss:
            loss_mode: "is_reshape_renyi"
            is_reshape_renyi:
              rho_min: 0.3     # Min ESS for γ_base
              tau: 1.0         # Correctness temperature
              T: 5.0           # γ_target temperature
              gamma_min: 0.05  # Minimum gamma value
              clip_weight: false
              clip_threshold: 10.0

    References:
        IS Reshape v3 theory: recipe/is_shape/theory/is_reshape_v3_renyi.md
    """
    assert config is not None, "ActorConfig must be provided for IS Reshape Rényi"

    # Extract config
    renyi_config = config.policy_loss.get("is_reshape_renyi", {}) if hasattr(config, "policy_loss") else {}
    rho_min = renyi_config.get("rho_min", 0.3)
    tau = renyi_config.get("tau", 1.0)
    T = renyi_config.get("T", 5.0)
    gamma_min = renyi_config.get("gamma_min", 0.05)
    clip_weight = renyi_config.get("clip_weight", False)
    clip_threshold = renyi_config.get("clip_threshold", 10.0)

    # Step 1: Compute log w
    log_w = log_prob - old_log_prob
    log_w = torch.clamp(log_w, min=-20.0, max=20.0)

    # Step 2: Compute γ_base (global baseline from ESS constraint)
    with torch.no_grad():
        log_w_flat = log_w[response_mask.bool()]
        if log_w_flat.numel() == 0:
            gamma_base = 1.0
            sigma_sq = 0.0
        else:
            sigma_sq = torch.var(log_w_flat, unbiased=True).item()
            if sigma_sq < 1e-8:
                gamma_base = 1.0
            else:
                gamma_base = min(1.0, math.sqrt(-math.log(rho_min) / sigma_sq))
                gamma_base = max(gamma_min, gamma_base)  # Ensure γ_base >= gamma_min

    # Step 3: Correctness metric C = A · log w
    C = advantages * log_w

    # Step 4: Probability that policy is correct: P_correct = σ(C/τ)
    P_correct = torch.sigmoid(C / tau)

    # Step 5: Target gamma: γ_target = σ(-log w · T)
    # When w < 1 (log w < 0) → γ_target → 1
    # When w > 1 (log w > 0) → γ_target → 0
    gamma_target = torch.sigmoid(-log_w * T)

    # Step 6: Optimal gamma (per-sample):
    # γ* = γ_base + (γ_target - γ_base) · P_correct
    gamma = gamma_base + (gamma_target - gamma_base) * P_correct

    # Step 7: Compute w^γ
    # Note: gamma is per-sample, so w_gamma is also per-sample
    # IMPORTANT: Detach gamma so gradients don't flow through the γ computation.
    # This ensures the gradient is simply: ∇L = -E[γ * w^γ * A * ∇log_prob]
    # Without detach, we'd get extra gradients through ∂γ/∂log_w which causes instability.
    gamma_detached = gamma.detach()
    w_gamma = torch.exp(gamma_detached * log_w)

    # Optional: clip weights
    if clip_weight:
        w_gamma = torch.clamp(w_gamma, max=clip_threshold)

    # Step 8: Policy gradient loss
    # L = -E_μ[w^γ · A · log π_θ]
    # Note: Unlike v2, we don't have a 1/γ factor because γ is per-sample
    # The objective is simply: maximize E[w^γ · A]
    w_gamma_advantages = w_gamma * advantages * response_mask
    objective = w_gamma_advantages.sum() / (response_mask.sum() + 1e-8)
    pg_loss = -objective

    # Compute diagnostics
    with torch.no_grad():
        # KL divergence
        negative_approx_kl = log_prob - old_log_prob
        ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

        # IS ratio metrics
        w_original = torch.exp(log_w)
        is_ratio_original = verl_F.masked_mean(w_original, response_mask)
        is_ratio_used = verl_F.masked_mean(w_gamma, response_mask)

        # Weight statistics
        w_gamma_flat = w_gamma[response_mask.bool()]
        if w_gamma_flat.numel() > 0:
            max_weight = w_gamma_flat.max().item()
            mean_weight = verl_F.masked_mean(w_gamma, response_mask).item()
            ess = (w_gamma_flat.sum() ** 2) / (w_gamma_flat**2).sum()
            ess_ratio = (ess / w_gamma_flat.numel()).item()
        else:
            max_weight = 1.0
            mean_weight = 1.0
            ess_ratio = 1.0

        # Rényi-specific metrics
        gamma_mean = verl_F.masked_mean(gamma, response_mask).item()
        gamma_std = torch.std(gamma[response_mask.bool()]).item() if w_gamma_flat.numel() > 1 else 0.0

        C_mean = verl_F.masked_mean(C, response_mask).item()
        C_std = torch.std(C[response_mask.bool()]).item() if w_gamma_flat.numel() > 1 else 0.0

        P_correct_mean = verl_F.masked_mean(P_correct, response_mask).item()

        gamma_target_mean = verl_F.masked_mean(gamma_target, response_mask).item()

        # Count tokens in each correctness regime
        C_flat = C[response_mask.bool()]
        if C_flat.numel() > 0:
            correct_ratio = (C_flat > 0).float().mean().item()  # Fraction of tokens where C > 0
        else:
            correct_ratio = 0.5

    # Collect metrics
    pg_metrics = {
        "actor/ppo_kl": ppo_kl.item(),
        "actor/is_ratio_original": is_ratio_original.item(),
        "actor/is_ratio_used": is_ratio_used.item(),
        "actor/is_reshape_max_weight": max_weight,
        "actor/is_reshape_mean_weight": mean_weight,
        "actor/is_reshape_ess_ratio_actual": ess_ratio,
        # Rényi-specific metrics
        "actor/renyi_gamma_base": gamma_base,
        "actor/renyi_gamma_mean": gamma_mean,
        "actor/renyi_gamma_std": gamma_std,
        "actor/renyi_gamma_target_mean": gamma_target_mean,
        "actor/renyi_C_mean": C_mean,
        "actor/renyi_C_std": C_std,
        "actor/renyi_P_correct_mean": P_correct_mean,
        "actor/renyi_correct_ratio": correct_ratio,
        "actor/renyi_sigma_sq": sigma_sq,
    }

    return pg_loss, pg_metrics


def _sech2(x: torch.Tensor) -> torch.Tensor:
    """Compute sech²(x) = 1/cosh²(x) in a numerically stable way."""
    return 1.0 / torch.cosh(x).pow(2)


@register_policy_loss("is_reshape_v4")
def compute_policy_loss_is_reshape_v4(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute policy loss using IS Reshape v4 with reward-modulated trust region.

    This implements IS Reshape v4 which combines:
    1. SAPO-style soft trust region via sech²(τ(r-1)/2)
    2. Reward-modulated γ: γ(|A|) = σ(β|A|)
    3. SFT-RL interpolation: w(r,A) = (1-γ) + γ·r·sech²(τ(r-1)/2)

    Key features:
        - γ → 0: Pure SFT (weight=1, forward KL)
        - γ → 1: SAPO-like (soft trust region, reverse KL)
        - High |A| → more RL-like, Low |A| → more SFT-like

    Args:
        old_log_prob: Log probabilities from behavior policy μ
            Shape: (batch_size, response_length)
        log_prob: Log probabilities from current policy π_θ
            Shape: (batch_size, response_length)
        advantages: Advantage estimates (reward signal)
            Shape: (batch_size, response_length)
        response_mask: Valid token mask
            Shape: (batch_size, response_length)
        loss_agg_mode: Loss aggregation strategy
        config: Actor config, should contain:
            - is_reshape_v4.tau: Trust region temperature (default: 1.0)
            - is_reshape_v4.beta: Reward sensitivity for γ modulation (default: 1.0)
            - is_reshape_v4.gamma_min: Minimum γ value (default: 0.0)
            - is_reshape_v4.gamma_max: Maximum γ value (default: 1.0)
            - is_reshape_v4.clip_weight: Whether to clip weights (default: False)
            - is_reshape_v4.clip_threshold: Weight clipping threshold (default: 10.0)
        rollout_is_weights: Ignored (for interface compatibility)

    Returns:
        pg_loss: Policy gradient loss
        pg_metrics: Dictionary of metrics

    Config example:
        actor:
          policy_loss:
            loss_mode: "is_reshape_v4"
            is_reshape_v4:
              tau: 1.0           # Trust region temperature
              beta: 1.0          # Reward sensitivity
              gamma_min: 0.0     # Minimum γ (SFT floor)
              gamma_max: 1.0     # Maximum γ (RL ceiling)
              clip_weight: false
              clip_threshold: 10.0

    Theory:
        The gradient weight function:
            w(r, A) = (1 - γ(|A|)) + γ(|A|) · r · sech²(τ(r-1)/2)

        where γ(|A|) = σ(β|A|) modulates between:
            - SFT (γ=0): w(r) = 1 (ignore distribution shift)
            - RL (γ=1): w(r) = r·sech²(...) (SAPO-style trust region)

        This gives an implicit trust region: when r deviates from 1,
        the SAPO component decays, limiting the gradient magnitude.

    References:
        IS Reshape v4 theory: recipe/is_shape/theory/is_reshape_v4_reward_modulated.md
        SAPO paper: arXiv:2511.20347
    """
    assert config is not None, "ActorConfig must be provided for IS Reshape v4"

    # Extract config
    v4_config = config.policy_loss.get("is_reshape_v4", {}) if hasattr(config, "policy_loss") else {}
    tau = v4_config.get("tau", 1.0)
    beta = v4_config.get("beta", 1.0)
    gamma_min = v4_config.get("gamma_min", 0.0)
    gamma_max = v4_config.get("gamma_max", 1.0)
    clip_weight = v4_config.get("clip_weight", False)
    clip_threshold = v4_config.get("clip_threshold", 10.0)

    # Step 1: Compute log importance ratio and ratio
    log_r = log_prob - old_log_prob
    log_r = torch.clamp(log_r, min=-20.0, max=20.0)
    r = torch.exp(log_r)

    # Step 2: Compute reward-modulated γ
    # γ(|A|) = γ_min + (γ_max - γ_min) · σ(β|A|)
    abs_A = torch.abs(advantages)
    gamma_raw = torch.sigmoid(beta * abs_A)
    gamma = gamma_min + (gamma_max - gamma_min) * gamma_raw

    # Step 3: Compute SAPO-style soft trust region weight
    # sapo_weight = r · sech²(τ(r-1)/2)
    sapo_weight = r * _sech2(tau * (r - 1) / 2)

    # Step 4: Compute mixed weight
    # w(r, A) = (1 - γ) + γ · sapo_weight
    weight = (1 - gamma) + gamma * sapo_weight

    # Optional: clip weights for stability
    if clip_weight:
        weight = torch.clamp(weight, max=clip_threshold)

    # Step 5: Policy gradient loss
    # L = -E_μ[w(r, A) · A]
    # Note: weight is detached so gradients only flow through log_prob via advantages
    weight_detached = weight.detach()
    pg_loss = -(weight_detached * advantages * response_mask).sum() / (response_mask.sum() + 1e-8)

    # Compute diagnostics
    with torch.no_grad():
        mask = response_mask.bool()
        total = response_mask.sum()

        # KL divergence
        ppo_kl = verl_F.masked_mean(-log_r, response_mask)

        # IS ratio metrics
        is_ratio_original = verl_F.masked_mean(r, response_mask)
        is_ratio_used = verl_F.masked_mean(weight, response_mask)

        # Weight statistics
        weight_flat = weight[mask]
        if weight_flat.numel() > 0:
            max_weight = weight_flat.max().item()
            mean_weight = verl_F.masked_mean(weight, response_mask).item()
        else:
            max_weight = 1.0
            mean_weight = 1.0

        # v4-specific metrics
        gamma_mean = verl_F.masked_mean(gamma, response_mask).item()
        gamma_std = torch.std(gamma[mask]).item() if weight_flat.numel() > 1 else 0.0

        sapo_weight_mean = verl_F.masked_mean(sapo_weight, response_mask).item()
        abs_A_mean = verl_F.masked_mean(abs_A, response_mask).item()

        # Trust region violation: how many tokens have r far from 1?
        r_flat = r[mask]
        if r_flat.numel() > 0:
            trust_region_violation = ((r_flat < 0.5) | (r_flat > 2.0)).float().mean().item()
        else:
            trust_region_violation = 0.0

    # Collect metrics
    pg_metrics = {
        "actor/ppo_kl": ppo_kl.item(),
        "actor/is_ratio_original": is_ratio_original.item(),
        "actor/is_ratio_used": is_ratio_used.item(),
        "actor/is_reshape_max_weight": max_weight,
        "actor/is_reshape_mean_weight": mean_weight,
        # v4-specific metrics
        "actor/v4_gamma_mean": gamma_mean,
        "actor/v4_gamma_std": gamma_std,
        "actor/v4_sapo_weight_mean": sapo_weight_mean,
        "actor/v4_abs_adv_mean": abs_A_mean,
        "actor/v4_trust_violation": trust_region_violation,
        "actor/v4_tau": tau,
        "actor/v4_beta": beta,
    }

    return pg_loss, pg_metrics


@register_policy_loss("is_reshape_v5")
def compute_policy_loss_is_reshape_v5(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute policy loss using IS Reshape v5 with bias-variance motivated trust region.

    This implements IS Reshape v5 based on first principles:
    1. SFT-RL gradient unification: g(w) interpolates between 1 (SFT) and w (RL)
    2. Bias-Variance tradeoff: soft trust region h(w) = sech²(τ(w-1)/2)
    3. Correctness-adaptive γ: γ(w,A) depends on C = A·log w

    Core formula:
        g(w, A) = (1 - γ(w,A)) + γ(w,A) · w · h(w)

    where:
        - h(w) = sech²(τ_h(w-1)/2): soft trust region
        - γ(w,A) = γ_base + (γ_max - γ_base) · σ(-C/τ_c): adaptive mixing
        - C = A · log w: correctness metric

    Key properties:
        - γ → 0: Pure SFT (weight=1)
        - γ → 1: RL with trust region (weight=w·h(w))
        - C > 0 (policy correct): lower γ, more SFT-like
        - C < 0 (policy wrong): higher γ, more RL-like

    Args:
        old_log_prob: Log probabilities from behavior policy μ
            Shape: (batch_size, response_length)
        log_prob: Log probabilities from current policy π_θ
            Shape: (batch_size, response_length)
        advantages: Advantage estimates
            Shape: (batch_size, response_length)
        response_mask: Valid token mask
            Shape: (batch_size, response_length)
        loss_agg_mode: Loss aggregation strategy
        config: Actor config, should contain:
            - is_reshape_v5.tau_h: Trust region temperature (default: 1.0)
            - is_reshape_v5.tau_c: Correctness temperature (default: 1.0)
            - is_reshape_v5.gamma_base: Minimum γ value (default: 0.1)
            - is_reshape_v5.gamma_max: Maximum γ value (default: 0.9)
            - is_reshape_v5.clip_weight: Whether to clip weights (default: False)
            - is_reshape_v5.clip_threshold: Weight clipping threshold (default: 10.0)
        rollout_is_weights: Ignored (for interface compatibility)

    Returns:
        pg_loss: Policy gradient loss
        pg_metrics: Dictionary of metrics

    Config example:
        actor:
          policy_loss:
            loss_mode: "is_reshape_v5"
            is_reshape_v5:
              tau_h: 1.0         # Trust region temperature
              tau_c: 1.0         # Correctness temperature
              gamma_base: 0.1    # Minimum γ (SFT floor)
              gamma_max: 0.9     # Maximum γ (RL ceiling)

    Four quadrant behavior (verified):
        | Case | w   | A  | C  | γ    | g(w,A) | Behavior              |
        |------|-----|----|----|------|--------|-----------------------|
        | 1    | >1  | >0 | +  | low  | ~1     | Moderate update       |
        | 2    | >1  | <0 | -  | high | ~w·h   | Correct with TR       |
        | 3    | <1  | >0 | -  | high | ~w·h   | Learn with IS weight  |
        | 4    | <1  | <0 | +  | low  | ~1     | Maintain              |

    References:
        IS Reshape v5 theory: recipe/is_shape/theory/is_reshape_v5_bias_variance.md
    """
    assert config is not None, "ActorConfig must be provided for IS Reshape v5"

    # Extract config
    v5_config = config.policy_loss.get("is_reshape_v5", {}) if hasattr(config, "policy_loss") else {}
    tau_h = v5_config.get("tau_h", 1.0)  # Trust region temperature
    tau_c = v5_config.get("tau_c", 1.0)  # Correctness temperature
    gamma_base = v5_config.get("gamma_base", 0.1)
    gamma_max = v5_config.get("gamma_max", 0.9)
    clip_weight = v5_config.get("clip_weight", False)
    clip_threshold = v5_config.get("clip_threshold", 10.0)

    # Step 1: Compute log w and w
    log_w = log_prob - old_log_prob
    log_w = torch.clamp(log_w, min=-20.0, max=20.0)
    w = torch.exp(log_w)

    # Step 2: Compute correctness metric C = A · log w
    C = advantages * log_w

    # Step 3: Compute P_wrong = σ(-C/τ_c)
    # C > 0 (correct) → P_wrong → 0
    # C < 0 (wrong) → P_wrong → 1
    P_wrong = torch.sigmoid(-C / tau_c)

    # Step 4: Compute adaptive γ(w, A)
    # γ = γ_base + (γ_max - γ_base) · P_wrong
    gamma = gamma_base + (gamma_max - gamma_base) * P_wrong

    # Step 5: Compute trust region h(w) = sech²(τ_h(w-1)/2)
    h_w = _sech2(tau_h * (w - 1) / 2)

    # Step 6: Compute gradient weight g(w, A)
    # g = (1 - γ) + γ · w · h(w)
    rl_weight = w * h_w
    weight = (1 - gamma) + gamma * rl_weight

    # Optional: clip weights for stability
    if clip_weight:
        weight = torch.clamp(weight, max=clip_threshold)

    # Step 7: Policy gradient loss
    # L = -E_μ[g(w, A) · A]
    weight_detached = weight.detach()
    pg_loss = -(weight_detached * advantages * response_mask).sum() / (response_mask.sum() + 1e-8)

    # Compute diagnostics
    with torch.no_grad():
        mask = response_mask.bool()
        total = response_mask.sum()

        # KL divergence
        ppo_kl = verl_F.masked_mean(-log_w, response_mask)

        # IS ratio metrics
        is_ratio_original = verl_F.masked_mean(w, response_mask)
        is_ratio_used = verl_F.masked_mean(weight, response_mask)

        # Weight statistics
        weight_flat = weight[mask]
        if weight_flat.numel() > 0:
            max_weight = weight_flat.max().item()
            mean_weight = verl_F.masked_mean(weight, response_mask).item()
        else:
            max_weight = 1.0
            mean_weight = 1.0

        # v5-specific metrics
        gamma_mean = verl_F.masked_mean(gamma, response_mask).item()
        gamma_std = torch.std(gamma[mask]).item() if weight_flat.numel() > 1 else 0.0

        C_mean = verl_F.masked_mean(C, response_mask).item()
        C_std = torch.std(C[mask]).item() if weight_flat.numel() > 1 else 0.0

        P_wrong_mean = verl_F.masked_mean(P_wrong, response_mask).item()

        h_w_mean = verl_F.masked_mean(h_w, response_mask).item()
        rl_weight_mean = verl_F.masked_mean(rl_weight, response_mask).item()

        # Quadrant statistics
        C_flat = C[mask]
        if C_flat.numel() > 0:
            correct_ratio = (C_flat > 0).float().mean().item()
        else:
            correct_ratio = 0.5

        # Trust region violation
        w_flat = w[mask]
        if w_flat.numel() > 0:
            trust_violation = ((w_flat < 0.5) | (w_flat > 2.0)).float().mean().item()
        else:
            trust_violation = 0.0

    # Collect metrics
    pg_metrics = {
        "actor/ppo_kl": ppo_kl.item(),
        "actor/is_ratio_original": is_ratio_original.item(),
        "actor/is_ratio_used": is_ratio_used.item(),
        "actor/is_reshape_max_weight": max_weight,
        "actor/is_reshape_mean_weight": mean_weight,
        # v5-specific metrics
        "actor/v5_gamma_mean": gamma_mean,
        "actor/v5_gamma_std": gamma_std,
        "actor/v5_C_mean": C_mean,
        "actor/v5_C_std": C_std,
        "actor/v5_P_wrong_mean": P_wrong_mean,
        "actor/v5_correct_ratio": correct_ratio,
        "actor/v5_h_w_mean": h_w_mean,
        "actor/v5_rl_weight_mean": rl_weight_mean,
        "actor/v5_trust_violation": trust_violation,
        "actor/v5_tau_h": tau_h,
        "actor/v5_tau_c": tau_c,
        "actor/v5_gamma_base": gamma_base,
        "actor/v5_gamma_max": gamma_max,
    }

    return pg_loss, pg_metrics


@register_policy_loss("is_reshape_v7")
def compute_policy_loss_is_reshape_v7(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute policy loss using IS Reshape v7 with discrete grouping by advantage sign.

    This implements the v7 theory (is_reshape_v7_final.md) which:
    1. Proves MSE is convex on both (0,1) and (1,2) intervals
    2. Uses separate γ+ for A > 0 and γ- for A < 0 (discrete grouping, not continuous)
    3. Derives Box-Cox form from f-divergence endpoint conditions

    Key insight from v7 theory:
        - γ+ ∈ (0, 1) for positive samples: concave function w^γ+
          → amplifies w < 1 (new good samples to learn)
          → shrinks w > 1 (known good samples, reduce variance)

        - γ- ∈ (1, 2) for negative samples: convex function w^γ-
          → shrinks w < 1 (already avoided bad samples)
          → amplifies w > 1 (not yet avoided, need to punish)

    Objective function (Box-Cox form):
        L(θ) = E_μ[(w^{γ(A)} - 1) / γ(A) · A]

    where γ(A) = γ+ if A > 0, γ- if A < 0

    Gradient:
        ∇L = E_μ[w^{γ(A)} · A · ∇log π]

    Args:
        old_log_prob: Log probabilities from behavior policy μ
            Shape: (batch_size, response_length)
        log_prob: Log probabilities from current policy π_θ
            Shape: (batch_size, response_length)
        advantages: Advantage estimates
            Shape: (batch_size, response_length)
        response_mask: Valid token mask
            Shape: (batch_size, response_length)
        loss_agg_mode: Loss aggregation strategy
        config: Actor config, should contain:
            - is_reshape_v7.rho_min: Min ESS ratio (default: 0.3)
            - is_reshape_v7.gamma_pos_min: Min γ for A > 0 (default: 0.05)
            - is_reshape_v7.gamma_pos_max: Max γ for A > 0 (default: 0.99)
            - is_reshape_v7.gamma_neg_min: Min γ for A < 0 (default: 1.01)
            - is_reshape_v7.gamma_neg_max: Max γ for A < 0 (default: 1.99)
            - is_reshape_v7.gamma_pos: Fixed γ for A > 0 (optional)
            - is_reshape_v7.gamma_neg: Fixed γ for A < 0 (optional)
            - is_reshape_v7.clip_weight: Whether to clip weights (default: False)
            - is_reshape_v7.clip_threshold: Weight clipping threshold (default: 10.0)
        rollout_is_weights: Ignored (for interface compatibility)

    Returns:
        pg_loss: Policy gradient loss
        pg_metrics: Dictionary of metrics

    Config example:
        actor:
          policy_loss:
            loss_mode: "is_reshape_v7"
            is_reshape_v7:
              rho_min: 0.3            # Min ESS ratio
              gamma_pos_min: 0.05     # Min γ for positive samples
              gamma_pos_max: 0.99     # Max γ for positive samples
              gamma_neg_min: 1.01     # Min γ for negative samples
              gamma_neg_max: 1.99     # Max γ for negative samples
              gamma_pos: null         # null for auto, or fixed value
              gamma_neg: null         # null for auto, or fixed value
              clip_weight: false
              clip_threshold: 10.0

    References:
        IS Reshape v7 theory: recipe/is_shape/theory/is_reshape_v7_final.md
    """
    assert config is not None, "ActorConfig must be provided for IS Reshape v7"

    # Extract config
    v7_config = config.policy_loss.get("is_reshape_v7", {}) if hasattr(config, "policy_loss") else {}
    rho_min = v7_config.get("rho_min", 0.3)
    gamma_pos_min = v7_config.get("gamma_pos_min", 0.05)
    gamma_pos_max = v7_config.get("gamma_pos_max", 0.99)
    gamma_neg_min = v7_config.get("gamma_neg_min", 1.01)
    gamma_neg_max = v7_config.get("gamma_neg_max", 1.99)
    gamma_pos_fixed = v7_config.get("gamma_pos", None)
    gamma_neg_fixed = v7_config.get("gamma_neg", None)
    clip_weight = v7_config.get("clip_weight", False)
    clip_threshold = v7_config.get("clip_threshold", 10.0)

    # Step 1: Compute log importance weights
    log_w = log_prob - old_log_prob
    log_w = torch.clamp(log_w, min=-20.0, max=20.0)

    # Step 2: Compute γ+ and γ- using v7 algorithm
    (gamma_pos, gamma_neg), gamma_metrics = compute_is_reshape_gamma_v7(
        log_w=log_w,
        advantages=advantages,
        response_mask=response_mask,
        rho_min=rho_min,
        gamma_pos_min=gamma_pos_min,
        gamma_pos_max=gamma_pos_max,
        gamma_neg_min=gamma_neg_min,
        gamma_neg_max=gamma_neg_max,
        gamma_pos_fixed=gamma_pos_fixed,
        gamma_neg_fixed=gamma_neg_fixed,
    )

    # Step 3: Create gamma tensor based on advantage sign
    # γ(A) = γ+ if A > 0, γ- if A < 0, 1.0 if A == 0
    gamma = torch.ones_like(advantages)
    pos_mask = advantages > 0
    neg_mask = advantages < 0
    gamma[pos_mask] = gamma_pos
    gamma[neg_mask] = gamma_neg

    # Step 4: Compute w^γ = exp(γ · log w)
    # IMPORTANT: Don't detach log_w! Gradients flow through w^γ
    w_gamma = torch.exp(gamma * log_w)

    # Optional: clip weights for stability
    if clip_weight:
        w_gamma = torch.clamp(w_gamma, max=clip_threshold)

    # Step 5: Compute Box-Cox objective: (w^γ - 1) / γ · A
    # Handle γ ≈ 0 case (limit is log w)
    safe_gamma = torch.where(
        gamma.abs() < 0.01,
        torch.ones_like(gamma),
        gamma
    )

    box_cox = torch.where(
        gamma.abs() < 0.01,
        log_w,  # γ → 0 limit
        (w_gamma - 1) / safe_gamma
    )

    # Objective: maximize E[(w^γ - 1)/γ · A] = minimize -E[(w^γ - 1)/γ · A]
    loss_mat = -box_cox * advantages

    # Use agg_loss to properly aggregate
    pg_loss = agg_loss(loss_mat, response_mask, loss_agg_mode)

    # Compute diagnostics
    with torch.no_grad():
        mask = response_mask > 0

        # KL divergence
        ppo_kl = verl_F.masked_mean(-log_w, response_mask)

        # Original IS ratio
        w_original = torch.exp(log_w)
        is_ratio_original = verl_F.masked_mean(w_original, response_mask)

        # Reshaped IS ratio
        is_ratio_used = verl_F.masked_mean(w_gamma, response_mask)

        # Weight statistics
        w_gamma_flat = w_gamma[mask]
        if w_gamma_flat.numel() > 0:
            max_weight = w_gamma_flat.max().item()
            mean_weight = verl_F.masked_mean(w_gamma, response_mask).item()

            # Actual ESS
            ess = (w_gamma_flat.sum() ** 2) / (w_gamma_flat**2).sum()
            ess_ratio = (ess / w_gamma_flat.numel()).item()
        else:
            max_weight = 1.0
            mean_weight = 1.0
            ess_ratio = 1.0

        # Per-group statistics
        w_gamma_pos = w_gamma[pos_mask & mask]
        w_gamma_neg = w_gamma[neg_mask & mask]

        if w_gamma_pos.numel() > 0:
            mean_weight_pos = w_gamma_pos.mean().item()
            max_weight_pos = w_gamma_pos.max().item()
        else:
            mean_weight_pos = 1.0
            max_weight_pos = 1.0

        if w_gamma_neg.numel() > 0:
            mean_weight_neg = w_gamma_neg.mean().item()
            max_weight_neg = w_gamma_neg.max().item()
        else:
            mean_weight_neg = 1.0
            max_weight_neg = 1.0

        # Box-Cox objective statistics
        objective = box_cox * advantages
        objective_mean = verl_F.masked_mean(objective, response_mask).item()

    # Collect metrics
    pg_metrics = {
        "actor/ppo_kl": ppo_kl.item(),
        "actor/is_ratio_original": is_ratio_original.item(),
        "actor/is_ratio_used": is_ratio_used.item(),
        "actor/is_reshape_max_weight": max_weight,
        "actor/is_reshape_mean_weight": mean_weight,
        "actor/is_reshape_ess_ratio_actual": ess_ratio,
        # v7-specific metrics
        "actor/v7_mean_weight_pos": mean_weight_pos,
        "actor/v7_mean_weight_neg": mean_weight_neg,
        "actor/v7_max_weight_pos": max_weight_pos,
        "actor/v7_max_weight_neg": max_weight_neg,
        "actor/v7_objective_mean": objective_mean,
    }
    pg_metrics.update(gamma_metrics)

    return pg_loss, pg_metrics


def compute_is_reshape_gamma_sym(
    log_w: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    rho_min: float = 0.3,
    gamma_min: float = 0.05,
    gamma_max: float = 0.99,
) -> tuple[tuple[float, float], dict[str, Any]]:
    """Compute symmetric gamma for IS Reshape.

    This computes gamma for positive samples using the standard ESS-based formula,
    then derives gamma for negative samples as symmetric around 1:
        γ- = 2 - γ+

    This ensures:
        - γ+ ∈ (0, 1) for positive samples
        - γ- ∈ (1, 2) for negative samples
        - γ+ and γ- are symmetric around 1

    Args:
        log_w: Log importance weights, shape (batch_size, seq_length)
        advantages: Advantage values, shape (batch_size, seq_length)
        response_mask: Valid token mask, shape (batch_size, seq_length)
        rho_min: Minimum ESS ratio threshold (default: 0.3)
        gamma_min: Minimum γ for positive samples (default: 0.05)
        gamma_max: Maximum γ for positive samples (default: 0.99)

    Returns:
        (gamma_pos, gamma_neg): Tuple of gamma values (symmetric around 1)
        metrics: Dictionary containing diagnostic metrics
    """
    with torch.no_grad():
        # Create mask for valid tokens
        mask = response_mask > 0

        # Extract all valid log_w (use entire batch for variance estimation)
        log_w_flat = log_w[mask]

        # Compute γ+ using standard ESS-based formula
        if log_w_flat.numel() < 2:
            gamma_pos = (gamma_min + gamma_max) / 2  # Default to middle
            sigma_sq = 0.0
        else:
            sigma_sq = torch.var(log_w_flat, unbiased=True).item()
            if sigma_sq < 1e-8:
                gamma_pos = gamma_max  # Near on-policy, can use high gamma
            else:
                # γ+ = min(γ_max, √(-log ρ_min / σ²))
                gamma_pos = min(gamma_max, math.sqrt(-math.log(rho_min) / sigma_sq))
                gamma_pos = max(gamma_min, gamma_pos)

        # Symmetric gamma: γ- = 2 - γ+
        gamma_neg = 2.0 - gamma_pos

        # Compute ESS ratios for diagnostics
        ess_ratio_pos = math.exp(-sigma_sq * gamma_pos * gamma_pos) if sigma_sq > 0 else 1.0
        ess_ratio_neg = math.exp(-sigma_sq * (gamma_neg - 1) * (gamma_neg - 1)) if sigma_sq > 0 else 1.0

        # Count positive and negative samples
        pos_mask = mask & (advantages > 0)
        neg_mask = mask & (advantages < 0)

        metrics = {
            "is_reshape_sym/gamma_pos": gamma_pos,
            "is_reshape_sym/gamma_neg": gamma_neg,
            "is_reshape_sym/sigma_sq": sigma_sq,
            "is_reshape_sym/n_pos": pos_mask.sum().item(),
            "is_reshape_sym/n_neg": neg_mask.sum().item(),
            "is_reshape_sym/ess_ratio_pos": ess_ratio_pos,
            "is_reshape_sym/ess_ratio_neg": ess_ratio_neg,
        }

        return (gamma_pos, gamma_neg), metrics


@register_policy_loss("is_reshape_sym")
def compute_policy_loss_is_reshape_sym(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute policy loss using IS Reshape with symmetric gamma around 1.

    This implements a symmetric version of IS Reshape where:
        - γ+ is computed using ESS-based formula (same as v7)
        - γ- = 2 - γ+ (symmetric around 1)

    Key insight:
        If γ+ = 0.5, then γ- = 1.5
        If γ+ = 0.3, then γ- = 1.7
        This maintains the property that positive samples use concave (γ < 1)
        and negative samples use convex (γ > 1) weight functions.

    Args:
        old_log_prob: Log probabilities from behavior policy μ
            Shape: (batch_size, response_length)
        log_prob: Log probabilities from current policy π_θ
            Shape: (batch_size, response_length)
        advantages: Advantage estimates
            Shape: (batch_size, response_length)
        response_mask: Valid token mask
            Shape: (batch_size, response_length)
        loss_agg_mode: Loss aggregation strategy
        config: Actor config, should contain:
            - is_reshape_sym.rho_min: Min ESS ratio (default: 0.3)
            - is_reshape_sym.gamma_min: Min γ for positive samples (default: 0.05)
            - is_reshape_sym.gamma_max: Max γ for positive samples (default: 0.99)
            - is_reshape_sym.clip_weight: Whether to clip weights (default: True)
            - is_reshape_sym.clip_threshold: Weight clipping threshold (default: 10.0)
        rollout_is_weights: Ignored (for interface compatibility)

    Returns:
        pg_loss: Policy gradient loss
        pg_metrics: Dictionary of metrics

    Config example:
        actor:
          policy_loss:
            loss_mode: "is_reshape_sym"
            is_reshape_sym:
              rho_min: 0.3
              gamma_min: 0.05
              gamma_max: 0.99
              clip_weight: true
              clip_threshold: 10.0
    """
    assert config is not None, "ActorConfig must be provided for IS Reshape Sym"

    # Extract config
    sym_config = config.policy_loss.get("is_reshape_sym", {}) if hasattr(config, "policy_loss") else {}
    rho_min = sym_config.get("rho_min", 0.3)
    gamma_min = sym_config.get("gamma_min", 0.05)
    gamma_max = sym_config.get("gamma_max", 1.0)
    clip_weight = sym_config.get("clip_weight", True)
    clip_threshold = sym_config.get("clip_threshold", 10.0)

    # Step 1: Compute log importance weights
    log_w = log_prob - old_log_prob
    log_w = torch.clamp(log_w, min=-20.0, max=20.0)

    # Step 2: Compute symmetric γ+ and γ-
    (gamma_pos, gamma_neg), gamma_metrics = compute_is_reshape_gamma_sym(
        log_w=log_w,
        advantages=advantages,
        response_mask=response_mask,
        rho_min=rho_min,
        gamma_min=gamma_min,
        gamma_max=gamma_max,
    )

    # Step 3: Create gamma tensor based on advantage sign
    gamma = torch.ones_like(advantages)
    pos_mask = advantages > 0
    neg_mask = advantages < 0
    gamma[pos_mask] = gamma_pos
    gamma[neg_mask] = gamma_neg

    # Step 4: Compute w^γ = exp(γ · log w)
    w_gamma = torch.exp(gamma * log_w)

    # Optional: clip weights for stability
    if clip_weight:
        w_gamma = torch.clamp(w_gamma, max=clip_threshold)

    # Step 5: Compute Box-Cox objective: (w^γ - 1) / γ · A
    safe_gamma = torch.where(
        gamma.abs() < 0.01,
        torch.ones_like(gamma),
        gamma
    )

    box_cox = torch.where(
        gamma.abs() < 0.01,
        log_w,  # γ → 0 limit
        (w_gamma - 1) / safe_gamma
    )

    # Objective: maximize E[(w^γ - 1)/γ · A] = minimize -E[(w^γ - 1)/γ · A]
    loss_mat = -box_cox * advantages

    # Use agg_loss to properly aggregate
    pg_loss = agg_loss(loss_mat, response_mask, loss_agg_mode)

    # Compute diagnostics
    with torch.no_grad():
        mask = response_mask > 0

        # KL divergence
        ppo_kl = verl_F.masked_mean(-log_w, response_mask)

        # Original IS ratio
        w_original = torch.exp(log_w)
        is_ratio_original = verl_F.masked_mean(w_original, response_mask)

        # Reshaped IS ratio
        is_ratio_used = verl_F.masked_mean(w_gamma, response_mask)

        # Weight statistics
        w_gamma_flat = w_gamma[mask]
        if w_gamma_flat.numel() > 0:
            max_weight = w_gamma_flat.max().item()
            mean_weight = verl_F.masked_mean(w_gamma, response_mask).item()

            # Actual ESS
            ess = (w_gamma_flat.sum() ** 2) / (w_gamma_flat**2).sum()
            ess_ratio = (ess / w_gamma_flat.numel()).item()
        else:
            max_weight = 1.0
            mean_weight = 1.0
            ess_ratio = 1.0

        # Per-group statistics
        w_gamma_pos = w_gamma[pos_mask & mask]
        w_gamma_neg = w_gamma[neg_mask & mask]

        if w_gamma_pos.numel() > 0:
            mean_weight_pos = w_gamma_pos.mean().item()
        else:
            mean_weight_pos = 1.0

        if w_gamma_neg.numel() > 0:
            mean_weight_neg = w_gamma_neg.mean().item()
        else:
            mean_weight_neg = 1.0

        # Box-Cox objective statistics
        objective = box_cox * advantages
        objective_mean = verl_F.masked_mean(objective, response_mask).item()

    # Collect metrics
    pg_metrics = {
        "actor/ppo_kl": ppo_kl.item(),
        "actor/is_ratio_original": is_ratio_original.item(),
        "actor/is_ratio_used": is_ratio_used.item(),
        "actor/is_reshape_max_weight": max_weight,
        "actor/is_reshape_mean_weight": mean_weight,
        "actor/is_reshape_ess_ratio_actual": ess_ratio,
        # sym-specific metrics
        "actor/sym_mean_weight_pos": mean_weight_pos,
        "actor/sym_mean_weight_neg": mean_weight_neg,
        "actor/sym_objective_mean": objective_mean,
    }
    pg_metrics.update(gamma_metrics)

    return pg_loss, pg_metrics


@register_policy_loss("is_reshape_static")
def compute_policy_loss_is_reshape_static(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute policy loss using IS Reshape with static symmetric gamma values.

    This implements IS Reshape with fixed gamma values:
        - γ+ = 0.5 for positive samples (A > 0)
        - γ- = 1.5 for negative samples (A < 0)

    These values are symmetric around 1 (0.5 + 1.5 = 2, midpoint = 1).

    Key properties:
        - γ+ = 0.5: Concave w^0.5 = √w
          → amplifies w < 1 (new good samples)
          → shrinks w > 1 (known good samples)

        - γ- = 1.5: Convex w^1.5 = w√w
          → shrinks w < 1 (already avoided bad samples)
          → amplifies w > 1 (not yet avoided bad samples)

    Args:
        old_log_prob: Log probabilities from behavior policy μ
            Shape: (batch_size, response_length)
        log_prob: Log probabilities from current policy π_θ
            Shape: (batch_size, response_length)
        advantages: Advantage estimates
            Shape: (batch_size, response_length)
        response_mask: Valid token mask
            Shape: (batch_size, response_length)
        loss_agg_mode: Loss aggregation strategy
        config: Actor config, should contain:
            - is_reshape_static.gamma_pos: γ for positive samples (default: 0.5)
            - is_reshape_static.gamma_neg: γ for negative samples (default: 1.5)
            - is_reshape_static.clip_weight: Whether to clip weights (default: True)
            - is_reshape_static.clip_threshold: Weight clipping threshold (default: 10.0)
        rollout_is_weights: Ignored (for interface compatibility)

    Returns:
        pg_loss: Policy gradient loss
        pg_metrics: Dictionary of metrics

    Config example:
        actor:
          policy_loss:
            loss_mode: "is_reshape_static"
            is_reshape_static:
              gamma_pos: 0.5       # Fixed γ for positive samples
              gamma_neg: 1.5       # Fixed γ for negative samples (symmetric: 2 - 0.5)
              clip_weight: true
              clip_threshold: 10.0
    """
    assert config is not None, "ActorConfig must be provided for IS Reshape Static"

    # Extract config
    static_config = config.policy_loss.get("is_reshape_static", {}) if hasattr(config, "policy_loss") else {}
    gamma_pos = static_config.get("gamma_pos", 0.5)
    gamma_neg = static_config.get("gamma_neg", 1.5)
    clip_weight = static_config.get("clip_weight", True)
    clip_threshold = static_config.get("clip_threshold", 10.0)

    # Step 1: Compute log importance weights
    log_w = log_prob - old_log_prob
    log_w = torch.clamp(log_w, min=-20.0, max=20.0)

    # Step 2: Create gamma tensor based on advantage sign (fixed values)
    gamma = torch.ones_like(advantages)
    pos_mask = advantages > 0
    neg_mask = advantages < 0
    gamma[pos_mask] = gamma_pos
    gamma[neg_mask] = gamma_neg

    # Step 3: Compute w^γ = exp(γ · log w)
    w_gamma = torch.exp(gamma * log_w)

    # Optional: clip weights for stability
    if clip_weight:
        w_gamma = torch.clamp(w_gamma, max=clip_threshold)

    # Step 4: Compute Box-Cox objective: (w^γ - 1) / γ · A
    safe_gamma = torch.where(
        gamma.abs() < 0.01,
        torch.ones_like(gamma),
        gamma
    )

    box_cox = torch.where(
        gamma.abs() < 0.01,
        log_w,  # γ → 0 limit
        (w_gamma - 1) / safe_gamma
    )

    # Objective: maximize E[(w^γ - 1)/γ · A] = minimize -E[(w^γ - 1)/γ · A]
    loss_mat = -box_cox * advantages

    # Use agg_loss to properly aggregate
    pg_loss = agg_loss(loss_mat, response_mask, loss_agg_mode)

    # Compute diagnostics
    with torch.no_grad():
        mask = response_mask > 0

        # KL divergence
        ppo_kl = verl_F.masked_mean(-log_w, response_mask)

        # Original IS ratio
        w_original = torch.exp(log_w)
        is_ratio_original = verl_F.masked_mean(w_original, response_mask)

        # Reshaped IS ratio
        is_ratio_used = verl_F.masked_mean(w_gamma, response_mask)

        # Weight statistics
        w_gamma_flat = w_gamma[mask]
        if w_gamma_flat.numel() > 0:
            max_weight = w_gamma_flat.max().item()
            mean_weight = verl_F.masked_mean(w_gamma, response_mask).item()

            # Actual ESS
            ess = (w_gamma_flat.sum() ** 2) / (w_gamma_flat**2).sum()
            ess_ratio = (ess / w_gamma_flat.numel()).item()
        else:
            max_weight = 1.0
            mean_weight = 1.0
            ess_ratio = 1.0

        # Per-group statistics
        w_gamma_pos = w_gamma[pos_mask & mask]
        w_gamma_neg = w_gamma[neg_mask & mask]

        n_pos = pos_mask.sum().item()
        n_neg = neg_mask.sum().item()

        if w_gamma_pos.numel() > 0:
            mean_weight_pos = w_gamma_pos.mean().item()
            max_weight_pos = w_gamma_pos.max().item()
        else:
            mean_weight_pos = 1.0
            max_weight_pos = 1.0

        if w_gamma_neg.numel() > 0:
            mean_weight_neg = w_gamma_neg.mean().item()
            max_weight_neg = w_gamma_neg.max().item()
        else:
            mean_weight_neg = 1.0
            max_weight_neg = 1.0

        # Log w statistics for diagnostics
        log_w_flat = log_w[mask]
        if log_w_flat.numel() > 0:
            sigma_sq = torch.var(log_w_flat, unbiased=True).item()
        else:
            sigma_sq = 0.0

        # Box-Cox objective statistics
        objective = box_cox * advantages
        objective_mean = verl_F.masked_mean(objective, response_mask).item()

    # Collect metrics
    pg_metrics = {
        "actor/ppo_kl": ppo_kl.item(),
        "actor/is_ratio_original": is_ratio_original.item(),
        "actor/is_ratio_used": is_ratio_used.item(),
        "actor/is_reshape_max_weight": max_weight,
        "actor/is_reshape_mean_weight": mean_weight,
        "actor/is_reshape_ess_ratio_actual": ess_ratio,
        # static-specific metrics
        "is_reshape_static/gamma_pos": gamma_pos,
        "is_reshape_static/gamma_neg": gamma_neg,
        "is_reshape_static/sigma_sq": sigma_sq,
        "is_reshape_static/n_pos": n_pos,
        "is_reshape_static/n_neg": n_neg,
        "actor/static_mean_weight_pos": mean_weight_pos,
        "actor/static_mean_weight_neg": mean_weight_neg,
        "actor/static_max_weight_pos": max_weight_pos,
        "actor/static_max_weight_neg": max_weight_neg,
        "actor/static_objective_mean": objective_mean,
    }

    return pg_loss, pg_metrics


@register_policy_loss("is_reshape_harmonic")
def compute_policy_loss_is_reshape_harmonic(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute policy loss using IS Reshape with harmonic weight functions.

    This implements IS Reshape with asymmetric harmonic-style weights:
        - A > 0: h+(w) = 1/(w + c)     (convex, decaying)
        - A < 0: h-(w) = w/(1 + cw)    (concave, growing)

    Key properties:
        h+(w) = 1/(w + c):
            - w → 0: h+ → 1/c (bounded, amplifies new good samples)
            - w = 1: h+ = 1/(1+c)
            - w → ∞: h+ → 0 (suppresses known good samples, variance reduction)

        h-(w) = w/(1 + cw):
            - w → 0: h- → 0 (suppresses already-avoided bad samples)
            - w = 1: h- = 1/(1+c) (same as h+ at w=1)
            - w → ∞: h- → 1/c (bounded, amplifies un-avoided bad samples)

    Both functions are bounded and meet at w=1, providing:
        1. Automatic variance control (no exploding weights)
        2. Soft trust region (implicit clipping)
        3. Asymmetric treatment matching learning objectives

    Args:
        old_log_prob: Log probabilities from behavior policy μ
            Shape: (batch_size, response_length)
        log_prob: Log probabilities from current policy π_θ
            Shape: (batch_size, response_length)
        advantages: Advantage estimates
            Shape: (batch_size, response_length)
        response_mask: Valid token mask
            Shape: (batch_size, response_length)
        loss_agg_mode: Loss aggregation strategy
        config: Actor config, should contain:
            - is_reshape_harmonic.c: Smoothing constant (default: 0.1)
            - is_reshape_harmonic.clip_ratio: Max w for numerical stability (default: 100.0)
        rollout_is_weights: Ignored (for interface compatibility)

    Returns:
        pg_loss: Policy gradient loss
        pg_metrics: Dictionary of metrics

    Config example:
        actor:
          policy_loss:
            loss_mode: "is_reshape_harmonic"
            is_reshape_harmonic:
              c: 0.1            # Smoothing constant (smaller = sharper transition)
              clip_ratio: 100.0 # Max IS ratio for stability
    """
    assert config is not None, "ActorConfig must be provided for IS Reshape Harmonic"

    # Extract config
    harmonic_config = config.policy_loss.get("is_reshape_harmonic", {}) if hasattr(config, "policy_loss") else {}
    c = harmonic_config.get("c", 0.1)
    clip_ratio = harmonic_config.get("clip_ratio", 100.0)

    # Step 1: Compute log importance weights and ratio
    log_w = log_prob - old_log_prob
    log_w = torch.clamp(log_w, min=-20.0, max=20.0)
    w = torch.exp(log_w)

    # Clip w for numerical stability
    w = torch.clamp(w, min=1e-6, max=clip_ratio)

    # Step 2: Compute asymmetric harmonic weights
    # h+(w) = 1/(w + c) for A > 0 (convex, decaying)
    # h-(w) = w/(1 + cw) for A < 0 (concave, growing)
    pos_mask = advantages > 0
    neg_mask = advantages < 0

    # Compute both weight functions
    h_pos = 1.0 / (w + c)           # Convex: 1/(w + c)
    h_neg = w / (1.0 + c * w)       # Concave: w/(1 + cw)

    # Select weight based on advantage sign
    # For A = 0, use weight = 1/(1+c) (the meeting point)
    weight = torch.where(
        pos_mask,
        h_pos,
        torch.where(neg_mask, h_neg, torch.full_like(w, 1.0 / (1.0 + c)))
    )

    # Step 3: Policy gradient loss
    # L = -E_μ[h(w) · A]
    # Note: weight is NOT detached - gradients flow through w via log_prob
    pg_loss = -(weight * advantages * response_mask).sum() / (response_mask.sum() + 1e-8)

    # Compute diagnostics
    with torch.no_grad():
        mask = response_mask > 0

        # KL divergence
        ppo_kl = verl_F.masked_mean(-log_w, response_mask)

        # IS ratio metrics
        is_ratio_original = verl_F.masked_mean(w, response_mask)
        is_ratio_used = verl_F.masked_mean(weight, response_mask)

        # Weight statistics
        weight_flat = weight[mask]
        if weight_flat.numel() > 0:
            max_weight = weight_flat.max().item()
            min_weight = weight_flat.min().item()
            mean_weight = verl_F.masked_mean(weight, response_mask).item()
        else:
            max_weight = 1.0
            min_weight = 1.0
            mean_weight = 1.0

        # Per-group statistics
        h_pos_flat = h_pos[pos_mask & mask]
        h_neg_flat = h_neg[neg_mask & mask]

        n_pos = (pos_mask & mask).sum().item()
        n_neg = (neg_mask & mask).sum().item()

        if h_pos_flat.numel() > 0:
            mean_weight_pos = h_pos_flat.mean().item()
            max_weight_pos = h_pos_flat.max().item()
            min_weight_pos = h_pos_flat.min().item()
        else:
            mean_weight_pos = 1.0 / (1.0 + c)
            max_weight_pos = 1.0 / (1.0 + c)
            min_weight_pos = 1.0 / (1.0 + c)

        if h_neg_flat.numel() > 0:
            mean_weight_neg = h_neg_flat.mean().item()
            max_weight_neg = h_neg_flat.max().item()
            min_weight_neg = h_neg_flat.min().item()
        else:
            mean_weight_neg = 1.0 / (1.0 + c)
            max_weight_neg = 1.0 / (1.0 + c)
            min_weight_neg = 1.0 / (1.0 + c)

        # w statistics
        w_flat = w[mask]
        if w_flat.numel() > 0:
            w_mean = w_flat.mean().item()
            w_max = w_flat.max().item()
            w_min = w_flat.min().item()
        else:
            w_mean = 1.0
            w_max = 1.0
            w_min = 1.0

        # Objective statistics
        objective = weight * advantages
        objective_mean = verl_F.masked_mean(objective, response_mask).item()

    # Collect metrics
    pg_metrics = {
        "actor/ppo_kl": ppo_kl.item(),
        "actor/is_ratio_original": is_ratio_original.item(),
        "actor/is_ratio_used": is_ratio_used.item(),
        "actor/is_reshape_max_weight": max_weight,
        "actor/is_reshape_min_weight": min_weight,
        "actor/is_reshape_mean_weight": mean_weight,
        # harmonic-specific metrics
        "is_reshape_harmonic/c": c,
        "is_reshape_harmonic/n_pos": n_pos,
        "is_reshape_harmonic/n_neg": n_neg,
        "is_reshape_harmonic/w_mean": w_mean,
        "is_reshape_harmonic/w_max": w_max,
        "is_reshape_harmonic/w_min": w_min,
        "actor/harmonic_mean_weight_pos": mean_weight_pos,
        "actor/harmonic_mean_weight_neg": mean_weight_neg,
        "actor/harmonic_max_weight_pos": max_weight_pos,
        "actor/harmonic_max_weight_neg": max_weight_neg,
        "actor/harmonic_min_weight_pos": min_weight_pos,
        "actor/harmonic_min_weight_neg": min_weight_neg,
        "actor/harmonic_objective_mean": objective_mean,
    }

    return pg_loss, pg_metrics


@register_policy_loss("grpo_clip_asymmetric")
def compute_policy_loss_grpo_clip_asymmetric(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute policy loss with asymmetric PPO clipping.

    This implements PPO/GRPO clipping that only applies to one sign of advantage:
        - clip_pos_only: Only clip when A > 0, use raw ratio when A <= 0
        - clip_neg_only: Only clip when A < 0, use raw ratio when A >= 0

    Standard PPO objective:
        L = min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)

    This variant selectively applies the min() based on advantage sign:
        - clip_pos_only: Apply min() only for A > 0
        - clip_neg_only: Apply min() only for A < 0

    Motivation:
        - clip_pos_only: Trust the policy more for avoiding bad actions (A < 0),
          but be conservative for good actions to prevent over-optimization
        - clip_neg_only: Trust the policy more for reinforcing good actions (A > 0),
          but be conservative for bad actions to ensure proper avoidance

    Args:
        old_log_prob: Log probabilities from behavior policy μ
            Shape: (batch_size, response_length)
        log_prob: Log probabilities from current policy π_θ
            Shape: (batch_size, response_length)
        advantages: Advantage estimates
            Shape: (batch_size, response_length)
        response_mask: Valid token mask
            Shape: (batch_size, response_length)
        loss_agg_mode: Loss aggregation strategy
        config: Actor config, should contain:
            - grpo_clip_asymmetric.mode: "clip_pos_only" or "clip_neg_only"
            - grpo_clip_asymmetric.clip_ratio: Clipping epsilon (default: 0.2)
        rollout_is_weights: Ignored (for interface compatibility)

    Returns:
        pg_loss: Policy gradient loss
        pg_metrics: Dictionary of metrics

    Config example:
        actor:
          policy_loss:
            loss_mode: "grpo_clip_asymmetric"
            grpo_clip_asymmetric:
              mode: "clip_pos_only"  # or "clip_neg_only"
              clip_ratio: 0.2        # epsilon for clipping
    """
    assert config is not None, "ActorConfig must be provided for GRPO Clip Asymmetric"

    # Extract config
    asym_config = config.policy_loss.get("grpo_clip_asymmetric", {}) if hasattr(config, "policy_loss") else {}
    mode = asym_config.get("mode", "clip_pos_only")
    clip_ratio_config = asym_config.get("clip_ratio", None)

    # Use config clip_ratio or fall back to actor's clip_ratio
    if clip_ratio_config is not None:
        clip_ratio = clip_ratio_config
    elif hasattr(config, "clip_ratio"):
        clip_ratio = config.clip_ratio
    else:
        clip_ratio = 0.2  # default

    # Step 1: Compute log ratio and ratio
    log_ratio = log_prob - old_log_prob
    log_ratio = torch.clamp(log_ratio, min=-20.0, max=20.0)
    ratio = torch.exp(log_ratio)

    # Step 2: Compute clipped ratio
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio)

    # Step 3: Compute PPO objective components
    # Unclipped: ratio * A
    # Clipped: clip(ratio, 1-ε, 1+ε) * A
    pg_obj_unclipped = ratio * advantages
    pg_obj_clipped = clipped_ratio * advantages

    # Standard PPO: pessimistic (min for maximization)
    pg_obj_ppo = torch.min(pg_obj_unclipped, pg_obj_clipped)

    # Step 4: Create masks for positive and negative advantages
    pos_mask = advantages > 0
    neg_mask = advantages < 0
    zero_mask = advantages == 0

    # Step 5: Apply asymmetric clipping based on mode
    if mode == "clip_pos_only":
        # Clip for A > 0, no clip for A <= 0
        # A > 0: use PPO clipped objective
        # A < 0: use unclipped objective (raw ratio * A)
        # A = 0: doesn't matter, both are 0
        pg_obj = torch.where(pos_mask, pg_obj_ppo, pg_obj_unclipped)
    elif mode == "clip_neg_only":
        # Clip for A < 0, no clip for A >= 0
        # A > 0: use unclipped objective (raw ratio * A)
        # A < 0: use PPO clipped objective
        # A = 0: doesn't matter, both are 0
        pg_obj = torch.where(neg_mask, pg_obj_ppo, pg_obj_unclipped)
    else:
        raise ValueError(f"Unknown mode: {mode}. Expected 'clip_pos_only' or 'clip_neg_only'")

    # Step 6: Compute loss (negate for minimization)
    loss_mat = -pg_obj
    pg_loss = agg_loss(loss_mat, response_mask, loss_agg_mode)

    # Compute diagnostics
    with torch.no_grad():
        mask = response_mask > 0

        # KL divergence
        ppo_kl = verl_F.masked_mean(-log_ratio, response_mask)

        # IS ratio metrics
        is_ratio_mean = verl_F.masked_mean(ratio, response_mask)

        # Ratio statistics
        ratio_flat = ratio[mask]
        if ratio_flat.numel() > 0:
            ratio_max = ratio_flat.max().item()
            ratio_min = ratio_flat.min().item()
            ratio_std = ratio_flat.std().item()
        else:
            ratio_max = 1.0
            ratio_min = 1.0
            ratio_std = 0.0

        # Clipping statistics
        # For positive advantages
        ratio_pos = ratio[pos_mask & mask]
        if ratio_pos.numel() > 0:
            clip_frac_pos_high = (ratio_pos > 1.0 + clip_ratio).float().mean().item()
            clip_frac_pos_low = (ratio_pos < 1.0 - clip_ratio).float().mean().item()
            clip_frac_pos = clip_frac_pos_high + clip_frac_pos_low
        else:
            clip_frac_pos_high = 0.0
            clip_frac_pos_low = 0.0
            clip_frac_pos = 0.0

        # For negative advantages
        ratio_neg = ratio[neg_mask & mask]
        if ratio_neg.numel() > 0:
            clip_frac_neg_high = (ratio_neg > 1.0 + clip_ratio).float().mean().item()
            clip_frac_neg_low = (ratio_neg < 1.0 - clip_ratio).float().mean().item()
            clip_frac_neg = clip_frac_neg_high + clip_frac_neg_low
        else:
            clip_frac_neg_high = 0.0
            clip_frac_neg_low = 0.0
            clip_frac_neg = 0.0

        # Overall clipping fraction
        if ratio_flat.numel() > 0:
            clip_frac_total = ((ratio_flat > 1.0 + clip_ratio) | (ratio_flat < 1.0 - clip_ratio)).float().mean().item()
        else:
            clip_frac_total = 0.0

        # Sample counts
        n_pos = pos_mask.sum().item()
        n_neg = neg_mask.sum().item()
        n_zero = zero_mask.sum().item()

        # Effective clipping (how many were actually clipped based on mode)
        if mode == "clip_pos_only":
            effective_clip_frac = clip_frac_pos * (n_pos / max(n_pos + n_neg, 1))
        else:  # clip_neg_only
            effective_clip_frac = clip_frac_neg * (n_neg / max(n_pos + n_neg, 1))

    # Collect metrics
    pg_metrics = {
        "actor/ppo_kl": ppo_kl.item(),
        "actor/is_ratio_mean": is_ratio_mean.item(),
        "actor/ratio_max": ratio_max,
        "actor/ratio_min": ratio_min,
        "actor/ratio_std": ratio_std,
        # Clipping metrics
        "actor/clip_frac_total": clip_frac_total,
        "actor/clip_frac_pos": clip_frac_pos,
        "actor/clip_frac_neg": clip_frac_neg,
        "actor/clip_frac_pos_high": clip_frac_pos_high,
        "actor/clip_frac_pos_low": clip_frac_pos_low,
        "actor/clip_frac_neg_high": clip_frac_neg_high,
        "actor/clip_frac_neg_low": clip_frac_neg_low,
        "actor/effective_clip_frac": effective_clip_frac,
        # Sample counts
        "actor/n_pos": n_pos,
        "actor/n_neg": n_neg,
        "actor/n_zero": n_zero,
        # Config
        "grpo_clip_asymmetric/mode": 0 if mode == "clip_pos_only" else 1,  # 0 for pos, 1 for neg
        "grpo_clip_asymmetric/clip_ratio": clip_ratio,
    }

    return pg_loss, pg_metrics


@register_policy_loss("sapo")
def compute_policy_loss_sapo(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "seq-mean-token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute the Smoothed Advantage Policy Objective (SAPO).

    SAPO uses a smooth gating function instead of hard PPO clipping:
        gate(r, τ) = sigmoid(τ * (r - 1)) * (4/τ)

    This provides a soft trust region that:
        - At r=1: gate ≈ 2/τ (normalized baseline)
        - As r → 0 or r → ∞: gate decays smoothly

    Different temperatures for positive and negative advantages:
        - τ_pos for A > 0: controls learning rate for good actions
        - τ_neg for A < 0: controls correction rate for bad actions

    See https://arxiv.org/pdf/2511.20347 for more details.

    Args:
        old_log_prob: Log probabilities from behavior policy μ
            Shape: (batch_size, response_length)
        log_prob: Log probabilities from current policy π_θ
            Shape: (batch_size, response_length)
        advantages: Advantage estimates
            Shape: (batch_size, response_length)
        response_mask: Valid token mask
            Shape: (batch_size, response_length)
        loss_agg_mode: Aggregation mode. For SAPO, recommended "seq-mean-token-mean"
        config: Actor config, should contain:
            - sapo.tau_pos: Temperature for positive advantages (default: 1.0)
            - sapo.tau_neg: Temperature for negative advantages (default: 1.05)
        rollout_is_weights: Optional IS weights for rollout correction

    Returns:
        pg_loss: Policy gradient loss
        pg_metrics: Dictionary of metrics

    Config example:
        actor:
          policy_loss:
            loss_mode: "sapo"
            sapo:
              tau_pos: 1.0   # Temperature for A > 0
              tau_neg: 1.05  # Temperature for A < 0
    """
    assert config is not None, "ActorConfig must be provided for SAPO"

    # Extract config
    sapo_config = config.policy_loss.get("sapo", {}) if hasattr(config, "policy_loss") else {}
    tau_pos = sapo_config.get("tau_pos", 1.0)
    tau_neg = sapo_config.get("tau_neg", 1.05)

    # Convert to tensors
    tau_pos = torch.as_tensor(tau_pos, dtype=advantages.dtype, device=advantages.device)
    tau_neg = torch.as_tensor(tau_neg, dtype=advantages.dtype, device=advantages.device)

    def gate_function(x: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        """The gating function used in SAPO: sigmoid(τ*(r-1)) * (4/τ)"""
        return torch.sigmoid(tau * (x - 1.0)) * (4.0 / tau)

    # Compute IS ratio at token level: r = π_θ / π_old
    log_ratio = log_prob - old_log_prob
    log_ratio = torch.clamp(log_ratio, min=-20.0, max=20.0)
    ratio = torch.exp(log_ratio)

    # Select tau based on advantage sign: τ_pos if A > 0, else τ_neg
    taus = torch.where(advantages > 0, tau_pos, tau_neg)

    # Compute gates: f(r, τ)
    gates = gate_function(ratio, taus)

    # Compute policy gradient loss: L = -gate * A
    pg_losses = -gates * advantages

    # Apply rollout correction weights if provided
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    # Aggregate loss (SAPO recommends seq-mean-token-mean)
    pg_loss = agg_loss(pg_losses, response_mask, loss_agg_mode)

    # Compute diagnostics
    with torch.no_grad():
        mask = response_mask > 0

        # KL divergence
        ppo_kl = verl_F.masked_mean(-log_ratio, response_mask)

        # Ratio statistics
        ratio_flat = ratio[mask]
        if ratio_flat.numel() > 0:
            ratio_mean = ratio_flat.mean().item()
            ratio_max = ratio_flat.max().item()
            ratio_min = ratio_flat.min().item()
        else:
            ratio_mean = 1.0
            ratio_max = 1.0
            ratio_min = 1.0

        # Gate statistics
        gates_flat = gates[mask]
        if gates_flat.numel() > 0:
            gate_mean = gates_flat.mean().item()
            gate_max = gates_flat.max().item()
            gate_min = gates_flat.min().item()
        else:
            gate_mean = 2.0 / tau_pos.item()
            gate_max = gate_mean
            gate_min = gate_mean

        # Per-sign statistics
        pos_mask = (advantages > 0) & mask
        neg_mask = (advantages <= 0) & mask

        gates_pos = gates[pos_mask]
        gates_neg = gates[neg_mask]

        n_pos = pos_mask.sum().item()
        n_neg = neg_mask.sum().item()

        if gates_pos.numel() > 0:
            gate_pos_mean = gates_pos.mean().item()
        else:
            gate_pos_mean = 2.0 / tau_pos.item()

        if gates_neg.numel() > 0:
            gate_neg_mean = gates_neg.mean().item()
        else:
            gate_neg_mean = 2.0 / tau_neg.item()

    # Collect metrics
    pg_metrics = {
        "actor/ppo_kl": ppo_kl.item(),
        "actor/ratio_mean": ratio_mean,
        "actor/ratio_max": ratio_max,
        "actor/ratio_min": ratio_min,
        # SAPO-specific metrics
        "sapo/gate_mean": gate_mean,
        "sapo/gate_max": gate_max,
        "sapo/gate_min": gate_min,
        "sapo/gate_pos_mean": gate_pos_mean,
        "sapo/gate_neg_mean": gate_neg_mean,
        "sapo/tau_pos": tau_pos.item(),
        "sapo/tau_neg": tau_neg.item(),
        "sapo/n_pos": n_pos,
        "sapo/n_neg": n_neg,
        # For compatibility with PPO metrics
        "actor/pg_clipfrac": 0.0,
        "actor/pg_clipfrac_lower": 0.0,
    }

    return pg_loss, pg_metrics


@register_policy_loss("sapo_is")
def compute_policy_loss_sapo_is(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "seq-mean-token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute SAPO with IS-Reshape aware asymmetric gating.

    This improves upon standard SAPO by using opposite sigmoid directions
    for positive and negative advantages, matching IS-Reshape theory:

    For A > 0 (good samples) - CONCAVE-like behavior:
        gate_pos(w) = 2 * sigmoid(-τ_pos * (w - 1))
        - w < 1 (new good samples): gate > 1 → amplify learning
        - w > 1 (known good samples): gate < 1 → reduce variance

    For A < 0 (bad samples) - CONVEX-like behavior:
        gate_neg(w) = 2 * sigmoid(τ_neg * (w - 1))
        - w < 1 (already avoided): gate < 1 → reduce gradient
        - w > 1 (not yet avoided): gate > 1 → amplify punishment

    Key properties:
        - Both gates are bounded in [0, 2]
        - Both gates equal 1 at w=1 (normalized)
        - Smooth and differentiable everywhere
        - Asymmetric treatment matches learning objectives

    Args:
        old_log_prob: Log probabilities from behavior policy μ
            Shape: (batch_size, response_length)
        log_prob: Log probabilities from current policy π_θ
            Shape: (batch_size, response_length)
        advantages: Advantage estimates
            Shape: (batch_size, response_length)
        response_mask: Valid token mask
            Shape: (batch_size, response_length)
        loss_agg_mode: Aggregation mode. Recommended "seq-mean-token-mean"
        config: Actor config, should contain:
            - sapo_is.tau_pos: Temperature for A > 0 (default: 1.0)
            - sapo_is.tau_neg: Temperature for A < 0 (default: 1.0)
        rollout_is_weights: Optional IS weights for rollout correction

    Returns:
        pg_loss: Policy gradient loss
        pg_metrics: Dictionary of metrics

    Config example:
        actor:
          policy_loss:
            loss_mode: "sapo_is"
            sapo_is:
              tau_pos: 1.0   # Temperature for A > 0 (controls concave sharpness)
              tau_neg: 1.0   # Temperature for A < 0 (controls convex sharpness)

    Comparison with standard SAPO:
        Standard SAPO: gate = sigmoid(τ*(w-1)) * (4/τ) for both signs
        SAPO-IS:       gate_pos = 2*sigmoid(-τ*(w-1))  # Flipped for A > 0
                       gate_neg = 2*sigmoid(τ*(w-1))   # Original for A < 0
    """
    assert config is not None, "ActorConfig must be provided for SAPO-IS"

    # Extract config
    sapo_is_config = config.policy_loss.get("sapo_is", {}) if hasattr(config, "policy_loss") else {}
    tau_pos = sapo_is_config.get("tau_pos", 1.0)
    tau_neg = sapo_is_config.get("tau_neg", 1.0)

    # Convert to tensors
    tau_pos = torch.as_tensor(tau_pos, dtype=advantages.dtype, device=advantages.device)
    tau_neg = torch.as_tensor(tau_neg, dtype=advantages.dtype, device=advantages.device)

    # Compute IS ratio at token level: r = π_θ / π_old
    log_ratio = log_prob - old_log_prob
    log_ratio = torch.clamp(log_ratio, min=-20.0, max=20.0)
    ratio = torch.exp(log_ratio)

    # Create masks
    pos_mask = advantages > 0
    neg_mask = advantages <= 0

    # Compute asymmetric gates:
    # For A > 0: gate = 2 * sigmoid(-τ_pos * (w - 1))  [FLIPPED - concave-like]
    # For A < 0: gate = 2 * sigmoid(τ_neg * (w - 1))   [ORIGINAL - convex-like]
    #
    # The key insight: flipping the sign of τ flips the sigmoid direction
    # - Flipped sigmoid: higher for w < 1, lower for w > 1 (concave behavior)
    # - Original sigmoid: lower for w < 1, higher for w > 1 (convex behavior)

    gate_pos = 2.0 * torch.sigmoid(-tau_pos * (ratio - 1.0))
    gate_neg = 2.0 * torch.sigmoid(tau_neg * (ratio - 1.0))

    # Select gate based on advantage sign
    gates = torch.where(pos_mask, gate_pos, gate_neg)

    # Compute policy gradient loss: L = -gate * A
    pg_losses = -gates * advantages

    # Apply rollout correction weights if provided
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    # Aggregate loss
    pg_loss = agg_loss(pg_losses, response_mask, loss_agg_mode)

    # Compute diagnostics
    with torch.no_grad():
        mask = response_mask > 0

        # KL divergence
        ppo_kl = verl_F.masked_mean(-log_ratio, response_mask)

        # Ratio statistics
        ratio_flat = ratio[mask]
        if ratio_flat.numel() > 0:
            ratio_mean = ratio_flat.mean().item()
            ratio_max = ratio_flat.max().item()
            ratio_min = ratio_flat.min().item()
        else:
            ratio_mean = 1.0
            ratio_max = 1.0
            ratio_min = 1.0

        # Gate statistics
        gates_flat = gates[mask]
        if gates_flat.numel() > 0:
            gate_mean = gates_flat.mean().item()
            gate_max = gates_flat.max().item()
            gate_min = gates_flat.min().item()
        else:
            gate_mean = 1.0
            gate_max = 1.0
            gate_min = 1.0

        # Per-sign gate statistics
        pos_valid = pos_mask & mask
        neg_valid = neg_mask & mask

        gate_pos_values = gate_pos[pos_valid]
        gate_neg_values = gate_neg[neg_valid]

        n_pos = pos_valid.sum().item()
        n_neg = neg_valid.sum().item()

        if gate_pos_values.numel() > 0:
            gate_pos_mean = gate_pos_values.mean().item()
            gate_pos_max = gate_pos_values.max().item()
            gate_pos_min = gate_pos_values.min().item()
        else:
            gate_pos_mean = 1.0
            gate_pos_max = 1.0
            gate_pos_min = 1.0

        if gate_neg_values.numel() > 0:
            gate_neg_mean = gate_neg_values.mean().item()
            gate_neg_max = gate_neg_values.max().item()
            gate_neg_min = gate_neg_values.min().item()
        else:
            gate_neg_mean = 1.0
            gate_neg_max = 1.0
            gate_neg_min = 1.0

        # Ratio distribution for positive/negative advantages
        ratio_pos = ratio[pos_valid]
        ratio_neg = ratio[neg_valid]

        if ratio_pos.numel() > 0:
            # For A > 0: how many have w < 1 (new good) vs w > 1 (known good)
            frac_new_good = (ratio_pos < 1.0).float().mean().item()
        else:
            frac_new_good = 0.5

        if ratio_neg.numel() > 0:
            # For A < 0: how many have w < 1 (avoided) vs w > 1 (not avoided)
            frac_not_avoided = (ratio_neg > 1.0).float().mean().item()
        else:
            frac_not_avoided = 0.5

    # Collect metrics
    pg_metrics = {
        "actor/ppo_kl": ppo_kl.item(),
        "actor/ratio_mean": ratio_mean,
        "actor/ratio_max": ratio_max,
        "actor/ratio_min": ratio_min,
        # Gate statistics
        "sapo_is/gate_mean": gate_mean,
        "sapo_is/gate_max": gate_max,
        "sapo_is/gate_min": gate_min,
        # Per-sign gate statistics
        "sapo_is/gate_pos_mean": gate_pos_mean,
        "sapo_is/gate_pos_max": gate_pos_max,
        "sapo_is/gate_pos_min": gate_pos_min,
        "sapo_is/gate_neg_mean": gate_neg_mean,
        "sapo_is/gate_neg_max": gate_neg_max,
        "sapo_is/gate_neg_min": gate_neg_min,
        # Config
        "sapo_is/tau_pos": tau_pos.item(),
        "sapo_is/tau_neg": tau_neg.item(),
        # Sample counts
        "sapo_is/n_pos": n_pos,
        "sapo_is/n_neg": n_neg,
        # Distribution diagnostics
        "sapo_is/frac_new_good": frac_new_good,  # Fraction of A>0 samples with w<1
        "sapo_is/frac_not_avoided": frac_not_avoided,  # Fraction of A<0 samples with w>1
        # For compatibility
        "actor/pg_clipfrac": 0.0,
        "actor/pg_clipfrac_lower": 0.0,
    }

    return pg_loss, pg_metrics


@register_policy_loss("sapo_is_mono")
def compute_policy_loss_sapo_is_mono(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "seq-mean-token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute SAPO-IS with monotonically increasing gate functions.

    This fixes the monotonicity issue in sapo_is by using properly designed
    gate functions that are:
    1. Monotonically increasing (required for valid IS)
    2. Bounded (for stability)
    3. Concave-like for A > 0, convex-like for A < 0

    Gate functions:

    For A > 0 (concave, bounded, increasing):
        g_pos(w) = (1 + τ_pos) * w / (τ_pos + w)

        Properties:
        - Monotonically increasing: g'(w) = τ_pos(1+τ_pos)/(τ_pos+w)² > 0 ✓
        - Bounded in [0, 1+τ_pos] ✓
        - g(1) = 1 ✓
        - g(w) > w for w < 1 (amplify new good samples) ✓
        - g(w) < w for w > 1 (reduce variance) ✓
        - Concave: g''(w) < 0 ✓

    For A < 0 (convex-like, bounded, increasing):
        g_neg(w) = (1 - exp(-τ_neg * w² / 2)) / (1 - exp(-τ_neg / 2))

        Properties:
        - Monotonically increasing: g'(w) = τ_neg * w * exp(...) / ... > 0 ✓
        - Bounded ✓
        - g(1) = 1 ✓
        - g(w) < w for w < 1 (reduce unnecessary punishment) ✓
        - g(w) > w for moderate w > 1 (amplify needed punishment) ✓

    Args:
        old_log_prob: Log probabilities from behavior policy μ
        log_prob: Log probabilities from current policy π_θ
        advantages: Advantage estimates
        response_mask: Valid token mask
        loss_agg_mode: Aggregation mode
        config: Actor config with:
            - sapo_is_mono.tau_pos: Temperature for A > 0 (default: 1.0)
            - sapo_is_mono.tau_neg: Temperature for A < 0 (default: 1.0)
        rollout_is_weights: Optional rollout correction weights

    Returns:
        pg_loss: Policy gradient loss
        pg_metrics: Dictionary of metrics

    Config example:
        actor:
          policy_loss:
            loss_mode: "sapo_is_mono"
            sapo_is_mono:
              tau_pos: 1.0  # Higher = more conservative (closer to linear)
              tau_neg: 1.0  # Higher = sharper transition
    """
    assert config is not None, "ActorConfig must be provided for SAPO-IS-Mono"

    # Extract config
    config_dict = config.policy_loss.get("sapo_is_mono", {}) if hasattr(config, "policy_loss") else {}
    tau_pos = config_dict.get("tau_pos", 1.0)
    tau_neg = config_dict.get("tau_neg", 1.0)

    # Ensure tau_pos > 0 and tau_neg > 0
    tau_pos = max(0.01, tau_pos)
    tau_neg = max(0.01, tau_neg)

    # Compute IS ratio at token level: w = π_θ / π_old
    log_ratio = log_prob - old_log_prob
    log_ratio = torch.clamp(log_ratio, min=-20.0, max=20.0)
    w = torch.exp(log_ratio)

    # Clamp w for numerical stability
    w = torch.clamp(w, min=1e-6, max=100.0)

    # Create masks
    pos_mask = advantages > 0
    neg_mask = advantages <= 0

    # Gate for A > 0: g_pos(w) = (1 + τ) * w / (τ + w)
    # This is a Michaelis-Menten style function:
    # - Concave ✓
    # - Monotonically increasing ✓
    # - Bounded by (1 + τ) ✓
    # - g(1) = (1+τ)/(τ+1) = 1 ✓
    # - g(w) > w for w < 1, g(w) < w for w > 1 ✓
    gate_pos = (1.0 + tau_pos) * w / (tau_pos + w)

    # Gate for A < 0: g_neg(w) = (1 - exp(-τw²/2)) / (1 - exp(-τ/2))
    # This is a Gaussian CDF-like function:
    # - Convex-like (initially) ✓
    # - Monotonically increasing ✓
    # - Bounded ✓
    # - g(1) = 1 ✓
    # - g(w) < w for w < 1, g(w) > w for moderate w > 1 ✓
    exp_term = torch.exp(-tau_neg * w * w / 2.0)
    normalizer = 1.0 - math.exp(-tau_neg / 2.0)
    # Avoid division by zero
    normalizer = max(normalizer, 1e-6)
    gate_neg = (1.0 - exp_term) / normalizer

    # Select gate based on advantage sign
    gates = torch.where(pos_mask, gate_pos, gate_neg)

    # Compute policy gradient loss: L = -gate * A
    pg_losses = -gates * advantages

    # Apply rollout correction weights if provided
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    # Aggregate loss
    pg_loss = agg_loss(pg_losses, response_mask, loss_agg_mode)

    # Compute diagnostics
    with torch.no_grad():
        mask = response_mask > 0

        # KL divergence
        ppo_kl = verl_F.masked_mean(-log_ratio, response_mask)

        # Ratio statistics
        w_flat = w[mask]
        if w_flat.numel() > 0:
            w_mean = w_flat.mean().item()
            w_max = w_flat.max().item()
            w_min = w_flat.min().item()
        else:
            w_mean = 1.0
            w_max = 1.0
            w_min = 1.0

        # Gate statistics
        gates_flat = gates[mask]
        if gates_flat.numel() > 0:
            gate_mean = gates_flat.mean().item()
            gate_max = gates_flat.max().item()
            gate_min = gates_flat.min().item()
        else:
            gate_mean = 1.0
            gate_max = 1.0
            gate_min = 1.0

        # Per-sign gate statistics
        pos_valid = pos_mask & mask
        neg_valid = neg_mask & mask

        gate_pos_values = gate_pos[pos_valid]
        gate_neg_values = gate_neg[neg_valid]

        n_pos = pos_valid.sum().item()
        n_neg = neg_valid.sum().item()

        if gate_pos_values.numel() > 0:
            gate_pos_mean = gate_pos_values.mean().item()
            gate_pos_max = gate_pos_values.max().item()
            gate_pos_min = gate_pos_values.min().item()
        else:
            gate_pos_mean = 1.0
            gate_pos_max = 1.0
            gate_pos_min = 1.0

        if gate_neg_values.numel() > 0:
            gate_neg_mean = gate_neg_values.mean().item()
            gate_neg_max = gate_neg_values.max().item()
            gate_neg_min = gate_neg_values.min().item()
        else:
            gate_neg_mean = 1.0
            gate_neg_max = 1.0
            gate_neg_min = 1.0

        # Verify concave/convex behavior by checking g(w) vs w
        w_pos = w[pos_valid]
        w_neg = w[neg_valid]

        # For A > 0: fraction where g(w) > w (should be high for w < 1)
        if w_pos.numel() > 0:
            g_pos_vals = gate_pos[pos_valid]
            frac_amplified_pos = (g_pos_vals > w_pos).float().mean().item()
            # Expected: amplified when w < 1
            frac_w_lt_1_pos = (w_pos < 1.0).float().mean().item()
        else:
            frac_amplified_pos = 0.5
            frac_w_lt_1_pos = 0.5

        # For A < 0: fraction where g(w) > w (should be high for w > 1)
        if w_neg.numel() > 0:
            g_neg_vals = gate_neg[neg_valid]
            frac_amplified_neg = (g_neg_vals > w_neg).float().mean().item()
            # Expected: amplified when w > 1
            frac_w_gt_1_neg = (w_neg > 1.0).float().mean().item()
        else:
            frac_amplified_neg = 0.5
            frac_w_gt_1_neg = 0.5

    # Collect metrics
    pg_metrics = {
        "actor/ppo_kl": ppo_kl.item(),
        "actor/w_mean": w_mean,
        "actor/w_max": w_max,
        "actor/w_min": w_min,
        # Gate statistics
        "sapo_is_mono/gate_mean": gate_mean,
        "sapo_is_mono/gate_max": gate_max,
        "sapo_is_mono/gate_min": gate_min,
        # Per-sign gate statistics
        "sapo_is_mono/gate_pos_mean": gate_pos_mean,
        "sapo_is_mono/gate_pos_max": gate_pos_max,
        "sapo_is_mono/gate_pos_min": gate_pos_min,
        "sapo_is_mono/gate_neg_mean": gate_neg_mean,
        "sapo_is_mono/gate_neg_max": gate_neg_max,
        "sapo_is_mono/gate_neg_min": gate_neg_min,
        # Config
        "sapo_is_mono/tau_pos": tau_pos,
        "sapo_is_mono/tau_neg": tau_neg,
        # Sample counts
        "sapo_is_mono/n_pos": n_pos,
        "sapo_is_mono/n_neg": n_neg,
        # Behavior verification
        "sapo_is_mono/frac_amplified_pos": frac_amplified_pos,  # Should correlate with w < 1
        "sapo_is_mono/frac_amplified_neg": frac_amplified_neg,  # Should correlate with w > 1
        "sapo_is_mono/frac_w_lt_1_pos": frac_w_lt_1_pos,
        "sapo_is_mono/frac_w_gt_1_neg": frac_w_gt_1_neg,
        # For compatibility
        "actor/pg_clipfrac": 0.0,
        "actor/pg_clipfrac_lower": 0.0,
    }

    return pg_loss, pg_metrics


@register_policy_loss("ib_is")
def compute_policy_loss_ib_is(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute policy loss using Information Bottleneck IS framework (v9).

    This implements IS-Reshape v9 based on Information Bottleneck theory:
    1. Views IS weight processing as an information compression problem
    2. Uses Softplus for entropy-regularized soft constraints (Fenchel-Legendre dual)
    3. Applies asymmetric compression for positive/negative advantages

    Core insight from IB theory:
        - For A > 0: Information is concentrated in w < 1 region (new good samples)
                     → Compress w > 1 region (upper bound truncation)
        - For A < 0: Information is concentrated in w > 1 region (un-avoided bad samples)
                     → Compress w < 1 region (lower bound truncation)

    Core formulas:
        A > 0: ρ̃ = C - τ·Softplus((C - ρ)/τ)  [upper bound]
        A < 0: ρ̃ = -C + τ·Softplus((C + ρ)/τ) [lower bound]

    where ρ = log(π/μ), C is bandwidth, τ is temperature.

    Boundary behavior:
        Upper bound (A > 0):
            - ρ → -∞: ρ̃ → ρ (linear, preserve gradient)
            - ρ → +∞: ρ̃ → C (saturate, limit weight)

        Lower bound (A < 0):
            - ρ → -∞: ρ̃ → -C (saturate, limit weight)
            - ρ → +∞: ρ̃ → ρ (linear, preserve gradient)

    Args:
        old_log_prob: Log probabilities from behavior policy μ
            Shape: (batch_size, response_length)
        log_prob: Log probabilities from current policy π_θ
            Shape: (batch_size, response_length)
        advantages: Advantage estimates
            Shape: (batch_size, response_length)
        response_mask: Valid token mask
            Shape: (batch_size, response_length)
        loss_agg_mode: Loss aggregation strategy
        config: Actor config, should contain:
            - ib_is.bandwidth: Default information bandwidth C (default: 0.5)
            - ib_is.temperature: Temperature τ for smoothness (default: 1.0)
            - ib_is.bandwidth_pos: Bandwidth for A > 0 (optional, defaults to bandwidth)
            - ib_is.bandwidth_neg: Bandwidth for A < 0 (optional, defaults to bandwidth)
        rollout_is_weights: Ignored (for interface compatibility)

    Returns:
        pg_loss: Policy gradient loss
        pg_metrics: Dictionary of metrics

    Config example:
        actor:
          policy_loss:
            loss_mode: "ib_is"
            ib_is:
              bandwidth: 0.5       # Default information bandwidth
              temperature: 1.0    # Smoothness (smaller = harder truncation)
              bandwidth_pos: null # Bandwidth for A > 0 (null = use default)
              bandwidth_neg: null # Bandwidth for A < 0 (null = use default)

    References:
        IS-Reshape v9 theory: recipe/is_shape/theory/is_reshape_v9_information_bottleneck.md
    """
    assert config is not None, "ActorConfig must be provided for IB-IS"

    # Extract config
    ib_config = config.policy_loss.get("ib_is", {}) if hasattr(config, "policy_loss") else {}
    bandwidth = ib_config.get("bandwidth", 0.5)
    temperature = ib_config.get("temperature", 1.0)
    bandwidth_pos = ib_config.get("bandwidth_pos", None)
    bandwidth_neg = ib_config.get("bandwidth_neg", None)

    # Use default bandwidth if not specified
    C_pos = bandwidth_pos if bandwidth_pos is not None else bandwidth
    C_neg = bandwidth_neg if bandwidth_neg is not None else bandwidth
    tau = max(0.01, temperature)  # Ensure tau > 0

    # Step 1: Compute log IS ratio ρ = log(π/μ)
    rho = log_prob - old_log_prob
    rho = torch.clamp(rho, -20.0, 20.0)  # Numerical stability

    # Step 2: Asymmetric Softplus truncation based on IB theory
    #
    # For A > 0 (good samples):
    #   - Information in w < 1 region (new good samples to learn)
    #   - Compress w > 1 region → upper bound truncation
    #   - ρ̃ = C - τ·Softplus((C - ρ)/τ)
    #
    # For A < 0 (bad samples):
    #   - Information in w > 1 region (un-avoided bad samples to punish)
    #   - Compress w < 1 region → lower bound truncation
    #   - ρ̃ = -C + τ·Softplus((C + ρ)/τ)

    # Upper bound truncation (for A > 0)
    # ρ̃ = C - τ·Softplus((C - ρ)/τ)
    # Properties: ρ → -∞ gives ρ̃ → ρ; ρ → +∞ gives ρ̃ → C
    delta_upper = (C_pos - rho) / tau
    rho_upper = C_pos - tau * torch.nn.functional.softplus(delta_upper)

    # Lower bound truncation (for A < 0)
    # ρ̃ = -C + τ·Softplus((C + ρ)/τ)
    # Properties: ρ → -∞ gives ρ̃ → -C; ρ → +∞ gives ρ̃ → ρ
    delta_lower = (C_neg + rho) / tau
    rho_lower = -C_neg + tau * torch.nn.functional.softplus(delta_lower)

    # Step 3: Select truncation based on advantage sign
    pos_mask = advantages > 0
    neg_mask = advantages < 0
    # For A = 0, use original ρ (no truncation)
    rho_smooth = torch.where(pos_mask, rho_upper, torch.where(neg_mask, rho_lower, rho))

    # Step 4: Convert to weights
    w_smooth = torch.exp(rho_smooth)

    # Step 5: Compute policy gradient loss
    # L = -E[w̃ · A]
    loss_mat = -w_smooth * advantages
    pg_loss = agg_loss(loss_mat, response_mask, loss_agg_mode)

    # Step 6: Compute diagnostics
    with torch.no_grad():
        mask = response_mask > 0
        w_original = torch.exp(rho)

        # KL divergence
        ppo_kl = verl_F.masked_mean(-rho, response_mask)

        # Weight statistics
        w_smooth_mean = verl_F.masked_mean(w_smooth, response_mask)
        w_original_mean = verl_F.masked_mean(w_original, response_mask)

        w_smooth_flat = w_smooth[mask]
        if w_smooth_flat.numel() > 0:
            w_smooth_max = w_smooth_flat.max().item()
            w_smooth_min = w_smooth_flat.min().item()
        else:
            w_smooth_max = 1.0
            w_smooth_min = 1.0

        # Compression statistics
        rho_flat = rho[mask]
        rho_smooth_flat = rho_smooth[mask]
        if rho_flat.numel() > 0:
            # How much was clipped (on average)
            compression_amount = (rho_flat - rho_smooth_flat).abs().mean().item()
            # Fraction of tokens that hit the upper bound (for A > 0)
            pos_valid = pos_mask & mask
            if pos_valid.any():
                frac_upper_clipped = (rho[pos_valid] > C_pos - 0.1).float().mean().item()
            else:
                frac_upper_clipped = 0.0
            # Fraction of tokens that hit the lower bound (for A < 0)
            neg_valid = neg_mask & mask
            if neg_valid.any():
                frac_lower_clipped = (rho[neg_valid] < -C_neg + 0.1).float().mean().item()
            else:
                frac_lower_clipped = 0.0
        else:
            compression_amount = 0.0
            frac_upper_clipped = 0.0
            frac_lower_clipped = 0.0

        # Per-group statistics
        n_pos = pos_mask.sum().item()
        n_neg = neg_mask.sum().item()

        pos_valid = pos_mask & mask
        neg_valid = neg_mask & mask

        if pos_valid.any():
            w_smooth_pos_mean = w_smooth[pos_valid].mean().item()
            w_smooth_pos_max = w_smooth[pos_valid].max().item()
        else:
            w_smooth_pos_mean = 1.0
            w_smooth_pos_max = 1.0

        if neg_valid.any():
            w_smooth_neg_mean = w_smooth[neg_valid].mean().item()
            w_smooth_neg_min = w_smooth[neg_valid].min().item()
        else:
            w_smooth_neg_mean = 1.0
            w_smooth_neg_min = 1.0

    # Collect metrics
    pg_metrics = {
        "actor/ppo_kl": ppo_kl.item(),
        "actor/w_original_mean": w_original_mean.item(),
        "actor/w_smooth_mean": w_smooth_mean.item(),
        "actor/w_smooth_max": w_smooth_max,
        "actor/w_smooth_min": w_smooth_min,
        # IB-IS specific metrics
        "ib_is/bandwidth": bandwidth,
        "ib_is/bandwidth_pos": C_pos,
        "ib_is/bandwidth_neg": C_neg,
        "ib_is/temperature": tau,
        "ib_is/compression_amount": compression_amount,
        "ib_is/frac_upper_clipped": frac_upper_clipped,
        "ib_is/frac_lower_clipped": frac_lower_clipped,
        # Per-group
        "ib_is/n_pos": n_pos,
        "ib_is/n_neg": n_neg,
        "ib_is/w_smooth_pos_mean": w_smooth_pos_mean,
        "ib_is/w_smooth_pos_max": w_smooth_pos_max,
        "ib_is/w_smooth_neg_mean": w_smooth_neg_mean,
        "ib_is/w_smooth_neg_min": w_smooth_neg_min,
        # For compatibility
        "actor/pg_clipfrac": frac_upper_clipped,
        "actor/pg_clipfrac_lower": frac_lower_clipped,
    }

    return pg_loss, pg_metrics


@register_policy_loss("harmonic_is")
def compute_policy_loss_harmonic_is(
        old_log_prob: torch.Tensor,
        log_prob: torch.Tensor,
        advantages: torch.Tensor,
        response_mask: torch.Tensor,
        loss_agg_mode: str = "token-mean",
        config: Optional[ActorConfig] = None,
        rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute policy loss using Harmonic Importance Sampling (IS-Reshape v10).

    Objective:
        Maximize J(θ) = E[ f(w) * A ]
        where f(w) = C * ln(λw + 1)
        and C = (λ+1)/λ is the normalization factor to ensure gradient=A at w=1.

    Effective Gradient Weight (φ):
        The autograd via chain rule yields: φ(w) = f'(w) * w
        φ(w) = C * [λ / (λw + 1)] * w = (λ+1) * [w / (λw + 1)]

    Properties:
        - At w=1 (On-policy): φ(1) = 1.0 (Matches standard PG/PPO)
        - At w→0 (Unexplored): φ(w) ≈ (λ+1)w (Linear, like IS)
        - At w→∞ (Overfitted): φ(w) → (λ+1)/λ (Saturated, like SFT)

    Lambda behavior (with Normalization):
        - λ = 1.0: φ maxes at 2.0. Balanced RL/SFT. (Recommended)
        - λ → 0: φ approaches Linear IS (Unbounded variance).
        - λ → ∞: φ approaches Step Function (SFT-like, strictly bounded near 1.0).

    Args:
        ... (same as before) ...
    """
    assert config is not None, "ActorConfig must be provided for Harmonic IS"

    # Extract config
    harmonic_config = config.policy_loss.get("harmonic_is", {}) if hasattr(config, "policy_loss") else {}
    lambda_ = harmonic_config.get("lambda_", 1.0)

    # Ensure λ > 0 to avoid division by zero
    lambda_ = max(1e-4, lambda_)

    # Step 1: Compute log IS ratio
    log_w = log_prob - old_log_prob
    # Clip for numerical stability in exp()
    log_w = torch.clamp(log_w, -10.0, 10.0)
    w = torch.exp(log_w)

    # Step 2: Compute objective function f(w)
    #
    # Normalization Factor C = (λ + 1) / λ
    # This ensures that when w=1, the effective gradient weight is 1.0.
    norm_factor = (lambda_ + 1.0) / lambda_

    # f(w) = C * ln(λw + 1)
    # We add 1e-6 inside log just in case, though w>=0 and λ>0 makes it safe.
    f_w = norm_factor * torch.log(lambda_ * w + 1.0)

    # Step 3: Compute policy gradient loss
    # L = -E[f(w) · A]
    # The gradient flow:
    # ∇L = -A · f'(w) · ∇w
    #    = -A · f'(w) · w · ∇logπ
    #    = -A · φ(w) · ∇logπ
    loss_mat = -f_w * advantages
    pg_loss = agg_loss(loss_mat, response_mask, loss_agg_mode)

    # Step 4: Compute diagnostics (Analysis only)
    with torch.no_grad():
        mask = response_mask > 0
        ppo_kl = verl_F.masked_mean(-log_w, response_mask)

        # Re-calculate explicit phi for logging to verify normalization
        # φ(w) = (λ+1) * w / (λw + 1)
        phi = (lambda_ + 1.0) * w / (lambda_ * w + 1.0)

        # Weight statistics
        phi_mean = verl_F.masked_mean(phi, response_mask)
        w_mean = verl_F.masked_mean(w, response_mask)

        # Flatten for min/max
        phi_flat = phi[mask]
        w_flat = w[mask]
        f_w_flat = f_w[mask]

        if phi_flat.numel() > 0:
            phi_max = phi_flat.max().item()
            phi_min = phi_flat.min().item()
            w_max = w_flat.max().item()
            w_std = w_flat.std().item()
            f_w_max = f_w_flat.max().item()
            f_w_min = f_w_flat.min().item()
            f_w_mean = f_w_flat.mean().item()
            f_w_std = f_w_flat.std().item()
        else:
            phi_max = phi_min = 1.0
            w_max = 1.0
            w_std = 0.0
            f_w_max = f_w_min = f_w_mean = 1.0
            f_w_std = 0.0

    # Collect metrics
    pg_metrics = {
        "actor/ppo_kl": ppo_kl.item(),
        "actor/w_mean": w_mean.item(),
        "actor/w_max": w_max,
        "actor/w_std": w_std,
        # Harmonic IS specific metrics
        "harmonic_is/phi_mean": phi_mean.item(),
        "harmonic_is/phi_max": phi_max,
        "harmonic_is/phi_min": phi_min,  # Should be close to 0
        "harmonic_is/f_w_max": f_w_max,
        "harmonic_is/f_w_min": f_w_min,
        "harmonic_is/f_w_mean": f_w_mean,
        "harmonic_is/f_w_std": f_w_std,
        "harmonic_is/lambda": lambda_,
    }

    return pg_loss, pg_metrics



@register_policy_loss("cauchy_is")
def compute_policy_loss_cauchy_is(
        old_log_prob: torch.Tensor,
        log_prob: torch.Tensor,
        advantages: torch.Tensor,
        response_mask: torch.Tensor,
        loss_agg_mode: str = "token-mean",
        config: Optional[ActorConfig] = None,
        rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute policy loss using Normalized Cauchy/Arctan IS (IS-Reshape v12).

    Theory:
        Derived from Min MSE with Advantage-Modulated Variance.
        Optimal weight: φ(w, A) = (1 + λ) * w / (1 + λw²)
        Loss function:  f(w, A) = [(1 + λ) / √λ] * arctan(√λ * w)

        where λ(A) = λ_base * exp(-scale * A)

    Why Normalization (1+λ)?
        We require φ(1) = 1.0 to ensure that for on-policy data (w=1),
        the gradient is unbiased regardless of how large λ is.
        Without this, high λ (safety) would aggressively suppress even valid on-policy gradients.

    Args:
        config.cauchy_is:
            lambda_: Static λ (optional).
            lambda_base: Value of λ when A=0 (Default: 1.0). Controls global Trust Region tightness.
            urgency_scale: Controls asymmetry. λ(A) = λ_base * exp(-scale * A).
                           High scale -> SFT-like for A>0, Hard-Clip for A<0.
    """
    assert config is not None

    # 1. Config Extraction (Simplified)
    cauchy_config = config.policy_loss.get("cauchy_is", {}) if hasattr(config, "policy_loss") else {}
    lambda_fixed = cauchy_config.get("lambda_", None)

    # λ_base replaces the ratio of risk/urgency
    lambda_base = cauchy_config.get("lambda_base", 1.0)
    urgency_scale = cauchy_config.get("urgency_scale", 1.0)

    lambda_min = cauchy_config.get("lambda_min", 1e-4)  # 1e-4 is safer than 0.01 for SFT-like behavior
    lambda_max = cauchy_config.get("lambda_max", 100.0)

    # 2. Compute IS Ratio w
    # Clamp for numerical safety before exp
    log_w = log_prob - old_log_prob
    log_w = torch.clamp(log_w, -10.0, 10.0)
    w = torch.exp(log_w)

    # 3. Compute Dynamic Lambda
    if lambda_fixed is not None:
        lambda_val = torch.full_like(advantages, max(1e-4, lambda_fixed))
        is_static_mode = True
    else:
        # λ(A) = λ_base * exp(-scale * A)
        # Using negative exponent directly is more intuitive: A↑ -> λ↓ (Trust↑)
        lambda_val = lambda_base * torch.exp(-urgency_scale * advantages)
        lambda_val = torch.clamp(lambda_val, lambda_min, lambda_max)
        is_static_mode = False

    # Detach lambda to ensure no gradients flow through the hyperparameter branch
    # (Though theoretically A is already detached, this is good practice)
    lambda_val = lambda_val.detach()

    # 4. Compute Normalized Objective
    # f(w) = C * arctan(√λ * w)
    # C = (1 + λ) / √λ
    sqrt_lambda = torch.sqrt(lambda_val)
    norm_factor = (1.0 + lambda_val) / (sqrt_lambda + 1e-8)

    f_w = norm_factor * torch.atan(sqrt_lambda * w)

    # 5. Compute Loss
    loss_mat = -f_w * advantages
    pg_loss = agg_loss(loss_mat, response_mask, loss_agg_mode)

    # 6. Metrics & Diagnostics
    with torch.no_grad():
        mask = response_mask > 0

        # Reconstruct Effective Gradient Weight φ(w) for checking
        # φ = f'(w) * w = (1+λ) * w / (1 + λw²)
        phi = (1.0 + lambda_val) * w / (1.0 + lambda_val * w ** 2)

        # Flatten for statistics
        w_flat = w[mask]
        phi_flat = phi[mask]
        lambda_flat = lambda_val[mask]
        f_w_flat = f_w[mask]
        adv_flat = advantages[mask]

        # Asymmetry Check
        pos_mask = (advantages > 0) & mask
        neg_mask = (advantages < 0) & mask

        # ===== Basic Stats =====
        metrics = {
            "actor/ppo_kl": verl_F.masked_mean(-log_w, response_mask).item(),
            # Mode: 0 = static, 1 = dynamic
            "cauchy_is/mode": 0.0 if is_static_mode else 1.0,
        }

        # ===== w (IS ratio) statistics =====
        if w_flat.numel() > 0:
            metrics["cauchy_is/w_mean"] = w_flat.mean().item()
            metrics["cauchy_is/w_std"] = w_flat.std().item()
            metrics["cauchy_is/w_max"] = w_flat.max().item()
            metrics["cauchy_is/w_min"] = w_flat.min().item()
            metrics["cauchy_is/w_median"] = w_flat.median().item()
            # Distribution shape
            metrics["cauchy_is/w_gt_1_frac"] = (w_flat > 1.0).float().mean().item()
            metrics["cauchy_is/w_gt_2_frac"] = (w_flat > 2.0).float().mean().item()
            metrics["cauchy_is/w_lt_0.5_frac"] = (w_flat < 0.5).float().mean().item()
        else:
            metrics["cauchy_is/w_mean"] = 1.0
            metrics["cauchy_is/w_std"] = 0.0
            metrics["cauchy_is/w_max"] = 1.0
            metrics["cauchy_is/w_min"] = 1.0
            metrics["cauchy_is/w_median"] = 1.0
            metrics["cauchy_is/w_gt_1_frac"] = 0.0
            metrics["cauchy_is/w_gt_2_frac"] = 0.0
            metrics["cauchy_is/w_lt_0.5_frac"] = 0.0

        # ===== φ (effective gradient weight) statistics =====
        if phi_flat.numel() > 0:
            metrics["cauchy_is/phi_mean"] = phi_flat.mean().item()
            metrics["cauchy_is/phi_std"] = phi_flat.std().item()
            metrics["cauchy_is/phi_max"] = phi_flat.max().item()
            metrics["cauchy_is/phi_min"] = phi_flat.min().item()
            # Verify normalization: φ should be ~1 when w~1
            w_near_1_mask = (w_flat > 0.9) & (w_flat < 1.1)
            if w_near_1_mask.any():
                metrics["cauchy_is/phi_at_w1"] = phi_flat[w_near_1_mask].mean().item()
            else:
                metrics["cauchy_is/phi_at_w1"] = 1.0
        else:
            metrics["cauchy_is/phi_mean"] = 1.0
            metrics["cauchy_is/phi_std"] = 0.0
            metrics["cauchy_is/phi_max"] = 1.0
            metrics["cauchy_is/phi_min"] = 1.0
            metrics["cauchy_is/phi_at_w1"] = 1.0

        # ===== λ statistics =====
        if lambda_flat.numel() > 0:
            metrics["cauchy_is/lambda_mean"] = lambda_flat.mean().item()
            metrics["cauchy_is/lambda_std"] = lambda_flat.std().item()
            metrics["cauchy_is/lambda_max"] = lambda_flat.max().item()
            metrics["cauchy_is/lambda_min"] = lambda_flat.min().item()
        else:
            metrics["cauchy_is/lambda_mean"] = 1.0
            metrics["cauchy_is/lambda_std"] = 0.0
            metrics["cauchy_is/lambda_max"] = 1.0
            metrics["cauchy_is/lambda_min"] = 1.0

        # ===== f(w) (objective function) statistics =====
        if f_w_flat.numel() > 0:
            metrics["cauchy_is/f_w_mean"] = f_w_flat.mean().item()
            metrics["cauchy_is/f_w_std"] = f_w_flat.std().item()
            metrics["cauchy_is/f_w_max"] = f_w_flat.max().item()
            metrics["cauchy_is/f_w_min"] = f_w_flat.min().item()
        else:
            metrics["cauchy_is/f_w_mean"] = 0.0
            metrics["cauchy_is/f_w_std"] = 0.0
            metrics["cauchy_is/f_w_max"] = 0.0
            metrics["cauchy_is/f_w_min"] = 0.0

        # ===== Per-sign (A>0 vs A<0) statistics =====
        # For A > 0 (positive advantages)
        if pos_mask.any():
            w_pos = w[pos_mask]
            phi_pos = phi[pos_mask]
            lambda_pos = lambda_val[pos_mask]
            metrics["cauchy_is/n_pos"] = pos_mask.sum().item()
            metrics["cauchy_is/w_pos_mean"] = w_pos.mean().item()
            metrics["cauchy_is/phi_pos_mean"] = phi_pos.mean().item()
            metrics["cauchy_is/lambda_pos_mean"] = lambda_pos.mean().item()
            # Behavior check: for A>0, we want φ ≈ w (IS-like) when λ is small
            metrics["cauchy_is/phi_over_w_pos"] = (phi_pos / (w_pos + 1e-8)).mean().item()
        else:
            metrics["cauchy_is/n_pos"] = 0
            metrics["cauchy_is/w_pos_mean"] = 1.0
            metrics["cauchy_is/phi_pos_mean"] = 1.0
            metrics["cauchy_is/lambda_pos_mean"] = 1.0
            metrics["cauchy_is/phi_over_w_pos"] = 1.0

        # For A < 0 (negative advantages)
        if neg_mask.any():
            w_neg = w[neg_mask]
            phi_neg = phi[neg_mask]
            lambda_neg = lambda_val[neg_mask]
            metrics["cauchy_is/n_neg"] = neg_mask.sum().item()
            metrics["cauchy_is/w_neg_mean"] = w_neg.mean().item()
            metrics["cauchy_is/phi_neg_mean"] = phi_neg.mean().item()
            metrics["cauchy_is/lambda_neg_mean"] = lambda_neg.mean().item()
            # Behavior check: for A<0, we want φ to be truncated when w is large
            metrics["cauchy_is/phi_over_w_neg"] = (phi_neg / (w_neg + 1e-8)).mean().item()
        else:
            metrics["cauchy_is/n_neg"] = 0
            metrics["cauchy_is/w_neg_mean"] = 1.0
            metrics["cauchy_is/phi_neg_mean"] = 1.0
            metrics["cauchy_is/lambda_neg_mean"] = 1.0
            metrics["cauchy_is/phi_over_w_neg"] = 1.0

        # ===== Four quadrant analysis =====
        # Q1: w < 1, A > 0 (new good samples - should amplify)
        # Q2: w > 1, A > 0 (known good samples - moderate)
        # Q3: w > 1, A < 0 (un-avoided bad samples - should truncate)
        # Q4: w < 1, A < 0 (avoided bad samples - maintain)
        q1_mask = (w < 1) & (advantages > 0) & mask
        q2_mask = (w > 1) & (advantages > 0) & mask
        q3_mask = (w > 1) & (advantages < 0) & mask
        q4_mask = (w < 1) & (advantages < 0) & mask

        metrics["cauchy_is/n_q1"] = q1_mask.sum().item()
        metrics["cauchy_is/n_q2"] = q2_mask.sum().item()
        metrics["cauchy_is/n_q3"] = q3_mask.sum().item()
        metrics["cauchy_is/n_q4"] = q4_mask.sum().item()

        if q1_mask.any():
            metrics["cauchy_is/phi_q1_mean"] = phi[q1_mask].mean().item()
        else:
            metrics["cauchy_is/phi_q1_mean"] = 1.0

        if q2_mask.any():
            metrics["cauchy_is/phi_q2_mean"] = phi[q2_mask].mean().item()
        else:
            metrics["cauchy_is/phi_q2_mean"] = 1.0

        if q3_mask.any():
            metrics["cauchy_is/phi_q3_mean"] = phi[q3_mask].mean().item()
        else:
            metrics["cauchy_is/phi_q3_mean"] = 1.0

        if q4_mask.any():
            metrics["cauchy_is/phi_q4_mean"] = phi[q4_mask].mean().item()
        else:
            metrics["cauchy_is/phi_q4_mean"] = 1.0

        # ===== Gradient contribution analysis =====
        # Effective gradient = φ * A
        grad_contrib = phi * advantages
        grad_contrib_flat = grad_contrib[mask]
        if grad_contrib_flat.numel() > 0:
            metrics["cauchy_is/grad_contrib_mean"] = grad_contrib_flat.mean().item()
            metrics["cauchy_is/grad_contrib_std"] = grad_contrib_flat.std().item()
            metrics["cauchy_is/grad_contrib_pos_sum"] = grad_contrib_flat[grad_contrib_flat > 0].sum().item() if (grad_contrib_flat > 0).any() else 0.0
            metrics["cauchy_is/grad_contrib_neg_sum"] = grad_contrib_flat[grad_contrib_flat < 0].sum().item() if (grad_contrib_flat < 0).any() else 0.0
        else:
            metrics["cauchy_is/grad_contrib_mean"] = 0.0
            metrics["cauchy_is/grad_contrib_std"] = 0.0
            metrics["cauchy_is/grad_contrib_pos_sum"] = 0.0
            metrics["cauchy_is/grad_contrib_neg_sum"] = 0.0

        # ===== Config tracking =====
        if is_static_mode:
            metrics["cauchy_is/lambda_fixed"] = lambda_fixed if lambda_fixed is not None else 0.0
        else:
            metrics["cauchy_is/lambda_base"] = lambda_base
            metrics["cauchy_is/urgency_scale"] = urgency_scale

    return pg_loss, metrics


@register_policy_loss("welsch_is")
def compute_policy_loss_welsch_is(
        old_log_prob: torch.Tensor,
        log_prob: torch.Tensor,
        advantages: torch.Tensor,
        response_mask: torch.Tensor,
        loss_agg_mode: str = "token-mean",
        config: Optional[ActorConfig] = None,
        rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute policy loss using Welsch-IS with Correct Gradient Detachment.

    Gradient Correctness:
        We want the update: Δθ ∝ φ(w) * A * ∇logπ
        where φ(w) = w * Gate(w)

        Using the identity ∇w = w * ∇logπ, we construct the proxy loss:
        L = - (Gate(w).detach() * w) * A

        This ensures the Gate acts as a robust filter/weight, not as an objective
        that pulls w towards 1 (which would cause gradient sign flipping).
    """
    assert config is not None

    # 1. Config
    welsch_config = config.policy_loss.get("welsch_is", {}) if hasattr(config, "policy_loss") else {}
    lambda_base = welsch_config.get("lambda_base", 1.0)
    urgency_scale = welsch_config.get("urgency_scale", 1.0)
    lambda_min = welsch_config.get("lambda_min", 0.1)
    lambda_max = welsch_config.get("lambda_max", 100.0)

    # 2. Compute IS Ratio w (with Gradients)
    log_w = log_prob - old_log_prob
    log_w = torch.clamp(log_w, -10.0, 10.0)
    w = torch.exp(log_w)

    # 3. Compute Dynamic Lambda (Detached by default logic)
    # Note: advantages are usually detached in PPO, but explicitly detaching lambda is safer.
    raw_lambda = lambda_base * torch.exp(-urgency_scale * advantages)
    lambda_val = torch.clamp(raw_lambda, lambda_min, lambda_max).detach()

    # 4. Compute Trust Gate (THE CRITICAL FIX)
    # We calculate the suppression factor based on current w.
    dist_sq = (w - 1.0) ** 2
    trust_gate = torch.exp(-0.5 * lambda_val * dist_sq)

    # === CRITICAL: DETACH THE GATE ===
    # We treat the gate as a "weighting coefficient" computed from data,
    # NOT as part of the optimization landscape.
    # If we don't detach, backprop will try to minimize (w-1)^2 to open the gate,
    # which introduces a "bias towards 1" force that fights the RL objective.
    gate_coeff = trust_gate.detach()

    # 5. Hard Tail Cutoff (Applied to the coefficient)
    # Mask out extreme outliers completely
    # gate_coeff = torch.where(w.detach() > 20.0, torch.zeros_like(gate_coeff), gate_coeff)

    # 6. Construct Loss using the Proxy Form
    # L = - (Weight * w) * A
    # ∇L = - Weight * A * ∇w = - Weight * A * (w ∇logπ) = - (w*Gate) * A * ∇logπ
    phi_proxy = gate_coeff * w
    loss_mat = -phi_proxy * advantages

    pg_loss = agg_loss(loss_mat, response_mask, loss_agg_mode)


    # 7. Metrics (Enhanced Debugging)
    with torch.no_grad():
        mask = response_mask > 0

        # Flatten tensors for statistical analysis
        w_flat = w[mask]
        gate_flat = trust_gate[mask]  # The suppression factor (0.0 ~ 1.0)
        phi_real = w_flat * gate_flat  # The effective gradient weight
        lambda_flat = lambda_val[mask]
        adv_flat = advantages[mask]

        # Basic Stats
        metrics = {
            "actor/ppo_kl": verl_F.masked_mean(-log_w, response_mask).item(),

            # W stats: How far is the policy drifting?
            "welsch_is/w_mean": w_flat.mean().item(),
            "welsch_is/w_std": w_flat.std().item(),
            "welsch_is/w_max": w_flat.max().item(),
            "welsch_is/w_median": w_flat.median().item(),

            # Gate stats: How hard are we braking?
            "welsch_is/gate_mean": gate_flat.mean().item(),
            "welsch_is/gate_min": gate_flat.min().item(),
            "welsch_is/gate_std": gate_flat.std().item(),

            # Phi stats: The actual force applied
            "welsch_is/phi_mean": phi_real.mean().item(),
            "welsch_is/phi_max": phi_real.max().item(),

            # Lambda stats: The stiffness of the trust region
            "welsch_is/lambda_mean": lambda_flat.mean().item(),
        }

        # ===== Advanced Analysis: Quadrants =====
        # We classify samples into 4 Quadrants based on Drift (w) and Quality (A)
        # Q1: w < 1, A > 0  (Underexplored Good) -> Should Boost (Gate ≈ 1)
        # Q2: w > 1, A > 0  (Overexplored Good)  -> Should Moderate (Gate < 1 if w >> 1)
        # Q3: w > 1, A < 0  (Overexplored Bad)   -> DANGER ZONE! Should Kill (Gate ≈ 0)
        # Q4: w < 1, A < 0  (Underexplored Bad)  -> Correctly avoided. Maintain (Gate ≈ 1)

        q1_mask = (w_flat < 1.0) & (adv_flat > 0)
        q2_mask = (w_flat >= 1.0) & (adv_flat > 0)
        q3_mask = (w_flat >= 1.0) & (adv_flat < 0)  # The most critical group for Step 16
        q4_mask = (w_flat < 1.0) & (adv_flat < 0)

        # Count proportions
        total_valid = mask.sum().item()
        if total_valid > 0:
            metrics["welsch_is/n_q3_frac"] = q3_mask.sum().item() / total_valid

        # Analyze Gate behavior per Quadrant
        # This tells us: Is the mechanism acting differently for Good vs Bad samples?
        if q1_mask.any(): metrics["welsch_is/gate_q1_new_good"] = gate_flat[q1_mask].mean().item()
        if q2_mask.any(): metrics["welsch_is/gate_q2_old_good"] = gate_flat[q2_mask].mean().item()

        # *** CRITICAL MONITORING ***
        # If this value is high (>0.5) during step=16, the defense is failing.
        # Ideally, this should be < 0.1 for outliers.
        if q3_mask.any():
            metrics["welsch_is/gate_q3_drift_bad"] = gate_flat[q3_mask].mean().item()
            # Also check the max gate in this region - did ANY bad outlier slip through?
            metrics["welsch_is/gate_q3_max"] = gate_flat[q3_mask].max().item()
            # Check effective weight for these bad samples
            metrics["welsch_is/phi_q3_mean"] = phi_real[q3_mask].mean().item()
        else:
            metrics["welsch_is/gate_q3_drift_bad"] = 1.0  # No bad drift samples

        if q4_mask.any(): metrics["welsch_is/gate_q4_avoid_bad"] = gate_flat[q4_mask].mean().item()

        # ===== Tail Risk Analysis =====
        # Check behavior on extreme outliers (w > 5.0)
        # These are the "Entropy Destroyers"
        outlier_mask = w_flat > 5.0
        if outlier_mask.any():
            metrics["welsch_is/n_outliers"] = outlier_mask.sum().item()
            # How much are we suppressing extremes? (Expect extremely close to 0)
            metrics["welsch_is/gate_outliers"] = gate_flat[outlier_mask].mean().item()
            metrics["welsch_is/phi_outliers"] = phi_real[outlier_mask].mean().item()
        else:
            metrics["welsch_is/n_outliers"] = 0
            metrics["welsch_is/gate_outliers"] = 0.0

        # ===== Asymmetry Check =====
        # Did we successfully make the gate tighter for Negative Advantages?
        pos_mask = adv_flat > 0
        neg_mask = adv_flat < 0
        if pos_mask.any() and neg_mask.any():
            avg_lambda_pos = lambda_flat[pos_mask].mean().item()
            avg_lambda_neg = lambda_flat[neg_mask].mean().item()
            # This ratio should be > 1.0 (Stiffer penalty for bad actions)
            metrics["welsch_is/lambda_asym_ratio"] = avg_lambda_neg / (avg_lambda_pos + 1e-6)

    return pg_loss, metrics


import torch
import torch.nn.functional as F
from typing import Optional, Any, Tuple


@register_policy_loss("gamma_is")
def compute_policy_loss_gamma_is(
        old_log_prob: torch.Tensor,
        log_prob: torch.Tensor,
        advantages: torch.Tensor,
        response_mask: torch.Tensor,
        loss_agg_mode: str = "token-mean",
        config: Optional[Any] = None,  # Type hint adjusted for general usage
        rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Gamma-IS Policy Loss V2 (Robust M-Estimator / Min-MSE variant).

    Theory:
        Optimizing for Minimum Mean Squared Error (MSE) under heavy-tailed
        noise (Log-Normal w) requires a Redescending M-Estimator.

        Weighting Function: φ(w) ∝ w^k * e^{-λw}

        Normalization Constraint: max φ(w) = 1.0.
        (We consider the Peak w as the most trusted "Inlier". We never amplify
         gradients > 1.0, only attenuate less trustworthy samples).

    Hyperparameters:
        k: Controls the "High-pass" filter (suppress irrelevant small w).
           Determines the ascent rate.
        λ: Controls the "Low-pass" filter (suppress high-variance large w).
           Determines the decay rate.

        Peak Location (Most trusted region): w_peak = k / λ
    """
    assert config is not None

    # 1. Config Extraction
    gamma_config = config.policy_loss.get("gamma_is", {}) if hasattr(config, "policy_loss") else {}

    # Positive Advantage: The learning signal
    # Recommendation: k=2.0, lambda=2.0 (Peak=1.0) -> Bell curve
    k_pos = float(gamma_config.get("k_pos", 2.0))
    lambda_pos = float(gamma_config.get("lambda_pos", 2.0))

    # Negative Advantage: The safety constraint
    # Recommendation: k=4.0, lambda=2.5 -> Strong suppression of bad outliers
    k_neg = float(gamma_config.get("k_neg", 4.0))
    lambda_neg = float(gamma_config.get("lambda_neg", 2.5))

    # 2. Compute IS Ratio w
    # log_w = log_prob_new - log_prob_old
    log_w = log_prob - old_log_prob

    # Numerical clamp: Prevent float16 overflow/underflow in exp()
    # -15 to 15 covers w range from 3e-7 to 3e6, sufficient for RL.
    log_w = torch.clamp(log_w, -10.0, 10.0)
    w = torch.exp(log_w)

    # 3. Dynamic Parameter Selection (Vectorized)
    # Create tensors for k and lambda based on advantage sign
    pos_mask = (advantages >= 0).float()
    neg_mask = 1.0 - pos_mask

    k = pos_mask * k_pos + neg_mask * k_neg
    lam = pos_mask * lambda_pos + neg_mask * lambda_neg

    # 4. Compute Normalized Coefficient (The "Detach" Trick)
    # We want the effective gradient scaler to be φ(w).
    # Since standard gradient is derived from (w * log_prob), we need a coeff such that:
    # coeff * w = φ(w)
    # -> coeff = φ(w) / w = Z * w^(k-1) * e^{-λw}

    # 4.1 Calculate Normalization Constant Z so that φ(1) = 1.0
    # Unnormalized Value: M = (1^k) * e^{-λ*1} = e^{-λ}
    # Log Max: ln(M) = k * ln(1) - λ*1 = -λ

    # Small epsilon to prevent log(0) if someone sets k=0 (though k should be >= 1)
    k_safe = torch.clamp(k, min=1e-4)
    lam_safe = torch.clamp(lam, min=1e-4)

    log_k = torch.log(k_safe)
    log_lam = torch.log(lam_safe)

    # Calculate log normalization factor (-ln M)
    # derived from: \phi(1) = 1
    # ln(Z) = - [k * ln(1) - λ * 1] = λ
    log_Z = lam_safe  # Since ln(1) = 0

    # 4.2 Calculate Log Coefficient
    # ln(coeff) = ln(Z) + (k-1)ln(w) - λw
    log_coeff = log_Z + (k - 1.0) * log_w - lam * w

    # 4.3 Detach and Exponentiate
    # CRITICAL: coeff is a scalar re-weighting term, not part of the gradient graph.
    coeff = torch.exp(log_coeff).detach()

    # Safety: Handle NaNs (e.g., if w is massive, -λw becomes -inf)
    coeff = torch.nan_to_num(coeff, nan=0.0, posinf=0.0, neginf=0.0)

    # 5. Construct Loss
    # We construct a proxy variable `phi_proxy` whose value is φ(w) but gradient is w's gradient.
    # This preserves the direction of PPO updates but scales the magnitude by φ(w).
    phi_proxy = coeff * w

    # Standard Policy Gradient Loss: - Σ (Weighted_Prob * Advantage)
    loss_mat = -phi_proxy * advantages

    pg_loss = agg_loss(loss_mat, response_mask, loss_agg_mode)

    # 6. Detailed Metrics (For diagnosing Lower/Upper/Dual Clip behavior)
    with torch.no_grad():
        # Ensure mask is boolean
        mask = response_mask > 0.5

        # 基础数据准备
        w_flat = w[mask]
        coeff_flat = coeff[mask]
        phi_real = coeff_flat * w_flat
        adv_flat = advantages[mask]

        metrics = {
            # 基础分布
            "actor/approx_kl": -log_w[mask].mean().item(),  # approx KL = E[log(p_old/p_new)] = E[-log_w]
            "gamma_is/w_mean": w_flat.mean().item(),
            "gamma_is/w_max": w_flat.max().item(),
            "gamma_is/w_std": w_flat.std().item(),  # 监控方差很有用

            # 核心权重监控
            "gamma_is/phi_mean": phi_real.mean().item(),
            "gamma_is/phi_max": phi_real.max().item(),  # CRITICAL: Should never exceed 1.0

            # Sanity Check: 如果 phi_max > 1.01，说明归一化数学推导有 Bug
            "gamma_is/normalization_error": max(0.0, phi_real.max().item() - 1.0),
        }

        # --- Breakdown by Advantage Sign ---

        # 1. Positive Advantage (Learning Good Actions) -> Focus on Upper Clip
        p_mask = adv_flat > 0
        if p_mask.any():
            w_pos = w_flat[p_mask]
            phi_pos = phi_real[p_mask]

            metrics["gamma_is/pos_w_mean"] = w_pos.mean().item()
            metrics["gamma_is/pos_phi_mean"] = phi_pos.mean().item()

            # Upper Clip Diagnosis:
            # How much are we suppressing outliers (w > 2.0)?
            outlier_mask = w_pos > 2.0
            if outlier_mask.any():
                # Ratio = Effective_Weight / Raw_Weight.
                # 1.0 = Trust completely. 0.1 = Distrust.
                # Corresponds to `coeff`
                metrics["gamma_is/pos_upper_clip_ratio"] = (phi_pos[outlier_mask] / w_pos[outlier_mask]).mean().item()
                metrics["gamma_is/pos_outlier_count"] = outlier_mask.sum().item()
            else:
                metrics["gamma_is/pos_upper_clip_ratio"] = 1.0
                metrics["gamma_is/pos_outlier_count"] = 0

        # 2. Negative Advantage (Avoiding Bad Actions) -> Focus on Lower & Dual Clip
        n_mask = adv_flat < 0
        if n_mask.any():
            w_neg = w_flat[n_mask]
            phi_neg = phi_real[n_mask]

            metrics["gamma_is/neg_w_mean"] = w_neg.mean().item()

            # Lower Clip Diagnosis:
            # For w < 0.5 (insignificant mistakes), phi should be tiny.
            low_mask = w_neg < 0.5
            if low_mask.any():
                metrics["gamma_is/neg_lower_clip_phi"] = phi_neg[low_mask].mean().item()
            else:
                metrics["gamma_is/neg_lower_clip_phi"] = 0.0

            # Dual Clip (Safety) Diagnosis:
            # For w > 5.0 (Step 16 Noise), phi MUST be effectively 0.
            # This is the "Crash Preventer" metric.
            noise_mask = w_neg > 5.0
            if noise_mask.any():
                metrics["gamma_is/neg_dual_clip_phi"] = phi_neg[noise_mask].mean().item()
                metrics["gamma_is/neg_noise_count"] = noise_mask.sum().item()
                # 记录一下最大的噪音 w 是多少，看看有多离谱
                metrics["gamma_is/neg_noise_max_w"] = w_neg[noise_mask].max().item()
            else:
                metrics["gamma_is/neg_dual_clip_phi"] = 0.0
                metrics["gamma_is/neg_noise_count"] = 0
                metrics["gamma_is/neg_noise_max_w"] = 0.0

    return pg_loss, metrics



@register_policy_loss("gamma_is_adaptive")
def compute_policy_loss_gamma_is_adaptive(
        old_log_prob: torch.Tensor,
        log_prob: torch.Tensor,
        advantages: torch.Tensor,
        response_mask: torch.Tensor,
        loss_agg_mode: str = "token-mean",
        config: Optional[Any] = None,
        rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Adaptive Gamma-IS Policy Loss (ESS-Driven).

    Dynamic Mechanism:
        Uses Effective Sample Size (ESS) to detect O.O.D. (Out-of-Distribution) batches.
        Let alpha = Normalized ESS (0 to 1).

        1. Sharpness Adaptation (Iris Mechanism):
           k_new = k_base / alpha
           -> As ESS drops, k -> infinity (distribution becomes a Dirac delta).

        2. Peak Adaptation (Return-to-Center):
           Peak_new = 1.0 + (Peak_base - 1.0) * alpha
           -> As ESS drops, Peak moves to 1.0.

    Result:
        - High ESS (I.D.): Acts like standard Gamma-IS (G49/G33).
        - Low ESS (O.O.D.): Acts like Identity Mapping (No Update), preventing collapse.
    """
    assert config is not None

    # 1. Config Extraction
    gamma_config = config.policy_loss.get("gamma_is", {}) if hasattr(config, "policy_loss") else {}

    # Base Parameters (Ideal scenario parameters, e.g., G49 or G33)
    k_pos_base = float(gamma_config.get("k_pos",2.0))
    # Usually Peak=0.75 -> lambda = k/P = 3.0/0.75 = 4.0
    lambda_pos_base = float(gamma_config.get("lambda_pos", 1.0))

    k_neg_base = float(gamma_config.get("k_neg", 6.0))
    # Usually Peak=2.0 -> lambda = k/P = 4.0/2.0 = 2.0
    lambda_neg_base = float(gamma_config.get("lambda_neg", 2.0))

    # Adaptive Settings
    enable_adaptive = gamma_config.get("enable_adaptive", True)  # Default to True now
    min_alpha = float(gamma_config.get("min_alpha", 1e-3))  # Prevent div by zero

    # 2. Compute IS Ratio w
    log_w = log_prob - old_log_prob
    log_w = torch.clamp(log_w, -10.0, 10.0)
    w = torch.exp(log_w)

    # 3. Calculate Adaptive Coefficients (ESS)
    # CRITICAL: Compute ESS only on generated tokens (using response_mask)
    # CRITICAL: Detach gradients! We don't want the policy to optimize for higher ESS.
    alpha = 1.0  # Default if adaptive is off
    ess_val = 0.0

    if enable_adaptive:
        with torch.no_grad():
            # Flatten and mask: only valid response tokens contribute to ESS
            # Ensure mask is boolean for indexing
            bool_mask = response_mask > 0.5

            if bool_mask.any():
                w_valid = w[bool_mask]

                # ESS Formula: (Sum w)^2 / Sum (w^2)
                sum_w = w_valid.sum()
                sum_w_sq = (w_valid ** 2).sum()

                # Add epsilon to prevent div by zero
                ess_val = (sum_w ** 2) / (sum_w_sq + 1e-6)

                # Normalized ESS: alpha in (0, 1]
                # If w are all 1.0, ESS = N, alpha = 1.0
                N = w_valid.numel()
                raw_alpha = ess_val / N

                # Clamp alpha to avoid numerical explosion
                alpha = torch.clamp(raw_alpha, min=min_alpha, max=1.0).item()
            else:
                # Fallback if batch is empty (shouldn't happen)
                alpha = 1.0

    # 4. Dynamic Parameter Calculation

    def get_adaptive_params(k_base, lam_base, alpha_val):
        """
        Applies Inverse-ESS scaling to k and Linear Interpolation to Peak.
        Returns new k and lambda tensor/scalar.
        """
        # Calculate Base Peak
        p_base = k_base / (lam_base + 1e-8)

        # 1. Adapt Peak: Move towards 1.0 as alpha drops
        # alpha=1 -> p_new = p_base
        # alpha=0 -> p_new = 1.0
        p_new = 1.0 + (p_base - 1.0) * alpha_val

        # 2. Adapt Sharpness: Increase as alpha drops
        # alpha=1 -> k_new = k_base
        # alpha=min -> k_new = huge
        k_new = k_base / alpha_val

        # 3. Recalculate Lambda to maintain the new Peak
        # Peak = k / lambda => lambda = k / Peak
        lam_new = k_new / p_new

        return k_new, lam_new

    # Compute adaptive parameters for both Pos and Neg cases
    # (These are scalars now, derived from the global batch alpha)
    k_pos_new, lambda_pos_new = get_adaptive_params(k_pos_base, lambda_pos_base, alpha)
    k_neg_new, lambda_neg_new = get_adaptive_params(k_neg_base, lambda_neg_base, alpha)

    # 5. Apply Parameters based on Advantage Sign
    pos_mask = (advantages >= 0).float()
    neg_mask = 1.0 - pos_mask

    # Now k and lam are tensors matching the shape of w
    k = pos_mask * k_pos_new + neg_mask * k_neg_new
    lam = pos_mask * lambda_pos_new + neg_mask * lambda_neg_new

    # 6. Compute Normalized Coefficient (Same as static version)
    # Normalization Constant Z so that max(φ(w)) = 1

    k_safe = torch.clamp(k, min=1e-4)
    lam_safe = torch.clamp(lam, min=1e-4)

    log_k = torch.log(k_safe)
    log_lam = torch.log(lam_safe)

    # Log Max: - [ k(ln k - ln λ) - k ]
    # log_Z = - (k * (log_k - log_lam) - k)
    log_Z = lam_safe

    # ln(coeff) = ln(Z) + (k-1)ln(w) - λw
    log_coeff = log_Z + (k - 1.0) * log_w - lam * w

    # Detach coefficient
    coeff = alpha * torch.exp(log_coeff).detach()
    coeff = torch.nan_to_num(coeff, nan=0.0, posinf=0.0, neginf=0.0)

    # 7. Construct Loss
    phi_proxy = coeff * w
    loss_mat = -phi_proxy * advantages
    pg_loss = agg_loss(loss_mat, response_mask, loss_agg_mode)

    # 8. Metrics (Updated with Adaptive Info)
    with torch.no_grad():
        mask = response_mask > 0.5
        w_flat = w[mask]
        coeff_flat = coeff[mask]
        phi_real = coeff_flat * w_flat

        metrics = {
            "actor/approx_kl": -log_w[mask].mean().item(),
            "gamma_is/w_mean": w_flat.mean().item(),
            "gamma_is/w_max": w_flat.max().item(),
            "gamma_is/w_std": w_flat.std().item(),

            "gamma_is/phi_mean": phi_real.mean().item(),
            "gamma_is/phi_max": phi_real.max().item(),

            # --- Adaptive Metrics ---
            "gamma_is/adaptive_ess": ess_val.item() if isinstance(ess_val, torch.Tensor) else ess_val,
            "gamma_is/adaptive_alpha": alpha,  # 0 (bad) to 1 (good)
            "gamma_is/adaptive_k_neg": k_neg_new,  # Monitor this spiking
            "gamma_is/adaptive_p_neg": k_neg_new / lambda_neg_new,  # Monitor this moving to 1.0
        }

        # Outlier diagnostics (Same as before)
        adv_flat = advantages[mask]

        # Neg Analysis
        n_mask = adv_flat < 0
        if n_mask.any():
            w_neg = w_flat[n_mask]
            phi_neg = phi_real[n_mask]

            noise_mask = w_neg > 5.0
            if noise_mask.any():
                metrics["gamma_is/neg_noise_count"] = noise_mask.sum().item()
                metrics["gamma_is/neg_noise_max_w"] = w_neg[noise_mask].max().item()
                # Key metric: Did the adaptive mechanism kill the gradient for noise?
                metrics["gamma_is/neg_noise_phi_mean"] = phi_neg[noise_mask].mean().item()
            else:
                metrics["gamma_is/neg_noise_count"] = 0
                metrics["gamma_is/neg_noise_max_w"] = 0.0
                metrics["gamma_is/neg_noise_phi_mean"] = 0.0


    return pg_loss, metrics


@register_policy_loss("gamma_is_amplitude")
def compute_policy_loss_gamma_is_amplitude(
        old_log_prob: torch.Tensor,
        log_prob: torch.Tensor,
        advantages: torch.Tensor,
        response_mask: torch.Tensor,
        loss_agg_mode: str = "token-mean",
        config: Optional[Any] = None,
        rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Adaptive Gamma-IS (G60): Static Shape + Dynamic Amplitude.

    Logic:
    1. Shape: Fixed Gamma distribution defined by (k, lambda).
       - Does NOT change with ESS.
    2. Amplitude: Scaled by Linear ESS (alpha).
       - Low ESS -> Low alpha -> Small gradients (Trust Region Control).
    """
    assert config is not None
    gamma_config = config.policy_loss.get("gamma_is", {}) if hasattr(config, "policy_loss") else {}

    # 1. Config: Static Shape Parameters
    k_pos = float(gamma_config.get("k_pos", 1.5))
    lambda_pos = float(gamma_config.get("lambda_pos", 0.91))

    k_neg = float(gamma_config.get("k_neg", 6.0))
    lambda_neg = float(gamma_config.get("lambda_neg", 2.0))

    # Adaptive Settings
    enable_adaptive = gamma_config.get("enable_adaptive", True)
    min_alpha = float(gamma_config.get("min_alpha", 1e-3))

    # 2. Compute IS Ratio w
    log_w = log_prob - old_log_prob
    log_w = torch.clamp(log_w, -10.0, 10.0)  # Safety clamp
    w = torch.exp(log_w)

    # 3. Compute Dynamic Amplitude (alpha) via Linear ESS
    # This is the "Dimming Switch" for O.O.D. data
    alpha = 1.0
    ess_val = 0.0

    if enable_adaptive:
        with torch.no_grad():
            bool_mask = response_mask > 0.5
            if bool_mask.any():
                w_valid = w[bool_mask]

                # ESS = (Sum w)^2 / Sum (w^2)
                # Sensitive to outliers (Variance-based)
                sum_w = w_valid.sum()
                sum_w_sq = (w_valid ** 2).sum()

                ess_val = (sum_w ** 2) / (sum_w_sq + 1e-8)
                N = w_valid.numel()

                # Calculate and clamp alpha
                alpha = ess_val / N
                alpha = torch.clamp(alpha, min=min_alpha, max=1.0).item()
            else:
                # Fallback for empty batch
                alpha = 0.0

                # 4. Vectorize Static Parameters
    pos_mask = (advantages >= 0).float()
    neg_mask = 1.0 - pos_mask

    k = pos_mask * k_pos + neg_mask * k_neg
    lam = pos_mask * lambda_pos + neg_mask * lambda_neg

    # 5. Compute Normalized Gamma Weight (Phi Shape)
    # This defines "What looks like a good sample" (Does not change)
    k_safe = torch.clamp(k, min=1e-4)
    lam_safe = torch.clamp(lam, min=1e-4)

    # Log-Space Normalization (Max=1.0 at w = k/lambda)
    log_Z = - (k * (torch.log(k_safe) - torch.log(lam_safe)) - k)
    log_phi_shape = log_Z + (k - 1.0) * log_w - lam * w

    # Detach to stop gradient flowing through the weight itself
    phi_shape = torch.exp(log_phi_shape).detach()
    phi_shape = torch.nan_to_num(phi_shape, nan=0.0)

    # 6. Apply Dynamic Amplitude
    # This defines "How much we trust this batch"
    coeff = phi_shape * alpha

    # 7. Compute Loss
    # - (weight * w * A) * grad(log_pi)
    loss_mat = -(coeff * w) * advantages
    pg_loss = agg_loss(loss_mat, response_mask, loss_agg_mode)

    # 8. Metrics
    with torch.no_grad():
        if bool_mask.any():
            metrics = {
                "gamma_is/adaptive_ess": ess_val.item() if isinstance(ess_val, torch.Tensor) else ess_val,
                "gamma_is/adaptive_alpha": alpha,
                "gamma_is/phi_mean": (coeff * w)[bool_mask].mean().item(),
                "gamma_is/w_max": w[bool_mask].max().item(),
            }
        else:
            metrics = {"gamma_is/adaptive_alpha": alpha}

    return pg_loss, metrics
