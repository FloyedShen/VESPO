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
IS Reshape Policy Loss Implementation

This implements the unified SFT-RL framework where:
- γ=0: Equivalent to supervised learning (no IS correction)
- γ=1: Standard policy gradient / PPO
- γ ∈ (0,1): Variance-controlled interpolation

The gradient is: g_γ = E_μ[w^γ · A · ∇log π]

Optimal γ is computed via closed-form: γ* = min(1, √(-log ρ_min / σ²))
where σ² = Var(log w) and ρ_min is the minimum acceptable ESS ratio.

Theory:
- Target distribution evolves as: p_γ^* ∝ (μ·r)^{1-γ} · e^{γr/τ}
- γ=0 (SFT): Forward KL, mean-seeking, r as linear weight
- γ=1 (RL): Reverse KL, mode-seeking, r as exponential objective
- Variance controlled by 2γ-order Rényi divergence
"""

import math
from typing import Any, Optional

import torch

import verl.utils.torch_functional as verl_F
from verl.trainer.config import AlgoConfig
from verl.trainer.ppo.core_algos import agg_loss, register_policy_loss
from verl.workers.config import ActorConfig


def compute_adaptive_gamma(
    log_w: torch.Tensor,
    response_mask: torch.Tensor,
    rho_min: float = 0.3,
    gamma_min: float = 0.0,
    gamma_max: float = 1.0,
) -> tuple[float, float]:
    """
    Compute optimal γ using closed-form solution.

    γ* = min(γ_max, √(-log ρ_min / σ²))

    This is an O(1) computation (given pre-computed variance).

    Args:
        log_w: Log importance weights, shape (batch_size, response_length)
        response_mask: Valid token mask, shape (batch_size, response_length)
        rho_min: Minimum acceptable ESS ratio (default: 0.3)
        gamma_min: Minimum γ value (default: 0.0)
        gamma_max: Maximum γ value (default: 1.0)

    Returns:
        gamma: Optimal γ value
        sigma_sq: Variance of log weights
    """
    # Compute σ² = Var(log w) over valid tokens
    valid_log_w = log_w[response_mask.bool()]

    if valid_log_w.numel() > 1:
        sigma_sq = valid_log_w.var().item()
    else:
        sigma_sq = 0.0

    # Closed-form optimal gamma: γ* = min(1, √(-log ρ_min / σ²))
    if sigma_sq < 1e-8:
        gamma = gamma_max
    else:
        gamma = math.sqrt(-math.log(rho_min) / sigma_sq)
        gamma = min(gamma_max, max(gamma_min, gamma))

    return gamma, sigma_sq


def compute_ess_ratio(
    weights: torch.Tensor,
    response_mask: torch.Tensor,
) -> float:
    """
    Compute Effective Sample Size (ESS) ratio.

    ESS = (Σ w_i)² / Σ w_i²
    ESS_ratio = ESS / n

    Under Log-Normal assumption: ESS_γ/n ≈ exp(-σ²γ²)

    Args:
        weights: Importance weights w^γ, shape (batch_size, response_length)
        response_mask: Valid token mask

    Returns:
        ESS ratio (between 0 and 1)
    """
    valid_weights = weights[response_mask.bool()]
    n_tokens = valid_weights.numel()

    if n_tokens == 0:
        return 1.0

    sum_w = valid_weights.sum()
    sum_w_sq = valid_weights.pow(2).sum()

    ess = (sum_w ** 2) / (sum_w_sq + 1e-8)
    ess_ratio = (ess / n_tokens).item()

    return ess_ratio


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
    """
    Compute the IS Reshape policy gradient loss.

    This implements the unified SFT-RL framework:

    Loss: L_γ = -E_μ[w^γ · A]  (negative because we minimize)
    Gradient: g_γ = E_μ[w^γ · A · ∇log π]

    The reshaped importance weight w^γ provides variance control:
    - γ=0: No IS correction (SFT-like)
    - γ=1: Full IS correction (standard PPO)
    - γ ∈ (0,1): Interpolation with controlled variance

    Args:
        old_log_prob: Log-probabilities under behavior policy μ (rollout policy)
            Shape: (batch_size, response_length)
        log_prob: Log-probabilities under current policy π_θ
            Shape: (batch_size, response_length)
        advantages: Advantage estimates A(s,a)
            Shape: (batch_size, response_length)
        response_mask: Mask for valid tokens
            Shape: (batch_size, response_length)
        loss_agg_mode: Loss aggregation mode ("token-mean", "seq-mean-token-sum", etc.)
        config: Actor configuration containing IS Reshape parameters:
            - is_reshape.rho_min: Minimum ESS ratio (default: 0.3)
            - is_reshape.gamma_min: Minimum γ (default: 0.0)
            - is_reshape.gamma_max: Maximum γ (default: 1.0)
            - is_reshape.adaptive_gamma: Whether to use adaptive γ (default: True)
            - is_reshape.fixed_gamma: Fixed γ value if not adaptive (default: 0.5)
            - is_reshape.use_ppo_clip: Whether to also apply PPO clipping (default: True)
            - clip_ratio, clip_ratio_low, clip_ratio_high: PPO clip parameters
        rollout_is_weights: Pre-computed rollout IS weights (optional, for compatibility)

    Returns:
        pg_loss: Scalar policy gradient loss
        pg_metrics: Dictionary of metrics for logging
    """
    assert config is not None, "config is required for IS Reshape"
    assert not isinstance(config, AlgoConfig), "ActorConfig expected, not AlgoConfig"

    # ========== Extract IS Reshape Configuration ==========
    is_reshape_config = config.get("is_reshape", {})

    # Core IS Reshape parameters
    rho_min = is_reshape_config.get("rho_min", 0.3)
    gamma_min = is_reshape_config.get("gamma_min", 0.0)
    gamma_max = is_reshape_config.get("gamma_max", 1.0)
    adaptive_gamma = is_reshape_config.get("adaptive_gamma", True)
    fixed_gamma = is_reshape_config.get("fixed_gamma", 0.5)
    use_ppo_clip = is_reshape_config.get("use_ppo_clip", True)

    # Standard PPO clipping config
    clip_ratio = config.clip_ratio
    clip_ratio_low = config.clip_ratio_low if config.clip_ratio_low is not None else clip_ratio
    clip_ratio_high = config.clip_ratio_high if config.clip_ratio_high is not None else clip_ratio
    clip_ratio_c = config.get("clip_ratio_c", 3.0)

    # ========== Compute Log Importance Weight ==========
    # log w = log(π_θ / μ) = log π_θ - log μ
    log_w = log_prob - old_log_prob
    log_w_clamped = torch.clamp(log_w, min=-20.0, max=20.0)  # Numerical stability

    # ========== Compute Adaptive γ ==========
    with torch.no_grad():
        if adaptive_gamma:
            gamma, sigma_sq = compute_adaptive_gamma(
                log_w_clamped.detach(),
                response_mask,
                rho_min=rho_min,
                gamma_min=gamma_min,
                gamma_max=gamma_max,
            )
        else:
            gamma = fixed_gamma
            # Still compute sigma_sq for metrics
            valid_log_w = log_w_clamped[response_mask.bool()]
            sigma_sq = valid_log_w.var().item() if valid_log_w.numel() > 1 else 0.0

    # ========== Compute Reshaped Importance Weight ==========
    # w^γ = exp(γ · log w)
    reshaped_log_ratio = gamma * log_w_clamped
    reshaped_ratio = torch.exp(reshaped_log_ratio)

    # Standard ratio for metrics and optional PPO clipping
    ratio = torch.exp(log_w_clamped)

    # ========== Compute KL Divergence ==========
    ppo_kl = verl_F.masked_mean(-log_w_clamped, response_mask)

    # ========== Compute Policy Gradient Loss ==========
    # IS Reshape loss: L = -w^γ · A
    pg_losses1 = -advantages * reshaped_ratio

    if use_ppo_clip:
        # Apply PPO-style clipping on reshaped ratio
        pg_losses2 = -advantages * torch.clamp(
            reshaped_ratio, 1 - clip_ratio_low, 1 + clip_ratio_high
        )
        clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)
        pg_clipfrac = verl_F.masked_mean(
            torch.gt(pg_losses2, pg_losses1).float(), response_mask
        )

        # Dual-clip for negative advantages (from dual-clip PPO)
        pg_losses3 = -advantages * clip_ratio_c
        clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
        pg_clipfrac_lower = verl_F.masked_mean(
            torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(),
            response_mask
        )

        pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
    else:
        # Pure IS Reshape without PPO clipping
        pg_losses = pg_losses1
        pg_clipfrac = torch.tensor(0.0, device=pg_losses.device)
        pg_clipfrac_lower = torch.tensor(0.0, device=pg_losses.device)

    # ========== Apply Additional Rollout Correction (if provided) ==========
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    # ========== Aggregate Loss ==========
    pg_loss = agg_loss(
        loss_mat=pg_losses,
        loss_mask=response_mask,
        loss_agg_mode=loss_agg_mode,
        **config.global_batch_info
    )

    # ========== Compute Metrics ==========
    with torch.no_grad():
        # ESS ratio
        ess_ratio = compute_ess_ratio(reshaped_ratio.detach(), response_mask)

        # Ratio statistics
        valid_ratio = ratio[response_mask.bool()]
        ratio_mean = valid_ratio.mean().item() if valid_ratio.numel() > 0 else 1.0
        ratio_max = valid_ratio.max().item() if valid_ratio.numel() > 0 else 1.0
        ratio_min = valid_ratio.min().item() if valid_ratio.numel() > 0 else 1.0

        # Reshaped ratio statistics
        valid_reshaped = reshaped_ratio[response_mask.bool()]
        reshaped_mean = valid_reshaped.mean().item() if valid_reshaped.numel() > 0 else 1.0
        reshaped_max = valid_reshaped.max().item() if valid_reshaped.numel() > 0 else 1.0

    pg_metrics = {
        # Standard PPO metrics
        "actor/pg_clipfrac": pg_clipfrac.detach().item() if torch.is_tensor(pg_clipfrac) else pg_clipfrac,
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item() if torch.is_tensor(pg_clipfrac_lower) else pg_clipfrac_lower,

        # IS Reshape specific metrics
        "actor/is_reshape_gamma": gamma,
        "actor/is_reshape_sigma_sq": sigma_sq,
        "actor/is_reshape_ess_ratio": ess_ratio,

        # Ratio diagnostics
        "actor/ratio_mean": ratio_mean,
        "actor/ratio_max": ratio_max,
        "actor/ratio_min": ratio_min,
        "actor/reshaped_ratio_mean": reshaped_mean,
        "actor/reshaped_ratio_max": reshaped_max,
    }

    return pg_loss, pg_metrics


@register_policy_loss("is_reshape_pure")
def compute_policy_loss_is_reshape_pure(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Pure IS Reshape without PPO clipping.

    This is a convenience wrapper that forces use_ppo_clip=False.
    Use this when you want to rely solely on γ-based variance control.

    Args:
        Same as compute_policy_loss_is_reshape

    Returns:
        Same as compute_policy_loss_is_reshape
    """
    assert config is not None

    # Override use_ppo_clip to False
    if "is_reshape" not in config:
        config["is_reshape"] = {}
    config["is_reshape"]["use_ppo_clip"] = False

    return compute_policy_loss_is_reshape(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        loss_agg_mode=loss_agg_mode,
        config=config,
        rollout_is_weights=rollout_is_weights,
    )


@register_policy_loss("is_reshape_sft")
def compute_policy_loss_is_reshape_sft(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    IS Reshape with γ=0 (SFT-like behavior).

    This is equivalent to reward-weighted supervised fine-tuning:
    L = -E_μ[A · log π]

    No importance sampling correction is applied.

    Args:
        Same as compute_policy_loss_is_reshape

    Returns:
        Same as compute_policy_loss_is_reshape
    """
    assert config is not None

    # Force γ=0 (SFT mode)
    if "is_reshape" not in config:
        config["is_reshape"] = {}
    config["is_reshape"]["adaptive_gamma"] = False
    config["is_reshape"]["fixed_gamma"] = 0.0
    config["is_reshape"]["use_ppo_clip"] = False

    return compute_policy_loss_is_reshape(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        loss_agg_mode=loss_agg_mode,
        config=config,
        rollout_is_weights=rollout_is_weights,
    )
