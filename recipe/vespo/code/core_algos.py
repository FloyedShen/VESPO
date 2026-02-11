# Copyright 2024 VESPO Authors
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
VESPO Core Algorithms

This module contains the VESPO policy loss implementation.
Import this module to register the VESPO policy loss with veRL.

Usage:
    # Import to register the policy loss
    import recipe.vespo.code.core_algos

    # Then use loss_mode="vespo" in config
"""

from typing import Any, Optional

import torch
from verl.trainer.ppo.core_algos import agg_loss, register_policy_loss


__all__ = [
    "compute_policy_loss_vespo"
]


@register_policy_loss("vespo")
def compute_policy_loss_vespo(
        old_log_prob: torch.Tensor,
        log_prob: torch.Tensor,
        advantages: torch.Tensor,
        response_mask: torch.Tensor,
        loss_agg_mode: str = "token-mean",
        config: Optional[Any] = None,
        rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Sequence-Level Gamma-IS Policy Loss (REINFORCE-style gradient).

    Gradient Form:
        ∇J = E[φ(w_seq) × A × ∇log π]

    where:
        - w_seq = ∏ₜ (π_θ/π_old)ₜ = exp(Σₜ log(π_θ/π_old)) is the TRUE sequence IS ratio (product)
        - φ(w) = w^k × e^{-λw} is the gamma weighting (normalized so φ(1)=1)

    Key Design:
        1. Uses TRUE sequence IS ratio (product, not geometric mean) - UNBIASED
        2. φ(w) is detached, only acts as gradient scaling coefficient
        3. Loss = -φ(w).detach() × A × log_prob, gradient naturally gives φ(w) × A × ∇log π
        4. Avoids numerical issues from exp() in the computation graph

    Args:
        old_log_prob: Log probabilities from behavior policy μ
        log_prob: Log probabilities from current policy π_θ
        advantages: Advantage estimates (sequence-level, broadcasted to tokens)
        response_mask: Valid token mask
        loss_agg_mode: Loss aggregation strategy
        config: Actor config with:
            - vespo.k_pos: k for positive advantages (default: 2.0)
            - vespo.lambda_pos: λ for positive advantages (default: 1.5)
            - vespo.k_neg: k for negative advantages (default: 4.0)
            - vespo.lambda_neg: λ for negative advantages (default: 2.5)
        rollout_is_weights: Optional TIS weights (π_old / π_sampler)

    Returns:
        pg_loss: Policy gradient loss
        metrics: Dictionary of metrics
    """
    assert config is not None

    # 1. Config Extraction
    gamma_config = config.policy_loss.get("vespo", {}) if hasattr(config, "policy_loss") else {}

    k_pos = float(gamma_config.get("k_pos", 2.0))
    lambda_pos = float(gamma_config.get("lambda_pos", 3.0))
    k_neg = float(gamma_config.get("k_neg", 3.0))
    lambda_neg = float(gamma_config.get("lambda_neg", 2.0))

    # 2. Compute TRUE Sequence-Level IS Ratio (product, not geometric mean)
    # Token-level log ratio: log(π_θ / π_old)
    log_ratio = log_prob - old_log_prob
    log_ratio = torch.clamp(log_ratio, -20.0, 20.0)

    # Sequence-level log ratio: Σₜ log(π_θ/π_old) = log(∏ₜ wₜ)
    # This is the TRUE sequence IS ratio in log space (NO division by seq_lengths)
    seq_log_ratio = torch.sum(log_ratio * response_mask, dim=-1)  # (batch_size,)

    # 3. Apply TIS Correction in LOG space (if provided)
    # w_corrected = (π_θ/π_old) × (π_old/π_sampler) = π_θ/π_sampler
    # In log space: log(w_corrected) = log(π_θ/π_old) + log(π_old/π_sampler)
    seq_log_tis = None
    if rollout_is_weights is not None:
        # Aggregate TIS to sequence level in log space (product = sum of logs)
        log_tis = torch.log(rollout_is_weights.clamp(min=1e-8))
        log_tis = torch.clamp(log_tis, -20.0, 20.0)  # Clamp per-token log TIS
        seq_log_tis = torch.sum(log_tis * response_mask, dim=-1)
        # Combine in log space: log(w_seq) = log(π_θ/π_old) + log(π_old/π_sampler)
        seq_log_ratio_combined = seq_log_ratio + seq_log_tis
    else:
        seq_log_ratio_combined = seq_log_ratio

    # 4. Compute w_seq for gamma weighting (detached, not in gradient graph)
    # Single clamp at the end after combining all log terms
    seq_log_ratio_clamped = torch.clamp(seq_log_ratio_combined, -20.0, 20.0)
    w_seq = torch.exp(seq_log_ratio_clamped.detach())  # (batch_size,)

    # 5. Get sequence-level advantage
    # In GRPO, advantages are typically the same for all tokens in a sequence
    seq_adv = advantages[:, 0] if advantages.dim() > 1 else advantages  # (batch_size,)

    # 6. Select k and lambda based on advantage sign (sequence level)
    pos_mask_seq = (seq_adv >= 0).float()
    neg_mask_seq = 1.0 - pos_mask_seq

    k_seq = pos_mask_seq * k_pos + neg_mask_seq * k_neg
    lam_seq = pos_mask_seq * lambda_pos + neg_mask_seq * lambda_neg

    # 7. Compute Gamma Weight φ(w_seq)
    # φ(w) = Z × w^k × e^{-λw}, normalized so φ(1) = 1
    # At w=1: φ(1) = Z × e^{-λ} = 1, so Z = e^λ
    #
    # In log space:
    # log φ(w) = λ + k×log(w) - λ×w

    lam_safe = torch.clamp(lam_seq, min=1e-4)
    log_w_seq = torch.log(w_seq.clamp(min=1e-8))

    log_phi = lam_safe + k_seq * log_w_seq - lam_safe * w_seq

    # φ(w) - detached, only acts as coefficient
    phi_seq = torch.exp(log_phi).detach()  # (batch_size,)
    phi_seq = torch.nan_to_num(phi_seq, nan=0.0, posinf=0.0, neginf=0.0)

    # 8. Broadcast φ to token level
    phi_token = phi_seq.unsqueeze(-1)  # (batch_size, 1)

    # 9. Compute Loss: L = -φ(w).detach() × A × log_prob
    # Gradient: ∇L = -φ(w) × A × ∇log_prob = -φ(w) × A × ∇log π
    # This is exactly what we want!
    loss_mat = -phi_token * advantages * log_prob

    pg_loss = agg_loss(loss_mat, response_mask, loss_agg_mode)
    # pg_loss = agg_loss(loss_mat, response_mask, "seq-mean-token-mean")

    # 10. Metrics
    with torch.no_grad():
        mask = response_mask > 0.5

        # Sequence lengths for normalized metrics
        seq_lengths = torch.sum(response_mask, dim=-1).clamp(min=1)

        # Normalized seq_log_ratio (geometric mean) for interpretable KL
        seq_log_ratio_normalized = seq_log_ratio / seq_lengths

        metrics = {
            # KL divergence (using normalized log ratio)
            "actor/approx_kl": (-seq_log_ratio_normalized).mean().item(),

            # Sequence-level w statistics (true product)
            "vespo/w_seq_mean": w_seq.mean().item(),
            "vespo/w_seq_max": w_seq.max().item(),
            "vespo/w_seq_min": w_seq.min().item(),
            "vespo/w_seq_std": w_seq.std().item(),

            # Log-space statistics (more interpretable for products)
            "vespo/log_w_seq_mean": seq_log_ratio_clamped.mean().item(),
            "vespo/log_w_seq_max": seq_log_ratio_clamped.max().item(),
            "vespo/log_w_seq_min": seq_log_ratio_clamped.min().item(),

            # φ(w) statistics
            "vespo/phi_mean": phi_seq.mean().item(),
            "vespo/phi_max": phi_seq.max().item(),
            "vespo/phi_min": phi_seq.min().item(),

            # Config
            "vespo/k_pos": k_pos,
            "vespo/lambda_pos": lambda_pos,
            "vespo/k_neg": k_neg,
            "vespo/lambda_neg": lambda_neg,
        }

        # Per-sign breakdown (always add all keys with defaults)
        pos_seq_mask = seq_adv > 0
        neg_seq_mask = seq_adv < 0

        # Positive advantage stats
        if pos_seq_mask.any():
            metrics["vespo/w_seq_pos_mean"] = w_seq[pos_seq_mask].mean().item()
            metrics["vespo/phi_pos_mean"] = phi_seq[pos_seq_mask].mean().item()
            metrics["vespo/n_pos_seq"] = pos_seq_mask.sum().item()
        else:
            metrics["vespo/w_seq_pos_mean"] = 0.0
            metrics["vespo/phi_pos_mean"] = 0.0
            metrics["vespo/n_pos_seq"] = 0

        # Negative advantage stats
        if neg_seq_mask.any():
            metrics["vespo/w_seq_neg_mean"] = w_seq[neg_seq_mask].mean().item()
            metrics["vespo/phi_neg_mean"] = phi_seq[neg_seq_mask].mean().item()
            metrics["vespo/n_neg_seq"] = neg_seq_mask.sum().item()

            # Safety check: outlier handling for negative advantages
            neg_w = w_seq[neg_seq_mask]
            noise_mask = neg_w > 100.0  # Higher threshold since we use product
            if noise_mask.any():
                metrics["vespo/neg_noise_count"] = noise_mask.sum().item()
                metrics["vespo/neg_noise_max_w"] = neg_w[noise_mask].max().item()
                metrics["vespo/neg_noise_phi_mean"] = phi_seq[neg_seq_mask][noise_mask].mean().item()
            else:
                metrics["vespo/neg_noise_count"] = 0
                metrics["vespo/neg_noise_max_w"] = 0.0
                metrics["vespo/neg_noise_phi_mean"] = 0.0
        else:
            metrics["vespo/w_seq_neg_mean"] = 0.0
            metrics["vespo/phi_neg_mean"] = 0.0
            metrics["vespo/n_neg_seq"] = 0
            metrics["vespo/neg_noise_count"] = 0
            metrics["vespo/neg_noise_max_w"] = 0.0
            metrics["vespo/neg_noise_phi_mean"] = 0.0

        # TIS statistics (always add all keys)
        if rollout_is_weights is not None:
            # Compute seq_tis for metrics only (already combined into w_seq)
            seq_tis_for_metrics = torch.exp(torch.clamp(seq_log_tis, -20.0, 20.0))
            metrics["vespo/tis_enabled"] = 1.0
            metrics["vespo/seq_log_tis_mean"] = seq_log_tis.mean().item()
            metrics["vespo/seq_log_tis_max"] = seq_log_tis.max().item()
            metrics["vespo/seq_log_tis_min"] = seq_log_tis.min().item()
            metrics["vespo/seq_tis_mean"] = seq_tis_for_metrics.mean().item()
            metrics["vespo/seq_tis_max"] = seq_tis_for_metrics.max().item()
        else:
            metrics["vespo/tis_enabled"] = 0.0
            metrics["vespo/seq_log_tis_mean"] = 0.0
            metrics["vespo/seq_log_tis_max"] = 0.0
            metrics["vespo/seq_log_tis_min"] = 0.0
            metrics["vespo/seq_tis_mean"] = 0.0
            metrics["vespo/seq_tis_max"] = 0.0

    return pg_loss, metrics
