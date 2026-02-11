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
VESPO Data Parallel Actor

This extends the base DataParallelPPOActor to track per-policy-update metrics,
which is crucial for understanding how VESPO's gamma adapts across
multiple PPO updates on the same data.

Metric granularity:
- Each policy update (optimizer.step on a mini_batch) is recorded as a separate step
- This provides finer-grained view than per-epoch logging
"""

import torch

from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty

# Import to register VESPO policy loss
import recipe.vespo.code.core_algos  # noqa: F401
from verl.utils.metric import reduce_metrics
from verl.utils.py_functional import append_to_dict
from verl.workers.actor import DataParallelPPOActor
from verl.workers.actor.dp_actor import prepare_dynamic_batch, get_device_id
from verl.utils.seqlen_balancing import get_reverse_idx

__all__ = ["ISReshapeDataParallelPPOActor"]


class ISReshapeDataParallelPPOActor(DataParallelPPOActor):
    """
    Extended DataParallelPPOActor that tracks per-policy-update metrics for VESPO.

    Key additions:
    - Each policy update (mini_batch optimizer step) is a separate metric step
    - Records metrics per policy update for VESPO (gamma, sigma_sq, ess_ratio)
    - Allows analysis of how gamma evolves as π_θ drifts from π_old
    - Supports NoMu mode: replaces μ with group mean of current π
    """

    def _compute_nomu_group_mean(
        self,
        micro_batches: list,
        micro_bsz_idx: list[list[int]],
        mini_batch_size: int,
        temperature: float,
    ) -> torch.Tensor:
        """
        Compute group-mean log probabilities for NoMu mode.

        NoMu (No-Mu) replaces the behavior policy μ with the group mean of current policy π.
        This transforms the importance ratio from W = π/μ to W = π/π_group_mean.

        Steps:
            1. Forward pass all micro-batches to collect log_prob (no grad)
            2. Reorder to original sequence using micro_bsz_idx
            3. Compute group mean for each group of n_samples
            4. Return group mean expanded to token level

        Args:
            micro_batches: List of micro-batch DataProto objects
            micro_bsz_idx: List of index lists, micro_bsz_idx[i] contains original indices for micro_batch[i]
            mini_batch_size: Total number of samples in mini-batch
            temperature: Temperature for log prob computation

        Returns:
            group_mean_log_probs: Tensor of shape (mini_batch_size, response_length)
                Each sample's value is the group mean log prob (expanded to token level)
        """
        n_samples = self.config.rollout_n
        device = get_device_id()

        # Phase 1: Collect log_prob from all micro-batches (no grad)
        all_log_probs = []
        all_response_masks = []
        all_indices = []

        with torch.no_grad():
            for micro_batch, indices in zip(micro_batches, micro_bsz_idx):
                micro_batch = micro_batch.to(device)
                model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}

                outputs = self._forward_micro_batch(
                    model_inputs, temperature=temperature, calculate_entropy=False
                )
                all_log_probs.append(outputs["log_probs"])
                all_response_masks.append(model_inputs["response_mask"])
                all_indices.extend(indices)

        # Concatenate all micro-batch results
        all_log_probs = torch.cat(all_log_probs, dim=0)  # (total_samples, response_length)
        all_response_masks = torch.cat(all_response_masks, dim=0)  # (total_samples, response_length)

        # Phase 2: Reorder to original sequence
        reverse_idx = get_reverse_idx(all_indices)
        reverse_idx_tensor = torch.tensor(reverse_idx, device=device, dtype=torch.long)
        log_probs_sorted = all_log_probs[reverse_idx_tensor]  # Back to original order
        response_masks_sorted = all_response_masks[reverse_idx_tensor]

        # Phase 3: Compute sequence-level log prob and group mean
        seq_lengths = response_masks_sorted.sum(dim=-1).clamp(min=1)  # (mini_batch_size,)
        seq_log_probs = (log_probs_sorted * response_masks_sorted).sum(dim=-1) / seq_lengths  # (mini_batch_size,)

        # Compute group mean
        n_groups = mini_batch_size // n_samples
        seq_log_probs_grouped = seq_log_probs.view(n_groups, n_samples)  # (n_groups, n_samples)
        group_mean_seq = seq_log_probs_grouped.mean(dim=1, keepdim=True)  # (n_groups, 1)
        group_mean_seq_expanded = group_mean_seq.expand(n_groups, n_samples).reshape(mini_batch_size)  # (mini_batch_size,)

        # Phase 4: Expand group mean back to token level
        # For each sample, all tokens share the same group mean (normalized by seq_length)
        # So that: sum(group_mean_log_probs * mask) / seq_len = group_mean_seq
        response_length = log_probs_sorted.shape[1]
        group_mean_per_token = group_mean_seq_expanded / seq_lengths  # (mini_batch_size,)
        group_mean_log_probs = group_mean_per_token.unsqueeze(-1).expand(mini_batch_size, response_length)
        group_mean_log_probs = group_mean_log_probs * response_masks_sorted  # Zero out padding

        # Reorder back to micro-batch order for later use
        forward_idx_tensor = torch.tensor(all_indices, device=device, dtype=torch.long)
        group_mean_log_probs = group_mean_log_probs[forward_idx_tensor]

        return group_mean_log_probs

    def update_policy(self, data: DataProto):
        """
        Update policy with per-policy-update metric tracking.

        Returns metrics with a "update_metrics" key containing a list of
        per-policy-update metric dictionaries, allowing the trainer to log each
        optimizer step as a separate global step.
        """
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]

        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
        ]
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        if "rollout_is_weights" in data.batch.keys():
            select_keys.append("rollout_is_weights")
        if "rollout_log_probs" in data.batch.keys():
            select_keys.append("rollout_log_probs")

        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        mini_batches = data.split(self.config.ppo_mini_batch_size)
        on_policy = len(mini_batches) == 1 and self.config.ppo_epochs == 1

        metrics = {}  # Aggregated metrics (for backward compatibility)
        all_update_metrics = []  # List of per-policy-update metrics
        num_epochs = self.config.ppo_epochs
        num_mini_batches = len(mini_batches)

        # Track update index across all epochs and mini_batches
        update_idx = 0

        for epoch_idx in range(num_epochs):
            for batch_idx, mini_batch in enumerate(mini_batches):
                # Metrics for this single policy update (optimizer step)
                update_metrics_raw = {}

                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, micro_bsz_idx = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)
                    # Build index list for non-dynamic case (sequential split)
                    micro_bs = self.config.ppo_micro_batch_size_per_gpu
                    micro_bsz_idx = [list(range(i * micro_bs, min((i + 1) * micro_bs, len(mini_batch))))
                                     for i in range(len(micro_batches))]

                # === NoMu: Replace old_log_prob with group mean of current policy ===
                # This transforms W = π/μ to W = π/π_group_mean, making it independent of behavior policy
                # NoMu computes group mean at the start of each mini-batch using current π
                # Handle both OmegaConf (.get) and dataclass (getattr) config access
                policy_loss_cfg = self.config.policy_loss
                if hasattr(policy_loss_cfg, 'get'):
                    use_nomu = policy_loss_cfg.get("use_nomu", False)
                else:
                    use_nomu = getattr(policy_loss_cfg, "use_nomu", False)
                nomu_group_mean_log_probs = None

                if use_nomu:
                    nomu_group_mean_log_probs = self._compute_nomu_group_mean(
                        micro_batches=micro_batches,
                        micro_bsz_idx=micro_bsz_idx,
                        mini_batch_size=len(mini_batch),
                        temperature=temperature,
                    )
                    metrics["nomu/enabled"] = [1.0]
                else:
                    metrics["nomu/enabled"] = [0.0]

                self.actor_optimizer.zero_grad()

                # Track current position in nomu_group_mean_log_probs
                nomu_offset = 0

                for micro_batch in micro_batches:
                    micro_batch = micro_batch.to(get_device_id())
                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    response_mask = model_inputs["response_mask"]
                    old_log_prob = model_inputs["old_log_probs"]
                    advantages = model_inputs["advantages"]
                    micro_bs = response_mask.shape[0]

                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    calculate_entropy = self.config.calculate_entropy or (entropy_coeff != 0)

                    if self.config.use_dynamic_bsz:
                        loss_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
                    else:
                        loss_scale_factor = 1 / self.gradient_accumulation

                    outputs = self._forward_micro_batch(
                        model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                    )
                    log_prob = outputs["log_probs"]
                    entropy = outputs["entropys"] if calculate_entropy else None

                    # Determine old_log_prob based on mode
                    if nomu_group_mean_log_probs is not None:
                        # NoMu mode: use pre-computed group mean as old_log_prob
                        # This makes log_ratio = log_π - group_mean(log_π)
                        old_log_prob = nomu_group_mean_log_probs[nomu_offset:nomu_offset + micro_bs]
                        nomu_offset += micro_bs

                    elif hasattr(self.config, "use_rollout_log_probs") and self.config.use_rollout_log_probs:
                        old_log_prob = model_inputs["old_log_probs"]
                    else:
                        if on_policy:
                            old_log_prob = log_prob.detach()
                        else:
                            old_log_prob = model_inputs["old_log_probs"]

                    loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
                    rollout_is_weights = model_inputs.get("rollout_is_weights", None)

                    policy_loss_fn = get_policy_loss_fn(loss_mode)

                    pg_loss, pg_metrics = policy_loss_fn(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=advantages,
                        response_mask=response_mask,
                        loss_agg_mode=loss_agg_mode,
                        config=self.config,
                        rollout_is_weights=rollout_is_weights,
                    )
                    micro_batch_metrics.update(pg_metrics)

                    rollout_log_prob = model_inputs.get("rollout_log_probs", None)
                    if loss_mode != "rollout_correction" and rollout_log_prob is not None:
                        from verl.trainer.ppo.rollout_corr_helper import compute_rollout_corr_metrics_from_logprobs
                        rollout_corr_metrics = compute_rollout_corr_metrics_from_logprobs(
                            log_prob=log_prob,
                            rollout_log_prob=rollout_log_prob,
                            response_mask=response_mask,
                        )
                        micro_batch_metrics.update(rollout_corr_metrics)

                    policy_loss = pg_loss
                    if calculate_entropy and entropy is not None:
                        entropy_agg = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                        micro_batch_metrics["actor/entropy"] = entropy_agg.detach().item()
                        if entropy_coeff != 0:
                            policy_loss -= entropy_agg * entropy_coeff

                    if self.config.use_kl_loss:
                        ref_log_prob = model_inputs["ref_log_prob"]
                        kld = kl_penalty(
                            logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type
                        )
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        micro_batch_metrics["actor/kl_loss"] = kl_loss.detach().item() * loss_scale_factor
                        micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    if self.config.use_dynamic_bsz:
                        loss = policy_loss * loss_scale_factor
                    else:
                        loss = policy_loss * loss_scale_factor

                    if self.scaler is not None:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    micro_batch_metrics["actor/pg_loss"] = pg_loss.detach().item() * loss_scale_factor

                    append_to_dict(metrics, micro_batch_metrics)
                    append_to_dict(update_metrics_raw, micro_batch_metrics)

                grad_norm = self._optimizer_step()
                mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, mini_batch_metrics)
                append_to_dict(update_metrics_raw, mini_batch_metrics)

                # Reduce update metrics to single values and store after each optimizer step
                update_metrics_reduced = reduce_metrics(update_metrics_raw)
                update_metrics_reduced["actor/ppo_epoch_idx"] = epoch_idx
                update_metrics_reduced["actor/mini_batch_idx"] = batch_idx
                update_metrics_reduced["actor/update_idx"] = update_idx
                all_update_metrics.append(update_metrics_reduced)
                update_idx += 1

        # Add total ppo_epochs and mini_batches for reference
        metrics["actor/ppo_epochs"] = [num_epochs]
        metrics["actor/num_mini_batches"] = [num_mini_batches]

        self.actor_optimizer.zero_grad()

        # Return both aggregated metrics and per-update metrics
        return {"metrics": metrics, "update_metrics": all_update_metrics}
