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
IS Reshape Megatron Workers

Extends veRL Megatron workers for IS Reshape experiments with per-policy-update metrics tracking.

Core modifications:
1. Monkey-patches update_policy to track metrics per optimizer.step() instead of per epoch
2. Custom policy_loss support: Handles is_reshape, gamma_is, etc. configs
3. Passes per-update metrics through meta_info["update_metrics"]
"""

import logging
import os
import types
from typing import Iterable

import psutil
from codetiming import Timer
from megatron.core import parallel_state as mpu
from omegaconf import OmegaConf, open_dict

import torch
from verl import DataProto
from verl.single_controller.base.decorator import Dispatch, make_nd_compute_dataproto_dispatch_fn, register
from verl.utils.device import get_torch_device
from verl.utils.megatron.router_replay_patch import RouterReplay, RouterReplayAction
from verl.utils.megatron_utils import (
    load_megatron_model_to_gpu,
    load_megatron_optimizer,
    offload_megatron_model_to_cpu,
    offload_megatron_optimizer,
)
from verl.utils.memory_utils import aggressive_empty_cache
from verl.utils.metric import reduce_metrics
from verl.utils.profiler import GPUMemoryLogger, log_gpu_memory_usage
from verl.utils.py_functional import append_to_dict
from verl.workers.config.actor import PolicyLossConfig
from verl.workers.megatron_workers import AsyncActorRolloutRefWorker

# Import to register custom policy loss functions (is_reshape, gamma_is, etc.)
# This import has side effects - it registers policy loss functions via decorators
import recipe.vespo.code.core_algos  # noqa: F401

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_PPO_LOGGING_LEVEL", "WARN"))

__all__ = ["ISReshapeMegatronActorRolloutRefWorker"]

# Known fields in PolicyLossConfig dataclass - any other fields are custom loss configs
_POLICY_LOSS_KNOWN_FIELDS = frozenset(
    f.name for f in __import__("dataclasses").fields(PolicyLossConfig)
)


def _compute_nomu_group_mean(self, data: DataProto) -> torch.Tensor:
    """
    Compute group-mean log probabilities for NoMu mode.

    NoMu (No-Mu) replaces the behavior policy μ with the group mean of current policy π.
    This transforms the importance ratio from W = π/μ to W = π/π_group_mean.

    Args:
        data: DataProto containing the mini-batch data

    Returns:
        group_mean_log_probs: Tensor of shape (batch_size, response_length)
            Each sample's value is the group mean log prob (expanded to token level)
    """
    n_samples = self.config.rollout_n

    # Phase 1: Collect log_prob using compute_log_prob (no grad, forward only)
    with torch.no_grad():
        # compute_log_prob returns tuple: (log_probs, entropys, layers_topk_idx)
        all_log_probs, _, _ = self.compute_log_prob(data, calculate_entropy=False)

    # all_log_probs shape: (batch_size, response_length), note: compute_log_prob returns CPU tensor
    response_mask = data.batch["response_mask"]

    # Ensure tensors are on the same device
    device = response_mask.device
    all_log_probs = all_log_probs.to(device)
    batch_size = all_log_probs.shape[0]

    # Phase 2: Compute sequence-level log prob and group mean
    seq_lengths = response_mask.sum(dim=-1).clamp(min=1)  # (batch_size,)
    seq_log_probs = (all_log_probs * response_mask).sum(dim=-1) / seq_lengths  # (batch_size,)

    # Compute group mean
    n_groups = batch_size // n_samples
    seq_log_probs_grouped = seq_log_probs.view(n_groups, n_samples)  # (n_groups, n_samples)
    group_mean_seq = seq_log_probs_grouped.mean(dim=1, keepdim=True)  # (n_groups, 1)
    group_mean_seq_expanded = group_mean_seq.expand(n_groups, n_samples).reshape(batch_size)  # (batch_size,)

    # Phase 3: Expand group mean back to token level
    # For each sample, all tokens share the same group mean (normalized by seq_length)
    # So that: sum(group_mean_log_probs * mask) / seq_len = group_mean_seq
    response_length = all_log_probs.shape[1]
    group_mean_per_token = group_mean_seq_expanded / seq_lengths  # (batch_size,)
    group_mean_log_probs = group_mean_per_token.unsqueeze(-1).expand(batch_size, response_length)
    group_mean_log_probs = group_mean_log_probs * response_mask  # Zero out padding

    return group_mean_log_probs


def _is_reshape_update_policy(self, dataloader: Iterable[DataProto]) -> dict:
    """
    Patched update_policy that tracks per-policy-update metrics.

    This is monkey-patched onto the actor to avoid creating a second actor instance
    (which would cause patch_fused_forward to be called twice).

    Returns:
        {"metrics": {...}, "update_metrics": [...]}
    """
    metrics = {}
    all_update_metrics = []

    if self.use_torch_profiler and self.prof and self.prof.enable:
        self.prof.start()

    update_idx = 0

    # Check if NoMu is enabled (read from custom configs stored on actor)
    custom_configs = getattr(self, '_custom_policy_loss_configs', {})
    use_nomu = custom_configs.get("use_nomu", False)

    for data in dataloader:
        update_metrics_raw = {}

        # === NoMu: Compute group mean and replace old_log_probs ===
        if use_nomu:
            nomu_group_mean_log_probs = _compute_nomu_group_mean(self, data)
            # Replace old_log_probs with group mean
            data.batch["old_log_probs"] = nomu_group_mean_log_probs
            if update_idx == 0:
                metrics["nomu/enabled"] = [1.0]

        if self.config.router_replay.mode in ["R2", "R3"]:
            RouterReplay.set_global_router_replay_action(RouterReplayAction.REPLAY_FORWARD)

        self.actor_optimizer.zero_grad()
        for chunk in self.actor_module:
            chunk.zero_grad_buffer()

        calculate_entropy = getattr(self.config, 'calculate_entropy', False) or (self.config.entropy_coeff != 0)
        if data.meta_info.get("micro_batch_size", None) is not None:
            micro_batch_size = data.meta_info["micro_batch_size"]
        else:
            micro_batch_size = self.config.ppo_micro_batch_size_per_gpu
        max_token_len = None
        if self.config.use_dynamic_bsz:
            max_token_len = self.config.ppo_max_token_len_per_gpu * self.config.megatron.context_parallel_size

        metric_micro_batch = self.forward_backward_batch(
            data,
            calculate_entropy=calculate_entropy,
            use_dynamic_bsz=self.config.use_dynamic_bsz,
            micro_batch_size=micro_batch_size,
            max_token_len=max_token_len,
            mini_batch_size=self.config.ppo_mini_batch_size,
        )
        metric_micro_batch = metric_micro_batch["output"]
        for metric in metric_micro_batch:
            append_to_dict(metrics, metric[0])
            append_to_dict(update_metrics_raw, metric[0])

        update_successful, grad_norm, num_zeros_in_grad = self.actor_optimizer.step()
        grad_data = {"actor/grad_norm": grad_norm}
        append_to_dict(metrics, grad_data)
        append_to_dict(update_metrics_raw, grad_data)

        if update_successful:
            pass
        else:
            raise NotImplementedError("Update failed - gradient overflow or NaN detected")

        if self.use_torch_profiler and self.prof and self.prof.enable:
            self.prof.step()

        if self.config.router_replay.mode in ["R2", "R3"]:
            RouterReplay.clear_global_router_replay_action()
            RouterReplay.clear_global_indices()

        # Collect per-update metrics on last pipeline stage
        if mpu.is_pipeline_last_stage(ignore_virtual=True):
            update_metrics_reduced = reduce_metrics(update_metrics_raw)
            update_metrics_reduced["actor/update_idx"] = update_idx
            all_update_metrics.append(update_metrics_reduced)

        update_idx += 1

    if self.use_torch_profiler and self.prof and self.prof.enable:
        self.prof.stop_and_save()
        self.prof.stop_trace()
    get_torch_device().empty_cache()

    return {"metrics": metrics, "update_metrics": all_update_metrics}


class ISReshapeMegatronActorRolloutRefWorker(AsyncActorRolloutRefWorker):
    """
    Extended Megatron AsyncActorRolloutRefWorker for IS Reshape experiments.

    Core changes from parent:
    1. Monkey-patches update_policy for per-update metrics tracking
    2. Handles custom policy_loss configs (is_reshape, gamma_is, etc.)
    3. Extracts per-update metrics from update_policy return format

    Note: Extends AsyncActorRolloutRefWorker for vLLM async mode support.
    """

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        """Initialize model and patch update_policy for per-update metrics."""
        # Step 1: Temporarily remove custom policy_loss configs to avoid dataclass conversion errors
        # This is needed because omega_conf_to_dataclass will fail on unknown fields
        custom_loss_configs = {}
        if self._is_actor and hasattr(self.config.actor, 'policy_loss'):
            policy_loss = self.config.actor.policy_loss
            all_keys = list(policy_loss.keys()) if hasattr(policy_loss, 'keys') else []
            custom_keys = [k for k in all_keys if k not in _POLICY_LOSS_KNOWN_FIELDS]

            if custom_keys:
                OmegaConf.set_struct(policy_loss, False)
                with open_dict(policy_loss):
                    for key in custom_keys:
                        custom_loss_configs[key] = policy_loss[key]
                        del policy_loss[key]

        # Step 2: Call parent's init_model (creates MegatronPPOActor, vLLM rollout, etc.)
        super().init_model()

        # Step 3: Restore custom configs after parent init
        if custom_loss_configs:
            OmegaConf.set_struct(self.config.actor.policy_loss, False)
            with open_dict(self.config.actor.policy_loss):
                for key, value in custom_loss_configs.items():
                    self.config.actor.policy_loss[key] = value
            OmegaConf.set_struct(self.config.actor.policy_loss, True)

        # Step 4: Store custom configs on actor and monkey-patch update_policy
        # We store custom configs on actor because actor.config.policy_loss is a dataclass
        # that was created before we could add custom fields
        if self._is_actor:
            self.actor._custom_policy_loss_configs = custom_loss_configs
            self.actor.update_policy = types.MethodType(_is_reshape_update_policy, self.actor)
            logger.info(f"[ISReshape] Patched actor.update_policy for per-update metrics, custom_configs={custom_loss_configs}")

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    @GPUMemoryLogger(role="update_actor", logger=logger)
    def update_actor(self, data: DataProto):
        """
        Override update_actor to handle per-update metrics.

        Patched update_policy returns:
            {"metrics": {...}, "update_metrics": [...]}

        We extract both and pass update_metrics through via metrics["__update_metrics__"].
        """
        assert self._is_actor
        if self._is_offload_param:
            load_megatron_model_to_gpu(self.actor_module)
            log_gpu_memory_usage("After load actor params and grad during update_actor", logger=logger)
        if self._is_offload_optimizer:
            load_megatron_optimizer(self.actor_optimizer)
            log_gpu_memory_usage("After load actor optimizer during update_actor", logger=logger)

        micro_batch_size = self.config.actor.ppo_micro_batch_size_per_gpu
        data.meta_info["micro_batch_size"] = micro_batch_size
        dataloader = self.actor.make_minibatch_iterator(data=data)

        with Timer(name="update_policy", logger=None) as timer:
            result = self.actor.update_policy(dataloader=dataloader)

        # Handle ISReshape return format: {"metrics": {...}, "update_metrics": [...]}
        if isinstance(result, dict) and "update_metrics" in result:
            metrics = result["metrics"]
            update_metrics = result["update_metrics"]
        else:
            metrics = result
            update_metrics = []

        # Add perf metrics (same as parent)
        delta_time = timer.last
        global_num_tokens = data.meta_info["global_token_num"]
        images_seqlens = data.meta_info.get("images_seqlens", None)
        estimated_flops, promised_flops = self.flops_counter.estimate_flops(
            global_num_tokens, delta_time, images_seqlens=images_seqlens
        )
        metrics["perf/mfu/actor"] = estimated_flops * self.config.actor.ppo_epochs / promised_flops / self.world_size
        metrics["perf/max_memory_allocated_gb"] = get_torch_device().max_memory_allocated() / (1024**3)
        metrics["perf/max_memory_reserved_gb"] = get_torch_device().max_memory_reserved() / (1024**3)
        metrics["perf/cpu_memory_used_gb"] = psutil.virtual_memory().used / (1024**3)

        from verl.utils.megatron.optimizer import get_megatron_last_lr

        metrics["actor/lr"] = get_megatron_last_lr(self.actor_optimizer)
        self.actor_optimizer_scheduler.step(1)

        # Put update_metrics inside metrics dict with special key to ensure proper handling across workers
        # (Regular meta_info keys must be identical across all workers during DataProto.concat)
        metrics["__update_metrics__"] = update_metrics
        output = DataProto(meta_info={"metrics": metrics})
        output = output.to("cpu")

        if self._is_offload_param:
            offload_megatron_model_to_cpu(self.actor_module)
            log_gpu_memory_usage("After offload actor params and grad during update_actor", logger=logger)
        if self._is_offload_optimizer:
            offload_megatron_optimizer(self.actor_optimizer)
            log_gpu_memory_usage("After offload actor optimizer during update_actor", logger=logger)

        aggressive_empty_cache(force_sync=True)
        return output
