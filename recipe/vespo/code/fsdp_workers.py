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
VESPO FSDP Workers

This module extends the veRL FSDP workers to use the custom ISReshapeDataParallelPPOActor,
which tracks per-policy-update metrics for VESPO experiments.

Usage:
    In your training script, use ISReshapeRolloutRefWorker instead of ActorRolloutRefWorker:

    ```python
    from recipe.vespo.code.fsdp_workers import ISReshapeRolloutRefWorker

    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ISReshapeRolloutRefWorker),
        # ... other roles
    }
    ```
"""

import logging
import os

import psutil
import torch
from codetiming import Timer
from omegaconf import OmegaConf, open_dict

from verl import DataProto
from verl.single_controller.base.decorator import Dispatch, register
from verl.utils.debug import log_gpu_memory_usage
from verl.utils.fsdp_utils import (
    load_fsdp_model_to_gpu,
    load_fsdp_optimizer,
    offload_fsdp_model_to_cpu,
    offload_fsdp_optimizer,
)
from verl.utils.torch_functional import get_torch_device
from verl.workers.config.actor import PolicyLossConfig
from verl.workers.fsdp_workers import AsyncActorRolloutRefWorker, get_device_id
from verl.utils.config import omega_conf_to_dataclass

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_PPO_LOGGING_LEVEL", "WARN"))

__all__ = ["ISReshapeRolloutRefWorker"]

# Known fields in PolicyLossConfig dataclass - any other fields are custom loss configs
_POLICY_LOSS_KNOWN_FIELDS = frozenset(
    f.name for f in __import__("dataclasses").fields(PolicyLossConfig)
)


class ISReshapeRolloutRefWorker(AsyncActorRolloutRefWorker):
    """
    Extended AsyncActorRolloutRefWorker that uses ISReshapeDataParallelPPOActor.

    This worker overrides init_model to instantiate the custom actor class
    that tracks per-policy-update VESPO metrics (gamma, sigma_sq, ess_ratio).

    Note: We extend AsyncActorRolloutRefWorker instead of ActorRolloutRefWorker
    because vLLM only supports async mode and requires methods like get_zeromq_address.
    """

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        """
        Initialize model with automatic handling of custom policy loss configs.

        The parent's init_model() tries to convert config.actor to ActorConfig dataclass,
        which includes converting policy_loss to PolicyLossConfig. Since PolicyLossConfig
        has fixed fields, any custom loss config keys (like is_reshape, is_reshape_pos, etc.)
        would cause errors.

        This method automatically:
        1. Detects custom keys in policy_loss that are not in PolicyLossConfig
        2. Temporarily removes them before calling super().init_model()
        3. Restores them afterward for use by the policy loss functions
        """
        custom_loss_configs = {}

        if hasattr(self.config.actor, 'policy_loss'):
            policy_loss = self.config.actor.policy_loss

            # Find all custom keys (not in PolicyLossConfig dataclass)
            all_keys = list(policy_loss.keys()) if hasattr(policy_loss, 'keys') else []
            custom_keys = [k for k in all_keys if k not in _POLICY_LOSS_KNOWN_FIELDS]

            # Save and remove custom configs
            if custom_keys:
                OmegaConf.set_struct(policy_loss, False)
                with open_dict(policy_loss):
                    for key in custom_keys:
                        custom_loss_configs[key] = policy_loss[key]
                        del policy_loss[key]

        # Call parent's init_model to handle all the standard initialization
        super().init_model()

        # Replace the actor with our custom VESPO actor
        # Note: super().init_model() already created self.actor with a properly converted ActorConfig
        # We reuse that config and just replace the actor class
        if self._is_actor:
            from recipe.vespo.code.dp_actor import ISReshapeDataParallelPPOActor

            # Reuse the ActorConfig that was already created by parent's init_model
            # This is properly converted from DictConfig and required by GSPO and other policy losses
            actor_cfg = self.actor.config

            # Add custom loss configs to the ActorConfig's policy_loss
            # These were extracted before super().init_model() to avoid dataclass conversion errors
            if custom_loss_configs and hasattr(actor_cfg, 'policy_loss'):
                for key, value in custom_loss_configs.items():
                    # Use object.__setattr__ to bypass frozen check if needed
                    if hasattr(actor_cfg.policy_loss, '_mutable_fields'):
                        actor_cfg.policy_loss._mutable_fields.add(key)
                    setattr(actor_cfg.policy_loss, key, value)

            self.actor = ISReshapeDataParallelPPOActor(
                config=actor_cfg,
                actor_module=self.actor_module_fsdp,
                actor_optimizer=self.actor_optimizer,
            )

        # Replace reference policy actor if needed
        if self._is_ref:
            from recipe.vespo.code.dp_actor import ISReshapeDataParallelPPOActor

            # Reuse the RefConfig that was already created by parent's init_model
            ref_cfg = self.ref_policy.config
            self.ref_policy = ISReshapeDataParallelPPOActor(
                config=ref_cfg,
                actor_module=self.ref_module_fsdp,
            )

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def update_actor(self, data: DataProto):
        """
        Override update_actor to handle per-update metrics from ISReshapeDataParallelPPOActor.

        The custom actor returns {"metrics": {...}, "update_metrics": [...]} instead of
        just a metrics dict. This method extracts both and passes them through to the
        trainer via meta_info.
        """
        assert self._is_actor
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)
        if self._is_offload_optimizer:
            load_fsdp_optimizer(optimizer=self.actor_optimizer, device_id=get_device_id())

        with self.ulysses_sharding_manager:
            data = data.to("cpu")  # data will to device with each micro batch on actor.update_policy

            # perform training
            with Timer(name="update_policy", logger=None) as timer:
                result = self.actor.update_policy(data=data)

            # Handle the new return format: {"metrics": {...}, "update_metrics": [...]}
            if isinstance(result, dict) and "update_metrics" in result:
                metrics = result["metrics"]
                update_metrics = result["update_metrics"]
            else:
                # Backward compatibility: old format returns just metrics dict
                metrics = result
                update_metrics = []

            delta_time = timer.last
            global_num_tokens = data.meta_info["global_token_num"]
            estimated_flops, promised_flops = self.flops_counter.estimate_flops(global_num_tokens, delta_time)
            metrics["perf/mfu/actor"] = (
                estimated_flops * self.config.actor.ppo_epochs / promised_flops / self.world_size
            )
            metrics["perf/max_memory_allocated_gb"] = get_torch_device().max_memory_allocated() / (1024**3)
            metrics["perf/max_memory_reserved_gb"] = get_torch_device().max_memory_reserved() / (1024**3)
            metrics["perf/cpu_memory_used_gb"] = psutil.virtual_memory().used / (1024**3)

            lr = self.actor_lr_scheduler.get_last_lr()[0]
            metrics["actor/lr"] = lr.item() if torch.is_tensor(lr) else lr
            self.actor_lr_scheduler.step()

            # Return both aggregated metrics and per-update metrics
            # Put update_metrics inside metrics dict with special key to ensure proper handling across workers
            # (Regular meta_info keys must be identical across all workers during DataProto.concat)
            metrics["__update_metrics__"] = update_metrics
            output = DataProto(meta_info={"metrics": metrics})

            output = output.to("cpu")

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
            log_gpu_memory_usage("After offload actor model during update_actor", logger=logger)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.actor_optimizer)
            log_gpu_memory_usage("After offload actor optimizer during update_actor", logger=logger)

        return output
