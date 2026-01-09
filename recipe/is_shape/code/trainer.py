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
IS Reshape Trainer

Custom trainer that logs metrics at each policy update step.

Step counting:
- global_steps: counts policy updates (mini_batch optimizer steps)
- batch_idx: counts rollout batches (used for val/save frequency)

Frequency settings:
- test_freq: based on rollout batch count (batch_idx), same as original RayPPOTrainer
- save_freq: based on rollout batch count (batch_idx), same as original RayPPOTrainer

Metric logging:
- Each policy update is logged as a separate step for fine-grained analysis
"""

from collections import defaultdict
from copy import deepcopy
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm

from verl import DataProto
from verl.trainer.ppo.core_algos import AdvantageEstimator
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
)
from verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer,
    ResourcePoolManager,
    apply_kl_penalty,
    compute_advantage,
    compute_response_mask,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics

__all__ = ["ISReshapeTrainer", "ResourcePoolManager"]


class ISReshapeTrainer(RayPPOTrainer):
    """
    Custom trainer for IS Reshape experiments.

    Key difference from RayPPOTrainer:
    - Logs metrics at each policy update (optimizer step), not just each batch
    - global_steps counts policy updates for fine-grained metric logging
    - val/save frequency still based on rollout batch count (same as original)
    """

    def fit(self):
        """
        Training loop with per-policy-update step counting.

        Each policy update (optimizer step on a mini_batch) is counted as a
        separate global_step, allowing visualization of how metrics evolve
        during training.

        val/save frequency is based on rollout batch count (batch_idx).
        """
        from omegaconf import OmegaConf
        from verl.utils.tracking import Tracking
        import ray
        import uuid

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0  # Counts policy updates (for metric logging)
        self._load_checkpoint()

        ppo_epochs = self.config.actor_rollout_ref.actor.ppo_epochs
        ppo_mini_batch_size = self.config.actor_rollout_ref.actor.ppo_mini_batch_size
        train_batch_size = self.config.data.train_batch_size
        rollout_n = self.config.actor_rollout_ref.rollout.n

        # Calculate policy updates per rollout batch
        effective_batch_size = train_batch_size * rollout_n
        num_mini_batches = effective_batch_size // ppo_mini_batch_size
        policy_updates_per_batch = ppo_epochs * num_mini_batches

        # Total rollout batches for the entire training
        total_rollout_batches = len(self.train_dataloader) * self.config.trainer.total_epochs
        if self.config.trainer.total_training_steps is not None:
            total_rollout_batches = self.config.trainer.total_training_steps

        # Total policy updates
        total_policy_updates = total_rollout_batches * policy_updates_per_batch

        # Resume from checkpoint: calculate current batch_idx from global_steps
        current_batch_idx = self.global_steps // policy_updates_per_batch
        current_epoch = current_batch_idx // len(self.train_dataloader)

        # Validation before training
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            if val_metrics:
                from pprint import pprint
                pprint(f"Initial validation metrics: {val_metrics}")
                logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        progress_bar = tqdm(total=total_policy_updates, initial=self.global_steps, desc="Training Progress")

        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0
        rollout_batch_idx = current_batch_idx  # Counts rollout batches (for val/save freq)

        for epoch in range(current_epoch, self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                timing_raw = {}
                batch_metrics = {}  # Metrics computed once per batch (rollout, reward, etc.)

                batch: DataProto = DataProto.from_single_dict(batch_dict)
                batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )

                gen_batch = self._get_gen_batch(batch)
                gen_batch.meta_info["global_steps"] = self.global_steps
                gen_batch_output = gen_batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                )

                with marked_timer("step", timing_raw):
                    # Generate sequences (once per batch)
                    with marked_timer("gen", timing_raw, color="red"):
                        if not self.async_rollout_mode:
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch_output)
                        else:
                            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)
                        timing_raw.update(gen_batch_output.meta_info.get("timing", {}))
                        gen_batch_output.meta_info.pop("timing", None)

                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)

                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(batch)

                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=batch_metrics)

                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    # Compute reward (once per batch)
                    with marked_timer("reward", timing_raw, color="yellow"):
                        if self.use_rm and "rm_scores" not in batch.batch.keys():
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(
                                data=batch, config=self.config, tokenizer=self.tokenizer
                            )
                        else:
                            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

                    # Compute old_log_prob (once per batch)
                    rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
                    bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)

                    if bypass_recomputing_logprobs:
                        from verl.trainer.ppo.rollout_corr_helper import apply_rollout_correction
                        apply_rollout_correction(
                            batch=batch,
                            rollout_corr_config=rollout_corr_config,
                            policy_loss_config=self.config.actor_rollout_ref.actor.policy_loss,
                        )
                    else:
                        with marked_timer("old_log_prob", timing_raw, color="blue"):
                            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                            old_log_prob.batch.pop("entropys", None)
                            batch = batch.union(old_log_prob)

                    # Compute ref_log_prob if needed (once per batch)
                    if self.use_reference_policy:
                        with marked_timer("ref_log_prob", timing_raw, color="olive"):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # Compute values if using critic (once per batch)
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    # Compute advantages (once per batch)
                    with marked_timer("adv", timing_raw, color="brown"):
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            batch_metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )

                    # Update critic (once per batch, before actor epochs)
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        batch_metrics.update(critic_output_metrics)

                    # Now do PPO updates - each policy update is a separate step
                    if self.config.trainer.critic_warmup <= (rollout_batch_idx + 1):
                        with marked_timer("update_actor", timing_raw, color="red"):
                            rollout_config = self.config.actor_rollout_ref.rollout
                            batch.meta_info["multi_turn"] = rollout_config.multi_turn.enable
                            batch.meta_info["temperature"] = rollout_config.temperature

                            # Call custom update that returns per-update metrics
                            actor_output = self.actor_rollout_wg.update_actor(batch)

                # Compute batch-level metrics after step timing is complete
                batch_metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                batch_metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                n_gpus = self.resource_pool_manager.get_n_gpus()
                batch_metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                # Handle per-policy-update metrics logging
                if self.config.trainer.critic_warmup <= (rollout_batch_idx + 1):
                    actor_metrics = actor_output.meta_info.get("metrics", {})

                    # Extract update_metrics if available
                    all_update_metrics = []
                    if "__update_metrics__" in actor_metrics:
                        update_metrics_raw = actor_metrics["__update_metrics__"]
                        if isinstance(update_metrics_raw, list) and len(update_metrics_raw) > 0:
                            if isinstance(update_metrics_raw[0], list):
                                all_update_metrics = update_metrics_raw[0]
                            else:
                                all_update_metrics = update_metrics_raw

                    if all_update_metrics:
                        # Log each policy update as a separate step
                        for update_idx, update_metrics in enumerate(all_update_metrics):
                            step_metrics = {}

                            # Add batch-level metrics only at first update of this batch
                            if update_idx == 0:
                                step_metrics.update(batch_metrics)

                            # Add update-level metrics (filter out internal keys)
                            for key, value in update_metrics.items():
                                if not key.startswith("__"):
                                    step_metrics[key] = value

                            # Add training info
                            step_metrics["training/global_step"] = self.global_steps
                            step_metrics["training/rollout_batch_idx"] = rollout_batch_idx
                            step_metrics["training/epoch"] = epoch
                            step_metrics["training/update_idx"] = update_idx

                            logger.log(data=step_metrics, step=self.global_steps)
                            progress_bar.update(1)
                            self.global_steps += 1

                        # Check for last step (based on policy updates)
                        is_last_step = self.global_steps >= total_policy_updates
                    else:
                        # Fallback: For standard actors that don't return per-update metrics
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        actor_output_metrics = {k: v for k, v in actor_output_metrics.items() if not k.startswith("__")}

                        # Log each policy update with the same aggregated actor metrics
                        for update_idx in range(policy_updates_per_batch):
                            step_metrics = {}

                            if update_idx == 0:
                                step_metrics.update(batch_metrics)

                            step_metrics.update(actor_output_metrics)

                            step_metrics["training/global_step"] = self.global_steps
                            step_metrics["training/rollout_batch_idx"] = rollout_batch_idx
                            step_metrics["training/epoch"] = epoch
                            step_metrics["training/update_idx"] = update_idx

                            logger.log(data=step_metrics, step=self.global_steps)
                            progress_bar.update(1)
                            self.global_steps += 1

                        is_last_step = self.global_steps >= total_policy_updates
                else:
                    # Critic warmup - just count steps without logging metrics
                    for _ in range(policy_updates_per_batch):
                        progress_bar.update(1)
                        self.global_steps += 1
                    is_last_step = self.global_steps >= total_policy_updates

                # Increment rollout batch counter
                rollout_batch_idx += 1

                # Validation check after each rollout batch (based on rollout_batch_idx)
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (is_last_step or rollout_batch_idx % self.config.trainer.test_freq == 0)
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    logger.log(data=val_metrics, step=self.global_steps)

                # Checkpointing check after each rollout batch (based on rollout_batch_idx)
                esi_close_to_expiration = should_save_ckpt_esi(
                    max_steps_duration=self.max_steps_duration,
                    redundant_time=self.config.trainer.esi_redundant_time,
                )
                if self.config.trainer.save_freq > 0 and (
                    is_last_step or rollout_batch_idx % self.config.trainer.save_freq == 0 or esi_close_to_expiration
                ):
                    with marked_timer("save_checkpoint", timing_raw, color="green"):
                        self._save_checkpoint()

                if is_last_step:
                    from pprint import pprint
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return
