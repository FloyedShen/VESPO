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
IS Reshape PPO Trainer

This is the main entry point for IS Reshape experiments.
It uses ISReshapeRolloutRefWorker to track per-epoch metrics.

Usage:
    python3 -u -m recipe.is_shape.code.main_ppo \\
        algorithm.adv_estimator=grpo \\
        actor_rollout_ref.actor.policy_loss.loss_mode=is_reshape \\
        actor_rollout_ref.actor.policy_loss.is_reshape.rho_min=0.3 \\
        actor_rollout_ref.actor.policy_loss.is_reshape.gamma_min=0.05 \\
        actor_rollout_ref.actor.policy_loss.is_reshape.gamma_max=1.0 \\
        ...
"""

import os

import hydra
import ray

# Import custom reward loop manager for async mode acceleration
import recipe.is_shape.code.reward_loop_prime  # noqa: F401 - Register "prime" reward loop manager

from verl.trainer.ppo.ray_trainer import ResourcePoolManager
from verl.trainer.ppo.reward import get_custom_reward_fn
from verl.trainer.ppo.utils import Role, need_critic, need_reference_policy, need_reward_model

# Import IS Reshape custom trainer
from recipe.is_shape.code.trainer import ISReshapeTrainer


@hydra.main(config_path="../../../verl/trainer/config", config_name="ppo_trainer", version_base=None)
def main(config):
    run_is_reshape_ppo(config)


def run_is_reshape_ppo(config) -> None:
    """Run PPO training with IS Reshape."""
    os.environ["ENSURE_CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not ray.is_initialized():
        ray.init(
            runtime_env={
                "env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN", "VLLM_LOGGING_LEVEL": "WARN"}
            }
        )

    runner = TaskRunner.remote()
    ray.get(runner.run.remote(config))


@ray.remote(num_cpus=1)
class TaskRunner:
    def run(self, config):
        from pprint import pprint

        from omegaconf import OmegaConf

        from verl.utils.fs import copy_to_local

        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        # Import IS Reshape custom worker
        if config.actor_rollout_ref.actor.strategy in {"fsdp", "fsdp2"}:
            from recipe.is_shape.code.fsdp_workers import ISReshapeRolloutRefWorker
            from verl.single_controller.ray import RayWorkerGroup

            ray_worker_group_cls = RayWorkerGroup
            ActorRolloutRefWorkerCls = ISReshapeRolloutRefWorker

            if config.get("critic", None) and config.critic.get("strategy", None):
                assert config.critic.strategy in {"fsdp", "fsdp2"}
                from verl.workers.fsdp_workers import CriticWorker
        else:
            raise NotImplementedError(f"Strategy {config.actor_rollout_ref.actor.strategy} not supported for IS Reshape")

        # Setup role-worker mapping
        role_worker_mapping = {}
        use_reference_policy = config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss

        if use_reference_policy:
            role_worker_mapping[Role.ActorRolloutRef] = ray.remote(ActorRolloutRefWorkerCls)
        else:
            role_worker_mapping[Role.ActorRollout] = ray.remote(ActorRolloutRefWorkerCls)

        if need_critic(config):
            role_worker_mapping[Role.Critic] = ray.remote(CriticWorker)

        if need_reward_model(role_worker_mapping) and config.reward_model.enable:
            if config.reward_model.strategy in {"fsdp", "fsdp2"}:
                from verl.workers.fsdp_workers import RewardModelWorker
            else:
                raise NotImplementedError
            role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)

        # Resource pool setup
        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }

        mapping = {}
        for role in role_worker_mapping:
            mapping[role] = global_pool_id

        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        # Load tokenizer
        local_path = copy_to_local(config.actor_rollout_ref.model.path)

        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, use_fast=True)

        # Setup reward functions
        from verl.workers.reward_manager import get_reward_manager_cls

        reward_manager_name = config.reward_model.get("reward_manager", "naive")
        reward_manager_cls = get_reward_manager_cls(reward_manager_name)

        compute_score = get_custom_reward_fn(config)
        reward_kwargs = dict(config.reward_model.get("reward_kwargs", {}))
        reward_fn = reward_manager_cls(
            tokenizer=tokenizer,
            num_examine=0,
            compute_score=compute_score,
            reward_fn_key=config.data.reward_fn_key,
            **reward_kwargs,
        )

        val_reward_fn = reward_manager_cls(
            tokenizer=tokenizer,
            num_examine=1,
            compute_score=compute_score,
            reward_fn_key=config.data.reward_fn_key,
        )

        # Create and run trainer
        trainer = ISReshapeTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
        )
        trainer.init_workers()
        trainer.fit()


if __name__ == "__main__":
    main()
