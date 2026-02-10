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

Main entry point for IS Reshape experiments with Megatron/FSDP backend.

Features:
1. Per-update metrics tracking: Each optimizer.step() is logged as a separate step
2. Custom policy_loss support: Handles is_reshape, gamma_is, etc. configs from core_algos.py
3. Compatible with vLLM async rollout mode

Usage:
    python3 -m recipe.vespo.code.main_ppo \\
        --config-path=../../../verl/trainer/config \\
        --config-name=ppo_megatron_trainer \\
        algorithm.adv_estimator=grpo \\
        actor_rollout_ref.actor.policy_loss.loss_mode=gamma_is \\
        ...
"""

import os
import socket

import hydra
import ray
from omegaconf import OmegaConf

from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
from verl.trainer.ppo.ray_trainer import ResourcePoolManager
from verl.trainer.ppo.reward import load_reward_manager
from verl.trainer.ppo.utils import Role, need_critic, need_reference_policy
from verl.utils.config import validate_config
from verl.workers.config.actor import PolicyLossConfig

# Import IS Reshape custom trainer
from recipe.vespo.code.trainer import ISReshapeTrainer

# Import reward_loop_prime to register the "prime_loop" reward manager
# This must be done before load_reward_manager is called
import recipe.vespo.code.reward_loop_prime  # noqa: F401

# Known fields in PolicyLossConfig dataclass - any other fields are custom loss configs
_POLICY_LOSS_KNOWN_FIELDS = frozenset(
    f.name for f in __import__("dataclasses").fields(PolicyLossConfig)
)


def _remove_custom_policy_loss_fields(config):
    """
    Temporarily remove custom policy_loss fields to avoid dataclass conversion errors.

    Returns dict of removed fields to be restored later.
    """
    custom_loss_configs = {}
    if hasattr(config.actor_rollout_ref.actor, 'policy_loss'):
        policy_loss = config.actor_rollout_ref.actor.policy_loss
        all_keys = list(policy_loss.keys()) if hasattr(policy_loss, 'keys') else []
        custom_keys = [k for k in all_keys if k not in _POLICY_LOSS_KNOWN_FIELDS]

        if custom_keys:
            OmegaConf.set_struct(policy_loss, False)
            from omegaconf import open_dict
            with open_dict(policy_loss):
                for key in custom_keys:
                    custom_loss_configs[key] = policy_loss[key]
                    del policy_loss[key]
    return custom_loss_configs


def _restore_custom_policy_loss_fields(config, custom_loss_configs):
    """Restore previously removed custom policy_loss fields."""
    if custom_loss_configs:
        OmegaConf.set_struct(config.actor_rollout_ref.actor.policy_loss, False)
        from omegaconf import open_dict
        with open_dict(config.actor_rollout_ref.actor.policy_loss):
            for key, value in custom_loss_configs.items():
                config.actor_rollout_ref.actor.policy_loss[key] = value
        OmegaConf.set_struct(config.actor_rollout_ref.actor.policy_loss, True)


@hydra.main(config_path="../../../verl/trainer/config", config_name="ppo_trainer", version_base=None)
def main(config):
    run_is_reshape_ppo(config)


def run_is_reshape_ppo(config) -> None:
    """Run PPO training with IS Reshape - uses same structure as official main_ppo."""
    if not ray.is_initialized():
        default_runtime_env = get_ppo_ray_runtime_env()
        ray_init_kwargs = config.ray_kwargs.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})

        if config.transfer_queue.enable:
            runtime_env_vars = runtime_env_kwargs.get("env_vars", {})
            runtime_env_vars["TRANSFER_QUEUE_ENABLE"] = "1"
            runtime_env_kwargs["env_vars"] = runtime_env_vars

        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        print(f"ray init kwargs: {ray_init_kwargs}")

        ray_init_dict = OmegaConf.to_container(ray_init_kwargs)

        is_ray_job = os.environ.get('RAY_JOB_ID') is not None
        has_ray_address_env = os.environ.get('RAY_ADDRESS') is not None
        has_address = 'address' in ray_init_dict and ray_init_dict['address'] is not None

        if is_ray_job or has_ray_address_env or has_address:
            print("Connecting to existing Ray cluster, not setting object_store_memory")
            ray.init(**ray_init_dict)
        else:
            print("Starting new Ray cluster with object_store_memory")
            ray.init(**ray_init_dict, object_store_memory=536870912000)

    task_runner_class = ray.remote(num_cpus=1)(TaskRunner)
    runner = task_runner_class.remote()
    ray.get(runner.run.remote(config))


class TaskRunner:
    """Task runner for IS Reshape training - simplified structure matching official."""

    def __init__(self):
        self.role_worker_mapping = {}
        self.mapping = {}

    def add_actor_rollout_worker(self, config):
        """Add actor rollout worker based on the actor strategy."""
        from verl.single_controller.ray import RayWorkerGroup

        # Use IS Reshape custom workers for both FSDP and Megatron strategies
        if config.actor_rollout_ref.actor.strategy in {"fsdp", "fsdp2"}:
            from recipe.vespo.code.fsdp_workers import ISReshapeRolloutRefWorker
            actor_rollout_cls = ISReshapeRolloutRefWorker
            ray_worker_group_cls = RayWorkerGroup

        elif config.actor_rollout_ref.actor.strategy == "megatron":
            from recipe.vespo.code.megatron_workers import ISReshapeMegatronActorRolloutRefWorker
            actor_rollout_cls = ISReshapeMegatronActorRolloutRefWorker
            ray_worker_group_cls = RayWorkerGroup

        else:
            raise NotImplementedError(f"Strategy {config.actor_rollout_ref.actor.strategy} not supported")

        self.role_worker_mapping[Role.ActorRollout] = ray.remote(actor_rollout_cls)
        self.mapping[Role.ActorRollout] = "global_pool"
        return actor_rollout_cls, ray_worker_group_cls

    def add_critic_worker(self, config):
        """Add critic worker to role mapping."""
        if not need_critic(config):
            return

        if config.critic.strategy in {"fsdp", "fsdp2"}:
            from verl.workers.fsdp_workers import CriticWorker
        elif config.critic.strategy == "megatron":
            from verl.workers.megatron_workers import CriticWorker
        else:
            raise NotImplementedError

        self.role_worker_mapping[Role.Critic] = ray.remote(CriticWorker)
        self.mapping[Role.Critic] = "global_pool"

    def add_reward_model_worker(self, config):
        """Add reward model worker if enabled."""
        if config.reward_model.enable:
            if config.reward_model.strategy in {"fsdp", "fsdp2"}:
                from verl.workers.fsdp_workers import RewardModelWorker
            elif config.reward_model.strategy == "megatron":
                from verl.workers.megatron_workers import RewardModelWorker
            else:
                raise NotImplementedError

            self.role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            if config.reward_model.enable_resource_pool:
                self.mapping[Role.RewardModel] = "reward_pool"
            else:
                self.mapping[Role.RewardModel] = "global_pool"

    def add_ref_policy_worker(self, config, ref_policy_cls):
        """Add reference policy worker if KL loss or KL reward is used."""
        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            self.role_worker_mapping[Role.RefPolicy] = ray.remote(ref_policy_cls)
            self.mapping[Role.RefPolicy] = "global_pool"

    def init_resource_pool_mgr(self, config):
        """Initialize resource pool manager."""
        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }

        if config.reward_model.enable_resource_pool:
            if config.reward_model.n_gpus_per_node <= 0:
                raise ValueError("config.reward_model.n_gpus_per_node must be greater than 0")
            if config.reward_model.nnodes <= 0:
                raise ValueError("config.reward_model.nnodes must be greater than 0")

            reward_pool = [config.reward_model.n_gpus_per_node] * config.reward_model.nnodes
            resource_pool_spec["reward_pool"] = reward_pool

        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=self.mapping)
        return resource_pool_manager

    def run(self, config):
        """Execute the main PPO training workflow."""
        from pprint import pprint
        from verl.utils.fs import copy_to_local

        print(f"TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        # Temporarily remove custom policy_loss fields to avoid dataclass conversion errors
        custom_loss_configs = _remove_custom_policy_loss_fields(config)

        actor_rollout_cls, ray_worker_group_cls = self.add_actor_rollout_worker(config)
        self.add_critic_worker(config)
        self.add_reward_model_worker(config)
        self.add_ref_policy_worker(config, actor_rollout_cls)

        validate_config(
            config=config,
            use_reference_policy=need_reference_policy(self.role_worker_mapping),
            use_critic=need_critic(config),
        )

        # Restore custom policy_loss fields after validation
        _restore_custom_policy_loss_fields(config, custom_loss_configs)

        local_path = copy_to_local(
            config.actor_rollout_ref.model.path, use_shm=config.actor_rollout_ref.model.get("use_shm", False)
        )

        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        reward_fn = load_reward_manager(
            config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {})
        )
        val_reward_fn = load_reward_manager(
            config, tokenizer, num_examine=1, **config.reward_model.get("reward_kwargs", {})
        )

        resource_pool_manager = self.init_resource_pool_mgr(config)

        from verl.utils.dataset.rl_dataset import collate_fn, get_dataset_class

        dataset_cls = get_dataset_class(config.data)
        train_dataset = dataset_cls(
            data_files=config.data.train_files,
            tokenizer=tokenizer,
            processor=processor,
            config=config.data,
        )
        val_dataset = dataset_cls(
            data_files=config.data.val_files,
            tokenizer=tokenizer,
            processor=processor,
            config=config.data,
        )

        # Create sampler
        import torch
        from torchdata.stateful_dataloader.sampler import RandomSampler
        from torch.utils.data import SequentialSampler

        if config.data.shuffle:
            train_dataloader_generator = torch.Generator()
            seed = config.data.get("seed")
            if seed is not None:
                train_dataloader_generator.manual_seed(seed)
            train_sampler = RandomSampler(data_source=train_dataset, generator=train_dataloader_generator)
        else:
            train_sampler = SequentialSampler(data_source=train_dataset)

        # Use ISReshapeTrainer for per-update metrics logging
        trainer = ISReshapeTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=self.role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
        )
        trainer.init_workers()
        trainer.fit()


if __name__ == "__main__":
    main()
