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
Parallel Reward Loop Manager for Async Rollout Mode

This module provides a ProcessPoolExecutor-based reward loop manager that
can significantly speed up reward computation in async rollout mode.

Usage:
    1. Import this file in your training script to register the "prime" reward loop manager
    2. Set reward_model.reward_manager=prime in your config
"""

import asyncio
import inspect
from concurrent.futures import ProcessPoolExecutor
from functools import partial

from verl import DataProto
from verl.experimental.reward.reward_loop import register
from verl.experimental.reward.reward_loop.base import RewardLoopManagerBase
from verl.utils.reward_score import default_compute_score


def _compute_score_wrapper(compute_score_func, data_source, solution_str, ground_truth, extra_info):
    """Wrapper function for ProcessPoolExecutor to call compute_score."""
    try:
        result = compute_score_func(
            data_source=data_source,
            solution_str=solution_str,
            ground_truth=ground_truth,
            extra_info=extra_info,
        )
        return result
    except Exception as e:
        print(f"[Error] Reward computation failed: {e}")
        return {"score": 0.0, "acc": False, "pred": "[ERROR]"}


@register("prime")
class PrimeRewardLoopManager(RewardLoopManagerBase):
    """
    Parallel reward loop manager using ProcessPoolExecutor for true parallelization.

    This manager uses a process pool to compute rewards in parallel, avoiding
    Python's GIL limitation. It provides significant speedup for CPU-intensive
    reward functions like math_verify.

    Performance:
        - NaiveRewardLoopManager: Uses ThreadPoolExecutor (GIL-limited)
        - PrimeRewardLoopManager: Uses ProcessPoolExecutor (true parallelism)
        - Expected speedup: ~10-60x depending on CPU cores

    Args:
        config: Config object
        tokenizer: Tokenizer for decoding
        compute_score: Reward computation function
        reward_router_address: Optional reward router address
        reward_model_tokenizer: Optional reward model tokenizer
        num_processes: Number of worker processes (default: 64)
    """

    def __init__(
        self,
        config,
        tokenizer,
        compute_score=None,
        reward_router_address=None,
        reward_model_tokenizer=None,
        num_processes=128
    ):
        super().__init__(config, tokenizer)
        self.compute_score = compute_score or default_compute_score
        self.is_async_reward_score = inspect.iscoroutinefunction(self.compute_score)
        self.reward_router_address = reward_router_address
        self.reward_model_tokenizer = reward_model_tokenizer

        # Get num_processes from config if available
        reward_kwargs = config.reward_model.get("reward_kwargs", {})
        self.num_processes = reward_kwargs.get("num_processes", num_processes)

        # Create process pool
        # Note: ProcessPoolExecutor is created lazily to avoid issues with Ray
        self._executor = None

        print(f"[PrimeRewardLoopManager] Initialized with num_processes={self.num_processes}")

    def _get_executor(self):
        """Lazy initialization of ProcessPoolExecutor."""
        if self._executor is None:
            self._executor = ProcessPoolExecutor(max_workers=self.num_processes)
        return self._executor

    async def run_single(self, data: DataProto) -> dict:
        """
        Compute reward for a single data item using process pool.

        Note: Even though this is "run_single", when called from RewardLoopWorker.compute_score_batch(),
        multiple instances of this method run concurrently via asyncio.gather(), and each one
        uses the shared process pool for parallel execution.
        """
        assert len(data) == 1, "Only support single data item"
        data_item = data[0]
        response_ids = data_item.batch["responses"]
        response_length = response_ids.shape[-1]
        valid_response_length = data_item.batch["attention_mask"][-response_length:].sum()
        valid_response_ids = response_ids[:valid_response_length]

        data_source = data_item.non_tensor_batch["data_source"]
        ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
        extra_info = data_item.non_tensor_batch.get("extra_info", {})
        tool_extra_fields = data_item.non_tensor_batch.get("tool_extra_fields", None)
        if tool_extra_fields is not None:
            extra_info.update(tool_extra_fields.items())

        num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
        rollout_reward_scores = data_item.non_tensor_batch.get("reward_scores", {})
        extra_info["num_turns"] = num_turns
        extra_info["rollout_reward_scores"] = rollout_reward_scores

        # Decode response
        response_str = await self.loop.run_in_executor(
            None, lambda: self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
        )

        # For async reward functions, we still need to run them sequentially
        # because ProcessPoolExecutor doesn't support async functions
        if self.is_async_reward_score:
            extra_reward_kwargs = (
                {
                    "reward_router_address": self.reward_router_address,
                    "reward_model_tokenizer": self.reward_model_tokenizer,
                }
                if self.reward_router_address is not None
                else {}
            )
            result = await self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
                **extra_reward_kwargs,
            )
        else:
            # Use ProcessPoolExecutor for true parallelism
            executor = self._get_executor()

            # Create a partial function that captures all arguments
            compute_func = partial(
                _compute_score_wrapper,
                self.compute_score,
                data_source,
                response_str,
                ground_truth,
                extra_info,
            )

            # Submit to process pool
            result = await self.loop.run_in_executor(executor, compute_func)

        reward_extra_info = {}

        score: float
        if isinstance(result, dict):
            score = result["score"]
            for key, value in result.items():
                reward_extra_info[key] = value
        else:
            score = result
            reward_extra_info["acc"] = score

        reward = score

        return {"reward_score": reward, "reward_extra_info": reward_extra_info}

    def __del__(self):
        """Clean up process pool on deletion."""
        if self._executor is not None:
            self._executor.shutdown(wait=True)
