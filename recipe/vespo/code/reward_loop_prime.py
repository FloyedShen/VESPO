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
Prime Reward Manager for IS Reshape Training

This module provides a Ray actor-based reward manager that uses separate processes
for reward computation, solving the math-verify hang issue.

The key insight from https://github.com/volcengine/verl/pull/4752:
- Math-verify uses signal-based timeouts which don't work in threads
- ProcessPoolExecutor can have pickling issues with complex functions
- Ray actors provide true process isolation and handle function initialization
  inside each worker, bypassing pickling issues

Usage:
    1. Import this file in your training script to register the "prime_loop" reward manager
    2. Set reward_model.reward_manager=prime_loop in your config
    3. Set reward_model.num_workers=N to control the number of Ray actor workers
"""

import asyncio
import itertools
import time
from collections import defaultdict

import ray
import torch
from omegaconf import DictConfig
from transformers import AutoTokenizer

from verl import DataProto
from verl.experimental.reward_loop.reward_manager import register as register_manager
from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register as register_manager_legacy


@ray.remote(num_cpus=1)
class RewardComputeWorker:
    """
    Ray actor worker for computing rewards in a separate process.

    Uses multiprocessing.Process internally to enforce hard timeout,
    solving the math-verify hang issue where signal-based timeouts don't work.
    """

    def __init__(self, compute_score_fn):
        self.compute_score_fn = compute_score_fn
        self.worker_id = id(self)
        print(f"[RewardComputeWorker-{self.worker_id}] Initialized")

    def compute_score(self, timeout: float = 60.0, **kwargs) -> dict:
        """Compute reward score with hard timeout using multiprocessing."""
        import multiprocessing as mp
        import traceback

        start_time = time.time()
        process = None
        parent_conn = None

        # Extract info for logging
        solution_str = kwargs.get("solution_str", "")
        ground_truth = kwargs.get("ground_truth", "")
        # Truncate for logging
        solution_preview = solution_str[:500] + "..." if len(solution_str) > 500 else solution_str
        gt_preview = str(ground_truth)[:200] + "..." if len(str(ground_truth)) > 200 else str(ground_truth)

        try:
            # Use Pipe instead of Queue for better reliability
            parent_conn, child_conn = mp.Pipe(duplex=False)

            def _worker(conn, compute_fn, kwargs):
                import signal
                signal.signal(signal.SIGTERM, signal.SIG_DFL)
                signal.signal(signal.SIGINT, signal.SIG_DFL)
                try:
                    result = compute_fn(**kwargs)
                    conn.send(("success", result))
                except Exception as e:
                    conn.send(("error", str(e)))
                finally:
                    conn.close()

            process = mp.Process(target=_worker, args=(child_conn, self.compute_score_fn, kwargs))
            process.start()
            child_conn.close()  # Close child's end in parent

            # Wait for result with timeout
            if parent_conn.poll(timeout=timeout):
                status, result = parent_conn.recv()
                elapsed = time.time() - start_time
                if status == "success":
                    return result
                else:
                    print(f"[RewardComputeWorker-{self.worker_id}] Error after {elapsed:.1f}s: {result}")
                    print(f"  Ground Truth: {gt_preview}")
                    print(f"  Response: {solution_preview}")
                    return {"score": 0.0, "acc": False, "pred": f"[ERROR: {result}]"}
            else:
                # Timeout - kill the process with SIGKILL
                elapsed = time.time() - start_time
                print(f"[RewardComputeWorker-{self.worker_id}] TIMEOUT after {elapsed:.1f}s (limit={timeout}s), killing process")
                print(f"  Ground Truth: {gt_preview}")
                print(f"  Response: {solution_preview}")
                process.kill()
                process.join(timeout=5)
                return {"score": 0.0, "acc": False, "pred": "[TIMEOUT]"}

        except Exception as e:
            elapsed = time.time() - start_time
            print(f"[RewardComputeWorker-{self.worker_id}] Exception after {elapsed:.1f}s: {e}")
            print(f"  Ground Truth: {gt_preview}")
            print(f"  Response: {solution_preview}")
            traceback.print_exc()
            return {"score": 0.0, "acc": False, "pred": f"[ERROR: {str(e)}]"}
        finally:
            # Ensure cleanup
            if parent_conn is not None:
                try:
                    parent_conn.close()
                except:
                    pass
            if process is not None and process.is_alive():
                process.kill()
                process.join(timeout=1)


@register_manager("prime_loop")
@register_manager_legacy("prime_loop")
class PrimeRewardManager(RewardManagerBase):
    """
    Prime Reward Manager using Ray actors for true process isolation.

    This manager solves the math-verify hang issue by:
    1. Using Ray actors (separate processes) instead of threads
    2. Initializing the compute_score function inside each worker process
    3. Using node affinity to schedule workers on the same node for efficiency

    The "prime_loop" name distinguishes it from the official "prime" reward manager.

    Args:
        config: Config object containing reward_model settings
        tokenizer: Tokenizer for decoding response tokens
        compute_score: Reward computation function (e.g., math verification)
        reward_router_address: Optional address for external reward router
        reward_model_tokenizer: Optional tokenizer for reward model
        num_examine: (Legacy) Number of samples to examine (unused, for compatibility)
        reward_fn_key: (Legacy) Key for reward function (unused, for compatibility)

    Config options:
        reward_model.num_workers: Number of Ray actor workers (default: 8)
    """

    def __init__(
        self,
        config: DictConfig | None = None,
        tokenizer: AutoTokenizer | None = None,
        compute_score=None,
        reward_router_address=None,
        reward_model_tokenizer=None,
        # Legacy (AbstractRewardManager) kwargs for compatibility
        num_examine: int | None = None,
        reward_fn_key: str | None = None,
        **kwargs,
    ):
        # Handle case where config is None (legacy AbstractRewardManager signature)
        if config is None:
            config = DictConfig({"reward_model": {}})
        if tokenizer is None:
            raise TypeError("PrimeRewardManager requires `tokenizer`.")

        super().__init__(config, tokenizer)
        self.compute_score = compute_score or default_compute_score
        self.reward_router_address = reward_router_address
        self.reward_model_tokenizer = reward_model_tokenizer

        # Get number of workers from config (default: 16)
        rm_cfg = config.get("reward_model") or {}
        num_reward_workers = rm_cfg.get("num_workers", 16)

        print(f"[PrimeRewardManager] Config reward_model: {rm_cfg}")
        print(f"[PrimeRewardManager] num_workers from config: {num_reward_workers}")

        # Create Ray actor workers on the same node for efficiency
        # Each worker runs in a separate process, solving the math-verify hang issue
        self.reward_workers = [
            RewardComputeWorker.options(
                scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                    node_id=ray.get_runtime_context().get_node_id(),
                    soft=True,
                ),
            ).remote(self.compute_score)
            for _ in range(num_reward_workers)
        ]
        self.reward_worker_pool = itertools.cycle(self.reward_workers)

        print(f"[PrimeRewardManager] Initialized with {num_reward_workers} Ray actor workers")

    def choose_reward_worker(self):
        """Round-robin selection of reward workers for load balancing."""
        return next(self.reward_worker_pool)

    async def run_single(self, data: DataProto) -> dict:
        """
        Compute reward for a single data item using Ray actor workers.

        This method is called concurrently via asyncio.gather() from the reward loop,
        and each call uses a Ray actor worker for true parallel execution.

        Args:
            data: DataProto containing a single sample

        Returns:
            dict with reward_score and reward_extra_info
        """
        assert len(data) == 1, "Only support single data item"
        data_item = data[0]

        # Extract response information
        response_ids = data_item.batch["responses"]
        response_length = response_ids.shape[-1]
        valid_response_length = data_item.batch["attention_mask"][-response_length:].sum()
        valid_response_ids = response_ids[:valid_response_length]

        # Extract metadata
        data_source = data_item.non_tensor_batch["data_source"]
        ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
        extra_info = data_item.non_tensor_batch.get("extra_info", {})

        # Handle tool-related extra fields
        tool_extra_fields = data_item.non_tensor_batch.get("tool_extra_fields", None)
        if tool_extra_fields is not None:
            extra_info.update(tool_extra_fields.items())

        # Add turn and reward score information
        num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
        rollout_reward_scores = data_item.non_tensor_batch.get("reward_scores", {})
        extra_info["num_turns"] = num_turns
        extra_info["rollout_reward_scores"] = rollout_reward_scores

        # Decode response string
        response_str = await self.loop.run_in_executor(
            None, lambda: self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
        )

        # Prepare extra kwargs for reward router if available
        extra_reward_kwargs = (
            {
                "reward_router_address": self.reward_router_address,
                "reward_model_tokenizer": self.reward_model_tokenizer,
            }
            if self.reward_router_address is not None
            else {}
        )

        # Submit to Ray actor worker for computation with timeout
        reward_worker = self.choose_reward_worker()
        ray_timeout = 120.0  # Timeout for Ray call itself

        try:
            # Use asyncio.wait_for to add timeout on the Ray call
            result = await asyncio.wait_for(
                reward_worker.compute_score.remote(
                    timeout=60.0,  # Timeout for subprocess
                    data_source=data_source,
                    solution_str=response_str,
                    ground_truth=ground_truth,
                    extra_info=extra_info,
                    **extra_reward_kwargs,
                ),
                timeout=ray_timeout
            )
        except asyncio.TimeoutError:
            print(f"[PrimeRewardManager.run_single] Ray call timeout after {ray_timeout}s")
            result = {"score": 0.0, "acc": False, "pred": "[RAY_TIMEOUT]"}
        except Exception as e:
            print(f"[PrimeRewardManager.run_single] Error: {e}")
            result = {"score": 0.0, "acc": False, "pred": f"[ERROR: {str(e)}]"}

        # Process result
        reward_extra_info = {}
        if isinstance(result, dict):
            score = result["score"]
            for key, value in result.items():
                reward_extra_info[key] = value
        else:
            score = result
            reward_extra_info["acc"] = score

        return {"reward_score": score, "reward_extra_info": reward_extra_info}

    def __call__(self, data: DataProto, return_dict: bool = False):
        """
        Compute reward for a batch of data using Ray actor workers.

        This method provides compatibility with the standard reward manager interface.
        It processes the batch in parallel using Ray actors.

        Args:
            data: DataProto containing the batch data
            return_dict: If True, return dict with reward_tensor and reward_extra_info.
                        If False, return only reward_tensor.

        Returns:
            torch.Tensor | dict: Reward tensor or dict with tensor and extra info
        """
        # If there are pre-computed rm_scores, return them directly
        if "rm_scores" in data.batch.keys():
            if return_dict:
                reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
                reward_extra_info = {key: data.non_tensor_batch[key] for key in reward_extra_keys}
                return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": reward_extra_info}
            else:
                return data.batch["rm_scores"]

        # Initialize reward tensor
        prompt_ids = data.batch["prompts"]
        prompt_length = prompt_ids.shape[-1]
        response_ids = data.batch["responses"]
        valid_response_length = data.batch["attention_mask"][:, prompt_length:].sum(dim=-1)
        reward_tensor = torch.zeros_like(response_ids, dtype=torch.float32)

        # Decode all responses
        sequences_str = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
        data_sources = data.non_tensor_batch["data_source"]
        ground_truths = [data_item.non_tensor_batch["reward_model"]["ground_truth"] for data_item in data]
        extra_infos = data.non_tensor_batch.get("extra_info", None)
        if extra_infos is None:
            extra_infos = [{}] * len(data)

        # Submit all tasks to Ray actors
        print(f"[PrimeRewardManager.__call__] Submitting {len(data)} tasks to Ray actors...")
        start_time = time.time()

        futures = []
        for i in range(len(data)):
            reward_worker = self.choose_reward_worker()
            extra_info = extra_infos[i] if isinstance(extra_infos, list) else {}
            if not isinstance(extra_info, dict):
                extra_info = {}
            future = reward_worker.compute_score.remote(
                timeout=60.0,  # Timeout for subprocess
                data_source=data_sources[i],
                solution_str=sequences_str[i],
                ground_truth=ground_truths[i],
                extra_info=extra_info,
            )
            futures.append(future)

        # Gather results with timeout to prevent hanging
        total_timeout = max(300.0, len(futures) * 5.0)  # At least 5 min, or 5s per sample
        print(f"[PrimeRewardManager.__call__] Waiting for results with timeout={total_timeout}s...")

        # Initialize results list with None (for timeout cases)
        results = [None] * len(futures)

        # Create mapping from ObjectRef to index BEFORE ray.wait
        # Use ray.ObjectRef's internal id for mapping since ObjectRefs are hashable
        future_to_idx = {f: i for i, f in enumerate(futures)}

        try:
            # Try to get all results with timeout
            ready_futures, not_ready = ray.wait(
                futures,
                num_returns=len(futures),
                timeout=total_timeout
            )

            elapsed = time.time() - start_time
            print(f"[PrimeRewardManager.__call__] ray.wait returned after {elapsed:.1f}s: "
                  f"{len(ready_futures)} ready, {len(not_ready)} not ready")

            # Get results from ready futures
            if ready_futures:
                ready_results = ray.get(ready_futures)
                for future, result in zip(ready_futures, ready_results):
                    # ObjectRefs are hashable, so we can use them directly as keys
                    idx = future_to_idx.get(future)
                    if idx is not None:
                        results[idx] = result
                    else:
                        print(f"[PrimeRewardManager.__call__] WARNING: Could not find index for future")

            # Handle timed out futures
            if not_ready:
                print(f"[PrimeRewardManager.__call__] WARNING: {len(not_ready)} tasks timed out after {total_timeout}s")
                for future in not_ready:
                    try:
                        ray.cancel(future, force=True)
                    except Exception as e:
                        print(f"[PrimeRewardManager.__call__] Failed to cancel future: {e}")

        except Exception as e:
            elapsed = time.time() - start_time
            print(f"[PrimeRewardManager.__call__] Error after {elapsed:.1f}s: {e}")
            import traceback
            traceback.print_exc()

        # Process results and build reward tensor
        reward_extra_info = defaultdict(list)
        timeout_count = 0
        success_count = 0

        for i, result in enumerate(results):
            if result is None:
                # Timeout or error case
                score = 0.0
                reward_extra_info["score"].append(0.0)
                reward_extra_info["acc"].append(False)
                reward_extra_info["pred"].append("[TIMEOUT]")
                timeout_count += 1
            elif isinstance(result, dict):
                score = result.get("score", 0.0)
                for key, value in result.items():
                    reward_extra_info[key].append(value)
                if result.get("acc", False):
                    success_count += 1
            else:
                score = float(result) if result is not None else 0.0
                reward_extra_info["acc"].append(score)

            # Place reward at the last valid position
            reward_tensor[i, valid_response_length[i].item() - 1] = score

        total_elapsed = time.time() - start_time
        print(f"[PrimeRewardManager.__call__] Completed in {total_elapsed:.1f}s: "
              f"{success_count} correct, {timeout_count} timeouts, {len(data) - success_count - timeout_count} incorrect")

        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": dict(reward_extra_info)}
        else:
            return reward_tensor
