# Copyright 2024 GeoGuessr RLHF Project
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
GeoGuessr Reward Manager with Ray-based Parallel Computation

This reward manager uses Ray remote functions for parallel computation,
which works correctly within Ray workers (unlike ProcessPoolExecutor).
Supports dictionary return values with multiple metrics (score, distance,
accuracy, etc.).
"""

from collections import defaultdict
from typing import Any, Callable, Optional

import ray
import torch
from transformers import PreTrainedTokenizer

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager


@ray.remote
def compute_single_score_ray(module_info, task, completion, reference, task_extra_info):
    """
    Ray remote function to compute score for a single sample.

    This function re-imports the reward function in the remote worker to avoid
    serialization issues with custom modules.

    Args:
        module_info: Dict with 'type' and function info for loading the reward function
        task: Task/data source identifier
        completion: Model's output string
        reference: Ground truth
        task_extra_info: Extra information dict

    Returns:
        Score (float or dict)
    """
    try:
        # Load the reward function in the remote worker
        if module_info['type'] == 'custom':
            # Re-import the custom module in this worker
            import importlib.util
            import sys

            file_path = module_info['file_path']
            function_name = module_info['function_name']
            reward_kwargs = module_info.get('reward_kwargs', {})

            # Check if already loaded
            if 'custom_module' not in sys.modules:
                spec = importlib.util.spec_from_file_location("custom_module", file_path)
                module = importlib.util.module_from_spec(spec)
                sys.modules["custom_module"] = module
                spec.loader.exec_module(module)
            else:
                module = sys.modules["custom_module"]

            raw_fn = getattr(module, function_name)

            # Wrap with kwargs if provided (mimicking _call_with_kwargs behavior)
            if reward_kwargs:
                # Merge reward_kwargs with task_extra_info
                merged_extra_info = {**(task_extra_info or {}), **reward_kwargs}
                result = raw_fn(task, completion, reference, merged_extra_info)
            else:
                result = raw_fn(task, completion, reference, task_extra_info)

        elif module_info['type'] == 'default':
            from verl.utils.reward_score import default_compute_score
            result = default_compute_score(task, completion, reference, task_extra_info)
        else:
            raise ValueError(f"Unknown module_info type: {module_info['type']}")

        return result
    except Exception as e:
        print(f"[Error] Score computation failed: {e}, completion: {completion[:80] if completion else 'N/A'}")
        import traceback
        traceback.print_exc()
        # Return default failure dict
        return {
            "score": 0.0,
            "distance@km": 20000.0,
            "acc@1km": 0.0,
            "acc@25km": 0.0,
            "acc@200km": 0.0,
            "acc@750km": 0.0,
            "acc@2500km": 0.0,
            "geoguessr@point": 0.0,
            "parse_success": False
        }


def run_reward_scoring_ray(module_info, completions, references, tasks, extra_info=None):
    """
    Compute scores for multiple samples in parallel using Ray.

    This function supports reward functions that return either:
    - A single float/int score
    - A dictionary with 'score' key and additional metrics

    Args:
        module_info: Dict containing info to load the reward function in remote workers
        completions: List of model output strings
        references: List of ground truth values
        tasks: List of task/data source identifiers
        extra_info: List of extra information dicts (optional)

    Returns:
        List of score dicts
    """
    if extra_info is None:
        extra_info = [None] * len(tasks)

    # Launch all tasks in parallel using Ray
    futures = []
    for c, r, t, ei in zip(completions, references, tasks, extra_info, strict=True):
        future = compute_single_score_ray.remote(module_info, t, c, r, ei)
        futures.append(future)

    # Get all results (with timeout per task handled by Ray)
    try:
        results = ray.get(futures, timeout=300.0)
    except ray.exceptions.GetTimeoutError:
        print("[Timeout] Some reward scoring tasks timed out")
        # Get partial results
        ready, not_ready = ray.wait(futures, num_returns=len(futures), timeout=0)
        results = []
        for i, future in enumerate(futures):
            if future in ready:
                try:
                    results.append(ray.get(future))
                except Exception as e:
                    print(f"[Error] Task {i} failed: {e}")
                    results.append(None)
            else:
                print(f"[Timeout] Task {i} timed out")
                results.append(None)
    except Exception as e:
        print(f"[Error] Reward scoring failed: {e}")
        import traceback
        traceback.print_exc()
        results = [None] * len(completions)

    # Process results - handle both dict and numeric returns
    scores = []
    for result, completion, reference, task in zip(results, completions, references, tasks, strict=True):
        if isinstance(result, Exception) or result is None:
            # Handle failed or timed-out tasks
            scores.append({
                "score": 0.0,
                "distance@km": 20000.0,
                "acc@1km": 0.0,
                "acc@25km": 0.0,
                "acc@200km": 0.0,
                "acc@750km": 0.0,
                "acc@2500km": 0.0,
                "geoguessr@point": 0.0,
                "parse_success": False
            })
        elif isinstance(result, dict):
            # Dictionary return (expected for GeoGuessr)
            scores.append(result)
        elif isinstance(result, int | float | bool):
            # Scalar return (backward compatibility)
            scores.append({"score": float(result)})
        elif isinstance(result, (list, tuple)) and len(result) > 0:
            # Legacy format: (score, ...) tuple
            scores.append({"score": float(result[0])})
        else:
            # Unknown format
            print(f"[Warning] Unknown result format: {type(result)}, using 0.0")
            scores.append({"score": 0.0})

    return scores


@register("geoguessr")
class GeoGuessrRewardManager(AbstractRewardManager):
    """
    Reward Manager optimized for GeoGuessr tasks with Ray-based parallel computation.

    This manager:
    - Computes rewards in parallel using Ray remote functions (Ray-compatible)
    - Supports reward functions that return dictionaries with multiple metrics
    - Tracks and returns all metrics (score, distance, accuracy, etc.)
    - Compatible with the verl reward manager interface

    Example usage in config:
        reward_model.reward_manager=geoguessr
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        num_examine: int,
        compute_score: Optional[Callable] = None,
        reward_fn_key: str = "data_source",
    ) -> None:
        """
        Initialize GeoGuessr Reward Manager.

        Args:
            tokenizer: Tokenizer for decoding model outputs
            num_examine: Number of samples to print for debugging
            compute_score: Reward function (should return dict with 'score' key)
            reward_fn_key: Key in non_tensor_batch for data source
        """
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key

        # Prepare module info for Ray remote workers
        self._prepare_module_info()

    def _prepare_module_info(self):
        """
        Prepare module information for loading the reward function in remote workers.
        This avoids serialization issues with custom modules.
        """
        import sys
        from functools import partial

        # Check if compute_score is a wrapped function (partial)
        actual_func = self.compute_score
        reward_kwargs = {}

        if isinstance(self.compute_score, partial):
            # It's wrapped by partial(_call_with_kwargs, raw_fn, reward_kwargs)
            # Extract the actual reward function
            if len(self.compute_score.args) >= 2:
                actual_func = self.compute_score.args[0]  # The raw_fn
                reward_kwargs = self.compute_score.args[1] if len(self.compute_score.args) > 1 else {}
                print(f"[GeoGuessrRewardManager] Detected wrapped function, extracted actual_func")

        # Check if using custom reward function
        if 'custom_module' in sys.modules:
            custom_module = sys.modules['custom_module']

            # Find which function is being used
            function_name = None
            for name in dir(custom_module):
                obj = getattr(custom_module, name)
                if callable(obj) and obj == actual_func:
                    function_name = name
                    break

            if function_name:
                self.module_info = {
                    'type': 'custom',
                    'file_path': custom_module.__file__,
                    'function_name': function_name,
                    'reward_kwargs': reward_kwargs
                }
                print(f"[GeoGuessrRewardManager] Using custom reward function: {function_name} from {custom_module.__file__}")
            else:
                self.module_info = {'type': 'default', 'reward_kwargs': {}}
                print("[GeoGuessrRewardManager] Using default compute_score")
        else:
            self.module_info = {'type': 'default', 'reward_kwargs': {}}
            print("[GeoGuessrRewardManager] Using default compute_score")

    def verify(self, data):
        """
        Compute rewards for a batch and store as 'acc' tensor.

        Args:
            data: DataProto batch

        Returns:
            List of score dictionaries
        """
        # Extract data
        prompt_ids = data.batch["prompts"]
        response_ids = data.batch["responses"]

        # Decode responses
        sequences_str = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)

        # Get ground truths and data sources
        ground_truth = [data_item.non_tensor_batch["reward_model"]["ground_truth"] for data_item in data]
        data_sources = data.non_tensor_batch[self.reward_fn_key]
        extra_info = data.non_tensor_batch.get("extra_info", None)

        assert len(sequences_str) == len(ground_truth) == len(data_sources)

        # Compute scores in parallel using Ray
        try:
            score_dicts = run_reward_scoring_ray(
                self.module_info,
                completions=sequences_str,
                references=ground_truth,
                tasks=data_sources,
                extra_info=extra_info,
            )
        except Exception as e:
            print(f"[Error] Unexpected error during scoring. Setting all as 0. {e}")
            import traceback
            traceback.print_exc()
            score_dicts = [{
                "score": 0.0,
                "distance@km": 20000.0,
                "acc@1km": 0.0,
                "acc@25km": 0.0,
                "acc@200km": 0.0,
                "acc@750km": 0.0,
                "acc@2500km": 0.0,
                "geoguessr@point": 0.0,
                "parse_success": False
            } for _ in range(len(sequences_str))]

        # Extract scores and store in batch
        scores = []
        for score_dict in score_dicts:
            if isinstance(score_dict, dict):
                scores.append(score_dict.get("score", 0.0))
            else:
                scores.append(float(score_dict))

        data.batch["acc"] = torch.tensor(scores, dtype=torch.float32, device=prompt_ids.device)

        return score_dicts

    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        """
        Compute rewards for a batch of data.

        Args:
            data: DataProto containing prompts, responses, and metadata
            return_dict: Whether to return dictionary with extra info

        Returns:
            If return_dict=False: reward_tensor (torch.Tensor)
            If return_dict=True: dict with 'reward_tensor' and 'reward_extra_info'
        """
        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
                reward_extra_info = {key: data.non_tensor_batch[key] for key in reward_extra_keys}
                return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": reward_extra_info}
            else:
                return data.batch["rm_scores"]

        # Initialize reward tensor
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)

        # Track extra metrics
        reward_extra_info = defaultdict(list)

        # Get prompt and response lengths
        prompt_ids = data.batch["prompts"]
        prompt_length = prompt_ids.shape[-1]
        response_ids = data.batch["responses"]
        valid_response_length = data.batch["attention_mask"][:, prompt_length:].sum(dim=-1)

        # Decode for printing
        sequences_str = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
        data_sources = data.non_tensor_batch["data_source"]

        # Compute scores
        score_dicts = self.verify(data)

        # Track which data sources we've already printed
        already_print_data_sources = {}

        # Process each sample
        for i in range(len(data)):
            data_source = data_sources[i]
            score_dict = score_dicts[i]

            # Extract score
            if isinstance(score_dict, dict):
                score = score_dict.get("score", 0.0)
                # Store all metrics
                for key, value in score_dict.items():
                    reward_extra_info[key].append(value)
            else:
                score = float(score_dict)
                reward_extra_info["score"].append(score)

            # Set reward at the last valid token position
            reward_tensor[i, valid_response_length[i].item() - 1] = score

            # Print samples for debugging
            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print(f"\n[Data Source] {data_source}")
                print(f"[Response] {sequences_str[i]}...")
                print(f"[Score Dict] {score_dict}")

        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
        else:
            return reward_tensor
