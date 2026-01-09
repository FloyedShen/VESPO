# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
Unified reward function using math-verify for mathematical answer verification.

This module provides a standardized reward function interface that uses the
math-verify library (https://github.com/huggingface/Math-Verify) for accurate
mathematical answer verification.

Requirements:
    pip install math-verify

Important Note:
    This implementation uses parsing_timeout=None in the parse() calls to disable
    the signal-based timeout mechanism. This is necessary for compatibility with
    multithreaded environments (e.g., Ray workers in distributed training).

    If you need timeout protection, you should implement it at a higher level
    (e.g., using concurrent.futures.ThreadPoolExecutor with timeout).

Usage in training scripts:
    custom_reward_function.path=recipe/is_shape/code/reward_function.py \
    custom_reward_function.name=math \
"""

# Import reward_loop_prime to register PrimeRewardLoopManager
# This ensures the "prime" reward loop manager is available in Ray worker processes
import os
import sys
import importlib.util

_current_dir = os.path.dirname(os.path.abspath(__file__))
_reward_loop_prime_path = os.path.join(_current_dir, "reward_loop_prime.py")

if os.path.exists(_reward_loop_prime_path):
    try:
        _spec = importlib.util.spec_from_file_location("reward_loop_prime_module", _reward_loop_prime_path)
        if _spec and _spec.loader:
            _reward_loop_prime_module = importlib.util.module_from_spec(_spec)
            sys.modules["reward_loop_prime_module"] = _reward_loop_prime_module
            _spec.loader.exec_module(_reward_loop_prime_module)
    except Exception as e:
        print(f"[Warning] Failed to load reward_loop_prime: {e}")

try:
    from math_verify import parse, verify
    from math_verify.errors import TimeoutException
except ImportError as e:
    raise ImportError(
        "Math-Verify is required for this reward function. "
        "Please install it by running: pip install math-verify"
    ) from e


def math(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    sandbox_fusion_url=None,
    concurrent_semaphore=None,
    memory_limit_mb=None,
    **kwargs,
) -> dict[str, float]:
    """
    Compute reward score for mathematical answers using math-verify.

    This function uses the math-verify library's parse() and verify() functions
    to accurately compare model outputs with ground truth answers. It handles
    various mathematical notations and formats.

    Args:
        solution_str (str): The model's generated solution/answer
        ground_truth (str): The ground truth answer (may contain LaTeX, expressions, etc.)
        timeout_score (float, optional): Score to return if verification times out. Defaults to 0.0.

    Returns:
        dict: Dictionary containing:
            - 'score' (float): 1.0 if correct, 0.0 if incorrect or timeout
            - 'acc' (bool): True if correct, False otherwise
            - 'pred' (str): The extracted prediction string

    Examples:
        >>> gold = parse("${1,3} \\cup {2,4}$")
        >>> answer = parse("${1,2,3,4}$")
        >>> verify(gold, answer)
        True

        >>> result = math("The answer is {1,2,3,4}", "${1,3} \\cup {2,4}$")
        >>> result
        {'score': 1.0, 'acc': True, 'pred': '{1,2,3,4}'}
    """
    ret_score = 0.0
    acc = False
    extracted_pred = None

    try:
        # Parse both the ground truth and the model output
        # The parse() function automatically handles various extraction configurations
        # IMPORTANT: Use parsing_timeout=None to disable signal-based timeout which
        # doesn't work in multithreaded environments (e.g., Ray workers)
        gold_parsed = parse(ground_truth, parsing_timeout=300)
        pred_parsed = parse(solution_str, parsing_timeout=300)

        # Extract the string representation for logging
        extracted_pred = str(pred_parsed) if pred_parsed else "[INVALID]"

        # Verify if the parsed answers match
        # verify() returns True if mathematically equivalent, False otherwise
        # IMPORTANT: Set timeout_seconds=None to disable signal-based timeout
        is_correct = verify(gold_parsed, pred_parsed, timeout_seconds=300)

        if is_correct:
            ret_score = 1.0
            acc = True

    except Exception as e:
        # Handle any other parsing or verification errors
        ret_score = 0.0
        acc = False
        extracted_pred = f"[ERROR: {str(e)}]"

    return {
        "score": ret_score,
        "acc": acc,
        "pred": extracted_pred,
    }


def instruction_following(solution_str: str, ground_truth: str, **kwargs) -> dict[str, float]:
    """
    Placeholder reward function for instruction following tasks.

    This is a simple exact match implementation. You may want to replace this
    with a more sophisticated instruction-following evaluation method.

    Args:
        solution_str (str): The model's generated output
        ground_truth (str): The expected output
        **kwargs: Additional arguments (ignored)

    Returns:
        dict: Dictionary containing score, acc, and pred
    """
    # Simple exact match for now
    # TODO: Replace with proper instruction-following evaluation
    is_correct = solution_str.strip() == ground_truth.strip()

    return {
        "score": 1.0 if is_correct else 0.0,
        "acc": is_correct,
        "pred": solution_str.strip(),
    }


# Mapping of ability types to reward functions
REWARD_FUNCTIONS = {
    "math": math,
    "instruction_following": instruction_following,
}


def get_reward_function(ability: str):
    """
    Get the appropriate reward function for a given ability type.

    Args:
        ability (str): The ability type (e.g., 'math', 'instruction_following')

    Returns:
        callable: The reward function

    Raises:
        ValueError: If the ability type is not supported
    """
    if ability not in REWARD_FUNCTIONS:
        raise ValueError(
            f"Unsupported ability type: {ability}. "
            f"Supported types: {list(REWARD_FUNCTIONS.keys())}"
        )
    return REWARD_FUNCTIONS[ability]


# Expose the main reward function
__all__ = ["math", "instruction_following", "get_reward_function", "REWARD_FUNCTIONS"]
