#!/usr/bin/env python3
"""
Test reward function in multithreaded environment.

This script tests that the math reward function works correctly with
parsing_timeout=None in a multithreaded context (simulating Ray workers).
"""

import concurrent.futures
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from reward_function import math as math_reward


def test_single_case(solution_str, ground_truth, expected_acc):
    """Test a single case."""
    result = math_reward(
        data_source="test",
        solution_str=solution_str,
        ground_truth=ground_truth,
    )

    success = result["acc"] == expected_acc
    status = "✓" if success else "✗"
    print(f"{status} ground_truth={ground_truth[:20]:20s} pred={result['pred']:20s} acc={result['acc']} score={result['score']}")
    return success


def test_case_wrapper(args):
    """Wrapper for testing in thread pool."""
    return test_single_case(*args)


def main():
    print("=" * 80)
    print("Testing Math Reward Function in Multithreaded Environment")
    print("=" * 80)

    # Test cases: (solution_str, ground_truth, expected_acc)
    test_cases = [
        # Correct answers
        ("The answer is 27", "27", True),
        ("The result is \\boxed{27}", "27", True),
        ("\\boxed{1.5}", "1.5", True),
        ("The answer is \\boxed{\\frac{3}{2}}", "1.5", True),

        # Incorrect answers
        ("The answer is 28", "27", False),
        ("\\boxed{2}", "1.5", False),

        # Complex cases
        ("{1,2,3,4}", "{1,3} \\cup {2,4}", True),
        ("\\boxed{\\frac{27}{1}}", "27", True),
    ]

    print("\n1. Sequential Test:")
    print("-" * 80)
    sequential_results = []
    for args in test_cases:
        sequential_results.append(test_single_case(*args))

    print(f"\nSequential: {sum(sequential_results)}/{len(test_cases)} passed")

    print("\n2. Multithreaded Test (simulating Ray workers):")
    print("-" * 80)

    # Test with ThreadPoolExecutor (simulates multithreaded environment)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        parallel_results = list(executor.map(test_case_wrapper, test_cases))

    print(f"\nMultithreaded: {sum(parallel_results)}/{len(test_cases)} passed")

    # Overall result
    print("\n" + "=" * 80)
    if all(sequential_results) and all(parallel_results):
        print("✓ All tests passed!")
        print("✓ The reward function works correctly in multithreaded environments.")
        return 0
    else:
        print("✗ Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
