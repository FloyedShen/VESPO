#!/usr/bin/env python3
"""
Test script for the custom reward function.

This script tests the math reward function to ensure it works correctly
with the math-verify library.
"""

import sys
sys.path.insert(0, '/mnt/tidal-alsh-hilab/usr/shenguobin/verl/recipe/is_shape/code')

from reward_function import math, get_reward_function


def test_basic_math():
    """Test basic mathematical answer verification."""
    print("=" * 60)
    print("Test 1: Basic Set Operations")
    print("=" * 60)

    # Test case from your example
    gold = "${1,3} \\cup {2,4}$"
    answer = "${1,2,3,4}$"

    result = math(answer, gold)
    print(f"Ground truth: {gold}")
    print(f"Model answer: {answer}")
    print(f"Result: {result}")
    print(f"Expected: Correct (score=1.0)")
    print()


def test_incorrect_answer():
    """Test with incorrect answer."""
    print("=" * 60)
    print("Test 2: Incorrect Answer")
    print("=" * 60)

    gold = "$42$"
    answer = "$43$"

    result = math(answer, gold)
    print(f"Ground truth: {gold}")
    print(f"Model answer: {answer}")
    print(f"Result: {result}")
    print(f"Expected: Incorrect (score=0.0)")
    print()


def test_latex_expressions():
    """Test with LaTeX expressions."""
    print("=" * 60)
    print("Test 3: LaTeX Expressions")
    print("=" * 60)

    # Fraction equivalence
    gold = "$\\frac{1}{2}$"
    answer = "$0.5$"

    result = math(answer, gold)
    print(f"Ground truth: {gold}")
    print(f"Model answer: {answer}")
    print(f"Result: {result}")
    print(f"Expected: Correct (should be equivalent)")
    print()


def test_boxed_answer():
    """Test with boxed answers."""
    print("=" * 60)
    print("Test 4: Boxed Answers")
    print("=" * 60)

    gold = "$100$"
    answer = "The answer is \\boxed{100}"

    result = math(answer, gold)
    print(f"Ground truth: {gold}")
    print(f"Model answer: {answer}")
    print(f"Result: {result}")
    print(f"Expected: Correct (score=1.0)")
    print()


def test_get_reward_function():
    """Test the get_reward_function utility."""
    print("=" * 60)
    print("Test 5: Get Reward Function")
    print("=" * 60)

    # Get math function
    math_func = get_reward_function("math")
    result = math_func("$42$", "$42$")
    print(f"Math function result: {result}")
    print(f"Expected: score=1.0, acc=True")
    print()


def test_return_format():
    """Verify the return format matches verl expectations."""
    print("=" * 60)
    print("Test 6: Return Format Verification")
    print("=" * 60)

    result = math("$42$", "$42$")

    # Check that result is a dict with required keys
    assert isinstance(result, dict), "Result must be a dictionary"
    assert "score" in result, "Result must contain 'score' key"
    assert "acc" in result, "Result must contain 'acc' key"
    assert "pred" in result, "Result must contain 'pred' key"

    # Check types
    assert isinstance(result["score"], float), "score must be float"
    assert isinstance(result["acc"], bool), "acc must be bool"
    assert isinstance(result["pred"], str), "pred must be str"

    print(f"✓ Return format is correct")
    print(f"  Keys: {list(result.keys())}")
    print(f"  Types: score={type(result['score']).__name__}, "
          f"acc={type(result['acc']).__name__}, pred={type(result['pred']).__name__}")
    print()


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("Testing Custom Reward Function")
    print("=" * 60 + "\n")

    try:
        test_basic_math()
        test_incorrect_answer()
        test_latex_expressions()
        test_boxed_answer()
        test_get_reward_function()
        test_return_format()

        print("=" * 60)
        print("✅ All tests completed!")
        print("=" * 60)

    except Exception as e:
        print("=" * 60)
        print(f"❌ Test failed with error: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
