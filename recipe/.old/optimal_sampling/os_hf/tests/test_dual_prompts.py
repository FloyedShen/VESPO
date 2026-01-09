#!/usr/bin/env python3
"""
Test script for dual prompts and chat template functionality

Tests:
1. Dual prompts creation (different for base and teacher)
2. Chat template formatting
3. Natural language template for base model
4. Solution hint for teacher
5. End-to-end generation with different prompts
"""

import sys
import torch
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_imports():
    """Test that chat template functions can be imported"""
    print("\n" + "=" * 80)
    print("Test 1: Imports")
    print("=" * 80)
    try:
        from optimal_sampling_hf import (
            OptimalSamplingModel,
            create_dual_prompts,
            NATURAL_LANGUAGE_TEMPLATE
        )
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False


def test_natural_language_template():
    """Test natural language template formatting"""
    print("\n" + "=" * 80)
    print("Test 2: Natural Language Template")
    print("=" * 80)
    try:
        from optimal_sampling_hf import NATURAL_LANGUAGE_TEMPLATE
        from jinja2 import Template

        # Test message
        messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "What is 2+2?"}
        ]

        # Render template
        template = Template(NATURAL_LANGUAGE_TEMPLATE)
        prompt = template.render(messages=messages, add_generation_prompt=True)

        print("Messages:")
        for msg in messages:
            print(f"  {msg['role']}: {msg['content'][:50]}...")

        print(f"\nRendered prompt:")
        print(f"  {prompt[:200]}...")

        # Check format
        assert "Question:" in prompt or "Answer:" in prompt, "Template should contain Question/Answer"

        print("\n✓ Natural language template works")
        return True
    except Exception as e:
        print(f"✗ Template test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_create_dual_prompts():
    """Test creating dual prompts for base and teacher"""
    print("\n" + "=" * 80)
    print("Test 3: Create Dual Prompts")
    print("=" * 80)
    try:
        from optimal_sampling_hf import OptimalSamplingModel, create_dual_prompts, NATURAL_LANGUAGE_TEMPLATE

        # Initialize model to get tokenizers
        print("Initializing models...")
        model = OptimalSamplingModel(
            model_theta_path="Qwen/Qwen2.5-1.5B-Instruct",
            model_t_path="Qwen/Qwen2.5-3B-Instruct",
            alpha_method="fixed",
            fixed_alpha=0.5,
            dtype=torch.bfloat16,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )

        # Create test messages
        messages_list = [
            [
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": "What is the capital of France?"}
            ]
        ]

        # Create dual prompts
        print("\nCreating dual prompts...")
        prompts_theta, prompts_t = create_dual_prompts(
            messages_list,
            model.tokenizer_theta,
            model.tokenizer_t,
            force_nlt_in_theta=True,
            base_template=NATURAL_LANGUAGE_TEMPLATE,
            add_generation_prompt=True
        )

        print(f"\nBase model (θ) prompt (Natural Language):")
        print(f"  {prompts_theta[0][:150]}...")

        print(f"\nTeacher model (t) prompt (Chat Template):")
        print(f"  {prompts_t[0][:150]}...")

        # Verify they are different
        assert prompts_theta[0] != prompts_t[0], "Prompts should be different"
        assert "Question:" in prompts_theta[0], "Base should use natural language format"

        print("\n✓ Dual prompts creation works")
        return True
    except Exception as e:
        print(f"✗ Dual prompts test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_solution_hint():
    """Test that teacher can see solution hint while base only sees problem"""
    print("\n" + "=" * 80)
    print("Test 4: Solution Hint for Teacher")
    print("=" * 80)
    try:
        from optimal_sampling_hf import OptimalSamplingModel, create_dual_prompts, NATURAL_LANGUAGE_TEMPLATE

        # Initialize model
        print("Initializing models...")
        model = OptimalSamplingModel(
            model_theta_path="Qwen/Qwen2.5-1.5B-Instruct",
            model_t_path="Qwen/Qwen2.5-3B-Instruct",
            alpha_method="fixed",
            fixed_alpha=0.5,
            dtype=torch.bfloat16,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )

        # Problem for base (no hint)
        messages_theta = [[
            {"role": "system", "content": "Solve the problem step by step"},
            {"role": "user", "content": "What is 2 + 2?"}
        ]]

        # Problem for teacher (with hint)
        messages_teacher = [[
            {"role": "system", "content": "Solve the problem step by step"},
            {"role": "user", "content": "What is 2 + 2?\n\n##Hint\nAdd the two numbers together: 2 + 2 = 4"}
        ]]

        # Create prompts
        prompts_theta, _ = create_dual_prompts(
            messages_theta,
            model.tokenizer_theta,
            model.tokenizer_t,
            force_nlt_in_theta=True,
            base_template=NATURAL_LANGUAGE_TEMPLATE
        )

        _, prompts_teacher = create_dual_prompts(
            messages_teacher,
            model.tokenizer_theta,
            model.tokenizer_t,
            force_nlt_in_theta=False  # Teacher uses standard template
        )

        print(f"\nBase sees (no hint):")
        print(f"  {prompts_theta[0][:150]}...")

        print(f"\nTeacher sees (with hint):")
        print(f"  {prompts_teacher[0][:200]}...")

        # Verify hint is in teacher prompt but not in base
        assert "##Hint" not in prompts_theta[0], "Base should not see hint"
        assert "##Hint" in prompts_teacher[0], "Teacher should see hint"

        print("\n✓ Solution hint works correctly")
        return True
    except Exception as e:
        print(f"✗ Solution hint test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_end_to_end_generation():
    """Test end-to-end generation with dual prompts"""
    print("\n" + "=" * 80)
    print("Test 5: End-to-End Generation with Dual Prompts")
    print("=" * 80)
    try:
        from optimal_sampling_hf import OptimalSamplingModel, create_dual_prompts, NATURAL_LANGUAGE_TEMPLATE

        # Initialize model
        print("Initializing models...")
        model = OptimalSamplingModel(
            model_theta_path="Qwen/Qwen2.5-1.5B-Instruct",
            model_t_path="Qwen/Qwen2.5-3B-Instruct",
            alpha_method="kl_symmetry",
            dtype=torch.bfloat16,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )

        # Create different messages for base and teacher
        messages_theta = [[
            {"role": "system", "content": "Reason step by step"},
            {"role": "user", "content": "What is the capital of France?"}
        ]]

        messages_teacher = [[
            {"role": "system", "content": "Demonstrate clear thinking"},
            {"role": "user", "content": "What is the capital of France?\n\n##Hint\nFrance is in Europe. Paris is the capital."}
        ]]

        # Create dual prompts
        prompts_theta, _ = create_dual_prompts(
            messages_theta,
            model.tokenizer_theta,
            model.tokenizer_t,
            force_nlt_in_theta=True,
            base_template=NATURAL_LANGUAGE_TEMPLATE
        )

        _, prompts_teacher = create_dual_prompts(
            messages_teacher,
            model.tokenizer_theta,
            model.tokenizer_t,
            force_nlt_in_theta=False
        )

        print("\nGenerating with dual prompts...")
        print(f"Base prompt: {prompts_theta[0][:80]}...")
        print(f"Teacher prompt: {prompts_teacher[0][:80]}...")

        # Generate
        outputs = model.generate(
            prompts=prompts_theta,      # Base sees natural language
            prompts_t=prompts_teacher,  # Teacher sees chat template with hint
            max_new_tokens=50,
            temperature=0.8,
            return_diagnostics=True
        )

        print(f"\nGenerated text:")
        print(f"  {outputs.generated_texts[0][:200]}...")

        print(f"\nDiagnostics:")
        print(f"  Tokens: {outputs.generated_ids.shape[1]}")
        print(f"  Alpha mean: {outputs.alpha_values.mean():.3f}")
        print(f"  ESS ratio: {outputs.ess_ratios.mean():.3f}")

        # Validate output
        assert len(outputs.generated_texts) == 1, "Should generate 1 text"
        assert outputs.generated_ids.shape[1] > 0, "Should generate some tokens"
        assert outputs.alpha_values.shape[1] > 0, "Should have alpha values"

        print("\n✓ End-to-end generation with dual prompts works")
        return True
    except Exception as e:
        print(f"✗ End-to-end test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("=" * 80)
    print("Testing Dual Prompts and Chat Template Functionality")
    print("=" * 80)

    # Run tests
    results = []
    results.append(("Imports", test_imports()))

    if results[-1][1]:  # Only continue if imports successful
        results.append(("Natural Language Template", test_natural_language_template()))
        results.append(("Create Dual Prompts", test_create_dual_prompts()))
        results.append(("Solution Hint", test_solution_hint()))
        results.append(("End-to-End Generation", test_end_to_end_generation()))

    # Print summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {test_name}")

    total_tests = len(results)
    passed_tests = sum(1 for _, passed in results if passed)
    print(f"\nTotal: {passed_tests}/{total_tests} tests passed")

    if passed_tests == total_tests:
        print("\n✅ All tests passed!")
        return 0
    else:
        print(f"\n❌ {total_tests - passed_tests} test(s) failed")
        return 1


if __name__ == "__main__":
    exit(main())
