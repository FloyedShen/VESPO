#!/usr/bin/env python3
"""
Basic tests for Optimal Sampling HF

Tests:
1. Package imports
2. Model initialization
3. Basic generation
4. Output format validation
"""

import sys
import torch
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_imports():
    """Test that all main components can be imported"""
    print("Testing imports...")
    try:
        from optimal_sampling_hf import (
            OptimalSamplingModel,
            AlphaComputer,
            DiagnosticComputer,
            SamplingOutput
        )
        print("  ✓ All imports successful")
        return True
    except ImportError as e:
        print(f"  ✗ Import failed: {e}")
        return False

def test_model_initialization():
    """Test model initialization"""
    print("\nTesting model initialization...")
    try:
        from optimal_sampling_hf import OptimalSamplingModel

        model = OptimalSamplingModel(
            model_theta_path="Qwen/Qwen2.5-1.5B-Instruct",
            model_t_path="Qwen/Qwen2.5-3B-Instruct",
            alpha_method="fixed",
            fixed_alpha=0.5,
            dtype=torch.bfloat16,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        print("  ✓ Model initialized successfully")
        return True
    except Exception as e:
        print(f"  ✗ Model initialization failed: {e}")
        return False

def test_basic_generation():
    """Test basic text generation"""
    print("\nTesting basic generation...")
    try:
        from optimal_sampling_hf import OptimalSamplingModel

        model = OptimalSamplingModel(
            model_theta_path="Qwen/Qwen2.5-1.5B-Instruct",
            model_t_path="Qwen/Qwen2.5-3B-Instruct",
            alpha_method="fixed",
            fixed_alpha=0.5,
            dtype=torch.bfloat16,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )

        outputs = model.generate(
            prompts=["What is 2+2?"],
            max_new_tokens=20,
            temperature=0.8,
            return_diagnostics=True
        )

        # Validate output
        assert len(outputs.generated_texts) == 1, "Expected 1 generated text"
        assert outputs.generated_ids.shape[0] == 1, "Expected 1 sequence"
        assert outputs.alpha_values.shape[0] == 1, "Expected alpha values for 1 sequence"
        assert outputs.ess_ratios.shape[0] == 1, "Expected ESS ratios for 1 sequence"

        print(f"  Generated: {outputs.generated_texts[0][:50]}...")
        print(f"  Tokens: {outputs.generated_ids.shape[1]}")
        print(f"  Alpha mean: {outputs.alpha_values.mean():.3f}")
        print("  ✓ Basic generation successful")
        return True
    except Exception as e:
        print(f"  ✗ Generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_output_format():
    """Test output format validation"""
    print("\nTesting output format...")
    try:
        from optimal_sampling_hf import OptimalSamplingModel, SamplingOutput

        model = OptimalSamplingModel(
            model_theta_path="Qwen/Qwen2.5-1.5B-Instruct",
            model_t_path="Qwen/Qwen2.5-3B-Instruct",
            alpha_method="kl_symmetry",
            dtype=torch.bfloat16,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )

        outputs = model.generate(
            prompts=["Test prompt"],
            max_new_tokens=10,
            temperature=0.8,
            return_diagnostics=True
        )

        # Check output type
        assert isinstance(outputs, SamplingOutput), "Output should be SamplingOutput"

        # Check attributes
        assert hasattr(outputs, 'generated_texts'), "Missing generated_texts"
        assert hasattr(outputs, 'generated_ids'), "Missing generated_ids"
        assert hasattr(outputs, 'alpha_values'), "Missing alpha_values"
        assert hasattr(outputs, 'ess_ratios'), "Missing ess_ratios"
        assert hasattr(outputs, 'diagnostics'), "Missing diagnostics"

        # Check diagnostics
        assert outputs.diagnostics is not None, "Diagnostics should not be None"
        assert 'kl_theta' in outputs.diagnostics, "Missing kl_theta in diagnostics"
        assert 'kl_t' in outputs.diagnostics, "Missing kl_t in diagnostics"

        print("  ✓ Output format valid")
        return True
    except Exception as e:
        print(f"  ✗ Output format validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("=" * 80)
    print("Running Optimal Sampling HF Tests")
    print("=" * 80)

    # Run tests
    results = []
    results.append(("Imports", test_imports()))

    if results[-1][1]:  # Only continue if imports successful
        results.append(("Model Initialization", test_model_initialization()))

        if results[-1][1]:  # Only continue if initialization successful
            results.append(("Basic Generation", test_basic_generation()))
            results.append(("Output Format", test_output_format()))

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
