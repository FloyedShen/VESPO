#!/usr/bin/env python3
"""
Tests for different alpha computation methods

Tests:
1. Fixed alpha
2. KL symmetry
3. Reverse KL symmetry
4. ESS balance
5. Entropy-based
"""

import sys
import torch
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_alpha_method(method_name, **kwargs):
    """Test a specific alpha method"""
    try:
        from optimal_sampling_hf import OptimalSamplingModel

        print(f"\n  Testing {method_name}...")
        model = OptimalSamplingModel(
            model_theta_path="Qwen/Qwen2.5-1.5B-Instruct",
            model_t_path="Qwen/Qwen2.5-3B-Instruct",
            alpha_method=method_name,
            dtype=torch.bfloat16,
            device="cuda" if torch.cuda.is_available() else "cpu",
            **kwargs
        )

        outputs = model.generate(
            prompts=["What is 2+2?"],
            max_new_tokens=10,
            temperature=0.8,
            return_diagnostics=True
        )

        # Validate alpha values
        alpha_mean = outputs.alpha_values.mean().item()
        alpha_min = outputs.alpha_values.min().item()
        alpha_max = outputs.alpha_values.max().item()

        print(f"    Alpha range: [{alpha_min:.3f}, {alpha_max:.3f}], mean: {alpha_mean:.3f}")

        # Basic sanity checks
        assert 0.0 <= alpha_min <= 1.0, f"Alpha min out of range: {alpha_min}"
        assert 0.0 <= alpha_max <= 1.0, f"Alpha max out of range: {alpha_max}"
        assert alpha_min <= alpha_mean <= alpha_max, "Alpha mean not in range"

        # Check diagnostics
        assert outputs.diagnostics is not None, "Diagnostics missing"
        kl_theta = outputs.diagnostics['kl_theta'].mean().item()
        kl_t = outputs.diagnostics['kl_t'].mean().item()
        print(f"    KL divergence: θ={kl_theta:.4f}, t={kl_t:.4f}")

        print(f"    ✓ {method_name} passed")
        return True

    except Exception as e:
        print(f"    ✗ {method_name} failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("=" * 80)
    print("Testing Alpha Computation Methods")
    print("=" * 80)

    # Test configurations
    tests = [
        ("fixed", {"fixed_alpha": 0.5}),
        ("fixed", {"fixed_alpha": 0.3}),
        ("fixed", {"fixed_alpha": 0.7}),
        ("kl_symmetry", {}),
        ("reverse_kl_symmetry", {}),
        ("ess_balance", {}),
        ("entropy", {}),
    ]

    results = []
    for method, kwargs in tests:
        config_name = f"{method} (alpha={kwargs.get('fixed_alpha', 'N/A')})" if method == "fixed" else method
        passed = test_alpha_method(method, **kwargs)
        results.append((config_name, passed))

    # Print summary
    print("\n" + "=" * 80)
    print("Alpha Methods Test Summary")
    print("=" * 80)
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {test_name}")

    total_tests = len(results)
    passed_tests = sum(1 for _, passed in results if passed)
    print(f"\nTotal: {passed_tests}/{total_tests} tests passed")

    if passed_tests == total_tests:
        print("\n✅ All alpha methods work correctly!")
        return 0
    else:
        print(f"\n❌ {total_tests - passed_tests} test(s) failed")
        return 1

if __name__ == "__main__":
    exit(main())
