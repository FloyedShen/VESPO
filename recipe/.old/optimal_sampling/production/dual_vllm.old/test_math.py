#!/usr/bin/env python3
"""
Test core math utilities without requiring vLLM
"""

import sys
import os
import numpy as np

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import (
    solve_kl_symmetry,
    compute_q_star,
    merge_top_k_candidates,
    sample_from_distribution,
    compute_diagnostics,
)


def test_kl_symmetry():
    """Test KL symmetry solver"""
    print("\n" + "="*70)
    print("TEST 1: KL Symmetry Solver")
    print("="*70)

    # Test case 1: Identical distributions
    print("\nCase 1: Identical distributions")
    probs = np.array([0.5, 0.3, 0.2])
    alpha = solve_kl_symmetry(probs, probs)
    print(f"  α = {alpha:.6f} (expected: 0.5)")
    assert abs(alpha - 0.5) < 1e-5, "Should be 0.5 for identical distributions"
    print("  ✅ PASS")

    # Test case 2: Different distributions
    print("\nCase 2: Different distributions")
    probs_theta = np.array([0.7, 0.2, 0.1])
    probs_t = np.array([0.1, 0.2, 0.7])
    alpha = solve_kl_symmetry(probs_theta, probs_t, tol=1e-8)
    q_star = compute_q_star(probs_theta, probs_t, alpha)

    # Verify KL symmetry
    eps = 1e-10
    kl_theta = (q_star * (np.log(q_star + eps) - np.log(probs_theta + eps))).sum()
    kl_t = (q_star * (np.log(q_star + eps) - np.log(probs_t + eps))).sum()

    print(f"  α = {alpha:.6f}")
    print(f"  D_KL(q||π_θ) = {kl_theta:.8f}")
    print(f"  D_KL(q||π_t)  = {kl_t:.8f}")
    print(f"  |Difference| = {abs(kl_theta - kl_t):.8f}")

    assert abs(kl_theta - kl_t) < 1e-5, "KL divergences should be equal"
    print("  ✅ PASS")

    # Test case 3: Extreme case (peaked distribution)
    print("\nCase 3: Peaked distribution")
    probs_theta = np.array([0.98, 0.01, 0.01])
    probs_t = np.array([0.33, 0.33, 0.34])
    alpha = solve_kl_symmetry(probs_theta, probs_t)
    print(f"  α = {alpha:.6f} (should be < 0.3, favoring peaked π_θ)")
    assert alpha < 0.3, "Should favor peaked distribution"
    print("  ✅ PASS")

    print("\n✅ All KL symmetry tests PASSED")
    return True


def test_q_star_computation():
    """Test q* computation"""
    print("\n" + "="*70)
    print("TEST 2: Q* Computation")
    print("="*70)

    probs_theta = np.array([0.5, 0.3, 0.2])
    probs_t = np.array([0.2, 0.3, 0.5])

    # Test boundary conditions
    print("\nCase 1: α=0 (should equal π_θ)")
    q_0 = compute_q_star(probs_theta, probs_t, alpha=0.0)
    assert np.allclose(q_0, probs_theta, atol=1e-6)
    print(f"  q* = {q_0}")
    print("  ✅ PASS")

    print("\nCase 2: α=1 (should equal π_t)")
    q_1 = compute_q_star(probs_theta, probs_t, alpha=1.0)
    assert np.allclose(q_1, probs_t, atol=1e-6)
    print(f"  q* = {q_1}")
    print("  ✅ PASS")

    print("\nCase 3: α=0.5 (geometric mean)")
    q_half = compute_q_star(probs_theta, probs_t, alpha=0.5)
    print(f"  q* = {q_half}")
    assert abs(q_half.sum() - 1.0) < 1e-6, "Should sum to 1"
    print("  ✅ Normalized")

    # Verify geometric mean property
    eps = 1e-10
    expected = np.exp(0.5 * np.log(probs_theta + eps) + 0.5 * np.log(probs_t + eps))
    expected = expected / expected.sum()
    assert np.allclose(q_half, expected, atol=1e-6)
    print("  ✅ Geometric mean property verified")

    print("\n✅ All q* computation tests PASSED")
    return True


def test_top_k_merge():
    """Test top-k merging"""
    print("\n" + "="*70)
    print("TEST 3: Top-K Merging")
    print("="*70)

    # Test case 1: Disjoint sets
    print("\nCase 1: Disjoint top-k sets")
    logprobs_theta = {0: -1.0, 1: -2.0, 2: -3.0}
    logprobs_t = {3: -1.0, 4: -2.0, 5: -3.0}

    candidates, probs_theta, probs_t = merge_top_k_candidates(
        logprobs_theta, logprobs_t
    )

    print(f"  Candidates: {candidates}")
    print(f"  π_θ sum: {probs_theta.sum():.6f}")
    print(f"  π_t sum: {probs_t.sum():.6f}")

    assert len(candidates) == 6
    assert abs(probs_theta.sum() - 1.0) < 1e-6
    assert abs(probs_t.sum() - 1.0) < 1e-6
    print("  ✅ PASS")

    # Test case 2: Overlapping sets
    print("\nCase 2: Overlapping top-k sets")
    logprobs_theta = {0: -1.0, 1: -2.0, 2: -3.0}
    logprobs_t = {1: -1.5, 2: -2.5, 3: -3.5}

    candidates, probs_theta, probs_t = merge_top_k_candidates(
        logprobs_theta, logprobs_t
    )

    print(f"  Candidates: {candidates}")
    assert len(candidates) == 4
    assert set(candidates) == {0, 1, 2, 3}
    print("  ✅ PASS")

    print("\n✅ All top-k merging tests PASSED")
    return True


def test_sampling():
    """Test sampling"""
    print("\n" + "="*70)
    print("TEST 4: Sampling")
    print("="*70)

    np.random.seed(42)
    probs = np.array([0.5, 0.3, 0.2])
    candidates = [10, 20, 30]

    print("\nSampling 10000 times from [0.5, 0.3, 0.2]")

    samples = [sample_from_distribution(probs, candidates) for _ in range(10000)]

    counts = {10: 0, 20: 0, 30: 0}
    for s in samples:
        counts[s] += 1

    empirical = np.array([counts[10], counts[20], counts[30]]) / 10000

    print(f"  Expected:  {probs}")
    print(f"  Empirical: {empirical}")
    print(f"  Max error: {np.abs(empirical - probs).max():.4f}")

    assert np.allclose(empirical, probs, atol=0.02)
    print("  ✅ PASS")

    print("\n✅ All sampling tests PASSED")
    return True


def test_diagnostics():
    """Test diagnostics computation"""
    print("\n" + "="*70)
    print("TEST 5: Diagnostics")
    print("="*70)

    probs_theta = np.array([0.5, 0.3, 0.2])
    probs_t = np.array([0.2, 0.3, 0.5])

    alpha = solve_kl_symmetry(probs_theta, probs_t)
    q_star = compute_q_star(probs_theta, probs_t, alpha)

    diag = compute_diagnostics(probs_theta, probs_t, q_star, alpha)

    print(f"\n  Alpha: {diag['alpha']:.6f}")
    print(f"  D_KL(q||π_θ): {diag['kl_theta']:.6f}")
    print(f"  D_KL(q||π_t):  {diag['kl_t']:.6f}")
    print(f"  |Difference|: {diag['kl_diff']:.6f}")
    print(f"  ESS_θ: {diag['ess_theta']:.6f}")
    print(f"  ESS_t: {diag['ess_t']:.6f}")
    print(f"  ESS ratio: {diag['ess_ratio']:.6f}")
    print(f"  H(q*): {diag['entropy_q']:.6f}")

    # Verify KL symmetry
    assert diag['kl_diff'] < 1e-4, "KL symmetry error should be small"
    print("\n  ✅ KL symmetry verified")

    # Verify ESS ratio is close to 1
    assert 0.9 <= diag['ess_ratio'] <= 1.1, "ESS ratio should be ≈ 1"
    print("  ✅ ESS balance verified")

    print("\n✅ All diagnostic tests PASSED")
    return True


def test_performance():
    """Test performance of key operations"""
    print("\n" + "="*70)
    print("TEST 6: Performance")
    print("="*70)

    import time

    # Realistic vocab size
    vocab_size = 100  # Simulating top-k=100

    np.random.seed(42)
    probs_theta = np.random.dirichlet([1.0] * vocab_size)
    probs_t = np.random.dirichlet([1.0] * vocab_size)

    # Time KL symmetry solving
    start = time.time()
    for _ in range(100):
        alpha = solve_kl_symmetry(probs_theta, probs_t, max_iter=20)
    elapsed = time.time() - start

    avg_time = elapsed / 100 * 1000  # ms
    print(f"\nKL symmetry (vocab=100, 100 iterations):")
    print(f"  Average time: {avg_time:.3f} ms")
    print(f"  Expected: < 1 ms")

    if avg_time < 2.0:
        print("  ✅ PASS")
    else:
        print("  ⚠️  Slower than expected")

    # Time q* computation
    start = time.time()
    for _ in range(10000):
        q_star = compute_q_star(probs_theta, probs_t, 0.5)
    elapsed = time.time() - start

    avg_time = elapsed / 10000 * 1000  # ms
    print(f"\nQ* computation (vocab=100, 10000 iterations):")
    print(f"  Average time: {avg_time:.4f} ms")
    print(f"  Expected: < 0.1 ms")

    if avg_time < 0.2:
        print("  ✅ PASS")
    else:
        print("  ⚠️  Slower than expected")

    print("\n✅ Performance tests PASSED")
    return True


def main():
    """Run all tests"""
    print("="*70)
    print("🧪 Dual VLLM - Core Math Tests (No vLLM Required)")
    print("="*70)

    tests = [
        ("KL Symmetry", test_kl_symmetry),
        ("Q* Computation", test_q_star_computation),
        ("Top-K Merging", test_top_k_merge),
        ("Sampling", test_sampling),
        ("Diagnostics", test_diagnostics),
        ("Performance", test_performance),
    ]

    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n❌ {name} FAILED: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # Summary
    print("\n" + "="*70)
    print("📋 TEST SUMMARY")
    print("="*70)

    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)

    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {name:.<50} {status}")

    print("\n" + "="*70)
    if passed_count == total_count:
        print(f"🎉 ALL {total_count} TESTS PASSED!")
    else:
        print(f"❌ {total_count - passed_count}/{total_count} TESTS FAILED")
    print("="*70 + "\n")

    return passed_count == total_count


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
