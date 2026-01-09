"""
Unit tests for dual_vllm module

Run: pytest test_dual_vllm.py -v
"""

import pytest
import numpy as np
from unittest.mock import Mock, AsyncMock, patch

from utils import (
    solve_kl_symmetry,
    compute_q_star,
    merge_top_k_candidates,
    sample_from_distribution,
    compute_diagnostics,
)


class TestKLSymmetry:
    """Test KL symmetry solver"""

    def test_identical_distributions(self):
        """When π_θ = π_t, α* should be 0.5"""
        probs = np.array([0.5, 0.3, 0.2])
        alpha = solve_kl_symmetry(probs, probs)
        assert abs(alpha - 0.5) < 1e-6

    def test_symmetry_property(self):
        """Verify that α* achieves KL symmetry"""
        np.random.seed(42)
        probs_theta = np.random.dirichlet([1.0] * 10)
        probs_t = np.random.dirichlet([1.0] * 10)

        alpha = solve_kl_symmetry(probs_theta, probs_t, tol=1e-8)
        q_star = compute_q_star(probs_theta, probs_t, alpha)

        # Compute KL divergences
        eps = 1e-10
        kl_theta = (q_star * (np.log(q_star + eps) - np.log(probs_theta + eps))).sum()
        kl_t = (q_star * (np.log(q_star + eps) - np.log(probs_t + eps))).sum()

        # Should be approximately equal
        assert abs(kl_theta - kl_t) < 1e-5

    def test_extreme_case_theta_peak(self):
        """When π_θ is very peaked, α should be close to 0"""
        probs_theta = np.array([0.99, 0.005, 0.005])
        probs_t = np.array([0.33, 0.33, 0.34])

        alpha = solve_kl_symmetry(probs_theta, probs_t)
        # α should be close to 0 (favor base model)
        assert alpha < 0.3

    def test_extreme_case_t_peak(self):
        """When π_t is very peaked, α should be close to 1"""
        probs_theta = np.array([0.33, 0.33, 0.34])
        probs_t = np.array([0.99, 0.005, 0.005])

        alpha = solve_kl_symmetry(probs_theta, probs_t)
        # α should be close to 1 (favor teacher)
        assert alpha > 0.7

    def test_convergence(self):
        """Test that binary search converges"""
        np.random.seed(123)
        probs_theta = np.random.dirichlet([1.0] * 50)
        probs_t = np.random.dirichlet([1.0] * 50)

        alpha = solve_kl_symmetry(probs_theta, probs_t, tol=1e-8, max_iter=30)

        # Should be in valid range
        assert 0.0 <= alpha <= 1.0

    def test_input_validation(self):
        """Test input validation"""
        probs_theta = np.array([0.5, 0.5])
        probs_t = np.array([0.3, 0.3, 0.4])  # Different length

        with pytest.raises(AssertionError):
            solve_kl_symmetry(probs_theta, probs_t)


class TestQStarComputation:
    """Test q* computation"""

    def test_q_star_normalization(self):
        """q* should be a valid probability distribution"""
        probs_theta = np.array([0.5, 0.3, 0.2])
        probs_t = np.array([0.2, 0.3, 0.5])
        alpha = 0.5

        q_star = compute_q_star(probs_theta, probs_t, alpha)

        # Should sum to 1
        assert abs(q_star.sum() - 1.0) < 1e-6

        # Should be non-negative
        assert (q_star >= 0).all()

    def test_alpha_zero(self):
        """When α=0, q* should equal π_θ"""
        probs_theta = np.array([0.5, 0.3, 0.2])
        probs_t = np.array([0.2, 0.3, 0.5])

        q_star = compute_q_star(probs_theta, probs_t, alpha=0.0)

        assert np.allclose(q_star, probs_theta, atol=1e-6)

    def test_alpha_one(self):
        """When α=1, q* should equal π_t"""
        probs_theta = np.array([0.5, 0.3, 0.2])
        probs_t = np.array([0.2, 0.3, 0.5])

        q_star = compute_q_star(probs_theta, probs_t, alpha=1.0)

        assert np.allclose(q_star, probs_t, atol=1e-6)

    def test_geometric_mean_property(self):
        """q* should be geometric mean in log space"""
        probs_theta = np.array([0.5, 0.3, 0.2])
        probs_t = np.array([0.2, 0.3, 0.5])
        alpha = 0.6

        q_star = compute_q_star(probs_theta, probs_t, alpha)

        # In log space: log q* = (1-α) log π_θ + α log π_t
        eps = 1e-10
        expected_log_q = ((1 - alpha) * np.log(probs_theta + eps) +
                          alpha * np.log(probs_t + eps))
        expected_q = np.exp(expected_log_q)
        expected_q = expected_q / expected_q.sum()

        assert np.allclose(q_star, expected_q, atol=1e-6)


class TestTopKMerge:
    """Test top-k merging"""

    def test_merge_disjoint(self):
        """Test merging disjoint top-k sets"""
        logprobs_theta = {0: -1.0, 1: -2.0, 2: -3.0}
        logprobs_t = {3: -1.0, 4: -2.0, 5: -3.0}

        candidates, probs_theta, probs_t = merge_top_k_candidates(
            logprobs_theta, logprobs_t
        )

        # Should have 6 candidates
        assert len(candidates) == 6
        assert set(candidates) == {0, 1, 2, 3, 4, 5}

        # Probabilities should sum to 1
        assert abs(probs_theta.sum() - 1.0) < 1e-6
        assert abs(probs_t.sum() - 1.0) < 1e-6

    def test_merge_overlapping(self):
        """Test merging overlapping top-k sets"""
        logprobs_theta = {0: -1.0, 1: -2.0, 2: -3.0}
        logprobs_t = {1: -1.5, 2: -2.5, 3: -3.5}

        candidates, probs_theta, probs_t = merge_top_k_candidates(
            logprobs_theta, logprobs_t
        )

        # Should have 4 unique candidates
        assert len(candidates) == 4
        assert set(candidates) == {0, 1, 2, 3}

    def test_missing_token_handling(self):
        """Test that missing tokens get very small probability"""
        logprobs_theta = {0: -1.0, 1: -2.0}
        logprobs_t = {2: -1.0, 3: -2.0}

        candidates, probs_theta, probs_t = merge_top_k_candidates(
            logprobs_theta, logprobs_t
        )

        # Token 2 should have very small prob in probs_theta
        idx_2 = candidates.index(2)
        assert probs_theta[idx_2] < 1e-20


class TestSampling:
    """Test sampling function"""

    def test_sampling_distribution(self):
        """Test that sampling follows the distribution"""
        np.random.seed(42)
        probs = np.array([0.5, 0.3, 0.2])
        candidates = [10, 20, 30]

        # Sample many times
        samples = [sample_from_distribution(probs, candidates) for _ in range(10000)]

        # Check empirical distribution
        counts = {10: 0, 20: 0, 30: 0}
        for s in samples:
            counts[s] += 1

        empirical = np.array([counts[10], counts[20], counts[30]]) / 10000

        # Should be close to true probabilities
        assert np.allclose(empirical, probs, atol=0.02)


class TestDiagnostics:
    """Test diagnostic computation"""

    def test_diagnostics_keys(self):
        """Test that all expected keys are present"""
        probs_theta = np.array([0.5, 0.3, 0.2])
        probs_t = np.array([0.2, 0.3, 0.5])
        q_star = np.array([0.35, 0.3, 0.35])
        alpha = 0.5

        diag = compute_diagnostics(probs_theta, probs_t, q_star, alpha)

        expected_keys = {
            "alpha", "kl_theta", "kl_t", "kl_diff",
            "ess_theta", "ess_t", "ess_ratio", "entropy_q"
        }
        assert set(diag.keys()) == expected_keys

    def test_ess_ratio_balance(self):
        """When KL symmetric, ESS ratio should be close to 1"""
        np.random.seed(42)
        probs_theta = np.random.dirichlet([1.0] * 10)
        probs_t = np.random.dirichlet([1.0] * 10)

        alpha = solve_kl_symmetry(probs_theta, probs_t)
        q_star = compute_q_star(probs_theta, probs_t, alpha)

        diag = compute_diagnostics(probs_theta, probs_t, q_star, alpha)

        # ESS ratio should be close to 1.0
        assert abs(diag["ess_ratio"] - 1.0) < 0.2


class TestCoordinatorConfig:
    """Test configuration"""

    def test_default_config(self):
        """Test default configuration values"""
        from config import CoordinatorConfig

        config = CoordinatorConfig()
        assert config.top_k == 100
        assert config.alpha_tol == 1e-6
        assert config.theta_url == "http://localhost:8000"

    def test_config_validation(self):
        """Test configuration validation"""
        from config import CoordinatorConfig

        # Valid config
        config = CoordinatorConfig(top_k=50)
        config.validate()  # Should not raise

        # Invalid config
        config = CoordinatorConfig(top_k=-1)
        with pytest.raises(AssertionError):
            config.validate()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
