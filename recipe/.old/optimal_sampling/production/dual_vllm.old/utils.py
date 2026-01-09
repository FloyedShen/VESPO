"""
Utility functions for optimal sampling computation
"""

import numpy as np
from typing import Dict, Tuple, List


def solve_kl_symmetry(
    probs_theta: np.ndarray,
    probs_t: np.ndarray,
    tol: float = 1e-6,
    max_iter: int = 20
) -> float:
    """
    Solve KL symmetry condition via binary search

    Goal: Find α* such that D_KL(q_α* || π_θ) = D_KL(q_α* || π_t)
    Equivalent to: E_q[log(π_t/π_θ)] = 0

    Args:
        probs_theta: Probability distribution from π_θ [n_candidates]
        probs_t: Probability distribution from π_t [n_candidates]
        tol: Convergence tolerance
        max_iter: Maximum iterations

    Returns:
        alpha_star: Optimal α ∈ [0, 1]

    Theoretical basis:
        See theory/proof_final.md Section 6.4 (Theorem 5)
    """
    # Input validation
    assert len(probs_theta) == len(probs_t), "Probability arrays must have same length"
    assert np.allclose(probs_theta.sum(), 1.0, atol=1e-4), "probs_theta must sum to 1"
    assert np.allclose(probs_t.sum(), 1.0, atol=1e-4), "probs_t must sum to 1"

    # Handle edge case: distributions are identical
    if np.allclose(probs_theta, probs_t, atol=1e-6):
        return 0.5

    # Precompute log ratio
    eps = 1e-10
    log_ratio = np.log(probs_t + eps) - np.log(probs_theta + eps)

    # Binary search
    alpha_low, alpha_high = 0.0, 1.0

    for iteration in range(max_iter):
        alpha_mid = (alpha_low + alpha_high) / 2

        # Compute q_α in log space (numerical stability)
        log_q = ((1 - alpha_mid) * np.log(probs_theta + eps) +
                 alpha_mid * np.log(probs_t + eps))

        # Normalize
        log_q = log_q - np.max(log_q)  # Subtract max for stability
        q_alpha = np.exp(log_q)
        q_alpha = q_alpha / q_alpha.sum()

        # Compute Δ(α) = E_q[log(π_t/π_θ)]
        delta = (q_alpha * log_ratio).sum()

        # Update interval
        # When delta > 0: q is too close to π_t, need to decrease α
        # When delta < 0: q is too close to π_θ, need to increase α
        if delta > 0:
            alpha_high = alpha_mid
        else:
            alpha_low = alpha_mid

        # Check convergence
        if alpha_high - alpha_low < tol:
            break

    alpha_star = (alpha_low + alpha_high) / 2

    # Sanity check
    if alpha_star < 0.0 or alpha_star > 1.0:
        print(f"Warning: alpha_star={alpha_star:.6f} out of bounds, clipping")
        alpha_star = np.clip(alpha_star, 0.0, 1.0)

    return alpha_star


def compute_q_star(
    probs_theta: np.ndarray,
    probs_t: np.ndarray,
    alpha: float
) -> np.ndarray:
    """
    Compute optimal sampling distribution q*

    q*(y) ∝ π_θ(y)^(1-α) * π_t(y)^α

    Args:
        probs_theta: Probability distribution from π_θ
        probs_t: Probability distribution from π_t
        alpha: Mixing parameter (α=0 → π_θ, α=1 → π_t)

    Returns:
        q_star: Optimal sampling distribution (normalized)
    """
    eps = 1e-10

    # Compute in log space for numerical stability
    log_q = ((1 - alpha) * np.log(probs_theta + eps) +
             alpha * np.log(probs_t + eps))

    # Normalize via softmax
    log_q = log_q - np.max(log_q)
    q_star = np.exp(log_q)
    q_star = q_star / q_star.sum()

    return q_star


def merge_top_k_candidates(
    logprobs_theta: Dict[str, float],
    logprobs_t: Dict[str, float]
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """
    Merge top-k candidates from two models

    Args:
        logprobs_theta: {token_string: log_prob} from π_θ
        logprobs_t: {token_string: log_prob} from π_t

    Returns:
        candidates: List of candidate token strings (sorted)
        probs_theta: Normalized probabilities for π_θ [len(candidates)]
        probs_t: Normalized probabilities for π_t [len(candidates)]

    Theoretical note:
        By merging top-k from both models, we cover the probability mass where
        either model has high confidence. This ensures the approximation error
        is bounded by O((1 - p_covered) * log(V)) where p_covered ≈ 0.95-0.99.
    """
    # Merge candidate sets
    candidates = sorted(set(logprobs_theta.keys()) | set(logprobs_t.keys()))

    # For tokens not in top-k, assign very small log prob
    missing_logprob = -100.0  # exp(-100) ≈ 3.7e-44

    # Build probability vectors
    logprobs_theta_full = np.array([
        logprobs_theta.get(tid, missing_logprob)
        for tid in candidates
    ])
    logprobs_t_full = np.array([
        logprobs_t.get(tid, missing_logprob)
        for tid in candidates
    ])

    # Convert to probabilities and renormalize
    probs_theta = np.exp(logprobs_theta_full)
    probs_t = np.exp(logprobs_t_full)

    probs_theta = probs_theta / probs_theta.sum()
    probs_t = probs_t / probs_t.sum()

    return candidates, probs_theta, probs_t


def sample_from_distribution(probs: np.ndarray, candidates: List[str]) -> str:
    """
    Sample a token from a probability distribution

    Args:
        probs: Probability distribution [n_candidates]
        candidates: List of candidate token strings

    Returns:
        sampled_token_string: The sampled token string
    """
    assert len(probs) == len(candidates), "Mismatched lengths"
    assert np.allclose(probs.sum(), 1.0, atol=1e-4), "Probabilities must sum to 1"

    # Sample
    idx = np.random.choice(len(candidates), p=probs)
    return candidates[idx]


def compute_diagnostics(
    probs_theta: np.ndarray,
    probs_t: np.ndarray,
    q_star: np.ndarray,
    alpha: float
) -> Dict[str, float]:
    """
    Compute diagnostic metrics

    Returns:
        Dict with keys:
            - alpha: The α value used
            - kl_theta: D_KL(q* || π_θ)
            - kl_t: D_KL(q* || π_t)
            - kl_diff: |D_KL(q* || π_θ) - D_KL(q* || π_t)|
            - ess_theta: Effective sample size w.r.t. π_θ
            - ess_t: Effective sample size w.r.t. π_t
            - ess_ratio: ess_theta / ess_t
            - entropy_q: H(q*)
    """
    eps = 1e-10

    # KL divergences
    kl_theta = (q_star * (np.log(q_star + eps) - np.log(probs_theta + eps))).sum()
    kl_t = (q_star * (np.log(q_star + eps) - np.log(probs_t + eps))).sum()
    kl_diff = abs(kl_theta - kl_t)

    # ESS (Effective Sample Size)
    # ESS = 1 / Σ(π²/q)
    ess_theta = 1.0 / ((probs_theta ** 2) / (q_star + eps)).sum()
    ess_t = 1.0 / ((probs_t ** 2) / (q_star + eps)).sum()
    ess_ratio = ess_theta / (ess_t + eps)

    # Entropy
    entropy_q = -(q_star * np.log(q_star + eps)).sum()

    return {
        "alpha": alpha,
        "kl_theta": kl_theta,
        "kl_t": kl_t,
        "kl_diff": kl_diff,
        "ess_theta": ess_theta,
        "ess_t": ess_t,
        "ess_ratio": ess_ratio,
        "entropy_q": entropy_q,
    }
