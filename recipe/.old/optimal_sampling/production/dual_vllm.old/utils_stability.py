"""
Enhanced utility functions with stability detection

Adds:
1. Overlap ratio detection
2. JS divergence detection
3. Auto-fallback to π_t when unstable
"""

import numpy as np
from typing import Dict, Tuple, List


def compute_js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """
    Compute Jensen-Shannon Divergence

    JS(P||Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M)
    where M = 0.5 * (P + Q)

    Range: [0, ln(2)] ≈ [0, 0.693]
    - 0: identical distributions
    - ln(2): completely different distributions
    """
    eps = 1e-10
    m = 0.5 * (p + q)

    kl_pm = (p * (np.log(p + eps) - np.log(m + eps))).sum()
    kl_qm = (q * (np.log(q + eps) - np.log(m + eps))).sum()

    return 0.5 * kl_pm + 0.5 * kl_qm


def merge_top_k_candidates_with_stability(
    logprobs_theta: Dict[str, float],
    logprobs_t: Dict[str, float],
    stability_threshold_js: float = 0.5,
    stability_threshold_overlap: float = 0.1,
    auto_fallback: bool = True
) -> Tuple[List[str], np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Merge top-k candidates with stability detection

    Args:
        logprobs_theta: {token_string: log_prob} from π_θ
        logprobs_t: {token_string: log_prob} from π_t
        stability_threshold_js: JS divergence threshold (default: 0.5)
            - If JS > threshold, distributions are too different
        stability_threshold_overlap: Overlap probability mass threshold (default: 0.1)
            - If overlap < threshold, distributions have little common support
        auto_fallback: If True, automatically fall back to π_t when unstable

    Returns:
        candidates: List of candidate token strings
        probs_theta: Normalized probabilities for π_θ
        probs_t: Normalized probabilities for π_t
        diagnostics: Dict with stability metrics
            - overlap_count: Number of overlapping tokens
            - overlap_mass_theta: Probability mass of overlap in π_θ
            - overlap_mass_t: Probability mass of overlap in π_t
            - js_divergence: JS divergence between distributions
            - is_stable: Whether distributions are stable for mixing
            - fallback_to_t: Whether we should fall back to π_t
    """
    # Compute overlap
    overlap_tokens = set(logprobs_theta.keys()) & set(logprobs_t.keys())
    overlap_count = len(overlap_tokens)

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

    # Compute overlap mass
    overlap_mass_theta = sum(
        probs_theta[i] for i, tok in enumerate(candidates)
        if tok in overlap_tokens
    )
    overlap_mass_t = sum(
        probs_t[i] for i, tok in enumerate(candidates)
        if tok in overlap_tokens
    )

    # Compute JS divergence
    js_div = compute_js_divergence(probs_theta, probs_t)

    # Stability check
    is_stable = (
        js_div < stability_threshold_js and
        min(overlap_mass_theta, overlap_mass_t) > stability_threshold_overlap
    )

    # Fallback decision
    fallback_to_t = auto_fallback and not is_stable

    # If fallback, replace π_θ with π_t
    if fallback_to_t:
        probs_theta = probs_t.copy()

    # Diagnostics
    diagnostics = {
        "overlap_count": overlap_count,
        "overlap_mass_theta": float(overlap_mass_theta),
        "overlap_mass_t": float(overlap_mass_t),
        "js_divergence": float(js_div),
        "is_stable": is_stable,
        "fallback_to_t": fallback_to_t,
    }

    return candidates, probs_theta, probs_t, diagnostics


def solve_kl_symmetry_with_fallback(
    probs_theta: np.ndarray,
    probs_t: np.ndarray,
    tol: float = 1e-6,
    max_iter: int = 20,
    stability_diagnostics: Dict[str, float] = None
) -> Tuple[float, bool]:
    """
    Solve KL symmetry with fallback detection

    Args:
        probs_theta: Probability distribution from π_θ
        probs_t: Probability distribution from π_t
        tol: Convergence tolerance
        max_iter: Maximum iterations
        stability_diagnostics: Optional diagnostics from merge function

    Returns:
        alpha_star: Optimal α (or 1.0 if fallback)
        did_fallback: Whether we fell back to α=1
    """
    # Check if we should fallback based on diagnostics
    did_fallback = False
    if stability_diagnostics and stability_diagnostics.get("fallback_to_t", False):
        # Fall back to π_t (α = 1.0)
        did_fallback = True
        return 1.0, did_fallback

    # Handle edge case: distributions are identical
    if np.allclose(probs_theta, probs_t, atol=1e-6):
        return 0.5, did_fallback

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
        log_q = log_q - np.max(log_q)
        q_alpha = np.exp(log_q)
        q_alpha = q_alpha / q_alpha.sum()

        # Compute Δ(α) = E_q[log(π_t/π_θ)]
        delta = (q_alpha * log_ratio).sum()

        # Update interval
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
        alpha_star = np.clip(alpha_star, 0.0, 1.0)

    return alpha_star, did_fallback
