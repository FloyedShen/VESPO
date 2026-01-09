"""
Enhanced Dual VLLM Coordinator

Includes advanced features:
1. Dual prompt support (different templates for π_θ and π_t)
2. First token forcing (use π_t for first token)
3. Support constraint (trust region limiting)
4. Special token handling (basic)
"""

import asyncio
import aiohttp
import logging
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
import numpy as np
from tqdm.asyncio import tqdm

# Support both relative and absolute imports
try:
    from .config_enhanced import EnhancedCoordinatorConfig
    from .utils import (
        solve_kl_symmetry,
        compute_q_star,
        merge_top_k_candidates,
        sample_from_distribution,
        compute_diagnostics,
    )
    from .coordinator import GenerationOutput
except ImportError:
    from config_enhanced import EnhancedCoordinatorConfig
    from utils import (
        solve_kl_symmetry,
        compute_q_star,
        merge_top_k_candidates,
        sample_from_distribution,
        compute_diagnostics,
    )
    from coordinator import GenerationOutput


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


class EnhancedDualVLLMCoordinator:
    """
    Enhanced Dual VLLM Coordinator with advanced features

    New capabilities:
    - Dual prompts: π_θ and π_t can see different inputs
    - First token forcing: Force α=1 for first token
    - Support constraint: Limit to π_t's trust region
    - Special token handling: Exclude special tokens from mixing

    Example:
        config = EnhancedCoordinatorConfig(
            theta_url="http://localhost:8000",
            t_url="http://localhost:8001",
            force_first_token=True,
            constraint_to_target=True,
            target_top_p=0.95
        )

        async with EnhancedDualVLLMCoordinator(config) as coordinator:
            # Dual prompt: different templates for base and teacher
            results = await coordinator.generate_batch_dual_prompts(
                prompts_theta=["Question: What is AI?"],
                prompts_t=["<|im_start|>user\nWhat is AI?<|im_end|>"],
                max_tokens=100
            )
    """

    def __init__(self, config: Optional[EnhancedCoordinatorConfig] = None):
        """Initialize enhanced coordinator"""
        self.config = config or EnhancedCoordinatorConfig()
        self.config.validate()

        # Setup logging
        self.logger = logging.getLogger("EnhancedDualVLLMCoordinator")
        if self.config.enable_logging:
            logging.basicConfig(
                level=self.config.log_level,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )

        # HTTP session
        self.session: Optional[aiohttp.ClientSession] = None

        # Statistics
        self.stats = {
            "total_requests": 0,
            "failed_requests": 0,
            "total_tokens": 0,
            "first_token_forced": 0,  # Count of forced first tokens
            "constraint_applied": 0,   # Count of constraint applications
            "stability_fallback": 0,   # Count of stability-induced fallbacks
            "stability_checks": 0,     # Total stability checks performed
        }

    async def __aenter__(self):
        """Async context manager entry"""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()

    async def start(self):
        """Start the coordinator"""
        if self.session is None:
            timeout = aiohttp.ClientTimeout(total=self.config.request_timeout)
            connector = aiohttp.TCPConnector(limit=self.config.connection_pool_size)
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                connector=connector
            )
            self.logger.info("Enhanced Coordinator started")

    async def close(self):
        """Close the coordinator"""
        if self.session is not None:
            await self.session.close()
            self.session = None
            self.logger.info("Enhanced Coordinator closed")

    async def generate_batch(
        self,
        prompts: List[str],
        max_tokens: int = 100,
        temperature: float = 1.0,
        return_diagnostics: bool = False,
        show_progress: bool = True
    ) -> List[GenerationOutput]:
        """
        Generate with same prompts for both models (backward compatible)
        """
        return await self.generate_batch_dual_prompts(
            prompts_theta=prompts,
            prompts_t=prompts,  # Same prompts
            max_tokens=max_tokens,
            temperature=temperature,
            return_diagnostics=return_diagnostics,
            show_progress=show_progress
        )

    async def generate_batch_dual_prompts(
        self,
        prompts_theta: List[str],
        prompts_t: List[str],
        max_tokens: int = 100,
        temperature: float = 1.0,
        return_diagnostics: bool = False,
        show_progress: bool = True
    ) -> List[GenerationOutput]:
        """
        Generate with different prompts for π_θ and π_t

        This is the key feature for supporting different chat templates!

        Args:
            prompts_theta: Prompts for π_θ (base model)
                Example: "Question: What is AI?"
            prompts_t: Prompts for π_t (teacher model)
                Example: "<|im_start|>user\nWhat is AI?<|im_end|>"
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            return_diagnostics: Whether to return detailed diagnostics
            show_progress: Show progress bar

        Returns:
            List of GenerationOutput objects

        Note:
            Both models will generate the SAME tokens (from q*),
            but they see different input contexts.
            This is important for alignment with different training data.
        """
        if self.session is None:
            raise RuntimeError("Coordinator not started")

        if len(prompts_theta) != len(prompts_t):
            raise ValueError(
                f"prompts_theta and prompts_t must have same length: "
                f"{len(prompts_theta)} vs {len(prompts_t)}"
            )

        self.logger.info(
            f"Generating {len(prompts_theta)} prompts with max_tokens={max_tokens}"
        )

        # Generate concurrently
        if show_progress:
            tasks = [
                self._generate_one_dual_prompt(
                    prompt_theta, prompt_t, max_tokens, temperature, return_diagnostics
                )
                for prompt_theta, prompt_t in zip(prompts_theta, prompts_t)
            ]
            results = await tqdm.gather(*tasks, desc="Generating")
        else:
            results = await asyncio.gather(*[
                self._generate_one_dual_prompt(
                    prompt_theta, prompt_t, max_tokens, temperature, return_diagnostics
                )
                for prompt_theta, prompt_t in zip(prompts_theta, prompts_t)
            ])

        # Update statistics
        self.stats["total_requests"] += len(prompts_theta)
        self.stats["total_tokens"] += sum(
            len(r.generated_tokens) for r in results if r.error is None
        )
        self.stats["failed_requests"] += sum(1 for r in results if r.error is not None)

        return results

    async def _generate_one_dual_prompt(
        self,
        prompt_theta: str,
        prompt_t: str,
        max_tokens: int,
        temperature: float,
        return_diagnostics: bool
    ) -> GenerationOutput:
        """
        Generate for a single prompt pair with dual prompts

        Key difference from base coordinator:
        - Maintains separate contexts for π_θ and π_t
        - But samples the SAME token from q* for both
        """
        context_theta = prompt_theta
        context_t = prompt_t

        generated_tokens = []
        alpha_history = []
        diagnostics_history = []

        try:
            for step in range(max_tokens):
                # ============================================
                # Step 1: Get logprobs from both models (parallel)
                # ============================================
                logprobs_theta, logprobs_t = await asyncio.gather(
                    self._get_next_token_logprobs(
                        context_theta, self.config.theta_url, temperature, self.config.theta_model_name
                    ),
                    self._get_next_token_logprobs(
                        context_t, self.config.t_url, temperature, self.config.t_model_name
                    ),
                )

                # ============================================
                # Step 2: Merge top-k candidates
                # ============================================
                candidates, probs_theta, probs_t = merge_top_k_candidates(
                    logprobs_theta, logprobs_t
                )

                # ============================================
                # Step 2.5: Stability detection (if enabled)
                # ============================================
                stability_diag = None
                if self.config.enable_stability_check:
                    self.stats["stability_checks"] += 1

                    # Compute overlap
                    overlap_tokens = set(logprobs_theta.keys()) & set(logprobs_t.keys())
                    overlap_count = len(overlap_tokens)

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
                        js_div < self.config.stability_threshold_js and
                        min(overlap_mass_theta, overlap_mass_t) > self.config.stability_threshold_overlap
                    )

                    # Fallback decision
                    fallback_to_t = self.config.auto_fallback and not is_stable

                    # Store diagnostics
                    stability_diag = {
                        "overlap_count": overlap_count,
                        "overlap_mass_theta": float(overlap_mass_theta),
                        "overlap_mass_t": float(overlap_mass_t),
                        "js_divergence": float(js_div),
                        "is_stable": is_stable,
                        "fallback_to_t": fallback_to_t,
                    }

                    # If fallback, replace π_θ with π_t
                    if fallback_to_t:
                        probs_theta = probs_t.copy()
                        self.stats["stability_fallback"] += 1

                        self.logger.debug(
                            f"Step {step}: Unstable distribution detected! "
                            f"JS={js_div:.3f}, Overlap={overlap_mass_theta:.3f}. "
                            f"Falling back to π_t"
                        )

                # ============================================
                # Step 3: Apply support constraint (if enabled)
                # ============================================
                if self.config.constraint_to_target:
                    candidates, probs_theta, probs_t = self._apply_support_constraint(
                        candidates, probs_theta, probs_t
                    )
                    self.stats["constraint_applied"] += 1

                # ============================================
                # Step 4: Compute α and q*
                # ============================================
                if step == 0 and self.config.force_first_token:
                    # ✨ Force first token to use π_t (α = 1)
                    alpha_star = 1.0
                    q_star = probs_t  # Directly use π_t
                    self.stats["first_token_forced"] += 1

                    self.logger.debug(
                        f"First token forced: using π_t directly (α=1.0)"
                    )
                elif stability_diag and stability_diag["fallback_to_t"]:
                    # ✨ Stability-induced fallback (α = 1)
                    alpha_star = 1.0
                    q_star = probs_t  # Fallback to π_t

                    self.logger.debug(
                        f"Step {step}: Stability fallback (α=1.0)"
                    )
                else:
                    # Normal optimal sampling
                    alpha_star = solve_kl_symmetry(
                        probs_theta, probs_t,
                        tol=self.config.alpha_tol,
                        max_iter=self.config.alpha_max_iter
                    )
                    q_star = compute_q_star(probs_theta, probs_t, alpha_star)

                # ============================================
                # Step 5: Sample next token
                # ============================================
                next_token = sample_from_distribution(q_star, candidates)

                # ============================================
                # Step 6: Compute diagnostics (optional)
                # ============================================
                if return_diagnostics:
                    diag = compute_diagnostics(probs_theta, probs_t, q_star, alpha_star)
                    diagnostics_history.append(diag)

                # ============================================
                # Step 7: Update states
                # ============================================
                generated_tokens.append(next_token)
                alpha_history.append(alpha_star)

                # Update BOTH contexts with the sampled token string
                # next_token is already a token string from vLLM API (e.g., "!", ",", "Hello")
                context_theta += next_token
                context_t += next_token

                # TODO: Proper EOS detection
                # if next_token == "<|endoftext|>":
                #     break

            # Aggregate diagnostics
            aggregated_diagnostics = None
            if return_diagnostics and diagnostics_history:
                aggregated_diagnostics = self._aggregate_diagnostics(diagnostics_history)

            return GenerationOutput(
                prompt=prompt_theta,  # Return θ's prompt as primary
                generated_text=context_theta,
                generated_tokens=generated_tokens,
                alpha_history=alpha_history,
                diagnostics=aggregated_diagnostics,
                error=None
            )

        except Exception as e:
            self.logger.error(f"Error generating: {str(e)}")
            return GenerationOutput(
                prompt=prompt_theta,
                generated_text="",
                generated_tokens=[],
                alpha_history=[],
                diagnostics=None,
                error=str(e)
            )

    def _apply_support_constraint(
        self,
        candidates: List[int],
        probs_theta: np.ndarray,
        probs_t: np.ndarray
    ) -> Tuple[List[int], np.ndarray, np.ndarray]:
        """
        Apply support constraint: limit to π_t's trust region

        Strategy: Keep only tokens in π_t's top-p cumulative probability

        Args:
            candidates: List of candidate token IDs
            probs_theta: Probabilities for π_θ [len(candidates)]
            probs_t: Probabilities for π_t [len(candidates)]

        Returns:
            (filtered_candidates, filtered_probs_theta, filtered_probs_t)

        Theory:
            By limiting to π_t's top-p, we ensure:
            1. We don't sample tokens π_t considers very unlikely
            2. Numerical stability (avoid extreme importance weights)
            3. Better alignment with teacher's distribution
        """
        # Sort by π_t's probabilities (descending)
        sorted_indices = np.argsort(-probs_t)

        # Compute cumulative probability
        cumsum = np.cumsum(probs_t[sorted_indices])

        # Find cutoff: keep tokens until cumsum exceeds target_top_p
        n_keep = np.searchsorted(cumsum, self.config.target_top_p) + 1
        n_keep = min(n_keep, len(candidates))  # Ensure we keep at least one

        # Keep top-p tokens
        keep_indices = sorted_indices[:n_keep]

        # Filter
        filtered_candidates = [candidates[i] for i in keep_indices]
        filtered_probs_theta = probs_theta[keep_indices]
        filtered_probs_t = probs_t[keep_indices]

        # Renormalize
        filtered_probs_theta = filtered_probs_theta / filtered_probs_theta.sum()
        filtered_probs_t = filtered_probs_t / filtered_probs_t.sum()

        self.logger.debug(
            f"Support constraint: {len(candidates)} -> {len(filtered_candidates)} "
            f"(kept {n_keep}/{len(candidates)})"
        )

        return filtered_candidates, filtered_probs_theta, filtered_probs_t

    async def _get_next_token_logprobs(
        self,
        context: str,
        vllm_url: str,
        temperature: float,
        model_name: str
    ) -> Dict[str, float]:
        """Get next token log probabilities from vLLM OpenAI API

        Args:
            context: The context/prompt to use
            vllm_url: The vLLM server URL
            temperature: Sampling temperature
            model_name: The model name (e.g., "Qwen/Qwen3-4B-Base")

        Returns:
            Dict mapping token_string -> logprob (using strings as keys)
        """
        payload = {
            "model": model_name,  # Use actual model name instead of "dummy"
            "prompt": context,
            "max_tokens": 1,
            "temperature": temperature,
            "logprobs": self.config.top_k,
            "echo": False,
        }

        # Retry logic
        for attempt in range(self.config.max_retries + 1):
            try:
                async with self.session.post(
                    f"{vllm_url}/v1/completions",
                    json=payload
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise RuntimeError(
                            f"vLLM API error: {response.status} - {error_text}"
                        )

                    data = await response.json()

                    # OpenAI API format: top_logprobs is a dict with token strings as keys
                    logprobs_data = data["choices"][0]["logprobs"]["top_logprobs"][0]

                    # Return directly as {token_string: logprob}
                    # Both models use the same tokenizer (Qwen), so strings should match
                    return logprobs_data

            except Exception as e:
                if attempt < self.config.max_retries:
                    self.logger.warning(
                        f"Request failed (attempt {attempt + 1}/"
                        f"{self.config.max_retries + 1}): {e}"
                    )
                    await asyncio.sleep(self.config.retry_delay)
                else:
                    raise RuntimeError(
                        f"Failed after {self.config.max_retries + 1} attempts: {e}"
                    )

    def _aggregate_diagnostics(self, diagnostics_history: List[Dict]) -> Dict[str, Any]:
        """Aggregate diagnostics across all generated tokens"""
        keys = diagnostics_history[0].keys()
        aggregated = {}

        for key in keys:
            values = [d[key] for d in diagnostics_history]
            aggregated[f"{key}_mean"] = np.mean(values)
            aggregated[f"{key}_std"] = np.std(values)
            aggregated[f"{key}_min"] = np.min(values)
            aggregated[f"{key}_max"] = np.max(values)

        return aggregated

    def get_statistics(self) -> Dict[str, Any]:
        """Get enhanced coordinator statistics"""
        stats = self.stats.copy()

        # Add some computed stats
        if stats["total_requests"] > 0:
            stats["success_rate"] = 1.0 - (
                stats["failed_requests"] / stats["total_requests"]
            )

        return stats


# ============================================
# Convenience function
# ============================================

async def generate_with_dual_prompts(
    prompts_theta: List[str],
    prompts_t: List[str],
    theta_url: str = "http://localhost:8000",
    t_url: str = "http://localhost:8001",
    max_tokens: int = 100,
    top_k: int = 100,
    temperature: float = 1.0,
    force_first_token: bool = True,
    constraint_to_target: bool = False,
    target_top_p: float = 0.95,
    return_diagnostics: bool = False,
    show_progress: bool = True
) -> List[GenerationOutput]:
    """
    Convenience function for dual prompt generation

    Example:
        # Base model sees simple format
        prompts_theta = ["Question: What is AI?"]

        # Teacher sees chat template format
        prompts_t = ["<|im_start|>user\nWhat is AI?<|im_end|>\n<|im_start|>assistant"]

        results = await generate_with_dual_prompts(
            prompts_theta=prompts_theta,
            prompts_t=prompts_t,
            force_first_token=True,
            constraint_to_target=True
        )
    """
    config = EnhancedCoordinatorConfig(
        theta_url=theta_url,
        t_url=t_url,
        top_k=top_k,
        force_first_token=force_first_token,
        constraint_to_target=constraint_to_target,
        target_top_p=target_top_p,
    )

    async with EnhancedDualVLLMCoordinator(config) as coordinator:
        results = await coordinator.generate_batch_dual_prompts(
            prompts_theta=prompts_theta,
            prompts_t=prompts_t,
            max_tokens=max_tokens,
            temperature=temperature,
            return_diagnostics=return_diagnostics,
            show_progress=show_progress
        )

    return results
