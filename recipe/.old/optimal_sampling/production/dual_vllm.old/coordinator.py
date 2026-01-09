"""
Dual VLLM Coordinator

Core implementation of the async dual-vllm coordination approach.
"""

import asyncio
import aiohttp
import logging
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
import numpy as np
from tqdm.asyncio import tqdm

# Support both relative and absolute imports
try:
    from .config import CoordinatorConfig
    from .utils import (
        solve_kl_symmetry,
        compute_q_star,
        merge_top_k_candidates,
        sample_from_distribution,
        compute_diagnostics,
    )
except ImportError:
    from config import CoordinatorConfig
    from utils import (
        solve_kl_symmetry,
        compute_q_star,
        merge_top_k_candidates,
        sample_from_distribution,
        compute_diagnostics,
    )


@dataclass
class GenerationOutput:
    """Output from generation"""
    prompt: str
    generated_text: str
    generated_tokens: List[int]
    alpha_history: List[float]
    diagnostics: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class DualVLLMCoordinator:
    """
    Dual VLLM Coordinator for Optimal Sampling

    This class coordinates two independent vLLM instances to generate samples
    from the theoretically optimal distribution q* ∝ π_θ^(1-α*) * π_t^α*,
    where α* is determined by KL symmetry.

    Architecture:
        ┌─────────────┐         ┌─────────────┐
        │  vllm       │         │  vllm       │
        │  (π_θ)      │         │  (π_t)      │
        └──────┬──────┘         └──────┬──────┘
               │                       │
               └───────────┬───────────┘
                           │
                    ┌──────▼──────┐
                    │ Coordinator │
                    │ - Sync      │
                    │ - Compute α*│
                    │ - Sample q* │
                    └─────────────┘

    Usage:
        coordinator = DualVLLMCoordinator(config)

        # Async context manager (recommended)
        async with coordinator:
            results = await coordinator.generate_batch(
                prompts=["Hello, how are you?"],
                max_tokens=100
            )

        # Or manually manage lifecycle
        await coordinator.start()
        results = await coordinator.generate_batch(...)
        await coordinator.close()
    """

    def __init__(self, config: Optional[CoordinatorConfig] = None):
        """
        Initialize coordinator

        Args:
            config: Configuration object (default: CoordinatorConfig())
        """
        self.config = config or CoordinatorConfig()
        self.config.validate()

        # Setup logging
        self.logger = logging.getLogger("DualVLLMCoordinator")
        if self.config.enable_logging:
            logging.basicConfig(
                level=self.config.log_level,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )

        # HTTP session (will be initialized in start())
        self.session: Optional[aiohttp.ClientSession] = None

        # Statistics
        self.stats = {
            "total_requests": 0,
            "failed_requests": 0,
            "total_tokens": 0,
        }

    async def __aenter__(self):
        """Async context manager entry"""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()

    async def start(self):
        """Start the coordinator (initialize HTTP session)"""
        if self.session is None:
            timeout = aiohttp.ClientTimeout(total=self.config.request_timeout)
            connector = aiohttp.TCPConnector(limit=self.config.connection_pool_size)
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                connector=connector
            )
            self.logger.info("Coordinator started")

    async def close(self):
        """Close the coordinator (cleanup resources)"""
        if self.session is not None:
            await self.session.close()
            self.session = None
            self.logger.info("Coordinator closed")

    async def generate_batch(
        self,
        prompts: List[str],
        max_tokens: int = 100,
        temperature: float = 1.0,
        return_diagnostics: bool = False,
        show_progress: bool = True
    ) -> List[GenerationOutput]:
        """
        Generate text for a batch of prompts using optimal sampling

        Args:
            prompts: List of input prompts
            max_tokens: Maximum tokens to generate per prompt
            temperature: Sampling temperature (applied before coordination)
            return_diagnostics: Whether to compute and return diagnostics
            show_progress: Show progress bar

        Returns:
            List of GenerationOutput objects
        """
        if self.session is None:
            raise RuntimeError("Coordinator not started. Use 'async with' or call start()")

        self.logger.info(f"Generating {len(prompts)} prompts with max_tokens={max_tokens}")

        # Generate concurrently with progress bar
        if show_progress:
            tasks = [
                self._generate_one(prompt, max_tokens, temperature, return_diagnostics)
                for prompt in prompts
            ]
            results = await tqdm.gather(*tasks, desc="Generating")
        else:
            results = await asyncio.gather(*[
                self._generate_one(prompt, max_tokens, temperature, return_diagnostics)
                for prompt in prompts
            ])

        # Update statistics
        self.stats["total_requests"] += len(prompts)
        self.stats["total_tokens"] += sum(len(r.generated_tokens) for r in results if r.error is None)
        self.stats["failed_requests"] += sum(1 for r in results if r.error is not None)

        return results

    async def _generate_one(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        return_diagnostics: bool
    ) -> GenerationOutput:
        """
        Generate text for a single prompt

        This is the core generation loop that coordinates the two models
        at each token position.
        """
        context = prompt
        generated_tokens = []
        alpha_history = []
        diagnostics_history = []

        try:
            for step in range(max_tokens):
                # ============================================
                # Step 1: Get logprobs from both models (parallel)
                # ============================================
                logprobs_theta, logprobs_t = await asyncio.gather(
                    self._get_next_token_logprobs(context, self.config.theta_url, temperature),
                    self._get_next_token_logprobs(context, self.config.t_url, temperature),
                )

                # ============================================
                # Step 2: Merge top-k candidates
                # ============================================
                candidates, probs_theta, probs_t = merge_top_k_candidates(
                    logprobs_theta, logprobs_t
                )

                # ============================================
                # Step 3: Solve for α* (KL symmetry)
                # ============================================
                alpha_star = solve_kl_symmetry(
                    probs_theta, probs_t,
                    tol=self.config.alpha_tol,
                    max_iter=self.config.alpha_max_iter
                )

                # ============================================
                # Step 4: Compute q* and sample
                # ============================================
                q_star = compute_q_star(probs_theta, probs_t, alpha_star)
                next_token = sample_from_distribution(q_star, candidates)

                # ============================================
                # Step 5: Compute diagnostics (optional)
                # ============================================
                if return_diagnostics:
                    diag = compute_diagnostics(probs_theta, probs_t, q_star, alpha_star)
                    diagnostics_history.append(diag)

                # ============================================
                # Step 6: Update state
                # ============================================
                generated_tokens.append(next_token)
                alpha_history.append(alpha_star)

                # Decode token and update context
                # Note: This is a simplified version. In production, you'd need
                # to use the actual tokenizer from vLLM
                context = await self._decode_and_append(context, next_token)

                # Check for EOS (simplified - would need actual EOS token ID)
                # if next_token == eos_token_id:
                #     break

            # Aggregate diagnostics
            aggregated_diagnostics = None
            if return_diagnostics and diagnostics_history:
                aggregated_diagnostics = self._aggregate_diagnostics(diagnostics_history)

            return GenerationOutput(
                prompt=prompt,
                generated_text=context,  # Full text including prompt
                generated_tokens=generated_tokens,
                alpha_history=alpha_history,
                diagnostics=aggregated_diagnostics,
                error=None
            )

        except Exception as e:
            self.logger.error(f"Error generating for prompt: {str(e)}")
            return GenerationOutput(
                prompt=prompt,
                generated_text="",
                generated_tokens=[],
                alpha_history=[],
                diagnostics=None,
                error=str(e)
            )

    async def _get_next_token_logprobs(
        self,
        context: str,
        vllm_url: str,
        temperature: float
    ) -> Dict[int, float]:
        """
        Get next token log probabilities from vLLM

        This uses vLLM's completion API with specific parameters to get
        the probability distribution over the next token.

        Args:
            context: Current text context
            vllm_url: URL of the vLLM instance
            temperature: Sampling temperature

        Returns:
            Dict mapping token_id -> log_prob (top-k tokens)
        """
        payload = {
            "prompt": context,
            "max_tokens": 1,  # We only need logprobs for next token
            "temperature": temperature,
            "logprobs": self.config.top_k,  # Request top-k log probabilities
            "echo": False,  # Don't echo the prompt
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
                        raise RuntimeError(f"vLLM API error: {response.status} - {error_text}")

                    data = await response.json()

                    # Extract logprobs from response
                    # vLLM returns logprobs in: data["choices"][0]["logprobs"]["top_logprobs"]
                    # This is a list (one entry per token), we want the first one
                    logprobs = data["choices"][0]["logprobs"]["top_logprobs"][0]

                    return logprobs

            except Exception as e:
                if attempt < self.config.max_retries:
                    self.logger.warning(
                        f"Request failed (attempt {attempt + 1}/{self.config.max_retries + 1}): {e}"
                    )
                    await asyncio.sleep(self.config.retry_delay)
                else:
                    raise RuntimeError(f"Failed after {self.config.max_retries + 1} attempts: {e}")

    async def _decode_and_append(self, context: str, token_id: int) -> str:
        """
        Decode a token and append to context

        Note: This is a placeholder. In production, you'd need to:
        1. Get the tokenizer from vLLM
        2. Use it to decode the token
        3. Handle special tokens properly

        For now, we'll make a request to vLLM to do the decoding.
        """
        # Simple approach: use vLLM to generate with the token_id
        # This is not ideal but works as a placeholder
        # In production, you'd want to cache the tokenizer
        return context + f"[TOKEN_{token_id}]"  # Placeholder

    def _aggregate_diagnostics(self, diagnostics_history: List[Dict]) -> Dict[str, Any]:
        """
        Aggregate diagnostics across all generated tokens

        Returns:
            Dict with aggregated statistics (mean, std, min, max)
        """
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
        """Get coordinator statistics"""
        return self.stats.copy()


# ============================================
# Convenience function
# ============================================

async def generate_with_optimal_sampling(
    prompts: List[str],
    theta_url: str = "http://localhost:8000",
    t_url: str = "http://localhost:8001",
    max_tokens: int = 100,
    top_k: int = 100,
    temperature: float = 1.0,
    return_diagnostics: bool = False,
    show_progress: bool = True
) -> List[GenerationOutput]:
    """
    Convenience function for one-off generation

    Args:
        prompts: List of input prompts
        theta_url: URL for π_θ vLLM instance
        t_url: URL for π_t vLLM instance
        max_tokens: Maximum tokens to generate
        top_k: Number of top tokens to consider
        temperature: Sampling temperature
        return_diagnostics: Whether to return diagnostics
        show_progress: Show progress bar

    Returns:
        List of GenerationOutput objects

    Example:
        results = await generate_with_optimal_sampling(
            prompts=["Hello, how are you?"],
            theta_url="http://localhost:8000",
            t_url="http://localhost:8001",
            max_tokens=50
        )

        for result in results:
            print(result.generated_text)
            print(f"Average α: {np.mean(result.alpha_history):.3f}")
    """
    config = CoordinatorConfig(
        theta_url=theta_url,
        t_url=t_url,
        top_k=top_k
    )

    async with DualVLLMCoordinator(config) as coordinator:
        results = await coordinator.generate_batch(
            prompts=prompts,
            max_tokens=max_tokens,
            temperature=temperature,
            return_diagnostics=return_diagnostics,
            show_progress=show_progress
        )

    return results
