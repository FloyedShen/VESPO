"""
Dual-Engine Optimal Sampling - Main Interface

Implements optimal sampling with two independent vLLM engines that synchronize
at the LogitsProcessor layer. Achieves O(n) complexity instead of O(n²).
"""

import uuid
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
from vllm import LLM, SamplingParams

from .sync_state_v2 import DualEngineSyncStateV2
from .alpha_computer import AlphaComputer
from .sync_processor_a import SyncLogitsProcessorA
from .sync_processor_b import SyncLogitsProcessorB
from . import proxy_storage

logger = logging.getLogger(__name__)


@dataclass
class OptimalSamplingOutput:
    """Output from dual-engine optimal sampling"""
    generated_texts: List[str]
    generated_ids: List[List[int]]
    num_tokens: List[int]
    alpha_stats: Optional[Dict[str, float]] = None


class OptimalSamplingDual:
    """
    Dual-Engine Optimal Sampling with O(n) Complexity

    Architecture:
    - Two independent vLLM engines (A and B)
    - Both maintain their own KV cache (O(n) per token)
    - Synchronize ONLY at LogitsProcessor layer
    - No recomputation (eliminates O(n²) overhead)

    Performance:
    - Current architecture: O(n²) for Theta model (recomputes entire sequence)
    - Dual-engine: O(n) for both models
    - Expected speedup: 3-8x for long sequences (2K-8K tokens)

    Usage:
        sampler = OptimalSamplingDual(
            model_teacher="Qwen/Qwen2.5-7B",
            model_theta="Qwen/Qwen2.5-1.5B",
            alpha_method="kl_symmetry",
            sync_timeout=5.0
        )

        outputs = sampler.generate(
            prompts=["What is AI?", "Explain ML."],
            max_tokens=2048
        )
    """

    def __init__(
        self,
        model_teacher: str,
        model_theta: str,
        alpha_method: str = "kl_symmetry",
        fixed_alpha: float = 0.5,
        alpha_min: float = 0.5,
        alpha_max: float = 1.0,
        sync_timeout: float = 5.0,
        gpu_memory_teacher: float = 0.4,
        gpu_memory_theta: float = 0.4,
        enable_prefix_caching: bool = True,
        enable_optimal_sampling: bool = True,
        **kwargs
    ):
        """
        Initialize dual-engine optimal sampling

        Args:
            model_teacher: Path to teacher model (typically larger)
            model_theta: Path to theta model (typically smaller)
            alpha_method: Alpha computation method (fixed, kl_symmetry, ess_balance, entropy)
            fixed_alpha: Fixed alpha value (when method="fixed")
            alpha_min: Minimum alpha (teacher weight)
            alpha_max: Maximum alpha (teacher weight)
            sync_timeout: Maximum sync wait time (seconds)
            gpu_memory_teacher: GPU memory utilization for teacher
            gpu_memory_theta: GPU memory utilization for theta
            enable_prefix_caching: Enable vLLM prefix caching
            enable_optimal_sampling: Enable optimal sampling (False = use teacher only)
            **kwargs: Additional vLLM arguments
        """
        self.model_teacher = model_teacher
        self.model_theta = model_theta
        self.enable_optimal_sampling = enable_optimal_sampling

        # Generate unique session ID for this dual-engine instance
        self.session_id = str(uuid.uuid4())

        print("=" * 80)
        print("🚀 Initializing Dual-Engine Optimal Sampling")
        print("=" * 80)
        print(f"Teacher Model: {model_teacher}")
        print(f"Theta Model: {model_theta}")
        print(f"Alpha Method: {alpha_method}")
        print(f"Sync Timeout: {sync_timeout}s")
        print(f"Optimal Sampling: {'Enabled' if enable_optimal_sampling else 'Disabled'}")
        print(f"Session ID: {self.session_id[:8]}...")
        print()

        # Initialize shared synchronization state (V2 with multiprocessing.Manager)
        self.sync_state = DualEngineSyncStateV2.create(
            timeout=sync_timeout,
            enable_logging=False  # Disable verbose logging
        )

        # Get proxy for cross-process sharing
        sync_proxy = self.sync_state.get_proxy()

        # Initialize alpha computer
        self.alpha_computer = AlphaComputer(
            method=alpha_method,
            fixed_alpha=fixed_alpha,
            alpha_min=alpha_min,
            alpha_max=alpha_max
        )

        print(f"   - Alpha method: {alpha_method}")
        print(f"   - Alpha range: [{alpha_min:.2f}, {alpha_max:.2f}]")

        # Register session in proxy's registry_dict
        registry_dict = sync_proxy['registry_dict']
        registry_dict[self.session_id] = {
            'alpha_computer': self.alpha_computer,
            'enable_optimal_sampling': enable_optimal_sampling
        }

        # Save proxy to file for subprocess access
        proxy_file = proxy_storage.save_proxy_to_file(sync_proxy)
        proxy_storage.set_proxy_file_env(proxy_file)

        # Set session ID in environment variable for subprocesses
        proxy_storage.set_session_id_env(self.session_id)

        # Initialize Engine A (Teacher) with SyncLogitsProcessorA
        print("Initializing Engine A (Teacher)...")
        self.engine_a = LLM(
            model=model_teacher,
            gpu_memory_utilization=gpu_memory_teacher,
            enable_prefix_caching=enable_prefix_caching,
            disable_log_stats=True,
            logits_processors=[SyncLogitsProcessorA],  # Pass class, not instance
            **kwargs
        )

        # Initialize Engine B (Theta) with SyncLogitsProcessorB
        print("Initializing Engine B (Theta)...")
        self.engine_b = LLM(
            model=model_theta,
            gpu_memory_utilization=gpu_memory_theta,
            enable_prefix_caching=enable_prefix_caching,
            disable_log_stats=True,
            logits_processors=[SyncLogitsProcessorB],  # Pass class, not instance
            **kwargs
        )

        print("=" * 80)
        print("✅ Dual-Engine Initialization Complete")
        print("=" * 80)
        print()

    def generate(
        self,
        prompts: List[str],
        max_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        **kwargs
    ) -> OptimalSamplingOutput:
        """
        Generate text with dual-engine optimal sampling

        Args:
            prompts: List of input prompts
            max_tokens: Maximum tokens to generate per prompt
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            **kwargs: Additional sampling parameters

        Returns:
            OptimalSamplingOutput with generated texts and statistics
        """
        # Generate unique request IDs for synchronization
        request_ids = [str(uuid.uuid4()) for _ in prompts]

        # Create sampling parameters with request IDs
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            extra_args={'request_ids': request_ids},  # For ID alignment
            **kwargs
        )

        logger.info(f"Starting generation for {len(prompts)} prompts")

        # Launch both engines in parallel
        def run_engine_a():
            logger.info(f"[Engine A] Starting generation...")
            return self.engine_a.generate(
                prompts=prompts,
                sampling_params=sampling_params
            )

        def run_engine_b():
            logger.info(f"[Engine B] Starting generation...")
            return self.engine_b.generate(
                prompts=prompts,
                sampling_params=sampling_params
            )

        # Execute in parallel with thread pool
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_a = executor.submit(run_engine_a)
            future_b = executor.submit(run_engine_b)

            # Wait for both to complete
            # We primarily use Engine A's output (teacher)
            outputs_a = future_a.result()
            outputs_b = future_b.result()  # Ensure B completes too

        logger.info(f"Generation complete")

        # Extract results from Engine A
        generated_texts = [output.outputs[0].text for output in outputs_a]
        generated_ids = [output.outputs[0].token_ids for output in outputs_a]
        num_tokens = [len(ids) for ids in generated_ids]

        # Collect alpha statistics
        # Note: Alpha stats are tracked in LogitsProcessor instances
        # which are created internally by vLLM. For now, we don't have
        # direct access to them. This can be improved by adding class-level
        # alpha history tracking if needed.
        alpha_stats = None

        # Print sync statistics
        sync_stats = self.sync_state.get_statistics()
        logger.info(f"Sync statistics: {sync_stats}")

        return OptimalSamplingOutput(
            generated_texts=generated_texts,
            generated_ids=generated_ids,
            num_tokens=num_tokens,
            alpha_stats=alpha_stats
        )

    def get_sync_statistics(self) -> Dict:
        """
        Get synchronization statistics

        Returns:
            Dictionary with sync metrics:
            - sync_count: Total successful syncs
            - timeout_count: Number of timeouts
            - avg_wait_time: Average wait time per sync
            - timeout_rate: Timeout percentage
        """
        return self.sync_state.get_statistics()

    def reset_statistics(self):
        """Reset all statistics counters"""
        self.sync_state.reset_statistics()

    def __repr__(self):
        return (
            f"OptimalSamplingDual("
            f"teacher={self.model_teacher}, "
            f"theta={self.model_theta}, "
            f"optimal_sampling={self.enable_optimal_sampling})"
        )
