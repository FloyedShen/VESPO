"""
Dual VLLM Coordinator for Optimal Sampling

This module implements the dual-vllm coordination approach for optimal sampling
in RLHF data generation. It uses two independent vLLM instances and coordinates
their outputs to sample from the theoretically optimal distribution q*.

Key features:
- Top-k approximation with theoretical error bounds
- Async coordination for low latency
- Batch generation support
- Full compatibility with vLLM's API

Usage:
    from dual_vllm import DualVLLMCoordinator

    coordinator = DualVLLMCoordinator(
        theta_url="http://localhost:8000",
        t_url="http://localhost:8001"
    )

    results = await coordinator.generate_batch(
        prompts=["Hello, how are you?"],
        max_tokens=100
    )
"""

from .coordinator import DualVLLMCoordinator
from .config import CoordinatorConfig
from .utils import solve_kl_symmetry, compute_q_star

__version__ = "0.1.0"
__all__ = [
    "DualVLLMCoordinator",
    "CoordinatorConfig",
    "solve_kl_symmetry",
    "compute_q_star",
]
