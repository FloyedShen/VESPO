"""
Configuration for Dual VLLM Coordinator
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CoordinatorConfig:
    """
    Configuration for DualVLLMCoordinator

    Attributes:
        theta_url: URL for π_θ (base model) vLLM instance
        t_url: URL for π_t (teacher model) vLLM instance
        top_k: Number of top tokens to consider (default: 100)
            Theoretical error bound: O((1-coverage) * log(V))
            With k=100, coverage ≈ 0.95-0.99, error < 1%

        alpha_tol: Tolerance for KL symmetry binary search
        alpha_max_iter: Max iterations for binary search

        max_retries: Max retries for failed requests
        retry_delay: Delay between retries (seconds)
        request_timeout: Timeout for HTTP requests (seconds)

        connection_pool_size: Size of HTTP connection pool
        enable_logging: Enable detailed logging
    """

    # vLLM endpoints
    theta_url: str = "http://localhost:8000"
    t_url: str = "http://localhost:8001"

    # Top-k approximation
    top_k: int = 100

    # Alpha computation
    alpha_tol: float = 1e-6
    alpha_max_iter: int = 20

    # Network settings
    max_retries: int = 3
    retry_delay: float = 1.0
    request_timeout: float = 60.0
    connection_pool_size: int = 100

    # Logging
    enable_logging: bool = True
    log_level: str = "INFO"

    def validate(self):
        """Validate configuration"""
        assert self.top_k > 0, "top_k must be positive"
        assert self.alpha_tol > 0, "alpha_tol must be positive"
        assert self.alpha_max_iter > 0, "alpha_max_iter must be positive"
        assert self.max_retries >= 0, "max_retries must be non-negative"
