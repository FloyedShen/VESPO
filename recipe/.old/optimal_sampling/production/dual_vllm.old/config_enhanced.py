"""
Enhanced Configuration for Dual VLLM Coordinator

Includes advanced features from optimal_sampling_model.py:
- Support constraint (trust region limiting)
- First token forcing
- Dual prompt support
- Special token handling (partial)
"""

from dataclasses import dataclass
from typing import Optional
from config import CoordinatorConfig


@dataclass
class EnhancedCoordinatorConfig(CoordinatorConfig):
    """
    Enhanced configuration with advanced features

    New features compared to base CoordinatorConfig:

    1. Support Constraint (Trust Region):
       - constraint_to_target: Limit sampling to π_t's support
       - target_top_p: Use top-p of π_t as trust region
       - Ensures we don't sample tokens that π_t considers unlikely

    2. First Token Forcing:
       - force_first_token: Force first token to use π_t directly
       - Helps with better initial direction (based on theory)

    3. Dual Prompt Support:
       - Allows different prompts for π_θ and π_t
       - Useful when using different chat templates
       - Example: Base model sees "Question: ...", Teacher sees "<|im_start|>..."

    4. Model Names:
       - theta_model_name: Model name for π_θ (used in OpenAI API requests)
       - t_model_name: Model name for π_t (used in OpenAI API requests)
    """

    # Model names for OpenAI API
    theta_model_name: str = "Qwen/Qwen3-4B-Base"
    """Model name for π_θ (base model) - used in OpenAI API requests"""

    t_model_name: str = "Qwen/Qwen3-14B"
    """Model name for π_t (teacher model) - used in OpenAI API requests"""

    # Support constraint / Trust region
    constraint_to_target: bool = False
    """Whether to constraint sampling to π_t's support (trust region)"""

    target_top_p: float = 0.95
    """Top-p threshold for π_t's trust region (only used if constraint_to_target=True)"""

    # First token forcing
    force_first_token: bool = True
    """Force first token to be sampled from π_t directly (α=1 for first token)"""

    # Special token handling (basic support)
    exclude_special_tokens: bool = False
    """Exclude special tokens from mixing (use π_t directly for them)"""

    special_token_ids: Optional[list] = None
    """List of special token IDs to exclude (if known)"""

    # Stability detection (NEW)
    enable_stability_check: bool = False
    """Whether to enable stability detection (overlap + JS divergence)"""

    stability_threshold_js: float = 0.5
    """JS divergence threshold (0-0.693). If JS > threshold, distributions are too different"""

    stability_threshold_overlap: float = 0.1
    """Overlap probability mass threshold (0-1). If overlap < threshold, distributions have little common support"""

    auto_fallback: bool = True
    """Whether to automatically fallback to π_t when unstable"""

    def validate(self):
        """Validate enhanced configuration"""
        super().validate()  # Call base validation

        assert 0.0 <= self.target_top_p <= 1.0, "target_top_p must be in [0, 1]"
        assert 0.0 <= self.stability_threshold_js <= 0.693, "stability_threshold_js must be in [0, ln(2)]"
        assert 0.0 <= self.stability_threshold_overlap <= 1.0, "stability_threshold_overlap must be in [0, 1]"

        if self.constraint_to_target and self.target_top_p == 1.0:
            # Warning: constraint enabled but threshold is 1.0 (no effect)
            pass
