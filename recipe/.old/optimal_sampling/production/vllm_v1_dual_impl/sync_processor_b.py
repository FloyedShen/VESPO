"""
Sync LogitsProcessor for Engine B (Theta Model)

Simplified version using BaseSyncLogitsProcessor
"""

import torch
from .sync_processor_base import BaseSyncLogitsProcessor


class SyncLogitsProcessorB(BaseSyncLogitsProcessor):
    """
    LogitsProcessor for Engine B (Theta Model) in dual-engine architecture

    Engine B typically runs the smaller/theta model.
    Inherits all functionality from BaseSyncLogitsProcessor.
    """

    def __init__(
        self,
        vllm_config,
        device: torch.device,
        is_pin_memory: bool
    ):
        """
        Initialize Engine B sync processor

        Args:
            vllm_config: vLLM configuration
            device: torch device
            is_pin_memory: Whether to pin memory
        """
        super().__init__(
            engine_id='B',
            vllm_config=vllm_config,
            device=device,
            is_pin_memory=is_pin_memory
        )
