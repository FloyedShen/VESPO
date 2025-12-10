"""
Sync LogitsProcessor for Engine A (Teacher Model)

Simplified version using BaseSyncLogitsProcessor
"""

import torch
from .sync_processor_base import BaseSyncLogitsProcessor


class SyncLogitsProcessorA(BaseSyncLogitsProcessor):
    """
    LogitsProcessor for Engine A (Teacher Model) in dual-engine architecture

    Engine A typically runs the larger/teacher model.
    Inherits all functionality from BaseSyncLogitsProcessor.
    """

    def __init__(
        self,
        vllm_config,
        device: torch.device,
        is_pin_memory: bool
    ):
        """
        Initialize Engine A sync processor

        Args:
            vllm_config: vLLM configuration
            device: torch device
            is_pin_memory: Whether to pin memory
        """
        super().__init__(
            engine_id='A',
            vllm_config=vllm_config,
            device=device,
            is_pin_memory=is_pin_memory
        )
