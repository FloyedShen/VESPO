"""
Dual-Engine Optimal Sampling for vLLM V1

High-performance optimal sampling using two synchronized vLLM engines.
Achieves O(n) complexity instead of O(n²).

Architecture:
- Engine A (Teacher, typically larger model) runs independently with full KV cache
- Engine B (Theta, typically smaller model) runs independently with full KV cache
- Both engines synchronize at LogitsProcessor layer to exchange logits
- No recomputation needed (eliminates O(n²) overhead)

Performance Benefits:
- Original architecture: Theta model has O(n²) complexity (recomputes full sequence each token)
- Dual-engine: Both models have O(n) complexity
- Expected speedup: 3-8x for long sequences (2K-8K tokens)

Key Features:
- ✅ O(n) complexity for both models (vs O(n²) for inner model)
- ✅ Full KV cache benefits for both models
- ✅ Per-request synchronization (handles batch size mismatch)
- ✅ Timeout protection (prevents deadlocks)
- ✅ Graceful degradation on failures
- ✅ Comprehensive statistics tracking
- ✅ Unified base class eliminates code duplication

Example:
    from production.vllm_v1_dual_impl import OptimalSamplingDual

    sampler = OptimalSamplingDual(
        model_teacher="Qwen/Qwen2.5-7B",
        model_theta="Qwen/Qwen2.5-1.5B",
        alpha_method="kl_symmetry",
        sync_timeout=5.0
    )

    outputs = sampler.generate(
        prompts=["What is artificial intelligence?"],
        max_tokens=2048
    )

    print(outputs.generated_texts[0])
    print(f"Alpha stats: {outputs.alpha_stats}")
    print(f"Sync stats: {sampler.get_sync_statistics()}")
"""

from .optimal_sampling_dual import OptimalSamplingDual, OptimalSamplingOutput
from .sync_state_v2 import DualEngineSyncStateV2
from .sync_processor_base import BaseSyncLogitsProcessor
from .sync_processor_a import SyncLogitsProcessorA
from .sync_processor_b import SyncLogitsProcessorB
from .alpha_computer import AlphaComputer

__all__ = [
    "OptimalSamplingDual",
    "OptimalSamplingOutput",
    "DualEngineSyncStateV2",
    "BaseSyncLogitsProcessor",
    "SyncLogitsProcessorA",
    "SyncLogitsProcessorB",
    "AlphaComputer",
]

__version__ = "2.1.0"  # Updated: Added BaseSyncLogitsProcessor
