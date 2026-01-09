"""
Optimal Sampling V1 Implementation for vLLM V1 Engine

High-performance optimal sampling using vLLM V1 LogitsProcessor API.

Architecture:
- Teacher model (π_t, larger model) runs in the outer vLLM engine (benefits from KV cache)
- Theta model (π_θ, smaller model) runs in the inner guide model
- Optimal distribution: q*(y|x) = π_θ(y|x)^(1-α) × π_t(y|x)^α / Z

Features:
- ✅ True KV cache reuse for teacher model
- ✅ Different system prompts for each model
- ✅ Chat template support
- ✅ Alpha statistics tracking
"""

from .optimal_sampling_v1 import OptimalSamplingV1, OptimalSamplingOutput
from .logits_processor_v1 import OptimalSamplingLogitsProcessorV1
from .guide_model_v1 import ThetaModelV1
from .alpha_computer import AlphaComputer

__all__ = [
    "OptimalSamplingV1",
    "OptimalSamplingOutput",
    "OptimalSamplingLogitsProcessorV1",
    "ThetaModelV1",
    "AlphaComputer",
]
