"""
SGLang Implementation of Optimal Sampling

Clean and efficient implementation using SGLang's native control flow.

Key advantages over vLLM:
- 60% less code
- Native support for complex sampling
- Easier to debug and maintain
- No dependency on logits processors

Usage:
    from production.sglang_impl import OptimalSamplingSGLang

    sampler = OptimalSamplingSGLang(
        model_base="Qwen/Qwen2.5-0.5B",
        model_teacher="Qwen/Qwen2.5-1.5B",
        alpha_method="kl_symmetry"
    )

    outputs = sampler.generate(
        prompts=["What is AI?"],
        max_tokens=100
    )
"""

from .optimal_sampling_sglang import OptimalSamplingSGLang
from .alpha_computer import AlphaComputer

__all__ = [
    "OptimalSamplingSGLang",
    "AlphaComputer",
]
