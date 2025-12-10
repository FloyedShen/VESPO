"""
Optimal Sampling HF: HuggingFace Transformers + Flash Attention Implementation

High-performance optimal sampling using HuggingFace Transformers with Flash Attention 2.
6.4x faster than vLLM V1 for data generation tasks.

Basic Usage:
    from optimal_sampling_hf import OptimalSamplingModel

    model = OptimalSamplingModel(
        model_theta_path="Qwen/Qwen2.5-1.5B-Instruct",
        model_t_path="Qwen/Qwen2.5-3B-Instruct",
        alpha_method="kl_symmetry"
    )

    outputs = model.generate(
        prompts=["What is 2+2?"],
        max_new_tokens=100,
        temperature=0.8
    )

    print(outputs.generated_texts[0])
"""

from .optimal_sampling import (
    OptimalSamplingModel,
    AlphaComputer,
    DiagnosticComputer,
    SamplingOutput,
    create_dual_prompts,
    NATURAL_LANGUAGE_TEMPLATE,
)

__version__ = "1.0.0"

__all__ = [
    "OptimalSamplingModel",
    "AlphaComputer",
    "DiagnosticComputer",
    "SamplingOutput",
    "create_dual_prompts",
    "NATURAL_LANGUAGE_TEMPLATE",
]
