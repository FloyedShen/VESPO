"""
Optimal Sampling V1 - Main Interface for vLLM V1 Engine

High-performance optimal sampling using vLLM V1 LogitsProcessor API.

Architecture:
- Teacher model (π_t, larger model) runs in the outer vLLM engine (benefits from KV cache)
- Theta model (π_θ, smaller model) runs in the inner guide model
- Optimal distribution: q*(y|x) = π_θ(y|x)^(1-α) × π_t(y|x)^α / Z
"""

from typing import List, Optional, Dict
from dataclasses import dataclass
from vllm import LLM, SamplingParams

from .logits_processor_v1 import OptimalSamplingLogitsProcessorV1


@dataclass
class OptimalSamplingOutput:
    """Output from Optimal Sampling"""
    generated_texts: List[str]
    generated_ids: List[List[int]]
    num_tokens: List[int]
    alpha_stats: Optional[Dict[str, float]] = None  # Alpha statistics


class OptimalSamplingV1:
    """
    Optimal Sampling V1 using vLLM V1 Engine

    This implementation uses vLLM V1's LogitsProcessor API for maximum
    performance with automatic KV cache management.

    Key Features:
    - ✅ True KV cache reuse (teacher model benefits from vLLM's KV cache)
    - ✅ vLLM V1 native integration
    - ✅ Batch processing support
    - ✅ Efficient memory usage
    - ✅ Different system prompts for teacher and theta models
    - ✅ Chat template support
    - ✅ Alpha statistics tracking

    Architecture:
    - Teacher model (larger) runs in the outer vLLM engine
    - Theta model (smaller) runs in the inner guide model
    - This maximizes KV cache benefits for the larger model

    Usage:
        sampler = OptimalSamplingV1(
            model_teacher="Qwen/Qwen2.5-1.5B",  # Larger model (outer)
            model_theta="Qwen/Qwen2.5-0.5B",     # Smaller model (inner)
            alpha_method="kl_symmetry",
            teacher_system_prompt="You are a helpful assistant.",
            theta_system_prompt="You are a helpful assistant.",
            enable_chat_template=True
        )

        outputs = sampler.generate(
            prompts=["What is AI?"],
            max_tokens=100,
            temperature=0.8
        )

    Args:
        model_teacher: Teacher model path (π_t, larger model, outer vLLM)
        model_theta: Theta model path (π_θ, smaller model, inner guide)
        alpha_method: Alpha computation method ("kl_symmetry", "ess_balance", "entropy", "fixed")
        fixed_alpha: Fixed alpha value (when alpha_method="fixed")
        alpha_min: Minimum alpha value (teacher weight)
        alpha_max: Maximum alpha value (teacher weight)
        teacher_system_prompt: System prompt for teacher model
        theta_system_prompt: System prompt for theta model
        enable_chat_template: Whether to use chat template
        track_alpha_stats: Whether to track alpha statistics
        gpu_memory_utilization: GPU memory fraction for teacher model
        **kwargs: Additional vLLM arguments
    """

    def __init__(
        self,
        model_teacher: str,
        model_theta: str,
        alpha_method: str = "kl_symmetry",
        fixed_alpha: float = 0.5,
        alpha_min: float = 0.5,
        alpha_max: float = 1.0,
        teacher_system_prompt: Optional[str] = None,
        theta_system_prompt: Optional[str] = None,
        enable_chat_template: bool = False,
        track_alpha_stats: bool = True,
        gpu_memory_utilization: float = 0.5,
        **kwargs
    ):
        self.model_teacher = model_teacher
        self.model_theta = model_theta
        self.alpha_method = alpha_method
        self.fixed_alpha = fixed_alpha
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.teacher_system_prompt = teacher_system_prompt
        self.theta_system_prompt = theta_system_prompt
        self.enable_chat_template = enable_chat_template
        self.track_alpha_stats = track_alpha_stats

        print("=" * 80)
        print("🚀 Initializing Optimal Sampling V1")
        print("=" * 80)
        print(f"Teacher Model (π_t, outer): {model_teacher}")
        print(f"Theta Model (π_θ, inner): {model_theta}")
        print(f"Alpha Method: {alpha_method}")
        print(f"GPU Memory (teacher): {gpu_memory_utilization:.1%}")
        if teacher_system_prompt:
            print(f"Teacher System Prompt: {teacher_system_prompt[:50]}...")
        if theta_system_prompt:
            print(f"Theta System Prompt: {theta_system_prompt[:50]}...")
        print(f"Chat Template: {'Enabled' if enable_chat_template else 'Disabled'}")
        print(f"Alpha Stats Tracking: {'Enabled' if track_alpha_stats else 'Disabled'}")
        print()

        # Initialize teacher model (outer vLLM) with logits processor
        print("📦 Loading teacher model with Optimal Sampling Logits Processor...")
        self.llm = LLM(
            model=model_teacher,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
            enable_prefix_caching=True,
            disable_log_stats=True,
            logits_processors=[OptimalSamplingLogitsProcessorV1],
            **kwargs
        )

        # Get tokenizer for chat template support
        if enable_chat_template:
            self.teacher_tokenizer = self.llm.get_tokenizer()
            print("✅ Loaded tokenizer for chat template support")

        print("\n" + "=" * 80)
        print("✅ Optimal Sampling V1 Initialized!")
        print("=" * 80)
        print()

    def _apply_system_prompt(self, prompt: str, system_prompt: Optional[str]) -> str:
        """Apply system prompt to a user prompt"""
        if system_prompt is None:
            return prompt
        return f"{system_prompt}\n\n{prompt}"

    def _apply_chat_template(
        self,
        prompt: str,
        system_prompt: Optional[str],
        tokenizer
    ) -> str:
        """Apply chat template to format the prompt"""
        messages = []

        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })

        messages.append({
            "role": "user",
            "content": prompt
        })

        # Apply chat template
        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        return formatted

    def _prepare_prompts(self, prompts: List[str]) -> List[str]:
        """Prepare prompts with system prompt and/or chat template for teacher model"""
        prepared = []

        for prompt in prompts:
            if self.enable_chat_template:
                # Use chat template
                formatted = self._apply_chat_template(
                    prompt,
                    self.teacher_system_prompt,
                    self.teacher_tokenizer
                )
                prepared.append(formatted)
            elif self.teacher_system_prompt:
                # Just add system prompt
                formatted = self._apply_system_prompt(prompt, self.teacher_system_prompt)
                prepared.append(formatted)
            else:
                # No modification
                prepared.append(prompt)

        return prepared

    def generate(
        self,
        prompts: List[str],
        max_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        use_optimal_sampling: bool = True
    ) -> OptimalSamplingOutput:
        """
        Generate text using optimal sampling

        Args:
            prompts: List of input prompts (raw text, will be formatted)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            use_optimal_sampling: Whether to use optimal sampling (if False, uses teacher model only)

        Returns:
            OptimalSamplingOutput with generated texts and metadata
        """
        # Prepare prompts for teacher model
        prepared_prompts = self._prepare_prompts(prompts)

        # Build sampling params
        extra_args = {}
        if use_optimal_sampling:
            extra_args = {
                "theta_model_path": self.model_theta,
                "alpha_method": self.alpha_method,
                "fixed_alpha": self.fixed_alpha,
                "alpha_min": self.alpha_min,
                "alpha_max": self.alpha_max,
                "theta_system_prompt": self.theta_system_prompt,
                "enable_chat_template": self.enable_chat_template,
                "track_alpha_stats": self.track_alpha_stats,
                # Pass original prompts for theta model
                "original_prompts": prompts
            }

        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            extra_args=extra_args if extra_args else None
        )

        # Generate
        outputs = self.llm.generate(
            prompts=prepared_prompts,
            sampling_params=sampling_params,
            use_tqdm=False
        )

        # Extract results
        generated_texts = [output.outputs[0].text for output in outputs]
        generated_ids = [output.outputs[0].token_ids for output in outputs]
        num_tokens = [len(ids) for ids in generated_ids]

        # Extract alpha stats if available
        alpha_stats = None
        if self.track_alpha_stats and use_optimal_sampling and hasattr(outputs[0], 'alpha_stats'):
            alpha_stats = outputs[0].alpha_stats

        return OptimalSamplingOutput(
            generated_texts=generated_texts,
            generated_ids=generated_ids,
            num_tokens=num_tokens,
            alpha_stats=alpha_stats
        )

    def __repr__(self):
        return (
            f"OptimalSamplingV1(\n"
            f"  teacher={self.model_teacher},\n"
            f"  theta={self.model_theta},\n"
            f"  alpha_method={self.alpha_method},\n"
            f"  chat_template={'enabled' if self.enable_chat_template else 'disabled'}\n"
            f")"
        )
