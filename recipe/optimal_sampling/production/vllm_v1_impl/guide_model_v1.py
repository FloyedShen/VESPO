"""
Theta Model V1 for vLLM V1 Engine

Manages a separate vLLM instance for the theta model (π_θ, smaller model).
Designed to work with vLLM V1 LogitsProcessor API.

Architecture:
- Theta model (π_θ) runs as inner guide model
- Supports different system prompts from teacher model
- Supports chat template formatting
"""

import torch
from typing import List, Optional, Dict
from vllm import LLM, SamplingParams


class ThetaModelV1:
    """
    Theta Model (π_θ) for vLLM V1 Optimal Sampling

    Maintains a separate vLLM instance for the theta (smaller) model and provides
    logits on-demand for the logits processor.

    Key features:
    - Separate vLLM instance for theta model
    - Prefix caching for KV reuse
    - Different system prompt support
    - Chat template support
    - Efficient batch processing

    Usage:
        theta_model = ThetaModelV1(
            model_path="Qwen/Qwen2.5-0.5B",
            vllm_config=vllm_config,
            device=device,
            system_prompt="You are a helpful assistant.",
            enable_chat_template=True
        )

        logits = theta_model.get_logits_for_requests(
            request_data={
                0: {"token_ids": [...], "original_prompt": "What is AI?"},
                1: {"token_ids": [...], "original_prompt": "Explain ML."}
            },
            temperature=0.8
        )
    """

    def __init__(
        self,
        model_path: str,
        vllm_config,
        device: torch.device,
        system_prompt: Optional[str] = None,
        enable_chat_template: bool = False,
        gpu_memory_utilization: float = 0.4,
        enable_prefix_caching: bool = True,
    ):
        """
        Initialize Theta Model

        Args:
            model_path: Path to theta model (π_θ, smaller model)
            vllm_config: vLLM configuration object
            device: torch device
            system_prompt: System prompt for theta model (can differ from teacher)
            enable_chat_template: Whether to use chat template
            gpu_memory_utilization: GPU memory fraction for theta model
            enable_prefix_caching: Enable KV cache reuse
        """
        self.model_path = model_path
        self.device = device
        self.system_prompt = system_prompt
        self.enable_chat_template = enable_chat_template

        print(f"🔧 Initializing Theta Model V1 (π_θ): {model_path}")
        print(f"   - GPU memory: {gpu_memory_utilization:.1%}")
        print(f"   - Prefix caching: {enable_prefix_caching}")
        if system_prompt:
            print(f"   - System prompt: {system_prompt[:50]}...")
        print(f"   - Chat template: {'Enabled' if enable_chat_template else 'Disabled'}")

        # Initialize theta model LLM instance
        self.llm = LLM(
            model=model_path,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
            enable_prefix_caching=enable_prefix_caching,
            disable_log_stats=True,
        )

        # Get vocab size
        self.vocab_size = self.llm.llm_engine.model_config.hf_config.vocab_size

        # Get tokenizer for chat template support
        if enable_chat_template:
            self.tokenizer = self.llm.get_tokenizer()
            print(f"   - Loaded tokenizer for chat template")

        print(f"✅ Theta model initialized (vocab_size={self.vocab_size})")

    def _apply_system_prompt(self, prompt: str) -> str:
        """Apply system prompt to a user prompt"""
        if self.system_prompt is None:
            return prompt
        return f"{self.system_prompt}\n\n{prompt}"

    def _apply_chat_template(self, prompt: str) -> str:
        """Apply chat template to format the prompt"""
        messages = []

        if self.system_prompt:
            messages.append({
                "role": "system",
                "content": self.system_prompt
            })

        messages.append({
            "role": "user",
            "content": prompt
        })

        # Apply chat template
        formatted = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        return formatted

    def _prepare_prompt(self, original_prompt: str) -> str:
        """Prepare prompt with system prompt and/or chat template"""
        if self.enable_chat_template:
            return self._apply_chat_template(original_prompt)
        elif self.system_prompt:
            return self._apply_system_prompt(original_prompt)
        else:
            return original_prompt

    def get_logits_for_requests(
        self,
        request_data: Dict[int, Dict[str, any]],
        temperature: float = 1.0
    ) -> Dict[int, torch.Tensor]:
        """
        Get logits from theta model for a set of requests

        Args:
            request_data: Dict mapping request index -> {
                "token_ids": full token sequence,
                "original_prompt": original user prompt (for applying theta's system prompt)
            }
            temperature: Temperature for logits

        Returns:
            Dict mapping request index -> logits tensor [vocab_size]
        """
        if not request_data:
            return {}

        # Prepare batch
        indices = list(request_data.keys())

        # Build prompts as TokensPrompt dicts (vLLM V1 API format)
        # Note: We use token_ids from teacher model's tokenization for consistency
        prompts = [
            {"prompt_token_ids": request_data[idx]["token_ids"]}
            for idx in indices
        ]

        # Sampling params to extract logits
        sampling_params = SamplingParams(
            temperature=0.0,  # Greedy (we only need logits)
            max_tokens=1,     # Generate 1 token to get logits
            logprobs=min(self.vocab_size, 20),  # vLLM limits logprobs
            prompt_logprobs=0
        )

        # Generate (uses prefix caching automatically)
        outputs = self.llm.generate(
            prompts=prompts,
            sampling_params=sampling_params,
            use_tqdm=False
        )

        # Extract logits
        result = {}
        for idx, output in zip(indices, outputs):
            logits = self._extract_logits_from_output(output)

            # Apply temperature
            if temperature != 1.0:
                logits = logits / temperature

            result[idx] = logits

        return result

    def _extract_logits_from_output(self, output) -> torch.Tensor:
        """Extract logits tensor from vLLM output"""
        logits = torch.full(
            (self.vocab_size,),
            float('-inf'),
            dtype=torch.float32,
            device=self.device
        )

        if len(output.outputs) > 0 and len(output.outputs[0].logprobs) > 0:
            logprobs_dict = output.outputs[0].logprobs[0]

            for token_id, logprob_obj in logprobs_dict.items():
                logits[token_id] = logprob_obj.logprob

        return logits

    def __repr__(self):
        return (
            f"ThetaModelV1(model={self.model_path}, "
            f"system_prompt={'set' if self.system_prompt else 'none'}, "
            f"chat_template={'enabled' if self.enable_chat_template else 'disabled'})"
        )
