"""
Optimal Sampling Logits Processor for vLLM V1

Implements optimal sampling using vLLM V1 LogitsProcessor API.

Architecture:
- Teacher model (π_t, larger) runs in outer vLLM (this processor receives its logits)
- Theta model (π_θ, smaller) runs in inner guide model
- Optimal distribution: q*(y|x) = π_θ(y|x)^(1-α) × π_t(y|x)^α / Z
"""

import torch
import torch.nn.functional as F
from typing import TYPE_CHECKING, Optional, Dict, List
import numpy as np
from vllm import SamplingParams
from vllm.v1.sample.logits_processor.interface import (
    BatchUpdate,
    LogitsProcessor,
    MoveDirectionality,
)

from .alpha_computer import AlphaComputer
from .guide_model_v1 import ThetaModelV1

if TYPE_CHECKING:
    from vllm.config import VllmConfig


class OptimalSamplingLogitsProcessorV1(LogitsProcessor):
    """
    Optimal Sampling Logits Processor for vLLM V1

    Architecture:
    - Receives logits from teacher model (π_t) - the outer vLLM model
    - Calls theta model (π_θ) - the inner guide model
    - Computes optimal distribution: q*(y|x) = π_θ^(1-α) × π_t^α / Z

    Where α is computed to satisfy KL symmetry:
        D_KL(q* || π_θ) = D_KL(q* || π_t)

    Args:
        vllm_config: vLLM configuration (for teacher model)
        device: torch device
        is_pin_memory: whether to use pinned memory
    """

    @classmethod
    def validate_params(cls, sampling_params: SamplingParams):
        """Validate sampling parameters for optimal sampling"""
        if sampling_params.extra_args:
            theta_model_path = sampling_params.extra_args.get("theta_model_path")
            if theta_model_path and not isinstance(theta_model_path, str):
                raise ValueError(f"theta_model_path must be a string, got {type(theta_model_path)}")

            alpha_method = sampling_params.extra_args.get("alpha_method", "kl_symmetry")
            if alpha_method not in ["kl_symmetry", "ess_balance", "entropy", "fixed"]:
                raise ValueError(f"Invalid alpha_method: {alpha_method}")

    def __init__(
        self,
        vllm_config: "VllmConfig",
        device: torch.device,
        is_pin_memory: bool
    ):
        """
        Initialize Optimal Sampling Logits Processor

        This is called once when vLLM engine starts.
        """
        print("=" * 80)
        print("🚀 Initializing Optimal Sampling Logits Processor V1")
        print("=" * 80)

        self.device = device
        self.vllm_config = vllm_config
        self.vocab_size = vllm_config.model_config.hf_config.vocab_size

        # Request state: index -> (prompt_token_ids, output_token_ids, sampling_params, original_prompt)
        # output_token_ids is a reference to the running list of generated tokens
        self.request_states: Dict[int, tuple[List[int], List[int], SamplingParams, str]] = {}

        # Theta model (inner) will be initialized lazily when first needed
        self.theta_model: Optional[ThetaModelV1] = None
        self.alpha_computer: Optional[AlphaComputer] = None

        # Track which requests use optimal sampling
        self.enabled_requests: set[int] = set()

        # Alpha statistics tracking (per request)
        self.alpha_history: Dict[int, List[float]] = {}
        self.track_alpha_stats: bool = False

        print(f"   Device: {device}")
        print(f"   Vocab size: {self.vocab_size}")
        print("=" * 80)

    def _ensure_theta_model(
        self,
        theta_model_path: str,
        theta_system_prompt: Optional[str],
        enable_chat_template: bool
    ):
        """Lazily initialize theta model when first needed"""
        if self.theta_model is None:
            print(f"\n📦 Initializing theta model (π_θ): {theta_model_path}")
            self.theta_model = ThetaModelV1(
                model_path=theta_model_path,
                vllm_config=self.vllm_config,
                device=self.device,
                system_prompt=theta_system_prompt,
                enable_chat_template=enable_chat_template,
                gpu_memory_utilization=0.4,
                enable_prefix_caching=True
            )

    def _ensure_alpha_computer(self, alpha_method: str, alpha_params: dict):
        """Lazily initialize alpha computer when first needed"""
        if self.alpha_computer is None:
            print(f"🧮 Initializing Alpha Computer (method={alpha_method})")
            self.alpha_computer = AlphaComputer(
                method=alpha_method,
                fixed_alpha=alpha_params.get("fixed_alpha", 0.5),
                alpha_min=alpha_params.get("alpha_min", 0.5),
                alpha_max=alpha_params.get("alpha_max", 1.0)
            )

    def is_argmax_invariant(self) -> bool:
        """
        Optimal sampling modifies logits and can change argmax.
        Return False to ensure it's applied even in greedy sampling.
        """
        return False

    def update_state(self, batch_update: Optional[BatchUpdate]) -> None:
        """
        Update internal state based on batch changes

        Called at the beginning of each engine step before apply().
        """
        if not batch_update:
            return

        # Process added requests
        for index, params, prompt_tok_ids, output_tok_ids in batch_update.added:
            if params.extra_args and params.extra_args.get("theta_model_path"):
                # This request wants optimal sampling
                # Get original prompt for theta model
                original_prompt = params.extra_args.get("original_prompts", [""])[0] if params.extra_args.get("original_prompts") else ""

                # Store prompt, output tokens, params, and original prompt
                self.request_states[index] = (prompt_tok_ids, output_tok_ids, params, original_prompt)
                self.enabled_requests.add(index)

                # Initialize alpha history for this request
                if params.extra_args.get("track_alpha_stats", True):
                    self.alpha_history[index] = []
                    self.track_alpha_stats = True

                # Lazily initialize theta model if needed
                theta_model_path = params.extra_args["theta_model_path"]
                theta_system_prompt = params.extra_args.get("theta_system_prompt")
                enable_chat_template = params.extra_args.get("enable_chat_template", False)
                self._ensure_theta_model(theta_model_path, theta_system_prompt, enable_chat_template)

                # Lazily initialize alpha computer if needed
                alpha_method = params.extra_args.get("alpha_method", "kl_symmetry")
                alpha_params = {
                    "fixed_alpha": params.extra_args.get("fixed_alpha", 0.5),
                    "alpha_min": params.extra_args.get("alpha_min", 0.5),
                    "alpha_max": params.extra_args.get("alpha_max", 1.0)
                }
                self._ensure_alpha_computer(alpha_method, alpha_params)
            else:
                # Regular request, not using optimal sampling
                self.request_states.pop(index, None)
                self.enabled_requests.discard(index)
                self.alpha_history.pop(index, None)

        if not self.enabled_requests:
            return

        # Process removed requests
        for index in batch_update.removed:
            self.request_states.pop(index, None)
            self.enabled_requests.discard(index)
            self.alpha_history.pop(index, None)

        # Process moved requests (unidirectional move or swap)
        for adx, bdx, direct in batch_update.moved:
            a_state = self.request_states.pop(adx, None)
            b_state = self.request_states.pop(bdx, None)
            a_history = self.alpha_history.pop(adx, None)
            b_history = self.alpha_history.pop(bdx, None)

            if a_state is not None:
                self.request_states[bdx] = a_state
                if a_history is not None:
                    self.alpha_history[bdx] = a_history
                if adx in self.enabled_requests:
                    self.enabled_requests.discard(adx)
                    self.enabled_requests.add(bdx)

            if direct == MoveDirectionality.SWAP and b_state is not None:
                self.request_states[adx] = b_state
                if b_history is not None:
                    self.alpha_history[adx] = b_history
                if bdx in self.enabled_requests:
                    self.enabled_requests.discard(bdx)
                    self.enabled_requests.add(adx)

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply optimal sampling to batch logits

        Args:
            logits: [batch_size, vocab_size] logits from teacher model (π_t)

        Returns:
            Modified logits tensor (can be in-place)
        """
        # Early exit if no requests use optimal sampling
        if not self.enabled_requests or self.theta_model is None:
            return logits

        # Get current token sequences and original prompts for enabled requests
        request_data = {}
        for idx in self.enabled_requests:
            if idx in self.request_states:
                prompt_tok_ids, output_tok_ids, params, original_prompt = self.request_states[idx]
                # Build full sequence: prompt + output tokens
                # Note: output_tok_ids is a reference to running output list
                full_sequence = prompt_tok_ids + output_tok_ids.copy()
                request_data[idx] = {
                    "token_ids": full_sequence,
                    "original_prompt": original_prompt
                }

        if not request_data:
            return logits

        # Get logits from theta model (π_θ)
        temperature = 1.0  # Get from params if needed
        theta_logits_dict = self.theta_model.get_logits_for_requests(
            request_data,
            temperature=temperature
        )

        # Mix logits for each enabled request
        for idx in theta_logits_dict.keys():
            if idx >= logits.shape[0]:
                continue  # Safety check

            # Get logits for this request
            logits_t = logits[idx]  # [vocab_size] - from teacher (outer)
            logits_theta = theta_logits_dict[idx]  # [vocab_size] - from theta (inner)

            # Convert to probabilities
            probs_t = F.softmax(logits_t, dim=-1)
            probs_theta = F.softmax(logits_theta, dim=-1)

            # Compute alpha (teacher weight)
            alpha = self.alpha_computer.compute(
                probs_theta.unsqueeze(0),
                probs_t.unsqueeze(0)
            ).item()

            # Track alpha if enabled
            if idx in self.alpha_history:
                self.alpha_history[idx].append(alpha)

            # Compute q* = π_θ^(1-α) × π_t^α
            q_star = self.alpha_computer.compute_q_star(
                probs_theta.unsqueeze(0),
                probs_t.unsqueeze(0),
                alpha
            ).squeeze(0)

            # Convert back to logits
            mixed_logits = torch.log(q_star + 1e-10)

            # Update in-place
            logits[idx] = mixed_logits

        return logits

    def get_alpha_stats(self, request_idx: int) -> Optional[Dict[str, float]]:
        """Get alpha statistics for a specific request"""
        if request_idx not in self.alpha_history:
            return None

        alphas = self.alpha_history[request_idx]
        if not alphas:
            return None

        alphas_array = np.array(alphas)
        return {
            "mean": float(np.mean(alphas_array)),
            "std": float(np.std(alphas_array)),
            "min": float(np.min(alphas_array)),
            "max": float(np.max(alphas_array)),
            "count": len(alphas),
            "history": alphas  # Full history
        }
