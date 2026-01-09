"""
Optimal Sampling with SGLang - Practical Implementation

Uses dual backends for maximum flexibility and control.
This version is immediately usable and doesn't depend on unreleased SGLang features.
"""

# MUST set before importing vLLM
import os

import torch
import numpy as np
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

from .alpha_computer import AlphaComputer


@dataclass
class SGLangSamplingOutput:
    """Output from SGLang optimal sampling"""
    generated_texts: List[str]
    generated_ids: List[List[int]]
    num_tokens: List[int]
    alphas: Optional[List[List[float]]] = None
    ess_ratios: Optional[List[List[float]]] = None


class OptimalSamplingSGLang:
    """
    Optimal Sampling with clean control flow

    This implementation uses two vLLM backends but with explicit
    control over the generation process, making it much cleaner
    than the logits processor approach.

    Key advantages:
    - Direct control over each generation step
    - Easy to debug and understand
    - Can print/inspect intermediate values
    - No dependency on logits processors
    - Works with any vLLM version

    Usage:
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

    def __init__(
        self,
        model_base: str,
        model_teacher: str,
        alpha_method: str = "kl_symmetry",
        fixed_alpha: float = 0.5,
        alpha_min: float = 0.5,
        alpha_max: float = 1.0,
        gpu_memory_utilization: float = 0.85,
        tensor_parallel_size: int = 1,
        dtype: str = "auto",
        constraint_to_target: bool = False,
        target_top_k: int = -1,
        target_top_p: float = 1.0,
        **kwargs
    ):
        """
        Initialize Optimal Sampling

        Args:
            model_base: Base model path (π_θ)
            model_teacher: Teacher model path (π_t)
            alpha_method: Alpha computation method
            fixed_alpha: Fixed alpha (when method="fixed")
            alpha_min: Minimum alpha
            alpha_max: Maximum alpha
            gpu_memory_utilization: Total GPU memory to use (split between models)
            tensor_parallel_size: Tensor parallel size
            dtype: Model dtype
            constraint_to_target: Constrain to teacher support
            target_top_k: Top-k constraint
            target_top_p: Top-p constraint
        """
        self.model_base_path = model_base
        self.model_teacher_path = model_teacher
        self.constraint_to_target = constraint_to_target
        self.target_top_k = target_top_k
        self.target_top_p = target_top_p

        print("=" * 80)
        print("🚀 Initializing Optimal Sampling (SGLang-style)")
        print("=" * 80)
        print(f"Base Model (π_θ): {model_base}")
        print(f"Teacher Model (π_t): {model_teacher}")
        print(f"Alpha Method: {alpha_method}")
        print(f"GPU Memory: {gpu_memory_utilization:.1%} total")
        print()

        # Initialize alpha computer
        print("🧮 Initializing Alpha Computer...")
        self.alpha_computer = AlphaComputer(
            method=alpha_method,
            fixed_alpha=fixed_alpha,
            alpha_min=alpha_min,
            alpha_max=alpha_max
        )

        # Calculate memory split
        mem_per_model = gpu_memory_utilization / 2.0

        # Initialize base model (π_θ)
        print(f"\n📦 Loading Base Model (π_θ)...")
        print(f"   Memory: {mem_per_model:.1%}")

        self.llm_base = LLM(
            model=model_base,
            gpu_memory_utilization=mem_per_model,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            trust_remote_code=True,
            enforce_eager=True,  # Disable cuda graphs for logits access
            disable_log_stats=True,
            **kwargs
        )
        print("✅ Base model loaded")

        # Initialize teacher model (π_t)
        print(f"\n📦 Loading Teacher Model (π_t)...")
        print(f"   Memory: {mem_per_model:.1%}")

        self.llm_teacher = LLM(
            model=model_teacher,
            gpu_memory_utilization=mem_per_model,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            trust_remote_code=True,
            enforce_eager=True,
            disable_log_stats=True,
            **kwargs
        )
        print("✅ Teacher model loaded")

        # Get tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_base, trust_remote_code=True)
        self.vocab_size = len(self.tokenizer)

        print(f"\n   Vocab size: {self.vocab_size}")
        print("\n" + "=" * 80)
        print("✅ Initialization Complete!")
        print("=" * 80)
        print()

    def generate(
        self,
        prompts: List[str],
        max_tokens: int = 100,
        temperature: float = 1.0,
        collect_diagnostics: bool = False,
        verbose: bool = False
    ) -> SGLangSamplingOutput:
        """
        Generate text using optimal sampling with TRUE BATCH processing

        This implementation processes multiple prompts in parallel for
        maximum GPU efficiency!

        Args:
            prompts: Input prompts
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            collect_diagnostics: Collect alpha/ESS values
            verbose: Print progress

        Returns:
            SGLangSamplingOutput
        """
        if verbose:
            print(f"🔄 Generating {len(prompts)} sequences in BATCH mode...")
            print(f"   Batch size: {len(prompts)}")
            print(f"   Max tokens: {max_tokens}")
            print(f"   Temperature: {temperature}")

        # Use batch generation for efficiency
        result = self._generate_batch(
            prompts=prompts,
            max_tokens=max_tokens,
            temperature=temperature,
            collect_diagnostics=collect_diagnostics,
            verbose=verbose
        )

        return result

    def _generate_batch(
        self,
        prompts: List[str],
        max_tokens: int,
        temperature: float,
        collect_diagnostics: bool,
        verbose: bool
    ) -> SGLangSamplingOutput:
        """
        Generate multiple sequences in TRUE BATCH mode

        This processes all prompts together for maximum GPU efficiency!
        """
        batch_size = len(prompts)

        # Tokenize all prompts
        batch_input_ids = [self.tokenizer.encode(p) for p in prompts]
        batch_current_ids = [ids.copy() for ids in batch_input_ids]

        # Track state for each sequence
        batch_generated_ids = [[] for _ in range(batch_size)]
        batch_finished = [False] * batch_size
        batch_alphas = [[] for _ in range(batch_size)] if collect_diagnostics else None
        batch_ess_ratios = [[] for _ in range(batch_size)] if collect_diagnostics else None

        if verbose:
            print(f"\n{'─' * 80}")
            print(f"Starting batch generation...")

        for step in range(max_tokens):
            # Get active (unfinished) sequences
            active_indices = [i for i in range(batch_size) if not batch_finished[i]]

            if not active_indices:
                break  # All sequences finished

            active_ids = [batch_current_ids[i] for i in active_indices]

            # Batch get logits from both models
            logits_base_batch = self._get_batch_logits(self.llm_base, active_ids)
            logits_teacher_batch = self._get_batch_logits(self.llm_teacher, active_ids)

            # Process each active sequence
            for idx, seq_idx in enumerate(active_indices):
                logits_base = logits_base_batch[idx]
                logits_teacher = logits_teacher_batch[idx]

                # Convert to probabilities
                probs_base = torch.softmax(logits_base / temperature, dim=-1)
                probs_teacher = torch.softmax(logits_teacher / temperature, dim=-1)

                # Compute alpha
                alpha = self.alpha_computer.compute(
                    probs_base.unsqueeze(0),
                    probs_teacher.unsqueeze(0)
                ).item()

                # Compute q*
                q_star = self.alpha_computer.compute_q_star(
                    probs_base.unsqueeze(0),
                    probs_teacher.unsqueeze(0),
                    alpha
                ).squeeze(0)

                # Apply constraint if enabled
                if self.constraint_to_target:
                    q_star = self._apply_constraint(q_star, probs_teacher)

                # Sample from q*
                next_token = torch.multinomial(q_star, num_samples=1).item()

                # Store
                batch_generated_ids[seq_idx].append(next_token)
                batch_current_ids[seq_idx].append(next_token)

                # Diagnostics
                if collect_diagnostics:
                    batch_alphas[seq_idx].append(alpha)

                    # ESS
                    ess_base = 1.0 / ((probs_base ** 2) / (q_star + 1e-10)).sum().item()
                    ess_teacher = 1.0 / ((probs_teacher ** 2) / (q_star + 1e-10)).sum().item()
                    batch_ess_ratios[seq_idx].append(ess_base / (ess_teacher + 1e-10))

                # Check EOS
                if next_token == self.tokenizer.eos_token_id:
                    batch_finished[seq_idx] = True

            # Verbose output
            if verbose and (step + 1) % 20 == 0:
                active_count = len(active_indices)
                finished_count = batch_size - active_count
                if collect_diagnostics and batch_alphas:
                    # Compute average alpha for active sequences
                    active_alphas = [batch_alphas[i][-1] for i in active_indices if batch_alphas[i]]
                    avg_alpha = np.mean(active_alphas) if active_alphas else 0
                    print(f"   Step {step+1}: Active={active_count}, Finished={finished_count}, α_avg={avg_alpha:.3f}")
                else:
                    print(f"   Step {step+1}: Active={active_count}, Finished={finished_count}")

        # Decode all results
        generated_texts = [
            self.tokenizer.decode(ids, skip_special_tokens=True)
            for ids in batch_generated_ids
        ]
        num_tokens = [len(ids) for ids in batch_generated_ids]

        if verbose:
            print(f"\n✅ Batch generation completed!")
            print(f"   Total tokens: {sum(num_tokens)}")
            print(f"   Avg tokens/seq: {np.mean(num_tokens):.1f}")
            if collect_diagnostics and batch_alphas:
                avg_alpha = np.mean([np.mean(a) if a else 0 for a in batch_alphas])
                print(f"   Average alpha: {avg_alpha:.3f}")

        return SGLangSamplingOutput(
            generated_texts=generated_texts,
            generated_ids=batch_generated_ids,
            num_tokens=num_tokens,
            alphas=batch_alphas,
            ess_ratios=batch_ess_ratios
        )

    def _generate_single(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        collect_diagnostics: bool,
        verbose: bool
    ) -> Dict[str, Any]:
        """
        Generate a single sequence

        This is where the magic happens - clean, explicit control!
        """
        # Tokenize
        input_ids = self.tokenizer.encode(prompt)
        current_ids = input_ids.copy()

        generated_ids = []
        alphas = [] if collect_diagnostics else None
        ess_ratios = [] if collect_diagnostics else None

        for step in range(max_tokens):
            # Step 1: Get logits from both models
            logits_base, logits_teacher = self._get_dual_logits(current_ids)

            # Step 2: Convert to probabilities
            probs_base = torch.softmax(logits_base / temperature, dim=-1)
            probs_teacher = torch.softmax(logits_teacher / temperature, dim=-1)

            # Step 3: Compute alpha
            alpha = self.alpha_computer.compute(
                probs_base.unsqueeze(0),
                probs_teacher.unsqueeze(0)
            ).item()

            # Step 4: Compute q*
            q_star = self.alpha_computer.compute_q_star(
                probs_base.unsqueeze(0),
                probs_teacher.unsqueeze(0),
                alpha
            ).squeeze(0)

            # Apply constraint if enabled
            if self.constraint_to_target:
                q_star = self._apply_constraint(q_star, probs_teacher)

            # Step 5: Sample from q*
            next_token = torch.multinomial(q_star, num_samples=1).item()

            # Store
            generated_ids.append(next_token)
            current_ids.append(next_token)

            # Diagnostics
            if collect_diagnostics:
                alphas.append(alpha)

                # ESS
                ess_base = 1.0 / ((probs_base ** 2) / (q_star + 1e-10)).sum().item()
                ess_teacher = 1.0 / ((probs_teacher ** 2) / (q_star + 1e-10)).sum().item()
                ess_ratios.append(ess_base / (ess_teacher + 1e-10))

            # Check EOS
            if next_token == self.tokenizer.eos_token_id:
                break

            # Verbose output
            if verbose and (step + 1) % 20 == 0:
                decoded = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                print(f"   Step {step+1}: α={alpha:.3f}, text={decoded[:40]}...")

        # Decode
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        return {
            "text": generated_text,
            "token_ids": generated_ids,
            "alphas": alphas,
            "ess_ratios": ess_ratios
        }

    def _get_dual_logits(self, input_ids: List[int]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get logits from both models

        This uses vLLM's generate() with special parameters to extract logits.
        Note: vLLM limits logprobs to max 20, so we get top-k and fill rest with -inf
        """
        # Prepare input
        prompt_token_ids = input_ids

        # Sampling params to get logits (vLLM 0.6.3 limits to max 20 logprobs)
        sampling_params = SamplingParams(
            max_tokens=1,
            temperature=0.0,
            logprobs=20,  # vLLM 0.6.3 max limit
            prompt_logprobs=0
        )

        # Get from base model
        output_base = self.llm_base.generate(
            prompt_token_ids=[prompt_token_ids],
            sampling_params=sampling_params,
            use_tqdm=False
        )[0]

        # Get from teacher model
        output_teacher = self.llm_teacher.generate(
            prompt_token_ids=[prompt_token_ids],
            sampling_params=sampling_params,
            use_tqdm=False
        )[0]

        # Extract logits
        logits_base = self._extract_logits_from_output(output_base)
        logits_teacher = self._extract_logits_from_output(output_teacher)

        return logits_base, logits_teacher

    def _get_batch_logits(self, llm, batch_input_ids: List[List[int]]) -> List[torch.Tensor]:
        """
        Get logits from a model for a BATCH of sequences

        This is the key to efficient batch processing!

        Args:
            llm: vLLM model instance
            batch_input_ids: List of token ID sequences

        Returns:
            List of logits tensors, one per sequence
        """
        # Sampling params to get logits
        sampling_params = SamplingParams(
            max_tokens=1,
            temperature=0.0,
            logprobs=20,  # vLLM 0.6.3 max limit
            prompt_logprobs=0
        )

        # Batch generate - THIS IS WHERE THE MAGIC HAPPENS!
        outputs = llm.generate(
            prompt_token_ids=batch_input_ids,
            sampling_params=sampling_params,
            use_tqdm=False
        )

        # Extract logits for each sequence
        batch_logits = []
        for output in outputs:
            logits = self._extract_logits_from_output(output)
            batch_logits.append(logits)

        return batch_logits

    def _extract_logits_from_output(self, output) -> torch.Tensor:
        """Extract logits from vLLM output"""
        # vLLM returns logprobs, convert back to logits
        logits = torch.full(
            (self.vocab_size,),
            float('-inf'),
            dtype=torch.float32
        )

        if len(output.outputs) > 0 and len(output.outputs[0].logprobs) > 0:
            logprobs_dict = output.outputs[0].logprobs[0]

            for token_id, logprob_obj in logprobs_dict.items():
                logits[token_id] = logprob_obj.logprob

        return logits

    def _apply_constraint(
        self,
        q_star: torch.Tensor,
        probs_teacher: torch.Tensor
    ) -> torch.Tensor:
        """Apply support constraint"""
        mask = torch.ones_like(probs_teacher, dtype=torch.bool)

        # Top-k
        if self.target_top_k > 0:
            k = min(self.target_top_k, len(probs_teacher))
            _, top_k_indices = torch.topk(probs_teacher, k=k)
            mask_k = torch.zeros_like(probs_teacher, dtype=torch.bool)
            mask_k[top_k_indices] = True
            mask = mask & mask_k

        # Top-p
        if self.target_top_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(probs_teacher, descending=True)
            cumsum_probs = torch.cumsum(sorted_probs, dim=0)
            indices_to_keep = cumsum_probs <= self.target_top_p
            indices_to_keep[0] = True

            mask_p = torch.zeros_like(probs_teacher, dtype=torch.bool)
            mask_p[sorted_indices[indices_to_keep]] = True
            mask = mask & mask_p

        # Apply and renormalize
        q_star_masked = q_star * mask.float()
        q_star_masked = q_star_masked / (q_star_masked.sum() + 1e-10)

        return q_star_masked

    def __repr__(self):
        return (
            f"OptimalSamplingSGLang(\n"
            f"  base={self.model_base_path},\n"
            f"  teacher={self.model_teacher_path},\n"
            f"  method={self.alpha_computer.method}\n"
            f")"
        )
