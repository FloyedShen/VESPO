#!/usr/bin/env python3
"""
Large-scale data generation example for Optimal Sampling HF

This example demonstrates:
1. Batch processing of large datasets
2. Saving outputs to JSONL format
3. Progress tracking and checkpointing
4. Efficient memory usage
"""

import torch
import json
from pathlib import Path
from tqdm import tqdm
from optimal_sampling_hf import OptimalSamplingModel

def save_to_jsonl(outputs, prompts, output_file, mode='a'):
    """Save generation outputs to JSONL file"""
    with open(output_file, mode) as f:
        for i, text in enumerate(outputs.generated_texts):
            data = {
                "prompt": prompts[i],
                "generated_text": text,
                "num_tokens": outputs.generated_ids[i].shape[0],
                "alpha_mean": float(outputs.alpha_values[i].mean()),
                "alpha_std": float(outputs.alpha_values[i].std()),
                "ess_ratio_mean": float(outputs.ess_ratios[i].mean()),
            }

            # Add diagnostics if available
            if outputs.diagnostics:
                for key in ["kl_theta", "kl_t"]:
                    if key in outputs.diagnostics:
                        data[f"{key}_mean"] = float(outputs.diagnostics[key][i].mean())

            f.write(json.dumps(data) + '\n')

def main():
    print("=" * 80)
    print("Optimal Sampling HF - Data Generation Example")
    print("=" * 80)

    # Configuration
    output_dir = Path("./data_generation_output")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "generated_data.jsonl"

    num_examples = 100
    batch_size = 4
    max_new_tokens = 256

    # Initialize model
    print("\n[1/3] Initializing model...")
    model = OptimalSamplingModel(
        model_theta_path="Qwen/Qwen2.5-1.5B-Instruct",
        model_t_path="Qwen/Qwen2.5-3B-Instruct",
        alpha_method="kl_symmetry",
        dtype=torch.bfloat16,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # Example prompts (in practice, load from dataset)
    print(f"\n[2/3] Generating {num_examples} examples...")
    all_prompts = [
        f"Question {i}: What is the capital of France?" for i in range(num_examples)
    ]

    # Clear output file
    if output_file.exists():
        output_file.unlink()

    # Process in batches
    num_batches = (num_examples + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(num_batches), desc="Generating batches"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_examples)
        batch_prompts = all_prompts[start_idx:end_idx]

        # Generate
        outputs = model.generate(
            prompts=batch_prompts,
            max_new_tokens=max_new_tokens,
            temperature=0.8,
            return_diagnostics=True,
            skip_decode=False
        )

        # Save to file
        save_to_jsonl(outputs, batch_prompts, output_file)

    # Print statistics
    print("\n[3/3] Generation complete!")
    print("=" * 80)
    print(f"Output file: {output_file}")
    print(f"Total examples: {num_examples}")
    print(f"File size: {output_file.stat().st_size / 1024 / 1024:.2f} MB")
    print("=" * 80)

    # Show sample output
    print("\nSample outputs (first 3):")
    with open(output_file) as f:
        for i, line in enumerate(f):
            if i >= 3:
                break
            data = json.loads(line)
            print(f"\n--- Example {i+1} ---")
            print(f"Prompt: {data['prompt'][:50]}...")
            print(f"Generated: {data['generated_text'][:100]}...")
            print(f"Tokens: {data['num_tokens']}")
            print(f"Alpha: {data['alpha_mean']:.3f} ± {data['alpha_std']:.3f}")

    print("\n✓ Data generation completed successfully!")

if __name__ == "__main__":
    main()
