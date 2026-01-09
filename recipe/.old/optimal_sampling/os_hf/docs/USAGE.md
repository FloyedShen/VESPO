# Optimal Sampling HF - Usage Guide

Complete guide to using the Optimal Sampling HF package.

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [API Reference](#api-reference)
4. [Examples](#examples)
5. [Advanced Usage](#advanced-usage)
6. [Performance Tips](#performance-tips)

---

## Installation

### From Source

```bash
cd os_hf
pip install -e .
```

### Install Flash Attention 2 (Recommended)

```bash
pip install flash-attn --no-build-isolation
```

Flash Attention 2 provides significant speed improvements (2-4x faster) and memory efficiency.

---

## Quick Start

### Basic Example

```python
import torch
from optimal_sampling_hf import OptimalSamplingModel

# Initialize model
model = OptimalSamplingModel(
    model_theta_path="Qwen/Qwen2.5-1.5B-Instruct",  # Base model
    model_t_path="Qwen/Qwen2.5-3B-Instruct",        # Teacher model
    alpha_method="kl_symmetry",
    dtype=torch.bfloat16,
    device="cuda"
)

# Generate
outputs = model.generate(
    prompts=["What is 2+2?", "Explain quantum computing"],
    max_new_tokens=100,
    temperature=0.8
)

# Access results
for text in outputs.generated_texts:
    print(text)
```

---

## API Reference

### OptimalSamplingModel

Main model class for optimal sampling with HuggingFace Transformers.

#### Constructor Parameters

```python
OptimalSamplingModel(
    model_theta_path: str,              # Base model (π_θ)
    model_t_path: str = None,           # Teacher model (π_t)
    alpha_method: str = "kl_symmetry",  # Alpha computation method
    fixed_alpha: float = 0.5,           # Fixed alpha (if method="fixed")
    alpha_min: float = 0.5,             # Minimum alpha value
    alpha_max: float = 1.0,             # Maximum alpha value
    constraint_to_target: bool = False, # Limit to π_t support
    target_top_k: int = -1,             # Top-k constraint
    target_top_p: float = 1.0,          # Top-p constraint
    device: str = "cuda",               # Device
    dtype: torch.dtype = torch.float16, # Data type
    **kwargs                            # Additional transformers args
)
```

#### Alpha Methods

| Method | Description | Use Case |
|--------|-------------|----------|
| `kl_symmetry` | Balances KL(q\*‖π_θ) and KL(q\*‖π_t) | **Recommended** for most cases |
| `ess_balance` | Balances effective sample sizes | Good for diversity |
| `entropy` | Maximizes entropy of q\* | Exploration-focused |
| `fixed` | Uses fixed α value | Simple baseline |
| `reverse_kl_symmetry` | Balances KL(π_θ‖q\*) and KL(π_t‖q\*) | Mode-covering |

#### generate() Method

```python
outputs = model.generate(
    prompts: List[str],                # Input prompts
    prompts_t: Optional[List[str]] = None,  # Teacher prompts (optional)
    max_new_tokens: int = 100,         # Max tokens to generate
    temperature: float = 1.0,          # Sampling temperature
    top_p: float = 1.0,                # Nucleus sampling
    top_k: int = -1,                   # Top-k sampling
    return_diagnostics: bool = True,   # Return diagnostics
    skip_decode: bool = False,         # Skip decoding
    return_logits: bool = False,       # Return logits
    return_q_star_probs: bool = False, # Return q* probs
)
```

Returns `SamplingOutput` with:
- `generated_texts`: List[str] - Decoded text
- `generated_ids`: torch.Tensor - Token IDs [batch, seq_len]
- `alpha_values`: torch.Tensor - Alpha values [batch, seq_len]
- `ess_ratios`: torch.Tensor - ESS ratios [batch, seq_len]
- `diagnostics`: Dict - KL divergences, etc.
- `logits`: Optional[Dict] - Raw logits (if requested)
- `q_star_probs`: Optional[torch.Tensor] - q* probabilities (if requested)

---

## Examples

### Example 1: Different Alpha Methods

```python
from optimal_sampling_hf import OptimalSamplingModel

# KL Symmetry (recommended)
model_kl = OptimalSamplingModel(
    model_theta_path="Qwen/Qwen2.5-1.5B-Instruct",
    model_t_path="Qwen/Qwen2.5-3B-Instruct",
    alpha_method="kl_symmetry"
)

# Fixed alpha
model_fixed = OptimalSamplingModel(
    model_theta_path="Qwen/Qwen2.5-1.5B-Instruct",
    model_t_path="Qwen/Qwen2.5-3B-Instruct",
    alpha_method="fixed",
    fixed_alpha=0.5  # 50% mix
)

# Entropy-based
model_entropy = OptimalSamplingModel(
    model_theta_path="Qwen/Qwen2.5-1.5B-Instruct",
    model_t_path="Qwen/Qwen2.5-3B-Instruct",
    alpha_method="entropy"
)
```

### Example 2: Batch Processing

```python
import torch
from optimal_sampling_hf import OptimalSamplingModel

model = OptimalSamplingModel(
    model_theta_path="Qwen/Qwen2.5-1.5B-Instruct",
    model_t_path="Qwen/Qwen2.5-3B-Instruct",
    alpha_method="kl_symmetry",
    dtype=torch.bfloat16
)

# Process multiple prompts
prompts = [f"Question {i}: What is the capital of France?" for i in range(10)]

outputs = model.generate(
    prompts=prompts,
    max_new_tokens=50,
    temperature=0.8
)

# Save to file
import json
with open("outputs.jsonl", "w") as f:
    for i, text in enumerate(outputs.generated_texts):
        data = {
            "prompt": prompts[i],
            "output": text,
            "alpha_mean": float(outputs.alpha_values[i].mean())
        }
        f.write(json.dumps(data) + "\n")
```

### Example 3: With Diagnostics

```python
outputs = model.generate(
    prompts=["Explain machine learning"],
    max_new_tokens=200,
    temperature=0.8,
    return_diagnostics=True
)

# Analyze diagnostics
print(f"Average α: {outputs.alpha_values.mean():.3f}")
print(f"α range: [{outputs.alpha_values.min():.3f}, {outputs.alpha_values.max():.3f}]")
print(f"ESS ratio: {outputs.ess_ratios.mean():.3f}")

if outputs.diagnostics:
    kl_theta = outputs.diagnostics["kl_theta"].mean()
    kl_t = outputs.diagnostics["kl_t"].mean()
    print(f"KL(q||π_θ): {kl_theta:.4f}")
    print(f"KL(q||π_t): {kl_t:.4f}")
```

---

## Advanced Usage

### Dual Prompts (Different for Base and Teacher)

```python
# Teacher sees instruction-formatted prompt
# Base sees natural language prompt
outputs = model.generate(
    prompts=["Question: What is AI?\nAnswer:"],  # For base model
    prompts_t=["<|im_start|>user\nWhat is AI?<|im_end|>\n<|im_start|>assistant\n"],  # For teacher
    max_new_tokens=100
)
```

### Memory-Efficient Generation (4-bit Quantization)

```python
model = OptimalSamplingModel(
    model_theta_path="Qwen/Qwen2.5-7B-Instruct",
    model_t_path="Qwen/Qwen2.5-14B-Instruct",
    dtype=torch.bfloat16,
    load_in_4bit=True,  # 4-bit quantization
    bnb_4bit_compute_dtype=torch.bfloat16
)
```

### Support Constraints (Numerical Stability)

```python
model = OptimalSamplingModel(
    model_theta_path="Qwen/Qwen2.5-1.5B-Instruct",
    model_t_path="Qwen/Qwen2.5-3B-Instruct",
    alpha_method="kl_symmetry",
    constraint_to_target=True,  # Limit to π_t support
    target_top_k=100,            # Only top-100 tokens from π_t
    target_top_p=0.95            # Or top-p
)
```

---

## Performance Tips

### For Small Models (1.5B-3B)

```python
model = OptimalSamplingModel(
    model_theta_path="Qwen/Qwen2.5-1.5B-Instruct",
    model_t_path="Qwen/Qwen2.5-3B-Instruct",
    dtype=torch.bfloat16,  # Use bfloat16
    device="cuda"
)

# Use larger batches
outputs = model.generate(
    prompts=prompts,
    max_new_tokens=512,
    batch_size=16  # Larger batch
)
```

### For Large Models (7B-14B)

```python
from transformers import BitsAndBytesConfig

model = OptimalSamplingModel(
    model_theta_path="Qwen/Qwen2.5-7B-Instruct",
    model_t_path="Qwen/Qwen2.5-14B-Instruct",
    dtype=torch.bfloat16,
    load_in_4bit=True,  # 4-bit quantization
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# Use smaller batches
outputs = model.generate(
    prompts=prompts,
    max_new_tokens=512,
    batch_size=4  # Smaller batch
)
```

### Enable Flash Attention 2

Flash Attention is automatically enabled if the package is installed:

```bash
pip install flash-attn --no-build-isolation
```

Verify:
```python
# Will print if Flash Attention is enabled
model = OptimalSamplingModel(...)
# Output: "✅ Flash Attention 2 auto-enabled"
```

### Batch Size Tuning

| Model Size | GPU Memory | Recommended Batch Size |
|-----------|-----------|------------------------|
| 1.5B + 3B | 24GB | 16-32 |
| 3B + 7B | 40GB | 8-16 |
| 7B + 14B | 80GB | 4-8 |

---

## Command-Line Tools

### Data Generation Script

```bash
cd scripts/

# Basic usage
python generate_data.py \
  --model_theta Qwen/Qwen2.5-1.5B-Instruct \
  --model_t Qwen/Qwen2.5-3B-Instruct \
  --dataset synthetic \
  --num_examples 1000 \
  --max_tokens 256 \
  --batch_size 8 \
  --output_dir ./outputs

# With custom alpha settings
python generate_data.py \
  --model_theta Qwen/Qwen2.5-1.5B-Instruct \
  --model_t Qwen/Qwen2.5-3B-Instruct \
  --alpha_method kl_symmetry \
  --alpha_min 0.3 \
  --alpha_max 1.0 \
  --num_examples 5000 \
  --max_tokens 512 \
  --output_dir ./outputs
```

---

## Troubleshooting

### Out of Memory?

- Reduce `batch_size`
- Use `load_in_4bit=True`
- Use `dtype=torch.float16` or `torch.bfloat16`
- Reduce `max_new_tokens`

### Flash Attention not working?

```bash
# Install flash-attn
pip install flash-attn --no-build-isolation

# Verify
python -c "import flash_attn; print('Flash Attention available')"
```

### Slow generation?

- Ensure Flash Attention 2 is installed
- Use `dtype=torch.bfloat16`
- Increase batch size if memory allows
- Use GPU instead of CPU

---

For more examples, see the `examples/` directory.
