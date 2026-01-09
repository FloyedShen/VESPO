# Optimal Sampling HF

**High-Performance Optimal Sampling with HuggingFace Transformers + Flash Attention 2**


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True accelerate launch \
    --num_processes=8 \
    --multi_gpu \
    --main_process_port 29501 \
    trl/scripts/sft.py \
    --model_name_or_path Qwen/Qwen3-1.7B \
    --dataset_name /diancpfs/user/guobin/verl/recipe/optimal_sampling/os_hf/outputs/sft/optimal_sampling_sft \
    --fsdp "full_shard auto_wrap" \
    --learning_rate 2.0e-5 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 4 \
    --max_length 16384 \
    --gradient_checkpointing \
    --eos_token '<|im_end|>' \
    --save_steps 20 \
    --use_liger_kernel \
    --output_dir checkpoints/qwen3-1.7b-DeepScaleR-os-sft \
    --report_to none \
    --packing False


CUDA_VISIBLE_DEVICES=0,1,2,3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True accelerate launch \
    --num_processes=4 \
    --multi_gpu \
    --main_process_port 29501 \
    trl/scripts/sft.py \
    --model_name_org_path Qwen/Qwen3-1.7B \
    --dataset_name /mnt/tidal-alsh-hilab/usr/shenguobin/verl/recipe/optimal_sampling/os_hf/outputs/sft/baseline_sft \
    --fsdp "full_shard auto_wrap" \
    --learning_rate 2.0e-5 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 4 \
    --max_length 16384 \
    --gradient_checkpointing \
    --eos_token '<|im_end|>' \
    --save_steps 20 \
    --use_liger_kernel \
    --output_dir checkpoints/qwen3-1.7b-DeepScaleR-baseline-sft \
    --report_to none \
    --packing False



CUDA_VISIBLE_DEVICES=4,5,6,7 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True accelerate launch \
    --num_processes=4 \
    --multi_gpu \
    --main_process_port 29501 \
    trl/scripts/sft.py \
    --model_name_or_path Qwen/Qwen3-1.7B \
    --dataset_name /mnt/tidal-alsh-hilab/usr/shenguobin/verl/recipe/optimal_sampling/os_hf/outputs/sft/warmstart_sft \
    --fsdp "full_shard auto_wrap" \
    --learning_rate 2.0e-5 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 4 \
    --max_length 16384 \
    --gradient_checkpointing \
    --eos_token '<|im_end|>' \
    --save_steps 20 \
    --use_liger_kernel \
    --output_dir checkpoints/qwen3-1.7b-DeepScaleR-ws-sft \
    --report_to none \
    --packing False


python eval_checkpoints.py \
    --checkpoint-dir /mnt/tidal-alsh-hilab/usr/shenguobin/trl/checkpoints \
    --output-dir ../data/eval_results \
    --tensor-parallel-size 8 \
    --batch-size 256 \
    --resume \
    --verbose
#   --extra-models Qwen/Qwen3-1.7B \


[![Performance](https://img.shields.io/badge/Speed-6.4x%20faster%20than%20vLLM-green)]()
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)]()
[![License](https://img.shields.io/badge/License-Apache%202.0-blue)]()

## 🚀 Performance Highlights

- **6.4x faster than vLLM V1** for data generation tasks
- **36.71 tokens/s** vs vLLM's 5.70 tokens/s (tested on Qwen models)
- **2.4x faster initialization** (7.0s vs 17.0s)
- **Native Flash Attention 2 support** for optimal memory efficiency

## 📋 What is Optimal Sampling?

Optimal Sampling mixes distributions from two models:
- **π_θ (Theta)**: Base model (e.g., Qwen-1.5B-Instruct)
- **π_t (Teacher)**: Teacher model (e.g., Qwen-3B-Instruct)
- **q\* (Optimal)**: Mixed distribution using geometric mean

The mixing parameter α is computed dynamically using methods like KL symmetry to balance exploration and exploitation.

## 🔧 Installation

### From Source

```bash
cd os_hf
pip install -e .
```

### Install Flash Attention 2 (Recommended)

```bash
pip install flash-attn --no-build-isolation
```

## 💡 Quick Start

### Basic Usage

```python
from optimal_sampling_hf import OptimalSamplingModel

# Initialize model
model = OptimalSamplingModel(
    model_theta_path="Qwen/Qwen2.5-1.5B-Instruct",  # Base model
    model_t_path="Qwen/Qwen2.5-3B-Instruct",        # Teacher model
    alpha_method="kl_symmetry",                      # Alpha computation
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

### Advanced Usage

```python
# With detailed diagnostics
outputs = model.generate(
    prompts=["Your question here"],
    max_new_tokens=200,
    temperature=0.8,
    save_logits=True,          # Save logits for analysis
    save_q_star_probs=True,    # Save q* distribution
)

# Access diagnostics
print(f"Average α: {outputs.diagnostics['alpha_mean']:.3f}")
print(f"ESS ratio: {outputs.diagnostics['ess_ratio_mean']:.3f}")
```

## 📊 Alpha Computation Methods

| Method | Description | Use Case |
|--------|-------------|----------|
| `kl_symmetry` | Balances KL(q\*‖π_θ) and KL(q\*‖π_t) | **Recommended** for most cases |
| `ess_balance` | Balances effective sample sizes | Good for diversity |
| `entropy` | Maximizes entropy of q\* | Exploration-focused |
| `fixed` | Uses fixed α value | Simple baseline |

## 🔥 Features

### Why Choose HF + Flash Attention?

✅ **Blazing Fast**: 6.4x faster than vLLM for data generation
✅ **Memory Efficient**: Flash Attention 2 reduces memory usage
✅ **Simple & Clean**: Direct PyTorch/HF implementation
✅ **Easy to Debug**: No cross-process communication overhead
✅ **Flexible**: Easy to customize and extend

### Key Capabilities

- **Automatic Flash Attention 2 detection**
- **Different tokenizers support** (θ and teacher can have different tokenizers)
- **Special token handling** (EOS, PAD, etc.)
- **Rich diagnostics** (α values, ESS, KL divergence)
- **Batch processing** for efficiency
- **Support constraints** (top-k, top-p on teacher distribution)

## 📁 Package Structure

```
os_hf/
├── optimal_sampling_hf/    # Core package
│   ├── __init__.py
│   └── optimal_sampling.py # Main model implementation
├── examples/               # Usage examples
│   ├── basic_usage.py
│   └── data_generation.py
├── scripts/               # Data generation scripts
│   └── generate_data.py
├── docs/                  # Documentation
├── setup.py              # Package installation
└── README.md             # This file
```

## 🎯 Use Cases

### 1. Large-Scale Data Generation

Perfect for generating training data with optimal sampling:

```bash
python scripts/generate_data.py \
    --model_theta Qwen/Qwen2.5-1.5B-Instruct \
    --model_t Qwen/Qwen2.5-3B-Instruct \
    --dataset your-dataset \
    --output_dir ./data/outputs \
    --num_examples 10000
```

### 2. Distillation Data Preparation

Generate high-quality reasoning traces for knowledge distillation:

```bash
python scripts/generate_data_multigpu.py   \
  --model_teacher "/mnt/tidal-alsh-hilab/usr/shenguobin/modelscope/Qwen/Qwen3-1.7B"   \
  --model_theta "/mnt/tidal-alsh-hilab/usr/shenguobin/modelscope/Qwen/Qwen3-1.7B"   \
  --dataset agentica-org/DeepScaleR-Preview-Dataset   \
  --batch_size 128   \
  --max_tokens 16384   \
  --output_dir ./outputs/q3_1.7b-os   \
  --resume   \
  --data_slice 0::8
```

bs 8, 15:37 -> 
bs 16 31:23
bs 32 58:43
bs 64 1:53:03 -> 0.566 sample / min
bs 128 1:53:03 -> 1.133 sample / min
bs 256 OOM

### 3. Research & Analysis

Study the behavior of optimal sampling:

```python
# Track alpha evolution
for i, alpha_hist in enumerate(outputs.alpha_values):
    plt.plot(alpha_hist, label=f"Example {i}")
plt.xlabel("Generation Step")
plt.ylabel("Alpha (Teacher Weight)")
plt.legend()
plt.show()
```

## ⚡ Performance Tips

### For Small Models (1.5B-3B)

```python
model = OptimalSamplingModel(
    model_theta_path="Qwen/Qwen2.5-1.5B-Instruct",
    model_t_path="Qwen/Qwen2.5-3B-Instruct",
    dtype=torch.bfloat16,  # Use bfloat16 for speed
    device="cuda"
)

outputs = model.generate(
    prompts=prompts,
    max_new_tokens=512,
    batch_size=16,  # Larger batch for small models
)
```

### For Large Models (7B-14B)

```python
from transformers import BitsAndBytesConfig

# Use 4-bit quantization
model = OptimalSamplingModel(
    model_theta_path="Qwen/Qwen2.5-7B-Instruct",
    model_t_path="Qwen/Qwen2.5-14B-Instruct",
    dtype=torch.bfloat16,
    load_in_4bit=True,  # 4-bit quantization
    bnb_4bit_compute_dtype=torch.bfloat16,
)

outputs = model.generate(
    prompts=prompts,
    max_new_tokens=512,
    batch_size=4,  # Smaller batch for large models
)
```

## 📈 Benchmark Results

Tested on Qwen-1.5B + Qwen-3B with 2 examples, 64 tokens:

| Implementation | Speed (tokens/s) | Init Time (s) | Total Time (s) |
|---------------|------------------|---------------|----------------|
| **HF + Flash Attn** | **36.71** | **7.00** | **10.49** |
| vLLM V1 | 5.70 | 17.04 | 39.50 |
| **Speedup** | **6.4x** | **2.4x** | **3.8x** |

See `optimal_sampling_package/scripts/WORK_REPORT_20251202.md` for detailed benchmarks.

## 🔬 Alpha Methods Explained

### KL Symmetry (Recommended)

Finds α that balances divergences from both models:
```
KL(q* || π_θ) ≈ KL(q* || π_t)
```

This ensures q\* is not too far from either model.

### ESS Balance

Balances effective sample sizes to maintain diversity from both distributions.

### Entropy

Maximizes entropy of q\* for maximum exploration.

## 🐛 Troubleshooting

### Flash Attention not working?

```bash
# Install flash-attn
pip install flash-attn --no-build-isolation

# Verify installation
python -c "import flash_attn; print('Flash Attention available')"
```

### Out of Memory?

- Reduce `max_new_tokens`
- Use smaller batch size
- Enable 4-bit quantization (`load_in_4bit=True`)
- Use `dtype=torch.float16` or `torch.bfloat16`

### Slow generation?

- Ensure Flash Attention 2 is installed
- Use `dtype=torch.bfloat16` instead of float32
- Increase batch size if memory allows

## 📚 Citation

If you use this implementation, please cite:

```bibtex
@software{optimal_sampling_hf,
  title = {Optimal Sampling HF: High-Performance Implementation with Flash Attention 2},
  year = {2024},
  url = {https://github.com/your-repo/optimal_sampling_hf}
}
```

## 📄 License

Apache License 2.0

## 🤝 Contributing

Contributions welcome! Please open an issue or PR.

## 🔗 Related Projects

- [optimal_sampling_package](../optimal_sampling_package) - vLLM V1 implementation
- [Flash Attention](https://github.com/Dao-AILab/flash-attention) - Efficient attention implementation

## ⭐ Star History

If you find this useful, please star the repo!

---

**Built with ❤️ for fast and efficient optimal sampling**
