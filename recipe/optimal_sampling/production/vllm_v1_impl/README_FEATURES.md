# Optimal Sampling V1 - Feature Update

## 🎯 Major Improvements

### 1. **Model Role Swap** ✅
**Architecture Change**: Teacher (larger model) now runs in outer vLLM, Theta (smaller model) in inner.

**Before**:
```
Base Model (θ, smaller) → Outer vLLM
Guide Model (t, larger) → Inner vLLM
```

**After**:
```
Teacher Model (t, larger) → Outer vLLM (benefits from KV cache!)
Theta Model (θ, smaller) → Inner vLLM
```

**Benefits**:
- **Better KV Cache Utilization**: The larger teacher model fully benefits from vLLM's KV cache optimization
- **More Efficient**: Most computational cost is on the teacher model, which is now optimized
- **Faster Generation**: Reduced redundant computation on the larger model

### 2. **Different System Prompts** ✅
Support different system prompts for teacher and theta models.

**Usage**:
```python
sampler = OptimalSamplingV1(
    model_teacher="Qwen/Qwen2.5-1.5B",
    model_theta="Qwen/Qwen2.5-0.5B",
    teacher_system_prompt="You are a highly knowledgeable AI expert.",
    theta_system_prompt="You are a simple and concise assistant.",
)
```

**Benefits**:
- **Flexible Behavior**: Different models can have different personalities/behaviors
- **Specialized Roles**: Teacher can be verbose, theta can be concise
- **Better Control**: Fine-tune each model's output style independently

### 3. **Chat Template Support** ✅
Automatically apply chat templates to format prompts.

**Usage**:
```python
sampler = OptimalSamplingV1(
    model_teacher="Qwen/Qwen2.5-1.5B",
    model_theta="Qwen/Qwen2.5-0.5B",
    teacher_system_prompt="You are a helpful assistant.",
    theta_system_prompt="You are a helpful assistant.",
    enable_chat_template=True,  # Enable chat template
)
```

**Benefits**:
- **Proper Formatting**: Uses model's native chat template (e.g., ChatML, Llama format)
- **Better Quality**: Models trained with chat templates work better with proper formatting
- **Automatic**: No manual template formatting needed

### 4. **Alpha Statistics Tracking** ✅
Track and analyze alpha values during generation.

**Usage**:
```python
sampler = OptimalSamplingV1(
    model_teacher="Qwen/Qwen2.5-1.5B",
    model_theta="Qwen/Qwen2.5-0.5B",
    track_alpha_stats=True,  # Enable alpha tracking
)

outputs = sampler.generate(prompts=["..."], max_tokens=100)

# Access alpha statistics
print(f"Mean alpha: {outputs.alpha_stats['mean']:.4f}")
print(f"Std alpha:  {outputs.alpha_stats['std']:.4f}")
print(f"Min alpha:  {outputs.alpha_stats['min']:.4f}")
print(f"Max alpha:  {outputs.alpha_stats['max']:.4f}")
print(f"Alpha history: {outputs.alpha_stats['history']}")
```

**Statistics Provided**:
- `mean`: Average alpha across all generation steps
- `std`: Standard deviation of alpha
- `min`: Minimum alpha value
- `max`: Maximum alpha value
- `count`: Number of generation steps
- `history`: Full alpha value sequence

**Benefits**:
- **Analysis**: Understand how alpha changes during generation
- **Debugging**: Identify if alpha computation is working correctly
- **Insights**: See when teacher vs theta is more influential

## 📋 Complete API Reference

### OptimalSamplingV1

```python
sampler = OptimalSamplingV1(
    model_teacher: str,              # Path to teacher model (larger, outer)
    model_theta: str,                # Path to theta model (smaller, inner)
    alpha_method: str = "kl_symmetry",  # Alpha computation method
    fixed_alpha: float = 0.5,        # Fixed alpha (if method="fixed")
    alpha_min: float = 0.5,          # Minimum alpha (teacher weight)
    alpha_max: float = 1.0,          # Maximum alpha (teacher weight)
    teacher_system_prompt: Optional[str] = None,  # Teacher system prompt
    theta_system_prompt: Optional[str] = None,    # Theta system prompt
    enable_chat_template: bool = False,  # Enable chat template formatting
    track_alpha_stats: bool = True,  # Track alpha statistics
    gpu_memory_utilization: float = 0.5,  # GPU memory for teacher
    **kwargs  # Additional vLLM arguments
)
```

### Generation

```python
outputs = sampler.generate(
    prompts: List[str],              # Input prompts (raw text)
    max_tokens: int = 100,           # Maximum tokens to generate
    temperature: float = 1.0,        # Sampling temperature
    top_p: float = 1.0,              # Nucleus sampling
    top_k: int = -1,                 # Top-k sampling
    use_optimal_sampling: bool = True  # Enable/disable optimal sampling
)
```

### Output

```python
@dataclass
class OptimalSamplingOutput:
    generated_texts: List[str]       # Generated text sequences
    generated_ids: List[List[int]]   # Token IDs
    num_tokens: List[int]            # Number of tokens per sequence
    alpha_stats: Optional[Dict[str, float]]  # Alpha statistics (if enabled)
```

## 🎨 Usage Examples

### Example 1: Basic Usage with New Architecture
```python
from production.vllm_v1_impl import OptimalSamplingV1

sampler = OptimalSamplingV1(
    model_teacher="Qwen/Qwen2.5-1.5B",  # Larger model (outer)
    model_theta="Qwen/Qwen2.5-0.5B",     # Smaller model (inner)
    alpha_method="kl_symmetry"
)

outputs = sampler.generate(
    prompts=["What is AI?"],
    max_tokens=100
)

print(outputs.generated_texts[0])
```

### Example 2: Different System Prompts
```python
sampler = OptimalSamplingV1(
    model_teacher="Qwen/Qwen2.5-1.5B",
    model_theta="Qwen/Qwen2.5-0.5B",
    teacher_system_prompt="You are an expert AI researcher.",
    theta_system_prompt="You are a beginner-friendly tutor."
)

outputs = sampler.generate(prompts=["Explain transformers."], max_tokens=100)
```

### Example 3: Chat Template
```python
sampler = OptimalSamplingV1(
    model_teacher="Qwen/Qwen2.5-1.5B",
    model_theta="Qwen/Qwen2.5-0.5B",
    teacher_system_prompt="You are a helpful assistant.",
    enable_chat_template=True  # Automatically format with chat template
)

outputs = sampler.generate(prompts=["Hello!"], max_tokens=50)
```

### Example 4: Alpha Statistics
```python
sampler = OptimalSamplingV1(
    model_teacher="Qwen/Qwen2.5-1.5B",
    model_theta="Qwen/Qwen2.5-0.5B",
    track_alpha_stats=True
)

outputs = sampler.generate(prompts=["Explain ML."], max_tokens=100)

# Analyze alpha values
stats = outputs.alpha_stats
print(f"Average teacher weight: {stats['mean']:.3f}")
print(f"Alpha varied from {stats['min']:.3f} to {stats['max']:.3f}")
```

### Example 5: All Features Combined
```python
sampler = OptimalSamplingV1(
    model_teacher="Qwen/Qwen2.5-1.5B",     # Larger model (outer, KV cache optimized)
    model_theta="Qwen/Qwen2.5-0.5B",       # Smaller model (inner)
    alpha_method="kl_symmetry",
    teacher_system_prompt="You are a knowledgeable expert.",
    theta_system_prompt="You are a concise assistant.",
    enable_chat_template=True,
    track_alpha_stats=True,
    gpu_memory_utilization=0.4
)

outputs = sampler.generate(
    prompts=["What is deep learning?", "Explain neural networks."],
    max_tokens=100,
    temperature=0.8
)

for i, text in enumerate(outputs.generated_texts):
    print(f"Output {i+1}: {text}")

if outputs.alpha_stats:
    print(f"\nAlpha statistics: {outputs.alpha_stats}")
```

## 🔬 Testing

Run the comprehensive test suite:

```bash
python test_optimal_sampling_v1_new.py
```

Tests include:
1. ✅ Model role swap verification
2. ✅ Different system prompts
3. ✅ Chat template support
4. ✅ Alpha statistics tracking
5. ✅ Batch generation
6. ✅ Comparison with/without optimal sampling

## 🚀 Performance Benefits

### KV Cache Optimization
The new architecture maximizes KV cache benefits:

- **Teacher model (outer)**: Full KV cache reuse → Faster generation
- **Theta model (inner)**: Still benefits from prefix caching
- **Overall**: Significant speedup for longer sequences

### Memory Efficiency
- Teacher model runs in outer vLLM with optimized memory management
- Theta model uses smaller memory footprint in inner instance
- Both models can use independent GPU memory allocations

## 📊 Alpha Statistics Use Cases

### 1. Understanding Model Behavior
Track when the teacher vs theta model is more influential:
- High alpha (→1.0): Teacher model dominates
- Low alpha (→0.5): More balanced mixing

### 2. Quality Analysis
Correlate alpha values with output quality:
- Stable alpha: Consistent model agreement
- Varying alpha: Models disagree, adaptive mixing

### 3. Debugging
Verify alpha computation:
- Check if alpha is in expected range [alpha_min, alpha_max]
- Ensure alpha changes appropriately with different methods

### 4. Research
Analyze optimal sampling behavior:
- How does alpha evolve during generation?
- Does alpha correlate with token difficulty?
- Which alpha method works best for your use case?

## 🔄 Migration Guide

### From Old API (v0) to New API (v1)

**Old API**:
```python
sampler = OptimalSamplingV1(
    model_base="Qwen/Qwen2.5-0.5B",   # Smaller (was outer)
    model_guide="Qwen/Qwen2.5-1.5B",  # Larger (was inner)
)
```

**New API**:
```python
sampler = OptimalSamplingV1(
    model_teacher="Qwen/Qwen2.5-1.5B",  # Larger (now outer!) ⭐
    model_theta="Qwen/Qwen2.5-0.5B",     # Smaller (now inner)
    teacher_system_prompt="...",         # NEW ⭐
    theta_system_prompt="...",           # NEW ⭐
    enable_chat_template=True,           # NEW ⭐
    track_alpha_stats=True,              # NEW ⭐
)
```

**Key Changes**:
1. `model_base` → `model_teacher` (now the LARGER model)
2. `model_guide` → `model_theta` (now the SMALLER model)
3. Added `teacher_system_prompt` and `theta_system_prompt`
4. Added `enable_chat_template`
5. Added `track_alpha_stats`
6. Output now includes `alpha_stats`

## 📝 Notes

### Token ID Consistency
- Teacher model tokenizes the input prompt
- Theta model receives the SAME token IDs for consistency
- System prompts are applied to original text prompts, not token IDs

### Chat Template
- When enabled, applies the model's native chat template
- Includes system message if system prompt is provided
- Automatically adds generation prompt

### Alpha Statistics
- Tracked per request during generation
- Statistics computed when generation completes
- Minimal overhead (~negligible performance impact)

## 🎯 Recommended Configuration

For best results:

```python
sampler = OptimalSamplingV1(
    model_teacher="larger-model",      # e.g., Qwen2.5-7B
    model_theta="smaller-model",       # e.g., Qwen2.5-1.5B
    alpha_method="kl_symmetry",        # Recommended for balance
    alpha_min=0.5,                     # Minimum teacher weight
    alpha_max=1.0,                     # Maximum teacher weight
    teacher_system_prompt="Detailed and informative responses.",
    theta_system_prompt="Concise and clear responses.",
    enable_chat_template=True,         # If models support it
    track_alpha_stats=True,            # For analysis
    gpu_memory_utilization=0.5         # Adjust based on VRAM
)
```

## 🙏 Acknowledgments

All features implemented following vLLM V1 API best practices for maximum performance and compatibility.
