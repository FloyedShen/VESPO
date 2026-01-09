# Dual VLLM Quick Reference

## 🚀 Quick Start (3 Steps)

### 1. Start vLLM Instances

```bash
# Terminal 1: Base model
python -m vllm.entrypoints.api_server \
    --model meta-llama/Llama-2-7b-hf \
    --port 8000

# Terminal 2: Teacher model
python -m vllm.entrypoints.api_server \
    --model meta-llama/Llama-2-7b-chat-hf \
    --port 8001

# Or use the quick start script:
./start_vllm.sh
```

### 2. Install Dependencies

```bash
cd production/dual_vllm
pip install -r requirements.txt
```

### 3. Generate

```python
import asyncio
from dual_vllm import generate_with_optimal_sampling

async def main():
    results = await generate_with_optimal_sampling(
        prompts=["What is the capital of France?"],
        theta_url="http://localhost:8000",
        t_url="http://localhost:8001",
        max_tokens=50
    )
    print(results[0].generated_text)

asyncio.run(main())
```

## 📊 Key Metrics to Monitor

```python
result = results[0]

# 1. Alpha values (should be 0.3-0.7 for similar models)
alpha_mean = np.mean(result.alpha_history)
print(f"Alpha: {alpha_mean:.3f}")

# 2. KL symmetry error (should be < 1e-4)
if result.diagnostics:
    kl_diff = result.diagnostics['kl_diff_mean']
    print(f"KL error: {kl_diff:.6f}")  # Want: < 1e-4

# 3. ESS ratio (should be ≈ 1.0)
    ess_ratio = result.diagnostics['ess_ratio_mean']
    print(f"ESS ratio: {ess_ratio:.3f}")  # Want: 0.9-1.1
```

## ⚙️ Configuration Cheatsheet

```python
from dual_vllm import CoordinatorConfig

# Minimal config
config = CoordinatorConfig(
    theta_url="http://localhost:8000",
    t_url="http://localhost:8001"
)

# High-performance config
config = CoordinatorConfig(
    theta_url="http://localhost:8000",
    t_url="http://localhost:8001",
    top_k=100,              # Higher = more accurate
    connection_pool_size=100,  # For concurrent requests
)

# High-precision config
config = CoordinatorConfig(
    theta_url="http://localhost:8000",
    t_url="http://localhost:8001",
    top_k=200,              # More candidates
    alpha_tol=1e-8,         # Tighter convergence
    alpha_max_iter=30,      # More iterations
)
```

## 🔧 Common Patterns

### Pattern: Batch Processing

```python
async with DualVLLMCoordinator(config) as coordinator:
    results = await coordinator.generate_batch(
        prompts=[...],  # Your prompts
        max_tokens=100,
        show_progress=True
    )
```

### Pattern: With Diagnostics

```python
results = await coordinator.generate_batch(
    prompts=[...],
    return_diagnostics=True  # Get detailed metrics
)
```

### Pattern: Error Handling

```python
for result in results:
    if result.error:
        logger.error(f"Failed: {result.error}")
    else:
        process(result.generated_text)
```

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| Connection refused | Check vLLM instances: `curl http://localhost:8000/health` |
| Slow generation | Increase `connection_pool_size`, reduce `top_k` |
| High memory | Use vLLM quantization: `--quantization awq` |
| NaN alphas | Check if distributions are identical |

## 📈 Performance Tuning

```python
# For speed (acceptable accuracy loss)
config = CoordinatorConfig(
    top_k=50,               # Faster (90% coverage)
    alpha_tol=1e-4,         # Looser tolerance
    alpha_max_iter=10,      # Fewer iterations
)

# For accuracy (slower)
config = CoordinatorConfig(
    top_k=200,              # Better coverage
    alpha_tol=1e-8,         # Tighter tolerance
    alpha_max_iter=30,      # More iterations
)
```

## 📝 Files Overview

```
dual_vllm/
├── coordinator.py       # Main logic - start here
├── utils.py            # Math functions
├── config.py           # Configuration
├── example.py          # Usage examples
├── test_dual_vllm.py   # Unit tests
└── README.md           # Full docs
```

## 🔗 Next Steps

1. **Run examples**: `python example.py`
2. **Run tests**: `pytest test_dual_vllm.py -v`
3. **Read theory**: `../../theory/proof_final.md`
4. **Integration**: See `README.md` section "Integration with Training"
