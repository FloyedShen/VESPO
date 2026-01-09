# vLLM V1 Dual-Engine Optimal Sampling

High-performance optimal sampling using two synchronized vLLM engines. Achieves **O(n) complexity** instead of O(n²).

## 📁 Directory Structure

```
vllm_v1_dual_impl/
├── README.md                    # This file
├── __init__.py                  # Package exports
│
├── Core Implementation
│   ├── optimal_sampling_dual.py # Main user interface
│   ├── sync_state_v2.py         # Multiprocessing synchronization
│   ├── alpha_computer.py        # Alpha computation methods
│   └── proxy_storage.py         # File-based proxy storage
│
├── LogitsProcessors
│   ├── sync_processor_base.py   # Base class (NEW!)
│   ├── sync_processor_a.py      # Engine A (Teacher)
│   └── sync_processor_b.py      # Engine B (Theta)
│
├── docs/
│   └── README.md                # Full documentation
│
├── examples/
│   └── verify_benchmark.py      # Verification benchmark
│
└── .old/
    ├── sync_state.py            # Old threading version
    ├── global_registry.py       # Old registry approach
    └── verify_benchmark.log     # Old log file
```

## 🚀 Quick Start

```python
from production.vllm_v1_dual_impl import OptimalSamplingDual

# Initialize
sampler = OptimalSamplingDual(
    model_teacher="Qwen/Qwen2.5-7B",
    model_theta="Qwen/Qwen2.5-1.5B",
    alpha_method="kl_symmetry",
    sync_timeout=5.0
)

# Generate
outputs = sampler.generate(
    prompts=["What is artificial intelligence?"],
    max_tokens=2048
)

print(outputs.generated_texts[0])
print(f"Sync stats: {sampler.get_sync_statistics()}")
```

## 📊 Performance

| Sequence Length | Original (O(n²)) | Dual-Engine (O(n)) | Speedup |
|-----------------|------------------|--------------------|---------|
| 512 tokens      | 10 sec           | 5 sec              | **2x**  |
| 2048 tokens     | 160 sec          | 20 sec             | **8x**  |
| 8192 tokens     | 2.5 hours        | 47 min             | **3.2x**|

## 📖 Documentation

- **Full Documentation**: See [docs/README.md](docs/README.md)
- **Examples**: See [examples/](examples/)
- **Deprecated Code**: See [.old/](.old/)

## 🆕 What's New in v2.1.0

- ✅ **Unified Base Class**: `BaseSyncLogitsProcessor` eliminates code duplication
- ✅ **Simplified Implementation**: Engine A/B processors now only ~40 lines each (was ~350 lines)
- ✅ **Better Organization**: Docs, examples, and deprecated code properly separated
- ✅ **Multiprocessing Support**: Works with vLLM V1's spawn mode (v2.0)
- ✅ **Zero Race Conditions**: Reference counting prevents premature barrier deletion (v2.0)

## 🔧 Key Features

- **O(n) Complexity**: Both models maintain O(n) instead of O(n²)
- **Full KV Cache**: Both engines benefit from KV caching
- **Parallel Sync**: Thread pool for batch synchronization
- **Timeout Protection**: Graceful degradation on sync failures
- **4 Alpha Methods**: fixed, kl_symmetry, ess_balance, entropy
- **Statistics Tracking**: Comprehensive sync and alpha statistics

## 📝 Version History

- **v2.1.0** (Current): Added BaseSyncLogitsProcessor, improved organization
- **v2.0.0**: Multiprocessing support, race condition fix
- **v1.0.0**: Initial threading-based implementation

## 🤝 Contributing

When adding new features:
1. Update the base class if common functionality
2. Keep docs/ and examples/ synchronized
3. Move deprecated code to .old/
4. Update version in `__init__.py`

---

**For full documentation, see [docs/README.md](docs/README.md)**
