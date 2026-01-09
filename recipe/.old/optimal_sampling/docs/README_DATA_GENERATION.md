# 最优采样数据生成管线

基于最优采样分布 q* 理论的 RLHF 数据生成工具。

## 🎯 核心特性

- ✅ 支持 **3种Alpha计算方法**: fixed, kl_symmetry (理论最优), entropy (快速近似)
- ✅ 支持 **2种Backend**: transformers (完整实现), VLLM (规划中)
- ✅ **自动数据集适配**: 自动检测数据集格式，支持多种HuggingFace数据集
- ✅ **OpenAI API格式输出**: messages格式，可直接用于训练
- ✅ **完整诊断信息**: ESS ratio, KL散度, Alpha分布等
- ✅ **批量生成**: 支持大规模数据生成和断点续传

## 📦 安装

```bash
# 克隆仓库 (如果需要)
cd optimal_sampling

# 安装依赖
pip install -r requirements_data_generation.txt
```

## 🚀 快速开始

### 1. 测试管线

```bash
# 使用GPT-2快速测试
python test_pipeline.py
```

### 2. 生成数据

```bash
# 基础使用 - 固定alpha
python generate_data.py \
    --model_theta meta-llama/Llama-2-7b-hf \
    --dataset agentica-org/DeepScaleR-Preview-Dataset \
    --output data/generated.jsonl \
    --alpha_method fixed \
    --fixed_alpha 0.5 \
    --num_samples 1000 \
    --save_diagnostics

# 理论最优 - KL对称
python generate_data.py \
    --model_theta meta-llama/Llama-2-7b-hf \
    --model_t meta-llama/Llama-2-7b-chat-hf \
    --dataset agentica-org/DeepScaleR-Preview-Dataset \
    --output data/generated_optimal.jsonl \
    --alpha_method kl_symmetry \
    --num_samples 1000 \
    --batch_size 4 \
    --save_diagnostics
```

### 3. 分析结果

```bash
# 分析诊断信息
python analyze_diagnostics.py data/generated.diagnostics.jsonl --output_dir analysis/

# 对比不同方法
python analyze_diagnostics.py \
    data/generated_fixed.diagnostics.jsonl \
    data/generated_kl.diagnostics.jsonl \
    data/generated_entropy.diagnostics.jsonl
```

## 📚 文档

- **[完整使用指南](DATA_GENERATION_GUIDE.md)** - 详细的使用说明和最佳实践
- **[实验设计](experiment_design.md)** - 完整的实验方案
- **[理论证明](proof_final.md)** - q* 的理论推导

## 🏗️ 架构

```
optimal_sampling/
├── optimal_sampling_model.py     # 核心模型类
│   ├── AlphaComputer             # Alpha参数计算
│   ├── DiagnosticComputer        # 诊断信息
│   └── OptimalSamplingModel      # 主模型
│
├── generate_data.py              # 数据生成脚本
│   ├── DatasetAdapter            # 数据集适配器
│   └── DataGenerator             # 数据生成器
│
├── analyze_diagnostics.py        # 诊断分析工具
├── test_pipeline.py              # 测试脚本
└── DATA_GENERATION_GUIDE.md      # 使用指南
```

## 🔬 Alpha计算方法对比

| 方法 | 速度 | 理论保证 | 推荐场景 |
|------|------|----------|----------|
| **fixed** | ⭐⭐⭐ | ❌ | 快速测试 |
| **entropy** | ⭐⭐ | 近似 | 快速生成大规模数据 |
| **kl_symmetry** | ⭐ | ✅ 完整 | 最终训练数据 |

## 📊 输出格式

### 主数据文件 (`.jsonl`)

```json
{
  "messages": [
    {"role": "user", "content": "What is AI?"},
    {"role": "assistant", "content": "AI is..."}
  ],
  "sample_idx": 0
}
```

### 诊断文件 (`.diagnostics.jsonl`)

```json
{
  "sample_idx": 0,
  "alpha_mean": 0.523,
  "alpha_std": 0.045,
  "ess_ratio_mean": 0.987,
  "ess_ratio_std": 0.112,
  "kl_theta_mean": 0.234,
  "kl_t_mean": 0.231
}
```

## 🎯 理论验证指标

生成的数据应满足：

1. **ESS Ratio ≈ 1.0** (在 [0.8, 1.2] 范围内)
   - 表示Fisher信息平衡

2. **Alpha ∈ [0.2, 0.8]**
   - 过于极端的alpha可能表示分布不匹配

3. **KL对称** (仅kl_symmetry方法)
   - D_KL(q*||π_θ) ≈ D_KL(q*||π_t)
   - 差异应 < 0.05

## 💡 使用技巧

1. **先用小模型测试**: 使用 `gpt2` 或 `--num_samples 10` 验证管线
2. **监控诊断信息**: 使用 `--save_diagnostics` 跟踪ESS ratio
3. **批量大小调整**: 根据GPU内存调整 `--batch_size`
4. **断点续传**: 使用 `--start_idx` 从中断处继续

## 🐛 常见问题

**Q: CUDA out of memory?**
```bash
--batch_size 2 --dtype float16
```

**Q: 生成速度慢?**
```bash
--alpha_method entropy --batch_size 16
```

**Q: 数据集格式不兼容?**
```bash
--dataset_adapter generic --prompt_field your_field
```

详见 [DATA_GENERATION_GUIDE.md](DATA_GENERATION_GUIDE.md)

## 📈 性能参考

在 A100 (80GB) 上的性能参考：

| 模型 | Batch Size | Alpha方法 | 速度 (samples/min) |
|------|-----------|-----------|-------------------|
| GPT-2 | 16 | fixed | ~100 |
| LLaMA-7B | 8 | fixed | ~20 |
| LLaMA-7B | 4 | kl_symmetry | ~8 |
| LLaMA-13B | 4 | kl_symmetry | ~4 |

## 🤝 贡献

欢迎提出问题和改进建议！

## 📝 引用

如果这个工作对你有帮助，请引用：

```bibtex
@article{optimal_sampling_rlhf,
  title={Optimal Sampling Distribution for RLHF via Fisher Information Balance},
  author={Your Name},
  year={2024}
}
```

## 📄 许可

MIT License

---

**准备好开始了吗？** 运行 `python test_pipeline.py` 进行快速测试！ 🚀
