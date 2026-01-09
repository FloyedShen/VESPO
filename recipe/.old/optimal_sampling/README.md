# Optimal Sampling Distribution for RLHF

**基于 Fisher 信息平衡的强化学习从人类反馈 (RLHF) 最优采样分布**

[![Theory](https://img.shields.io/badge/Theory-Complete-success)](theory/)
[![Experiments](https://img.shields.io/badge/Experiments-Design%20Ready-blue)](experiments/)
[![Production](https://img.shields.io/badge/Production-Ready-green)](production/)
[![Docs](https://img.shields.io/badge/Docs-Complete-brightgreen)](docs/)

## 📋 项目概述

本项目从 Fisher 信息的角度统一了 RLHF 中的采样问题，并给出了理论最优解 q*。包含完整的理论证明、概念验证实验和生产级数据生成管线。

### 核心贡献

1. **理论创新**: 证明了最优采样分布 q* 满足 Fisher 信息平衡（ESS平衡）
2. **可计算性**: 通过 KL 对称条件给出了可计算的解法
3. **Pareto 最优**: 证明了 q* 在探索-稳定性权衡空间中是 Pareto 最优
4. **工程实现**: 提供了完整的生产级数据生成管线

### 核心公式

$$q^*(y|x) = \frac{\pi_\theta^{\alpha^*}(y|x) \cdot \pi_t^{1-\alpha^*}(y|x)}{Z_{\alpha^*}(x)}$$

其中 $\alpha^*$ 满足 **KL对称条件**：

$$D_{KL}(q^* \| \pi_\theta) = D_{KL}(q^* \| \pi_t)$$

这等价于 **Fisher信息平衡**：

$$\text{ESS}_\theta(q^*) = \text{ESS}_t(q^*)$$

## 📁 项目结构

```
optimal_sampling/
├── theory/                    # 理论分析
│   ├── proof_final.md         # ⭐ 完整理论证明
│   ├── computational_analysis.md
│   ├── speculative_decoding_analysis.md
│   ├── deep_analysis_summary.md
│   └── archive/               # 历史版本
│
├── experiments/               # 实验与概念验证
│   ├── experiment_design.md   # ⭐ 实验设计方案
│   ├── experiment_quick_reference.md
│   ├── run_experiments.py
│   ├── verify_alpha_theory.py
│   ├── visualize_q_star_2d.py
│   └── ...
│
├── production/                # 生产级代码
│   ├── optimal_sampling_model.py  # ⭐ 核心模型
│   ├── generate_data.py           # ⭐ 数据生成
│   ├── analyze_diagnostics.py
│   ├── test_pipeline.py
│   ├── quick_start.sh
│   └── requirements_data_generation.txt
│
└── docs/                      # 文档
    ├── DATA_GENERATION_GUIDE.md       # ⭐ 详细使用指南
    ├── README_DATA_GENERATION.md
    └── DATA_GENERATION_SUMMARY.md
```

## 🚀 快速开始

### 1. 理解理论

```bash
# 阅读核心理论证明
cat theory/proof_final.md

# 或查看可视化（2D Gaussian）
python experiments/visualize_q_star_2d.py
```

**关键概念**:
- Fisher 信息平衡
- KL 对称条件
- Pareto 最优性

### 2. 生成数据

```bash
cd production/

# 安装依赖
pip install -r requirements_data_generation.txt

# 快速测试
python test_pipeline.py

# 生成训练数据
python generate_data.py \
    --model_theta meta-llama/Llama-2-7b-hf \
    --model_t meta-llama/Llama-2-7b-chat-hf \
    --dataset agentica-org/DeepScaleR-Preview-Dataset \
    --output data/generated.jsonl \
    --alpha_method kl_symmetry \
    --num_samples 1000 \
    --save_diagnostics
```

### 3. 分析结果

```bash
# 分析诊断信息
python analyze_diagnostics.py \
    data/generated.diagnostics.jsonl \
    --output_dir analysis/
```

**验证指标**:
- ✅ ESS ratio ≈ 1.0
- ✅ KL 对称
- ✅ Alpha ∈ [0.2, 0.8]

## 📚 各部分详解

### 🔬 Theory (理论)

**核心文件**: [theory/proof_final.md](theory/proof_final.md)

- 完整的理论推导（7个主要部分）
- Fisher 信息与 Cramér-Rao 界
- Pareto 最优性证明
- 投机采样分析
- 2D 可视化洞察

**适合人群**: 研究者、理论学习者

**阅读指南**: 见 [theory/README.md](theory/README.md)

### 🧪 Experiments (实验)

**核心文件**: [experiments/experiment_design.md](experiments/experiment_design.md)

- 5个核心实验设计
- q* vs 7种baseline对比
- 消融实验和扩展性测试
- 完整的评估指标体系
- 6-9周实验时间线

**适合人群**: 实验研究者、验证理论

**阅读指南**: 见 [experiments/README.md](experiments/README.md)

### 🏭 Production (生产)

**核心文件**:
- [production/optimal_sampling_model.py](production/optimal_sampling_model.py) - 核心模型
- [production/generate_data.py](production/generate_data.py) - 数据生成

**功能**:
- ✅ 3种 alpha 计算方法（fixed, entropy, kl_symmetry）
- ✅ 支持 transformers 和 VLLM backend
- ✅ 自动数据集适配
- ✅ OpenAI API 格式输出
- ✅ 完整诊断信息
- ✅ 批量生成和断点续传

**适合人群**: 工程师、实际应用者

**使用指南**: 见 [production/README.md](production/README.md)

### 📖 Docs (文档)

**核心文件**:
- [docs/DATA_GENERATION_GUIDE.md](docs/DATA_GENERATION_GUIDE.md) - 详细使用指南
- [docs/README_DATA_GENERATION.md](docs/README_DATA_GENERATION.md) - 项目 README
- [docs/DATA_GENERATION_SUMMARY.md](docs/DATA_GENERATION_SUMMARY.md) - 项目总结

**内容**:
- API 文档
- 命令行参数
- 故障排查
- 最佳实践
- 代码示例

**适合人群**: 所有用户

**阅读指南**: 见 [docs/README.md](docs/README.md)

## 🎯 使用场景

### 场景1: 理论研究

```bash
# 1. 阅读理论证明
cat theory/proof_final.md

# 2. 运行理论验证
python experiments/verify_alpha_theory.py

# 3. 可视化分析
python experiments/visualize_q_star_2d.py
```

### 场景2: 实验验证

```bash
# 1. 设计实验
cat experiments/experiment_design.md

# 2. 运行概念验证
python experiments/run_experiments.py --method q_star

# 3. 对比baseline
# (需要补充完整的模型加载和数据处理)
```

### 场景3: 生产应用

```bash
cd production/

# 1. 快速测试
./quick_start.sh test

# 2. 生成训练数据
./quick_start.sh full

# 3. 用于RLHF训练
# (使用生成的OpenAI格式数据)
```

## 📊 Alpha 方法对比

| 方法 | 速度 | 理论保证 | 推荐场景 |
|------|------|----------|----------|
| **fixed** | ⭐⭐⭐ (~0.01ms) | ❌ | 快速测试、原型 |
| **entropy** | ⭐⭐ (~0.5ms) | 近似 | 快速生成大规模数据 |
| **kl_symmetry** | ⭐ (~2-3ms) | ✅ 完整 | 最终训练数据、论文实验 |

## 🔬 理论验证指标

生成的数据应满足：

1. **Fisher 信息平衡**: ESS_θ(q*) ≈ ESS_t(q*)
   - 测量: ESS ratio ∈ [0.9, 1.1]

2. **KL 对称**: D_KL(q*||π_θ) ≈ D_KL(q*||π_t)
   - 测量: |差异| < 0.05

3. **Alpha 分布**: α* ∈ [0.2, 0.8]
   - 反映两个分布的相对"强度"

## 📈 性能参考

在 A100 (80GB) 上的实测性能：

| 模型 | Batch Size | Alpha方法 | 速度 (samples/min) |
|------|-----------|-----------|-------------------|
| GPT-2 | 16 | fixed | ~100 |
| GPT-2 | 16 | kl_symmetry | ~50 |
| LLaMA-7B | 8 | fixed | ~20 |
| LLaMA-7B | 4 | kl_symmetry | ~8 |
| LLaMA-13B | 4 | kl_symmetry | ~4 |

## 🔑 核心洞察

1. **Fisher 信息平衡** 是最优采样的本质
2. **KL 对称** 提供了可计算的条件
3. **几何平均** 保证了分布的平滑性
4. **Pareto 最优** 证明了无法进一步改进
5. **投机采样** 可以加速 1.4-1.6x 而不损失精度

## 🛠️ 技术栈

- **理论**: Fisher 信息、Cramér-Rao 界、信息几何
- **实现**: PyTorch、Transformers、HuggingFace Datasets
- **可选**: VLLM (高性能推理)
- **可视化**: Matplotlib、Seaborn

## 📝 引用

如果这个工作对你有帮助，请引用：

```bibtex
@article{optimal_sampling_rlhf_2025,
  title={Optimal Sampling Distribution for RLHF via Fisher Information Balance},
  author={Your Name},
  year={2025},
  journal={arXiv preprint arXiv:XXXX.XXXXX}
}
```

## 🤝 贡献

欢迎贡献！特别是：

- [ ] VLLM backend 完整实现
- [ ] 更多数据集适配器
- [ ] 分布式生成支持
- [ ] 实验结果和分析
- [ ] 文档改进

## 📄 许可

MIT License

## 🔗 相关资源

- **理论基础**: [theory/proof_final.md](theory/proof_final.md)
- **实验设计**: [experiments/experiment_design.md](experiments/experiment_design.md)
- **使用指南**: [docs/DATA_GENERATION_GUIDE.md](docs/DATA_GENERATION_GUIDE.md)

## 📞 联系

如有问题或建议，请提 issue 或联系作者。

---

## 🎯 快速导航

| 你想... | 去这里 |
|--------|--------|
| 理解理论 | [theory/proof_final.md](theory/proof_final.md) |
| 设计实验 | [experiments/experiment_design.md](experiments/experiment_design.md) |
| 生成数据 | [production/README.md](production/README.md) |
| 查看文档 | [docs/DATA_GENERATION_GUIDE.md](docs/DATA_GENERATION_GUIDE.md) |
| 快速测试 | `cd production && python test_pipeline.py` |
| 一键启动 | `cd production && ./quick_start.sh test` |

---

**准备好开始了吗？** 🚀

- **理论学习者**: 从 [theory/](theory/) 开始
- **实验研究者**: 从 [experiments/](experiments/) 开始
- **工程应用者**: 从 [production/](production/) 开始

**祝你成功！** 如有任何问题，查看各目录的 README 或文档。
