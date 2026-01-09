# 文档 (Documentation)

本目录包含数据生成管线的完整使用文档。

## 📚 文档清单

### 主要文档

**[DATA_GENERATION_GUIDE.md](DATA_GENERATION_GUIDE.md)** ⭐ **详细使用指南**
- 完整的使用说明
- API 文档
- 命令行参数详解
- 实验建议
- 故障排查
- 代码示例
- 最佳实践

**[README_DATA_GENERATION.md](README_DATA_GENERATION.md)** ⭐ **项目 README**
- 项目概述
- 核心特性
- 快速开始
- 架构说明
- 性能参考
- 常见问题
- 使用示例

**[DATA_GENERATION_SUMMARY.md](DATA_GENERATION_SUMMARY.md)** ⭐ **项目总结**
- 项目完成情况
- 文件清单（12个核心文件）
- 使用流程
- 技术特点
- 理论验证
- 输出格式
- 功能检查清单

## 📖 阅读指南

### 快速入门
1. 先读 **README_DATA_GENERATION.md** 了解项目概况
2. 运行 `python test_pipeline.py` 测试环境
3. 参考 **DATA_GENERATION_GUIDE.md** 开始生成数据

### 深入使用
1. 阅读 **DATA_GENERATION_GUIDE.md** 的"详细文档"部分
2. 了解不同 alpha 方法的区别
3. 学习自定义数据集适配器
4. 掌握故障排查技巧

### 项目总览
1. 阅读 **DATA_GENERATION_SUMMARY.md** 了解完整项目
2. 查看文件清单和功能完整性
3. 了解已实现和待完成的功能

## 🎯 核心概念

### 最优采样分布 q*

$$q^*(y|x) = \frac{\pi_\theta^{\alpha^*}(y|x) \cdot \pi_t^{1-\alpha^*}(y|x)}{Z_{\alpha^*}(x)}$$

- $\pi_\theta$: 当前策略模型
- $\pi_t$: 目标策略模型
- $\alpha^*$: 最优混合参数（通过KL对称求解）

### Alpha 计算方法

1. **fixed**: 固定值（如 0.5）
   - 优点: 最快
   - 缺点: 非最优
   - 用途: 快速测试

2. **entropy**: 熵公式近似
   - 公式: $\alpha \approx H(\pi_\theta) / [H(\pi_\theta) + H(\pi_t)]$
   - 优点: 快速，合理近似
   - 用途: 快速生成大规模数据

3. **kl_symmetry**: KL对称条件（理论最优）
   - 条件: $D_{KL}(q^* \| \pi_\theta) = D_{KL}(q^* \| \pi_t)$
   - 优点: 理论最优，Fisher信息平衡
   - 用途: 最终训练数据

### 理论验证指标

1. **ESS Ratio ≈ 1.0**
   - 定义: $\text{ESS}_\theta / \text{ESS}_t$
   - 期望: ∈ [0.9, 1.1]
   - 意义: Fisher信息平衡

2. **KL对称** (kl_symmetry方法)
   - 定义: $|D_{KL}(q^* \| \pi_\theta) - D_{KL}(q^* \| \pi_t)|$
   - 期望: < 0.05
   - 意义: 最优混合

3. **Alpha分布**
   - 期望: ∈ [0.2, 0.8]
   - 意义: 反映分布相对强度

## 🚀 快速开始命令

### 测试管线
```bash
python production/test_pipeline.py
```

### 生成数据（基础）
```bash
python production/generate_data.py \
    --model_theta meta-llama/Llama-2-7b-hf \
    --dataset agentica-org/DeepScaleR-Preview-Dataset \
    --output data/generated.jsonl \
    --alpha_method kl_symmetry \
    --num_samples 1000 \
    --save_diagnostics
```

### 快速启动
```bash
cd production/
./quick_start.sh test   # 测试
./quick_start.sh small  # 小规模
./quick_start.sh full   # 完整运行
```

### 分析结果
```bash
python production/analyze_diagnostics.py \
    data/generated.diagnostics.jsonl \
    --output_dir analysis/
```

## 📊 输出说明

### 主数据文件格式

```jsonl
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}], "sample_idx": 0}
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}], "sample_idx": 1}
...
```

符合 OpenAI fine-tuning API 格式，可直接用于训练。

### 诊断信息格式

```jsonl
{"sample_idx": 0, "alpha_mean": 0.523, "ess_ratio_mean": 0.987, "kl_theta_mean": 0.234, "kl_t_mean": 0.231}
{"sample_idx": 1, "alpha_mean": 0.518, "ess_ratio_mean": 0.993, "kl_theta_mean": 0.229, "kl_t_mean": 0.226}
...
```

用于验证理论条件和监控生成质量。

## 🔧 常见任务

### 任务1: 快速验证管线

```bash
# 1. 测试
python production/test_pipeline.py

# 2. 小规模生成（10个样本）
python production/generate_data.py \
    --model_theta gpt2 \
    --dataset agentica-org/DeepScaleR-Preview-Dataset \
    --output data/test.jsonl \
    --num_samples 10
```

### 任务2: 生成训练数据

```bash
# 使用理论最优方法生成1000个样本
python production/generate_data.py \
    --model_theta meta-llama/Llama-2-7b-hf \
    --model_t meta-llama/Llama-2-7b-chat-hf \
    --dataset agentica-org/DeepScaleR-Preview-Dataset \
    --output data/train_optimal.jsonl \
    --alpha_method kl_symmetry \
    --num_samples 1000 \
    --batch_size 4 \
    --save_diagnostics
```

### 任务3: 对比不同方法

```bash
cd production/
./quick_start.sh full
```

这将自动生成3种方法的数据并对比分析。

### 任务4: 验证数据质量

```bash
# 分析诊断信息
python production/analyze_diagnostics.py \
    data/train_optimal.diagnostics.jsonl \
    --output_dir analysis/

# 检查以下指标:
# - ESS ratio 是否接近1
# - Alpha 是否在合理范围
# - KL散度是否对称
```

## 💡 使用技巧

### 提高生成速度

1. **使用 entropy 方法**: 比 kl_symmetry 快 3-4x
2. **增大 batch size**: 在内存允许的情况下
3. **减少 max_new_tokens**: 如果不需要长响应
4. **使用更小的模型**: GPT-2 用于快速测试

### 提高生成质量

1. **使用 kl_symmetry 方法**: 理论最优
2. **使用两个不同的模型**: π_θ (base) 和 π_t (chat)
3. **调整 temperature**: 默认1.0，可以调节探索度
4. **监控诊断信息**: ESS ratio 应该接近1

### 调试技巧

1. **先用小数据测试**: `--num_samples 10`
2. **检查诊断文件**: 看alpha和ESS ratio是否正常
3. **降低batch size**: 如果遇到OOM
4. **使用float16**: 节省内存

## 📈 性能优化

### 内存优化

```bash
# 小GPU（<16GB）
--batch_size 2 --dtype float16 --model_theta gpt2

# 中等GPU（16-40GB）
--batch_size 4 --dtype float16 --model_theta meta-llama/Llama-2-7b-hf

# 大GPU（40GB+）
--batch_size 8 --dtype float16 --model_theta meta-llama/Llama-2-13b-hf
```

### 速度优化

```bash
# 快速模式（牺牲理论精度）
--alpha_method entropy --batch_size 16

# 平衡模式
--alpha_method entropy --batch_size 8

# 精确模式（理论最优）
--alpha_method kl_symmetry --batch_size 4
```

## 🐛 故障排查速查表

| 问题 | 解决方案 |
|------|---------|
| CUDA OOM | 减小 `--batch_size` 和 `--max_new_tokens` |
| 速度慢 | 使用 `--alpha_method entropy` |
| Alpha异常 | 检查两个模型是否差异过大 |
| ESS ratio远离1 | 使用 `kl_symmetry` 方法 |
| 数据集不兼容 | 使用 `--dataset_adapter generic` 并手动指定字段 |

详细排查见 **DATA_GENERATION_GUIDE.md** 的"故障排查"部分。

## 🔗 相关链接

### 内部链接
- [理论基础](../theory/proof_final.md)
- [实验设计](../experiments/experiment_design.md)
- [生产代码](../production/)

### 外部资源
- [Transformers 文档](https://huggingface.co/docs/transformers)
- [HuggingFace Datasets](https://huggingface.co/docs/datasets)
- [OpenAI Fine-tuning API](https://platform.openai.com/docs/guides/fine-tuning)

## 📝 更新日志

### 2025-01
- ✅ 完成数据生成管线
- ✅ 实现3种alpha计算方法
- ✅ 添加诊断分析工具
- ✅ 完善文档

### 待完成
- [ ] VLLM backend 完整实现
- [ ] 更多数据集适配器
- [ ] 分布式生成支持

## ❓ 常见问题

**Q: 应该使用哪种alpha方法？**
- 测试: `fixed`
- 快速生成: `entropy`
- 最终数据: `kl_symmetry`

**Q: 为什么ESS ratio不是1？**
- `fixed` 和 `entropy` 只是近似，不保证ESS平衡
- `kl_symmetry` 应该接近1，如果不是，检查迭代收敛

**Q: 可以使用VLLM加速吗？**
- 目前VLLM backend未完全实现
- 需要自行实现token级采样控制

**Q: 生成的数据如何用于训练？**
- 输出是标准OpenAI格式，可直接用于fine-tuning
- 或转换为其他框架需要的格式

更多问题见各文档的FAQ部分。

---

**文档完整性**: ✅ 所有核心功能都有详细文档

**准备好了吗？** 从 **README_DATA_GENERATION.md** 开始！ 🚀
