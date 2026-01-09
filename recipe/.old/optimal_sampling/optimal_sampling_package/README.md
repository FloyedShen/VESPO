# Optimal Sampling for vLLM V1

高性能的最优采样实现，基于 vLLM V1 引擎，专为 semi on-policy 蒸馏和高质量数据生成设计。

## 🌟 核心特性

- **真正的 KV Cache 复用**: Teacher 模型享受 vLLM 完整的 KV cache 优化
- **灵活的前缀控制**: 支持两个模型使用不同的 chat template 和 system prompt
- **高效的批处理**: 原生支持 vLLM V1 的批处理能力
- **Semi On-Policy 蒸馏**: 专为从不同前缀采样高质量 on-policy 数据设计
- **数学推理优化**: 特别适合需要标准答案引导的场景

## 🎯 设计目标：Semi On-Policy Distillation

在数学推理等复杂任务中，小模型直接学习 `p(·|x)` 通常很困难。我们的方法：

1. **条件生成**: 使用 `p(·|x, y*, r)`
   - `x`: 原始问题
   - `y*`: 大模型/Oracle 生成的标准答案
   - `r`: 引导 prompt

2. **减少 Off-Policy Gap**: 通过从不同前缀采样，生成高质量且 on-policy 的数据

3. **灵活的模型组合**: Teacher 和 Theta 模型可以使用完全不同的提示策略

## 📦 安装

```bash
# 基础安装
pip install -e .

# 开发模式
pip install -e ".[dev]"
```

## 🚀 快速开始

### 基础用法

```python
from optimal_sampling import OptimalSamplingV1

# 初始化采样器
sampler = OptimalSamplingV1(
    model_teacher="Qwen/Qwen2.5-32B-Instruct",
    model_theta="Qwen/Qwen2.5-7B-Instruct",
    alpha_method="kl_symmetry",
    gpu_memory_utilization=0.5,
)

# 生成
outputs = sampler.generate(
    prompts=["Solve: 2x + 3 = 7"],
    max_tokens=512,
    temperature=0.8
)

print(outputs.generated_texts[0])
```

### Semi On-Policy Distillation 示例（推荐）

**关键设计：Teacher 和 Student 接收不同的输入！**

```python
from optimal_sampling import OptimalSamplingV1

# 初始化采样器
sampler = OptimalSamplingV1(
    model_teacher="Qwen/Qwen2.5-32B-Instruct",  # Oracle
    model_theta="Qwen/Qwen2.5-7B-Instruct",      # Student

    # Teacher: 给定标准答案，生成推理
    teacher_system_prompt=(
        "Given the problem and correct answer, "
        "generate detailed reasoning steps."
    ),

    # Student: 学习从问题直接推理（不能看答案！）
    theta_system_prompt="You are a math problem solver.",

    enable_chat_template=False,
    alpha_method="kl_symmetry",
)

# 准备不同的 prompts
problem = "Solve: 2x + 3 = 7"
oracle_answer = "x = 2"

# Teacher 看到答案（条件生成）
teacher_prompt = f"Problem: {problem}\nAnswer: {oracle_answer}\nReasoning:"

# Student 不能看答案（学习直接推理）
student_prompt = f"Problem: {problem}\nReasoning:"

# 生成：Optimal mixing 平衡质量和 on-policy
outputs = sampler.generate(
    prompts=[teacher_prompt],        # Teacher 接收
    theta_prompts=[student_prompt],  # Student 接收 ✅
    max_tokens=512,
    temperature=0.8
)

# 结果：高质量的 on-policy 推理数据
print(outputs.generated_texts[0])
```

**为什么这样设计？**
- ✅ Teacher 有答案引导 → 生成高质量推理
- ✅ Student 不看答案 → 保持 on-policy 分布
- ✅ Optimal mixing (α) 平衡两者
- ✅ 结果：比纯 off-policy 数据的 gap 小很多

## 📚 文档

- [安装指南](docs/installation.md)
- [使用说明](docs/usage.md)
- [Semi On-Policy Distillation 详解](docs/distillation.md)
- [API 文档](docs/api.md)

## 🔧 配置

查看 `configs/` 目录获取预定义配置：

- `base.yaml`: 基础配置
- `distillation.yaml`: 蒸馏场景配置
- `math_qa.yaml`: 数学问答优化配置

## 📊 性能基准测试

### 重度负载测试结果 (Qwen 3B + 1.5B)

| 配置 | 批量大小 | Max Tokens | 吞吐量 | 延迟 | 推荐场景 |
|------|---------|-----------|--------|------|---------|
| ⭐ **推荐** | 16 | 512 | 138.81 tok/s | 2.37s | 平衡配置 |
| 🚀 **最佳** | 32 | 512 | **152.68 tok/s** | 2.30s | 最大吞吐 |
| 📝 长推理 | 16 | 1024 | 119.47 tok/s | 3.93s | 详细推理 |

**关键发现**:
- ✅ 100% 成功率，稳定可靠
- ✅ 批量处理效率高（BS=32 最优）
- ✅ 冷启动后热启动快 4.5x
- ⚠️ 比 Teacher-only 慢 20-30x（运行两个模型）

**详细报告**:
- 📄 [完整性能分析报告](BENCHMARK_REPORT.md)
- 📄 [快速总结](BENCHMARK_SUMMARY.md)
- 📊 [原始数据](heavy_benchmark_results.json)

## 🔬 理论基础

基于 Cramér-Rao 下界和 KL 散度对称性的最优 α 计算：

```
D_KL(q* || π_θ) = D_KL(q* || π_t)
```

其中 `q*(y|x) = π_θ(y|x)^(1-α) × π_t(y|x)^α / Z`

详见 `theory/proof_final.md`


## 🙏 致谢

基于 vLLM V1 LogitsProcessor API 构建。
