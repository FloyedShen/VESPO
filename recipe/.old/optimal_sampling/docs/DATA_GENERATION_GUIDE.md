# 数据生成管线使用指南

## 📋 概述

本数据生成管线实现了基于最优采样分布 q* 的数据生成功能，用于 RLHF 训练数据的准备。

## 🏗️ 架构

```
optimal_sampling_model.py    # 核心模型类
    ├── AlphaComputer         # Alpha参数计算器
    ├── DiagnosticComputer    # 诊断信息计算器
    └── OptimalSamplingModel  # 主模型类

generate_data.py              # 数据生成脚本
    ├── DatasetAdapter        # 数据集适配器基类
    ├── DeepScaleRAdapter     # DeepScaleR数据集适配器
    └── DataGenerator         # 数据生成器
```

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install torch transformers datasets tqdm
# 可选: VLLM支持
pip install vllm
```

### 2. 基础使用

```bash
# 使用相同模型 (π_θ = π_t), 固定alpha
python generate_data.py \
    --model_theta meta-llama/Llama-2-7b-hf \
    --dataset agentica-org/DeepScaleR-Preview-Dataset \
    --output data/generated_fixed_alpha.jsonl \
    --alpha_method fixed \
    --fixed_alpha 0.5 \
    --num_samples 1000 \
    --batch_size 8 \
    --save_diagnostics

# 使用不同模型, KL对称方法
python generate_data.py \
    --model_theta meta-llama/Llama-2-7b-hf \
    --model_t meta-llama/Llama-2-7b-chat-hf \
    --dataset agentica-org/DeepScaleR-Preview-Dataset \
    --output data/generated_kl_symmetry.jsonl \
    --alpha_method kl_symmetry \
    --num_samples 1000 \
    --batch_size 4 \
    --save_diagnostics

# 使用熵公式 (快速近似)
python generate_data.py \
    --model_theta meta-llama/Llama-2-7b-hf \
    --model_t meta-llama/Llama-2-7b-chat-hf \
    --dataset agentica-org/DeepScaleR-Preview-Dataset \
    --output data/generated_entropy.jsonl \
    --alpha_method entropy \
    --num_samples 1000 \
    --batch_size 8 \
    --save_diagnostics
```

### 3. 输出格式

生成的数据为 JSONL 格式，每行一个样本，符合 OpenAI API 的 messages 格式：

```json
{
  "messages": [
    {
      "role": "user",
      "content": "What is the capital of France?"
    },
    {
      "role": "assistant",
      "content": "The capital of France is Paris..."
    }
  ],
  "sample_idx": 0,
  "original_field1": "...",
  "original_field2": "..."
}
```

诊断信息文件 (`.diagnostics.jsonl`):

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

## 📚 详细文档

### OptimalSamplingModel 类

#### 初始化参数

```python
model = OptimalSamplingModel(
    model_theta_path="meta-llama/Llama-2-7b-hf",  # π_θ 模型路径
    model_t_path="meta-llama/Llama-2-7b-chat-hf", # π_t 模型路径 (可选)
    backend="transformers",                        # "transformers" 或 "vllm"
    alpha_method="kl_symmetry",                   # "fixed", "kl_symmetry", "entropy"
    fixed_alpha=0.5,                              # 固定alpha值
    alpha_tol=1e-6,                               # KL对称求解容差
    device="cuda",                                 # 设备
    dtype=torch.float16                           # 数据类型
)
```

#### Alpha 计算方法

1. **fixed**: 固定alpha值
   - 最快，不需要计算
   - 适合快速测试
   - 参数: `fixed_alpha`

2. **kl_symmetry**: KL对称条件 (理论最优)
   - 求解 D_KL(q*||π_θ) = D_KL(q*||π_t)
   - 二分法，20次迭代
   - 每个token约 2-3ms
   - **推荐用于最终训练数据**

3. **entropy**: 熵公式快速近似
   - α ≈ H(π_θ) / [H(π_θ) + H(π_t)]
   - 最快，无迭代
   - 近似精度较高
   - 适合需要速度的场景

#### 生成方法

```python
outputs = model.generate(
    prompts=["Hello, how are you?"],  # 输入prompts
    max_new_tokens=100,                # 最大生成token数
    temperature=1.0,                   # 采样温度
    top_p=1.0,                         # nucleus sampling
    top_k=-1,                          # top-k sampling
    return_diagnostics=True            # 返回诊断信息
)

# 输出
print(outputs.generated_texts)       # 生成的文本
print(outputs.alpha_values)          # [batch, seq_len]
print(outputs.ess_ratios)            # [batch, seq_len]
print(outputs.diagnostics)           # Dict[str, Tensor]
```

### 数据集适配器

#### DeepScaleRAdapter (自动检测格式)

```python
adapter = DeepScaleRAdapter(
    dataset_name="agentica-org/DeepScaleR-Preview-Dataset",
    split="train"
)

# 自动检测以下字段:
# - prompt: prompt, question, instruction, input, text
# - response: response, answer, output, completion
# - messages: messages (OpenAI格式)
```

#### GenericAdapter (通用)

```python
adapter = GenericAdapter(
    dataset_name="your/dataset",
    split="train",
    prompt_field="question",      # 指定prompt字段
    response_field="answer"       # 指定response字段
)
```

### 命令行参数完整列表

```bash
# 模型参数
--model_theta PATH              # π_θ 模型路径 (必需)
--model_t PATH                  # π_t 模型路径 (可选)
--backend {transformers,vllm}   # Backend选择

# Alpha方法
--alpha_method {fixed,kl_symmetry,entropy}  # Alpha计算方法
--fixed_alpha FLOAT             # 固定alpha值 (默认0.5)

# 数据集
--dataset NAME                  # HuggingFace数据集名称 (必需)
--dataset_split SPLIT           # 数据集split (默认train)
--dataset_adapter {auto,deepscaler,generic}
--prompt_field FIELD            # Prompt字段名
--response_field FIELD          # Response字段名

# 生成参数
--num_samples INT               # 生成样本数 (默认全部)
--start_idx INT                 # 起始索引 (用于断点续传)
--batch_size INT                # Batch大小 (默认8)
--max_new_tokens INT            # 最大生成token数 (默认512)
--temperature FLOAT             # 采样温度 (默认1.0)

# 输出
--output PATH                   # 输出文件路径 (必需)
--save_diagnostics              # 保存诊断信息

# 设备
--device DEVICE                 # 设备 (默认cuda)
--dtype {float16,bfloat16,float32}  # 数据类型 (默认float16)
```

## 🔬 实验建议

### 1. 对比不同Alpha方法

```bash
# 生成3种方法的数据进行对比
for method in fixed kl_symmetry entropy; do
    python generate_data.py \
        --model_theta meta-llama/Llama-2-7b-hf \
        --model_t meta-llama/Llama-2-7b-chat-hf \
        --dataset agentica-org/DeepScaleR-Preview-Dataset \
        --output data/generated_${method}.jsonl \
        --alpha_method $method \
        --num_samples 1000 \
        --save_diagnostics
done

# 分析诊断信息
python analyze_diagnostics.py data/generated_*.diagnostics.jsonl
```

### 2. 不同Alpha固定值

```bash
# 测试不同的固定alpha值
for alpha in 0.3 0.5 0.7; do
    python generate_data.py \
        --model_theta meta-llama/Llama-2-7b-hf \
        --model_t meta-llama/Llama-2-7b-chat-hf \
        --dataset agentica-org/DeepScaleR-Preview-Dataset \
        --output data/generated_alpha_${alpha}.jsonl \
        --alpha_method fixed \
        --fixed_alpha $alpha \
        --num_samples 1000 \
        --save_diagnostics
done
```

### 3. 断点续传

```bash
# 如果生成中断, 可以从指定位置继续
python generate_data.py \
    --model_theta meta-llama/Llama-2-7b-hf \
    --model_t meta-llama/Llama-2-7b-chat-hf \
    --dataset agentica-org/DeepScaleR-Preview-Dataset \
    --output data/generated.jsonl \
    --alpha_method kl_symmetry \
    --start_idx 500 \
    --num_samples 1000 \
    --save_diagnostics
```

## 📊 性能优化建议

### 1. Batch Size 调整

```bash
# 小模型 (GPT-2, Llama-7B) - 使用较大batch
--batch_size 16

# 大模型 (Llama-13B, 30B) - 使用较小batch
--batch_size 4

# 根据GPU内存调整
```

### 2. 数据类型选择

```bash
# A100/H100 - 使用 bfloat16 (更稳定)
--dtype bfloat16

# V100/其他 - 使用 float16 (更快)
--dtype float16

# 调试/小模型 - 使用 float32
--dtype float32
```

### 3. Alpha方法选择

| 方法 | 速度 | 精度 | 推荐场景 |
|------|------|------|----------|
| fixed | ⭐⭐⭐ | ⭐ | 快速测试 |
| entropy | ⭐⭐ | ⭐⭐ | 快速生成 |
| kl_symmetry | ⭐ | ⭐⭐⭐ | 最终训练数据 |

## 🐛 故障排查

### 问题1: CUDA out of memory

**解决方案**:
```bash
# 减小batch size
--batch_size 2

# 减小max_new_tokens
--max_new_tokens 256

# 使用float16
--dtype float16
```

### 问题2: 数据集格式不兼容

**解决方案**:
```bash
# 使用generic adapter并手动指定字段
--dataset_adapter generic \
--prompt_field your_prompt_field \
--response_field your_response_field
```

### 问题3: 生成速度慢

**解决方案**:
```bash
# 使用entropy方法 (更快)
--alpha_method entropy

# 增大batch size
--batch_size 16

# 考虑使用VLLM (注意: 当前VLLM不支持完整q*采样)
--backend vllm  # 需要自行实现近似方法
```

## 📝 代码示例

### Python API 使用

```python
from optimal_sampling_model import create_optimal_sampling_model

# 创建模型
model = create_optimal_sampling_model(
    model_theta="meta-llama/Llama-2-7b-hf",
    model_t="meta-llama/Llama-2-7b-chat-hf",
    alpha_method="kl_symmetry"
)

# 生成
outputs = model.generate(
    prompts=["What is AI?", "Explain quantum computing"],
    max_new_tokens=100,
    temperature=1.0
)

# 查看结果
for i, (text, alpha, ess) in enumerate(zip(
    outputs.generated_texts,
    outputs.alpha_values,
    outputs.ess_ratios
)):
    print(f"\n=== Sample {i+1} ===")
    print(f"Text: {text}")
    print(f"Alpha (mean): {alpha.mean():.3f}")
    print(f"ESS ratio (mean): {ess.mean():.3f}")
```

### 自定义数据集适配器

```python
from generate_data import DatasetAdapter

class MyCustomAdapter(DatasetAdapter):
    def get_prompt(self, idx: int) -> str:
        sample = self.dataset[idx]
        # 自定义prompt提取逻辑
        return sample["my_custom_field"]

    def get_metadata(self, idx: int) -> dict:
        sample = self.dataset[idx]
        return {
            "sample_idx": idx,
            "custom_meta": sample.get("meta", {})
        }

# 使用
from generate_data import DataGenerator
from optimal_sampling_model import create_optimal_sampling_model

model = create_optimal_sampling_model(...)
adapter = MyCustomAdapter("my/dataset", "train")
generator = DataGenerator(model, adapter, "output.jsonl")
generator.generate(num_samples=1000)
```

## 🎯 最佳实践

1. **先用小数据集测试**: 用 `--num_samples 10` 验证管线正常工作
2. **使用诊断信息**: 启用 `--save_diagnostics` 监控ESS ratio
3. **检查alpha分布**: alpha应该在 [0.2, 0.8] 范围内，过于极端可能有问题
4. **批量实验**: 使用脚本批量测试不同参数组合
5. **保存checkpoint**: 对于大规模生成，定期检查输出文件

## 📚 相关文件

- `optimal_sampling_model.py` - 核心模型实现
- `generate_data.py` - 数据生成脚本
- `experiment_design.md` - 实验设计方案
- `proof_final.md` - 理论证明

## 🤝 贡献

如果发现bug或有改进建议，欢迎提issue或PR。

---

**准备好开始生成数据了吗？** 🚀
