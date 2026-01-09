# 数据集预处理

本目录包含 IS Reshape 实验所需数据集的预处理脚本。

## 数据集概览

### 训练集

- **DeepScaleR-Preview-Dataset** (`agentica-org/DeepScaleR-Preview-Dataset`)
  - 用途: 训练数据
  - 类型: 数学推理题目
  - 字段: problem, answer, solution

### 测试集

| 数据集 | HuggingFace 路径 | 类型 | 样本数 | 说明 |
|--------|-----------------|------|--------|------|
| **AMC 2023** | `math-ai/amc23` | 数学竞赛 | 40 | AMC 12A 2023 |
| **AIME 2024** | `HuggingFaceH4/aime_2024` | 数学竞赛 | 30 | AIME 2024 |
| **AIME 2025** | `math-ai/aime25` | 数学竞赛 | 30 | AIME 2025 |
| **MATH-500** | `HuggingFaceH4/MATH-500` | 数学题库 | 500 | MATH benchmark 子集 |
| **MinervaMath** | `math-ai/minervamath` | 数学题库 | 272 | 物理/天文题目 |
| **OlympiadBench** | `math-ai/olympiadbench` | 奥赛题库 | 674 | 国际数学奥林匹克题目 |
| **IFEval** | `google/IFEval` | 指令跟随 | 541 | 指令跟随评测 |

## 快速开始

### 1. 环境要求

```bash
pip install datasets
```

确保 verl 已正确安装（需要 `verl.utils.reward_score.math_reward` 模块）。

### 2. 处理所有数据集

```bash
cd /mnt/tidal-alsh-hilab/usr/shenguobin/verl/recipe/is_shape/code
python preprocess_datasets.py --output_dir ../data --datasets all
```

### 3. 处理单个数据集

```bash
# 只处理训练集
python preprocess_datasets.py --datasets deepscaler

# 只处理特定测试集
python preprocess_datasets.py --datasets amc23 aime_2024

# 自定义输出目录
python preprocess_datasets.py --datasets all --output_dir /path/to/output
```

## 输出格式

所有数据集处理后遵循统一格式：

```json
{
  "data_source": "数据集名称",
  "prompt": [
    {
      "role": "user",
      "content": "问题文本 + 指令"
    }
  ],
  "ability": "math" 或 "instruction_following",
  "reward_model": {
    "style": "rule",
    "ground_truth": "答案"
  },
  "extra_info": {
    "split": "train" 或 "test",
    "index": 样本索引,
    // 其他元数据...
  }
}
```

## 输出目录结构

```
data/
├── deepscaler/
│   ├── train.parquet          # 训练数据
│   └── train_example.json     # 示例（用于检查）
├── amc23/
│   ├── test.parquet
│   └── test_example.json
├── aime_2024/
│   ├── test.parquet
│   └── test_example.json
├── aime25/
│   ├── test.parquet
│   └── test_example.json
├── math500/
│   ├── test.parquet
│   └── test_example.json
├── minervamath/
│   ├── test.parquet
│   └── test_example.json
├── olympiadbench/
│   ├── test.parquet
│   └── test_example.json
└── ifeval/
    ├── test.parquet
    └── test_example.json
```

## 数据集详细信息

### 1. DeepScaleR-Preview-Dataset (训练集)

**原始字段**:
- `problem`: 问题描述
- `answer`: 答案（可能带 LaTeX 格式）
- `solution`: 解答步骤

**处理后**:
- 问题添加指令: "Let's think step by step and output the final answer within \\boxed{}."
- `answer` 字段直接作为 ground_truth
- 保留 solution 信息在 extra_info

### 2. AMC 2023 (测试集)

**原始字段**:
- `id`: 题目 ID
- `question`: 问题描述
- `answer`: 答案
- `url`: AoPS 链接

**特点**:
- AMC 12A 2023 的 40 道题目
- 答案为数字或字母

### 3. AIME 2024 (测试集)

**原始字段**:
- `id`: 题目 ID
- `problem`: 问题描述
- `solution`: 官方解答
- `answer`: 答案
- `url`: AoPS 链接
- `year`: 年份

**特点**:
- AIME 2024 的 30 道题目
- 难度较高，答案为 000-999 的整数

### 4. AIME 2025 (测试集)

**原始字段**:
- `problem`: 问题描述
- `answer`: 答案
- `id`: 题目 ID

**特点**:
- AIME 2025 的 30 道题目
- 最新年份数据

### 5. MATH-500 (测试集)

**原始字段**:
- `problem`: 问题描述
- `solution`: 详细解答
- `answer`: 答案
- `subject`: 学科类别（Algebra, Geometry, 等）
- `level`: 难度等级（1-5）
- `unique_id`: 唯一标识符

**特点**:
- MATH benchmark 的 500 题子集
- 覆盖多个数学学科
- 难度分级

### 6. IFEval (测试集)

**原始字段**:
- `key`: 样本 ID
- `prompt`: 指令
- `instruction_id_list`: 指令类型列表
- `kwargs`: 评测参数

**特点**:
- 指令跟随能力评测
- 不是数学题，而是各种指令执行任务
- reward_model 使用特殊的 "ifeval" 模式

### 7. MinervaMath (测试集)

**原始字段**:
- `question`: 问题描述
- `answer`: 答案

**特点**:
- 包含物理、天文等科学计算题目
- 272 道题目
- 答案通常为数值

### 8. OlympiadBench (测试集)

**原始字段**:
- `id`: 题目 ID
- `question`: 问题描述
- `solution`: 详细解答
- `final_answer`: 最终答案
- `context`: 上下文信息
- `image_*`: 图片字段（1-5）
- `difficulty`: 难度级别
- `subject`: 学科（Math, Physics 等）
- `subfield`: 子领域（Algebra, Geometry, Combinatorics 等）
- `question_type`: 题目类型
- `modality`: 模态（Text-only, With-image 等）
- `is_multiple_answer`: 是否多答案
- `answer_type`: 答案类型（Numerical, Expression 等）
- `unit`: 单位
- `language`: 语言

**特点**:
- 国际数学奥林匹克竞赛题目
- 674 道高难度题目
- 包含丰富的元数据（难度、学科、子领域等）
- 部分题目可能包含图片（当前预处理未处理图片）
- 提供详细的解答步骤

**处理后**:
- 如果有 context，会与 question 合并
- 使用 final_answer 作为 ground_truth
- 元数据保存在 extra_info 中，便于按难度/学科筛选

## 使用示例

### 加载处理后的数据

```python
from datasets import load_dataset

# 加载训练集
train_dataset = load_dataset(
    "parquet",
    data_files="../data/deepscaler/train.parquet"
)["train"]

# 加载测试集
test_amc23 = load_dataset(
    "parquet",
    data_files="../data/amc23/test.parquet"
)["train"]

# 查看样本
print(train_dataset[0])
```

### 在训练脚本中使用

```python
# 在 verl 训练脚本中
data_config = {
    "train": {
        "path": "recipe/is_shape/data/deepscaler/train.parquet",
        "format": "parquet"
    },
    "test": {
        "amc23": "recipe/is_shape/data/amc23/test.parquet",
        "aime_2024": "recipe/is_shape/data/aime_2024/test.parquet",
        "aime25": "recipe/is_shape/data/aime25/test.parquet",
        "math500": "recipe/is_shape/data/math500/test.parquet",
    }
}
```

## 注意事项

1. **数据集大小**: DeepScaleR 训练集可能较大，首次下载需要时间

2. **答案格式**:
   - 数学题答案可能包含 LaTeX 格式（如 `\\frac{2}{3}`）
   - 使用 `verl.utils.reward_score.math_reward` 进行答案匹配

3. **指令文本**:
   - 数学题自动添加 "Let's think step by step and output the final answer within \\boxed{}."
   - IFEval 保持原始 prompt

4. **分割信息**:
   - 原始数据集的 split 信息保存在 `extra_info.split`
   - 处理后统一为 train 或 test

## 扩展

如需添加新数据集，参考 `preprocess_datasets.py` 中的函数模板：

```python
def process_your_dataset(output_dir):
    """Process your custom dataset."""
    data_source = "your-org/your-dataset"
    dataset = datasets.load_dataset(data_source, split="train")

    def process_fn(example, idx):
        # 提取问题和答案
        question = example["question"]
        answer = example["answer"]

        # 构建统一格式
        data = {
            "data_source": data_source,
            "prompt": [{"role": "user", "content": question}],
            "ability": "math",
            "reward_model": {
                "style": "rule",
                "ground_truth": answer
            },
            "extra_info": {
                "split": "train",
                "index": idx,
                # 添加其他元数据
            }
        }
        return data

    processed = dataset.map(function=process_fn, with_indices=True)
    # 保存...
```

## 故障排除

### 问题: 下载数据集超时

```bash
# 设置 HuggingFace 镜像
export HF_ENDPOINT=https://hf-mirror.com
```

### 问题: 内存不足

```bash
# 单独处理每个数据集
python preprocess_datasets.py --datasets deepscaler
python preprocess_datasets.py --datasets amc23
# ...
```

### 问题: 找不到 verl 模块

```bash
# 确保在 verl 环境中
cd /path/to/verl
pip install -e .
```

## 联系

如有问题或建议，请提交 issue 或联系项目维护者。
