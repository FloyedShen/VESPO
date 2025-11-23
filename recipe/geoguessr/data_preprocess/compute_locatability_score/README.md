# Locatability Score Computation Pipeline

为地理定位数据集（GAEA、OSV5M）计算图片的可定位性分数（locatability score）。

## 概述

本工具基于 GeoChain 的方法，通过语义分割分析图像内容，计算每张图片用于地理定位的难度：

- **输入**：HuggingFace Dataset 格式的地理定位数据集
- **输出**：添加了 `locatability_score` 和 `class_mapping` 字段的数据集
- **特性**：
  - ✅ 自动检测并使用多GPU并行处理
  - ✅ 支持断点续传（处理大数据集时可随时中断恢复）
  - ✅ 实时进度监控
  - ✅ 内存优化（分片处理）

## 方法原理

### Locatability Score 计算

```
1. 语义分割（Mask2Former + ADE20K 150类）
   输入图片 → 每个像素的类别标签

2. 统计类别占比
   percentage[类别] = 该类别像素数 / 总像素数

3. 加权求和
   score = Σ (weight[类别] × percentage[类别])
```

### 类别权重示例

| 类别 | 权重 | 说明 |
|------|------|------|
| road, route | 1.0 | 道路标识最重要 |
| signboard, sign | 0.567 | 标识牌很有价值 |
| building | 0.228 | 建筑风格有帮助 |
| car | 0.265 | 车辆类型有地域特征 |
| armchair | 0.0 | 室内家具无帮助 |

**Score 解读**：
- **0.6+**：容易定位（有明显地标、路标等）
- **0.3-0.6**：中等难度（有一些线索但不明显）
- **<0.3**：困难（室内场景或信息量少）

## 快速开始

### 1. 环境要求

```bash
# 已包含在 verl 环境中
pip install torch transformers datasets pillow tqdm
```

### 2. 处理数据集

**最简单的方式：使用批处理脚本**

```bash
cd /mnt/tidal-alsh-hilab/usr/shenguobin/verl/recipe/geoguessr/data_preprocess/compute_locatability_score

# 处理 GAEA 数据集（~1.4M样本）
./run_batch.sh --gaea

# 处理 OSV5M 数据集（~4.8M样本）
./run_batch.sh --osv5m

# 处理所有数据集
./run_batch.sh --all

# 查看所有选项
./run_batch.sh --help
```

### 3. 输出路径

处理后的数据集会保存到：

```
/mnt/tidal-alsh-hilab/usr/shenguobin/geoguessr/processed/
├── gaea_wlp/
│   ├── train/          # 带 locatability_score 的 GAEA train
│   └── bench/          # 带 locatability_score 的 GAEA bench
└── osv5m_wlp/
    ├── train/          # 带 locatability_score 的 OSV5M train
    └── test/           # 带 locatability_score 的 OSV5M test
```

`_wlp` = "with locatability and percentage"

## 高级用法

### 手动运行（更细粒度控制）

```bash
python3 scripts/run_parallel.py \
    --input_dataset /path/to/input/dataset \
    --output_dataset /path/to/output/dataset \
    --batch_size 20 \
    --num_workers 2 \
    --n_gpus 8
```

**参数说明**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--input_dataset` | 必需 | 输入数据集路径 |
| `--output_dataset` | 必需 | 输出数据集路径 |
| `--n_gpus` | 自动检测 | 使用的GPU数量 |
| `--batch_size` | 20 | 每个GPU的批处理大小 |
| `--num_workers` | 2 | 每个GPU的数据加载线程数 |
| `--checkpoint_interval` | 100 | 每N个batch保存一次断点 |
| `--no_amp` | False | 禁用混合精度（更慢但更稳定） |
| `--no_resume` | False | 不从断点恢复（重新开始） |

### 监控进度

```bash
# 查看单个GPU的日志
tail -f /path/to/output_temp/shard_0.log

# 监控所有GPU
watch -n 1 'tail -n 5 /path/to/output_temp/shard_*.log'
```

### 断点续传

如果处理中断（OOM、网络问题等），只需重新运行相同命令，会自动从断点继续：

```bash
# 第一次运行（中断）
./run_batch.sh --gaea

# 重新运行（自动恢复）
./run_batch.sh --gaea
```

强制重新处理：

```bash
./run_batch.sh --gaea --force --no_resume
```

## 项目结构

```
compute_locatability_score/
├── utils/                          # 工具模块
│   ├── __init__.py
│   ├── semantic_config.py          # 150类语义类别和权重
│   └── score_calculator.py         # 分数计算函数
├── scripts/                        # 核心脚本
│   ├── process_shard.py            # 单GPU处理脚本
│   └── run_parallel.py             # 多GPU调度器
├── run_batch.sh                    # 批处理启动脚本
└── README.md                       # 本文档
```

## 性能参考

在配置了 8×A100 (80GB) 的机器上：

| 数据集 | 样本数 | 处理时间（估计） | GPU显存占用 |
|--------|--------|-----------------|-------------|
| GAEA train | 1.4M | ~3-4 小时 | ~15GB/GPU |
| OSV5M train | 4.8M | ~12-15 小时 | ~15GB/GPU |

**优化建议**：
- 增加 `batch_size`（如果显存充足）
- 减少 `num_workers`（如果CPU瓶颈）
- 使用 `--no_amp` 如果遇到数值不稳定问题

## 使用计算好的数据

```python
import datasets

# 加载带 locatability score 的数据集
dataset = datasets.load_from_disk(
    "/mnt/tidal-alsh-hilab/usr/shenguobin/geoguessr/processed/gaea_wlp/train"
)

# 查看样本
sample = dataset[0]
print(f"Locatability score: {sample['locatability_score']:.4f}")
print(f"Class mapping: {sample['class_mapping'][:100]}...")

# 过滤数据：只保留中等难度的图片
medium_difficulty = dataset.filter(
    lambda x: 0.3 <= x['locatability_score'] <= 0.6
)

print(f"Medium difficulty samples: {len(medium_difficulty)}")
```

## 数据分析工具

创建分布统计：

```python
import json
import numpy as np

scores = [s['locatability_score'] for s in dataset]

print("Score distribution:")
print(f"  Min: {min(scores):.4f}")
print(f"  25%: {np.percentile(scores, 25):.4f}")
print(f"  50%: {np.percentile(scores, 50):.4f}")
print(f"  75%: {np.percentile(scores, 75):.4f}")
print(f"  Max: {max(scores):.4f}")
print(f"  Mean: {np.mean(scores):.4f}")

# 分析类别分布
class_mappings = [json.loads(s['class_mapping']) for s in dataset.select(range(1000))]
# ... 进一步分析
```

## 故障排查

### 问题1：OOM (Out of Memory)

**症状**：CUDA out of memory

**解决方案**：
```bash
# 减小batch size
./run_batch.sh --gaea --batch_size 10

# 或禁用AMP
./run_batch.sh --gaea --no_amp
```

### 问题2：进程卡住

**症状**：长时间无输出

**检查**：
```bash
# 查看GPU使用情况
nvidia-smi

# 查看进程日志
tail -f /path/to/output_temp/shard_*.log
```

### 问题3：合并失败

**症状**：All shards completed but merge failed

**解决方案**：
```bash
# 手动运行合并
python3 scripts/run_parallel.py \
    --input_dataset /path/to/input \
    --output_dataset /path/to/output \
    --temp_dir /path/to/output_temp \
    --skip_merge=false
```

## 下一步：数据过滤和课程学习

处理完成后，可以基于 locatability score 实现：

### 1. 难度分级

```python
# 简单：score > 0.6
easy_samples = dataset.filter(lambda x: x['locatability_score'] > 0.6)

# 中等：0.3 <= score <= 0.6
medium_samples = dataset.filter(lambda x: 0.3 <= x['locatability_score'] <= 0.6)

# 困难：score < 0.3
hard_samples = dataset.filter(lambda x: x['locatability_score'] < 0.3)
```

### 2. 课程学习

```python
# Stage 1: 训练初期，使用简单样本
train_data_stage1 = dataset.filter(lambda x: x['locatability_score'] > 0.5)

# Stage 2: 中期，加入中等难度
train_data_stage2 = dataset.filter(lambda x: x['locatability_score'] > 0.3)

# Stage 3: 后期，使用全部数据
train_data_stage3 = dataset
```

### 3. 混合采样

```python
from datasets import concatenate_datasets

# 按比例混合不同难度
mixed_dataset = concatenate_datasets([
    easy_samples.select(range(min(50000, len(easy_samples)))),    # 50k easy
    medium_samples.select(range(min(100000, len(medium_samples)))), # 100k medium
    hard_samples.select(range(min(30000, len(hard_samples))))     # 30k hard
])
```

## 参考

- GeoChain Paper: https://arxiv.org/abs/2506.00785
- Mask2Former: https://huggingface.co/facebook/mask2former-swin-large-ade-semantic
- ADE20K Dataset: https://groups.csail.mit.edu/vision/datasets/ADE20K/

## License

Apache License 2.0
