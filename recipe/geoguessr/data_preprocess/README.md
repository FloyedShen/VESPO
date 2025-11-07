# GeoGuessr 数据集预处理与 verl 训练指南

本文档完整说明如何将 GeoGuessr 数据集（OSV5M, GAEA, GeoChain）预处理并转换为 verl RLHF 训练格式，使用基于经纬度距离的 reward 进行强化学习训练。

## 📋 目录

- [整体流程](#整体流程)
- [环境配置](#环境配置)
- [文件说明](#文件说明)
- [快速开始](#快速开始)
- [Step 1: 数据预处理](#step-1-数据预处理)
- [Step 2: 转换为 verl 格式](#step-2-转换为-verl-格式)
- [Step 3: 验证转换结果](#step-3-验证转换结果)
- [统一数据格式](#统一数据格式)
- [verl 格式数据结构](#verl-格式数据结构)
- [Reward 计算详解](#reward-计算详解)
- [在 verl 中使用](#在-verl-中使用)
- [数据集统计](#数据集统计)
- [最佳实践](#最佳实践)
- [故障排查](#故障排查)

---

## 整体流程

```
原始数据集 (OSV5M, GAEA, GeoChain)
    ↓
Step 1: 预处理 (preprocess_*.py)
    ├── 加载原始数据
    ├── 调用逆地理编码服务（Nominatim）
    ├── 统一字段格式
    └── 保存为 HuggingFace Dataset
    ↓
统一格式数据 (HuggingFace Dataset)
    ↓
Step 2: 转换为 verl 格式 (convert_to_verl_format.py)
    ├── 构建 prompt (system + user)
    ├── 配置 reward_model
    ├── 添加 tools_kwargs
    └── 保存为 Parquet 文件（支持分块）
    ↓
verl RLHF 训练数据 (Parquet)
    ↓
Step 3: 训练 (RLHFDataset + verl trainer)
```

---

## 环境配置

### 设置环境变量

```bash
export GEOGUESSR_DIR=/path/to/your/geoguessr/data
```

### 目录结构

```
$GEOGUESSR_DIR/
├── osv5m/osv5m/          # OSV5M 数据集
│   ├── train.csv
│   ├── test.csv
│   └── images/
├── vistas/               # Vistas 数据集（用于 GeoChain）
└── processed/            # 处理后的数据集
    ├── gaea/
    │   ├── train/        # HuggingFace Dataset 格式
    │   └── bench/
    ├── geochain/
    │   ├── test/
    │   └── mini_test/
    ├── osv5m/
    │   ├── train/
    │   └── test/
    └── verl_format/      # verl 训练格式
        ├── gaea_train.parquet
        ├── gaea_bench.parquet
        ├── osv5m_train_chunk_0000.parquet
        ├── osv5m_train_chunk_0001.parquet
        └── ...
```

### 逆地理编码服务

所有预处理脚本都需要 Nominatim 服务来获取地址信息：

```bash
# 使用 Docker 启动 Nominatim
docker run -it --rm \
  -e PBF_URL=https://download.geofabrik.de/planet-latest.osm.pbf \
  -p 8080:8080 \
  mediagis/nominatim:5.1
```

---

## 文件说明

| 文件                          | 说明                                       |
|-----------------------------|------------------------------------------|
| `preprocess_gaea.py`        | GAEA 数据集预处理（从 HuggingFace 加载）            |
| `preprocess_geochain.py`    | GeoChain 数据集预处理（从 HuggingFace 加载）        |
| `preprocess_osv5m.py`       | OSV5M 数据集预处理（从本地 CSV 加载）                 |
| `convert_to_verl_format.py` | 转换为 verl RLHF 训练格式（⭐ 支持分块）               |
| `reward_calculator.py`      | Reward 计算工具（Haversine 距离 + 多种 reward 函数） |
| `test_verl_format.py`       | 验证转换后的数据格式                               |
| `quick_start.py`            | 快速测试整个流程                                 |
| `README.md`                 | 本文档                                      |

---

## 快速开始

```bash
# 1. 设置环境变量
export GEOGUESSR_DIR=/path/to/data

# 2. 运行快速测试（可选）
python quick_start.py

# 3. 预处理数据集
python preprocess_osv5m.py
python preprocess_gaea.py
python preprocess_geochain.py

# 4. 转换为 verl 格式
# 小数据集：不分块
python convert_to_verl_format.py --single_dataset gaea/train

# 大数据集：分块（推荐）
python convert_to_verl_format.py --single_dataset osv5m/train --chunk_size 50000

# 5. 验证转换结果
python test_verl_format.py $GEOGUESSR_DIR/processed/verl_format/gaea_train.parquet
```

---

## Step 1: 数据预处理

将原始数据集预处理为统一格式。

### 运行预处理脚本

```bash
# 服务 URL（根据你的 Nominatim 服务地址修改）
SERVICE_URL="http://localhost:8080"

# 预处理 OSV5M
python preprocess_osv5m.py

# 预处理 GAEA
python preprocess_gaea.py

# 预处理 GeoChain
python preprocess_geochain.py
```

### 输出

- **位置**: `$GEOGUESSR_DIR/processed/{osv5m,gaea,geochain}/{train,test}/`
- **格式**: HuggingFace Dataset，包含 image, lat, lon, 地址信息等

### 特点

- ✅ 支持断点续传（`resume=True`）
- ✅ 分块处理大数据集（`chunk_size`）
- ✅ 并行处理（`num_workers`）
- ✅ 自动跳过损坏的图像

---

## Step 2: 转换为 verl 格式

将统一格式转换为 verl RLHF 训练格式。

### 基础用法

```bash
# 转换所有数据集
python convert_to_verl_format.py

# 只转换特定数据集
python convert_to_verl_format.py --single_dataset gaea/train

# 限制样本数（用于快速测试）
python convert_to_verl_format.py --single_dataset gaea/train --max_samples 1000
```

### ⭐ 分块模式（推荐用于大数据集）

**为什么使用分块？**
- OSV5M train 有 ~450 万样本，单个 parquet 文件会超过 100GB
- 分块后可以更灵活地选择训练数据子集
- RLHFDataset **原生支持多个文件**，无需额外处理
- 降低内存压力

**使用方法：**

```bash
# 将 OSV5M train 拆分成每个 50K 样本的文件
python convert_to_verl_format.py \
    --single_dataset osv5m/train \
    --chunk_size 50000

# 输出文件：
# osv5m_train_chunk_0000.parquet (50K samples)
# osv5m_train_chunk_0001.parquet (50K samples)
# osv5m_train_chunk_0002.parquet (50K samples)
# ...
# 总共约 90 个文件
```

**推荐的分块策略：**

| 数据集 | 样本数 | 推荐 chunk_size | 预计文件数 | 原因 |
|--------|--------|-----------------|------------|------|
| GAEA train | ~150K | 不分块 | 1 | 数据量小 |
| GAEA bench | ~15K | 不分块 | 1 | 数据量小 |
| GeoChain test | ~60K | 不分块 | 1 | 数据量小 |
| OSV5M train | ~4.5M | 50000 | ~90 | 数据量大 ⭐ |
| OSV5M test | ~500K | 50000 | ~10 | 数据量中等 |

### 其他选项

```bash
# 在 prompt 中包含地址提示（用于有提示的训练）
python convert_to_verl_format.py --include_address_hints

# 添加自定义输出格式指令
python convert_to_verl_format.py \
    --instruction_following "Please provide coordinates in format: latitude: XX.XXX, longitude: YY.YYY"

# 组合使用
python convert_to_verl_format.py \
    --single_dataset osv5m/train \
    --chunk_size 50000 \
    --include_address_hints \
    --instruction_following "Provide precise coordinates."
```

### 输出

- **位置**: `$GEOGUESSR_DIR/processed/verl_format/`
- **格式**: Parquet 文件，包含 prompt, images, reward_model, tools_kwargs 等

---

## Step 3: 验证转换结果

```bash
# 验证单个文件
python test_verl_format.py $GEOGUESSR_DIR/processed/verl_format/gaea_train.parquet

# 验证 chunk 文件
python test_verl_format.py $GEOGUESSR_DIR/processed/verl_format/osv5m_train_chunk_0000.parquet
```

验证脚本会检查：
- ✅ 文件可以正常加载
- ✅ 所有必需字段存在
- ✅ prompt 结构正确（system + user）
- ✅ images 是列表格式
- ✅ reward_model 配置正确
- ✅ tools_kwargs 结构正确

---

## 统一数据格式

预处理后的数据集具有统一的字段格式。

### 核心字段（所有数据集）

```python
{
    'image': PIL.Image,           # 图像
    'lat': float,                 # 纬度
    'lon': float,                 # 经度
    'image_source': str,          # 图像来源 ('mapillary', 'vistas', etc.)
    'source': str,                # 数据集来源 ('gaea', 'geochain', 'osv5m')
    'messages': str/dict,         # JSON 格式的对话（GAEA 有，其他为空）

    # Nominatim 逆地理编码字段
    'road': str,
    'suburb': str,
    'ISO3166-2-lvl10': str,
    'city': str,
    'postcode': str,
    'country': str,
}
```

### 数据集特有字段

**GAEA:**
```python
{
    'question_type': str,         # 问题类型
    'subset': str,                # 子集名称
}
```

**GeoChain:**
```python
{
    'locatability_score': float,  # 可定位性分数
    'class_mapping': str,         # 类别映射
}
```

**OSV5M:**
```python
{
    'osv5m_country': str,         # 原始国家代码
    'osv5m_region': str,
    'osv5m_sub_region': str,
    'osv5m_city': str,

    # 环境特征
    'land_cover': float,
    'road_index': float,
    'drive_side': float,
    'climate': float,
    'soil': float,
    'dist_sea': float,

    # 元数据
    'captured_at': str,
    'sequence': str,
    'thumb_original_url': str,
}
```

---

## verl 格式数据结构

转换后的每个样本包含以下字段：

```python
{
    "data_source": "osv5m",       # 数据来源 ('osv5m', 'gaea', 'geochain')

    "prompt": [                   # OpenAI chat 格式
        {
            "role": "system",
            "content": "You are an expert in geography and image analysis..."
        },
        {
            "role": "user",
            "content": "Where was this photo taken? Please predict the latitude and longitude."
        }
    ],

    "images": [PIL.Image],        # ⭐ 图像列表（注意是复数）

    "ability": "geolocation",     # 任务类型

    "reward_model": {             # Reward 配置
        "style": "rule",
        "ground_truth": {"lat": 40.7128, "lon": -74.0060}
    },

    "extra_info": {
        "split": "train",         # 'train' 或 'test'
        "index": 0,               # 样本索引
        "answer": {"lat": 40.7128, "lon": -74.0060},
        "image_source": "mapillary",

        # ⭐ Tools 配置
        "need_tools_kwargs": True,
        "tools_kwargs": {
            "calc_geoguessr_reward": {
                "create_kwargs": {
                    "ground_truth": {"lat": 40.7128, "lon": -74.0060},
                    "reward_type": "exponential"  # 可选
                }
            }
        },

        # 地址信息（供参考）
        "address": {
            "country": "united states",
            "city": "new york",
            "road": "broadway"
        },

        # 数据集特定字段
        # ...
    }
}
```

---

## Reward 计算详解

基于预测坐标与真实坐标的距离计算 reward。

### 距离计算

使用 **Haversine 公式**计算球面距离（单位：公里）：

```python
from reward_calculator import haversine_distance

# 纽约到旧金山的距离
distance = haversine_distance(
    lat1=40.7128, lon1=-74.0060,   # 纽约
    lat2=37.7749, lon2=-122.4194   # 旧金山
)
# 输出: 约 4130 km
```

### Reward 函数

支持三种 reward 类型：

#### 1. **Exponential**（推荐）

```python
reward = exp(-distance / 1000)
```

**特点**: 指数衰减，类似 GeoGuessr 官方评分

**距离-Reward 对应表**:

| 距离 (km) | Reward |
|-----------|--------|
| 0 | 1.0 |
| 100 | 0.90 |
| 500 | 0.61 |
| 1000 | 0.37 |
| 2000 | 0.14 |
| 5000 | 0.007 |

#### 2. **Linear**

```python
reward = max(0, 1 - distance / 20000)
```

**特点**: 线性衰减，简单直观

#### 3. **Threshold**

```python
reward = 1.0 if distance <= 1000 else 0.0
```

**特点**: 二值 reward，稀疏信号

### 坐标解析

`reward_calculator.py` 支持多种格式的坐标输出：

```python
# ✅ 格式 1: 显式标签
"latitude: 40.7128, longitude: -74.0060"

# ✅ 格式 2: 简写
"lat: 40.7128, lon: -74.0060"

# ✅ 格式 3: 括号格式
"(40.7128, -74.0060)"

# ✅ 格式 4: 度数格式
"40.7128°N, 74.0060°W"
```

### 使用示例

```python
from reward_calculator import calculate_reward

ground_truth = {"lat": 40.7128, "lon": -74.0060}  # 纽约
prediction = "Based on the architecture, I believe this is latitude: 40.75, longitude: -73.95"

result = calculate_reward(
    predicted_text=prediction,
    ground_truth=ground_truth,
    reward_type="exponential"
)

print(f"Distance: {result['distance_km']:.2f} km")        # 7.2 km
print(f"Reward: {result['reward']:.4f}")                  # 0.9928
print(f"Parse success: {result['parse_success']}")        # True
print(f"Predicted coords: {result['predicted_coords']}")  # (40.75, -73.95)
```

---

## 在 verl 中使用

### ⭐ 重要：RLHFDataset 支持多文件

**RLHFDataset 原生支持传入文件列表**，会自动合并所有文件！

从源码可以看到：

```python
def __init__(
    self,
    data_files: str | list[str],  # 👈 支持 str 或 list[str]
    ...
):
    if not isinstance(data_files, list | ListConfig):
        data_files = [data_files]

    # 会自动合并所有文件
    for parquet_file in self.data_files:
        dataframe = datasets.load_dataset("parquet", data_files=parquet_file)["train"]
        dataframes.append(dataframe)
    self.dataframe = datasets.concatenate_datasets(dataframes)
```

### 方式 1: 使用单个文件

```python
from verl.utils.dataset.rlhf_dataset import RLHFDataset
from transformers import AutoTokenizer, AutoProcessor

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-VL-7B")
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B")

dataset = RLHFDataset(
    data_files="$GEOGUESSR_DIR/processed/verl_format/gaea_train.parquet",
    tokenizer=tokenizer,
    processor=processor,
    config={
        "cache_dir": "~/.cache/verl/rlhf",
        "prompt_key": "prompt",
        "image_key": "images",
        "max_prompt_length": 2048,
        "return_raw_chat": False,
        "return_multi_modal_inputs": True,
    }
)
```

### 方式 2: 使用 glob 匹配多个 chunk 文件

```python
from glob import glob

# 自动匹配所有 chunk 文件
data_files = glob("$GEOGUESSR_DIR/processed/verl_format/osv5m_train_chunk_*.parquet")
print(f"Found {len(data_files)} chunk files")  # 约 90 个文件

dataset = RLHFDataset(
    data_files=data_files,  # ⭐ 传入文件列表
    tokenizer=tokenizer,
    processor=processor,
    config=config
)

print(f"Total samples: {len(dataset)}")  # 约 450 万样本
```

### 方式 3: 手动指定文件列表

```python
data_files = [
    "$GEOGUESSR_DIR/processed/verl_format/osv5m_train_chunk_0000.parquet",
    "$GEOGUESSR_DIR/processed/verl_format/osv5m_train_chunk_0001.parquet",
    "$GEOGUESSR_DIR/processed/verl_format/osv5m_train_chunk_0002.parquet",
    # ... 更多文件
]

dataset = RLHFDataset(
    data_files=data_files,
    tokenizer=tokenizer,
    processor=processor,
    config=config
)
```

### 方式 4: 混合多个数据集

```python
from glob import glob

# 混合不同数据集
data_files = [
    # 单文件数据集
    "$GEOGUESSR_DIR/processed/verl_format/gaea_train.parquet",
    "$GEOGUESSR_DIR/processed/verl_format/geochain_test.parquet",
] + glob("$GEOGUESSR_DIR/processed/verl_format/osv5m_train_chunk_*.parquet")  # 多个 chunks

dataset = RLHFDataset(
    data_files=data_files,
    tokenizer=tokenizer,
    processor=processor,
    config=config
)

print(f"Total samples: {len(dataset)}")
```

### 配置 Reward 工具

在训练配置中注册 reward 工具：

```python
from reward_calculator import CalcGeoguesrRewardTool

# 注册工具（在 verl trainer 配置中）
trainer.register_tool(
    "calc_geoguessr_reward",
    CalcGeoguesrRewardTool  # 工具类
)

# verl 会自动：
# 1. 从 extra_info.tools_kwargs.calc_geoguessr_reward.create_kwargs 读取参数
# 2. 创建工具实例: tool = CalcGeoguesrRewardTool(**create_kwargs)
# 3. 调用工具: result = tool(model_output)
# 4. 提取 reward: reward = result['reward']
```

---

## 数据集统计

预处理后的数据集规模（估计）：

| 数据集 | Split | 样本数 | 大小 | 特点 |
|--------|-------|--------|------|------|
| OSV5M | train | ~4.5M | ~150GB | 数据量最大，环境特征丰富 |
| OSV5M | test | ~500K | ~20GB | |
| GAEA | train | ~150K | ~10GB | 有 Q&A 对话，质量高 |
| GAEA | bench | ~15K | ~1GB | 评测集 |
| GeoChain | test | ~60K | ~5GB | 有 locatability_score |
| GeoChain | mini_test | ~6K | ~500MB | 小型评测集 |

---

## 最佳实践

### 1. 渐进式训练

建议按照以下顺序训练：

#### **Stage 1**: 使用 GAEA（质量高，有对话）
- 样本数少（~150K），适合初期训练
- 有原始 Q&A，可以学习推理过程
- 图片质量好

```python
data_files = ["$GEOGUESSR_DIR/processed/verl_format/gaea_train.parquet"]
```

#### **Stage 2**: 加入 GeoChain（难度中等）
- 有 `locatability_score` 可以过滤样本
- 图片质量好，多样性强

```python
data_files = [
    "$GEOGUESSR_DIR/processed/verl_format/gaea_train.parquet",
    "$GEOGUESSR_DIR/processed/verl_format/geochain_test.parquet",
]
```

#### **Stage 3**: 加入 OSV5M（数据量大）
- 数据量最大（~4.5M），适合大规模训练
- 有环境特征（land_cover, climate 等）
- 覆盖范围广

```python
data_files = [
    "$GEOGUESSR_DIR/processed/verl_format/gaea_train.parquet",
    "$GEOGUESSR_DIR/processed/verl_format/geochain_test.parquet",
] + glob("$GEOGUESSR_DIR/processed/verl_format/osv5m_train_chunk_*.parquet")
```

### 2. Prompt 策略

尝试不同的 prompt 难度：

#### A. 无提示（最难）
```bash
python convert_to_verl_format.py
```

#### B. 有地址提示（中等）
```bash
python convert_to_verl_format.py --include_address_hints
```
会在 prompt 中包含国家/城市信息。

#### C. 指定输出格式（推荐）
```bash
python convert_to_verl_format.py \
    --instruction_following "Please provide coordinates in format: latitude: XX.XXX, longitude: YY.YYY"
```

### 3. Reward 调优

根据训练阶段调整 reward 函数：

- **Early stage**: 使用 `linear` 或 `threshold`
  - Reward 信号更稀疏
  - 鼓励模型先学习大致区域

- **Later stage**: 使用 `exponential`
  - Reward 信号更精细
  - 鼓励模型优化精确度

**修改方式**: 在 `reward_calculator.py` 中调整 `scale` 参数：

```python
# 修改 calculate_geoguessr_score 函数中的 scale
scale = 1000.0  # 默认值，1000 km → reward ≈ 0.37
scale = 500.0   # 更严格，500 km → reward ≈ 0.37
scale = 2000.0  # 更宽松，2000 km → reward ≈ 0.37
```

### 4. 数据过滤

根据任务难度过滤数据：

```python
# 在 convert_to_verl_format.py 的 convert_sample_to_verl_format 函数中添加：

# 示例 1: 只保留 locatability_score > 0.5 的样本（GeoChain）
if sample.get('locatability_score', 1.0) < 0.5:
    return None  # 跳过该样本

# 示例 2: 只保留特定国家的样本
if sample.get('country') not in ['united states', 'france', 'japan']:
    return None

# 示例 3: 只保留有城市信息的样本
if not sample.get('city'):
    return None
```

---

## 故障排查

### 问题 1: 内存不足

**症状**: 处理大数据集时内存溢出

**解决方案**:

```bash
# 方案 1: 减小样本数进行测试
python convert_to_verl_format.py --single_dataset osv5m/train --max_samples 10000

# 方案 2: 使用分块模式
python convert_to_verl_format.py --single_dataset osv5m/train --chunk_size 10000

# 方案 3: 增加系统 swap
```

### 问题 2: 图像加载失败

**症状**: 某些样本无法加载图像

**原因**: 图像文件损坏或缺失

**解决**: 转换脚本会自动跳过这些样本，检查日志中的错误信息

### 问题 3: Reward 始终为 0

**症状**: 训练时所有样本的 reward 都是 0

**可能原因**:
1. 模型输出格式无法被解析
2. tools_kwargs 配置错误
3. 工具未正确注册

**解决步骤**:

1. **检查模型输出**:
```python
# 查看模型输出是否包含坐标
print(model_output)
# 应该类似: "latitude: 40.71, longitude: -74.00"
```

2. **测试坐标解析**:
```python
from reward_calculator import parse_coordinates_from_text

text = "your model output here"
coords = parse_coordinates_from_text(text)
print(coords)  # 应该返回 (lat, lon) 或 None
```

3. **验证数据格式**:
```bash
python test_verl_format.py your_file.parquet
```

4. **检查工具注册**:
```python
# 确保在 trainer 中注册了工具
trainer.register_tool("calc_geoguessr_reward", CalcGeoguesrRewardTool)
```

### 问题 4: Nominatim 服务请求失败

**症状**: 预处理时频繁出现网络错误

**解决**:
1. 检查 Nominatim 服务是否正常运行
2. 调整 `retry_delay` 参数（默认 1 秒）
3. 使用 `max_retries=-1` 无限重试（默认）

### 问题 5: 分块文件太多，难以管理

**症状**: OSV5M 生成了 90 个 chunk 文件

**解决**: 使用 glob 模式自动加载：

```python
from glob import glob

# 加载所有 chunks
data_files = glob("$GEOGUESSR_DIR/processed/verl_format/osv5m_train_chunk_*.parquet")

# 或者只加载前 10 个 chunks 进行测试
data_files = sorted(glob(".../*_chunk_*.parquet"))[:10]

dataset = RLHFDataset(data_files=data_files, ...)
```

---

## 注意事项

### 存储空间

- 确保有足够的磁盘空间（每个数据集处理后约 50-200GB）
- OSV5M 数据集特别大，train 约 150GB，test 约 20GB
- 分块模式不会减少总存储空间，但更易管理

### 逆地理编码服务

- Nominatim 服务需要稳定运行
- 默认配置为无限重试（`max_retries=-1`）
- 建议使用本地 Docker 部署，避免网络延迟

### 图像处理

- 损坏的图像会被自动跳过，不影响处理流程
- 预处理脚本会显示跳过的样本数量
- 可以在日志中查看具体错误信息

### verl 格式转换

- 图像字段从 `image` (单数) 转换为 `images` (复数列表)
- Prompt 必须是 list 格式，包含 role 和 content
- tools_kwargs 结构必须严格按照 verl 要求

### 多文件加载

- RLHFDataset 会自动合并所有文件
- 文件顺序不影响最终数据集
- 可以混合不同来源的数据集

---

## 参考资料

- **verl 框架**: https://github.com/volcengine/verl
- **RLHFDataset 源码**: `verl/utils/dataset/rlhf_dataset.py`
- **GeoGuessr 评分系统**: https://geoguessr.com/scoring
- **Haversine 公式**: https://en.wikipedia.org/wiki/Haversine_formula
- **Nominatim API**: https://nominatim.org/release-docs/latest/api/Overview/

---

## 许可证

本项目遵循 Apache License 2.0。

## 贡献

欢迎提交 issue 和 pull request！

---

**祝训练顺利！** 🚀
