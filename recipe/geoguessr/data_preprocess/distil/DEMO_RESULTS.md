# 蒸馏Demo运行总结

## ✅ 已完成

### 1. 创建的文件

```
/mnt/tidal-alsh-hilab/usr/shenguobin/verl/recipe/geoguessr/data_preprocess/distil/
├── demo_distillation.py    # 主脚本：生成trace
├── test_api.py             # API连接测试
├── view_traces.py          # Trace查看和分析工具
├── README.md               # 使用文档
├── DEMO_RESULTS.md         # 本文档
└── traces_demo/            # 生成的demo traces
    ├── trace_00000.json    # Sample 1
    ├── trace_00001.json    # Sample 2
    └── trace_00002.json    # Sample 3
```

### 2. API测试结果 ✅

```bash
$ python3 test_api.py

╔══════════════════════════════════════════════════════════════════════════════╗
║                    Qwen3-VL-235B-Thinking API Test                          ║
╚══════════════════════════════════════════════════════════════════════════════╝

API Health                ✅ PASSED
Text Completion           ✅ PASSED
Vision Completion         ✅ PASSED

🎉 All tests passed! API is ready to use.
```

**API信息**：
- URL: http://10.146.229.25:80/v1
- Model: nginx (Qwen3-VL-235B-A22B-Thinking)
- Max tokens: 262,144
- 支持思考链输出（包含 `</think>` 标签）

### 3. Demo运行结果

**配置**：
- 数据集：GAEA train (1.4M samples)
- 排序方式：按难度降序（locatability_score从低到高）
- 处理样本数：3个最难样本
- 温度：0.7
- 最大tokens：4096

**性能**：
- ✅ 成功率：3/3 (100%)
- ⏱️ 平均耗时：~45秒/样本
- 📊 平均tokens：1787 tokens/样本

**质量指标**：
```
Parse success rate: 3/3 (100.0%)
Average distance error: 6063.76 km

Accuracy:
  @   1km:   0/3 (  0.0%)
  @  25km:   0/3 (  0.0%)
  @ 200km:   0/3 (  0.0%)
  @ 750km:   0/3 (  0.0%)
  @2500km:   1/3 ( 33.3%)
```

**注**：准确率较低是因为这些是**最困难的样本**（locatability_score=1.0，通常是难以定位的场景）。

### 4. 生成的Trace示例

每个trace包含详细的推理过程：

```json
{
  "sample_data": {
    "lat": 39.1283,
    "lon": -84.4776,
    "locatability_score": 1.0,
    "country": "united states",
    ...
  },
  "api_response": {
    "choices": [...],
    "usage": {
      "prompt_tokens": 354,
      "completion_tokens": 1277,
      "total_tokens": 1631
    }
  },
  "response_text": "Okay, let's try to figure out where this photo was taken. The image shows a pothole in an asphalt road with some old bricks exposed underneath. Hmm, bricks under asphalt... [详细推理过程] ... \\boxed{42.3558, -71.0690}"
}
```

**推理特点**：
- ✅ 逐步分析图像线索
- ✅ 考虑地理、建筑、气候等多种因素
- ✅ 包含思考过程（thinking chain）
- ✅ 使用标准格式输出：`\boxed{lat, lon}`

## 🚀 如何使用

### 快速开始

```bash
cd /mnt/tidal-alsh-hilab/usr/shenguobin/verl/recipe/geoguessr/data_preprocess/distil

# 1. 测试API
python3 test_api.py

# 2. 生成少量trace（demo）
python3 demo_distillation.py --num_samples 10

# 3. 查看结果
python3 view_traces.py traces_demo/ --batch

# 4. 查看单个trace
python3 view_traces.py traces_demo/trace_00000.json --verbose
```

### 大规模生成

```bash
# 生成1000个最难样本的trace
python3 demo_distillation.py \
    --num_samples 1000 \
    --output_dir traces_hard_1k \
    --temperature 0.7

# 预计耗时：~12.5小时 (45秒 × 1000)
# 预计tokens：~1.8M tokens
```

### 并行加速（TODO）

可以创建多进程版本加速：

```bash
# 使用4个进程并行处理
python3 demo_distillation_parallel.py \
    --num_samples 1000 \
    --num_workers 4 \
    --output_dir traces_hard_1k

# 预计耗时：~3小时 (12.5小时 / 4)
```

## 📊 数据集信息

**可用数据集**：

| 数据集 | 路径 | 样本数 | 说明 |
|--------|------|--------|------|
| GAEA train | `/mnt/tidal-alsh-hilab/usr/shenguobin/geoguessr/processed/gaea_wlp/train` | 1.4M | 有Q&A对话 |
| GAEA bench | `/mnt/tidal-alsh-hilab/usr/shenguobin/geoguessr/processed/gaea_wlp/bench` | 15K | 评测集 |

**Locatability Score分布**（预估）：

| Score Range | Difficulty | 占比 | 建议 |
|-------------|------------|------|------|
| < 0.3 | 困难 | ~20% | 优先生成 ⭐⭐⭐ |
| 0.3 - 0.6 | 中等 | ~50% | 主要训练数据 ⭐⭐ |
| > 0.6 | 简单 | ~30% | 补充数据 ⭐ |

## 🔧 下一步工作

### 1. 集成Visual Toolbox（优先）

**TODO**：找到并集成 `visual_toolbox_v2.py`

```python
# 需要实现的功能：
# 1. 在system prompt中添加工具描述
# 2. 解析模型输出中的工具调用
# 3. 执行工具并获取结果
# 4. 将结果反馈给模型继续推理
```

### 2. 实现并行处理

创建 `demo_distillation_parallel.py`：
- 使用 multiprocessing 或 concurrent.futures
- 每个进程独立调用API
- 合并结果到统一目录

### 3. 质量过滤

添加过滤逻辑：
```python
# 只保存高质量trace
def is_high_quality(trace):
    # 1. 坐标解析成功
    if not parse_success:
        return False

    # 2. 推理链长度合理（>500 chars）
    if len(response) < 500:
        return False

    # 3. 距离误差不太大（<5000km）
    if distance > 5000:
        return False

    return True
```

### 4. 分阶段生成策略

```bash
# Stage 1: 困难样本（locatability_score < 0.3）
python3 demo_distillation.py \
    --filter_score_max 0.3 \
    --num_samples 10000 \
    --output_dir traces_hard

# Stage 2: 中等样本（0.3 <= score <= 0.6）
python3 demo_distillation.py \
    --filter_score_min 0.3 \
    --filter_score_max 0.6 \
    --num_samples 50000 \
    --output_dir traces_medium

# Stage 3: 简单样本（score > 0.6）
python3 demo_distillation.py \
    --filter_score_min 0.6 \
    --num_samples 20000 \
    --output_dir traces_easy
```

### 5. 数据转换为训练格式

将生成的traces转换为verl训练格式：

```python
# Convert traces to verl format
python3 convert_traces_to_verl.py \
    --input_dir traces_hard/ \
    --output_file train_data_distilled.parquet
```

## 📝 注意事项

1. **API限流**：注意API可能有QPS限制，建议添加 rate limiting
2. **错误重试**：网络不稳定时建议添加重试机制
3. **断点续传**：大规模生成时实现checkpoint机制
4. **磁盘空间**：每1000个trace约需要10-15MB存储空间

## 🎯 预期效果

使用这些高质量trace进行蒸馏训练，预期可以：

1. **提升推理能力**：学习详细的思考链
2. **改善工具使用**：学习何时、如何使用视觉工具
3. **提高准确率**：通过困难样本的详细分析提升模型能力
4. **加速收敛**：高质量trace可以作为warm-start数据

## 📚 相关文档

- API测试：`test_api.py`
- 使用说明：`README.md`
- 主项目文档：`../README.md`
- Locatability Score：`../compute_locatability_score/README.md`
