# Enhanced Distillation with Tool Support and Reward Scoring

完整版蒸馏系统，包含：
- ✅ Visual toolbox支持（zoom, rotate）
- ✅ 自动reward打分
- ✅ 保存dataset路径+index（不保存图片）
- ✅ 多轮对话支持
- ✅ 标准prompt格式

## 🆕 新功能

### 1. 工具调用支持

模型可以使用两个视觉工具：

**image_zoom_in_tool**: 放大图片特定区域
```json
<tool_call>
{"name": "image_zoom_in_tool", "arguments": {"bbox_2d": [100, 100, 500, 500]}}
</tool_call>
```

**image_rotate_tool**: 旋转图片
```json
<tool_call>
{"name": "image_rotate_tool", "arguments": {"angle": 90}}
</tool_call>
```

### 2. 自动Reward打分

使用 `reward_function.py` 中的官方GeoGuessr评分系统：
- ✅ 距离计算（Haversine公式）
- ✅ 官方GeoGuessr分数（0-5000点）
- ✅ 多级准确率（@1km, @25km, @200km, @750km, @2500km）

### 3. 高效存储

**不保存图片**，而是保存：
- `dataset_path`: 数据集路径
- `sample_index`: 样本索引

可以随时通过以下方式还原图片：
```python
import datasets
ds = datasets.load_from_disk(dataset_path)
image = ds[sample_index]['image']
```

## 🚀 快速开始

### 测试新版本（2个样本）

```bash
cd /mnt/tidal-alsh-hilab/usr/shenguobin/verl/recipe/geoguessr/data_preprocess/distil

# 运行带工具的版本
python3 demo_distillation_with_tools.py \
    --num_samples 2 \
    --max_turns 5 \
    --output_dir traces_with_tools_demo

# 查看结果
python3 view_traces_enhanced.py traces_with_tools_demo/ --batch

# 查看单个trace详情
python3 view_traces_enhanced.py traces_with_tools_demo/trace_00000.json --verbose
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--dataset_path` | gaea_wlp/train | 数据集路径 |
| `--output_dir` | traces_with_tools | 输出目录 |
| `--num_samples` | 10 | 处理样本数 |
| `--max_turns` | 10 | 每个样本最大轮数 |
| `--temperature` | 0.7 | 采样温度 |
| `--max_tokens` | 2048 | 每轮最大tokens |

## 📋 Trace格式

生成的trace包含以下字段：

```json
{
  "dataset_path": "/path/to/dataset",
  "sample_index": 42,
  "sample_data": {
    "lat": 40.7128,
    "lon": -74.0060,
    "locatability_score": 0.35,
    "country": "united states",
    ...
  },
  "conversation_history": [
    {
      "turn": 1,
      "messages": [...],
      "response": "Let me analyze this image...",
      "usage": {"total_tokens": 1234}
    },
    ...
  ],
  "final_response": "Based on my analysis... \\boxed{40.7128, -74.0060}",
  "reward_score": {
    "score": 0.9234,
    "distance@km": 12.5,
    "geoguessr@point": 4617,
    "parse_success": true,
    "acc@1km": 0.0,
    "acc@25km": 1.0,
    ...
  },
  "tool_calls": [
    {
      "turn": 2,
      "tool_call": {
        "name": "image_zoom_in_tool",
        "arguments": {"bbox_2d": [100, 100, 500, 500]}
      },
      "success": true
    }
  ],
  "metadata": {
    "total_turns": 3,
    "num_tool_calls": 1,
    "parse_success": true,
    "distance_km": 12.5,
    "score": 0.9234
  }
}
```

## 📊 输出示例

```
[Sample 1/2]
  Locatability score: 1.0000
  Ground truth: lat=51.4535, lon=0.0051
  Country: united kingdom
  [SUCCESS] Generated trace
    Turns: 3
    Tool calls: 1
    Parse success: True
    Distance: 245.67 km
    Score: 0.7821
    Saved to: traces_with_tools_demo/trace_00000.json
```

## 🔍 还原图片

```python
import datasets
import json

# 加载trace
with open('traces_with_tools_demo/trace_00000.json', 'r') as f:
    trace = json.load(f)

# 还原图片
ds = datasets.load_from_disk(trace['dataset_path'])
image = ds[trace['sample_index']]['image']

# 显示
image.show()
```

## 📈 预期效果

### 工具使用率

在难度较高的样本（locatability_score < 0.3）中：
- 预期工具使用率：30-50%
- 常见工具调用：zoom_in（查看标识、文字）

### 准确率提升

带工具的trace应该比不带工具的更准确：
- 不带工具：avg distance ~6000 km
- 带工具：预期 avg distance <4000 km（对困难样本）

### Token消耗

- 平均每轮：800-1200 tokens
- 带工具调用：2-5轮（多1-4轮）
- 总计：~2000-4000 tokens/样本（vs 之前的1787）

## 🎯 大规模生成

### 分阶段策略

```bash
# Stage 1: 超困难样本（score < 0.2, 允许更多工具调用）
python3 demo_distillation_with_tools.py \
    --num_samples 1000 \
    --max_turns 15 \
    --output_dir traces_ultra_hard \
    --temperature 0.8

# Stage 2: 困难样本（0.2 <= score < 0.4）
python3 demo_distillation_with_tools.py \
    --num_samples 5000 \
    --max_turns 10 \
    --output_dir traces_hard \
    --temperature 0.7

# Stage 3: 中等样本（0.4 <= score < 0.6）
python3 demo_distillation_with_tools.py \
    --num_samples 10000 \
    --max_turns 8 \
    --output_dir traces_medium \
    --temperature 0.7
```

### 质量过滤

生成后可以过滤低质量trace：

```python
import json
from pathlib import Path

def filter_high_quality_traces(input_dir, output_dir, min_score=0.3):
    """
    Filter traces by quality.

    Criteria:
    - Parse success
    - Distance < 5000 km
    - Score > min_score
    - At least 1 turn
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    high_quality = 0

    for trace_file in input_dir.glob("trace_*.json"):
        with open(trace_file) as f:
            trace = json.load(f)

        reward = trace['reward_score']

        # Check quality
        if (reward.get('parse_success') and
            reward.get('distance@km', 10000) < 5000 and
            reward.get('score', 0) > min_score and
            len(trace['conversation_history']) > 0):

            # Copy to output
            output_file = output_dir / trace_file.name
            with open(output_file, 'w') as f:
                json.dump(trace, f, indent=2)
            high_quality += 1

    print(f"Filtered {high_quality} high-quality traces")

# 使用
filter_high_quality_traces('traces_hard', 'traces_hard_filtered', min_score=0.3)
```

## 📝 对比两个版本

| Feature | demo_distillation.py | demo_distillation_with_tools.py |
|---------|---------------------|--------------------------------|
| 工具支持 | ❌ | ✅ |
| Reward打分 | ❌ | ✅ |
| 多轮对话 | ❌ | ✅ |
| 图片保存 | ✅ (base64) | ❌ (保存index) |
| Token消耗 | ~1800/样本 | ~2500/样本 |
| 处理时间 | ~45秒/样本 | ~60-90秒/样本 |
| 推荐场景 | 快速测试 | 正式生成 |

## 🔧 故障排查

### 问题1: 工具调用失败

**症状**: `tool_calls` 为空或全部失败

**可能原因**:
- 模型没有生成 `<tool_call>` 标签
- JSON格式错误
- bbox超出图片范围

**解决**: 检查conversation_history中的response，确认模型输出格式

### 问题2: Parse success率低

**症状**: 大部分trace的 `parse_success=false`

**可能原因**:
- 模型没有生成 `<answer>` 标签
- 坐标格式不正确
- 达到max_turns仍未给出答案

**解决**: 增加 `max_turns` 或降低 `temperature`

### 问题3: 存储占用大

**症状**: trace文件很大（>100KB/个）

**原因**: conversation_history包含完整的图片base64（工具调用时）

**解决**:
- 这是正常的（需要保存完整对话）
- 可以后处理时移除base64数据

## 🎓 下一步

1. **并行处理**: 创建多进程版本加速
2. **动态max_turns**: 根据难度调整最大轮数
3. **工具优先级**: 在system prompt中引导模型优先使用某些工具
4. **质量检查**: 在生成时实时过滤低质量trace
5. **数据增强**: 同一样本用不同temperature生成多个trace

## 📚 相关文件

- `demo_distillation_with_tools.py`: 主脚本
- `view_traces_enhanced.py`: 查看工具
- `reward_function.py`: Reward计算（在上层目录）
- `visual_toolbox_v2.py`: 工具实现（dots-eval项目）
