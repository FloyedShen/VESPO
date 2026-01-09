# GeoGuessr Trace Distillation

使用 Qwen3-VL-235B-Thinking 生成高质量的思考链和工具调用轨迹，用于蒸馏训练。

## 概述

这个模块从已计算 locatability_score 的数据集中：
1. **按难度排序**：优先处理困难样本（locatability_score 低）
2. **调用大模型**：使用 Qwen3-VL-235B-Thinking 生成详细推理过程
3. **保存轨迹**：保存完整的输入输出和推理链

## 文件说明

```
distil/
├── demo_distillation.py      # 主脚本：生成trace demo
├── test_api.py               # 测试API连接
└── README.md                 # 本文档
```

## 快速开始

### 1. 测试API连接

```bash
cd /mnt/tidal-alsh-hilab/usr/shenguobin/verl/recipe/geoguessr/data_preprocess/distil

# 测试API是否可用
python3 test_api.py
```

### 2. 运行Demo（处理10个最难样本）

```bash
# 默认配置：处理10个最难样本
python3 demo_distillation.py

# 自定义参数
python3 demo_distillation.py \
    --dataset_path /mnt/tidal-alsh-hilab/usr/shenguobin/geoguessr/processed/gaea_wlp/train \
    --output_dir ./traces_demo \
    --num_samples 10 \
    --temperature 0.7 \
    --max_tokens 4096
```

### 3. 查看生成结果

```bash
# 查看生成的trace文件
ls -lh traces_demo/

# 查看某个trace的内容
python3 -c "
import json
with open('traces_demo/trace_00000.json', 'r') as f:
    trace = json.load(f)
    print('Locatability score:', trace['sample_data']['locatability_score'])
    print('Ground truth:', trace['sample_data']['lat'], trace['sample_data']['lon'])
    print('Response:', trace['response_text'][:500])
"
```

## 配置说明

### API配置

```python
API_BASE_URL = "http://10.146.229.25:80/v1"
MODEL_NAME = "nginx"
```

### System Prompt

默认使用标准的地理定位system prompt，引导模型：
- 分析图像中的线索（建筑、植被、路标等）
- 逐步推理
- 使用工具（待集成visual_toolbox）
- 输出格式：`\boxed{latitude, longitude}`

### 采样参数

- **temperature**: 0.7（平衡创造性和准确性）
- **max_tokens**: 4096（支持详细推理）

## 输出格式

每个样本生成一个JSON文件 `trace_XXXXX.json`：

```json
{
  "sample_data": {
    "lat": 43.772747,
    "lon": 11.255256,
    "locatability_score": 0.2255,
    "country": "italy",
    "class_mapping": "{...}",
    ...
  },
  "api_response": {
    "choices": [...],
    "usage": {...},
    ...
  },
  "response_text": "Let me analyze this image step by step...\n\n..."
}
```

## 难度排序策略

样本按 `locatability_score` **降序**排列（最难的优先）：

| Score Range | Difficulty | Priority |
|-------------|------------|----------|
| < 0.3 | 困难（室内/信息少） | ⭐⭐⭐ 最高 |
| 0.3 - 0.6 | 中等（部分线索） | ⭐⭐ 中等 |
| > 0.6 | 简单（明显标志） | ⭐ 较低 |

**原因**：困难样本需要更复杂的推理，生成的trace质量更高，更适合蒸馏训练。

## 进阶使用

### 处理大规模数据

```bash
# 处理前1000个最难样本
python3 demo_distillation.py --num_samples 1000

# 使用不同温度生成多样性trace
python3 demo_distillation.py --temperature 0.9 --num_samples 100
```

### 集成Visual Toolbox（TODO）

当 visual_toolbox_v2.py 可用时，需要在 `demo_distillation.py` 中集成：

```python
# 1. Import tools
from dots_evals.tools.deepeyes.visual_toolbox_v2 import VisualToolbox

# 2. Create tool instance
toolbox = VisualToolbox()

# 3. Add tool calling in system prompt
SYSTEM_PROMPT_WITH_TOOLS = """
...
Available tools:
- image_analysis: Analyze specific regions
- text_extraction: Extract text from signs
- ...
"""

# 4. Handle tool calls in response
# Parse tool calls from model output
# Execute tools
# Feed results back to model
```

## 数据集路径

- **GAEA train**: `/mnt/tidal-alsh-hilab/usr/shenguobin/geoguessr/processed/gaea_wlp/train` (1.4M samples)
- **GAEA bench**: `/mnt/tidal-alsh-hilab/usr/shenguobin/geoguessr/processed/gaea_wlp/bench` (15K samples)

## 故障排查

### API连接失败

```bash
# 1. 检查API服务是否运行
curl http://10.146.229.25:80/v1/models

# 2. 检查网络连接
ping 10.146.229.25

# 3. 查看详细错误日志
python3 demo_distillation.py --num_samples 1 2>&1 | tee distill.log
```

### 生成质量不佳

调整采样参数：
- 降低 temperature（0.3-0.5）：更保守、更准确
- 提高 temperature（0.8-1.0）：更多样、更创造性
- 增加 max_tokens：支持更长的推理链

### 处理速度慢

- 减少 `num_samples`
- 使用并行处理（多进程，每个进程调用API）
- 调整 timeout 参数

## 下一步

1. ✅ 验证API连接（test_api.py）
2. ✅ 生成demo traces（10个样本）
3. ⏳ 集成visual_toolbox工具
4. ⏳ 扩展到大规模生成（1000+样本）
5. ⏳ 实现并行处理加速
6. ⏳ 添加质量过滤（解析成功率、坐标准确度）

## 参考

- Qwen API文档: https://help.aliyun.com/zh/dashscope/
- GeoGuessr项目: `../README.md`
- Locatability Score: `../compute_locatability_score/README.md`
