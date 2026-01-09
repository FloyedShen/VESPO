# 修订版蒸馏系统

根据要求完成的4项修订：

## ✅ 修订内容

### 1. 简化System Prompt ✅
- **修改前**: 任务特定的详细说明
- **修改后**: `"You are a helpful assistant."`
- **User Prompt**: 保持提供的标准格式

### 2. 高效工具存储 ✅
- **修改前**: 保存工具处理后的图片（base64）
- **修改后**: 只保存工具调用参数
- **恢复方式**: 从数据集加载原图，重新执行工具调用

### 3. 并发采样 ✅
- **实现**: ThreadPoolExecutor并发调用API
- **顺序保证**: 使用字典存储结果，按index排序保存
- **并发数**: 可配置 `--max_workers`（默认4）

### 4. Python代码工具 ✅
- **工具**: `python_code_tool`
- **功能**: 模型编写Python代码处理图片
- **执行**: 本地安全执行（沙箱+超时）
- **支持**: PIL图像操作、numpy

## 📁 文件说明

```
distil/
├── demo_concurrent.py          # 并发版（工具：zoom, rotate）⭐
├── demo_python_code.py         # Python代码工具版 ⭐
├── demo_distillation_with_tools.py  # 原完整版
├── demo_distillation.py        # 原基础版
└── ...
```

## 🆚 版本对比

| Feature | demo_concurrent.py | demo_python_code.py |
|---------|-------------------|---------------------|
| System Prompt | ✅ Simple | ✅ Simple |
| User Prompt | ✅ Standard | ✅ Standard |
| 工具存储 | ✅ 仅参数 | ✅ 仅代码 |
| 并发处理 | ✅ | ✅ |
| 工具类型 | zoom, rotate | Python code |
| 灵活性 | 固定工具 | **任意PIL操作** ⭐ |
| 安全性 | 高 | 中（沙箱） |

## 🚀 使用方法

### 版本1: 并发版（固定工具）

```bash
cd /mnt/tidal-alsh-hilab/usr/shenguobin/verl/recipe/geoguessr/data_preprocess/distil

# 测试（4个worker并发处理10个样本）
python3 demo_concurrent.py \
    --num_samples 10 \
    --max_workers 4 \
    --output_dir traces_concurrent_test

# 大规模生成（8个worker处理1000个样本，预计4小时）
python3 demo_concurrent.py \
    --num_samples 1000 \
    --max_workers 8 \
    --output_dir traces_concurrent_1k
```

**特点**：
- ✅ 并发快速
- ✅ 工具调用：`image_zoom_in_tool`, `image_rotate_tool`
- ✅ 只保存工具参数（`bbox_2d`, `angle`）
- ✅ 可从原图+参数恢复

### 版本2: Python代码工具

```bash
# 测试
python3 demo_python_code.py \
    --num_samples 10 \
    --max_workers 4 \
    --output_dir traces_python_code_test

# 生产
python3 demo_python_code.py \
    --num_samples 1000 \
    --max_workers 8 \
    --output_dir traces_python_code_1k
```

**特点**：
- ✅ 模型可编写任意Python代码
- ✅ 支持复杂图像处理（滤波、增强、分割等）
- ✅ 只保存代码文本
- ✅ 可重新执行代码恢复结果

**模型可用的代码示例**：
```python
# 旋转和裁剪
result_image = image.rotate(45).crop((100, 100, 500, 500))

# 灰度化
from PIL import ImageOps
result_image = ImageOps.grayscale(image)

# 增强对比度
from PIL import ImageEnhance
enhancer = ImageEnhance.Contrast(image)
result_image = enhancer.enhance(2.0)

# 使用numpy
import numpy as np
arr = np.array(image)
# ... 处理 ...
result_image = Image.fromarray(arr)
```

## 📊 性能对比

### 并发加速效果

| Workers | 预计耗时（1000样本） | 相对加速 |
|---------|---------------------|---------|
| 1 | ~27小时 | 1x |
| 4 | ~7小时 | 4x |
| 8 | ~3.5小时 | 8x |

**注**: 实际加速比取决于API服务器负载和网络延迟

### 存储效率

**修改前**（保存图片）:
- 单个trace: ~50KB（含工具处理后的图片base64）
- 1000个trace: ~50MB

**修改后**（仅参数/代码）:
- 单个trace: ~10KB（仅工具参数或代码文本）
- 1000个trace: ~10MB
- **节省**: 80%存储空间 ✅

## 🔄 恢复工具处理结果

### 方案1: 从参数恢复（demo_concurrent.py）

```python
import json
import datasets
from PIL import Image

# 加载trace
with open('traces_concurrent/trace_00000.json') as f:
    trace = json.load(f)

# 加载原图
ds = datasets.load_from_disk(trace['dataset_path'])
image = ds[trace['sample_index']]['image']

# 重新执行工具调用
for tool_call in trace['tool_calls_log']:
    if tool_call['tool_name'] == 'image_zoom_in_tool':
        bbox = tool_call['tool_arguments']['bbox_2d']
        image = image.crop(bbox)
    elif tool_call['tool_name'] == 'image_rotate_tool':
        angle = tool_call['tool_arguments']['angle']
        image = image.rotate(angle, expand=True)

# 得到最终处理后的图片
image.show()
```

### 方案2: 从代码恢复（demo_python_code.py）

```python
import json
import datasets

# 加载trace
with open('traces_python_code/trace_00000.json') as f:
    trace = json.load(f)

# 加载原图
ds = datasets.load_from_disk(trace['dataset_path'])
image = ds[trace['sample_index']]['image']

# 重新执行代码
for tool_call in trace['tool_calls_log']:
    if tool_call['tool_name'] == 'python_code_tool':
        code = tool_call['code']
        # 执行代码
        globals_dict = {'image': image, 'result_image': None}
        exec(code, globals_dict)
        image = globals_dict['result_image']

# 得到最终处理后的图片
image.show()
```

## 🔒 安全性说明

### demo_concurrent.py
- ✅ **完全安全**: 只执行预定义的crop和rotate操作
- ✅ 无代码执行风险

### demo_python_code.py
- ⚠️ **需要注意**: 执行模型生成的代码
- ✅ **已实施的防护**:
  - 沙箱环境（限制可用函数）
  - 10秒超时
  - 禁止危险操作（文件I/O、网络、系统调用）
- ⚠️ **建议**: 生产环境中在隔离容器内运行

## 📋 Trace格式

### demo_concurrent.py输出

```json
{
  "dataset_path": "...",
  "sample_index": 0,
  "sample_data": {...},
  "conversation_log": [...],
  "tool_calls_log": [
    {
      "turn": 1,
      "tool_name": "image_zoom_in_tool",
      "tool_arguments": {"bbox_2d": [100, 100, 500, 500]},
      "success": true
    }
  ],
  "final_response": "...",
  "reward_score": {...},
  "metadata": {...}
}
```

### demo_python_code.py输出

```json
{
  "dataset_path": "...",
  "sample_index": 0,
  "sample_data": {...},
  "conversation_log": [...],
  "tool_calls_log": [
    {
      "turn": 1,
      "tool_name": "python_code_tool",
      "code": "result_image = image.rotate(90).crop((100, 100, 500, 500))",
      "success": true,
      "message": "Success: Image processed successfully"
    }
  ],
  "final_response": "...",
  "reward_score": {...},
  "metadata": {...}
}
```

## 🎯 推荐使用场景

### demo_concurrent.py（推荐）
- ✅ 大规模生产环境
- ✅ 需要快速生成
- ✅ 对安全性要求高
- ✅ 工具使用明确（zoom, rotate）

### demo_python_code.py
- ✅ 研究实验
- ✅ 需要灵活的图像处理
- ✅ 分析模型的代码生成能力
- ⚠️ 隔离环境中运行

## 🧪 快速测试

```bash
cd /mnt/tidal-alsh-hilab/usr/shenguobin/verl/recipe/geoguessr/data_preprocess/distil

# 测试并发版（2个样本，2个worker）
python3 demo_concurrent.py --num_samples 2 --max_workers 2

# 测试Python代码版（2个样本，2个worker）
python3 demo_python_code.py --num_samples 2 --max_workers 2

# 查看结果
python3 view_traces_enhanced.py traces_concurrent/ --batch
python3 view_traces_enhanced.py traces_python_code/ --batch
```

## ✅ 修订完成清单

- [x] System prompt简化为 "You are a helpful assistant."
- [x] User prompt使用提供的标准格式
- [x] 工具调用只保存参数/代码，不保存图片
- [x] 实现并发采样（ThreadPoolExecutor）
- [x] 保证问题答案顺序对应（字典+排序）
- [x] 提供Python代码工具版本
- [x] 本地执行图像处理代码
- [x] 沙箱+超时保护

## 📚 相关文档

- `README_ENHANCED.md`: 原完整版文档
- `DEMO_VERIFICATION.md`: 原版验证报告
- 本文档: 修订版说明
