# 🚀 生产级蒸馏脚本 - 断点续传版

完整的生产级脚本，支持断点续传、实时保存、灵活配置。

## ✨ 核心特性

### 1. 断点续传 ✅
- **自动保存进度**: 每个样本处理后立即更新checkpoint
- **智能跳过**: 重启后自动跳过已处理的样本
- **失败记录**: 记录失败样本，避免重复尝试
- **随时中断**: 可随时Ctrl+C中断，重启后继续

### 2. 实时保存 ✅
- **即时写入**: 每个trace生成后立即保存到磁盘
- **文件锁保护**: 使用fcntl防止并发写入冲突
- **原子操作**: 使用临时文件+重命名保证数据完整性
- **不丢数据**: 即使程序崩溃，已处理样本不会丢失

### 3. 灵活配置 ✅
- **指定数据集**: `--dataset_path`
- **并发度**: `--max_workers` (1-16)
- **采样策略**: `--sampling_strategy` (random/hardest/easiest)
- **输出目录**: `--output_dir`

## 🎯 快速使用

### 基础用法

```bash
cd /mnt/tidal-alsh-hilab/usr/shenguobin/verl/recipe/geoguessr/data_preprocess/distil

# 生成1000个样本，4个worker
python3 distill_production.py \
    --dataset_path /mnt/tidal-alsh-hilab/usr/shenguobin/geoguessr/processed/gaea_wlp/train \
    --output_dir traces_production_1k \
    --num_samples 1000 \
    --max_workers 4
```

### 断点续传示例

```bash
# 第一次运行（中断在500/1000）
python3 distill_production.py \
    --dataset_path /path/to/dataset \
    --output_dir traces_1k \
    --num_samples 1000 \
    --max_workers 4

# ... 处理到500个时按Ctrl+C中断 ...

# 第二次运行（自动从501开始）
python3 distill_production.py \
    --dataset_path /path/to/dataset \
    --output_dir traces_1k \
    --num_samples 1000 \
    --max_workers 4

# 输出：
# Resume from checkpoint:
#   Already processed: 500
#   Remaining: 500/1000
# Processing 500 samples with 4 workers
```

## 📊 参数说明

### 必需参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `--dataset_path` | 数据集路径 | `/path/to/gaea_wlp/train` |
| `--output_dir` | 输出目录（含checkpoint） | `traces_1k` |

### 可选参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--num_samples` | 1000 | 生成样本数 |
| `--max_workers` | 4 | 并发worker数（1-16） |
| `--max_turns` | 10 | 每样本最大轮数 |
| `--temperature` | 0.7 | 采样温度 |
| `--max_tokens` | 2048 | 每轮最大tokens |
| `--sampling_strategy` | random | 采样策略 |
| `--no_resume` | False | 不从checkpoint恢复 |

### 采样策略

| 策略 | 说明 | 适用场景 |
|------|------|---------|
| `random` | 随机采样 | 通用，无偏 |
| `hardest` | 困难优先 | 想要难样本 |
| `easiest` | 简单优先 | Warm-up训练 |

## 📁 输出结构

```
traces_production_1k/
├── checkpoint.json              # 断点文件 ⭐
├── trace_00000.json             # Trace 0
├── trace_00001.json             # Trace 1
├── trace_00002.json             # Trace 2
├── ...
└── trace_00999.json             # Trace 999
```

### checkpoint.json格式

```json
{
  "processed_indices": [0, 1, 2, ..., 499],
  "failed_indices": [5, 23, 147],
  "total_processed": 500,
  "total_failed": 3,
  "timestamp": 1701234567.89
}
```

## 🔍 监控和检查

### 1. 查看checkpoint状态

```bash
# 使用辅助脚本
python3 check_checkpoint.py traces_production_1k

# 输出：
# ============================================================
# Checkpoint Status: traces_production_1k
# ============================================================
# Processed samples: 500
# Failed samples: 3
# Total attempts: 503
# Last update: 2024-11-26 21:30:45
#
# Actual trace files: 500
# ============================================================
```

### 2. 实时监控进度

```bash
# 方法1: watch命令
watch -n 5 'python3 check_checkpoint.py traces_production_1k'

# 方法2: 查看checkpoint文件
watch -n 5 'cat traces_production_1k/checkpoint.json | jq ".total_processed, .total_failed"'

# 方法3: 统计trace文件
watch -n 5 'ls traces_production_1k/trace_*.json | wc -l'
```

### 3. 查看最新生成的trace

```bash
# 最新10个
ls -lt traces_production_1k/trace_*.json | head -10

# 查看某个trace
python3 view_traces_enhanced.py traces_production_1k/trace_00500.json
```

## 🎯 使用场景

### 场景1: 快速测试

```bash
# 测试：10个样本，2个worker
python3 distill_production.py \
    --dataset_path /path/to/dataset \
    --output_dir traces_test \
    --num_samples 10 \
    --max_workers 2
```

### 场景2: 中等规模生成

```bash
# 1000个样本，4个worker（预计7小时）
python3 distill_production.py \
    --dataset_path /path/to/gaea_wlp/train \
    --output_dir traces_1k \
    --num_samples 1000 \
    --max_workers 4
```

### 场景3: 大规模生成

```bash
# 10000个样本，8个worker（预计35小时，可中断续传）
python3 distill_production.py \
    --dataset_path /path/to/gaea_wlp/train \
    --output_dir traces_10k \
    --num_samples 10000 \
    --max_workers 8 \
    --sampling_strategy hardest
```

### 场景4: 困难样本优先

```bash
# 只采样最困难的1000个样本
python3 distill_production.py \
    --dataset_path /path/to/gaea_wlp/train \
    --output_dir traces_hardest_1k \
    --num_samples 1000 \
    --max_workers 4 \
    --sampling_strategy hardest
```

### 场景5: 不同数据集

```bash
# GAEA bench数据集
python3 distill_production.py \
    --dataset_path /path/to/gaea_wlp/bench \
    --output_dir traces_gaea_bench \
    --num_samples 500 \
    --max_workers 4
```

## 🔧 高级用法

### 1. 重新开始（忽略checkpoint）

```bash
# 使用--no_resume强制重新开始
python3 distill_production.py \
    --dataset_path /path/to/dataset \
    --output_dir traces_1k \
    --num_samples 1000 \
    --max_workers 4 \
    --no_resume
```

### 2. 增加采样温度（更多样性）

```bash
python3 distill_production.py \
    --dataset_path /path/to/dataset \
    --output_dir traces_diverse \
    --num_samples 1000 \
    --max_workers 4 \
    --temperature 0.9
```

### 3. 更多推理轮次

```bash
python3 distill_production.py \
    --dataset_path /path/to/dataset \
    --output_dir traces_complex \
    --num_samples 1000 \
    --max_workers 4 \
    --max_turns 15
```

### 4. 继续采样（追加更多样本）

```bash
# 已经生成了1000个，想要再生成1000个
# 将num_samples改为2000即可
python3 distill_production.py \
    --dataset_path /path/to/dataset \
    --output_dir traces_2k \
    --num_samples 2000 \
    --max_workers 4
```

## 📈 性能预估

### 处理时间（基于测试）

| 样本数 | Worker数 | 预计耗时 | 备注 |
|--------|---------|---------|------|
| 10 | 2 | ~2分钟 | 测试 |
| 100 | 4 | ~25分钟 | 小规模 |
| 1000 | 4 | ~7小时 | 中规模 |
| 1000 | 8 | ~3.5小时 | 中规模（快） |
| 10000 | 8 | ~35小时 | 大规模 |

### 并发效率

| Worker数 | 加速比 | CPU使用 | 内存使用 |
|---------|--------|---------|---------|
| 1 | 1x | ~10% | ~2GB |
| 4 | ~4x | ~40% | ~8GB |
| 8 | ~8x | ~80% | ~16GB |
| 16 | ~12x | 100% | ~32GB |

**建议**: 4-8个worker为最佳平衡

## 🛡️ 错误处理

### 自动处理的情况

1. **API临时失败**: 自动记录为失败，不影响其他样本
2. **单个样本错误**: 记录failed_indices，继续处理
3. **文件写入冲突**: 使用锁机制自动处理
4. **程序中断**: checkpoint自动保存，重启继续

### 手动干预的情况

1. **重复失败**: 检查failed_indices，可能需要调整参数
2. **磁盘满**: 清理空间后继续
3. **API长时间不可用**: 等待恢复后继续

## 🔍 故障排查

### 问题1: 无法恢复

**症状**: 运行时显示"Already processed: 0"

**原因**: checkpoint.json损坏或不存在

**解决**:
```bash
# 检查checkpoint
python3 check_checkpoint.py traces_1k

# 如果损坏，可以手动重建或使用--no_resume
```

### 问题2: 失败样本过多

**症状**: Failed samples > 10%

**原因**: API不稳定或参数不当

**解决**:
```bash
# 降低并发度
--max_workers 2

# 降低温度
--temperature 0.5
```

### 问题3: 进度缓慢

**症状**: 每个样本耗时>2分钟

**原因**: max_turns太大或网络慢

**解决**:
```bash
# 减少max_turns
--max_turns 5

# 检查API响应时间
curl -w "@curl-format.txt" http://10.146.229.25:80/v1/models
```

## 📊 实时统计

脚本会在最后输出详细统计：

```
================================================================================
Final Summary
================================================================================
Total requested: 1000
Already processed (before): 500
Newly processed: 495
Total processed: 995
Failed: 5
Remaining: 0
Output directory: traces_production_1k
Checkpoint: traces_production_1k/checkpoint.json
================================================================================

Statistics (sampled from 100 traces):
  Parse success rate: 95/100 (95.0%)
  Average distance: 4523.45 km
  Median distance: 3201.12 km
================================================================================
```

## ✅ 最佳实践

1. **小规模测试**: 先用10个样本测试配置
2. **监控进度**: 使用watch命令实时监控
3. **定期备份**: 定期备份output_dir（特别是checkpoint.json）
4. **合理并发**: 4-8 workers最佳
5. **灵活中断**: 可随时中断，不影响已生成数据

## 🎯 完整示例

```bash
#!/bin/bash
# 完整的生产环境示例

# 1. 配置
DATASET=/mnt/tidal-alsh-hilab/usr/shenguobin/geoguessr/processed/gaea_wlp/train
OUTPUT=traces_production_1k
WORKERS=4
SAMPLES=1000

# 2. 运行（支持随时中断）
python3 distill_production.py \
    --dataset_path $DATASET \
    --output_dir $OUTPUT \
    --num_samples $SAMPLES \
    --max_workers $WORKERS \
    --sampling_strategy hardest

# 3. 检查结果
python3 check_checkpoint.py $OUTPUT

# 4. 查看统计
python3 view_traces_enhanced.py $OUTPUT/ --batch
```

---

**推荐**: 这是最适合生产环境的版本，支持断点续传、实时保存、灵活配置！🚀
