# 🚀 生产环境快速开始指南

## 已完成功能 ✅

你的生产级蒸馏系统已经完全实现，包含以下核心功能：

### 1. **断点续传** ✅
- 自动保存进度到 `checkpoint.json`
- 重启后自动跳过已处理样本
- 记录失败样本，避免重复尝试

### 2. **实时保存** ✅
- 每个trace生成后立即保存到磁盘
- 使用文件锁防止并发冲突
- 原子写入保证数据完整性

### 3. **灵活配置** ✅
- 指定数据集路径
- 可调并发度 (1-16 workers)
- 多种采样策略 (random/hardest/easiest)

---

## 立即开始使用

### 方式1: 快速测试 (10个样本)

```bash
cd /mnt/tidal-alsh-hilab/usr/shenguobin/verl/recipe/geoguessr/data_preprocess/distil

python3 distill_production.py \
    --dataset_path /mnt/tidal-alsh-hilab/usr/shenguobin/geoguessr/processed/gaea_wlp/train \
    --output_dir traces_test \
    --num_samples 10 \
    --max_workers 2
```

**预计时间**: ~2分钟

### 方式2: 生产环境 (1000个样本)

```bash
python3 distill_production.py \
    --dataset_path /mnt/tidal-alsh-hilab/usr/shenguobin/geoguessr/processed/gaea_wlp/train \
    --output_dir traces_production_1k \
    --num_samples 1000 \
    --max_workers 4
```

**预计时间**: ~7小时
**可随时中断**: 按 Ctrl+C 中断后，再次运行相同命令即可继续

### 方式3: 困难样本优先

```bash
python3 distill_production.py \
    --dataset_path /mnt/tidal-alsh-hilab/usr/shenguobin/geoguessr/processed/gaea_wlp/train \
    --output_dir traces_hardest_1k \
    --num_samples 1000 \
    --max_workers 4 \
    --sampling_strategy hardest
```

---

## 监控进度

### 实时查看进度

```bash
# 方法1: 查看checkpoint
python3 check_checkpoint.py traces_production_1k

# 方法2: 统计trace文件数
ls traces_production_1k/trace_*.json | wc -l

# 方法3: 持续监控（每5秒刷新）
watch -n 5 'python3 check_checkpoint.py traces_production_1k'
```

### 查看统计信息

```bash
python3 view_traces_enhanced.py traces_production_1k/ --batch
```

---

## 断点续传演示

### 场景: 处理到500个时被中断

```bash
# 第一次运行
python3 distill_production.py \
    --dataset_path /path/to/dataset \
    --output_dir traces_1k \
    --num_samples 1000 \
    --max_workers 4

# ... 运行一段时间后按 Ctrl+C 中断 ...
# 输出: Processed: 500/1000

# 第二次运行（相同命令）
python3 distill_production.py \
    --dataset_path /path/to/dataset \
    --output_dir traces_1k \
    --num_samples 1000 \
    --max_workers 4

# 输出:
# Resume from checkpoint:
#   Already processed: 500
#   Remaining: 500/1000
# Processing 500 samples with 4 workers
# ✅ 自动从501开始继续
```

---

## 核心文件说明

### 1. distill_production.py
主要的生产脚本，支持所有功能。

**关键特性**:
- `CheckpointManager`: 管理断点状态
- `RealtimeTraceSaver`: 实时保存trace
- 线程池并发处理
- 自动重试机制

### 2. check_checkpoint.py
检查checkpoint状态的工具。

```bash
python3 check_checkpoint.py <output_dir>
```

### 3. examples_production.sh
包含10+个实用示例，覆盖各种使用场景。

```bash
# 查看所有示例
cat examples_production.sh
```

### 4. README_PRODUCTION.md
完整的文档，包含:
- 详细参数说明
- 性能预估
- 故障排查
- 最佳实践

---

## 输出结构

```
traces_production_1k/
├── checkpoint.json          # 断点文件 ⭐
├── trace_00000.json         # Trace 0
├── trace_00001.json         # Trace 1
├── trace_00002.json         # Trace 2
├── ...
└── trace_00999.json         # Trace 999
```

### checkpoint.json 格式

```json
{
  "processed_indices": [0, 1, 2, ..., 499],
  "failed_indices": [5, 23],
  "total_processed": 500,
  "total_failed": 2,
  "timestamp": 1732654321.12
}
```

---

## 参数速查

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--dataset_path` | (必需) | 数据集路径 |
| `--output_dir` | (必需) | 输出目录 |
| `--num_samples` | 1000 | 生成样本数 |
| `--max_workers` | 4 | 并发数 |
| `--max_turns` | 10 | 每样本最大轮数 |
| `--temperature` | 0.7 | 采样温度 |
| `--sampling_strategy` | random | random/hardest/easiest |
| `--no_resume` | False | 不从checkpoint恢复 |

---

## 性能参考

| 样本数 | Worker数 | 预计耗时 |
|--------|---------|---------|
| 10 | 2 | ~2分钟 |
| 100 | 4 | ~25分钟 |
| 1000 | 4 | ~7小时 |
| 1000 | 8 | ~3.5小时 |
| 10000 | 8 | ~35小时 |

**建议**: 使用4-8个worker获得最佳性价比

---

## 常见问题

### Q: 如何从头开始（忽略checkpoint）？
```bash
python3 distill_production.py \
    --output_dir traces_1k \
    --num_samples 1000 \
    --no_resume
```

### Q: 如何追加更多样本？
```bash
# 已生成1000个，想要2000个
# 只需将 num_samples 改为 2000
python3 distill_production.py \
    --output_dir traces_2k \
    --num_samples 2000  # 增加数量
```

### Q: 如何查看失败的样本？
```bash
python3 check_checkpoint.py traces_1k
# 输出会显示 failed_indices: [5, 23, 147]
```

---

## 下一步建议

1. **小规模测试**: 先运行10个样本验证配置
2. **监控运行**: 使用watch命令实时监控
3. **中等规模**: 确认无误后生成1000个样本
4. **大规模生产**: 利用断点续传生成10000+样本

---

## 更多文档

- **完整文档**: `README_PRODUCTION.md`
- **使用示例**: `examples_production.sh`
- **系统总览**: `FINAL_SUMMARY.md`
- **自适应采样**: `README_ADAPTIVE.md`

---

**系统已就绪，可以开始使用！** 🎉

建议从测试命令开始:
```bash
python3 distill_production.py \
    --dataset_path /mnt/tidal-alsh-hilab/usr/shenguobin/geoguessr/processed/gaea_wlp/train \
    --output_dir traces_test \
    --num_samples 10 \
    --max_workers 2
```
