# 实验脚本生成器使用说明

## 📁 生成的文件

1. **`generate_experiment_scripts.py`** - 主生成脚本
2. **`experiment_tracking.md`** - 实验追踪表格（支持手动编辑）
3. **`gamma_is_ppo_epochs_16_*.sh`** - 19个实验启动脚本
4. **`run_all_experiments.sh`** - 批量运行脚本
5. **`experiment_configs_summary.txt`** - 配置总结文本

## 🚀 快速开始

### 1. 生成所有脚本
```bash
python3 generate_experiment_scripts.py
```

### 2. 运行单个实验
```bash
# 直接运行（当前目录）
bash gamma_is_ppo_epochs_16_2.0_2.0_4.0_2.5_G5.sh

# 使用完整路径（推荐，可从任意目录执行）
bash recipe/is_shape/run/qwen3-4b-base/gamma/gamma_is_ppo_epochs_16_2.0_2.0_4.0_2.5_G5.sh
```

## 📊 实验追踪

### 编辑追踪表
在 `experiment_tracking.md` 中手动填写「运行机器」、「运行状态」和「说明」：

```markdown
| Group | ... | 运行机器 | 运行状态 | 说明 |
|-------|-----|---------|---------|------|
| G1    | ... | node01  | 已完成   | 测试基线配置，效果不错 |
| G2    | ... | node02  | 运行中   | 正在验证强 Lower Clip |
| G5    | ... | gpu-01  | 失败     | OOM 错误，需要调整参数 |
| G6    | ... | -       | -       | -    |
```

**可编辑列说明：**
- **运行机器**: 填写运行实验的机器名称（如 node01, gpu-server-01）
- **运行状态**: 实验的当前状态
- **说明**: 自定义备注、实验结果、遇到的问题等（初始为空，可自由填写）

**状态选项：**
- `-`: 未设置
- `待运行`: 等待执行
- `运行中`: 正在执行
- `已完成`: 执行完成
- `失败`: 执行失败
- `暂停`: 暂时停止

**配置参考：**
表格下方的「配置参考」部分保留了所有实验的原始说明，供你参考使用。你可以复制到主表格中，也可以写自己的说明。

### 重新生成（保留用户数据）
```bash
python3 generate_experiment_scripts.py
```

**重要：** 你填写的「运行机器」、「运行状态」和「说明」**不会被覆盖**！脚本会自动保留这些信息。

## 🔧 添加新实验配置

### 步骤 1: 编辑配置列表
在 `generate_experiment_scripts.py` 中的 `EXPERIMENT_CONFIGS` 添加新配置：

```python
EXPERIMENT_CONFIGS = [
    # 现有配置...
    ("G19", "Wild", 4.0, 3.2, 1.25, 4.0, 4.0, "模拟 PPO 的'平顶'感"),

    # 添加新配置
    ("G20", "Custom", 2.5, 2.5, 1.00, 4.5, 3.0, "我的自定义配置"),
]
```

### 步骤 2: 重新生成
```bash
python3 generate_experiment_scripts.py
```

**自动处理：**
- ✓ 生成新的 G20 实验脚本
- ✓ 在追踪表中添加 G20 行（机器和状态为空）
- ✓ 保留所有已有配置的用户填写数据

## 📝 配置参数说明

| 参数 | 说明 |
|-----|------|
| Group | 组别标识（如 G1, G2...） |
| Strategy | 策略类型（Mimic, Math, Drift, NegDef, Wild, Legacy） |
| k_pos | 正向 k 参数 |
| lam_pos | 正向 λ 参数 |
| peak | 峰值位置 |
| k_neg | 负向 k 参数 |
| lam_neg | 负向 λ 参数 |
| 说明 | 实验描述 |

## 🎯 19组预设配置

| 策略 | 组别 | 说明 |
|-----|------|------|
| **Mimic** | G1-G3 | 模拟 PPO 行为 |
| **Legacy** | G4 | 前冠军复刻 |
| **Math** | G5-G8 | 标准钟形分布（G5为推荐） |
| **Drift** | G9-G12 | 左偏和深锚实验 |
| **NegDef** | G13-G15 | 负向防御策略 |
| **Wild** | G16-G19 | 探索性配置 |

详细参数见 `experiment_configs_summary.txt` 或 `experiment_tracking.md`。

## 💡 工作流程示例

```bash
# 1. 初始化：生成所有脚本
python3 generate_experiment_scripts.py
# 提示：所有实验的说明列为空（-），可自由填写

# 2. 追踪：在 experiment_tracking.md 中标记实验状态和备注
vim experiment_tracking.md
# 填写：G1 运行在 node01，状态为"运行中"，说明为"测试基线配置"

# 3. 执行实验
bash recipe/is_shape/run/qwen3-4b-base/gamma/gamma_is_ppo_epochs_16_1.0_0.8_3.0_1.5_G1.sh

# 4. 更新状态和记录结果
vim experiment_tracking.md
# 更新：G1 状态改为"已完成"，说明改为"Loss 收敛良好，准确率 85%"

# 5. 添加新配置（可选）
vim generate_experiment_scripts.py
# 在 EXPERIMENT_CONFIGS 中添加 G20

# 6. 重新生成（保留已有追踪数据）
python3 generate_experiment_scripts.py
# ✓ G1-G19 的机器、状态和说明信息保留
# ✓ G20 自动添加（机器、状态和说明为空）
```

## ⚙️ 脚本配置

### 当前默认设置
- 数据集: `dapo_math`
- 模型: `Qwen3-4B-Base`
- Batch Size: 256
- PPO Epochs: 16
- GPU数量: 8

### 修改默认设置
编辑 `generate_experiment_scripts.py` 中的 `SCRIPT_TEMPLATE` 部分。

## 🔍 查看生成的文件

```bash
# 列出所有实验脚本
ls -1 gamma_is_ppo_epochs_16_*.sh

# 查看追踪表
cat experiment_tracking.md

# 查看配置总结
cat experiment_configs_summary.txt

# 查看批量运行脚本
cat run_all_experiments.sh
```

## ⚠️ 注意事项

1. **只修改可编辑列**：只修改「运行机器」、「运行状态」和「说明」列，其他列会在重新生成时自动更新
2. **使用 Group 作为标识**：脚本通过 Group 列（如 G1, G2）识别实验，修改 Group 会导致数据无法匹配
3. **保留表格格式**：不要修改 markdown 表格的结构（列数、分隔符等）
4. **说明列自由填写**：说明列初始为空（`-`），你可以填写任何内容，如实验结果、遇到的问题、参数调整记录等
5. **参考原始说明**：如果需要，可以从表格下方的「配置参考」部分复制原始说明

## 🆘 故障排除

**问题：重新生成后用户数据丢失**
- 检查 markdown 表格格式是否正确（列数应为10列）
- 确认 Group 列的值没有被修改
- 查看终端输出："✓ 已读取现有追踪数据，保留 X 个实验的用户填写信息"
- 确保「运行机器」、「运行状态」和「说明」列的值不是占位符（如"待填写"）

**问题：新增配置没有出现**
- 检查 `EXPERIMENT_CONFIGS` 语法是否正确
- 确认逗号和括号是否配对
- 运行脚本查看错误信息

**问题：脚本执行失败**
- 检查路径是否正确
- 确认数据文件是否存在
- 查看脚本中的模型路径配置

---

*生成时间：2025-12-24*
