# 实验与概念验证 (Experiments)

本目录包含理论验证实验、概念测试和可视化代码。

## 📚 主要文件

### 实验设计

**[experiment_design.md](experiment_design.md)** ⭐ **主要文档**
- 完整的实验设计方案
- 5个核心实验：
  1. 主对比实验（q* vs 7种baseline）
  2. 消融实验（alpha方法对比）
  3. 扩展性实验（模型规模、任务多样性）
  4. 投机采样实验
  5. 长期训练实验
- 评估指标体系
- 预期结果和时间线（6-9周）

**[experiment_quick_reference.md](experiment_quick_reference.md)**
- 快速参考指南
- MVP实验（1周，GPT-2）
- 关键指标速查表
- 实施检查清单
- 快速启动命令

### 实验框架

**[run_experiments.py](run_experiments.py)**
- 通用实验训练框架
- 支持多种方法：
  - q* (kl_symmetry, entropy, fixed)
  - PPO
  - SFT
  - DPO
- 包含占位符，需要填充实际模型加载等逻辑

## 🧪 理论验证脚本

### Alpha理论验证

**[verify_alpha_theory.py](verify_alpha_theory.py)**
- 验证KL对称条件
- 测试二分法求解alpha
- 检查ESS平衡

**[compare_alpha_principles.py](compare_alpha_principles.py)**
- 对比3种alpha计算方法
- 分析精度和速度权衡

### Fisher信息测试

**[test_fisher_information.py](test_fisher_information.py)**
- 验证Fisher信息计算
- 测试ESS与Fisher信息的关系
- Cramér-Rao界验证

**[test_learning_progress.py](test_learning_progress.py)**
- 测试学习进度跟踪
- 样本效率分析

### 可视化

**[visualize_q_star_2d.py](visualize_q_star_2d.py)**
- 2D Gaussian分布下的q*可视化
- 9种不同情况的对比
- Alpha演化动画
- 生成输出：
  - `q_star_behavior_2d.png`
  - `q_alpha_evolution.png`

### 反思与分析

**[reflection_on_proof.py](reflection_on_proof.py)**
- 理论证明的反思
- 边界条件分析
- 潜在改进方向

## 🚀 快速开始

### 1. 理论验证

```bash
# 验证alpha计算是否正确
python verify_alpha_theory.py

# 对比不同alpha方法
python compare_alpha_principles.py

# 测试Fisher信息
python test_fisher_information.py
```

### 2. 可视化

```bash
# 生成2D可视化
python visualize_q_star_2d.py
```

### 3. 运行实验（需要先配置）

```bash
# GPT-2快速验证
python run_experiments.py \
    --method q_star \
    --model gpt2 \
    --num_steps 1000 \
    --use_wandb

# LLaMA-7B主实验
python run_experiments.py \
    --method q_star \
    --model llama-7b \
    --num_steps 5000 \
    --use_wandb
```

## 📊 实验体系

### 阶段1: 快速验证（1-2周）

**目标**: 验证核心假设

**设置**:
- 模型: GPT-2 (124M)
- 数据: HH-RLHF (10K)
- 对比: PPO vs q*

**成功标准**:
- [ ] ESS ratio ∈ [0.8, 1.2]
- [ ] 样本效率 > PPO
- [ ] 训练稳定

### 阶段2: 主要实验（3-4周）

**目标**: 完整对比和消融实验

**设置**:
- 模型: LLaMA-7B
- 数据: HH-RLHF (100K)
- 对比: 7种方法

**评估**:
- Win rate (GPT-4评估)
- 学习曲线
- ESS平衡
- 训练稳定性

### 阶段3: 扩展实验（2-3周）

**目标**: 泛化性验证

**维度**:
- 模型规模: GPT-2 → LLaMA-13B
- 任务: 对话 / 摘要 / 代码
- 数据规模: 10K → 500K

## 🔬 关键指标

| 指标 | 计算方法 | 目标值 | 重要性 |
|------|---------|--------|--------|
| **Win Rate** | GPT-4评估 | > 65% | ⭐⭐⭐ |
| **样本效率** | Performance/Samples | > 1.2x | ⭐⭐⭐ |
| **ESS Ratio** | ESS_θ / ESS_t | ∈ [0.9, 1.1] | ⭐⭐⭐ |
| **α* 分布** | mean(α*) | ∈ [0.3, 0.7] | ⭐⭐ |
| **梯度方差** | std(∇J) | 最小 | ⭐⭐ |
| **KL散度** | D_KL(π_θ \|\| π_init) | 适中 | ⭐ |

## 📈 预期结果

根据理论分析，q* 应该达到：

```
方法          Win Rate    样本效率    ESS Ratio    训练时间
---------------------------------------------------------
PPO           60±2%       1.00x       N/A          10h
DPO           65±1%       1.20x       N/A          8h
q* (ours)     68±1%       1.30x       0.98±0.05    12h
```

## 🔧 实现检查清单

### 阶段1：环境准备
- [ ] 安装依赖 (transformers, torch, wandb)
- [ ] 下载模型 (LLaMA-7B, reward model)
- [ ] 准备数据集 (HH-RLHF)
- [ ] 配置GPU (8×A100)

### 阶段2：代码实现
- [ ] 完善 OptimalSamplingDistribution 类
- [ ] 实现 q* 训练循环
- [ ] 实现baseline方法 (PPO, DPO)
- [ ] 实现评估逻辑 (GPT-4, ROUGE)

### 阶段3：快速验证
- [ ] 在GPT-2上跑通
- [ ] 验证ESS ratio
- [ ] 检查梯度/loss正常

### 阶段4：主实验
- [ ] 运行所有baseline (7种方法)
- [ ] 收集所有指标
- [ ] 生成学习曲线图

### 阶段5：分析报告
- [ ] 统计显著性检验
- [ ] 消融实验分析
- [ ] 撰写实验报告

## 💡 实验技巧

### 调试建议

1. **先用小模型验证**
   - GPT-2 (124M) 快速迭代
   - 验证算法正确性

2. **监控关键信号**
   ```python
   assert 0.1 < alpha_star.mean() < 0.9
   assert 0.5 < ess_ratio.mean() < 2.0
   assert not torch.isnan(loss)
   ```

3. **可视化帮助理解**
   - 绘制 α*(t) 演化
   - 绘制 ESS ratio 分布
   - 绘制权重分布

### 常见问题

**Q: α* 总是接近0或1？**
- 检查 π_θ 和 π_t 是否定义正确
- 检查温度参数

**Q: ESS ratio 远离1？**
- 检查KL对称求解是否收敛
- 增加迭代次数

**Q: 训练不稳定？**
- 降低学习率
- 增加KL惩罚系数 β
- Clip梯度范数

## 📊 可视化示例

运行 `visualize_q_star_2d.py` 生成的图片展示：

### 1. q* 行为（9种情况）
- 相同方差，不同均值
- 不同方差，相同均值
- 方差和均值都不同
- 不同相关性

### 2. α 演化
- 从 α=0 (纯π_t) 到 α=1 (纯π_θ)
- 标记最优的 α*

**关键发现**:
- α* 反映分布的相对"强度"
- q* 集中在两个分布的重叠区域
- KL对称确保两个分布公平对待

## 🔗 相关资源

- **理论基础**: [../theory/proof_final.md](../theory/proof_final.md)
- **生产实现**: [../production/](../production/)
- **使用文档**: [../docs/](../docs/)

## 📝 注意事项

1. `run_experiments.py` 中的很多函数是占位符（返回随机数据）
2. 实际使用需要填充：
   - 模型加载逻辑
   - 数据集加载
   - Reward model推理
   - 评估逻辑

3. 建议先在 [../production/](../production/) 使用完整实现

---

**下一步**: 完成快速验证实验，然后进行主对比实验！
