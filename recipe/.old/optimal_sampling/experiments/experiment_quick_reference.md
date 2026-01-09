# 实验方案总结（Quick Reference）

## 📋 核心实验列表

### **实验1：主对比实验** ⭐⭐⭐（最重要）

**目标**: 验证 q* 优于所有baseline

**对比方法**:
1. On-policy (PPO)
2. Off-policy (from π_t)
3. SFT (忽略权重)
4. DPO
5. Uniform α (α=0.5)
6. Entropy α
7. **q* (KL对称)** ← 我们的方法

**数据**: Anthropic HH-RLHF (170K)
**模型**: LLaMA-7B
**样本数**: 100K per method

**关键指标**:
- ✅ Win rate vs baseline (GPT-4评估)
- ✅ 样本效率 (学习曲线)
- ✅ ESS ratio ≈ 1 (理论验证)
- ✅ 训练稳定性 (梯度方差)

**预期结果**:
```
方法          Win Rate    样本效率    ESS Ratio
---------------------------------------------------
PPO           60%         1.0x        N/A
DPO           65%         1.2x        N/A
q* (ours)     68%         1.3x        ~1.0 ✓
```

**时间**: 3-4周（8×A100）

---

### **实验2：消融实验** ⭐⭐

**目标**: 验证设计选择

**变量**:
- α的确定方法 (固定 vs 熵公式 vs KL对称)
- 计算精度 (迭代次数)
- 温度参数

**关键问题**:
- KL对称是否真的优于固定α？
- 熵公式的误差多大？

**时间**: 1-2周

---

### **实验3：扩展性实验** ⭐

**目标**: 验证泛化性

**维度**:
1. **模型规模**: GPT-2 → Pythia-6.9B → LLaMA-7B → LLaMA-13B
2. **任务多样性**: 对话 / 摘要 / 代码生成
3. **数据规模**: 10K / 50K / 100K / 500K

**关键问题**:
- 优势是否随模型规模增大？
- 在不同任务上是否一致有效？

**时间**: 2-3周

---

### **实验4：投机采样** ⭐ (可选)

**目标**: 验证加速效果

**设置**:
- Token级rejection sampling
- 候选数 k ∈ {3, 5, 10}

**预期**: 加速 1.4-1.6倍，完全保持分布

**时间**: 1周

---

## 🎯 最小可行实验（MVP）

如果时间/资源有限，优先做：

### **快速验证（1周）**

**模型**: GPT-2 (124M)
**数据**: HH-RLHF (10K样本)
**对比**: PPO vs q*

**检验点**:
1. [ ] ESS ratio ∈ [0.8, 1.2]
2. [ ] 样本效率 > PPO
3. [ ] 训练稳定（梯度方差 < PPO）

**如果通过** → 进行主对比实验
**如果不通过** → 调试理论实现

---

## 📊 关键指标速查表

| 指标 | 计算方法 | 目标值 | 重要性 |
|------|---------|--------|--------|
| **Win Rate** | GPT-4评估 | > 65% | ⭐⭐⭐ |
| **样本效率** | Performance/Samples | > 1.2x | ⭐⭐⭐ |
| **ESS Ratio** | ESS_θ / ESS_t | ∈ [0.9, 1.1] | ⭐⭐⭐ |
| **α* 分布** | mean(α*) | ∈ [0.3, 0.7] | ⭐⭐ |
| **梯度方差** | std(∇J) | 最小 | ⭐⭐ |
| **KL散度** | D_KL(π_θ \|\| π_init) | 适中 | ⭐ |

---

## 🔧 实现检查清单

### 阶段1：环境准备
- [ ] 安装依赖 (transformers, torch, wandb)
- [ ] 下载模型 (LLaMA-7B, reward model)
- [ ] 准备数据集 (HH-RLHF)
- [ ] 配置GPU (8×A100)

### 阶段2：代码实现
- [ ] 实现 OptimalSamplingDistribution 类
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

---

## 💡 实验技巧

### **调试建议**

1. **先用小模型验证**
   - GPT-2 (124M) 可以快速迭代
   - 验证算法正确性

2. **监控关键信号**
   ```python
   # 每步都检查
   assert 0.1 < alpha_star.mean() < 0.9, "α* 异常"
   assert 0.5 < ess_ratio.mean() < 2.0, "ESS不平衡"
   assert not torch.isnan(loss), "Loss NaN"
   ```

3. **可视化帮助理解**
   - 绘制 α*(t) 随时间演化
   - 绘制ESS ratio分布（直方图）
   - 绘制权重分布

### **常见问题**

**Q: α* 总是接近0或1？**
- 检查 π_θ 和 π_t 是否定义正确
- 检查温度参数（太小会导致极端）

**Q: ESS ratio 远离1？**
- 检查KL对称求解是否收敛
- 增加迭代次数或提高精度

**Q: 训练不稳定？**
- 降低学习率
- 增加KL惩罚系数 β
- Clip梯度范数

---

## 📈 预期时间线

```
Week 1-2:  快速验证（GPT-2）
  ├─ 实现代码
  ├─ 调试算法
  └─ 初步结果

Week 3-6:  主对比实验（LLaMA-7B）
  ├─ 运行7种方法
  ├─ 收集所有指标
  └─ 统计分析

Week 7-8:  消融实验
  ├─ α选择方法
  ├─ 超参数影响
  └─ 敏感性分析

Week 9:    扩展实验（可选）
  └─ 模型规模/多任务

Week 10:   总结报告
  └─ 撰写论文/技术报告
```

**总计**: 6-10周

---

## 🎨 结果可视化模板

### **Figure 1: 学习曲线**
```
x轴: 样本数（0-100K）
y轴: Win Rate (%)
曲线: 7种方法
突出: q* 在上方
```

### **Figure 2: ESS平衡验证**
```
散点图:
x轴: ESS_θ
y轴: ESS_t
点: 每个训练step
对角线: y=x（完美平衡）
q*的点应该聚集在对角线附近
```

### **Figure 3: α*演化**
```
时间序列:
x轴: 训练steps
y轴: α* (0-1)
显示: 均值 ± 标准差
观察: 是否收敛？
```

### **Table 1: 主对比结果**
```
| 方法 | Win Rate | 样本效率 | ESS Ratio | 训练时间 |
|------|---------|---------|-----------|---------|
| PPO  | 60±2%   | 1.00x   | -         | 10h     |
| DPO  | 65±1%   | 1.20x   | -         | 8h      |
| q*   | 68±1%   | 1.30x   | 0.98±0.05 | 12h     |
```

---

## 🚀 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 快速验证（GPT-2）
python run_experiments.py \
    --method q_star \
    --model gpt2 \
    --num_steps 1000 \
    --batch_size 16 \
    --use_wandb

# 3. 主实验（LLaMA-7B）
python run_experiments.py \
    --method q_star \
    --model llama-7b \
    --num_steps 5000 \
    --batch_size 32 \
    --use_wandb

# 4. 运行所有baseline
for method in ppo dpo sft uniform_alpha; do
    python run_experiments.py --method $method --model llama-7b
done

# 5. 分析结果
python analyze_results.py --experiment_dir ./results
```

---

## 📚 相关文件

- **理论文档**: `proof_final.md`
- **实验设计**: `experiment_design.md`
- **代码框架**: `run_experiments.py`
- **投机采样**: `speculative_decoding_analysis.md`
- **可视化**: `visualize_q_star_2d.py`

---

## ✅ 成功标准

实验成功的标志：

1. **性能**: Win rate > 所有baseline
2. **效率**: 样本效率 > 1.2x
3. **理论**: ESS ratio ∈ [0.9, 1.1]
4. **稳定**: 梯度方差 < baseline
5. **可重复**: 3次运行结果一致

如果满足3/5以上 → 实验成功 ✓

---

**准备好开始实验了吗？从快速验证开始！** 🚀
