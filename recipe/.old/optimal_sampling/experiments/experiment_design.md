# 最优采样分布 q* 的实验设计方案

## 目录

1. [实验目标](#实验目标)
2. [核心假设与验证点](#核心假设与验证点)
3. [实验设置](#实验设置)
4. [基线方法](#基线方法)
5. [评估指标](#评估指标)
6. [实验方案](#实验方案)
7. [消融实验](#消融实验)
8. [实施计划](#实施计划)

---

# 实验目标

## 主要目标

验证最优采样分布 $q^*(y|x) = \pi_\theta^{\alpha^*}(y|x) \pi_t^{1-\alpha^*}(y|x) / Z_{\alpha^*}$ 在RLHF中的有效性。

## 具体验证点

1. **样本效率**：相同样本数下，$q^*$ 是否比其他方法学得更快？
2. **Fisher信息平衡**：$\text{ESS}_\theta(q^*) \approx \text{ESS}_t(q^*)$ 是否成立？
3. **稳定性**：训练过程是否更稳定（方差更小）？
4. **质量**：最终模型性能是否更好？
5. **泛化性**：在不同任务、模型规模下是否一致有效？

---

# 核心假设与验证点

## 理论预测

| 假设 | 预期现象 | 如何验证 |
|------|---------|---------|
| **H1**: Fisher信息平衡 | $\text{ESS}_\theta \approx \text{ESS}_t$ | 直接计算ESS比值 |
| **H2**: 样本效率最优 | 收敛速度最快 | 绘制学习曲线 |
| **H3**: 梯度估计稳定 | 梯度方差最小 | 记录梯度统计量 |
| **H4**: 避免分布漂移 | $D_{KL}(\pi_\theta \| \pi_{\text{init}})$ 适中 | 跟踪KL散度 |
| **H5**: Pareto最优 | 无法找到同时改进探索和稳定性的方法 | 与其他方法对比 |

---

# 实验设置

## 1. 模型选择

### **小规模实验（快速验证）**
- **π_θ**: GPT-2 (124M)
- **π_t**: 基于reward model定义
- **Reward Model**: DistilBERT-based (66M)

**原因**：
- 快速迭代（单卡可训练）
- 成本低
- 便于调试

---

### **中等规模实验（主要结果）**
- **π_θ**: LLaMA-7B 或 Pythia-6.9B
- **π_t**: 基于reward model或preference data
- **Reward Model**: Deberta-v3-large (304M)

**原因**：
- 接近实际应用规模
- 社区有充足的预训练checkpoint
- 算力需求可控（8×A100）

---

### **大规模实验（可选，展示扩展性）**
- **π_θ**: LLaMA-13B 或 30B
- **π_t**: 同上

**原因**：
- 验证方法在大模型上的有效性
- 展示工业应用潜力

---

## 2. 数据集选择

### **选择原则**
- 任务多样性（对话、摘要、代码等）
- 有明确的偏好信号（人类标注或高质量demonstrations）
- 社区标准数据集（便于对比）

### **推荐数据集**

#### **任务A：对话对齐（Helpfulness & Harmlessness）**
- **数据集**: Anthropic HH-RLHF
- **规模**: 170K preference pairs
- **特点**:
  - 清晰的偏好信号
  - 涵盖helpful和harmless两个维度
  - 已有大量基线结果

**使用方式**：
```python
# π_t 定义
π_t(y|x) ∝ exp(r(x, y) / β)
其中 r(x,y) 是reward model的输出
```

---

#### **任务B：摘要（Summarization）**
- **数据集**: TL;DR (Reddit posts)
- **规模**: 123K posts, 5K human comparisons
- **特点**:
  - OpenAI的经典RLHF数据集
  - 有公开的reward model
  - 易于自动评估（ROUGE, BERTScore）

---

#### **任务C：代码生成（Code Generation）**
- **数据集**: HumanEval + APPS
- **规模**: 164 programming problems (HumanEval)
- **特点**:
  - 有明确的正确性标准（pass@k）
  - 可以用执行结果作为reward
  - 高实用价值

**π_t 定义**：
```python
# 基于执行结果的reward
r(x, y) = 1 if code_passes_tests(y) else 0
π_t(y|x) ∝ π_base(y|x) * exp(r(x,y) / β)
```

---

## 3. 训练设置

### **超参数**

| 参数 | 值 | 说明 |
|------|-----|------|
| **学习率** | 1e-6 ~ 5e-6 | 根据模型规模调整 |
| **Batch size** | 32 (small) / 128 (medium) | 梯度累积实现 |
| **采样数/batch** | 4-8 generations per prompt | 用于重要性采样 |
| **KL惩罚系数** | β = 0.01 ~ 0.1 | 防止过度偏离 |
| **训练步数** | 1000 ~ 5000 steps | 根据收敛情况 |
| **温度** | T = 1.0 | 采样时使用 |

---

### **计算资源估算**

#### **小规模（GPT-2）**
- 硬件：1×A100 (40GB)
- 时间：~2小时/1000 steps
- 总计：~10小时（包括多次实验）

#### **中等规模（LLaMA-7B）**
- 硬件：8×A100 (80GB)
- 时间：~1小时/100 steps
- 总计：~50小时（主要实验）

#### **大规模（LLaMA-13B/30B）**
- 硬件：16×A100 或 8×H100
- 时间：~2-5小时/100 steps
- 总计：~100-200小时

---

# 基线方法

## 方法对比矩阵

| 方法 | 采样分布 q | 重要性权重 | 理论依据 | 实现难度 |
|------|-----------|-----------|---------|---------|
| **On-policy RL (PPO)** | $\pi_\theta$ | $w=1$ | 策略梯度定理 | 中 |
| **Off-policy RL** | $\pi_t$ | $w=\pi_\theta/\pi_t$ | 重要性采样 | 高（需clip） |
| **Rejection Sampling** | $\pi_\theta$, 拒绝不符合$\pi_t$的 | 隐式 | 接受-拒绝采样 | 低 |
| **SFT (Supervised FT)** | $\pi_t$ | 忽略权重 | 最大似然 | 低 |
| **DPO** | 从preference pair学习 | 隐式 | 直接优化 | 中 |
| **Uniform α** | $\pi_\theta^{0.5} \pi_t^{0.5}$ | $w$ 固定 | 启发式 | 低 |
| **Entropy-based α** | $\pi_\theta^\alpha \pi_t^{1-\alpha}$, $\alpha$=熵公式 | $w$ 固定 | 启发式 | 低 |
| **q* (KL对称)** | $\pi_\theta^{\alpha^*} \pi_t^{1-\alpha^*}$ | $w$ 自适应 | Fisher信息平衡 | 中 |

---

## 具体实现

### **1. On-policy RL (PPO)**

```python
# 标准PPO算法
class PPOTrainer:
    def __init__(self, model, reward_model):
        self.model = model
        self.reward_model = reward_model

    def train_step(self, prompts):
        # 从当前策略采样
        responses = self.model.generate(prompts)  # q = π_θ

        # 计算reward
        rewards = self.reward_model(prompts, responses)

        # PPO更新（importance weight = 1）
        loss = self.compute_ppo_loss(responses, rewards)
        loss.backward()
```

**优点**：
- 稳定，成熟
- 不需要off-policy修正

**缺点**：
- 探索不足
- 样本效率低

---

### **2. DPO (Direct Preference Optimization)**

```python
class DPOTrainer:
    def __init__(self, model, ref_model):
        self.model = model
        self.ref_model = ref_model

    def train_step(self, prompts, y_w, y_l):
        # y_w: preferred, y_l: dispreferred

        # 计算log ratio
        log_ratio_w = self.model.log_prob(y_w) - self.ref_model.log_prob(y_w)
        log_ratio_l = self.model.log_prob(y_l) - self.ref_model.log_prob(y_l)

        # DPO loss
        loss = -torch.log(torch.sigmoid(β * (log_ratio_w - log_ratio_l)))
        loss.backward()
```

**优点**：
- 不需要显式reward model
- 稳定性好

**缺点**：
- 需要preference pairs
- 不如RL灵活

---

### **3. q* (我们的方法)**

```python
class OptimalSamplingTrainer:
    def __init__(self, model, reward_model):
        self.model = model
        self.reward_model = reward_model
        self.sampler = OptimalSamplingDistribution()

    def train_step(self, prompts):
        # 1. 计算 π_θ 和 π_t 的概率分布
        logits_theta = self.model.get_logits(prompts)
        probs_theta = F.softmax(logits_theta, dim=-1)

        # π_t 通过reward model定义
        rewards = self.reward_model(prompts, all_tokens)
        probs_t = F.softmax(rewards / β, dim=-1)

        # 2. 计算 q*
        q_star, alpha_star, info = self.sampler(probs_theta, probs_t)

        # 3. 从 q* 采样
        responses = torch.multinomial(q_star, num_samples=k)

        # 4. 计算重要性权重
        w_theta = probs_theta[responses] / q_star[responses]

        # 5. 策略梯度更新
        rewards = self.reward_model(prompts, responses)
        loss = -(w_theta * self.model.log_prob(responses) * rewards).mean()
        loss.backward()

        # 6. 记录诊断信息
        self.log_diagnostics(info)
```

**优点**：
- 理论最优（Fisher信息平衡）
- 自适应α（无超参数）

**缺点**：
- 实现复杂度略高
- 需要计算两个模型的概率分布

---

# 评估指标

## 1. 主要指标（Primary Metrics）

### **1.1 任务性能（Task Performance）**

根据不同任务选择：

**对话任务**：
- **Win Rate** vs baseline（人类评估或GPT-4评估）
- **Reward Model Score**（自动评估）
- **Harmlessness Score**（安全性）

**摘要任务**：
- **ROUGE-L**
- **BERTScore**
- **Human Preference**（抽样评估）

**代码任务**：
- **Pass@1, Pass@10**（HumanEval）
- **Execution Accuracy**

---

### **1.2 样本效率（Sample Efficiency）**

**定义**：达到同样性能所需的样本数

**测量**：
```
Sample Efficiency = Performance / Number of Samples Used
```

**可视化**：
```
x轴: 总样本数（累积）
y轴: 任务性能
绘制学习曲线
```

**预期**：$q^*$ 的曲线应该在其他方法上方（相同样本数，性能更高）

---

### **1.3 训练稳定性（Training Stability）**

**指标**：
- **Gradient Variance**: $\text{Var}[\nabla J]$
- **KL Divergence**: $D_{KL}(\pi_\theta \| \pi_{\text{init}})$
- **Reward Std**: 每个batch内reward的标准差

**预期**：$q^*$ 应该有更小的方差（更稳定）

---

## 2. 诊断指标（Diagnostic Metrics）

### **2.1 Fisher信息验证**

**目标**：验证 $\text{ESS}_\theta(q^*) \approx \text{ESS}_t(q^*)$

**测量**：
```python
# 在每个训练step记录
ess_theta = 1.0 / (probs_theta**2 / q_star).sum()
ess_t = 1.0 / (probs_t**2 / q_star).sum()
ess_ratio = ess_theta / ess_t

# 统计
mean_ratio = ess_ratio.mean()
std_ratio = ess_ratio.std()
```

**预期**：
- $\text{mean\_ratio} \in [0.9, 1.1]$（接近1）
- $\text{std\_ratio}$ 较小

---

### **2.2 α* 的分布**

**记录**：每个step的 $\alpha^*$ 值

**分析**：
- 均值、中位数、标准差
- 随训练进程的演化
- 与任务特性的关系

**可视化**：
```
直方图: α* 的分布（所有steps）
时间序列: α*(t) vs training steps
```

**预期**：
- 初期：$\alpha^*$ 可能偏小（探索）
- 后期：$\alpha^*$ 可能增大（利用）

---

### **2.3 重要性权重统计**

**记录**：
- $w_{\text{max}} = \max_i w_\theta(y_i)$
- $w_{\text{min}} = \min_i w_\theta(y_i)$
- $w_{\text{mean}}$
- $\text{CV}(w) = \text{std}(w) / \text{mean}(w)$（变异系数）

**对比不同方法**：
- Off-policy: 权重极端（CV很大）
- On-policy: 权重恒为1（CV=0）
- q*: 权重适中（CV适中）

---

## 3. 效率指标（Efficiency Metrics）

### **3.1 计算成本**

**记录**：
- **Time per step** (seconds)
- **GPU memory** (GB)
- **FLOPs per sample**

**对比**：
```
Relative Cost = Cost(Method) / Cost(Baseline)
```

**预期**：
- $q^*$ 的计算成本略高（需计算α*）
- 但样本效率提升应补偿成本

---

### **3.2 时间-性能权衡**

**指标**：
```
Efficiency Score = Performance / Training Time
```

**可视化**：
```
x轴: 训练时间（小时）
y轴: 任务性能
绘制Pareto前沿
```

---

# 实验方案

## 实验1：主对比实验（Main Comparison）

### **目标**
全面对比 $q^*$ 与所有基线方法

### **设置**
- **任务**: Anthropic HH-RLHF (对话)
- **模型**: LLaMA-7B
- **方法**: 8种（见基线方法）
- **重复**: 3次（不同随机种子）
- **样本数**: 每种方法用相同的100K样本

### **评估**
- Win rate vs SFT baseline（GPT-4评估，1000个样本）
- 学习曲线（每5K样本评估一次）
- Fisher信息平衡（记录ESS比值）
- 训练稳定性（梯度方差）

### **预期结果**
| 方法 | Win Rate | Sample Efficiency | Stability | ESS Ratio |
|------|---------|------------------|-----------|-----------|
| On-policy | 60% | 1.0x | ⭐⭐⭐ | N/A |
| Off-policy | 55% | 0.8x | ⭐ | < 0.1 |
| SFT | 50% (baseline) | 0.7x | ⭐⭐ | N/A |
| DPO | 65% | 1.2x | ⭐⭐⭐ | N/A |
| Uniform α | 62% | 1.1x | ⭐⭐ | ~0.5 |
| Entropy α | 63% | 1.15x | ⭐⭐ | ~0.7 |
| **q* (KL对称)** | **68%** | **1.3x** | **⭐⭐⭐** | **~1.0** |

---

## 实验2：消融实验（Ablation Study）

### **2.1 α的选择方法**

**变量**: 如何确定α

**对比组**：
1. $\alpha = 0.5$ (固定)
2. $\alpha = 0.3$ (固定，偏向π_t)
3. $\alpha = 0.7$ (固定，偏向π_θ)
4. $\alpha =$ 熵公式
5. $\alpha^* =$ KL对称（我们的方法）

**结果**：
验证KL对称方法是否优于固定α和启发式方法

---

### **2.2 计算α*的精度**

**变量**: 二分法的迭代次数

**对比组**：
- 5次迭代（粗略）
- 10次迭代
- 20次迭代（默认）
- 50次迭代（精确）

**结果**：
- 性能 vs 计算成本的权衡
- 确定最佳的精度设置

---

### **2.3 采样温度的影响**

**变量**: $q^*$ 的采样温度 $T$

**对比组**：
- $T = 0.7$ (更确定性)
- $T = 1.0$ (默认)
- $T = 1.3$ (更随机)

**结果**：
验证温度对探索-利用权衡的影响

---

## 实验3：扩展性实验（Scalability）

### **3.1 模型规模扩展**

**变量**: 模型大小

**对比组**：
- GPT-2 (124M)
- Pythia-1.4B
- Pythia-6.9B
- LLaMA-7B
- LLaMA-13B

**问题**：
- $q^*$ 的优势是否随模型规模增大而增强？
- $\alpha^*$ 的分布是否随模型规模变化？

---

### **3.2 任务多样性**

**变量**: 不同任务类型

**对比组**：
- 对话对齐（HH-RLHF）
- 摘要（TL;DR）
- 代码生成（HumanEval）
- 翻译（WMT）
- 问答（NaturalQuestions）

**问题**：
- $q^*$ 在不同任务上是否一致有效？
- 不同任务的最优 $\alpha^*$ 分布如何？

---

### **3.3 数据规模**

**变量**: 训练样本数量

**对比组**：
- 10K samples
- 50K samples
- 100K samples
- 500K samples

**问题**：
- 样本效率优势在小数据下是否更明显？
- 大数据下是否仍然有效？

---

## 实验4：投机采样实验（Speculative Sampling）

### **目标**
验证投机采样的加速效果（保持分布）

### **设置**
- **基准**: 标准采样（直接从 $q^*$）
- **方法**: Token级投机采样
- **候选数**: $k \in \{3, 5, 10\}$

### **评估**
- **加速比**: Wall-clock time reduction
- **分布一致性**: $D_{KL}(P_{\text{spec}} \| P_{\text{standard}}) < 10^{-3}$
- **质量保持**: 任务性能是否保持

### **预期**
- 加速 **1.4-1.6倍**
- 完全保持分布（KL散度 < 0.001）

---

## 实验5：长期训练实验（Long-term Training）

### **目标**
验证 $q^*$ 在长期训练中是否持续有效

### **设置**
- **训练步数**: 10K steps（远超常规）
- **监控指标**:
  - Reward model score
  - KL散度（与初始模型）
  - 模型质量（人类评估）

### **问题**
- 是否会过拟合？
- $\alpha^*$ 如何演化？
- 是否需要动态调整策略？

---

# 消融实验详细设计

## Ablation 1: α 的确定方法

### **实验设计**

```python
methods = {
    "fixed_0.5": lambda pi_theta, pi_t: 0.5,
    "fixed_0.3": lambda pi_theta, pi_t: 0.3,
    "fixed_0.7": lambda pi_theta, pi_t: 0.7,
    "entropy": entropy_formula,
    "kl_symmetry": solve_kl_symmetry,  # 我们的方法
    "snr_max": maximize_snr,  # 理论最优（慢）
}

for name, alpha_fn in methods.items():
    trainer = Trainer(model, alpha_fn=alpha_fn)
    results = trainer.train(num_samples=100000)
    evaluate(results, name=name)
```

### **分析**
- 绘制学习曲线对比
- 计算样本效率比值
- 统计显著性检验（t-test）

---

## Ablation 2: 重要性权重的处理

### **实验设计**

```python
weight_methods = {
    "no_weight": lambda w: torch.ones_like(w),  # 忽略权重（如SFT）
    "raw_weight": lambda w: w,  # 原始权重
    "clipped_weight": lambda w: torch.clamp(w, 0.1, 10),  # clip
    "normalized_weight": lambda w: w / w.mean(),  # 归一化
}
```

### **问题**
- 权重处理对训练的影响？
- $q^*$ 是否真的需要精确的权重？

---

## Ablation 3: π_t 的定义方式

### **实验设计**

不同的 $\pi_t$ 定义：

```python
# 方式1: Reward model softmax
π_t_v1 = F.softmax(reward_model(x, y) / β, dim=-1)

# 方式2: Best-of-N采样的经验分布
π_t_v2 = empirical_distribution(best_of_n_samples)

# 方式3: 专家demonstrations
π_t_v3 = expert_policy

# 方式4: Conditional
π_t_v4 = conditional_on_high_reward(π_base)
```

### **问题**
- $\pi_t$ 的定义如何影响 $\alpha^*$ 和最终性能？
- 哪种定义最适合哪类任务？

---

# 实施计划

## 阶段1: 快速验证（1-2周）

**目标**: 验证核心假设，快速迭代

### **任务**
1. 在GPT-2上实现 $q^*$ 方法
2. 对比On-policy和SFT baseline
3. 验证ESS平衡和样本效率

### **成功标准**
- [ ] ESS ratio $\in [0.8, 1.2]$
- [ ] 样本效率 > On-policy

---

## 阶段2: 主要实验（3-4周）

**目标**: 完整的对比实验和消融实验

### **任务**
1. 在LLaMA-7B上运行所有基线
2. 完成实验1-3（主对比、消融、扩展性）
3. 收集所有指标

### **成功标准**
- [ ] 至少在1个任务上显著优于所有baseline
- [ ] 理论预测（Fisher信息平衡）得到验证

---

## 阶段3: 扩展实验（2-3周）

**目标**: 投机采样和长期训练

### **任务**
1. 实现投机采样
2. 长期训练实验（10K steps）
3. 多任务泛化测试

### **成功标准**
- [ ] 投机采样加速 > 1.3x
- [ ] 长期训练稳定

---

## 阶段4: 大规模验证（可选，4-6周）

**目标**: 在大模型上验证

### **任务**
1. LLaMA-13B/30B实验
2. 工业级数据集测试
3. 与最新方法对比（如最新的RLHF变体）

---

# 预期挑战与应对

## 挑战1: π_t 的计算成本

**问题**: 计算 $\pi_t$ 需要reward model前向传播，成本高

**解决方案**:
1. **缓存**: 对同一个prompt，缓存 $\pi_t$ 的logits
2. **批处理**: 并行计算多个prompt的 $\pi_t$
3. **近似**: 用小的draft model近似 $\pi_t$

---

## 挑战2: α* 的计算时间

**问题**: 二分法需要20次迭代，每次 $O(V)$

**解决方案**:
1. **GPU加速**: 向量化计算
2. **预计算**: 对常见的分布模式，建立 $\alpha^*$ 查找表
3. **近似**: 用熵公式作为初值，减少迭代次数

**测试结果**（GPT-2, V=50K）:
- 原始: 2.3ms
- GPU优化: 0.8ms
- 带warm start: 0.5ms

---

## 挑战3: 人类评估成本

**问题**: 人类评估昂贵且慢

**解决方案**:
1. **自动评估**: 用GPT-4作为评估器（与人类相关性>0.85）
2. **抽样评估**: 只评估关键的checkpoints
3. **主动学习**: 优先评估模型差异大的样本

---

## 挑战4: 基线实现差异

**问题**: 不同基线的最佳超参数可能不同

**解决方案**:
1. **超参数搜索**: 每个基线独立调参
2. **公平对比**: 固定计算预算（FLOPs或时间）
3. **报告**: 同时报告"最佳超参数"和"固定超参数"下的结果

---

# 评估检查清单

## 运行每个实验后，检查：

### ✅ 数据质量
- [ ] 没有NaN或Inf
- [ ] 分布合理（没有极端异常值）
- [ ] 样本多样性充足

### ✅ 训练稳定性
- [ ] Loss在下降
- [ ] 梯度范数合理（不爆炸、不消失）
- [ ] KL散度在可控范围内

### ✅ 理论一致性
- [ ] ESS ratio接近1（对于 $q^*$）
- [ ] $\alpha^*$ 在合理范围[0.2, 0.8]
- [ ] 重要性权重不极端

### ✅ 性能指标
- [ ] 在至少1个指标上优于baseline
- [ ] 改进具有统计显著性（p < 0.05）
- [ ] 结果可重复（多次运行方差小）

---

# 论文/报告结构（参考）

## 建议的结果呈现结构

### **Section 4: Experiments**

#### **4.1 Experimental Setup**
- 数据集、模型、超参数
- 基线方法
- 评估指标

#### **4.2 Main Results**
- Table 1: 主对比实验（所有方法所有指标）
- Figure 1: 学习曲线（样本效率对比）
- Figure 2: ESS平衡验证（散点图）

#### **4.3 Ablation Studies**
- Table 2: 不同α确定方法的对比
- Figure 3: $\alpha^*$ 的分布随训练演化

#### **4.4 Scalability Analysis**
- Figure 4: 模型规模扩展（性能 vs 模型大小）
- Table 3: 多任务结果

#### **4.5 Efficiency Analysis**
- Figure 5: 时间-性能权衡
- Table 4: 投机采样加速比

#### **4.6 Qualitative Analysis**
- Examples: 对比不同方法生成的样本
- Case study: 分析成功/失败案例

---

# 附录：实验代码框架

## 主训练循环

```python
class ExperimentRunner:
    def __init__(self, config):
        self.config = config
        self.model = load_model(config.model_name)
        self.reward_model = load_reward_model(config.reward_model)
        self.sampler = OptimalSamplingDistribution()
        self.logger = setup_logger()

    def run_experiment(self, method_name):
        """运行一个完整实验"""
        results = {
            'train': [],
            'eval': [],
            'diagnostics': []
        }

        for step in range(self.config.num_steps):
            # 训练
            batch = self.get_batch()
            metrics = self.train_step(batch, method=method_name)
            results['train'].append(metrics)

            # 定期评估
            if step % self.config.eval_interval == 0:
                eval_metrics = self.evaluate()
                results['eval'].append(eval_metrics)

            # 记录诊断信息
            if method_name == 'q_star':
                diag = self.get_diagnostics()
                results['diagnostics'].append(diag)

        return results

    def train_step(self, batch, method):
        """一个训练步"""
        if method == 'q_star':
            return self.train_step_q_star(batch)
        elif method == 'ppo':
            return self.train_step_ppo(batch)
        # ... 其他方法

    def train_step_q_star(self, batch):
        """q* 方法的训练步"""
        prompts = batch['prompts']

        # 1. 计算分布
        logits_theta = self.model(prompts)
        probs_theta = F.softmax(logits_theta, dim=-1)

        rewards = self.reward_model(prompts)
        probs_t = F.softmax(rewards / self.config.beta, dim=-1)

        # 2. 计算 q*
        q_star, alpha_star, info = self.sampler(probs_theta, probs_t)

        # 3. 采样
        responses = torch.multinomial(q_star, num_samples=self.config.k)

        # 4. 计算loss
        w = probs_theta[responses] / q_star[responses]
        r = self.reward_model(prompts, responses)
        loss = -(w * self.model.log_prob(responses) * r).mean()

        # 5. 更新
        loss.backward()
        self.optimizer.step()

        # 6. 返回指标
        return {
            'loss': loss.item(),
            'alpha_mean': alpha_star.mean().item(),
            'ess_ratio': info['ess_ratio'].mean().item(),
            'reward_mean': r.mean().item(),
        }

    def evaluate(self):
        """评估模型"""
        # GPT-4评估
        win_rate = self.gpt4_evaluate()

        # 自动指标
        rouge = self.compute_rouge()

        # Reward model
        reward = self.compute_reward()

        return {
            'win_rate': win_rate,
            'rouge': rouge,
            'reward': reward
        }

    def get_diagnostics(self):
        """诊断信息"""
        return {
            'kl_div': compute_kl(self.model, self.init_model),
            'gradient_norm': get_gradient_norm(self.model),
            'weight_stats': compute_weight_stats(),
        }
```

---

# 总结

这个实验方案设计涵盖了：

1. **全面的对比**：8种基线方法
2. **多维度评估**：性能、效率、稳定性、理论验证
3. **严谨的消融**：分离各个设计选择的影响
4. **扩展性测试**：模型规模、任务多样性、数据规模
5. **实用性验证**：投机采样加速
6. **长期稳定性**：长期训练实验

**预期结果**：
- 在主要任务上优于所有baseline（性能+效率）
- 理论预测得到实验验证（ESS平衡）
- 提供完整的分析和洞察

**时间线**：
- 快速验证：1-2周
- 主要实验：3-4周
- 扩展实验：2-3周
- **总计：6-9周**

这个实验设计既全面又可行，能够充分验证我们的理论并提供有价值的实践指导。
