# Semi On-Policy Distillation 使用指南

## 核心概念

### 问题背景

在复杂任务（如数学推理）中，小模型直接学习 `p(answer | problem)` 很困难：
- 推理空间巨大
- 需要大量训练数据
- 容易学到错误的捷径

### 解决方案：Semi On-Policy Distillation

通过条件生成缓解问题：
```
p(reasoning | problem, answer*, prompt)
```

其中：
- `problem`: 原始问题
- `answer*`: 大模型/Oracle 生成的标准答案
- `prompt`: 引导性提示

### 关键挑战：Off-Policy Gap

如果只用大模型生成数据（off-policy），小模型在推理时会偏离训练分布，导致：
- 累积误差
- 泛化性差
- 实际性能下降

## Optimal Sampling 的解决方案

### 架构设计

```
Teacher Model (π_t)          Theta Model (π_θ)
     Large                       Small
       ↓                           ↓
"Problem + Answer"          "Problem only"
       ↓                           ↓
  High Quality              On-Policy Distribution
       ↓                           ↓
       └──────── Mix (α) ─────────┘
                   ↓
           q*(y|x) = π_θ^(1-α) × π_t^α
                   ↓
         High-Quality On-Policy Data
```

### 关键特性

1. **不同的输入**：
   - Teacher 看到答案 → 生成高质量推理
   - Student 不看答案 → 保持 on-policy

2. **最优混合**：
   - α 由 KL 对称性自动计算
   - 平衡质量和 on-policy 覆盖率

3. **渐进式采样**：
   - 从不同前缀采样
   - 提高数据多样性

## 使用示例

### 基础用法

```python
from optimal_sampling import OptimalSamplingV1

# 初始化
sampler = OptimalSamplingV1(
    model_teacher="Qwen/Qwen2.5-32B-Instruct",
    model_theta="Qwen/Qwen2.5-7B-Instruct",

    teacher_system_prompt="Given problem and answer, explain reasoning.",
    theta_system_prompt="You are a math problem solver.",

    alpha_method="kl_symmetry",
)

# 单个问题
problem = "If 2x + 3 = 7, solve for x."
answer = "x = 2"

teacher_prompt = f"Problem: {problem}\nAnswer: {answer}\nReasoning:"
student_prompt = f"Problem: {problem}\nReasoning:"

output = sampler.generate(
    prompts=[teacher_prompt],
    theta_prompts=[student_prompt],
    max_tokens=512,
)

print(output.generated_texts[0])
```

### 批量处理

```python
# 准备数据集
problems = [
    {"question": "What is 2+2?", "answer": "4"},
    {"question": "What is 3×5?", "answer": "15"},
    # ... more problems
]

# 批量生成
teacher_prompts = [
    f"Q: {p['question']}\nA: {p['answer']}\nExplain:"
    for p in problems
]

student_prompts = [
    f"Q: {p['question']}\nSolve:"
    for p in problems
]

outputs = sampler.generate(
    prompts=teacher_prompts,
    theta_prompts=student_prompts,
    max_tokens=512,
)

# 构建训练数据
training_data = [
    {
        "problem": p["question"],
        "reasoning": outputs.generated_texts[i],
        "answer": p["answer"],
    }
    for i, p in enumerate(problems)
]
```

### 多前缀策略

```python
# 定义不同的前缀模板
templates = [
    ("Problem: {q}\nAnswer: {a}\nDetailed steps:",
     "Problem: {q}\nSteps:"),

    ("Question: {q}\nSolution: {a}\nExplain:",
     "Question: {q}\nSolve:"),

    ("Given: {q}\nResult: {a}\nProof:",
     "Given: {q}\nProve:"),
]

# 对每个问题用多个模板采样
all_samples = []
for problem, answer in dataset:
    for teacher_tmpl, student_tmpl in templates:
        teacher_prompt = teacher_tmpl.format(q=problem, a=answer)
        student_prompt = student_tmpl.format(q=problem)

        output = sampler.generate(
            prompts=[teacher_prompt],
            theta_prompts=[student_prompt],
            max_tokens=512,
        )

        all_samples.append({
            "problem": problem,
            "answer": answer,
            "reasoning": output.generated_texts[0],
            "template": teacher_tmpl,
        })

# 结果：每个问题有多个不同风格的推理
```

### 质量过滤

```python
def is_high_quality(sample):
    """过滤低质量样本"""
    reasoning = sample["reasoning"]
    answer = sample["answer"]

    # 1. 检查长度
    if len(reasoning) < 100:
        return False

    # 2. 检查答案是否出现
    if answer.lower() not in reasoning.lower():
        return False

    # 3. 检查推理步骤标记
    step_markers = ["step", "first", "then", "finally"]
    if not any(marker in reasoning.lower() for marker in step_markers):
        return False

    return True

# 过滤数据
filtered_data = [s for s in all_samples if is_high_quality(s)]
print(f"Kept {len(filtered_data)}/{len(all_samples)} samples")
```

## 配置参数

### Alpha 方法

```python
# KL 对称性（推荐）
alpha_method="kl_symmetry"
# 自动平衡 D_KL(q*||π_θ) = D_KL(q*||π_t)

# ESS 平衡
alpha_method="ess_balance"
# 基于有效样本量平衡

# 固定值
alpha_method="fixed"
fixed_alpha=0.7  # 70% teacher, 30% student
```

### 温度控制

```python
# 较低温度：更确定的输出
temperature=0.7  # 适合数学推理

# 中等温度：平衡多样性和质量
temperature=0.8  # 默认推荐

# 较高温度：更多样化的输出
temperature=1.0  # 适合创意任务
```

### System Prompt 设计

```python
# Teacher: 强调解释清晰度
teacher_system_prompt = """
You are a math expert teacher. Given a problem and its correct answer,
explain the reasoning in a clear, step-by-step manner that a student
can follow and learn from.
"""

# Student: 强调独立推理
theta_system_prompt = """
You are a diligent math student. When given a problem, show your
complete reasoning process, explaining each step clearly.
"""
```

## 训练工作流

### 1. 数据生成

```python
# 生成训练数据
training_data = []

for batch in dataset.batches(batch_size=32):
    teacher_prompts = [...]
    student_prompts = [...]

    outputs = sampler.generate(
        prompts=teacher_prompts,
        theta_prompts=student_prompts,
        max_tokens=512,
    )

    training_data.extend(outputs)

# 保存数据
save_jsonl(training_data, "reasoning_data.jsonl")
```

### 2. 训练学生模型

```python
# 使用生成的数据微调学生模型
from transformers import Trainer

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B")

# 格式化训练数据
train_dataset = [
    {
        "input": sample["problem"],
        "output": sample["reasoning"],
    }
    for sample in training_data
]

# 训练
trainer = Trainer(model=model, train_dataset=train_dataset, ...)
trainer.train()
```

### 3. 迭代改进

```python
# Round 1: 用 32B 生成数据训练 7B
sampler_r1 = OptimalSamplingV1(
    model_teacher="Qwen/Qwen2.5-32B",
    model_theta="Qwen/Qwen2.5-7B",
)

# Round 2: 用训练后的 7B 作为 theta
sampler_r2 = OptimalSamplingV1(
    model_teacher="Qwen/Qwen2.5-32B",
    model_theta="path/to/trained-7B",  # 使用训练后的模型
)

# 继续生成更多数据...
```

## 最佳实践

### 1. 选择合适的模型组合

- **Teacher**: 尽可能大的高质量模型
- **Student**: 目标部署的模型大小
- **推荐比例**: Teacher 是 Student 的 2-5 倍大小

### 2. 前缀设计原则

- Teacher 前缀：明确包含答案，引导推理
- Student 前缀：只包含问题，鼓励独立思考
- 保持格式一致性

### 3. 批量大小

```python
# GPU 内存充足：大批量
batch_size = 64
outputs = sampler.generate(prompts=[...] * batch_size)

# GPU 内存受限：小批量多轮
for batch in batches(dataset, batch_size=16):
    outputs = sampler.generate(prompts=batch)
```

### 4. 质量控制

- 设置最小推理长度
- 验证答案出现在推理中
- 检查推理步骤的完整性
- 人工抽查样本质量

## 性能优化

### 首次运行

- 初始化慢（模型加载 + 编译）
- 预期时间：~30-60秒

### 后续生成

- KV cache 生效
- 编译图复用
- 速度提升 4-5x

### 推荐设置

```python
sampler = OptimalSamplingV1(
    model_teacher="Qwen/Qwen2.5-32B",
    model_theta="Qwen/Qwen2.5-7B",
    gpu_memory_utilization=0.45,  # 为 theta 模型留空间
    track_alpha_stats=False,       # 关闭统计以提速
)
```

## 常见问题

### Q: Student 是否完全不能看答案？

A: 在生成训练数据时不能看，但在**使用生成的数据训练时**，答案会作为标签。这样：
- 生成时：Student 保持 on-policy（不看答案）
- 训练时：Student 学习从问题推理到答案

### Q: 为什么不直接用 Teacher 模型生成数据？

A: 纯 Teacher 生成的数据是 off-policy 的：
- Student 推理时的分布 ≠ Teacher 的分布
- 导致累积误差和泛化性差
- Optimal mixing 缓解这个问题

### Q: α 的值代表什么？

A: α 是 Teacher 的权重：
- α = 0.5: 50% Teacher, 50% Student
- α = 0.8: 80% Teacher, 20% Student
- α 越大，越接近 off-policy（质量高但 gap 大）
- α 越小，越接近 on-policy（gap 小但质量可能差）

KL 对称性自动找到最优平衡点。

### Q: 能否用于其他任务？

A: 可以！只要满足：
1. 有高质量的 Oracle/大模型
2. 需要学习复杂的推理过程
3. Off-policy gap 是个问题

适用场景：
- 代码生成（给定正确代码，生成解释）
- 定理证明（给定证明，生成推导）
- 科学推理（给定结论，生成论证）

## 参考资料

- 理论基础：`theory/proof_final.md`
- 配置示例：`configs/distillation.yaml`
- 代码示例：`examples/distillation_correct.py`
