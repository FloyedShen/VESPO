# vLLM Batch Processing 与双 vLLM Core 对齐指南

本文档详细说明 Optimal Sampling V1 中批处理（batch processing）的关键 bug、vLLM 内部行为以及两个 vLLM 实例的对齐策略。

---

## 📋 目录

1. [Bug 总结](#bug-总结)
2. [vLLM 内部行为分析](#vllm-内部行为分析)
3. [两个 vLLM Core 的对齐策略](#两个-vllm-core-的对齐策略)
4. [关键要点总结](#关键要点总结)

---

## 🐛 Bug 总结

在批处理实现中发现了两个关键问题：

### Bug 1: Logits 索引理解错误（已修复）

#### 错误理解

最初认为 vLLM 的 logits tensor 大小是 `[current_batch_size, vocab_size]`，并尝试建立 Request ID → Batch Index 的映射。

```python
# ❌ 错误实现
request_id_to_batch_idx = {}
batch_idx = 0
for idx in self.enabled_requests:
    request_id_to_batch_idx[idx] = batch_idx
    batch_idx += 1

# 然后用 batch_idx 访问 logits
batch_idx = request_id_to_batch_idx[request_id]
logits[batch_idx] = mixed_logits
```

#### 正确理解

vLLM 使用**固定大小**的 logits tensor: `[max_num_reqs, vocab_size]`，可以**直接用 request index** 访问 logits。

```python
# ✅ 正确实现（与 vLLM 内置 LogitsProcessors 一致）
for request_idx, theta_logits in theta_logits_dict.items():
    logits[request_idx] = mixed_logits  # 直接使用 request_idx
```

#### 证据（来自 vLLM 源码）

vLLM 内置的 LogitsProcessors 都是直接用 request index 访问：

```python
# vllm/v1/sample/logits_processor/builtin.py

# LogitBiasLogitsProcessor
def apply(self, logits: torch.Tensor) -> torch.Tensor:
    if self.biases:
        logits[self.logits_slice] += self.bias_tensor
        # logits_slice = (req_indices, tok_indices)
    return logits

# MinTokensLogitsProcessor
def apply(self, logits: torch.Tensor) -> torch.Tensor:
    if self.min_toks:
        logits[self.logits_slice] = -float("inf")
        # logits_slice 包含 request indices
    return logits
```

### Bug 2: Original Prompt 索引错误（真正的 Bug）⚠️

这是导致批处理中所有请求都获取第一个 prompt 的**根本原因**。

#### Bug 代码

```python
# ❌ 错误：所有请求都取 original_prompts[0]
for index, params, prompt_tok_ids, output_tok_ids in batch_update.added:
    original_prompt = params.extra_args.get("original_prompts", [""])[0]
    #                                                              ^^^
    #                                                              总是 [0]！
```

#### 问题表现

```python
# 批处理请求
prompts = ["Problem 1", "Problem 2", "Problem 3"]
theta_prompts = ["Theta 1", "Theta 2", "Theta 3"]

# Bug 导致：
# Request 0 → original_prompts[0] = "Theta 1" ✅ 正确
# Request 1 → original_prompts[0] = "Theta 1" ❌ 错误！应该是 "Theta 2"
# Request 2 → original_prompts[0] = "Theta 1" ❌ 错误！应该是 "Theta 3"

# 结果：所有请求的 theta model 都看到了第一个问题的 prompt
```

#### 修复代码

```python
# ✅ 正确：根据 request index 获取对应 prompt
for index, params, prompt_tok_ids, output_tok_ids in batch_update.added:
    original_prompts_list = params.extra_args.get("original_prompts", [])
    original_prompt = original_prompts_list[index] if index < len(original_prompts_list) else ""
    #                                      ^^^^^
    #                                      使用 request index
```

#### 实际案例

**问题描述**：批处理两个数学问题时，第二个问题的输出混入了第一个问题的内容。

```python
# 输入
problem1 = "The operation ⊗ is defined as a⊗b = 3a+4b..."
problem2 = "Doug constructs a square window..."

# Bug 导致的输出（错误）
outputs.generated_texts[1]:
"<think>
Okay, let's try to figure out the problem step by step.
So, Doug has a square thing called a⊗ window..."
# ↑ 混入了 problem1 的 ⊗ 操作符

# 修复后的输出（正确）
outputs.generated_texts[1]:
"<think>
Doug constructs a square window using the following steps..."
# ✅ 只讨论 window 构造问题
```

---

## 🔍 vLLM 内部行为分析

### 1. Batch 和 Request 索引机制

vLLM 使用固定容量的稀疏数组来管理请求：

```
InputBatch 结构:
┌────────────────────────────────────────┐
│ max_num_reqs = 64 (固定容量)            │
├────────────────────────────────────────┤
│ _req_ids: [req0, req1, None, None,...] │ ← 稀疏数组
│                                        │
│ req_id_to_index: {                     │
│   "req0": 0,  ← request index          │
│   "req1": 1                            │
│ }                                      │
└────────────────────────────────────────┘

Logits Tensor:
┌─────────────────┐
│ [0]: logits_0   │ ← Request 0 的 logits
│ [1]: logits_1   │ ← Request 1 的 logits
│ [2]: unused     │
│ ...             │
│ [63]: unused    │
└─────────────────┘
Shape: [max_num_reqs, vocab_size]
```

**关键发现**：
- Logits tensor 大小固定为 `[max_num_reqs, vocab_size]`
- Request index 是在 batch 中的位置（0-based）
- 可以直接用 `logits[request_idx]` 访问对应请求的 logits

### 2. Index 复用机制（重要！）⚠️

当并发请求数超过 `max_num_reqs` 时，vLLM 会复用已完成请求的 index：

```python
# 场景：max_num_reqs = 64
# 时刻 T1: 64 个请求全满
requests = {0, 1, 2, ..., 63}

# 时刻 T2: Request 5 完成
# BatchUpdate: removed=[5]
# → Index 5 被释放

# 时刻 T3: 新请求 Request_new 加入
# BatchUpdate: added=[(5, params, prompt_toks, output_toks)]
#                     ^
#                     复用 index 5
```

#### Index 复用的安全处理

**关键**：必须按照 vLLM 规定的顺序处理 BatchUpdate：

```python
# ✅ 正确顺序（我们的实现）
def update_state(self, batch_update: BatchUpdate):
    # Step 1: 先处理 removed - 清理旧请求，释放 index
    for index in batch_update.removed:
        self.request_states.pop(index, None)
        self.enabled_requests.discard(index)
        self.alpha_history.pop(index, None)

    # Step 2: 再处理 added - 添加新请求，可能复用刚释放的 index
    for index, params, prompt_toks, output_toks in batch_update.added:
        self.request_states[index] = (...)  # 安全：旧数据已清理

    # Step 3: 最后处理 moved - 移动/交换请求
    for adx, bdx, direct in batch_update.moved:
        # ...
```

```python
# ❌ 错误顺序（会导致数据混淆）
def update_state(self, batch_update: BatchUpdate):
    # 先处理 added
    for index, params, ... in batch_update.added:
        self.request_states[index] = (...)  # 危险：可能覆盖还未清理的旧数据

    # 后处理 removed
    for index in batch_update.removed:
        self.request_states.pop(index, None)  # 可能删除刚添加的新数据！
```

#### Index 复用示例

```python
# Batch 状态演化
# ┌──────────────────────────────────────────────────┐
# │ T0: 初始状态 (3 个请求)                           │
# ├──────────────────────────────────────────────────┤
# │ Index 0: Request_A (prompt="What is AI?")        │
# │ Index 1: Request_B (prompt="What is ML?")        │
# │ Index 2: Request_C (prompt="What is DL?")        │
# └──────────────────────────────────────────────────┘
#
# ┌──────────────────────────────────────────────────┐
# │ T1: Request_A 完成                                │
# ├──────────────────────────────────────────────────┤
# │ BatchUpdate.removed = [0]                        │
# │ → 清理 Index 0 的所有数据                         │
# │   - request_states.pop(0)                        │
# │   - alpha_history.pop(0)                         │
# │   - 保存 alpha history 到文件                     │
# └──────────────────────────────────────────────────┘
#
# ┌──────────────────────────────────────────────────┐
# │ T2: 新请求 Request_D 加入，复用 Index 0          │
# ├──────────────────────────────────────────────────┤
# │ BatchUpdate.added = [                            │
# │   (0, params_D, prompt_toks_D, output_toks_D)    │
# │ ]                                                │
# │ → 安全：Index 0 已被清理，可以安全复用             │
# │   request_states[0] = (prompt_D, output_D, ...)  │
# │   alpha_history[0] = []  # 新的 alpha 历史        │
# └──────────────────────────────────────────────────┘
#
# 当前状态:
# Index 0: Request_D (prompt="What is RL?")  ← 复用
# Index 1: Request_B (prompt="What is ML?")
# Index 2: Request_C (prompt="What is DL?")
```

#### 潜在问题和解决方案

**问题 1**：如果处理顺序错误，可能导致数据混淆

```python
# ❌ 错误场景
# T1: Request_A (index=0) 还在运行
# T2: 错误地先处理 added，添加 Request_D 到 index 0
#     → 覆盖了 Request_A 的数据！
# T3: 再处理 removed，删除 index 0
#     → Request_D 的数据也被删除了！
```

**解决方案**：
- ✅ 严格按照 `removed → added → moved` 的顺序处理
- ✅ 在 `removed` 中彻底清理所有相关数据
- ✅ 在 `added` 中重新初始化所有数据

**问题 2**：Alpha history 文件名冲突

当 index 被复用时，不同请求可能使用相同的 index，导致文件名冲突：

```python
# Request_A (index=0) 完成，保存 alpha_history_0.json
# Request_D (index=0) 完成，保存 alpha_history_0.json  ← 覆盖！
```

**当前解决方案**：
- 每次 `removed` 时立即保存 alpha history
- 在主进程的 `generate()` 方法中立即读取并清理文件
- 使用 `time.sleep(0.01)` 确保文件 I/O 完成

**更好的方案**（未来优化）：
- 使用 UUID 或 timestamp 作为文件名的一部分
- 在 `extra_args` 中传递唯一的 request ID

```python
# 改进的文件命名
alpha_file = f"alpha_history_{request_uuid}_{index}.json"
# 或
alpha_file = f"alpha_history_{timestamp}_{index}.json"
```

### 3. BatchUpdate 和 LogitsProcessor 生命周期

vLLM 在每个生成 step 中调用 LogitsProcessor：

```python
# 每个 step:
1. update_state(batch_update)  # 更新请求状态
   ├─ batch_update.added: [(index, params, prompt_toks, output_toks), ...]
   ├─ batch_update.removed: [index1, index2, ...]
   └─ batch_update.moved: [(from_idx, to_idx, directionality), ...]

2. apply(logits)  # 处理 logits
   └─ logits shape: [max_num_reqs, vocab_size]

3. Sample token from logits

4. Append token to output_tok_ids (自动)
```

#### BatchUpdate 数据结构

```python
@dataclass(frozen=True)
class BatchUpdate:
    batch_size: int  # 当前 batch 中的请求数

    # 添加的请求：(index, params, prompt_tok_ids, output_tok_ids)
    added: Sequence[AddedRequest]

    # 移除的请求索引
    removed: Sequence[RemovedRequest]

    # 移动/交换的请求：(from_idx, to_idx, directionality)
    moved: Sequence[MovedRequest]
```

**重要**：`output_tok_ids` 是一个**引用**（list reference），vLLM 会自动在每个 step 后 append 新 token 到这个 list。

### 3. SamplingParams.extra_args 在批处理中的行为

**关键发现**：在批处理中，所有请求**共享同一个 SamplingParams 对象**！

```python
# optimal_sampling_v1.py 的 generate() 方法
sampling_params = SamplingParams(
    max_tokens=100,
    temperature=0.8,
    extra_args={
        "theta_model_path": "Qwen/Qwen2.5-1.5B",
        "original_prompts": ["prompt0", "prompt1", "prompt2"],  # ← List
        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        # 所有请求共享这个 list
    }
)

# vLLM 内部：所有请求都引用同一个 sampling_params
for i, prompt in enumerate(prompts):
    request = Request(
        prompt=prompt,
        sampling_params=sampling_params,  # ← 共享同一个对象
        ...
    )
```

**推论**：`extra_args` 中的 list 必须按 **request index** 索引，而不是用固定的 `[0]`。

---

## 🔗 两个 vLLM Core 的对齐策略

Optimal Sampling V1 使用嵌套的 vLLM 架构：

```
┌─────────────────────────────────────────────────────┐
│ Outer vLLM (Teacher Model π_t, 大模型)               │
│                                                     │
│  ┌────────────────────────────────────────┐         │
│  │ EngineCore (subprocess)                │         │
│  │                                        │         │
│  │  ┌──────────────────────────────────┐  │         │
│  │  │ OptimalSamplingLogitsProcessor  │  │         │
│  │  │                                  │  │         │
│  │  │  ┌────────────────────────────┐  │  │         │
│  │  │  │ Inner vLLM (Theta π_θ)     │  │  │         │
│  │  │  │ 小模型，获取 theta logits   │  │  │         │
│  │  │  └────────────────────────────┘  │  │         │
│  │  └──────────────────────────────────┘  │         │
│  └────────────────────────────────────────┘         │
└─────────────────────────────────────────────────────┘
```

### 对齐点 1: Prompt 对齐（Semi On-Policy Distillation）

**目标**：Teacher 和 Theta 看到**不同的初始 prompt**，但**相同的生成序列**。

#### 场景：隐式知识蒸馏

```python
# 示例：隐式知识蒸馏
teacher_prompt = "Problem: 2x+3=7\nAnswer: x=2\nReasoning:"
theta_prompt   = "Problem: 2x+3=7\nReasoning:"
                 # ↑ Theta 看不到答案

# Teacher tokenization:
teacher_tokens = tokenizer.encode(teacher_prompt)
# → [1, 2, 3, 4, 5, 6, 7, 8, 9]  (假设)

# Theta tokenization (单独进行):
theta_tokens = tokenizer.encode(theta_prompt)
# → [1, 2, 3, 4, 10]  (不同！因为没有 "Answer: x=2" 部分)

# 第一个生成 token: 11
# Teacher 看到: [1,2,3,4,5,6,7,8,9, 11]
#               ^^^^^^^^^^^^^^^^^ ^
#               teacher prompt    生成
#
# Theta 看到:   [1,2,3,4,10, 11]
#               ^^^^^^^^^ ^
#               theta prompt 生成
#               ↑ 不同前缀，相同后缀
```

#### 实现（在 `guide_model_v1.py`）

```python
def get_logits_for_requests(self, request_data: Dict[int, Dict]) -> Dict[int, torch.Tensor]:
    prompts = []
    for idx in indices:
        if request_data[idx].get("original_prompt"):
            # 1. 提取 teacher 已生成的 output tokens
            teacher_prompt_len = request_data[idx]["teacher_prompt_len"]
            full_sequence = request_data[idx]["token_ids"]
            output_tokens = full_sequence[teacher_prompt_len:]

            # 2. 单独 tokenize theta 的 prompt
            original_prompt = request_data[idx]["original_prompt"]

            if self.system_prompt:
                theta_prompt_text = f"{self.system_prompt}\n\n{original_prompt}"
            else:
                theta_prompt_text = original_prompt

            if self.enable_chat_template:
                messages = []
                if self.system_prompt:
                    messages.append({"role": "system", "content": self.system_prompt})
                if original_prompt:
                    messages.append({"role": "user", "content": original_prompt})
                theta_prompt_text = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )

            # 3. Tokenize theta's prompt
            theta_prompt_tokens = self.tokenizer.encode(theta_prompt_text, add_special_tokens=False)

            # 4. 组合：theta 的 prompt + 共享的 output tokens
            theta_full_sequence = theta_prompt_tokens + output_tokens
            #                     ^^^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^
            #                     theta 自己的 prompt    teacher 生成的

            prompts.append({"prompt_token_ids": theta_full_sequence})
        else:
            # Fallback: 使用 teacher 的 tokens（向后兼容）
            prompts.append({"prompt_token_ids": request_data[idx]["token_ids"]})

    # 获取 theta logits
    outputs = self.llm.generate(prompts=prompts, sampling_params=...)
    return result
```

### 对齐点 2: Request Index 对齐

**关键**：Teacher 的 request index 必须与 Theta 的 request index 一致。

```python
# logits_processor_v1.py

# update_state() 中：
for index, params, prompt_tok_ids, output_tok_ids in batch_update.added:
    # ✅ 用 request index 获取对应的 prompt
    original_prompts_list = params.extra_args.get("original_prompts", [])
    original_prompt = original_prompts_list[index]  # ← 关键！
    #                                      ^^^^^

    self.request_states[index] = (prompt_tok_ids, output_tok_ids, params, original_prompt)

# apply() 中：
request_data = {}
for idx in self.enabled_requests:
    if idx in self.request_states:
        prompt_tok_ids, output_tok_ids, params, original_prompt = self.request_states[idx]
        request_data[idx] = {
            "token_ids": full_sequence,
            "original_prompt": original_prompt,  # ← 正确的 prompt
            "teacher_prompt_len": len(prompt_tok_ids)
        }

# 获取 theta logits
theta_logits_dict = self.theta_model.get_logits_for_requests(request_data)
# → 返回 {idx: logits} 映射

# 混合 logits
for request_idx, theta_logits in theta_logits_dict.items():
    # ✅ 用相同的 request_idx 访问 teacher logits 和更新 logits
    logits_t = logits[request_idx]
    # ... compute mixed_logits ...
    logits[request_idx] = mixed_logits
```

### 对齐点 3: Output Tokens 对齐（自动对齐）

vLLM 通过**引用传递** `output_tok_ids` 实现自动同步：

```python
# BatchUpdate.added 中:
for index, params, prompt_tok_ids, output_tok_ids in batch_update.added:
    #                                ^^^^^^^^^^^^^^
    #                                这是一个 list reference！

    self.request_states[index] = (prompt_tok_ids, output_tok_ids, params, original_prompt)
    #                                              ^^^^^^^^^^^^^^
    #                                              保存引用

# 每次 apply() 时：
prompt_tok_ids, output_tok_ids, params, original_prompt = self.request_states[idx]
full_sequence = prompt_tok_ids + output_tok_ids.copy()
#                                ^^^^^^^^^^^^^^
#                                这个 list 已经被 vLLM 自动更新了！

# vLLM 在每个 step 后自动 append 新 token：
# Step 1: output_tok_ids = []
# Step 2: output_tok_ids = [token_1]  ← vLLM 自动 append
# Step 3: output_tok_ids = [token_1, token_2]  ← vLLM 自动 append
# ...
```

---

## ✅ 完整的对齐流程示例

```python
# Step 1: 初始化
sampler = OptimalSamplingV1(
    model_teacher="Qwen/Qwen2.5-3B",  # Outer vLLM
    model_theta="Qwen/Qwen2.5-1.5B",  # Inner vLLM
    alpha_method="kl_symmetry"
)

# Step 2: 批量生成（两个不同的问题）
outputs = sampler.generate(
    prompts=[
        "Problem: X\nAnswer: A\nReason:",  # Teacher prompt (index=0)
        "Problem: Y\nAnswer: B\nReason:"   # Teacher prompt (index=1)
    ],
    theta_prompts=[
        "Problem: X\nReason:",  # Theta prompt (index=0) - 没有答案
        "Problem: Y\nReason:"   # Theta prompt (index=1) - 没有答案
    ],
    max_tokens=100,
    temperature=0.8
)

# Step 3: vLLM 内部流程（每个生成 step）
# ┌──────────────────────────────────────────────────────┐
# │ Generation Loop (重复 max_tokens 次)                 │
# ├──────────────────────────────────────────────────────┤
# │ 1. update_state(batch_update) - 只在 step 0 调用     │
# │    ┌────────────────────────────────────────────┐    │
# │    │ Request 0:                                 │    │
# │    │  - teacher_prompt: "Problem: X\nAnswer:..."│    │
# │    │  - theta_prompt: "Problem: X\nReason:"     │    │
# │    │  - output_toks: [] (引用)                  │    │
# │    ├────────────────────────────────────────────┤    │
# │    │ Request 1:                                 │    │
# │    │  - teacher_prompt: "Problem: Y\nAnswer:..."│    │
# │    │  - theta_prompt: "Problem: Y\nReason:"     │    │
# │    │  - output_toks: [] (引用)                  │    │
# │    └────────────────────────────────────────────┘    │
# ├──────────────────────────────────────────────────────┤
# │ 2. apply(logits) - 每个 step 都调用                   │
# │    ┌────────────────────────────────────────────┐    │
# │    │ A. 构建 request_data                       │    │
# │    │    Request 0: {                            │    │
# │    │      "token_ids": teacher_prompt_toks +    │    │
# │    │                   output_toks,             │    │
# │    │      "original_prompt": "Problem: X...",   │    │
# │    │      "teacher_prompt_len": len(...)        │    │
# │    │    }                                       │    │
# │    │    Request 1: { ... }                      │    │
# │    ├────────────────────────────────────────────┤    │
# │    │ B. 获取 theta logits                       │    │
# │    │    theta_model.get_logits_for_requests()   │    │
# │    │    → 单独 tokenize theta prompt            │    │
# │    │    → 组合 theta_prompt_toks + output_toks  │    │
# │    │    → 获取 theta logits                     │    │
# │    ├────────────────────────────────────────────┤    │
# │    │ C. 混合 logits                             │    │
# │    │    for req_idx in [0, 1]:                  │    │
# │    │      logits_t = logits[req_idx]            │    │
# │    │      logits_theta = theta_logits[req_idx]  │    │
# │    │      alpha = compute_alpha(...)            │    │
# │    │      q_star = mix(logits_t, logits_theta)  │    │
# │    │      logits[req_idx] = q_star              │    │
# │    └────────────────────────────────────────────┘    │
# ├──────────────────────────────────────────────────────┤
# │ 3. Sample token from mixed logits                    │
# │    - Request 0: sample token_0                       │
# │    - Request 1: sample token_1                       │
# ├──────────────────────────────────────────────────────┤
# │ 4. Append token to output_tok_ids (vLLM 自动)        │
# │    - Request 0: output_toks[0].append(token_0)       │
# │    - Request 1: output_toks[1].append(token_1)       │
# └──────────────────────────────────────────────────────┘
# Repeat steps 2-4 until max_tokens or EOS
```

### 关键时刻的数据状态

假设两个问题，生成 3 个 token：

```python
# Initial state (Step 0)
Request 0:
  teacher_prompt_toks: [1, 2, 3, 4, 5]  # "Problem: X\nAnswer: A\nReason:"
  theta_prompt_toks:   [1, 2, 3, 6]     # "Problem: X\nReason:"
  output_toks: []

Request 1:
  teacher_prompt_toks: [10, 11, 12, 13, 14]  # "Problem: Y\nAnswer: B\nReason:"
  theta_prompt_toks:   [10, 11, 12, 15]      # "Problem: Y\nReason:"
  output_toks: []

# Step 1: Generate first token
apply() sees:
  Request 0:
    teacher: [1,2,3,4,5] + [] = [1,2,3,4,5]
    theta:   [1,2,3,6] + []   = [1,2,3,6]
  Request 1:
    teacher: [10,11,12,13,14] + [] = [10,11,12,13,14]
    theta:   [10,11,12,15] + []    = [10,11,12,15]

Sampled: token_0=20, token_1=30
vLLM appends: output_toks[0] = [20], output_toks[1] = [30]

# Step 2: Generate second token
apply() sees:
  Request 0:
    teacher: [1,2,3,4,5] + [20] = [1,2,3,4,5,20]
    theta:   [1,2,3,6] + [20]   = [1,2,3,6,20]
    #                     ^^^^
    #                     共享的生成序列
  Request 1:
    teacher: [10,11,12,13,14] + [30] = [10,11,12,13,14,30]
    theta:   [10,11,12,15] + [30]    = [10,11,12,15,30]
    #                         ^^^^
    #                         共享的生成序列

Sampled: token_0=21, token_1=31
vLLM appends: output_toks[0] = [20,21], output_toks[1] = [30,31]

# Step 3: Generate third token
apply() sees:
  Request 0:
    teacher: [1,2,3,4,5] + [20,21] = [1,2,3,4,5,20,21]
    theta:   [1,2,3,6] + [20,21]   = [1,2,3,6,20,21]
  Request 1:
    teacher: [10,11,12,13,14] + [30,31] = [10,11,12,13,14,30,31]
    theta:   [10,11,12,15] + [30,31]    = [10,11,12,15,30,31]

# ...继续直到 max_tokens 或 EOS
```

**关键观察**：
- Teacher 和 Theta 的**初始 prompt 不同**（实现隐式知识蒸馏）
- 但生成的 **output tokens 完全相同**（来自混合分布 q*）
- Request 0 和 Request 1 的数据**完全独立**（通过 request index 区分）

---

## 📝 关键要点总结

### 1. vLLM Logits Tensor 是固定大小的

```python
# ✅ 正确
logits.shape = [max_num_reqs, vocab_size]  # 固定大小
logits[request_idx] = mixed_logits          # 直接用 request index

# ❌ 错误
logits.shape = [current_batch_size, vocab_size]  # 动态大小
batch_idx = map_request_to_batch[request_idx]   # 不需要映射
```

### 2. Index 复用机制必须正确处理⚠️

**关键规则**：严格按照 `removed → added → moved` 的顺序处理 BatchUpdate

```python
# ✅ 正确
def update_state(self, batch_update):
    # 1. 先清理已完成的请求（释放 index）
    for index in batch_update.removed:
        self.request_states.pop(index, None)
        self.alpha_history.pop(index, None)

    # 2. 再添加新请求（可能复用刚释放的 index）
    for index, params, ... in batch_update.added:
        self.request_states[index] = (...)
        self.alpha_history[index] = []

    # 3. 最后处理移动/交换
    for adx, bdx, direct in batch_update.moved:
        # ...

# ❌ 错误：顺序错误会导致数据混淆
```

**为什么重要**：
- 当并发请求数 > `max_num_reqs` 时，vLLM 会复用 index
- 错误的顺序可能导致新旧请求数据混淆
- Alpha history 文件可能被覆盖

### 3. 批处理中所有请求共享 SamplingParams

```python
# SamplingParams 是共享的
sampling_params = SamplingParams(
    extra_args={
        "original_prompts": ["p0", "p1", "p2"]  # List for all requests
    }
)

# 在 LogitsProcessor 中，必须根据 request index 索引
original_prompt = original_prompts_list[index]  # ✅ 正确
original_prompt = original_prompts_list[0]      # ❌ 错误
```

### 3. Request Index 是关键

Request index 用于：
- 索引 logits tensor: `logits[request_idx]`
- 索引 original_prompts: `original_prompts[request_idx]`
- 索引 alpha_history: `alpha_history[request_idx]`
- 索引 request_states: `request_states[request_idx]`

### 4. Output Tokens 自动同步

vLLM 通过引用传递 `output_tok_ids`，无需手动同步：

```python
# BatchUpdate.added:
for index, params, prompt_tok_ids, output_tok_ids in batch_update.added:
    self.request_states[index] = (..., output_tok_ids, ...)
    #                                  ^^^^^^^^^^^^^^
    #                                  这是一个 list reference

# apply() 中自动获取最新的 output_tok_ids
_, output_tok_ids, _, _ = self.request_states[idx]
# output_tok_ids 已经包含所有已生成的 tokens（vLLM 自动 append）
```

### 5. 两个 vLLM Core 的对齐

| 维度 | Teacher (Outer) | Theta (Inner) | 对齐方式 |
|------|----------------|---------------|---------|
| **Prompt** | 完整 prompt（含答案） | 部分 prompt（无答案） | 分别 tokenize |
| **Output Tokens** | 相同 | 相同 | 引用传递自动同步 |
| **Request Index** | 相同 | 相同 | 直接传递 |
| **Logits** | Teacher logits | Theta logits | 按 request_idx 混合 |

### 6. 调试技巧

```python
# 在 LogitsProcessor 中打印到 stderr（subprocess 可见）
import sys

# 1. 检查 request index 和 original_prompt 的对应关系
def update_state(self, batch_update):
    if batch_update:
        print(f"BatchUpdate: added={len(batch_update.added)}, "
              f"removed={len(batch_update.removed)}, "
              f"moved={len(batch_update.moved)}",
              file=sys.stderr, flush=True)

        for index in batch_update.removed:
            print(f"  Removing request {index}", file=sys.stderr, flush=True)

        for index, params, _, _ in batch_update.added:
            original_prompts = params.extra_args.get("original_prompts", [])
            prompt = original_prompts[index] if index < len(original_prompts) else "N/A"
            print(f"  Adding request {index}: prompt='{prompt[:30]}...'",
                  file=sys.stderr, flush=True)

# 2. 检查 index 复用
def apply(self, logits):
    print(f"apply() called: enabled_requests={sorted(self.enabled_requests)}",
          file=sys.stderr, flush=True)

    for idx in self.enabled_requests:
        if idx in self.request_states:
            _, output_toks, _, original_prompt = self.request_states[idx]
            print(f"  Request {idx}: {len(output_toks)} tokens, "
                  f"prompt='{original_prompt[:20]}...'",
                  file=sys.stderr, flush=True)

# 3. 检查 logits tensor 大小
def apply(self, logits):
    print(f"logits.shape = {logits.shape}, "
          f"max_num_reqs = {logits.shape[0]}, "
          f"enabled = {len(self.enabled_requests)}",
          file=sys.stderr, flush=True)

# 4. 检测 index 复用冲突
def update_state(self, batch_update):
    if batch_update:
        # 检测是否有 index 同时出现在 removed 和 added 中（正常情况）
        removed_set = set(batch_update.removed)
        added_indices = {idx for idx, _, _, _ in batch_update.added}
        reused = removed_set & added_indices
        if reused:
            print(f"  Index reuse detected: {sorted(reused)}",
                  file=sys.stderr, flush=True)

# 5. Alpha history 文件冲突检测
def update_state(self, batch_update):
    if batch_update and self.alpha_storage_dir:
        for index in batch_update.removed:
            alpha_file = Path(self.alpha_storage_dir) / f"alpha_history_{index}.json"
            if alpha_file.exists():
                print(f"  Warning: alpha_history_{index}.json already exists, "
                      f"will be overwritten on next reuse",
                      file=sys.stderr, flush=True)
```

---

## 📚 相关文档

- [SUMMARY.md](SUMMARY.md) - Optimal Sampling V1 总体架构
- [distillation_guide.md](distillation_guide.md) - 知识蒸馏使用指南
- [BENCHMARK_REPORT.md](BENCHMARK_REPORT.md) - 性能基准测试

---

## 📞 问题反馈

如果在批处理中遇到问题，请检查：

1. ✅ 是否使用 `original_prompts[index]` 而不是 `original_prompts[0]`
2. ✅ 是否直接用 `logits[request_idx]` 访问 logits
3. ✅ 是否在 `guide_model_v1.py` 中单独 tokenize theta prompt
4. ✅ 是否正确处理 `output_tok_ids` 的引用传递

如有其他问题，请查看源代码注释或提交 issue。
