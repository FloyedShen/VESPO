# Dual vLLM Optimal Sampling System

**生产就绪的双模型最优采样系统，基于 vLLM 分布式推理引擎**

[![Tests](https://img.shields.io/badge/tests-passing-brightgreen)]()
[![Python](https://img.shields.io/badge/python-3.8+-blue)]()
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)]()

---

## 🎯 系统概述

本系统实现了基于 vLLM 的**最优采样（Optimal Sampling）**框架，用于RLHF数据生成。核心思想：从两个模型（base π_θ 和 teacher π_t）的分布中采样，使得采样分布 q* 同时最小化与两个模型的 KL 散度。

### 核心特性

- ✅ **理论正确**：满足 KL 对称条件 D_KL(q*||π_θ) = D_KL(q*||π_t)
- ✅ **高效实现**：基于 vLLM HTTP API，支持异步并发
- ✅ **双提示支持**：Base 和 Teacher 可使用不同 prompt 格式
- ✅ **稳定性检测**：自动检测分布差异，不稳定时fallback到teacher
- ✅ **Trust Region**：限制采样到teacher的top-p区域
- ✅ **生产就绪**：完整的错误处理、重试、统计跟踪

### 性能指标

- **每token开销**：~1-2ms (KL对称求解 + 稳定性检测)
- **吞吐量**：19-48 tokens/second per sequence
- **内存占用**：协调器 <100MB，模型显存由vLLM管理
- **KL对称误差**：< 1e-6 (理论保证)

---

## 📁 文件结构

```
dual_vllm/
├── README_MAIN.md              # 本文件 - 主文档
├── requirements.txt            # Python依赖
│
├── config.py                   # 基础配置类
├── config_enhanced.py          # 增强配置（新功能）
│
├── utils.py                    # 核心算法（KL对称求解等）
├── utils_stability.py          # 稳定性检测工具（独立）
│
├── coordinator.py              # 基础协调器
├── coordinator_enhanced.py     # 增强协调器（集成稳定性检测）⭐
│
├── generate_data_vllm.py       # 数据生成管线（vLLM版）⭐
│
├── example.py                  # 基础示例
├── example_enhanced.py         # 增强功能示例
├── demo_qwen3.py              # Qwen3完整演示
│
├── test_*.py                   # 各种测试脚本
├── start_vllm.sh              # vLLM启动脚本
│
└── **文档**
    ├── ENHANCED_FEATURES.md       # 增强功能说明
    ├── STABILITY_GUIDE.md         # 稳定性检测指南⭐
    ├── QWEN3_TEST_GUIDE.md        # Qwen3测试指南
    ├── MANUAL_TEST.sh             # 手动测试脚本
    └── README_SUCCESS.md          # 测试成功报告
```

**⭐ 标记为最重要的文件**

---

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 启动 vLLM 服务器

**终端1 - Base模型：**
```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-4B-Base \
    --port 9000 \
    --gpu-memory-utilization 0.20 \
    --max-logprobs 20 \
    --trust-remote-code
```

**终端2 - Teacher模型：**
```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-14B \
    --port 9001 \
    --gpu-memory-utilization 0.55 \
    --max-logprobs 20 \
    --trust-remote-code
```

### 3. 测试系统

```bash
# 简单测试
python test_qwen3_simple.py

# 完整演示
python demo_qwen3.py
```

### 4. 生成数据（重要！）

```bash
python generate_data_vllm.py \
    --theta_url http://localhost:9000 \
    --t_url http://localhost:9001 \
    --dataset agentica-org/DeepScaleR-Preview-Dataset \
    --output generated_data.jsonl \
    --num_samples 100 \
    --max_tokens 512 \
    --enable_stability_check \
    --save_diagnostics
```

---

## 💡 核心配置

### 基础配置

```python
from config_enhanced import EnhancedCoordinatorConfig

config = EnhancedCoordinatorConfig(
    # vLLM服务器
    theta_url="http://localhost:9000",
    t_url="http://localhost:9001",

    # 模型名称
    theta_model_name="Qwen/Qwen3-4B-Base",
    t_model_name="Qwen/Qwen3-14B",

    # Top-k近似
    top_k=20,  # vLLM 0.11.0限制
)
```

### 推荐配置（全功能）

```python
config = EnhancedCoordinatorConfig(
    # 基础配置
    theta_url="http://localhost:9000",
    t_url="http://localhost:9001",
    top_k=20,

    # 🔥 首Token强制（推荐）
    force_first_token=True,

    # 🔥 Trust Region约束（推荐）
    constraint_to_target=True,
    target_top_p=0.95,

    # 🔥 稳定性检测（可选但推荐）
    enable_stability_check=True,
    stability_threshold_js=0.5,
    stability_threshold_overlap=0.1,
    auto_fallback=True,
)
```

---

## 📊 核心功能详解

### 1. 双提示支持（Dual Prompts）

允许Base和Teacher模型看到不同的prompt格式：

```python
# Base模型：简单格式
prompts_theta = ["Q: What is AI?\nA:"]

# Teacher模型：Chat template格式
prompts_t = ["<|im_start|>user\nWhat is AI?<|im_end|>\n<|im_start|>assistant\n"]

# 生成时两个模型采样相同的token，但看到不同上下文
results = await coordinator.generate_batch_dual_prompts(
    prompts_theta=prompts_theta,
    prompts_t=prompts_t,
    max_tokens=100
)
```

**为什么需要？**
- Base模型通常没有经过instruction tuning
- Teacher模型有专门的chat template
- 不同格式能让每个模型发挥最佳性能

### 2. 首Token强制（First Token Forcing）

强制首个token使用teacher模型（α=1.0）：

```python
config = EnhancedCoordinatorConfig(
    force_first_token=True  # ✅ 启用
)
```

**为什么需要？**
- Teacher模型（如Qwen3-14B）有Chain-of-Thought训练
- 首token常常是 `<think>`（开始推理）
- Base模型没有这个行为，强行混合会破坏推理结构

**效果**：
- Token 1: α=1.0 (强制使用teacher)
- Token 2-N: α由KL对称或稳定性检测决定

### 3. Trust Region约束（Support Constraint）

限制采样范围到teacher的top-p区域：

```python
config = EnhancedCoordinatorConfig(
    constraint_to_target=True,  # ✅ 启用trust region
    target_top_p=0.95           # 保留teacher的top-95%
)
```

**为什么需要？**
- 防止采样teacher认为不太可能的token
- 提供更好的数值稳定性
- 更强的对齐性

### 4. 稳定性检测（Stability Detection）⭐ NEW

自动检测两个模型分布的差异，不稳定时fallback：

```python
config = EnhancedCoordinatorConfig(
    enable_stability_check=True,        # ✅ 启用稳定性检测
    stability_threshold_js=0.5,         # JS divergence阈值
    stability_threshold_overlap=0.1,    # Overlap概率质量阈值
    auto_fallback=True                  # 不稳定时自动fallback
)
```

**检测指标**：
- **Overlap Count**：两个模型top-k的交集大小
- **Overlap Mass**：交集的概率质量
- **JS Divergence**：Jensen-Shannon散度 ∈ [0, ln(2)]

**Fallback条件**：
- JS Divergence > 0.5（分布差异太大）
- 或 Overlap Mass < 0.1（几乎没有重叠）

**效果**（Qwen3-4B + Qwen3-14B实测）：
- 50% tokens稳定 → 正常混合（α ≈ 0.2-0.3）
- 50% tokens不稳定 → fallback到teacher（α = 1.0）
- **避免了不合理的混合**

详细说明见：`STABILITY_GUIDE.md`

---

## 🎛️ 配置建议

### 保守配置（强依赖Teacher）

```python
EnhancedCoordinatorConfig(
    force_first_token=True,
    constraint_to_target=True,
    target_top_p=0.90,                 # 更严格的约束
    enable_stability_check=True,
    stability_threshold_js=0.4,         # 更低的阈值
    stability_threshold_overlap=0.15,   # 更高的要求
)
```

**适用**：安全关键应用、teacher明显优于base

### 平衡配置（推荐）⭐

```python
EnhancedCoordinatorConfig(
    force_first_token=True,
    constraint_to_target=True,
    target_top_p=0.95,
    enable_stability_check=True,
    stability_threshold_js=0.5,
    stability_threshold_overlap=0.10,
)
```

**适用**：一般用途、quality与diversity平衡

### 探索配置（更多样性）

```python
EnhancedCoordinatorConfig(
    force_first_token=False,           # 不强制首token
    constraint_to_target=False,         # 不约束
    enable_stability_check=False,       # 不检测
)
```

**适用**：创意任务、需要更多探索

---

## 📈 使用示例

### 示例1：基础使用

```python
import asyncio
from coordinator_enhanced import EnhancedDualVLLMCoordinator
from config_enhanced import EnhancedCoordinatorConfig

async def main():
    config = EnhancedCoordinatorConfig(
        theta_url="http://localhost:9000",
        t_url="http://localhost:9001",
    )

    prompts_theta = ["Q: What is machine learning?\nA:"]
    prompts_t = ["<|im_start|>user\nWhat is machine learning?<|im_end|>\n<|im_start|>assistant\n"]

    async with EnhancedDualVLLMCoordinator(config) as coordinator:
        results = await coordinator.generate_batch_dual_prompts(
            prompts_theta=prompts_theta,
            prompts_t=prompts_t,
            max_tokens=100,
            return_diagnostics=True
        )

        for result in results:
            print(f"Generated: {result.generated_text}")
            print(f"Alpha mean: {np.mean(result.alpha_history):.3f}")

asyncio.run(main())
```

### 示例2：批量生成

```python
# 准备100个prompts
prompts_theta = [f"Q: Question {i}?\nA:" for i in range(100)]
prompts_t = [f"<|im_start|>user\nQuestion {i}?<|im_end|>\n<|im_start|>assistant\n" for i in range(100)]

# 批量生成（自动并发）
results = await coordinator.generate_batch_dual_prompts(
    prompts_theta=prompts_theta,
    prompts_t=prompts_t,
    max_tokens=512,
    show_progress=True  # 显示进度条
)
```

### 示例3：数据生成管线

参考：`generate_data_vllm.py`

```bash
python generate_data_vllm.py \
    --theta_url http://localhost:9000 \
    --t_url http://localhost:9001 \
    --dataset agentica-org/DeepScaleR-Preview-Dataset \
    --output data.jsonl \
    --num_samples 1000 \
    --max_tokens 512 \
    --batch_size 16 \
    --enable_stability_check \
    --save_diagnostics
```

---

## 🧪 测试

### 单元测试

```bash
# 基础功能测试（无需vLLM）
python test_enhanced.py

# 稳定性检测测试（无需vLLM）
python test_stability_enhanced.py
```

### 集成测试

```bash
# 需要vLLM服务器运行
python test_qwen3_simple.py       # 简单测试
python test_qwen3_stability.py    # 稳定性测试
python test_sequential_stability.py  # 连续生成测试
```

---

## 🔬 理论基础

### 最优采样公式

给定两个分布 π_θ 和 π_t，最优采样分布 q* 满足：

```
q*(y) ∝ π_θ(y)^(1-α*) · π_t(y)^α*
```

其中 α* 通过二分搜索求解 KL对称条件：

```
D_KL(q*||π_θ) = D_KL(q*||π_t)
```

详细理论见：`../../theory/proof_final.md`

### 稳定性检测原理

使用 **Jensen-Shannon Divergence** 量化分布差异：

```
JS(P||Q) = 0.5 * D_KL(P||M) + 0.5 * D_KL(Q||M)
其中 M = 0.5 * (P + Q)
```

特性：
- 对称：JS(P||Q) = JS(Q||P)
- 有界：JS ∈ [0, ln(2)] ≈ [0, 0.693]
- JS = 0: 完全相同
- JS = ln(2): 完全不同

---

## 📚 文档索引

| 文档 | 说明 |
|------|------|
| `README_MAIN.md` (本文件) | 主文档 - 快速开始 |
| `STABILITY_GUIDE.md` | 稳定性检测详细指南⭐ |
| `ENHANCED_FEATURES.md` | 增强功能说明 |
| `QWEN3_TEST_GUIDE.md` | Qwen3模型测试指南 |
| `README_SUCCESS.md` | 测试成功报告和结果 |

---

## ⚙️ API参考

### EnhancedCoordinatorConfig

主要参数：

- `theta_url`, `t_url`: vLLM服务器URL
- `theta_model_name`, `t_model_name`: 模型名称
- `top_k`: Top-k近似大小（≤20）
- `force_first_token`: 是否强制首token
- `constraint_to_target`: 是否启用trust region
- `target_top_p`: Trust region的top-p阈值
- `enable_stability_check`: 是否启用稳定性检测
- `stability_threshold_js`: JS divergence阈值
- `stability_threshold_overlap`: Overlap质量阈值
- `auto_fallback`: 是否自动fallback

### EnhancedDualVLLMCoordinator

主要方法：

```python
async def generate_batch_dual_prompts(
    prompts_theta: List[str],
    prompts_t: List[str],
    max_tokens: int = 100,
    temperature: float = 1.0,
    return_diagnostics: bool = False,
    show_progress: bool = True
) -> List[GenerationOutput]
```

---

## 🤝 贡献指南

欢迎贡献！请确保：

1. 所有测试通过：`python test_enhanced.py`
2. 代码符合风格规范
3. 添加必要的文档字符串

---

## 📄 许可证

Apache 2.0 License

---

## 🆘 常见问题

### Q1: 为什么所有token都fallback到teacher？

**A**: 可能原因：
1. Base和Teacher模型差异太大（如4B vs 14B且训练数据不同）
2. 使用了不同的chat template（base用简单格式，teacher用ChatML）
3. Teacher有CoT训练，base没有

**解决方案**：
- 这是正常的！稳定性检测正确识别了不兼容
- 如果想要更多混合，降低`stability_threshold_js`阈值
- 或者使用更相似的模型对

### Q2: 性能如何优化？

**A**:
1. 增加batch_size（利用vLLM的批处理）
2. 禁用不必要的功能（如`enable_stability_check=False`）
3. 使用更快的模型（如7B替代14B）
4. 调整vLLM的`gpu_memory_utilization`

### Q3: 如何自定义chat template？

**A**: 在`generate_data_vllm.py`中修改prompt构造逻辑。

---

**🎉 系统已生产就绪！开始使用吧！**

如有问题，请查看 `STABILITY_GUIDE.md` 或其他文档。
