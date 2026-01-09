#!/bin/bash
# 手动启动 Qwen3-4B-Base + Qwen3-14B 测试
#
# 使用说明：
# 1. 在三个不同的终端中分别运行以下命令
# 2. 等待两个模型都启动完成（约 1-2 分钟）
# 3. 运行测试脚本

echo "==================================="
echo "Qwen3-4B-Base + Qwen3-14B 手动测试"
echo "==================================="
echo ""
echo "步骤 1: 启动 Base 模型 (4B)"
echo "-----------------------------------"
echo "在终端 1 中运行："
echo ""
echo "python -m vllm.entrypoints.api_server \\"
echo "    --model Qwen/Qwen3-4B-Base \\"
echo "    --port 9000 \\"
echo "    --gpu-memory-utilization 0.20 \\"
echo "    --max-model-len 2048 \\"
echo "    --dtype auto \\"
echo "    --trust-remote-code"
echo ""
echo "步骤 2: 启动 Teacher 模型 (14B)"
echo "-----------------------------------"
echo "在终端 2 中运行："
echo ""
echo "python -m vllm.entrypoints.api_server \\"
echo "    --model Qwen/Qwen3-14B \\"
echo "    --port 9001 \\"
echo "    --gpu-memory-utilization 0.55 \\"
echo "    --max-model-len 2048 \\"
echo "    --dtype auto \\"
echo "    --trust-remote-code"
echo ""
echo "步骤 3: 等待模型加载完成"
echo "-----------------------------------"
echo "看到类似以下信息表示准备就绪："
echo "  INFO:     Application startup complete."
echo "  INFO:     Uvicorn running on http://0.0.0.0:9000"
echo ""
echo "可以通过以下命令测试："
echo "  curl http://localhost:9000/health"
echo "  curl http://localhost:9001/health"
echo ""
echo "步骤 4: 运行测试"
echo "-----------------------------------"
echo "在终端 3 中运行以下 Python 脚本..."
echo ""

# 创建简单的测试脚本
cat > test_qwen3_simple.py << 'EOF'
#!/usr/bin/env python3
"""
简单测试 Qwen3-4B-Base + Qwen3-14B
假设两个 vLLM 实例已经在运行
"""

import asyncio
import numpy as np
from coordinator_enhanced import EnhancedDualVLLMCoordinator
from config_enhanced import EnhancedCoordinatorConfig


async def test():
    """简单测试"""
    print("\n" + "="*70)
    print("🧪 Qwen3-4B-Base + Qwen3-14B 简单测试")
    print("="*70)

    # 配置
    config = EnhancedCoordinatorConfig(
        theta_url="http://localhost:9000",
        t_url="http://localhost:9001",
        top_k=100,
        force_first_token=True,
        constraint_to_target=True,
        target_top_p=0.95,
        enable_logging=False,
    )

    # 测试提示
    prompts_theta = [
        "Q: What is machine learning?\nA:",
        "Q: Explain neural networks.\nA:",
    ]

    prompts_t = [
        "<|im_start|>user\nWhat is machine learning?<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nExplain neural networks.<|im_end|>\n<|im_start|>assistant\n",
    ]

    print(f"\n📝 测试 {len(prompts_theta)} 个提示...")
    print(f"   Base 格式: {prompts_theta[0][:40]}...")
    print(f"   Instruct 格式: {prompts_t[0][:40]}...")

    try:
        async with EnhancedDualVLLMCoordinator(config) as coordinator:
            results = await coordinator.generate_batch_dual_prompts(
                prompts_theta=prompts_theta,
                prompts_t=prompts_t,
                max_tokens=50,
                temperature=1.0,
                return_diagnostics=True,
                show_progress=True
            )

            # 分析结果
            print(f"\n{'='*70}")
            print("📊 结果")
            print("="*70)

            for i, result in enumerate(results):
                print(f"\n[{i+1}] {prompts_theta[i][:30]}...")

                if result.error:
                    print(f"  ❌ 错误: {result.error}")
                else:
                    alpha_mean = np.mean(result.alpha_history)
                    alpha_std = np.std(result.alpha_history)

                    print(f"  ✅ Tokens: {len(result.generated_tokens)}")
                    print(f"  📊 α: {alpha_mean:.3f} ± {alpha_std:.3f}")
                    print(f"     首 α: {result.alpha_history[0]:.3f}")

                    if result.diagnostics:
                        print(f"  📈 KL 对称误差: {result.diagnostics['kl_diff_mean']:.6f}")
                        print(f"     ESS 比例: {result.diagnostics['ess_ratio_mean']:.3f}")

            # 统计
            stats = coordinator.get_statistics()
            print(f"\n{'='*70}")
            print("📈 统计")
            print("="*70)
            print(f"  请求数: {stats['total_requests']}")
            print(f"  Token 数: {stats['total_tokens']}")
            print(f"  首 token 强制次数: {stats['first_token_forced']}")
            print(f"  约束应用次数: {stats['constraint_applied']}")

            print("\n" + "="*70)
            print("🎉 测试完成！")
            print("="*70)

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test())
EOF

chmod +x test_qwen3_simple.py

echo "创建了 test_qwen3_simple.py"
echo ""
echo "运行："
echo "  python test_qwen3_simple.py"
echo ""
echo "==================================="
echo "完成！"
echo "==================================="
