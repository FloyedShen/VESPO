#!/usr/bin/env python3
"""
Alpha 值保存功能使用示例

展示如何使用 Optimal Sampling 的 alpha 保存功能。

注意: 这个示例展示 API 用法。由于当前 vLLM V1 架构限制，
嵌套初始化 theta model 可能会遇到问题。在实际使用中，
OptimalSamplingOutput 会包含 alpha_history 和 alpha_stats 字段。
"""

from optimal_sampling import OptimalSamplingV1, OptimalSamplingOutput
import json


def example_basic_usage():
    """示例 1: 基本用法"""
    print("=" * 80)
    print("示例 1: 基本 Alpha 值访问")
    print("=" * 80)

    # 模拟一个已经生成的输出（实际使用中由 generate() 返回）
    # 实际代码:
    # sampler = OptimalSamplingV1(...)
    # outputs = sampler.generate(prompts=["What is AI?"], max_tokens=50)

    # 模拟输出数据
    mock_output = OptimalSamplingOutput(
        generated_texts=["AI is artificial intelligence..."],
        generated_ids=[[23, 45, 67, ...]],
        num_tokens=[50],
        alpha_history=[[0.523, 0.518, 0.521, 0.519, 0.522] * 10],  # 50 个 alpha 值
        alpha_stats=[{
            "mean": 0.5206,
            "std": 0.0018,
            "min": 0.518,
            "max": 0.523,
            "count": 50
        }]
    )

    # 访问 alpha history
    print("\n📊 Alpha History:")
    alpha_values = mock_output.alpha_history[0]
    print(f"  - 生成的 token 数: {mock_output.num_tokens[0]}")
    print(f"  - Alpha 值数量: {len(alpha_values)}")
    print(f"  - 前 10 个 alpha 值: {alpha_values[:10]}")

    # 访问统计信息
    print("\n📈 Alpha 统计:")
    stats = mock_output.alpha_stats[0]
    print(f"  - 平均值: {stats['mean']:.4f}")
    print(f"  - 标准差: {stats['std']:.4f}")
    print(f"  - 范围: [{stats['min']:.4f}, {stats['max']:.4f}]")
    print(f"  - 样本数: {stats['count']}")


def example_batch_processing():
    """示例 2: 批量处理"""
    print("\n" + "=" * 80)
    print("示例 2: 批量请求的 Alpha 值")
    print("=" * 80)

    # 模拟批量输出
    mock_batch_output = OptimalSamplingOutput(
        generated_texts=[
            "Machine learning is...",
            "Deep learning uses...",
            "Reinforcement learning involves..."
        ],
        generated_ids=[[1, 2, 3]] * 3,
        num_tokens=[30, 45, 38],
        alpha_history=[
            [0.52] * 30,
            [0.51] * 45,
            [0.53] * 38,
        ],
        alpha_stats=[
            {"mean": 0.520, "std": 0.002, "min": 0.518, "max": 0.522, "count": 30},
            {"mean": 0.510, "std": 0.003, "min": 0.507, "max": 0.513, "count": 45},
            {"mean": 0.530, "std": 0.001, "min": 0.529, "max": 0.531, "count": 38},
        ]
    )

    print("\n批量生成结果:")
    for i in range(len(mock_batch_output.generated_texts)):
        print(f"\n请求 {i+1}:")
        print(f"  Text: {mock_batch_output.generated_texts[i][:40]}...")
        print(f"  Tokens: {mock_batch_output.num_tokens[i]}")
        print(f"  Alpha count: {len(mock_batch_output.alpha_history[i])}")
        print(f"  Avg Alpha: {mock_batch_output.alpha_stats[i]['mean']:.4f}")


def example_save_to_file():
    """示例 3: 保存到文件"""
    print("\n" + "=" * 80)
    print("示例 3: 保存 Alpha 值到文件")
    print("=" * 80)

    # 手动创建 JSON 数据（演示格式）
    alpha_data = {
        "num_requests": 2,
        "requests": [
            {
                "request_index": 0,
                "alpha_history": [0.52, 0.51, 0.53, 0.52, 0.51],
                "num_tokens": 5,
                "statistics": {
                    "mean": 0.518,
                    "std": 0.007,
                    "min": 0.51,
                    "max": 0.53,
                    "count": 5
                }
            },
            {
                "request_index": 1,
                "alpha_history": [0.54, 0.53, 0.52, 0.53],
                "num_tokens": 4,
                "statistics": {
                    "mean": 0.530,
                    "std": 0.007,
                    "min": 0.52,
                    "max": 0.54,
                    "count": 4
                }
            }
        ]
    }

    # 保存到文件
    filepath = "example_alpha_history.json"
    with open(filepath, 'w') as f:
        json.dump(alpha_data, f, indent=2)

    print(f"\n✅ Alpha 历史已保存到: {filepath}")

    # 读取并验证
    with open(filepath, 'r') as f:
        loaded_data = json.load(f)

    print(f"\n📖 文件内容验证:")
    print(f"  - 请求数量: {loaded_data['num_requests']}")
    for req in loaded_data['requests']:
        print(f"\n  请求 {req['request_index']}:")
        print(f"    - Alpha 值数量: {len(req['alpha_history'])}")
        print(f"    - 平均值: {req['statistics']['mean']:.4f}")

    # 清理
    import os
    os.remove(filepath)
    print(f"\n✅ 清理示例文件: {filepath}")


def example_alpha_analysis():
    """示例 4: Alpha 值分析"""
    print("\n" + "=" * 80)
    print("示例 4: Alpha 值分析")
    print("=" * 80)

    # 模拟不同场景的 alpha 值
    scenarios = {
        "稳定生成": [0.52] * 50,
        "逐渐增加": [0.50 + i * 0.001 for i in range(50)],
        "波动较大": [0.52 if i % 2 == 0 else 0.48 for i in range(50)],
    }

    import numpy as np

    for name, alpha_values in scenarios.items():
        alpha_array = np.array(alpha_values)
        print(f"\n📊 场景: {name}")
        print(f"  - 平均值: {np.mean(alpha_array):.4f}")
        print(f"  - 标准差: {np.std(alpha_array):.4f}")
        print(f"  - 范围: [{np.min(alpha_array):.4f}, {np.max(alpha_array):.4f}]")

        # 判断稳定性
        if np.std(alpha_array) < 0.01:
            print(f"  - 稳定性: ✅ 稳定")
        elif np.std(alpha_array) < 0.02:
            print(f"  - 稳定性: ⚠️  中等")
        else:
            print(f"  - 稳定性: ❌ 波动较大")


def main():
    """运行所有示例"""
    print("\n" + "=" * 80)
    print("🎨 Alpha 值保存功能使用示例")
    print("=" * 80)

    example_basic_usage()
    example_batch_processing()
    example_save_to_file()
    example_alpha_analysis()

    print("\n" + "=" * 80)
    print("✅ 所有示例完成!")
    print("=" * 80)

    print("\n💡 实际使用方法:")
    print("""
    from optimal_sampling import OptimalSamplingV1

    # 1. 初始化
    sampler = OptimalSamplingV1(
        model_teacher="Qwen/Qwen2.5-3B-Instruct",
        model_theta="Qwen/Qwen2.5-1.5B-Instruct",
        alpha_method="kl_symmetry",
        track_alpha_stats=True,  # 开启 alpha 追踪
    )

    # 2. 生成
    outputs = sampler.generate(
        prompts=["What is AI?"],
        max_tokens=100,
        temperature=0.8,
    )

    # 3. 访问 alpha 值
    if outputs.alpha_history:
        print(f"Alpha 值: {outputs.alpha_history[0][:10]}...")
        print(f"统计: {outputs.alpha_stats[0]}")

    # 4. 保存到文件
    sampler.save_alpha_history(outputs, "alpha.json")
    """)


if __name__ == '__main__':
    main()
