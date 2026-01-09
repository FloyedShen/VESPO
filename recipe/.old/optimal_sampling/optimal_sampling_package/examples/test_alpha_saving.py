#!/usr/bin/env python3
"""
测试 Alpha 值保存功能

验证:
1. Alpha history 正确保存在 OptimalSamplingOutput 中
2. save_alpha_history() 方法能正确保存到文件
3. 可以从文件读取并使用 alpha 值
"""

import json
from optimal_sampling import OptimalSamplingV1


def test_alpha_saving():
    """测试 alpha 保存功能"""
    print("=" * 80)
    print("🧪 测试 Alpha 值保存功能")
    print("=" * 80)

    # 初始化 sampler
    print("\n[1] 初始化 Optimal Sampling...")
    sampler = OptimalSamplingV1(
        model_teacher="Qwen/Qwen2.5-3B-Instruct",
        model_theta="Qwen/Qwen2.5-1.5B-Instruct",
        alpha_method="kl_symmetry",
        gpu_memory_utilization=0.45,
        track_alpha_stats=True,  # 必须开启才能追踪 alpha
    )

    # 测试单个请求
    print("\n[2] 测试单个请求...")
    prompts = ["What is 2 + 2?"]

    outputs = sampler.generate(
        prompts=prompts,
        max_tokens=50,
        temperature=0.8,
        use_optimal_sampling=True
    )

    print(f"\n生成文本: {outputs.generated_texts[0][:100]}...")
    print(f"Token 数量: {outputs.num_tokens[0]}")

    # 检查 alpha_history
    if outputs.alpha_history and outputs.alpha_history[0]:
        alpha_hist = outputs.alpha_history[0]
        print(f"\n✅ Alpha history 已保存!")
        print(f"   Alpha 值数量: {len(alpha_hist)}")
        print(f"   前 5 个 alpha 值: {alpha_hist[:5]}")

        # 检查 alpha_stats
        if outputs.alpha_stats and outputs.alpha_stats[0]:
            stats = outputs.alpha_stats[0]
            print(f"\n📊 Alpha 统计:")
            print(f"   平均值: {stats['mean']:.4f}")
            print(f"   标准差: {stats['std']:.4f}")
            print(f"   最小值: {stats['min']:.4f}")
            print(f"   最大值: {stats['max']:.4f}")
            print(f"   样本数: {stats['count']}")
    else:
        print("❌ Alpha history 未保存")
        return False

    # 测试保存到文件
    print("\n[3] 测试保存到文件...")
    filepath = "test_alpha_history.json"
    sampler.save_alpha_history(outputs, filepath)

    # 验证文件内容
    print(f"\n[4] 验证文件内容...")
    with open(filepath, 'r') as f:
        data = json.load(f)

    print(f"✅ 文件已创建: {filepath}")
    print(f"   请求数量: {data['num_requests']}")
    print(f"   第一个请求的 alpha 数量: {len(data['requests'][0]['alpha_history'])}")

    if 'statistics' in data['requests'][0]:
        print(f"   包含统计信息: ✅")
        print(f"   统计信息: {data['requests'][0]['statistics']}")

    # 测试批量请求
    print("\n[5] 测试批量请求...")
    batch_prompts = [
        "Count from 1 to 5:",
        "What is the capital of France?",
        "Calculate 10 * 10:",
    ]

    batch_outputs = sampler.generate(
        prompts=batch_prompts,
        max_tokens=30,
        temperature=0.8,
        use_optimal_sampling=True
    )

    print(f"\n批量生成完成:")
    for i, text in enumerate(batch_outputs.generated_texts):
        print(f"  [{i+1}] {text[:60]}...")
        if batch_outputs.alpha_history and batch_outputs.alpha_history[i]:
            print(f"       Alpha 数量: {len(batch_outputs.alpha_history[i])}")

    # 保存批量结果
    batch_filepath = "test_alpha_history_batch.json"
    sampler.save_alpha_history(batch_outputs, batch_filepath)
    print(f"\n✅ 批量结果已保存: {batch_filepath}")

    # 读取并显示批量文件
    with open(batch_filepath, 'r') as f:
        batch_data = json.load(f)

    print(f"\n批量文件内容:")
    print(f"  总请求数: {batch_data['num_requests']}")
    for req in batch_data['requests']:
        idx = req['request_index']
        alpha_count = len(req['alpha_history']) if req['alpha_history'] else 0
        print(f"  请求 {idx}: {alpha_count} 个 alpha 值")
        if 'statistics' in req and req['statistics']:
            print(f"    统计: mean={req['statistics']['mean']:.4f}, "
                  f"std={req['statistics']['std']:.4f}")

    print("\n" + "=" * 80)
    print("✅ 所有测试通过!")
    print("=" * 80)

    # 清理
    import os
    os.remove(filepath)
    os.remove(batch_filepath)
    print("\n✅ 测试文件已清理")

    return True


if __name__ == '__main__':
    try:
        success = test_alpha_saving()
        exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
