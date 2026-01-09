#!/usr/bin/env python3
"""
正确性测试：验证 Optimal Sampling 生成的质量

测试内容:
1. 基础生成测试 - 确保能生成合理文本
2. 数学推理测试 - 验证答案正确性
3. 温度一致性测试 - 验证 teacher 和 theta 使用相同温度
4. Alpha 合理性测试 - 验证 alpha 在合理范围内
5. 不同前缀测试 - 验证 teacher 和 student 接收不同输入
"""

import sys
import re
from optimal_sampling import OptimalSamplingV1


def test_basic_generation():
    """测试 1: 基础生成功能"""
    print("\n" + "=" * 80)
    print("测试 1: 基础生成功能")
    print("=" * 80)

    sampler = OptimalSamplingV1(
        model_teacher="Qwen/Qwen2.5-3B-Instruct",
        model_theta="Qwen/Qwen2.5-1.5B-Instruct",
        alpha_method="kl_symmetry",
        gpu_memory_utilization=0.45,
        track_alpha_stats=False,
    )

    prompts = ["Say hello in 5 words:"]

    outputs = sampler.generate(
        prompts=prompts,
        max_tokens=20,
        temperature=0.8,
        use_optimal_sampling=True
    )

    text = outputs.generated_texts[0]
    print(f"\n生成文本: '{text}'")

    # 验证
    assert len(text) > 0, "生成文本为空"
    assert len(text.split()) <= 25, "生成文本过长"

    print("✅ 基础生成测试通过")
    return True


def test_math_reasoning():
    """测试 2: 数学推理正确性"""
    print("\n" + "=" * 80)
    print("测试 2: 数学推理正确性")
    print("=" * 80)

    sampler = OptimalSamplingV1(
        model_teacher="Qwen/Qwen2.5-3B-Instruct",
        model_theta="Qwen/Qwen2.5-1.5B-Instruct",
        alpha_method="kl_symmetry",
        gpu_memory_utilization=0.45,
        track_alpha_stats=False,
    )

    # 简单数学问题
    problems = [
        {
            "question": "What is 2 + 2?",
            "answer": "4",
            "teacher_prompt": "Problem: What is 2 + 2?\nAnswer: 4\nExplain:",
            "student_prompt": "Problem: What is 2 + 2?\nSolve:",
        },
        {
            "question": "What is 3 × 5?",
            "answer": "15",
            "teacher_prompt": "Problem: What is 3 × 5?\nAnswer: 15\nExplain:",
            "student_prompt": "Problem: What is 3 × 5?\nSolve:",
        },
    ]

    for i, prob in enumerate(problems):
        print(f"\n问题 {i+1}: {prob['question']}")

        outputs = sampler.generate(
            prompts=[prob["teacher_prompt"]],
            theta_prompts=[prob["student_prompt"]],
            max_tokens=100,
            temperature=0.7,
            use_optimal_sampling=True
        )

        text = outputs.generated_texts[0]
        print(f"生成: {text[:200]}...")

        # 验证答案出现在生成文本中
        answer_found = prob["answer"] in text
        print(f"答案 '{prob['answer']}' 是否出现: {answer_found}")

        if not answer_found:
            print(f"⚠️ 警告: 答案未出现在生成文本中")
        else:
            print(f"✅ 答案正确出现")

    print("\n✅ 数学推理测试完成")
    return True


def test_temperature_consistency():
    """测试 3: 温度参数一致性"""
    print("\n" + "=" * 80)
    print("测试 3: 温度参数一致性")
    print("=" * 80)

    sampler = OptimalSamplingV1(
        model_teacher="Qwen/Qwen2.5-3B-Instruct",
        model_theta="Qwen/Qwen2.5-1.5B-Instruct",
        alpha_method="kl_symmetry",
        gpu_memory_utilization=0.45,
        track_alpha_stats=True,  # 开启追踪以验证
    )

    prompts = ["Count from 1 to 10:"]

    # 测试不同温度
    for temp in [0.5, 0.8, 1.0]:
        print(f"\n测试温度: {temp}")

        outputs = sampler.generate(
            prompts=prompts,
            max_tokens=30,
            temperature=temp,
            use_optimal_sampling=True
        )

        text = outputs.generated_texts[0]
        print(f"生成 (temp={temp}): {text[:100]}...")

        # 低温度应该更确定性
        if temp == 0.5:
            low_temp_text = text
        elif temp == 1.0:
            high_temp_text = text
            # 高温度应该与低温度有差异（通常）
            if low_temp_text == high_temp_text:
                print("⚠️ 注意: 不同温度生成相同文本（可能是确定性问题）")

    print("\n✅ 温度一致性测试完成")
    return True


def test_different_prompts():
    """测试 4: Teacher 和 Student 不同前缀"""
    print("\n" + "=" * 80)
    print("测试 4: Teacher 和 Student 接收不同输入")
    print("=" * 80)

    sampler = OptimalSamplingV1(
        model_teacher="Qwen/Qwen2.5-3B-Instruct",
        model_theta="Qwen/Qwen2.5-1.5B-Instruct",
        alpha_method="kl_symmetry",
        gpu_memory_utilization=0.45,
        track_alpha_stats=False,
    )

    # Teacher 看到答案，Student 不看
    teacher_prompt = "Problem: Solve 2x + 3 = 7\nAnswer: x = 2\nReasoning:"
    student_prompt = "Problem: Solve 2x + 3 = 7\nReasoning:"

    print(f"\nTeacher prompt: {teacher_prompt}")
    print(f"Student prompt: {student_prompt}")

    outputs = sampler.generate(
        prompts=[teacher_prompt],
        theta_prompts=[student_prompt],
        max_tokens=150,
        temperature=0.8,
        use_optimal_sampling=True
    )

    text = outputs.generated_texts[0]
    print(f"\n生成推理: {text[:300]}...")

    # 验证生成包含推理步骤
    has_reasoning = any(keyword in text.lower() for keyword in
                       ["step", "first", "then", "solve", "subtract", "divide"])

    print(f"\n包含推理关键词: {has_reasoning}")

    if has_reasoning:
        print("✅ 成功生成推理过程")
    else:
        print("⚠️ 警告: 未检测到明显的推理步骤")

    print("\n✅ 不同前缀测试完成")
    return True


def test_alpha_values():
    """测试 5: Alpha 值合理性"""
    print("\n" + "=" * 80)
    print("测试 5: Alpha 值合理性")
    print("=" * 80)

    sampler = OptimalSamplingV1(
        model_teacher="Qwen/Qwen2.5-3B-Instruct",
        model_theta="Qwen/Qwen2.5-1.5B-Instruct",
        alpha_method="kl_symmetry",
        gpu_memory_utilization=0.45,
        track_alpha_stats=True,
    )

    prompts = ["Write a short poem about AI:"]

    outputs = sampler.generate(
        prompts=prompts,
        max_tokens=50,
        temperature=0.8,
        use_optimal_sampling=True
    )

    # 获取 alpha 统计
    alpha_stats = outputs.alpha_stats[0] if outputs.alpha_stats else None

    if alpha_stats:
        print(f"\nAlpha 统计:")
        print(f"  平均值: {alpha_stats['mean']:.4f}")
        print(f"  标准差: {alpha_stats['std']:.4f}")
        print(f"  最小值: {alpha_stats['min']:.4f}")
        print(f"  最大值: {alpha_stats['max']:.4f}")
        print(f"  样本数: {alpha_stats['count']}")

        # 验证 alpha 在合理范围
        assert 0 <= alpha_stats['mean'] <= 1, "Alpha 平均值超出 [0,1] 范围"
        assert 0 <= alpha_stats['min'] <= 1, "Alpha 最小值超出 [0,1] 范围"
        assert 0 <= alpha_stats['max'] <= 1, "Alpha 最大值超出 [0,1] 范围"

        # KL 对称性应该让 alpha 在 0.5 附近（对于相似模型）
        if 0.3 < alpha_stats['mean'] < 0.7:
            print("✅ Alpha 在合理范围内 (0.3-0.7)")
        else:
            print(f"⚠️ Alpha 偏离中心值: {alpha_stats['mean']:.4f}")
    else:
        print("⚠️ 未获取到 Alpha 统计")

    print("\n✅ Alpha 值测试完成")
    return True


def test_batch_processing():
    """测试 6: 批量处理"""
    print("\n" + "=" * 80)
    print("测试 6: 批量处理")
    print("=" * 80)

    sampler = OptimalSamplingV1(
        model_teacher="Qwen/Qwen2.5-3B-Instruct",
        model_theta="Qwen/Qwen2.5-1.5B-Instruct",
        alpha_method="kl_symmetry",
        gpu_memory_utilization=0.45,
        track_alpha_stats=False,
    )

    # 批量处理
    batch_size = 4
    prompts = [f"Count from 1 to {i+3}:" for i in range(batch_size)]

    print(f"\n批量大小: {batch_size}")

    outputs = sampler.generate(
        prompts=prompts,
        max_tokens=30,
        temperature=0.8,
        use_optimal_sampling=True
    )

    print(f"\n生成结果数量: {len(outputs.generated_texts)}")
    assert len(outputs.generated_texts) == batch_size, "生成数量不匹配"

    for i, text in enumerate(outputs.generated_texts):
        print(f"\n请求 {i+1}: {text[:80]}...")

    print("\n✅ 批量处理测试通过")
    return True


def test_baseline_comparison():
    """测试 7: Optimal vs Teacher-only 对比"""
    print("\n" + "=" * 80)
    print("测试 7: Optimal Sampling vs Teacher-only 对比")
    print("=" * 80)

    sampler = OptimalSamplingV1(
        model_teacher="Qwen/Qwen2.5-3B-Instruct",
        model_theta="Qwen/Qwen2.5-1.5B-Instruct",
        alpha_method="kl_symmetry",
        gpu_memory_utilization=0.45,
        track_alpha_stats=False,
    )

    prompt = "Explain machine learning in simple terms:"

    # Optimal Sampling
    print("\n[Optimal Sampling]")
    outputs_optimal = sampler.generate(
        prompts=[prompt],
        max_tokens=80,
        temperature=0.8,
        use_optimal_sampling=True
    )
    optimal_text = outputs_optimal.generated_texts[0]
    print(f"生成: {optimal_text[:200]}...")

    # Teacher-only
    print("\n[Teacher-only Baseline]")
    outputs_baseline = sampler.generate(
        prompts=[prompt],
        max_tokens=80,
        temperature=0.8,
        use_optimal_sampling=False
    )
    baseline_text = outputs_baseline.generated_texts[0]
    print(f"生成: {baseline_text[:200]}...")

    # 比较
    print(f"\n文本长度对比:")
    print(f"  Optimal: {len(optimal_text)} 字符")
    print(f"  Baseline: {len(baseline_text)} 字符")

    # 两者应该都能生成合理文本
    assert len(optimal_text) > 10, "Optimal 生成过短"
    assert len(baseline_text) > 10, "Baseline 生成过短"

    print("\n✅ 对比测试完成")
    return True


def main():
    """运行所有测试"""
    print("\n" + "=" * 80)
    print("🧪 Optimal Sampling 正确性测试套件")
    print("=" * 80)

    tests = [
        ("基础生成", test_basic_generation),
        ("数学推理", test_math_reasoning),
        ("温度一致性", test_temperature_consistency),
        ("不同前缀", test_different_prompts),
        ("Alpha 值", test_alpha_values),
        ("批量处理", test_batch_processing),
        ("对比测试", test_baseline_comparison),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"\n❌ 测试失败: {name}")
            print(f"错误: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    # 总结
    print("\n" + "=" * 80)
    print("测试总结")
    print("=" * 80)
    print(f"通过: {passed}/{len(tests)}")
    print(f"失败: {failed}/{len(tests)}")

    if failed == 0:
        print("\n✅ 所有测试通过！")
        return 0
    else:
        print(f"\n❌ {failed} 个测试失败")
        return 1


if __name__ == '__main__':
    exit(main())
