#!/usr/bin/env python3
"""
大规模性能基准测试：序列长度 vs 批量大小

测试矩阵:
- 序列长度 (max_tokens): [1k, 2k, 4k, 8k]
- 批量大小 (batch_size): [1, 4, 8, 16, 32, 64]
- 总配置: 4 × 6 = 24 个

目标:
1. 找出最佳吞吐量配置
2. 分析序列长度对性能的影响
3. 分析批量大小对性能的影响
4. 生成性能热力图数据
"""

import time
import torch
import json
import numpy as np
from typing import List, Dict
from optimal_sampling import OptimalSamplingV1


def generate_prompts(num_prompts: int, complexity: str = "medium") -> List[str]:
    """生成测试prompts"""
    if complexity == "short":
        templates = [
            "What is {a} + {b}?",
            "Calculate: {a} × {b}",
            "Solve for x: {a}x = {b}",
        ]
    elif complexity == "medium":
        templates = [
            "A train travels at {a} km/h for {b} hours. What distance?",
            "In a class of {a} students, {b}% are boys. How many boys?",
            "A store sells apples at ${a} each. Cost of {b} apples?",
        ]
    else:  # long
        templates = [
            "A company has {a} employees working {b} hours per week at ${c}/hour. "
            "Calculate monthly salary cost for 4 weeks.",
            "A rectangular garden is {a}m × {b}m. A path of {c}m width surrounds it. "
            "Calculate the path area.",
            "In a school with {a} classes of {b} students each, {c}% attend a trip. "
            "Each bus holds {d} students. How many buses needed?",
        ]

    prompts = []
    for i in range(num_prompts):
        template = templates[i % len(templates)]
        params = {chr(97+j): np.random.randint(2, 20) for j in range(4)}
        prompts.append(template.format(**params))

    return prompts


def run_benchmark_config(
    sampler: OptimalSamplingV1,
    batch_size: int,
    max_tokens: int,
    use_optimal_sampling: bool = True
) -> Dict:
    """运行单个配置的 benchmark"""

    mode = "optimal" if use_optimal_sampling else "baseline"

    # 生成 prompts
    complexity = "short" if max_tokens <= 1024 else "medium" if max_tokens <= 2048 else "long"
    prompts = generate_prompts(batch_size, complexity)

    # 预热 (如果是第一次运行)
    if not hasattr(run_benchmark_config, '_warmed_up'):
        print("  [预热] Running warmup...")
        sampler.generate(
            prompts=prompts[:min(2, batch_size)],
            max_tokens=min(100, max_tokens),
            temperature=0.8,
            use_optimal_sampling=use_optimal_sampling
        )
        run_benchmark_config._warmed_up = True

    # 测量GPU内存
    torch.cuda.reset_peak_memory_stats()
    mem_before = torch.cuda.memory_allocated() / 1024**3

    # 开始计时
    start_time = time.time()

    try:
        # 生成
        outputs = sampler.generate(
            prompts=prompts,
            max_tokens=max_tokens,
            temperature=0.8,
            use_optimal_sampling=use_optimal_sampling
        )

        end_time = time.time()
        elapsed = end_time - start_time

        # 计算指标
        mem_after = torch.cuda.memory_allocated() / 1024**3
        mem_peak = torch.cuda.max_memory_allocated() / 1024**3

        total_tokens = sum(outputs.num_tokens)
        throughput = total_tokens / elapsed
        latency_per_request = elapsed / batch_size
        avg_tokens_per_request = total_tokens / batch_size

        result = {
            "batch_size": batch_size,
            "max_tokens": max_tokens,
            "mode": mode,
            "elapsed_time": elapsed,
            "total_tokens": total_tokens,
            "throughput": throughput,
            "latency_per_request": latency_per_request,
            "avg_tokens_per_request": avg_tokens_per_request,
            "memory_before_gb": mem_before,
            "memory_after_gb": mem_after,
            "memory_peak_gb": mem_peak,
            "success": True,
            "error": None
        }

    except Exception as e:
        result = {
            "batch_size": batch_size,
            "max_tokens": max_tokens,
            "mode": mode,
            "success": False,
            "error": str(e)
        }
        print(f"  ❌ Error: {e}")

    return result


def main():
    print("=" * 80)
    print("大规模性能基准测试：序列长度 vs 批量大小")
    print("=" * 80)

    # 测试矩阵
    max_tokens_list = [1024, 2048, 4096, 8192]  # 1k, 2k, 4k, 8k
    batch_sizes = [1, 4, 8, 16, 32, 64]

    total_configs = len(max_tokens_list) * len(batch_sizes)

    print(f"\n测试矩阵:")
    print(f"  序列长度: {max_tokens_list}")
    print(f"  批量大小: {batch_sizes}")
    print(f"  总配置数: {total_configs}")
    print(f"  每个配置测试 2 种模式 (Optimal + Baseline)")
    print(f"  总测试数: {total_configs * 2}")

    # 初始化 sampler (全局复用)
    print(f"\n[初始化] Loading models...")
    sampler = OptimalSamplingV1(
        model_teacher="Qwen/Qwen2.5-3B-Instruct",
        model_theta="Qwen/Qwen2.5-1.5B-Instruct",
        alpha_method="kl_symmetry",
        gpu_memory_utilization=0.45,
        track_alpha_stats=False,  # 性能模式
    )
    print("✅ Models loaded!")

    # 运行所有配置
    all_results = []
    config_idx = 0

    for max_tokens in max_tokens_list:
        for batch_size in batch_sizes:
            config_idx += 1

            print(f"\n{'=' * 80}")
            print(f"配置 {config_idx}/{total_configs}: "
                  f"BS={batch_size}, MT={max_tokens}")
            print(f"{'=' * 80}")

            # Optimal Sampling
            print(f"\n[{config_idx}.1] Optimal Sampling")
            optimal_result = run_benchmark_config(
                sampler,
                batch_size=batch_size,
                max_tokens=max_tokens,
                use_optimal_sampling=True
            )

            if optimal_result["success"]:
                print(f"  ✅ 完成")
                print(f"     时间: {optimal_result['elapsed_time']:.2f}s")
                print(f"     吞吐量: {optimal_result['throughput']:.2f} tok/s")
                print(f"     延迟: {optimal_result['latency_per_request']:.3f}s/req")
                print(f"     Tokens/req: {optimal_result['avg_tokens_per_request']:.1f}")

            all_results.append(optimal_result)

            # Teacher-only Baseline
            print(f"\n[{config_idx}.2] Teacher-only Baseline")
            baseline_result = run_benchmark_config(
                sampler,
                batch_size=batch_size,
                max_tokens=max_tokens,
                use_optimal_sampling=False
            )

            if baseline_result["success"]:
                print(f"  ✅ 完成")
                print(f"     时间: {baseline_result['elapsed_time']:.2f}s")
                print(f"     吞吐量: {baseline_result['throughput']:.2f} tok/s")
                print(f"     延迟: {baseline_result['latency_per_request']:.3f}s/req")
                print(f"     Tokens/req: {baseline_result['avg_tokens_per_request']:.1f}")

                # 计算加速比
                if optimal_result["success"]:
                    speedup = baseline_result['throughput'] / optimal_result['throughput']
                    print(f"\n     ⚖️  Baseline vs Optimal: {speedup:.2f}x faster")

            all_results.append(baseline_result)

            # 保存中间结果
            with open("large_scale_benchmark_results.json", 'w') as f:
                json.dump(all_results, f, indent=2)

    # 生成总结报告
    print("\n\n" + "=" * 80)
    print("总结报告")
    print("=" * 80)

    # 分析最佳配置
    optimal_results = [r for r in all_results if r["success"] and r["mode"] == "optimal"]

    if optimal_results:
        # 按吞吐量排序
        sorted_by_throughput = sorted(optimal_results, key=lambda x: x["throughput"], reverse=True)
        best = sorted_by_throughput[0]

        print(f"\n🏆 最佳吞吐量配置 (Optimal Sampling):")
        print(f"   批量大小: {best['batch_size']}")
        print(f"   序列长度: {best['max_tokens']}")
        print(f"   吞吐量: {best['throughput']:.2f} tok/s")
        print(f"   延迟: {best['latency_per_request']:.3f}s/req")

        # 按延迟排序
        sorted_by_latency = sorted(optimal_results, key=lambda x: x["latency_per_request"])
        lowest_latency = sorted_by_latency[0]

        print(f"\n⚡ 最低延迟配置 (Optimal Sampling):")
        print(f"   批量大小: {lowest_latency['batch_size']}")
        print(f"   序列长度: {lowest_latency['max_tokens']}")
        print(f"   延迟: {lowest_latency['latency_per_request']:.3f}s/req")
        print(f"   吞吐量: {lowest_latency['throughput']:.2f} tok/s")

    # 生成热力图数据
    print(f"\n\n📊 吞吐量热力图 (Optimal Sampling, tok/s):")
    print(f"{'BS \\ MT':<10}", end="")
    for mt in max_tokens_list:
        print(f"{mt:>10}", end="")
    print()
    print("-" * (10 + 10 * len(max_tokens_list)))

    for bs in batch_sizes:
        print(f"{bs:<10}", end="")
        for mt in max_tokens_list:
            # 查找对应结果
            result = next(
                (r for r in optimal_results
                 if r["batch_size"] == bs and r["max_tokens"] == mt),
                None
            )
            if result:
                print(f"{result['throughput']:>10.1f}", end="")
            else:
                print(f"{'N/A':>10}", end="")
        print()

    print(f"\n\n📊 延迟热力图 (Optimal Sampling, s/req):")
    print(f"{'BS \\ MT':<10}", end="")
    for mt in max_tokens_list:
        print(f"{mt:>10}", end="")
    print()
    print("-" * (10 + 10 * len(max_tokens_list)))

    for bs in batch_sizes:
        print(f"{bs:<10}", end="")
        for mt in max_tokens_list:
            result = next(
                (r for r in optimal_results
                 if r["batch_size"] == bs and r["max_tokens"] == mt),
                None
            )
            if result:
                print(f"{result['latency_per_request']:>10.3f}", end="")
            else:
                print(f"{'N/A':>10}", end="")
        print()

    # 对比 Optimal vs Baseline
    print(f"\n\n⚖️  Speedup 热力图 (Baseline / Optimal, 倍数):")
    print(f"{'BS \\ MT':<10}", end="")
    for mt in max_tokens_list:
        print(f"{mt:>10}", end="")
    print()
    print("-" * (10 + 10 * len(max_tokens_list)))

    for bs in batch_sizes:
        print(f"{bs:<10}", end="")
        for mt in max_tokens_list:
            optimal = next(
                (r for r in all_results
                 if r["success"] and r["mode"] == "optimal"
                 and r["batch_size"] == bs and r["max_tokens"] == mt),
                None
            )
            baseline = next(
                (r for r in all_results
                 if r["success"] and r["mode"] == "baseline"
                 and r["batch_size"] == bs and r["max_tokens"] == mt),
                None
            )
            if optimal and baseline:
                speedup = baseline["throughput"] / optimal["throughput"]
                print(f"{speedup:>10.2f}", end="")
            else:
                print(f"{'N/A':>10}", end="")
        print()

    # 性能趋势分析
    print(f"\n\n📈 性能趋势分析:")

    # 按序列长度分析
    print(f"\n1. 序列长度的影响 (固定 BS=16):")
    for mt in max_tokens_list:
        result = next(
            (r for r in optimal_results
             if r["batch_size"] == 16 and r["max_tokens"] == mt),
            None
        )
        if result:
            print(f"   MT={mt:>4}: {result['throughput']:>7.2f} tok/s, "
                  f"{result['latency_per_request']:>6.3f}s/req")

    # 按批量大小分析
    print(f"\n2. 批量大小的影响 (固定 MT=2048):")
    for bs in batch_sizes:
        result = next(
            (r for r in optimal_results
             if r["batch_size"] == bs and r["max_tokens"] == 2048),
            None
        )
        if result:
            print(f"   BS={bs:>2}: {result['throughput']:>7.2f} tok/s, "
                  f"{result['latency_per_request']:>6.3f}s/req")

    # 保存最终结果
    output_file = "large_scale_benchmark_results.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n\n✅ 结果已保存到: {output_file}")

    print("\n" + "=" * 80)
    print("🎉 大规模基准测试完成！")
    print("=" * 80)


if __name__ == '__main__':
    main()
