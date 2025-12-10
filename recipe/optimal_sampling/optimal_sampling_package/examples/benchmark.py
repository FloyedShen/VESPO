#!/usr/bin/env python3
"""
Benchmark: Speed and Correctness for Optimal Sampling

This benchmark measures:
1. Throughput (tokens/second)
2. Latency (time per request)
3. Correctness (for math problems with known answers)
4. Alpha statistics
5. Comparison: Optimal Sampling vs Teacher-only baseline
"""

import time
import re
from typing import List, Dict, Tuple
from optimal_sampling import OptimalSamplingV1
import json


# Test dataset: Math problems with known answers
MATH_PROBLEMS = [
    {
        "problem": "If 2x + 3 = 7, solve for x.",
        "answer": "2",
        "type": "algebra"
    },
    {
        "problem": "What is 15% of 200?",
        "answer": "30",
        "type": "arithmetic"
    },
    {
        "problem": "A triangle has sides 3, 4, and 5. Is it a right triangle?",
        "answer": "yes",
        "type": "geometry"
    },
    {
        "problem": "What is the factorial of 5?",
        "answer": "120",
        "type": "arithmetic"
    },
    {
        "problem": "If y = 2x + 1 and x = 3, what is y?",
        "answer": "7",
        "type": "algebra"
    },
    {
        "problem": "What is the square root of 144?",
        "answer": "12",
        "type": "arithmetic"
    },
    {
        "problem": "Convert 3/4 to a decimal.",
        "answer": "0.75",
        "type": "arithmetic"
    },
    {
        "problem": "What is 7 × 8?",
        "answer": "56",
        "type": "arithmetic"
    },
]


def extract_answer(text: str) -> str:
    """
    Extract numerical answer from generated text

    Looks for patterns like "x = 2", "answer is 2", "= 2", etc.
    """
    text = text.lower()

    # Pattern 1: "x = 2" or "answer is 2"
    pattern1 = r'(?:x\s*=|answer\s*(?:is|:))\s*([0-9.]+)'
    match = re.search(pattern1, text)
    if match:
        return match.group(1)

    # Pattern 2: Final number in text
    pattern2 = r'([0-9.]+)\s*(?:\.|$)'
    matches = re.findall(pattern2, text)
    if matches:
        return matches[-1]  # Return last number

    # Pattern 3: Yes/No questions
    if 'yes' in text:
        return 'yes'
    if 'no' in text:
        return 'no'

    return ""


def check_correctness(generated: str, expected: str) -> bool:
    """Check if generated text contains the correct answer"""
    extracted = extract_answer(generated)
    return extracted.strip() == expected.strip()


def run_benchmark(
    sampler: OptimalSamplingV1,
    problems: List[Dict],
    num_rounds: int = 3,
    use_optimal_sampling: bool = True
) -> Dict:
    """
    Run benchmark on a set of problems

    Args:
        sampler: Initialized OptimalSamplingV1 instance
        problems: List of problem dicts with 'problem' and 'answer' keys
        num_rounds: Number of times to repeat for averaging
        use_optimal_sampling: If False, uses teacher-only baseline

    Returns:
        Dictionary with benchmark results
    """
    mode = "Optimal Sampling" if use_optimal_sampling else "Teacher-only Baseline"
    print(f"\n{'=' * 80}")
    print(f"Running Benchmark: {mode}")
    print(f"{'=' * 80}")
    print(f"Problems: {len(problems)}")
    print(f"Rounds: {num_rounds}")

    all_results = []
    total_tokens = 0
    total_time = 0
    correct_count = 0
    total_count = 0

    for round_idx in range(num_rounds):
        print(f"\n[Round {round_idx + 1}/{num_rounds}]")

        prompts = [p["problem"] for p in problems]

        # Measure time
        start_time = time.time()

        outputs = sampler.generate(
            prompts=prompts,
            max_tokens=256,
            temperature=0.7,
            use_optimal_sampling=use_optimal_sampling
        )

        end_time = time.time()
        elapsed = end_time - start_time

        # Calculate metrics
        round_tokens = sum(outputs.num_tokens)
        total_tokens += round_tokens
        total_time += elapsed

        throughput = round_tokens / elapsed

        print(f"  Time: {elapsed:.2f}s")
        print(f"  Tokens: {round_tokens}")
        print(f"  Throughput: {throughput:.2f} tok/s")

        # Check correctness
        for i, (problem, text) in enumerate(zip(problems, outputs.generated_texts)):
            is_correct = check_correctness(text, problem["answer"])
            if is_correct:
                correct_count += 1
            total_count += 1

            all_results.append({
                "round": round_idx + 1,
                "problem_idx": i,
                "problem": problem["problem"],
                "expected_answer": problem["answer"],
                "generated_text": text,
                "num_tokens": outputs.num_tokens[i],
                "is_correct": is_correct
            })

    # Aggregate metrics
    avg_throughput = total_tokens / total_time
    avg_latency = total_time / total_count
    accuracy = correct_count / total_count * 100

    print(f"\n{'─' * 80}")
    print(f"RESULTS: {mode}")
    print(f"{'─' * 80}")
    print(f"Average Throughput: {avg_throughput:.2f} tok/s")
    print(f"Average Latency: {avg_latency:.3f} s/request")
    print(f"Accuracy: {accuracy:.1f}% ({correct_count}/{total_count})")
    print(f"Total Tokens: {total_tokens}")
    print(f"Total Time: {total_time:.2f}s")

    return {
        "mode": mode,
        "throughput": avg_throughput,
        "latency": avg_latency,
        "accuracy": accuracy,
        "correct_count": correct_count,
        "total_count": total_count,
        "total_tokens": total_tokens,
        "total_time": total_time,
        "results": all_results
    }


def main():
    print("=" * 80)
    print("OPTIMAL SAMPLING BENCHMARK")
    print("Speed & Correctness Evaluation")
    print("=" * 80)

    # Initialize sampler
    print("\n[1/4] Initializing sampler...")
    sampler = OptimalSamplingV1(
        model_teacher="Qwen/Qwen2.5-3B-Instruct",
        model_theta="Qwen/Qwen2.5-1.5B-Instruct",
        alpha_method="kl_symmetry",
        gpu_memory_utilization=0.45,
        track_alpha_stats=False,  # Disable for fair speed comparison
    )
    print("✅ Sampler initialized!")

    # Run benchmark with optimal sampling
    print("\n[2/4] Benchmarking: Optimal Sampling...")
    optimal_results = run_benchmark(
        sampler,
        MATH_PROBLEMS,
        num_rounds=2,
        use_optimal_sampling=True
    )

    # Run benchmark with teacher-only baseline
    print("\n[3/4] Benchmarking: Teacher-only Baseline...")
    baseline_results = run_benchmark(
        sampler,
        MATH_PROBLEMS,
        num_rounds=2,
        use_optimal_sampling=False
    )

    # Compare results
    print("\n[4/4] Comparison:")
    print("=" * 80)
    print(f"{'Metric':<30} {'Optimal Sampling':<20} {'Teacher-only':<20} {'Ratio':<10}")
    print("─" * 80)

    # Throughput
    throughput_ratio = optimal_results['throughput'] / baseline_results['throughput']
    print(f"{'Throughput (tok/s)':<30} {optimal_results['throughput']:>18.2f} {baseline_results['throughput']:>18.2f} {throughput_ratio:>8.2f}x")

    # Latency
    latency_ratio = optimal_results['latency'] / baseline_results['latency']
    print(f"{'Latency (s/request)':<30} {optimal_results['latency']:>18.3f} {baseline_results['latency']:>18.3f} {latency_ratio:>8.2f}x")

    # Accuracy
    acc_diff = optimal_results['accuracy'] - baseline_results['accuracy']
    print(f"{'Accuracy (%)':<30} {optimal_results['accuracy']:>18.1f} {baseline_results['accuracy']:>18.1f} {acc_diff:>+8.1f}")

    print("=" * 80)

    # Analysis
    print("\n[Analysis]:")
    if throughput_ratio < 0.9:
        print(f"⚠️  Optimal sampling is {(1-throughput_ratio)*100:.1f}% slower (overhead from theta model)")
    elif throughput_ratio > 1.1:
        print(f"✅ Optimal sampling is {(throughput_ratio-1)*100:.1f}% faster (unexpected, check setup)")
    else:
        print(f"✅ Throughput comparable ({abs(throughput_ratio-1)*100:.1f}% difference)")

    if acc_diff > 5:
        print(f"✅ Optimal sampling improves accuracy by {acc_diff:.1f}%")
    elif acc_diff < -5:
        print(f"⚠️  Optimal sampling decreases accuracy by {abs(acc_diff):.1f}%")
    else:
        print(f"📊 Accuracy similar (difference: {acc_diff:.1f}%)")

    # Show sample outputs
    print("\n[Sample Outputs]:")
    print("=" * 80)
    for i in range(min(3, len(MATH_PROBLEMS))):
        problem = MATH_PROBLEMS[i]
        optimal_sample = [r for r in optimal_results['results'] if r['problem_idx'] == i and r['round'] == 1][0]
        baseline_sample = [r for r in baseline_results['results'] if r['problem_idx'] == i and r['round'] == 1][0]

        print(f"\nProblem {i+1}: {problem['problem']}")
        print(f"Expected: {problem['answer']}")
        print(f"\n[Optimal Sampling] {'✅' if optimal_sample['is_correct'] else '❌'}")
        print(f"  {optimal_sample['generated_text'][:150]}...")
        print(f"\n[Teacher-only] {'✅' if baseline_sample['is_correct'] else '❌'}")
        print(f"  {baseline_sample['generated_text'][:150]}...")
        print("─" * 80)

    # Save results
    output_file = "benchmark_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            "optimal_sampling": optimal_results,
            "teacher_only": baseline_results,
            "comparison": {
                "throughput_ratio": throughput_ratio,
                "latency_ratio": latency_ratio,
                "accuracy_diff": acc_diff
            }
        }, f, indent=2)

    print(f"\n✅ Results saved to: {output_file}")

    print("\n" + "=" * 80)
    print("✅ BENCHMARK COMPLETED!")
    print("=" * 80)


if __name__ == '__main__':
    main()
