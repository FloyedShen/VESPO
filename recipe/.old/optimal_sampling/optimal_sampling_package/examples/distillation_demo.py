#!/usr/bin/env python3
"""
Semi On-Policy Distillation Demo

This example demonstrates how to use optimal sampling for semi on-policy
distillation in mathematical reasoning tasks.

Scenario: Small model learns p(reasoning | problem, answer, prompt)
instead of struggling with p(answer | problem).
"""

from optimal_sampling import OptimalSamplingV1
import yaml


def load_config(config_path="configs/distillation.yaml"):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_conditional_prompt(problem, oracle_answer, template):
    """
    Create conditional generation prompt for semi on-policy distillation

    Args:
        problem: The math problem
        oracle_answer: Correct answer from large model/oracle
        template: Template string with {problem} and {answer} placeholders

    Returns:
        Formatted prompt
    """
    return template.format(problem=problem, answer=oracle_answer)


def main():
    print("=" * 80)
    print("SEMI ON-POLICY DISTILLATION DEMO")
    print("=" * 80)

    # Math problems with oracle answers (from larger model or ground truth)
    problems = [
        {
            "problem": "If 2x + 3 = 7, what is x?",
            "oracle_answer": "x = 2",
            "explanation": "Basic algebra"
        },
        {
            "problem": "What is the derivative of x^2 + 3x + 1?",
            "oracle_answer": "2x + 3",
            "explanation": "Calculus - power rule"
        },
        {
            "problem": "A rectangle has length 5 and width 3. What is its area?",
            "oracle_answer": "15 square units",
            "explanation": "Geometry - area formula"
        },
    ]

    # Initialize sampler with different prompts for teacher and theta
    print("\n[1/4] Initializing sampler with semi on-policy configuration...")
    sampler = OptimalSamplingV1(
        model_teacher="Qwen/Qwen2.5-3B-Instruct",  # Oracle (larger in real use)
        model_theta="Qwen/Qwen2.5-1.5B-Instruct",   # Student

        # Teacher: Given answer, generate reasoning
        teacher_system_prompt=(
            "Given a math problem and the correct answer, "
            "generate detailed step-by-step reasoning that leads to the answer. "
            "Focus on clear logical steps."
        ),

        # Theta: Learn to generate reasoning from scratch
        theta_system_prompt=(
            "You are a math problem solver. "
            "Solve problems step by step, showing your reasoning clearly."
        ),

        enable_chat_template=True,
        alpha_method="kl_symmetry",
        gpu_memory_utilization=0.45,
        track_alpha_stats=True,
    )
    print("✅ Sampler initialized!")

    # Generate with different prefix templates
    print("\n[2/4] Sampling with multiple prefix strategies...")

    prefix_templates = [
        "Problem: {problem}\nAnswer: {answer}\nDetailed reasoning:",
        "Q: {problem}\nA: {answer}\nExplanation:",
        "Given: {problem}\nSolution: {answer}\nSteps:",
    ]

    all_samples = []
    for template_idx, template in enumerate(prefix_templates):
        print(f"\n  Strategy {template_idx + 1}/{len(prefix_templates)}: {template.split(':')[0]}...")

        # Create conditional prompts
        prompts = [
            create_conditional_prompt(p["problem"], p["oracle_answer"], template)
            for p in problems
        ]

        # Generate
        outputs = sampler.generate(
            prompts=prompts,
            max_tokens=512,
            temperature=0.8,
            use_optimal_sampling=True
        )

        # Store samples
        for i, problem_data in enumerate(problems):
            all_samples.append({
                "problem": problem_data["problem"],
                "oracle_answer": problem_data["oracle_answer"],
                "template": template,
                "reasoning": outputs.generated_texts[i],
                "num_tokens": outputs.num_tokens[i],
            })

    print("✅ Sampling complete!")

    # Display results
    print("\n[3/4] Generated Samples:")
    print("=" * 80)
    for i, sample in enumerate(all_samples):
        if i % len(prefix_templates) == 0:
            print(f"\n{'─' * 80}")
            print(f"Problem: {sample['problem']}")
            print(f"Oracle Answer: {sample['oracle_answer']}")
            print(f"{'─' * 80}")

        print(f"\nTemplate {i % len(prefix_templates) + 1}:")
        print(f"Reasoning: {sample['reasoning'][:200]}...")
        print(f"Tokens: {sample['num_tokens']}")

    # Quality analysis
    print("\n[4/4] Quality Analysis:")
    print("=" * 80)

    avg_length = sum(s["num_tokens"] for s in all_samples) / len(all_samples)
    print(f"Average reasoning length: {avg_length:.1f} tokens")

    # Check if oracle answer appears in reasoning
    contains_answer = sum(
        1 for s in all_samples
        if s["oracle_answer"].lower() in s["reasoning"].lower()
    )
    answer_match_rate = contains_answer / len(all_samples) * 100
    print(f"Answer verification rate: {answer_match_rate:.1f}%")

    print(f"\nTotal samples generated: {len(all_samples)}")
    print(f"Unique problems: {len(problems)}")
    print(f"Prefix strategies: {len(prefix_templates)}")

    print("\n" + "=" * 80)
    print("✅ SEMI ON-POLICY DISTILLATION DEMO COMPLETED!")
    print("=" * 80)
    print("\n[Key Insights]:")
    print("• Teacher model guided by oracle answers generates high-quality reasoning")
    print("• Theta model learns from on-policy distribution")
    print("• Multiple prefix strategies improve coverage")
    print("• Optimal mixing (α) balances quality and diversity")


if __name__ == '__main__':
    main()
