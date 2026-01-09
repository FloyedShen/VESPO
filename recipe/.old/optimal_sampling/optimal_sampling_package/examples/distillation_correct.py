#!/usr/bin/env python3
"""
Semi On-Policy Distillation - Correct Implementation

Demonstrates the CORRECT way to do semi on-policy distillation:
- Teacher receives: Problem + Answer → Reasoning (conditional generation)
- Student receives: Problem → Reasoning (learns direct inference)

This ensures the student doesn't "cheat" by seeing the answer!
"""

from optimal_sampling import OptimalSamplingV1


def main():
    print("=" * 80)
    print("SEMI ON-POLICY DISTILLATION - CORRECT IMPLEMENTATION")
    print("=" * 80)
    print("\n[Key Idea]")
    print("• Teacher (π_t): Sees problem + answer → generates reasoning")
    print("• Student (π_θ): Only sees problem → learns to reason from scratch")
    print("• Optimal mixing balances quality (teacher) and on-policy (student)")
    print("=" * 80)

    # Math problems with ground truth answers
    problems = [
        {
            "problem": "If 2x + 3 = 7, solve for x.",
            "answer": "x = 2",
        },
        {
            "problem": "What is the derivative of x^2 + 3x + 1?",
            "answer": "2x + 3",
        },
        {
            "problem": "A rectangle has length 5 and width 3. What is its area?",
            "answer": "15 square units",
        },
    ]

    # Initialize sampler with different prompts for teacher and student
    print("\n[1/3] Initializing Optimal Sampling...")
    sampler = OptimalSamplingV1(
        model_teacher="Qwen/Qwen2.5-3B-Instruct",
        model_theta="Qwen/Qwen2.5-1.5B-Instruct",

        # Teacher: Given answer, generate reasoning
        teacher_system_prompt=(
            "You are a math expert. Given a problem and the correct answer, "
            "explain the detailed reasoning that leads to this answer."
        ),

        # Student: Learn to reason from scratch
        theta_system_prompt=(
            "You are a math student learning to solve problems. "
            "Show your step-by-step reasoning."
        ),

        enable_chat_template=False,  # Using simple text format
        alpha_method="kl_symmetry",
        gpu_memory_utilization=0.45,
        track_alpha_stats=True,
    )
    print("✅ Sampler initialized!")

    # Prepare DIFFERENT prompts for teacher and student
    print("\n[2/3] Preparing prompts...")

    teacher_prompts = []  # Teacher sees: Problem + Answer
    theta_prompts = []    # Student sees: Only Problem

    for p in problems:
        # Teacher prompt: includes the answer (conditional generation)
        teacher_prompt = (
            f"Problem: {p['problem']}\n"
            f"Correct Answer: {p['answer']}\n"
            f"Detailed Reasoning:"
        )
        teacher_prompts.append(teacher_prompt)

        # Student prompt: NO answer (must learn to infer)
        theta_prompt = (
            f"Problem: {p['problem']}\n"
            f"Reasoning:"
        )
        theta_prompts.append(theta_prompt)

        print(f"\n[Problem {len(teacher_prompts)}]")
        print(f"  Question: {p['problem']}")
        print(f"  Ground Truth: {p['answer']}")
        print(f"  Teacher Prompt: ...Answer: {p['answer']}... ✅")
        print(f"  Student Prompt: ...Problem only... ✅")

    # Generate with optimal sampling
    print("\n[3/3] Generating with Optimal Sampling...")
    print("  (Teacher guided by answer, Student learns from problem)")

    outputs = sampler.generate(
        prompts=teacher_prompts,       # Teacher: Problem + Answer
        theta_prompts=theta_prompts,    # Student: Problem only ✅
        max_tokens=256,
        temperature=0.8,
        use_optimal_sampling=True
    )

    print("✅ Generation complete!")

    # Display results
    print("\n" + "=" * 80)
    print("RESULTS: Optimal Sampling (Teacher + Student Mixed)")
    print("=" * 80)

    for i, (problem, text, num_tokens) in enumerate(zip(
        problems, outputs.generated_texts, outputs.num_tokens
    )):
        print(f"\n[Problem {i+1}]: {problem['problem']}")
        print(f"[Ground Truth]: {problem['answer']}")
        print(f"\n[Generated Reasoning]: ({num_tokens} tokens)")
        print(text)
        print("-" * 80)

    # Show alpha statistics
    if outputs.alpha_stats:
        print("\n[Alpha Statistics]:")
        print(f"  Mean α: {outputs.alpha_stats['mean']:.3f} (teacher weight)")
        print(f"  Std α: {outputs.alpha_stats['std']:.3f}")
        print(f"  Range: [{outputs.alpha_stats['min']:.3f}, {outputs.alpha_stats['max']:.3f}]")
        print(f"\n  Interpretation:")
        print(f"    • α = {outputs.alpha_stats['mean']:.3f} means {outputs.alpha_stats['mean']*100:.1f}% teacher, "
              f"{(1-outputs.alpha_stats['mean'])*100:.1f}% student")
        print(f"    • Balances teacher's quality with student's on-policy distribution")

    print("\n" + "=" * 80)
    print("✅ SEMI ON-POLICY DISTILLATION DEMO COMPLETED!")
    print("=" * 80)

    print("\n[Key Takeaways]:")
    print("  1. ✅ Teacher has access to answer → generates high-quality reasoning")
    print("  2. ✅ Student doesn't see answer → stays on-policy")
    print("  3. ✅ Optimal mixing (α) balances both distributions")
    print("  4. ✅ Result: High-quality, on-policy training data")

    print("\n[Usage in Training]:")
    print("  • Collect many such (problem, reasoning) pairs")
    print("  • Use them to fine-tune the student model")
    print("  • Student learns p(reasoning | problem) with teacher's quality")
    print("  • Much better than pure off-policy (teacher-only) data")


if __name__ == '__main__':
    main()
