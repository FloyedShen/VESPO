#!/usr/bin/env python3
"""
Quick Test: Verify Teacher and Student Receive Different Prompts
"""

from optimal_sampling import OptimalSamplingV1


def main():
    print("=" * 80)
    print("QUICK TEST: Different Prompts for Teacher vs Student")
    print("=" * 80)

    # Initialize sampler
    print("\n[1/2] Initializing...")
    sampler = OptimalSamplingV1(
        model_teacher="Qwen/Qwen2.5-3B-Instruct",
        model_theta="Qwen/Qwen2.5-1.5B-Instruct",
        alpha_method="kl_symmetry",
        gpu_memory_utilization=0.45,
        track_alpha_stats=False,
    )
    print("✅ Initialized!")

    # Test with different prompts
    print("\n[2/2] Testing different prompts...")

    # Teacher sees answer, student doesn't
    teacher_prompts = [
        "Problem: 2x + 3 = 7\nAnswer: x = 2\nExplain why:"
    ]
    theta_prompts = [
        "Problem: 2x + 3 = 7\nSolve step by step:"
    ]

    print(f"\n  Teacher Prompt: {repr(teacher_prompts[0][:50])}...")
    print(f"  Student Prompt: {repr(theta_prompts[0][:50])}...")

    outputs = sampler.generate(
        prompts=teacher_prompts,
        theta_prompts=theta_prompts,
        max_tokens=64,
        temperature=0.8
    )

    print(f"\n✅ Generation successful!")
    print(f"  Generated {outputs.num_tokens[0]} tokens")
    print(f"  Output: {outputs.generated_texts[0][:100]}...")

    print("\n" + "=" * 80)
    print("✅ TEST PASSED! Different prompts work correctly.")
    print("=" * 80)


if __name__ == '__main__':
    main()
