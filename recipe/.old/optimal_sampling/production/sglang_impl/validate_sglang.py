"""
Validation script for SGLang-style implementation

Comprehensive testing of the optimal sampling implementation.
"""

import os

import sys
import time
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from production.sglang_impl import OptimalSamplingSGLang


def print_gpu_info():
    """Print GPU information"""
    print(f"\n💾 GPU Information:")
    print(f"   Device: {torch.cuda.get_device_name(0)}")
    total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    available_mem = torch.cuda.mem_get_info()[0] / 1e9
    print(f"   Total memory: {total_mem:.2f} GB")
    print(f"   Available: {available_mem:.2f} GB")


def test_basic_generation():
    """Test 1: Basic generation with diagnostics"""
    print("\n" + "=" * 80)
    print("TEST 1: Basic Generation")
    print("=" * 80)

    try:
        sampler = OptimalSamplingSGLang(
            model_base="Qwen/Qwen2.5-0.5B",
            model_teacher="Qwen/Qwen2.5-1.5B",
            alpha_method="kl_symmetry",
            gpu_memory_utilization=0.85
        )

        prompts = [
            "What is machine learning?",
            "Explain neural networks in one sentence.",
        ]

        print(f"\n🔄 Generating...")
        start = time.time()

        outputs = sampler.generate(
            prompts=prompts,
            max_tokens=80,
            temperature=0.8,
            collect_diagnostics=True,
            verbose=True
        )

        elapsed = time.time() - start

        # Show results
        print("\n" + "=" * 80)
        print("📊 Results")
        print("=" * 80)

        for i, (prompt, text) in enumerate(zip(prompts, outputs.generated_texts)):
            print(f"\n{'─' * 80}")
            print(f"Prompt {i+1}: {prompt}")
            print(f"Generated: {text}")
            print(f"Tokens: {outputs.num_tokens[i]}")

            if outputs.alphas:
                alpha_mean = np.mean(outputs.alphas[i])
                ess_mean = np.mean(outputs.ess_ratios[i]) if outputs.ess_ratios else None
                print(f"Alpha mean: {alpha_mean:.3f}")
                if ess_mean:
                    print(f"ESS ratio: {ess_mean:.3f}")

        total_tokens = sum(outputs.num_tokens)
        throughput = total_tokens / elapsed

        print(f"\n⏱️  Performance:")
        print(f"   Total time: {elapsed:.2f}s")
        print(f"   Total tokens: {total_tokens}")
        print(f"   Throughput: {throughput:.1f} tokens/s")

        print("\n✅ Test 1 PASSED")
        return True

    except Exception as e:
        print(f"\n❌ Test 1 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_alpha_methods():
    """Test 2: Different alpha methods"""
    print("\n" + "=" * 80)
    print("TEST 2: Alpha Methods Comparison")
    print("=" * 80)

    methods = ["kl_symmetry", "entropy", "fixed"]
    prompt = "Explain the theory of relativity."

    results = {}

    for method in methods:
        print(f"\n{'─' * 80}")
        print(f"Method: {method}")
        print(f"{'─' * 80}")

        try:
            sampler = OptimalSamplingSGLang(
                model_base="Qwen/Qwen2.5-0.5B",
                model_teacher="Qwen/Qwen2.5-1.5B",
                alpha_method=method,
                fixed_alpha=0.7 if method == "fixed" else 0.5,
                gpu_memory_utilization=0.85
            )

            outputs = sampler.generate(
                prompts=[prompt],
                max_tokens=60,
                temperature=0.8,
                collect_diagnostics=True
            )

            text = outputs.generated_texts[0]
            alpha_mean = np.mean(outputs.alphas[0]) if outputs.alphas else None

            results[method] = {
                "text": text,
                "alpha_mean": alpha_mean,
                "num_tokens": outputs.num_tokens[0]
            }

            print(f"Output: {text[:100]}...")
            print(f"Alpha mean: {alpha_mean:.3f}")
            print(f"Tokens: {outputs.num_tokens[0]}")

        except Exception as e:
            print(f"❌ Failed: {e}")
            results[method] = None

    # Summary
    print("\n" + "=" * 80)
    print("📊 Method Comparison Summary")
    print("=" * 80)

    for method, result in results.items():
        if result:
            print(f"\n{method}:")
            print(f"  Alpha: {result['alpha_mean']:.3f}")
            print(f"  Tokens: {result['num_tokens']}")
            print(f"  Length: {len(result['text'])} chars")

    print("\n✅ Test 2 COMPLETED")


def test_memory_usage():
    """Test 3: Memory usage"""
    print("\n" + "=" * 80)
    print("TEST 3: Memory Usage")
    print("=" * 80)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    initial = torch.cuda.memory_allocated() / 1e9

    print(f"\n💾 Initial memory: {initial:.2f} GB")

    try:
        print("\n📦 Loading models...")
        sampler = OptimalSamplingSGLang(
            model_base="Qwen/Qwen2.5-0.5B",
            model_teacher="Qwen/Qwen2.5-1.5B",
            alpha_method="kl_symmetry",
            gpu_memory_utilization=0.85
        )

        after_load = torch.cuda.memory_allocated() / 1e9
        print(f"   After loading: {after_load:.2f} GB")
        print(f"   Model memory: {after_load - initial:.2f} GB")

        # Generate
        print("\n🔄 Generating...")
        outputs = sampler.generate(
            prompts=["Test prompt"] * 4,
            max_tokens=50
        )

        peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"   Peak during gen: {peak:.2f} GB")

        total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        utilization = peak / total_mem * 100

        print(f"\n📊 Memory Summary:")
        print(f"   Model memory: {after_load - initial:.2f} GB")
        print(f"   Peak memory: {peak:.2f} GB")
        print(f"   GPU utilization: {utilization:.1f}%")

        print("\n✅ Test 3 PASSED")

    except Exception as e:
        print(f"\n❌ Test 3 FAILED: {e}")
        import traceback
        traceback.print_exc()


def main():
    print("\n" + "=" * 80)
    print("🚀 SGLang-style Optimal Sampling Validation")
    print("=" * 80)

    print_gpu_info()

    # Test 1: Basic generation
    success = test_basic_generation()

    if not success:
        print("\n❌ Basic test failed. Stopping.")
        return

    # Clean up
    torch.cuda.empty_cache()

    # Test 2: Alpha methods
    test_alpha_methods()

    # Clean up
    torch.cuda.empty_cache()

    # Test 3: Memory
    test_memory_usage()

    print("\n" + "=" * 80)
    print("✅ All Validation Tests Completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
