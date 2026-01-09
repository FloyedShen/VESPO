"""
测试脚本: 快速验证数据生成管线

使用小模型 (GPT-2) 快速测试所有功能
"""

import torch
import sys
from pathlib import Path

# 添加当前目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from optimal_sampling_model import create_optimal_sampling_model


def test_alpha_methods():
    """测试不同的alpha计算方法"""
    print("\n" + "="*60)
    print("Testing Alpha Methods")
    print("="*60)

    # 使用GPT-2进行快速测试
    model_name = "gpt2"

    for method in ["fixed", "entropy", "kl_symmetry"]:
        print(f"\n--- Testing {method} ---")

        try:
            model = create_optimal_sampling_model(
                model_theta=model_name,
                alpha_method=method,
                fixed_alpha=0.5 if method == "fixed" else None,
                device="cuda" if torch.cuda.is_available() else "cpu",
                dtype=torch.float32  # 使用float32以提高稳定性
            )

            outputs = model.generate(
                prompts=["Hello, how are you?"],
                max_new_tokens=10,
                temperature=1.0,
                return_diagnostics=True
            )

            print(f"✓ {method} method works!")
            print(f"  Generated: {outputs.generated_texts[0][:50]}...")
            print(f"  Alpha mean: {outputs.alpha_values.mean():.3f}")
            print(f"  ESS ratio mean: {outputs.ess_ratios.mean():.3f}")

        except Exception as e:
            print(f"✗ {method} method failed: {e}")

        # 清理内存
        if 'model' in locals():
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None


def test_batch_generation():
    """测试批量生成"""
    print("\n" + "="*60)
    print("Testing Batch Generation")
    print("="*60)

    model = create_optimal_sampling_model(
        model_theta="gpt2",
        alpha_method="fixed",
        fixed_alpha=0.5,
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype=torch.float32
    )

    prompts = [
        "What is AI?",
        "Explain quantum computing.",
        "How does the internet work?"
    ]

    outputs = model.generate(
        prompts=prompts,
        max_new_tokens=15,
        temperature=1.0,
        return_diagnostics=True
    )

    print(f"\n✓ Generated {len(outputs.generated_texts)} responses")
    for i, text in enumerate(outputs.generated_texts):
        print(f"\n{i+1}. Prompt: {prompts[i]}")
        print(f"   Response: {text[:60]}...")
        print(f"   Alpha: {outputs.alpha_values[i].mean():.3f}")
        print(f"   ESS ratio: {outputs.ess_ratios[i].mean():.3f}")


def test_diagnostics():
    """测试诊断信息完整性"""
    print("\n" + "="*60)
    print("Testing Diagnostics")
    print("="*60)

    model = create_optimal_sampling_model(
        model_theta="gpt2",
        alpha_method="kl_symmetry",
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype=torch.float32
    )

    outputs = model.generate(
        prompts=["Test prompt"],
        max_new_tokens=10,
        temperature=1.0,
        return_diagnostics=True
    )

    print("\n✓ Diagnostics keys:")
    for key in outputs.diagnostics.keys():
        values = outputs.diagnostics[key][0]  # 第一个样本
        print(f"  - {key}: shape={values.shape}, mean={values.mean():.4f}, std={values.std():.4f}")

    # 检查理论条件
    kl_theta = outputs.diagnostics["kl_theta"][0].mean()
    kl_t = outputs.diagnostics["kl_t"][0].mean()
    kl_diff = abs(kl_theta - kl_t)

    print(f"\n✓ KL Symmetry Check:")
    print(f"  D_KL(q*||π_θ) = {kl_theta:.4f}")
    print(f"  D_KL(q*||π_t)  = {kl_t:.4f}")
    print(f"  |Difference|    = {kl_diff:.4f}")
    print(f"  Status: {'✓ GOOD' if kl_diff < 0.01 else '⚠ Check required'}")

    ess_ratio = outputs.ess_ratios[0].mean()
    print(f"\n✓ ESS Ratio Check:")
    print(f"  ESS ratio = {ess_ratio:.4f}")
    print(f"  Status: {'✓ GOOD' if 0.5 < ess_ratio < 2.0 else '⚠ Check required'}")


def main():
    """运行所有测试"""
    print("\n" + "="*60)
    print("OptimalSamplingModel Test Suite")
    print("="*60)

    try:
        test_alpha_methods()
        test_batch_generation()
        test_diagnostics()

        print("\n" + "="*60)
        print("✓ All tests completed!")
        print("="*60 + "\n")

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
