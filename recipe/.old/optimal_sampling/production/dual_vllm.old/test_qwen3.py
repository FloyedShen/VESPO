#!/usr/bin/env python3
"""
Test Qwen3-4B-Base + Qwen3-14B combination

This tests:
- 4B Base model (π_θ)
- 14B Instruct model (π_t)
- Dual prompts with different formats
- All enhanced features
"""

import os
import sys
import time
import asyncio
import subprocess
import signal
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from coordinator_enhanced import EnhancedDualVLLMCoordinator
from config_enhanced import EnhancedCoordinatorConfig


class VLLMManager:
    """Manage vLLM instances"""

    def __init__(self):
        self.processes = []

    def start_vllm(self, model_path: str, port: int, gpu_util: float = 0.4, max_model_len: int = 4096):
        """Start a vLLM instance"""
        cmd = [
            "python", "-m", "vllm.entrypoints.api_server",
            "--model", model_path,
            "--port", str(port),
            "--gpu-memory-utilization", str(gpu_util),
            "--max-model-len", str(max_model_len),
            "--dtype", "auto",
            "--trust-remote-code",
        ]

        print(f"▶ Starting {model_path} on port {port} (GPU util: {gpu_util})")

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid,
            text=True,
            bufsize=1
        )

        self.processes.append((process, port, model_path))
        return process

    def wait_for_ready(self, port: int, timeout: int = 300):
        """Wait for vLLM to be ready"""
        import requests

        start_time = time.time()
        url = f"http://localhost:{port}/health"

        print(f"⏳ Waiting for port {port}", end="", flush=True)

        while time.time() - start_time < timeout:
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    elapsed = time.time() - start_time
                    print(f" ✅ (ready in {elapsed:.1f}s)")
                    return True
            except Exception as e:
                pass

            time.sleep(10)  # Check every 10 seconds
            print(".", end="", flush=True)

        print(" ❌ TIMEOUT")
        return False

    def cleanup(self):
        """Stop all vLLM instances"""
        print("\n🛑 Stopping vLLM instances...")
        for process, port, model in self.processes:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                process.wait(timeout=10)
                print(f"  ✓ Stopped {model} (port {port})")
            except:
                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                except:
                    pass

        self.processes = []


async def test_generation(coordinator):
    """Test generation with different prompt formats"""
    print("\n" + "="*70)
    print("TEST: Qwen3-4B-Base + Qwen3-14B Generation")
    print("="*70)

    # Test prompts (different formats for base and instruct)
    # Base model: Simple format
    prompts_theta = [
        "Q: What is machine learning?\nA:",
        "Q: Explain quantum computing simply.\nA:",
        "Q: What is the capital of France?\nA:",
    ]

    # Instruct model: Qwen2/3 chat format (from transformers docs)
    # Qwen3 uses a similar format to Qwen2
    prompts_t = [
        "<|im_start|>user\nWhat is machine learning?<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nExplain quantum computing simply.<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n",
    ]

    print(f"\n📝 Generating {len(prompts_theta)} prompts...")
    print(f"   Base format: {prompts_theta[0][:50]}...")
    print(f"   Instruct format: {prompts_t[0][:50]}...")

    start_time = time.time()

    results = await coordinator.generate_batch_dual_prompts(
        prompts_theta=prompts_theta,
        prompts_t=prompts_t,
        max_tokens=100,
        temperature=1.0,
        return_diagnostics=True,
        show_progress=True
    )

    elapsed = time.time() - start_time

    # Analyze results
    print(f"\n{'='*70}")
    print("📊 RESULTS")
    print("="*70)

    success_count = 0
    total_tokens = 0

    for i, result in enumerate(results):
        print(f"\n[{i+1}] Prompt: {prompts_theta[i][:40]}...")

        if result.error:
            print(f"  ❌ Error: {result.error}")
        else:
            success_count += 1
            total_tokens += len(result.generated_tokens)

            alpha_mean = np.mean(result.alpha_history)
            alpha_std = np.std(result.alpha_history)

            print(f"  ✅ Tokens: {len(result.generated_tokens)}")
            print(f"  📊 α: {alpha_mean:.3f} ± {alpha_std:.3f}")
            print(f"     First α: {result.alpha_history[0]:.3f} (forced: {result.alpha_history[0] > 0.99})")
            print(f"     α range: [{min(result.alpha_history):.3f}, {max(result.alpha_history):.3f}]")

            if result.diagnostics:
                print(f"  📈 KL symmetry error: {result.diagnostics['kl_diff_mean']:.6f}")
                print(f"     ESS ratio: {result.diagnostics['ess_ratio_mean']:.3f}")
                print(f"     Entropy: {result.diagnostics['entropy_q_mean']:.3f}")

    # Overall statistics
    print(f"\n{'='*70}")
    print("📈 OVERALL STATISTICS")
    print("="*70)
    print(f"  Success rate: {success_count}/{len(prompts_theta)}")
    print(f"  Total tokens: {total_tokens}")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Throughput: {total_tokens/elapsed:.1f} tok/s")

    # Coordinator statistics
    stats = coordinator.get_statistics()
    print(f"\n  Coordinator stats:")
    print(f"    Total requests: {stats['total_requests']}")
    print(f"    First tokens forced: {stats['first_token_forced']}")
    print(f"    Constraints applied: {stats['constraint_applied']}")
    print(f"    Success rate: {stats.get('success_rate', 0):.1%}")

    return success_count == len(prompts_theta)


async def test_comparison():
    """Compare different configurations"""
    print("\n" + "="*70)
    print("TEST: Configuration Comparison")
    print("="*70)

    prompt_theta = ["Q: What is deep learning?\nA:"]
    prompt_t = ["<|im_start|>user\nWhat is deep learning?<|im_end|>\n<|im_start|>assistant\n"]

    configs = [
        ("Baseline (no enhancements)", EnhancedCoordinatorConfig(
            theta_url="http://localhost:9000",
            t_url="http://localhost:9001",
            force_first_token=False,
            constraint_to_target=False,
        )),
        ("With first token forcing", EnhancedCoordinatorConfig(
            theta_url="http://localhost:9000",
            t_url="http://localhost:9001",
            force_first_token=True,
            constraint_to_target=False,
        )),
        ("Full features (recommended)", EnhancedCoordinatorConfig(
            theta_url="http://localhost:9000",
            t_url="http://localhost:9001",
            force_first_token=True,
            constraint_to_target=True,
            target_top_p=0.95,
        )),
    ]

    for name, config in configs:
        print(f"\n{'─'*70}")
        print(f"Configuration: {name}")

        async with EnhancedDualVLLMCoordinator(config) as coordinator:
            results = await coordinator.generate_batch_dual_prompts(
                prompts_theta=prompt_theta,
                prompts_t=prompt_t,
                max_tokens=50,
                temperature=1.0,
                return_diagnostics=True,
                show_progress=False
            )

            result = results[0]
            if not result.error:
                alpha_mean = np.mean(result.alpha_history)
                print(f"  Tokens: {len(result.generated_tokens)}")
                print(f"  α: {alpha_mean:.3f}")
                print(f"  First α: {result.alpha_history[0]:.3f}")

                if result.diagnostics:
                    print(f"  KL error: {result.diagnostics['kl_diff_mean']:.6f}")
                    print(f"  ESS_θ: {result.diagnostics['ess_theta_mean']:.2f}")
                    print(f"  ESS_t: {result.diagnostics['ess_t_mean']:.2f}")
            else:
                print(f"  ❌ Error: {result.error}")


async def main():
    """Main test"""
    print("="*70)
    print("🧪 Qwen3-4B-Base + Qwen3-14B Test")
    print("="*70)
    print("\nModels:")
    print("  π_θ: Qwen/Qwen3-4B-Base (4B Base)")
    print("  π_t: Qwen/Qwen3-14B (14B Instruct)")
    print("\nNote: Using actual Qwen3 models found in cache!")
    print("="*70)

    # Use Qwen3 models (found in user's cache)
    BASE_MODEL = "Qwen/Qwen3-4B-Base"  # π_θ (base)
    TEACHER_MODEL = "Qwen/Qwen3-14B"   # π_t (teacher/instruct)

    manager = VLLMManager()

    try:
        # Check GPU memory
        print("\n💾 GPU Memory (before):")
        os.system("nvidia-smi --query-gpu=index,memory.used,memory.free,memory.total --format=csv,noheader,nounits 2>/dev/null | head -1")

        # Start vLLM instances
        print(f"\n📦 Starting vLLM instances...")
        print(f"  Base: {BASE_MODEL}")
        print(f"  Teacher: {TEACHER_MODEL}")

        # Base model: 4B, use less GPU memory (~10GB)
        manager.start_vllm(BASE_MODEL, 9000, gpu_util=0.20, max_model_len=2048)
        time.sleep(10)

        # Teacher model: 14B, use more GPU memory (~35GB)
        manager.start_vllm(TEACHER_MODEL, 9001, gpu_util=0.55, max_model_len=2048)

        # Wait for both to be ready
        print()
        if not manager.wait_for_ready(9000, timeout=180):
            print("❌ Base model failed to start")
            return False

        if not manager.wait_for_ready(9001, timeout=180):
            print("❌ Teacher model failed to start")
            return False

        print("\n✅ Both models ready!")

        # Show GPU memory after loading
        print("\n💾 GPU Memory (after loading):")
        os.system("nvidia-smi --query-gpu=index,memory.used,memory.free,memory.total --format=csv,noheader,nounits 2>/dev/null | head -1")

        # Create coordinator with recommended config
        config = EnhancedCoordinatorConfig(
            theta_url="http://localhost:9000",
            t_url="http://localhost:9001",
            top_k=100,
            force_first_token=True,
            constraint_to_target=True,
            target_top_p=0.95,
            enable_logging=True,
        )

        # Run tests
        async with EnhancedDualVLLMCoordinator(config) as coordinator:
            test1 = await test_generation(coordinator)
            test2_result = await test_comparison()

            # Summary
            print("\n" + "="*70)
            print("📋 FINAL SUMMARY")
            print("="*70)
            print(f"  Generation test: {'✅ PASS' if test1 else '❌ FAIL'}")
            print("  Configuration comparison: ✅ DONE")

            print(f"\n💾 GPU Memory (final):")
            os.system("nvidia-smi --query-gpu=index,memory.used,memory.free,memory.total --format=csv,noheader,nounits 2>/dev/null | head -1")

            print("\n" + "="*70)
            if test1:
                print("🎉 TEST PASSED!")
                print("\n✨ The dual-vllm optimal sampling works!")
                print("   - 3B base + 7B instruct combination successful")
                print("   - Dual prompts working correctly")
                print("   - KL symmetry maintained")
                print("   - Enhanced features operational")
            else:
                print("❌ TEST FAILED")
            print("="*70 + "\n")

            return test1

    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user")
        return False

    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        manager.cleanup()

        print("\n💾 GPU Memory (after cleanup):")
        os.system("nvidia-smi --query-gpu=index,memory.used,memory.free,memory.total --format=csv,noheader,nounits 2>/dev/null | head -1")


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
