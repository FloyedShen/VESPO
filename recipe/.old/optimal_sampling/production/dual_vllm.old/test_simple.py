#!/usr/bin/env python3
"""
Simplified integration test for dual_vllm (no pytest dependency)
Tests core functionality only
"""

import os
import sys
import time
import asyncio
import subprocess
import signal
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from coordinator import DualVLLMCoordinator, CoordinatorConfig


class VLLMManager:
    """Manage vLLM instances"""

    def __init__(self):
        self.processes = []

    def start_vllm(self, model_path: str, port: int, gpu_util: float = 0.35):
        """Start a vLLM instance"""
        cmd = [
            "python", "-m", "vllm.entrypoints.api_server",
            "--model", model_path,
            "--port", str(port),
            "--gpu-memory-utilization", str(gpu_util),
            "--max-model-len", "2048",
            "--dtype", "auto",
            "--trust-remote-code",
        ]

        print(f"▶ Starting {model_path} on port {port}")

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            preexec_fn=os.setsid
        )

        self.processes.append((process, port))
        return process

    def wait_for_ready(self, port: int, timeout: int = 180):
        """Wait for vLLM to be ready"""
        import requests

        start_time = time.time()
        url = f"http://localhost:{port}/health"

        print(f"⏳ Waiting for port {port}", end="", flush=True)

        while time.time() - start_time < timeout:
            try:
                response = requests.get(url, timeout=2)
                if response.status_code == 200:
                    print(" ✅")
                    return True
            except:
                pass

            time.sleep(3)
            print(".", end="", flush=True)

        print(" ❌")
        return False

    def cleanup(self):
        """Stop all vLLM instances"""
        print("\n🛑 Stopping vLLM instances...")
        for process, port in self.processes:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                process.wait(timeout=5)
                print(f"  ✓ Stopped port {port}")
            except:
                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                except:
                    pass

        self.processes = []


async def test_basic(coordinator):
    """Test basic generation"""
    print("\n" + "="*70)
    print("TEST 1: Basic Generation (2 prompts)")
    print("="*70)

    prompts = [
        "What is 2+2?",
        "Hello, how are you?",
    ]

    results = await coordinator.generate_batch(
        prompts=prompts,
        max_tokens=30,
        temperature=1.0,
        return_diagnostics=True,
        show_progress=False
    )

    success = 0
    for i, r in enumerate(results):
        print(f"\n[{i+1}] {r.prompt}")
        if r.error:
            print(f"  ❌ {r.error}")
        else:
            success += 1
            print(f"  ✅ Tokens: {len(r.generated_tokens)}, α: {np.mean(r.alpha_history):.3f}")

            if r.diagnostics:
                kl_err = r.diagnostics['kl_diff_mean']
                ess_ratio = r.diagnostics['ess_ratio_mean']
                print(f"  📊 KL error: {kl_err:.6f}, ESS ratio: {ess_ratio:.3f}")

    print(f"\n{'='*70}")
    print(f"Result: {success}/{len(prompts)} passed")
    return success == len(prompts)


async def test_batch(coordinator):
    """Test batch processing"""
    print("\n" + "="*70)
    print("TEST 2: Batch Processing (10 prompts)")
    print("="*70)

    prompts = [f"Count to {i}" for i in range(1, 11)]

    start = time.time()
    results = await coordinator.generate_batch(
        prompts=prompts,
        max_tokens=20,
        temperature=1.0,
        return_diagnostics=False,
        show_progress=False
    )
    elapsed = time.time() - start

    success = sum(1 for r in results if r.error is None)
    total_tokens = sum(len(r.generated_tokens) for r in results if r.error is None)

    print(f"\n📊 Results:")
    print(f"  Success: {success}/{len(prompts)}")
    print(f"  Tokens: {total_tokens}")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Throughput: {total_tokens/elapsed:.1f} tok/s")

    print(f"\n{'='*70}")
    return success == len(prompts)


async def main():
    """Main test"""
    print("\n" + "="*70)
    print("🧪 Dual VLLM Integration Test (H100)")
    print("="*70)

    # Use smaller models that are likely available
    BASE = "Qwen/Qwen2.5-1.5B"
    TEACHER = "Qwen/Qwen2.5-1.5B-Instruct"

    manager = VLLMManager()

    try:
        # Start vLLM instances
        print("\n📦 Starting vLLM instances...")
        print(f"  Base: {BASE}")
        print(f"  Teacher: {TEACHER}")

        manager.start_vllm(BASE, 8000, gpu_util=0.35)
        time.sleep(5)
        manager.start_vllm(TEACHER, 8001, gpu_util=0.35)

        # Wait for ready
        if not manager.wait_for_ready(8000):
            print("❌ Base model failed to start")
            return False

        if not manager.wait_for_ready(8001):
            print("❌ Teacher model failed to start")
            return False

        print("\n✅ Both models ready!")

        # Show GPU memory
        print("\n💾 GPU Memory:")
        os.system("nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader,nounits 2>/dev/null | head -1")

        # Run tests
        config = CoordinatorConfig(
            theta_url="http://localhost:8000",
            t_url="http://localhost:8001",
            top_k=100,
            enable_logging=False,
        )

        async with DualVLLMCoordinator(config) as coord:
            test1 = await test_basic(coord)
            test2 = await test_batch(coord)

            # Summary
            print("\n" + "="*70)
            print("📋 TEST SUMMARY")
            print("="*70)
            print(f"  Basic generation: {'✅ PASS' if test1 else '❌ FAIL'}")
            print(f"  Batch processing: {'✅ PASS' if test2 else '❌ FAIL'}")

            stats = coord.get_statistics()
            print(f"\n📊 Coordinator Stats:")
            print(f"  Requests: {stats['total_requests']}")
            print(f"  Failures: {stats['failed_requests']}")
            print(f"  Tokens: {stats['total_tokens']}")

            all_pass = test1 and test2

            print("\n" + "="*70)
            if all_pass:
                print("🎉 ALL TESTS PASSED!")
            else:
                print("❌ SOME TESTS FAILED")
            print("="*70 + "\n")

            return all_pass

    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted")
        return False

    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        manager.cleanup()


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
