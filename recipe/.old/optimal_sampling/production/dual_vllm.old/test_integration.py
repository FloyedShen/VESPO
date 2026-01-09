#!/usr/bin/env python3
"""
Test script for dual_vllm on single H100 GPU

Uses small models to fit both on one GPU:
- Base: Qwen/Qwen2.5-1.5B
- Teacher: Qwen/Qwen2.5-1.5B-Instruct

Expected memory: ~8GB total
"""

import os
import sys
import time
import asyncio
import subprocess
import signal
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from coordinator import DualVLLMCoordinator, CoordinatorConfig


class VLLMManager:
    """Manage vLLM instances"""

    def __init__(self):
        self.processes = []

    def start_vllm(
        self,
        model_path: str,
        port: int,
        gpu_memory_utilization: float = 0.4,
        max_model_len: int = 2048
    ):
        """Start a vLLM instance"""
        cmd = [
            "python", "-m", "vllm.entrypoints.api_server",
            "--model", model_path,
            "--port", str(port),
            "--gpu-memory-utilization", str(gpu_memory_utilization),
            "--max-model-len", str(max_model_len),
            "--dtype", "auto",
            "--trust-remote-code",
        ]

        print(f"Starting vLLM: {model_path} on port {port}")
        print(f"  GPU memory utilization: {gpu_memory_utilization}")
        print(f"  Max model len: {max_model_len}")

        # Redirect output to log files
        log_file = open(f"vllm_{port}.log", "w")
        process = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid  # Create new process group
        )

        self.processes.append((process, log_file, port))
        return process

    def wait_for_ready(self, port: int, timeout: int = 300):
        """Wait for vLLM to be ready"""
        import requests

        start_time = time.time()
        url = f"http://localhost:{port}/health"

        print(f"Waiting for vLLM on port {port} to be ready...", end="", flush=True)

        while time.time() - start_time < timeout:
            try:
                response = requests.get(url, timeout=2)
                if response.status_code == 200:
                    print(" ✅")
                    return True
            except:
                pass

            time.sleep(2)
            print(".", end="", flush=True)

        print(" ❌ Timeout")
        return False

    def cleanup(self):
        """Stop all vLLM instances"""
        print("\nStopping vLLM instances...")
        for process, log_file, port in self.processes:
            try:
                # Kill entire process group
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                process.wait(timeout=10)
                print(f"  ✓ Stopped vLLM on port {port}")
            except:
                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                except:
                    pass
            finally:
                log_file.close()

        self.processes = []


async def run_basic_test(coordinator):
    """Run basic generation test"""
    print("\n" + "="*80)
    print("Test 1: Basic Generation")
    print("="*80)

    prompts = [
        "What is 2+2?",
        "Write a haiku about AI.",
    ]

    results = await coordinator.generate_batch(
        prompts=prompts,
        max_tokens=50,
        temperature=1.0,
        return_diagnostics=True,
        show_progress=True
    )

    # Check results
    success_count = 0
    for i, result in enumerate(results):
        print(f"\n{'='*60}")
        print(f"Prompt {i+1}: {result.prompt}")
        print(f"{'='*60}")

        if result.error:
            print(f"❌ Error: {result.error}")
        else:
            success_count += 1
            print(f"✅ Generated: {result.generated_text[:100]}...")
            print(f"\nStatistics:")
            print(f"  Tokens: {len(result.generated_tokens)}")
            print(f"  Alpha mean: {np.mean(result.alpha_history):.3f}")
            print(f"  Alpha std: {np.std(result.alpha_history):.3f}")

            if result.diagnostics:
                diag = result.diagnostics
                print(f"\nDiagnostics:")
                print(f"  KL symmetry error: {diag['kl_diff_mean']:.6f}")
                print(f"  ESS ratio: {diag['ess_ratio_mean']:.3f}")

                # Validation
                if diag['kl_diff_mean'] < 1e-4:
                    print(f"  ✅ KL symmetry: GOOD")
                else:
                    print(f"  ⚠️  KL symmetry: ERROR TOO LARGE")

                if 0.9 <= diag['ess_ratio_mean'] <= 1.1:
                    print(f"  ✅ ESS balance: GOOD")
                else:
                    print(f"  ⚠️  ESS balance: UNBALANCED")

    return success_count == len(prompts)


async def run_unit_tests():
    """Run unit tests"""
    print("\n" + "="*80)
    print("Test 2: Unit Tests")
    print("="*80)

    import subprocess
    result = subprocess.run(
        ["pytest", "test_dual_vllm.py", "-v", "--tb=short"],
        cwd=os.path.dirname(os.path.abspath(__file__)),
        capture_output=True,
        text=True
    )

    print(result.stdout)
    if result.stderr:
        print(result.stderr)

    return result.returncode == 0


async def run_stress_test(coordinator):
    """Run stress test with multiple prompts"""
    print("\n" + "="*80)
    print("Test 3: Stress Test (20 prompts)")
    print("="*80)

    prompts = [f"Question {i}: What is {i} squared?" for i in range(1, 21)]

    start_time = time.time()
    results = await coordinator.generate_batch(
        prompts=prompts,
        max_tokens=20,
        temperature=1.0,
        return_diagnostics=False,  # Faster without diagnostics
        show_progress=True
    )
    elapsed = time.time() - start_time

    success_count = sum(1 for r in results if r.error is None)
    total_tokens = sum(len(r.generated_tokens) for r in results if r.error is None)

    print(f"\nResults:")
    print(f"  Success: {success_count}/{len(prompts)}")
    print(f"  Total tokens: {total_tokens}")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Throughput: {total_tokens/elapsed:.2f} tokens/sec")

    return success_count == len(prompts)


async def main():
    """Main test function"""
    print("="*80)
    print("Dual VLLM Test on H100")
    print("="*80)

    # Configuration
    BASE_MODEL = "Qwen/Qwen2.5-1.5B"
    TEACHER_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
    THETA_PORT = 8000
    T_PORT = 8001

    manager = VLLMManager()

    try:
        # Step 1: Start vLLM instances
        print("\n" + "="*80)
        print("Step 1: Starting vLLM Instances")
        print("="*80)

        # Start both on same GPU with memory limits
        manager.start_vllm(
            model_path=BASE_MODEL,
            port=THETA_PORT,
            gpu_memory_utilization=0.35,  # 35% for base
            max_model_len=2048
        )

        time.sleep(5)  # Small delay between starts

        manager.start_vllm(
            model_path=TEACHER_MODEL,
            port=T_PORT,
            gpu_memory_utilization=0.35,  # 35% for teacher
            max_model_len=2048
        )

        # Wait for both to be ready
        print("\n" + "="*80)
        print("Step 2: Waiting for vLLM instances to be ready")
        print("="*80)

        if not manager.wait_for_ready(THETA_PORT):
            print(f"❌ Failed to start base model on port {THETA_PORT}")
            print(f"Check log: vllm_{THETA_PORT}.log")
            return False

        if not manager.wait_for_ready(T_PORT):
            print(f"❌ Failed to start teacher model on port {T_PORT}")
            print(f"Check log: vllm_{T_PORT}.log")
            return False

        print("\n✅ Both vLLM instances are ready!")

        # Check GPU memory
        print("\n" + "="*80)
        print("GPU Memory Status")
        print("="*80)
        os.system("nvidia-smi --query-gpu=memory.used,memory.free --format=csv")

        # Step 3: Run tests
        print("\n" + "="*80)
        print("Step 3: Running Tests")
        print("="*80)

        # Create coordinator
        config = CoordinatorConfig(
            theta_url=f"http://localhost:{THETA_PORT}",
            t_url=f"http://localhost:{T_PORT}",
            top_k=100,
            enable_logging=True,
        )

        async with DualVLLMCoordinator(config) as coordinator:
            # Test 1: Basic generation
            test1_pass = await run_basic_test(coordinator)

            # Test 2: Unit tests (run separately, doesn't need coordinator)
            # Skipped in this run to avoid conflicts

            # Test 3: Stress test
            test3_pass = await run_stress_test(coordinator)

            # Print summary
            print("\n" + "="*80)
            print("Test Summary")
            print("="*80)
            print(f"  Basic test: {'✅ PASS' if test1_pass else '❌ FAIL'}")
            print(f"  Stress test: {'✅ PASS' if test3_pass else '❌ FAIL'}")

            stats = coordinator.get_statistics()
            print(f"\nCoordinator Statistics:")
            print(f"  Total requests: {stats['total_requests']}")
            print(f"  Failed requests: {stats['failed_requests']}")
            print(f"  Total tokens: {stats['total_tokens']}")

            all_pass = test1_pass and test3_pass

            print("\n" + "="*80)
            if all_pass:
                print("🎉 All Tests PASSED!")
            else:
                print("❌ Some Tests FAILED")
            print("="*80)

            return all_pass

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        return False

    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        # Always cleanup
        manager.cleanup()

        print("\nLog files:")
        print(f"  vllm_{THETA_PORT}.log")
        print(f"  vllm_{T_PORT}.log")


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
