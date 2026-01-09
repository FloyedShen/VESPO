#!/usr/bin/env python3
"""
快速测试数据生成管线

测试 generate_data_vllm.py 的基本功能
"""

import subprocess
import sys

def main():
    print("\n" + "="*80)
    print("🧪 测试 generate_data_vllm.py")
    print("="*80)

    # 检查vLLM服务器是否运行
    print("\n1️⃣  检查 vLLM 服务器...")
    import requests
    try:
        r1 = requests.get("http://localhost:9000/health", timeout=2)
        r2 = requests.get("http://localhost:9001/health", timeout=2)
        print("✅ 两个 vLLM 服务器都在运行")
    except Exception as e:
        print(f"❌ vLLM 服务器未运行: {e}")
        print("\n请先启动服务器:")
        print("  python -m vllm.entrypoints.openai.api_server \\")
        print("      --model Qwen/Qwen3-4B-Base --port 9000 --max-logprobs 20")
        print("\n  python -m vllm.entrypoints.openai.api_server \\")
        print("      --model Qwen/Qwen3-14B --port 9001 --max-logprobs 20")
        return

    # 运行数据生成（小样本）
    print("\n2️⃣  运行数据生成（10个样本）...")
    cmd = [
        "python", "generate_data_vllm.py",
        "--theta_url", "http://localhost:9000",
        "--t_url", "http://localhost:9001",
        "--dataset", "agentica-org/DeepScaleR-Preview-Dataset",
        "--output", "test_output.jsonl",
        "--num_samples", "10",
        "--max_tokens", "100",
        "--batch_size", "5",
        "--enable_stability_check",
        "--save_diagnostics",
        "--verbose"
    ]

    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print("\n✅ 数据生成成功!")

        # 检查输出文件
        import os
        if os.path.exists("test_output.jsonl"):
            with open("test_output.jsonl") as f:
                lines = f.readlines()
            print(f"\n3️⃣  输出文件: test_output.jsonl")
            print(f"   - 生成了 {len(lines)} 条数据")

            if os.path.exists("test_output.diagnostics.jsonl"):
                with open("test_output.diagnostics.jsonl") as f:
                    diag_lines = f.readlines()
                print(f"   - 诊断信息: test_output.diagnostics.jsonl ({len(diag_lines)} 条)")

            # 显示第一条
            if lines:
                import json
                first_data = json.loads(lines[0])
                print(f"\n4️⃣  第一条数据预览:")
                user_msg = first_data["messages"][0]["content"]
                assistant_msg = first_data["messages"][1]["content"]
                print(f"   User: {user_msg[:100]}...")
                print(f"   Assistant: {assistant_msg[:100]}...")

                if "diagnostics" in first_data:
                    diag = first_data["diagnostics"]
                    print(f"   Alpha: {diag['alpha_mean']:.3f} ± {diag['alpha_std']:.3f}")

            print("\n" + "="*80)
            print("🎉 测试完成！")
            print("="*80)

    except subprocess.CalledProcessError as e:
        print(f"\n❌ 数据生成失败: {e}")
        return


if __name__ == "__main__":
    main()
