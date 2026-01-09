#!/usr/bin/env python3
"""
直接使用 vLLM Python SDK 测试 Qwen3-4B-Base + Qwen3-14B

不使用 HTTP API，直接调用 vLLM Python 接口
"""

import asyncio
import numpy as np
from typing import List, Dict
from vllm import LLM, SamplingParams

from utils import (
    solve_kl_symmetry,
    compute_q_star,
    merge_top_k_candidates,
    sample_from_distribution,
    compute_diagnostics,
)


def test_direct_vllm():
    """直接使用 vLLM SDK 测试"""
    print("\n" + "="*70)
    print("🧪 Qwen3-4B-Base + Qwen3-14B 直接测试（vLLM SDK）")
    print("="*70)

    # 初始化两个模型
    print("\n📦 加载模型...")
    print("  加载 Base 模型 (4B)...")
    llm_theta = LLM(
        model="Qwen/Qwen3-4B-Base",
        gpu_memory_utilization=0.20,
        max_model_len=2048,
        dtype="auto",
        trust_remote_code=True,
    )

    print("  加载 Teacher 模型 (14B)...")
    llm_t = LLM(
        model="Qwen/Qwen3-14B",
        gpu_memory_utilization=0.55,
        max_model_len=2048,
        dtype="auto",
        trust_remote_code=True,
    )

    print("✅ 模型加载完成！\n")

    # 测试提示
    prompts_theta = [
        "Q: What is machine learning?\nA:",
    ]

    prompts_t = [
        "<|im_start|>user\nWhat is machine learning?<|im_end|>\n<|im_start|>assistant\n",
    ]

    # 生成参数
    sampling_params = SamplingParams(
        temperature=1.0,
        max_tokens=1,
        logprobs=20,  # 获取 top-20 logprobs
    )

    print("📝 测试生成...")
    print(f"   Base 格式: {prompts_theta[0][:40]}...")
    print(f"   Instruct 格式: {prompts_t[0][:40]}...")

    # 生成 50 个 token
    max_tokens = 50
    context_theta = prompts_theta[0]
    context_t = prompts_t[0]

    generated_tokens = []
    alpha_history = []

    for step in range(max_tokens):
        # 从两个模型获取 logprobs
        outputs_theta = llm_theta.generate([context_theta], sampling_params)
        outputs_t = llm_t.generate([context_t], sampling_params)

        # 提取 logprobs
        logprobs_theta_data = outputs_theta[0].outputs[0].logprobs[0]
        logprobs_t_data = outputs_t[0].outputs[0].logprobs[0]

        # 转换为 dict
        logprobs_theta = {token_id: logprob.logprob for token_id, logprob in logprobs_theta_data.items()}
        logprobs_t = {token_id: logprob.logprob for token_id, logprob in logprobs_t_data.items()}

        # 合并 top-k
        candidates, probs_theta, probs_t = merge_top_k_candidates(
            logprobs_theta, logprobs_t
        )

        # 计算 α*
        if step == 0:
            # 首 token 强制
            alpha_star = 1.0
            q_star = probs_t
        else:
            alpha_star = solve_kl_symmetry(probs_theta, probs_t)
            q_star = compute_q_star(probs_theta, probs_t, alpha_star)

        # 采样
        next_token = sample_from_distribution(q_star, candidates)

        generated_tokens.append(next_token)
        alpha_history.append(alpha_star)

        # 解码并更新上下文
        token_str = llm_theta.get_tokenizer().decode([next_token])
        context_theta += token_str
        context_t += token_str

        if step < 5:
            print(f"  Step {step}: token={next_token}, α={alpha_star:.3f}, text='{token_str}'")

    # 结果
    print(f"\n{'='*70}")
    print("📊 结果")
    print("="*70)
    print(f"  生成 tokens: {len(generated_tokens)}")
    print(f"  平均 α: {np.mean(alpha_history):.3f} ± {np.std(alpha_history):.3f}")
    print(f"  首 α: {alpha_history[0]:.3f} (应为 1.0)")
    print(f"  α 范围: [{np.min(alpha_history):.3f}, {np.max(alpha_history):.3f}]")

    print(f"\n  生成文本:")
    print(f"  {context_theta[:200]}...")

    print("\n" + "="*70)
    print("🎉 测试完成！")
    print("="*70)


if __name__ == "__main__":
    test_direct_vllm()
