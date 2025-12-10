#!/usr/bin/env python3
"""
从 vLLM 部署的模型中采样数据作为 SFT baseline (异步并发版本)

功能:
1. 连接到 vLLM 部署的模型（通过 OpenAI-compatible API）
2. 从指定数据集加载 prompts
3. 异步并发批量生成回答（大幅提升速度）
4. 保存为 SFT 格式的 JSONL 文件
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from datasets import load_dataset
from openai import AsyncOpenAI, OpenAI
from tqdm.asyncio import tqdm as atqdm
from tqdm import tqdm


def load_data_from_dataset(
    dataset_path: str,
    split: str = "train",
    num_samples: Optional[int] = None
) -> List[Dict]:
    """
    从 HuggingFace 或本地数据集加载数据

    Args:
        dataset_path: 数据集路径（HuggingFace 或本地路径）
        split: 数据集分割（train/test/validation）
        num_samples: 限制样本数量（None 表示全部）

    Returns:
        List[Dict]: 数据列表
    """
    print(f"Loading dataset from {dataset_path}...")

    try:
        # 尝试从 HuggingFace 加载
        dataset = load_dataset(dataset_path, split=split)
    except Exception as e:
        print(f"Failed to load from HuggingFace: {e}")
        # 尝试从本地加载
        try:
            if dataset_path.endswith('.parquet'):
                df = pd.read_parquet(dataset_path)
                dataset = df.to_dict('records')
            elif dataset_path.endswith('.jsonl') or dataset_path.endswith('.json'):
                with open(dataset_path, 'r') as f:
                    dataset = [json.loads(line) for line in f]
            else:
                raise ValueError(f"Unsupported file format: {dataset_path}")
        except Exception as e2:
            print(f"Failed to load from local: {e2}")
            raise

    # 转换为列表
    if hasattr(dataset, 'to_list'):
        data = dataset.to_list()
    elif isinstance(dataset, list):
        data = dataset
    else:
        data = list(dataset)

    # 限制样本数量
    if num_samples is not None and num_samples < len(data):
        data = data[:num_samples]

    print(f"Loaded {len(data)} samples")
    return data


def extract_prompt_from_item(item: Dict) -> str:
    """
    从数据项中提取 prompt

    Args:
        item: 数据项

    Returns:
        str: 提取的 prompt
    """
    # 优先级：prompt > question > problem > text
    if 'prompt' in item:
        prompt_value = item['prompt']

        # 处理 numpy.ndarray
        if isinstance(prompt_value, np.ndarray):
            prompt_value = prompt_value.tolist()

        # 处理 list of dicts (OpenAI 格式)
        if isinstance(prompt_value, list) and len(prompt_value) > 0:
            first_item = prompt_value[0]
            if isinstance(first_item, dict) and 'content' in first_item:
                return first_item['content']

        # 处理字符串
        if isinstance(prompt_value, str):
            return prompt_value

        # 兜底
        return str(prompt_value)

    elif 'question' in item:
        return item['question']

    elif 'problem' in item:
        return item['problem']

    elif 'text' in item:
        return item['text']

    else:
        # 如果没有明确的 prompt 字段，尝试从 extra_info 中提取
        if 'extra_info' in item:
            extra_info = item['extra_info']
            if isinstance(extra_info, dict):
                if 'original_problem' in extra_info:
                    return f"Question: {extra_info['original_problem']}\n\nAnswer: "

        raise ValueError(f"Cannot extract prompt from item: {item.keys()}")


async def sample_single_prompt(
    client: AsyncOpenAI,
    model_name: str,
    prompt: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    max_retries: int = 3,
    semaphore: Optional[asyncio.Semaphore] = None
) -> str:
    """
    从 vLLM 模型采样单个 prompt (异步)

    Args:
        client: AsyncOpenAI client
        model_name: 模型名称
        prompt: 单个 prompt
        temperature: 采样温度
        top_p: top-p 采样
        max_tokens: 最大生成 token 数
        max_retries: 最大重试次数
        semaphore: 信号量，用于控制并发数

    Returns:
        str: 生成的回答
    """
    async with semaphore if semaphore else asyncio.Semaphore(1):
        for retry in range(max_retries + 1):
            try:
                # 异步调用 vLLM API
                response = await client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens
                )

                # 提取生成的文本
                generated_text = response.choices[0].message.content
                return generated_text

            except Exception as e:
                if retry < max_retries:
                    # 指数退避
                    wait_time = 2 ** retry
                    await asyncio.sleep(wait_time)
                else:
                    # 所有重试都失败
                    print(f"\nError sampling prompt (after {max_retries} retries): {e}")
                    print(f"Prompt: {prompt[:100]}...")
                    return ""  # 返回空字符串


async def sample_from_vllm_async(
    client: AsyncOpenAI,
    model_name: str,
    prompts: List[str],
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 2048,
    max_concurrent: int = 64,
    max_retries: int = 3
) -> List[str]:
    """
    从 vLLM 模型批量采样 (异步并发)

    Args:
        client: AsyncOpenAI client
        model_name: 模型名称
        prompts: prompt 列表
        temperature: 采样温度
        top_p: top-p 采样
        max_tokens: 最大生成 token 数
        max_concurrent: 最大并发数
        max_retries: 最大重试次数

    Returns:
        List[str]: 生成的回答列表
    """
    # 创建信号量来限制并发数
    semaphore = asyncio.Semaphore(max_concurrent)

    # 创建所有任务
    tasks = [
        sample_single_prompt(
            client=client,
            model_name=model_name,
            prompt=prompt,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            max_retries=max_retries,
            semaphore=semaphore
        )
        for prompt in prompts
    ]

    # 并发执行所有任务，显示进度条
    print(f"Starting {len(tasks)} concurrent sampling tasks...")

    # IMPORTANT: Use asyncio.gather to preserve order!
    # asyncio.as_completed returns results in completion order, not submission order
    # This causes prompts and responses to be mismatched!
    responses = await atqdm.gather(*tasks, desc="Sampling")

    return responses


def sample_from_vllm(
    base_url: str,
    api_key: str,
    model_name: str,
    prompts: List[str],
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 2048,
    max_concurrent: int = 64,
    max_retries: int = 3
) -> List[str]:
    """
    从 vLLM 模型批量采样 (同步接口)

    Args:
        base_url: vLLM base URL
        api_key: API key
        model_name: 模型名称
        prompts: prompt 列表
        temperature: 采样温度
        top_p: top-p 采样
        max_tokens: 最大生成 token 数
        max_concurrent: 最大并发数
        max_retries: 最大重试次数

    Returns:
        List[str]: 生成的回答列表
    """
    # 创建异步客户端
    client = AsyncOpenAI(base_url=base_url, api_key=api_key)

    # 运行异步函数
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    responses = loop.run_until_complete(
        sample_from_vllm_async(
            client=client,
            model_name=model_name,
            prompts=prompts,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            max_concurrent=max_concurrent,
            max_retries=max_retries
        )
    )

    return responses


def create_sft_data(prompts: List[str], responses: List[str]) -> List[Dict]:
    """
    创建 SFT 格式的数据

    Args:
        prompts: prompt 列表
        responses: 回答列表

    Returns:
        List[Dict]: SFT 格式的数据
    """
    sft_data = []
    for prompt, response in zip(prompts, responses):
        sft_data.append({
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response}
            ]
        })
    return sft_data


def save_sft_data(sft_data: List[Dict], output_path: str):
    """
    保存 SFT 数据为 JSONL 格式

    Args:
        sft_data: SFT 数据
        output_path: 输出路径
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        for item in sft_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"Saved {len(sft_data)} samples to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Sample from vLLM deployed model for SFT baseline (Async Concurrent)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 从 HuggingFace 数据集采样
  python sample_from_vllm.py \\
      --base_url http://localhost:8000/v1 \\
      --model_name meta-llama/Llama-2-7b-hf \\
      --dataset agentica-org/DeepScaleR-Preview-Dataset \\
      --output baseline_samples.jsonl \\
      --num_samples 1000

  # 从本地 parquet 文件采样
  python sample_from_vllm.py \\
      --base_url http://localhost:8000/v1 \\
      --model_name Qwen2.5-Math-7B \\
      --dataset production/data/deepscaler_q14b_q4bb_new/kl_symmetry_filtered.parquet \\
      --output baseline_qwen_samples.jsonl

  # 调整采样参数和并发数（高并发）
  python sample_from_vllm.py \\
      --base_url http://localhost:8000/v1 \\
      --model_name meta-llama/Llama-2-7b-hf \\
      --dataset dataset.jsonl \\
      --output samples.jsonl \\
      --temperature 0.8 \\
      --top_p 0.95 \\
      --max_tokens 4096 \\
      --max_concurrent 128
        """
    )

    parser.add_argument(
        '--base_url',
        type=str,
        required=True,
        help='vLLM 服务的 base URL（例如：http://localhost:8000/v1）'
    )

    parser.add_argument(
        '--model_name',
        type=str,
        required=True,
        help='模型名称（例如：meta-llama/Llama-2-7b-hf）'
    )

    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='数据集路径（HuggingFace 或本地文件）'
    )

    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='输出 JSONL 文件路径'
    )

    parser.add_argument(
        '--split',
        type=str,
        default='train',
        help='数据集分割（默认: train）'
    )

    parser.add_argument(
        '--num_samples',
        type=int,
        default=None,
        help='限制采样数量（默认: 全部）'
    )

    parser.add_argument(
        '--temperature',
        type=float,
        default=0.7,
        help='采样温度（默认: 0.7）'
    )

    parser.add_argument(
        '--top_p',
        type=float,
        default=0.9,
        help='top-p 采样（默认: 0.9）'
    )

    parser.add_argument(
        '--max_tokens',
        type=int,
        default=8192,
        help='最大生成 token 数（默认: 8192）'
    )

    parser.add_argument(
        '--max_concurrent',
        type=int,
        default=64,
        help='最大并发请求数（默认: 64，可根据服务器性能调整）'
    )

    parser.add_argument(
        '--api_key',
        type=str,
        default='EMPTY',
        help='API key（默认: EMPTY，vLLM 通常不需要）'
    )

    parser.add_argument(
        '--save_stats',
        action='store_true',
        help='保存统计信息'
    )

    args = parser.parse_args()

    # 打印配置
    print("="*60)
    print("vLLM Async Sampling Configuration")
    print("="*60)
    print(f"Base URL:     {args.base_url}")
    print(f"Model:        {args.model_name}")
    print(f"Dataset:      {args.dataset}")
    print(f"Output:       {args.output}")
    print(f"Num samples:  {args.num_samples or 'All'}")
    print(f"Temperature:  {args.temperature}")
    print(f"Top-p:        {args.top_p}")
    print(f"Max tokens:   {args.max_tokens}")
    print(f"Concurrent:   {args.max_concurrent}")
    print("="*60)
    print()

    # 测试连接（使用同步客户端）
    print("Connecting to vLLM server...")
    try:
        test_client = OpenAI(base_url=args.base_url, api_key=args.api_key)
        models = test_client.models.list()
        print(f"Connected! Available models: {[m.id for m in models.data]}")
    except Exception as e:
        print(f"Error connecting to vLLM server: {e}")
        print("Please check if the server is running and the base_url is correct.")
        sys.exit(1)

    print()

    # 加载数据集
    data = load_data_from_dataset(
        args.dataset,
        split=args.split,
        num_samples=args.num_samples
    )[37200:]
    # 0:3100 0
    # 3100:6200 1
    # 6200:9300 2
    # 9300:12400 3
    # 12400:15500 4
    # 15500:18600 5
    # 18600:21700 6
    # 21700:24800 7
    # 24800:27900 8
    # 27900:31000 9
    # 31000:34100 10
    # 34100:37200 11
    # 37200:  12

    # 提取 prompts
    print("\nExtracting prompts...")
    prompts = []
    failed_count = 0
    for i, item in enumerate(data):
        try:
            prompt = extract_prompt_from_item(item)
            prompts.append(prompt)
        except Exception as e:
            print(f"Warning: Failed to extract prompt from item {i}: {e}")
            failed_count += 1

    print(f"Extracted {len(prompts)} prompts ({failed_count} failed)")

    if len(prompts) == 0:
        print("Error: No prompts extracted!")
        sys.exit(1)

    print()

    # 从 vLLM 采样（异步并发）
    print(f"Sampling from vLLM model: {args.model_name}")
    print(f"Using {args.max_concurrent} concurrent requests...")
    start_time = time.time()

    responses = sample_from_vllm(
        base_url=args.base_url,
        api_key=args.api_key,
        model_name=args.model_name,
        prompts=prompts,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        max_concurrent=args.max_concurrent
    )

    elapsed_time = time.time() - start_time
    throughput = len(prompts) / elapsed_time if elapsed_time > 0 else 0

    print(f"\nSampling completed in {elapsed_time:.2f}s")
    print(f"Average time per sample: {elapsed_time/len(prompts):.3f}s")
    print(f"Throughput: {throughput:.2f} samples/sec")

    # 创建 SFT 数据
    print("\nCreating SFT format data...")
    sft_data = create_sft_data(prompts, responses)

    # 统计非空回答
    non_empty_responses = sum(1 for r in responses if r.strip())
    print(f"Non-empty responses: {non_empty_responses}/{len(responses)} ({non_empty_responses/len(responses)*100:.2f}%)")

    # 保存数据
    print("\nSaving data...")
    save_sft_data(sft_data, args.output)

    # 保存统计信息
    if args.save_stats:
        stats_path = Path(args.output).with_suffix('.stats.json')
        stats = {
            "base_url": args.base_url,
            "model_name": args.model_name,
            "dataset": args.dataset,
            "total_samples": len(prompts),
            "non_empty_responses": non_empty_responses,
            "success_rate": non_empty_responses / len(responses) if len(responses) > 0 else 0,
            "elapsed_time_seconds": elapsed_time,
            "avg_time_per_sample": elapsed_time / len(prompts) if len(prompts) > 0 else 0,
            "throughput_samples_per_sec": throughput,
            "sampling_params": {
                "temperature": args.temperature,
                "top_p": args.top_p,
                "max_tokens": args.max_tokens,
                "max_concurrent": args.max_concurrent
            }
        }

        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)

        print(f"Stats saved to {stats_path}")

    print()
    print("="*60)
    print("Sampling Complete!")
    print("="*60)
    print(f"Output file: {args.output}")
    print(f"Total samples: {len(sft_data)}")
    print(f"Success rate: {non_empty_responses/len(responses)*100:.2f}%")
    print(f"Throughput: {throughput:.2f} samples/sec")
    print()
    print("Next steps:")
    print(f"  1. Check the output: head {args.output}")
    print(f"  2. Use for SFT training as baseline")
    print("="*60)


if __name__ == "__main__":
    main()

# python sample_from_vllm.py --dataset agentica-org/DeepScaleR-Preview-Dataset  --base_url  http://localhost:8123/v1  --model_name Qwen/Qwen3-14B  --output data/deepscaler_q14b.jsonl  --max_concurrent 128