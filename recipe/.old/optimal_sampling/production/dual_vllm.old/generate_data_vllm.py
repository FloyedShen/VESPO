#!/usr/bin/env python3
"""
基于 vLLM 的数据生成管线

使用 EnhancedDualVLLMCoordinator 通过 HTTP API 调用 vLLM 服务器生成数据

特性:
- ✅ 使用 vLLM HTTP API（无需加载模型到本地）
- ✅ 支持 HuggingFace datasets
- ✅ 双提示支持（不同chat template）
- ✅ 稳定性检测（可选）
- ✅ Trust region约束（可选）
- ✅ 批处理 + 异步并发
- ✅ 断点续传
- ✅ 多种输出格式（JSONL/Parquet）

使用示例:

    # 启动两个 vLLM 服务器（不同终端）
    python -m vllm.entrypoints.openai.api_server \\
        --model Qwen/Qwen3-4B-Base --port 9000 --max-logprobs 20

    python -m vllm.entrypoints.openai.api_server \\
        --model Qwen/Qwen3-14B --port 9001 --max-logprobs 20

    # 生成数据
    python generate_data_vllm.py \\
        --theta_url http://localhost:9000 \\
        --t_url http://localhost:9001 \\
        --dataset agentica-org/DeepScaleR-Preview-Dataset \\
        --output generated_data.jsonl \\
        --num_samples 1000 \\
        --max_tokens 512 \\
        --batch_size 16 \\
        --enable_stability_check \\
        --save_diagnostics
"""

import asyncio
import argparse
import json
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm
import numpy as np
from datasets import load_dataset, Dataset
import time

from coordinator_enhanced import EnhancedDualVLLMCoordinator
from config_enhanced import EnhancedCoordinatorConfig


class DatasetAdapter:
    """数据集适配器 - 自动检测格式并提取prompts"""

    def __init__(self, dataset_path: str, split: str = "train"):
        """
        Args:
            dataset_path: HuggingFace dataset name 或本地路径
            split: 数据集分割
        """
        print(f"📦 加载数据集: {dataset_path}")
        try:
            self.dataset = load_dataset(dataset_path, split=split)
            print(f"✅ 成功加载: {len(self.dataset)} 条数据")
        except Exception as e:
            raise ValueError(f"Failed to load dataset: {e}")

        # 检测格式
        self._detect_format()

    def _detect_format(self):
        """自动检测数据集格式"""
        sample = self.dataset[0]
        self.columns = list(sample.keys())

        # 检测prompt/question字段
        prompt_candidates = ["prompt", "question", "instruction", "input", "text", "query"]
        self.prompt_field = None
        for candidate in prompt_candidates:
            if candidate in self.columns:
                self.prompt_field = candidate
                break

        # 检测messages字段（OpenAI格式）
        self.messages_field = "messages" if "messages" in self.columns else None

        print(f"📋 数据集格式:")
        print(f"   - Columns: {self.columns}")
        print(f"   - Prompt field: {self.prompt_field}")
        print(f"   - Messages field: {self.messages_field}")

    def __len__(self):
        return len(self.dataset)

    def get_prompt(self, idx: int) -> str:
        """提取纯文本 prompt"""
        sample = self.dataset[idx]

        # 优先使用messages格式
        if self.messages_field and self.messages_field in sample:
            messages = sample[self.messages_field]
            if isinstance(messages, list):
                # 提取最后一个user消息
                for msg in reversed(messages):
                    if isinstance(msg, dict) and msg.get("role") == "user":
                        return msg.get("content", "")

        # 使用prompt字段
        if self.prompt_field and self.prompt_field in sample:
            content = sample[self.prompt_field]
            if isinstance(content, str):
                return content

        # Fallback: 返回第一个字符串字段
        for value in sample.values():
            if isinstance(value, str) and len(value) > 0:
                return value

        raise ValueError(f"Cannot extract prompt from sample {idx}")

    def get_messages(self, idx: int) -> List[Dict[str, str]]:
        """提取messages格式（OpenAI）"""
        sample = self.dataset[idx]

        # 如果已有messages格式
        if self.messages_field and self.messages_field in sample:
            messages = sample[self.messages_field]
            if isinstance(messages, list):
                return messages

        # 否则从prompt构造
        prompt = self.get_prompt(idx)
        return [{"role": "user", "content": prompt}]

    def get_metadata(self, idx: int) -> Dict:
        """获取元数据"""
        sample = self.dataset[idx]
        metadata = {"sample_idx": idx}

        # 保存原始数据的其他字段
        for key, value in sample.items():
            if key != self.prompt_field and key != self.messages_field:
                # 只保存简单类型
                if isinstance(value, (str, int, float, bool)):
                    metadata[f"original_{key}"] = value

        return metadata


def create_dual_prompts(
    messages_list: List[List[Dict[str, str]]],
    use_base_template: bool = True,
    use_instruct_template: bool = True
) -> tuple[List[str], List[str]]:
    """
    创建双提示：Base和Instruct格式

    Args:
        messages_list: OpenAI格式的messages列表
        use_base_template: 是否为base模型使用简单模板
        use_instruct_template: 是否为instruct模型使用chat template

    Returns:
        (prompts_theta, prompts_t)
    """
    prompts_theta = []
    prompts_t = []

    for messages in messages_list:
        # 提取user消息
        user_content = ""
        for msg in messages:
            if msg.get("role") == "user":
                user_content = msg.get("content", "")
                break

        # Base格式（简单）
        if use_base_template:
            prompt_theta = f"Question: {user_content}\n\nAnswer: "
        else:
            prompt_theta = user_content

        # Instruct格式（Qwen/ChatML）
        if use_instruct_template:
            prompt_t = f"<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n"
        else:
            prompt_t = user_content

        prompts_theta.append(prompt_theta)
        prompts_t.append(prompt_t)

    return prompts_theta, prompts_t


class CheckpointManager:
    """管理断点续传"""

    def __init__(self, output_path: str):
        self.output_path = Path(output_path)
        self.checkpoint_file = self.output_path.parent / f"{self.output_path.stem}.checkpoint"

    def load(self) -> Optional[Dict]:
        """加载checkpoint"""
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️  Warning: Failed to load checkpoint: {e}")
        return None

    def save(self, checkpoint: Dict):
        """保存checkpoint"""
        try:
            self.checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.checkpoint_file, 'w') as f:
                json.dump(checkpoint, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"⚠️  Warning: Failed to save checkpoint: {e}")

    def remove(self):
        """删除checkpoint"""
        if self.checkpoint_file.exists():
            try:
                self.checkpoint_file.unlink()
            except Exception as e:
                print(f"⚠️  Warning: Failed to remove checkpoint: {e}")


async def generate_data(args):
    """主数据生成函数"""
    print("\n" + "="*80)
    print("🚀 vLLM Optimal Sampling 数据生成管线")
    print("="*80)

    # 加载数据集
    adapter = DatasetAdapter(args.dataset, args.dataset_split)
    total_samples = len(adapter)

    # 确定处理范围
    if args.num_samples is None:
        args.num_samples = total_samples - args.start_idx
    else:
        args.num_samples = min(args.num_samples, total_samples - args.start_idx)

    end_idx = args.start_idx + args.num_samples

    print(f"\n📊 处理范围:")
    print(f"   - Total samples in dataset: {total_samples}")
    print(f"   - Processing: {args.start_idx} → {end_idx} ({args.num_samples} samples)")

    # Checkpoint管理
    checkpoint_mgr = CheckpointManager(args.output)
    checkpoint = checkpoint_mgr.load()

    start_from = args.start_idx
    if checkpoint:
        last_processed = checkpoint.get("last_processed_idx", args.start_idx - 1)
        if last_processed >= end_idx - 1:
            print(f"\n✅ Already completed! (checkpoint shows idx={last_processed})")
            return

        start_from = last_processed + 1
        print(f"\n🔄 Resuming from checkpoint: idx={start_from}")

    # 配置Coordinator
    config = EnhancedCoordinatorConfig(
        theta_url=args.theta_url,
        t_url=args.t_url,
        theta_model_name=args.theta_model,
        t_model_name=args.t_model,
        top_k=args.top_k,
        force_first_token=args.force_first_token,
        constraint_to_target=args.constraint_to_target,
        target_top_p=args.target_top_p,
        enable_stability_check=args.enable_stability_check,
        stability_threshold_js=args.stability_threshold_js,
        stability_threshold_overlap=args.stability_threshold_overlap,
        auto_fallback=args.auto_fallback,
        enable_logging=args.verbose,
    )

    print(f"\n⚙️  配置:")
    print(f"   - θ URL: {config.theta_url}")
    print(f"   - t URL: {config.t_url}")
    print(f"   - Top-k: {config.top_k}")
    print(f"   - Force first token: {config.force_first_token}")
    print(f"   - Trust region: {config.constraint_to_target} (p={config.target_top_p})")
    print(f"   - Stability check: {config.enable_stability_check}")
    if config.enable_stability_check:
        print(f"      JS threshold: {config.stability_threshold_js}")
        print(f"      Overlap threshold: {config.stability_threshold_overlap}")

    # 输出文件
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 打开输出文件（追加模式）
    mode = 'a' if checkpoint else 'w'
    f_out = open(output_path, mode)

    diag_file = None
    if args.save_diagnostics:
        diag_file = open(output_path.with_suffix('.diagnostics.jsonl'), mode)

    # 启动Coordinator
    print(f"\n🔗 连接到 vLLM 服务器...")
    async with EnhancedDualVLLMCoordinator(config) as coordinator:
        print(f"✅ 连接成功!")

        # 批处理生成
        processed_count = start_from - args.start_idx

        for batch_start in tqdm(
            range(start_from, end_idx, args.batch_size),
            desc="Generating",
            initial=processed_count // args.batch_size
        ):
            batch_end = min(batch_start + args.batch_size, end_idx)
            batch_indices = range(batch_start, batch_end)

            # 提取messages
            messages_list = []
            metadata_list = []
            for idx in batch_indices:
                try:
                    messages = adapter.get_messages(idx)
                    metadata = adapter.get_metadata(idx)
                    messages_list.append(messages)
                    metadata_list.append(metadata)
                except Exception as e:
                    print(f"\n⚠️  Warning: Failed to extract sample {idx}: {e}")
                    continue

            if not messages_list:
                continue

            # 创建双提示
            prompts_theta, prompts_t = create_dual_prompts(
                messages_list,
                use_base_template=args.use_base_template,
                use_instruct_template=args.use_instruct_template
            )

            # 生成
            try:
                results = await coordinator.generate_batch_dual_prompts(
                    prompts_theta=prompts_theta,
                    prompts_t=prompts_t,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    return_diagnostics=args.save_diagnostics,
                    show_progress=False
                )

                # 保存结果
                for i, result in enumerate(results):
                    if result.error:
                        print(f"\n⚠️  Error in sample {batch_start + i}: {result.error}")
                        continue

                    # 构造输出数据（OpenAI格式）
                    output_data = {
                        "messages": [
                            {"role": "user", "content": adapter.get_prompt(batch_start + i)},
                            {"role": "assistant", "content": result.generated_text[len(result.prompt):]}
                        ],
                        "metadata": metadata_list[i]
                    }

                    # 添加alpha等诊断信息
                    if args.save_diagnostics:
                        output_data["diagnostics"] = {
                            "alpha_mean": float(np.mean(result.alpha_history)),
                            "alpha_std": float(np.std(result.alpha_history)),
                            "alpha_first": float(result.alpha_history[0]) if result.alpha_history else None,
                        }

                        # 写入诊断文件
                        diag_data = {
                            "sample_idx": batch_start + i,
                            **output_data["diagnostics"]
                        }
                        diag_file.write(json.dumps(diag_data, ensure_ascii=False) + '\n')
                        diag_file.flush()

                    # 写入主输出
                    f_out.write(json.dumps(output_data, ensure_ascii=False) + '\n')
                    f_out.flush()

                    processed_count += 1

                # 保存checkpoint
                checkpoint_mgr.save({
                    "last_processed_idx": batch_end - 1,
                    "processed_count": processed_count,
                    "timestamp": time.time()
                })

            except Exception as e:
                print(f"\n❌ Error during generation for batch {batch_start}-{batch_end}: {e}")
                import traceback
                traceback.print_exc()
                continue

    # 关闭文件
    f_out.close()
    if diag_file:
        diag_file.close()

    # 删除checkpoint
    checkpoint_mgr.remove()

    # 统计信息
    stats = coordinator.get_statistics()
    print(f"\n{'='*80}")
    print("📈 统计信息")
    print("="*80)
    print(f"总样本数: {processed_count}")
    print(f"成功率: {stats.get('success_rate', 0):.1%}")
    print(f"总Tokens: {stats['total_tokens']}")
    print(f"首Token强制: {stats['first_token_forced']} 次")
    if config.enable_stability_check:
        print(f"稳定性检查: {stats['stability_checks']} 次")
        print(f"稳定性Fallback: {stats['stability_fallback']} 次 ({stats['stability_fallback']/max(stats['stability_checks'],1)*100:.1f}%)")

    print(f"\n✅ 完成！输出: {output_path}")
    if args.save_diagnostics:
        print(f"📊 诊断信息: {output_path.with_suffix('.diagnostics.jsonl')}")


def main():
    parser = argparse.ArgumentParser(description="vLLM数据生成管线")

    # vLLM服务器
    parser.add_argument("--theta_url", type=str, required=True,
                       help="π_θ (base) vLLM服务器URL")
    parser.add_argument("--t_url", type=str, required=True,
                       help="π_t (teacher) vLLM服务器URL")
    parser.add_argument("--theta_model", type=str, default="Qwen/Qwen3-4B-Base",
                       help="Base模型名称")
    parser.add_argument("--t_model", type=str, default="Qwen/Qwen3-14B",
                       help="Teacher模型名称")

    # 数据集
    parser.add_argument("--dataset", type=str, required=True,
                       help="HuggingFace dataset name")
    parser.add_argument("--dataset_split", type=str, default="train",
                       help="Dataset split")

    # 处理范围
    parser.add_argument("--num_samples", type=int, default=None,
                       help="Number of samples (default: all)")
    parser.add_argument("--start_idx", type=int, default=0,
                       help="Start index")
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Batch size")

    # 生成参数
    parser.add_argument("--max_tokens", type=int, default=512,
                       help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0,
                       help="Sampling temperature")

    # Coordinator配置
    parser.add_argument("--top_k", type=int, default=20,
                       help="Top-k for approximation (max 20 for vLLM 0.11.0)")
    parser.add_argument("--force_first_token", action="store_true", default=True,
                       help="Force first token to use π_t")
    parser.add_argument("--constraint_to_target", action="store_true", default=True,
                       help="Enable trust region constraint")
    parser.add_argument("--target_top_p", type=float, default=0.95,
                       help="Trust region top-p threshold")

    # 稳定性检测
    parser.add_argument("--enable_stability_check", action="store_true",
                       help="Enable stability detection")
    parser.add_argument("--stability_threshold_js", type=float, default=0.5,
                       help="JS divergence threshold")
    parser.add_argument("--stability_threshold_overlap", type=float, default=0.1,
                       help="Overlap probability mass threshold")
    parser.add_argument("--auto_fallback", action="store_true", default=True,
                       help="Auto fallback to π_t when unstable")

    # Prompt模板
    parser.add_argument("--use_base_template", action="store_true", default=True,
                       help="Use simple template for base model")
    parser.add_argument("--use_instruct_template", action="store_true", default=True,
                       help="Use chat template for instruct model")

    # 输出
    parser.add_argument("--output", type=str, required=True,
                       help="Output file path (JSONL)")
    parser.add_argument("--save_diagnostics", action="store_true",
                       help="Save diagnostic information")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")

    args = parser.parse_args()

    # 运行
    asyncio.run(generate_data(args))


if __name__ == "__main__":
    main()
