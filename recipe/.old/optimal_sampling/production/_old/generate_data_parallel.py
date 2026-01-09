"""
数据生成脚本 (多卡并行版本)

功能:
- ✅ 批量处理 (batch processing)
- ✅ 数据并行 (data parallel) - 多卡处理不同的数据批次
- ✅ 模型切片 (model parallelism) - 大模型自动分片到多GPU
- ✅ 动态配置：根据GPU数量和显存自动选择策略
- ✅ 断点续传
- ✅ 自动数据集格式检测

使用示例:

    # 小模型 + 数据并行（推荐用于7B-13B模型）
    python generate_data_parallel.py \
        --model_theta Qwen/Qwen2.5-7B \
        --model_t Qwen/Qwen2.5-7B-Instruct \
        --dataset /path/to/DeepScaleR \
        --output generated_data.jsonl \
        --strategy data_parallel \
        --num_gpus 4 \
        --batch_size_per_gpu 8

    # 大模型 + 模型切片（推荐用于70B模型）
    python generate_data_parallel.py \
        --model_theta meta-llama/Llama-2-70b-hf \
        --model_t meta-llama/Llama-2-70b-chat-hf \
        --dataset /path/to/DeepScaleR \
        --output generated_data.jsonl \
        --strategy model_parallel \
        --num_gpus 4 \
        --load_in_4bit \
        --attn_implementation flash_attention_2

    # 自动策略（根据GPU数量和显存自动选择）
    python generate_data_parallel.py \
        --model_theta Qwen/Qwen2.5-7B \
        --model_t Qwen/Qwen2.5-7B-Instruct \
        --dataset /path/to/DeepScaleR \
        --output generated_data.jsonl \
        --strategy auto
"""

import os
import json
import argparse
import torch
import torch.multiprocessing as mp
from typing import List, Dict, Optional
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset, load_from_disk, Dataset
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from optimal_sampling_model import create_optimal_sampling_model


# ============================================
# 数据集适配器
# ============================================

class DatasetAdapter:
    """数据集适配器 - 自动检测格式"""

    def __init__(self, dataset_path: str, split: str = "train"):
        """
        Args:
            dataset_path: HuggingFace dataset name 或本地路径
            split: 数据集分割
        """
        self.dataset_path = dataset_path

        # 尝试加载数据集
        try:
            # 尝试作为本地路径加载
            if Path(dataset_path).exists():
                self.dataset = load_from_disk(dataset_path)
                if isinstance(self.dataset, dict) and split in self.dataset:
                    self.dataset = self.dataset[split]
                print(f"✓ Loaded dataset from local path: {dataset_path}")
            else:
                # 作为HuggingFace dataset加载
                self.dataset = load_dataset(dataset_path, split=split)
                print(f"✓ Loaded dataset from HuggingFace: {dataset_path}")
        except Exception as e:
            raise ValueError(f"Failed to load dataset from {dataset_path}: {e}")

        # 自动检测格式
        self._detect_format()

    def _detect_format(self):
        """自动检测数据集格式"""
        sample = self.dataset[0]
        self.columns = list(sample.keys())

        # 检测prompt字段
        prompt_candidates = ["prompt", "question", "instruction", "input", "text", "query"]
        self.prompt_field = None
        for candidate in prompt_candidates:
            if candidate in self.columns:
                self.prompt_field = candidate
                break

        # 检测messages字段 (OpenAI格式)
        self.messages_field = "messages" if "messages" in self.columns else None

        print(f"✓ Detected dataset format:")
        print(f"  Columns: {self.columns}")
        print(f"  Prompt field: {self.prompt_field}")
        print(f"  Messages field: {self.messages_field}")

    def __len__(self):
        return len(self.dataset)

    def get_prompt(self, idx: int) -> str:
        """提取prompt"""
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

    def to_openai_format(self, prompt: str, response: str, metadata: Dict = None) -> Dict:
        """转换为OpenAI API格式"""
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response}
        ]

        result = {"messages": messages}

        if metadata:
            result.update(metadata)

        return result

    def to_verl_format(self, prompt: str, response: str, metadata: Dict = None) -> Dict:
        """
        转换为 verl 训练格式

        verl 格式示例:
        {
            "data_source": "optimal_sampling",
            "prompt": [{"role": "user", "content": "..."}],
            "ability": "general",
            "extra_info": {
                "alpha_mean": 0.65,
                "response": "...",
                ...
            }
        }
        """
        data = {
            "data_source": "optimal_sampling",
            "prompt": [{"role": "user", "content": prompt}],
            "ability": "general",  # 可以根据任务类型设置
            "extra_info": {
                "response": response,
                "sample_idx": metadata.get("sample_idx", -1) if metadata else -1,
            }
        }

        # 添加其他元数据
        if metadata:
            for key, value in metadata.items():
                if key not in ["sample_idx"]:
                    data["extra_info"][key] = value

        return data


# ============================================
# 多GPU数据生成器
# ============================================

def worker_process(
    rank: int,
    world_size: int,
    model_theta: str,
    model_t: str,
    dataset_path: str,
    dataset_split: str,
    output_path: str,
    start_idx: int,
    end_idx: int,
    batch_size: int,
    max_new_tokens: int,
    temperature: float,
    alpha_method: str,
    fixed_alpha: float,
    model_kwargs: Dict,
    save_diagnostics: bool,
    output_format: str = "jsonl"  # "jsonl", "parquet", or "both"
):
    """
    Worker进程：处理分配给该GPU的数据

    Args:
        rank: GPU rank (0, 1, 2, ...)
        world_size: 总GPU数量
        其他参数：模型和生成配置
    """
    try:
        # 设置CUDA device
        torch.cuda.set_device(rank)
        device = f"cuda:{rank}"

        print(f"[GPU {rank}] Initializing worker...")

        # 加载数据集
        adapter = DatasetAdapter(dataset_path, dataset_split)

        # 创建模型（每个进程独立加载）
        print(f"[GPU {rank}] Loading model...")
        model = create_optimal_sampling_model(
            model_theta=model_theta,
            model_t=model_t,
            alpha_method=alpha_method,
            fixed_alpha=fixed_alpha,
            device=device,
            **model_kwargs
        )
        print(f"[GPU {rank}] Model loaded successfully")

        # 计算该进程处理的索引范围
        total_samples = end_idx - start_idx
        samples_per_gpu = total_samples // world_size
        extra = total_samples % world_size

        # 分配样本（尽量均匀）
        if rank < extra:
            local_start = start_idx + rank * (samples_per_gpu + 1)
            local_end = local_start + samples_per_gpu + 1
        else:
            local_start = start_idx + extra * (samples_per_gpu + 1) + (rank - extra) * samples_per_gpu
            local_end = local_start + samples_per_gpu

        print(f"[GPU {rank}] Processing samples {local_start} to {local_end} ({local_end - local_start} total)")

        # 输出文件（每个GPU独立文件）
        output_file_base = Path(output_path).parent / f"{Path(output_path).stem}_gpu{rank}"

        # 根据输出格式决定文件扩展名
        if output_format == "parquet":
            output_file = output_file_base.with_suffix('.parquet')
        else:
            output_file = output_file_base.with_suffix('.jsonl')

        diag_file = output_file_base.with_suffix('.diagnostics.jsonl') if save_diagnostics else None

        # 收集所有数据（用于parquet）
        all_data = []
        all_diag_data = []

        # 如果需要jsonl，打开文件
        f_out = None
        f_diag = None
        if output_format in ["jsonl", "both"]:
            f_out = open(output_file if output_format == "jsonl" else output_file_base.with_suffix('.jsonl'), 'w')
            if save_diagnostics:
                f_diag = open(diag_file, 'w')

        # 批处理生成
        for batch_start in tqdm(
            range(local_start, local_end, batch_size),
            desc=f"GPU {rank}",
            position=rank
        ):
            batch_end = min(batch_start + batch_size, local_end)
            batch_indices = range(batch_start, batch_end)

            # 提取prompts
            prompts = []
            metadata_list = []
            for idx in batch_indices:
                try:
                    prompt = adapter.get_prompt(idx)
                    metadata = adapter.get_metadata(idx)
                    prompts.append(prompt)
                    metadata_list.append(metadata)
                except Exception as e:
                    print(f"[GPU {rank}] Warning: Failed to extract prompt at {idx}: {e}")
                    continue

            if not prompts:
                continue

            # 生成
            try:
                outputs = model.generate(
                    prompts=prompts,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    return_diagnostics=save_diagnostics
                )

                # 保存结果
                for i, (prompt, response) in enumerate(zip(prompts, outputs.generated_texts)):
                    # 准备元数据（包含alpha等诊断信息）
                    metadata_with_diag = metadata_list[i].copy()
                    if save_diagnostics:
                        metadata_with_diag["alpha_mean"] = outputs.alpha_values[i].mean().item()
                        metadata_with_diag["alpha_std"] = outputs.alpha_values[i].std().item()
                        metadata_with_diag["ess_ratio_mean"] = outputs.ess_ratios[i].mean().item()
                        metadata_with_diag["gpu_rank"] = rank

                    # Parquet格式（verl格式）
                    if output_format in ["parquet", "both"]:
                        verl_data = adapter.to_verl_format(
                            prompt=prompt,
                            response=response,
                            metadata=metadata_with_diag
                        )
                        all_data.append(verl_data)

                    # JSONL格式（OpenAI格式）
                    if output_format in ["jsonl", "both"]:
                        openai_data = adapter.to_openai_format(
                            prompt=prompt,
                            response=response,
                            metadata=metadata_list[i]
                        )
                        f_out.write(json.dumps(openai_data, ensure_ascii=False) + '\n')
                        f_out.flush()

                    # 保存诊断信息
                    if save_diagnostics:
                        diag_data = {
                            "sample_idx": metadata_list[i].get("sample_idx"),
                            "alpha_mean": outputs.alpha_values[i].mean().item(),
                            "alpha_std": outputs.alpha_values[i].std().item(),
                            "ess_ratio_mean": outputs.ess_ratios[i].mean().item(),
                            "gpu_rank": rank
                        }
                        if output_format in ["jsonl", "both"]:
                            f_diag.write(json.dumps(diag_data, ensure_ascii=False) + '\n')
                            f_diag.flush()
                        all_diag_data.append(diag_data)

            except Exception as e:
                print(f"[GPU {rank}] Error during generation for batch {batch_start}-{batch_end}: {e}")
                import traceback
                traceback.print_exc()
                continue

        # 关闭jsonl文件
        if f_out:
            f_out.close()
        if f_diag:
            f_diag.close()

        # 写入parquet文件
        if output_format in ["parquet", "both"]:
            if all_data:
                print(f"[GPU {rank}] Writing parquet file...")
                # 转换为Dataset并保存
                dataset = Dataset.from_list(all_data)
                parquet_file = output_file if output_format == "parquet" else output_file_base.with_suffix('.parquet')
                dataset.to_parquet(str(parquet_file))
                print(f"[GPU {rank}] ✓ Parquet saved: {parquet_file}")

        print(f"[GPU {rank}] ✓ Completed! Output: {output_file}")

    except Exception as e:
        print(f"[GPU {rank}] FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()


def merge_outputs(output_path: str, num_gpus: int, output_format: str = "jsonl", delete_temp: bool = True):
    """合并多个GPU的输出文件"""
    print("\n" + "="*60)
    print("Merging outputs from all GPUs...")
    print("="*60)

    output_file = Path(output_path)

    # 合并 JSONL
    if output_format in ["jsonl", "both"]:
        merged_data = output_file.with_suffix('.jsonl')
        merged_diag = output_file.with_suffix('.diagnostics.jsonl')

        # 合并主输出
        with open(merged_data, 'w') as f_out:
            for rank in range(num_gpus):
                temp_file = output_file.parent / f"{output_file.stem}_gpu{rank}.jsonl"
                if temp_file.exists():
                    with open(temp_file, 'r') as f_in:
                        for line in f_in:
                            f_out.write(line)
                    if delete_temp:
                        temp_file.unlink()

        # 合并诊断信息
        diag_exists = False
        for rank in range(num_gpus):
            temp_diag = output_file.parent / f"{output_file.stem}_gpu{rank}.diagnostics.jsonl"
            if temp_diag.exists():
                diag_exists = True
                break

        if diag_exists:
            with open(merged_diag, 'w') as f_out:
                for rank in range(num_gpus):
                    temp_diag = output_file.parent / f"{output_file.stem}_gpu{rank}.diagnostics.jsonl"
                    if temp_diag.exists():
                        with open(temp_diag, 'r') as f_in:
                            for line in f_in:
                                f_out.write(line)
                        if delete_temp:
                            temp_diag.unlink()

        print(f"✓ Merged JSONL output: {merged_data}")
        if diag_exists:
            print(f"✓ Merged diagnostics: {merged_diag}")

    # 合并 Parquet
    if output_format in ["parquet", "both"]:
        merged_parquet = output_file.with_suffix('.parquet')

        # 收集所有GPU的数据
        all_datasets = []
        for rank in range(num_gpus):
            temp_parquet = output_file.parent / f"{output_file.stem}_gpu{rank}.parquet"
            if temp_parquet.exists():
                try:
                    ds = Dataset.from_parquet(str(temp_parquet))
                    all_datasets.append(ds)
                    if delete_temp:
                        temp_parquet.unlink()
                except Exception as e:
                    print(f"Warning: Failed to load {temp_parquet}: {e}")

        if all_datasets:
            # 合并所有dataset
            from datasets import concatenate_datasets
            merged_dataset = concatenate_datasets(all_datasets)
            merged_dataset.to_parquet(str(merged_parquet))
            print(f"✓ Merged Parquet output: {merged_parquet}")
            print(f"  Total samples: {len(merged_dataset)}")

    print("="*60 + "\n")


# ============================================
# 主函数
# ============================================

def main():
    parser = argparse.ArgumentParser(description="Generate data with multi-GPU support")

    # 模型参数
    parser.add_argument("--model_theta", type=str, required=True,
                       help="Path to π_θ model")
    parser.add_argument("--model_t", type=str, default=None,
                       help="Path to π_t model (default: same as model_theta)")

    # 数据集
    parser.add_argument("--dataset", type=str, required=True,
                       help="HuggingFace dataset name or local path")
    parser.add_argument("--dataset_split", type=str, default="train",
                       help="Dataset split")

    # 多GPU配置
    parser.add_argument("--strategy", type=str, default="auto",
                       choices=["auto", "data_parallel", "model_parallel"],
                       help="Parallelization strategy")
    parser.add_argument("--num_gpus", type=int, default=None,
                       help="Number of GPUs to use (default: all available)")
    parser.add_argument("--batch_size_per_gpu", type=int, default=8,
                       help="Batch size per GPU (for data_parallel)")

    # 生成参数
    parser.add_argument("--num_samples", type=int, default=None,
                       help="Number of samples to generate (default: all)")
    parser.add_argument("--start_idx", type=int, default=0,
                       help="Start index (for resuming)")
    parser.add_argument("--max_new_tokens", type=int, default=512,
                       help="Maximum new tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0,
                       help="Sampling temperature")

    # Alpha方法
    parser.add_argument("--alpha_method", type=str, default="kl_symmetry",
                       choices=["fixed", "kl_symmetry", "entropy", "ess_balance"],
                       help="Method to compute alpha")
    parser.add_argument("--fixed_alpha", type=float, default=0.5,
                       help="Fixed alpha value (when alpha_method=fixed)")

    # 大模型加速参数
    parser.add_argument("--load_in_4bit", action="store_true",
                       help="Enable INT4 quantization")
    parser.add_argument("--load_in_8bit", action="store_true",
                       help="Enable INT8 quantization")
    parser.add_argument("--attn_implementation", type=str, default=None,
                       choices=["eager", "flash_attention_2"],
                       help="Attention implementation")
    parser.add_argument("--dtype", type=str, default="float16",
                       choices=["float16", "bfloat16", "float32"],
                       help="Data type")

    # 输出
    parser.add_argument("--output", type=str, required=True,
                       help="Output file path")
    parser.add_argument("--output_format", type=str, default="jsonl",
                       choices=["jsonl", "parquet", "both"],
                       help="Output format: jsonl (OpenAI), parquet (verl), or both")
    parser.add_argument("--save_diagnostics", action="store_true",
                       help="Save diagnostic information")
    parser.add_argument("--keep_temp_files", action="store_true",
                       help="Keep temporary per-GPU files")

    args = parser.parse_args()

    # 检测可用GPU
    if torch.cuda.is_available():
        available_gpus = torch.cuda.device_count()
        if args.num_gpus is None:
            args.num_gpus = available_gpus
        else:
            args.num_gpus = min(args.num_gpus, available_gpus)
    else:
        raise RuntimeError("No CUDA devices available!")

    print("\n" + "="*60)
    print("Multi-GPU Data Generation")
    print("="*60)
    print(f"GPUs available: {available_gpus}")
    print(f"GPUs to use: {args.num_gpus}")
    print(f"Strategy: {args.strategy}")
    print(f"Batch size per GPU: {args.batch_size_per_gpu}")
    print("="*60 + "\n")

    # 决定策略
    if args.strategy == "auto":
        # 简单启发式：小模型用data_parallel，大模型用model_parallel
        # 这里根据模型名称简单判断
        model_name = args.model_theta.lower()
        if "70b" in model_name or "65b" in model_name:
            strategy = "model_parallel"
        else:
            strategy = "data_parallel"
        print(f"Auto-selected strategy: {strategy}")
    else:
        strategy = args.strategy

    # 转换dtype
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32
    }

    # 构建模型kwargs
    model_kwargs = {
        "dtype": dtype_map[args.dtype]
    }

    if args.load_in_4bit:
        model_kwargs["load_in_4bit"] = True
        model_kwargs["bnb_4bit_compute_dtype"] = torch.bfloat16
        model_kwargs["bnb_4bit_use_double_quant"] = True
        model_kwargs["bnb_4bit_quant_type"] = "nf4"

    if args.load_in_8bit:
        model_kwargs["load_in_8bit"] = True

    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation

    # 根据策略配置device_map
    if strategy == "model_parallel":
        # 模型切片：让transformers自动分配
        model_kwargs["device_map"] = "auto"
        if args.num_gpus > 1:
            # 限制每个GPU的显存（均分）
            max_memory = {i: f"{75 // args.num_gpus}GB" for i in range(args.num_gpus)}
            model_kwargs["max_memory"] = max_memory

        print(f"\n⚙️  Using MODEL PARALLEL strategy:")
        print(f"   Model will be split across {args.num_gpus} GPUs")
        print(f"   Using single process with device_map='auto'")
        print(f"   Memory limit per GPU: {max_memory if args.num_gpus > 1 else 'auto'}\n")

        # 模型切片模式：单进程处理所有数据
        # 不使用multiprocessing，因为模型已经跨多GPU
        adapter = DatasetAdapter(args.dataset, args.dataset_split)

        total_samples = len(adapter)
        if args.num_samples is None:
            args.num_samples = total_samples - args.start_idx
        else:
            args.num_samples = min(args.num_samples, total_samples - args.start_idx)

        # 创建模型（跨GPU）
        print("Loading model (will be split across GPUs)...")
        model = create_optimal_sampling_model(
            model_theta=args.model_theta,
            model_t=args.model_t,
            alpha_method=args.alpha_method,
            fixed_alpha=args.fixed_alpha,
            **model_kwargs
        )
        print("✓ Model loaded and distributed")

        # 单进程批处理
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 收集数据（用于parquet）
        all_data = []
        all_diag_data = []

        # 打开文件（如果需要jsonl）
        f_out = None
        f_diag = None
        if args.output_format in ["jsonl", "both"]:
            output_file_jsonl = output_path.with_suffix('.jsonl') if args.output_format == "both" else output_path
            f_out = open(output_file_jsonl, 'w')
            if args.save_diagnostics:
                f_diag = open(output_path.with_suffix('.diagnostics.jsonl'), 'w')

        for batch_start in tqdm(
            range(args.start_idx, args.start_idx + args.num_samples, args.batch_size_per_gpu),
            desc="Processing"
        ):
            batch_end = min(batch_start + args.batch_size_per_gpu, args.start_idx + args.num_samples)
            batch_indices = range(batch_start, batch_end)

            prompts = []
            metadata_list = []
            for idx in batch_indices:
                try:
                    prompt = adapter.get_prompt(idx)
                    metadata = adapter.get_metadata(idx)
                    prompts.append(prompt)
                    metadata_list.append(metadata)
                except Exception as e:
                    print(f"Warning: Failed to extract prompt at {idx}: {e}")
                    continue

            if not prompts:
                continue

            try:
                outputs = model.generate(
                    prompts=prompts,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    return_diagnostics=args.save_diagnostics
                )

                for i, (prompt, response) in enumerate(zip(prompts, outputs.generated_texts)):
                    # 准备元数据（包含alpha等诊断信息）
                    metadata_with_diag = metadata_list[i].copy()
                    if args.save_diagnostics:
                        metadata_with_diag["alpha_mean"] = outputs.alpha_values[i].mean().item()
                        metadata_with_diag["alpha_std"] = outputs.alpha_values[i].std().item()
                        metadata_with_diag["ess_ratio_mean"] = outputs.ess_ratios[i].mean().item()

                    # Parquet格式（verl格式）
                    if args.output_format in ["parquet", "both"]:
                        verl_data = adapter.to_verl_format(
                            prompt=prompt,
                            response=response,
                            metadata=metadata_with_diag
                        )
                        all_data.append(verl_data)

                    # JSONL格式（OpenAI格式）
                    if args.output_format in ["jsonl", "both"]:
                        openai_data = adapter.to_openai_format(
                            prompt=prompt,
                            response=response,
                            metadata=metadata_list[i]
                        )
                        f_out.write(json.dumps(openai_data, ensure_ascii=False) + '\n')
                        f_out.flush()

                    # 保存诊断信息
                    if args.save_diagnostics:
                        diag_data = {
                            "sample_idx": metadata_list[i].get("sample_idx"),
                            "alpha_mean": outputs.alpha_values[i].mean().item(),
                            "alpha_std": outputs.alpha_values[i].std().item(),
                            "ess_ratio_mean": outputs.ess_ratios[i].mean().item()
                        }
                        if args.output_format in ["jsonl", "both"]:
                            f_diag.write(json.dumps(diag_data, ensure_ascii=False) + '\n')
                            f_diag.flush()
                        all_diag_data.append(diag_data)

            except Exception as e:
                print(f"Error during generation: {e}")
                import traceback
                traceback.print_exc()
                continue

        # 关闭jsonl文件
        if f_out:
            f_out.close()
        if f_diag:
            f_diag.close()

        # 写入parquet文件
        if args.output_format in ["parquet", "both"]:
            if all_data:
                print("Writing parquet file...")
                dataset = Dataset.from_list(all_data)
                output_file_parquet = output_path.with_suffix('.parquet') if args.output_format == "both" else output_path
                dataset.to_parquet(str(output_file_parquet))
                print(f"✓ Parquet saved: {output_file_parquet}")

        print(f"\n✓ Generation completed!")
        if args.output_format == "jsonl":
            print(f"Output: {output_path}")
        elif args.output_format == "parquet":
            print(f"Output: {output_path}")
        else:  # both
            print(f"JSONL output: {output_path.with_suffix('.jsonl')}")
            print(f"Parquet output: {output_path.with_suffix('.parquet')}")

    else:  # data_parallel
        # 数据并行：每个GPU独立加载模型，处理不同的数据
        print(f"\n⚙️  Using DATA PARALLEL strategy:")
        print(f"   {args.num_gpus} independent processes, each with full model")
        print(f"   Data will be split across GPUs\n")

        # 加载数据集（只为了获取大小）
        adapter = DatasetAdapter(args.dataset, args.dataset_split)
        total_samples = len(adapter)

        if args.num_samples is None:
            args.num_samples = total_samples - args.start_idx
        else:
            args.num_samples = min(args.num_samples, total_samples - args.start_idx)

        end_idx = args.start_idx + args.num_samples

        # 为每个GPU指定device
        model_kwargs["device"] = "cuda"  # Will be set by worker

        # 启动多进程
        mp.set_start_method('spawn', force=True)

        processes = []
        for rank in range(args.num_gpus):
            p = mp.Process(
                target=worker_process,
                args=(
                    rank,
                    args.num_gpus,
                    args.model_theta,
                    args.model_t,
                    args.dataset,
                    args.dataset_split,
                    args.output,
                    args.start_idx,
                    end_idx,
                    args.batch_size_per_gpu,
                    args.max_new_tokens,
                    args.temperature,
                    args.alpha_method,
                    args.fixed_alpha,
                    model_kwargs,
                    args.save_diagnostics,
                    args.output_format  # ✨ 新增：传递output_format
                )
            )
            p.start()
            processes.append(p)

        # 等待所有进程完成
        for p in processes:
            p.join()

        # 合并输出
        merge_outputs(args.output, args.num_gpus, output_format=args.output_format, delete_temp=not args.keep_temp_files)

    print("\n" + "="*60)
    print("✅ All done!")
    print("="*60)


if __name__ == "__main__":
    main()
