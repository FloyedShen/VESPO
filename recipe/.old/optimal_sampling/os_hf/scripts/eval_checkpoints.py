#!/usr/bin/env python3
"""
Checkpoint 评测脚本

功能：
1. 遍历目录中的所有 HF model checkpoint
2. 对每个模型使用 vLLM 进行部署
3. 在多个数学数据集上评估性能
4. 使用 math_verify 验证答案
5. 保存详细结果到 jsonl 和汇总表格

支持的数据集：
- math-ai/amc23 (question, answer)
- HuggingFaceH4/aime_2024 (problem, answer)
- math-ai/aime25 (problem, answer)
- HuggingFaceH4/MATH-500 (problem, answer)
"""

import argparse
import json
import os
import sys
import time
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

try:
    from math_verify import parse, verify
    USE_MATH_VERIFY = True
except ImportError:
    print("Warning: math_verify not installed. Install with: pip install math-verify")
    USE_MATH_VERIFY = False

@dataclass
class EvalResult:
    """单条评测结果"""
    dataset: str
    problem: str
    ground_truth: str
    model_response: str
    model_answer: str  # 从 response 中提取的答案
    is_correct: bool
    model_name: str
    checkpoint_step: Optional[int] = None


@dataclass
class DatasetConfig:
    """数据集配置"""
    name: str
    hf_name: str
    question_col: str
    answer_col: str
    split: str = "test"


# 数据集配置
DATASETS = [
    DatasetConfig(
        name="amc23",
        hf_name="math-ai/amc23",
        question_col="question",
        answer_col="answer",
        split="test"
    ),
    DatasetConfig(
        name="aime_2024",
        hf_name="HuggingFaceH4/aime_2024",
        question_col="problem",
        answer_col="answer",
        split="train"  # 这个数据集可能只有 train split
    ),
    DatasetConfig(
        name="aime25",
        hf_name="math-ai/aime25",
        question_col="problem",
        answer_col="answer",
        split="test"
    ),
    DatasetConfig(
        name="MATH-500",
        hf_name="HuggingFaceH4/MATH-500",
        question_col="problem",
        answer_col="answer",
        split="test"
    ),
]


class VLLMServer:
    """vLLM 管理器 - 使用 Python API"""

    def __init__(self, model_path: str, port: int = 8000, tensor_parallel_size: Optional[int] = None, verbose: bool = False, chat_template: Optional[str] = None):
        self.model_path = model_path
        self.port = port
        self.tensor_parallel_size = tensor_parallel_size or self._get_gpu_count()
        self.verbose = verbose
        self.chat_template = chat_template
        self.llm = None
        self.tokenizer = None  # 用于应用chat template

    def _get_gpu_count(self) -> int:
        """获取可用的 GPU 数量"""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--list-gpus"],
                capture_output=True,
                text=True,
                check=True
            )
            return len(result.stdout.strip().split('\n'))
        except Exception:
            return 1

    def start(self) -> bool:
        """初始化 vLLM 模型"""
        print(f"\n{'='*60}")
        print(f"Loading vLLM model: {self.model_path}")
        print(f"Tensor Parallel Size: {self.tensor_parallel_size}")
        if self.chat_template:
            print(f"Chat Template: {self.chat_template}")
        print(f"{'='*60}\n")

        try:
            from vllm import LLM, SamplingParams
            from transformers import AutoTokenizer

            # 加载tokenizer用于应用chat template
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )

            # 初始化 LLM
            self.llm = LLM(
                model=self.model_path,
                tensor_parallel_size=self.tensor_parallel_size,
                trust_remote_code=True,
                max_model_len=16384,
                gpu_memory_utilization=0.95,
                enforce_eager=True,
            )

            print("✅ Model loaded successfully!")
            return True

        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            import traceback
            traceback.print_exc()
            return False

    def generate(
        self,
        prompt: str,
        max_tokens: int = 40960,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> Optional[str]:
        """生成单个回复"""
        results = self.generate_batch(
            [prompt],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p
        )
        return results[0] if results else None

    def generate_batch(
        self,
        prompts: List,  # Can be List[str] or List[List[Dict]]
        max_tokens: int = 40960,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> List[Optional[str]]:
        """批量生成回复

        Args:
            prompts: 可以是字符串列表或chat messages列表
                     - List[str]: 直接的prompt文本
                     - List[List[Dict]]: chat messages格式，会自动应用chat template
        """
        if self.llm is None:
            print("❌ Model not loaded")
            return [None] * len(prompts)

        try:
            from vllm import SamplingParams

            # 如果prompts是chat messages格式，使用tokenizer应用chat template
            processed_prompts = []
            for prompt in prompts:
                if isinstance(prompt, list) and len(prompt) > 0 and isinstance(prompt[0], dict):
                    # Chat messages格式，使用tokenizer的chat template
                    if self.tokenizer and hasattr(self.tokenizer, 'apply_chat_template'):
                        text = self.tokenizer.apply_chat_template(
                            prompt,
                            tokenize=False,
                            add_generation_prompt=True
                        )
                        processed_prompts.append(text)
                    else:
                        # 如果tokenizer不支持chat template，退回到简单拼接
                        text = prompt[0]["content"] if prompt else ""
                        processed_prompts.append(text)
                else:
                    # 字符串格式，直接使用
                    processed_prompts.append(prompt)

            sampling_params = SamplingParams(
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )

            outputs = self.llm.generate(processed_prompts, sampling_params)

            results = []
            for output in outputs:
                if output and output.outputs:
                    results.append(output.outputs[0].text.strip())
                else:
                    results.append(None)

            return results

        except Exception as e:
            print(f"❌ Generation error: {e}")
            import traceback
            traceback.print_exc()
            return [None] * len(prompts)

    def stop(self):
        """清理资源"""
        print(f"\n🛑 Cleaning up vLLM resources...")
        if self.llm is not None:
            # vLLM 会在 Python 退出时自动清理，但我们可以显式删除
            del self.llm
            self.llm = None

            # 清理 CUDA 缓存
            import torch
            torch.cuda.empty_cache()

            print("✅ Resources cleaned")


def find_checkpoints(root_dir: str, filter_checkpoint_pattern: bool = True) -> List[Path]:
    """
    查找目录中的所有 HF model checkpoint（递归查找）

    Args:
        root_dir: 根目录
        filter_checkpoint_pattern: 是否只保留路径中包含 'checkpoint-' 的目录（避免评测训练根路径的初始模型）

    Returns:
        List[Path]: checkpoint 路径列表
    """
    root_path = Path(root_dir)
    checkpoints = []

    print(f"🔍 Recursively searching for checkpoints in: {root_dir}")
    if filter_checkpoint_pattern:
        print("   Filtering: only directories containing 'checkpoint-' in path")

    # 递归查找包含 config.json 的目录
    for path in root_path.rglob("config.json"):
        model_dir = path.parent

        # 过滤：只保留路径中包含 'checkpoint-' 的目录
        if filter_checkpoint_pattern:
            # 检查路径的任何部分是否包含 'checkpoint-' 模式
            path_parts = model_dir.parts
            has_checkpoint_pattern = any('checkpoint-' in part for part in path_parts)
            if not has_checkpoint_pattern:
                continue

        # 确保这是一个有效的 HF model（包含 config.json 和模型权重文件）
        has_weights = (
            (model_dir / "pytorch_model.bin").exists() or
            (model_dir / "model.safetensors").exists() or
            list(model_dir.glob("pytorch_model-*.bin")) or
            list(model_dir.glob("model-*.safetensors"))
        )

        if has_weights:
            checkpoints.append(model_dir)

    # 按路径排序（通常 checkpoint-100, checkpoint-200 等会按数字顺序）
    checkpoints.sort()

    return checkpoints


def extract_checkpoint_info(checkpoint_path) -> Tuple[str, Optional[int]]:
    """
    从 checkpoint 路径提取模型名称和步数

    Args:
        checkpoint_path: checkpoint 路径（Path 或 str）

    Returns:
        Tuple[str, Optional[int]]: (model_name, step)
    """
    # 如果是字符串（HF 模型名），直接返回
    if isinstance(checkpoint_path, str):
        # HuggingFace 模型名称，使用斜杠替换为下划线作为目录名
        model_name = checkpoint_path.replace('/', '_')
        return model_name, None

    # 如果是 Path 对象，按原逻辑处理
    path_str = str(checkpoint_path)

    # 尝试从路径中提取 step 信息
    step = None
    if "checkpoint-" in path_str:
        try:
            step_str = path_str.split("checkpoint-")[-1].split("/")[0]
            step = int(step_str)
        except ValueError:
            pass

    # 提取模型名称：父目录名称 + checkpoint 名称
    # 例如：qwen3-4b-base-DeepScaleR-pure-q14b-sft/checkpoint-10
    parent_name = checkpoint_path.parent.name
    checkpoint_name = checkpoint_path.name

    # 如果父目录不是 checkpoints 这样的根目录，则包含父目录名称
    if parent_name and parent_name not in ['checkpoints', 'models', 'outputs']:
        model_name = f"{parent_name}/{checkpoint_name}"
    else:
        model_name = checkpoint_name

    return model_name, step


def is_checkpoint_evaluated(checkpoint_path, output_root: Path) -> bool:
    """
    检查 checkpoint 是否已经完成评测

    Args:
        checkpoint_path: checkpoint 路径（Path 或 str）
        output_root: 输出根目录

    Returns:
        bool: 是否已完成评测
    """
    model_name, _ = extract_checkpoint_info(checkpoint_path)
    output_dir = output_root / model_name

    # 检查是否存在 summary.json 文件
    summary_file = output_dir / "summary.json"
    if not summary_file.exists():
        return False

    # 检查 summary.json 是否包含所有数据集的结果
    try:
        with open(summary_file, 'r', encoding='utf-8') as f:
            summary = json.load(f)

        # 检查是否所有数据集都有结果
        expected_datasets = {ds.name for ds in DATASETS}
        evaluated_datasets = set(summary.keys())

        if expected_datasets.issubset(evaluated_datasets):
            return True
        else:
            missing = expected_datasets - evaluated_datasets
            print(f"   ⚠️  Incomplete evaluation: missing {missing}")
            return False

    except Exception as e:
        print(f"   ⚠️  Error reading summary.json: {e}")
        return False


def extract_answer_from_response(response: str) -> str:
    """
    从模型回复中提取答案

    常见格式：
    - \\boxed{答案}
    - Answer: 答案
    - 最后一行

    Args:
        response: 模型回复

    Returns:
        str: 提取的答案
    """
    # 尝试提取 \boxed{} 中的内容
    if "\\boxed{" in response:
        start = response.rfind("\\boxed{")
        if start != -1:
            # 找到匹配的右括号
            count = 1
            i = start + 7  # len("\\boxed{")
            while i < len(response) and count > 0:
                if response[i] == '{':
                    count += 1
                elif response[i] == '}':
                    count -= 1
                i += 1

            if count == 0:
                return response[start + 7:i - 1].strip()

    # 尝试提取 "Answer:" 后面的内容
    if "Answer:" in response:
        answer = response.split("Answer:")[-1].strip()
        # 取第一行
        answer = answer.split('\n')[0].strip()
        return answer

    # 尝试提取 "答案是" 后面的内容
    if "答案是" in response:
        answer = response.split("答案是")[-1].strip()
        answer = answer.split('\n')[0].strip()
        return answer

    # 返回最后一个非空行
    lines = [line.strip() for line in response.split('\n') if line.strip()]
    if lines:
        return lines[-1]

    return response.strip()


def verify_answer(model_answer: str, ground_truth: str) -> bool:
    """
    验证答案是否正确

    Args:
        model_answer: 模型给出的答案
        ground_truth: 正确答案

    Returns:
        bool: 是否正确
    """
    if not USE_MATH_VERIFY:
        # 如果没有 math_verify，使用简单的字符串比较
        return model_answer.strip() == ground_truth.strip()

    try:
        gold = parse(ground_truth)
        answer = parse(model_answer)
        return verify(gold, answer)
    except Exception:
        # 如果解析失败，使用字符串比较
        return model_answer.strip() == ground_truth.strip()


def create_prompt(question: str, dataset_name: str):
    """
    创建推理 prompt

    Args:
        question: 问题
        dataset_name: 数据集名称

    Returns:
        List[Dict]: Chat messages格式，会自动应用模型的chat template
                    [{"role": "user", "content": "..."}]
    """
    # 构建用户消息内容
    content = f"{question}\n\nPlease provide your answer in the format \\boxed{{answer}}."

    # 返回 chat messages 格式，vLLM会自动应用chat template
    return [{"role": "user", "content": content}]


def evaluate_on_dataset(
    server: VLLMServer,
    dataset_config: DatasetConfig,
    model_name: str,
    checkpoint_step: Optional[int],
    max_samples: Optional[int] = None,
    batch_size: int = 32,
) -> List[EvalResult]:
    """
    在单个数据集上评估

    Args:
        server: vLLM server
        dataset_config: 数据集配置
        model_name: 模型名称
        checkpoint_step: checkpoint 步数
        max_samples: 最大样本数（用于测试）
        batch_size: 批量推理的批次大小

    Returns:
        List[EvalResult]: 评测结果列表
    """
    print(f"\n{'='*60}")
    print(f"Evaluating on: {dataset_config.name}")
    print(f"{'='*60}\n")

    # 加载数据集
    try:
        dataset = load_dataset(dataset_config.hf_name, split=dataset_config.split)
    except Exception as e:
        print(f"❌ Failed to load dataset {dataset_config.hf_name}: {e}")
        # 尝试使用其他 split
        try:
            dataset = load_dataset(dataset_config.hf_name, split="train")
            print(f"✅ Loaded 'train' split instead")
        except Exception as e2:
            print(f"❌ Failed to load dataset with 'train' split: {e2}")
            return []

    # 限制样本数
    if max_samples and len(dataset) > max_samples:
        dataset = dataset.select(range(max_samples))

    print(f"Dataset size: {len(dataset)}")
    print(f"Batch size: {batch_size}")

    results = []

    # 批量处理数据集
    dataset_list = list(dataset)
    num_batches = (len(dataset_list) + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(num_batches), desc=f"Evaluating {dataset_config.name}"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(dataset_list))
        batch_examples = dataset_list[start_idx:end_idx]

        # 准备批量数据
        questions = [ex[dataset_config.question_col] for ex in batch_examples]
        ground_truths = [str(ex[dataset_config.answer_col]) for ex in batch_examples]
        prompts = [create_prompt(q, dataset_config.name) for q in questions]

        # 批量生成回复
        responses = server.generate_batch(prompts)

        # 处理批量结果
        for idx, (question, ground_truth, response) in enumerate(zip(questions, ground_truths, responses)):
            if response is None:
                print(f"⚠️  Generation failed for sample {start_idx + idx}")
                response = ""

            # 提取答案
            model_answer = extract_answer_from_response(response)

            # 验证答案
            is_correct = verify_answer(model_answer, ground_truth)

            print(
                f"[PROMPT] {question}\n"
                f"[RESPONSE] {response}\n"
                f"[MODEL_ANSWER] {model_answer}\n"
                f"[GROUND TRUTH] {ground_truth}\n"
                f"[IS_CORRECT] {is_correct}"
            )

            # 保存结果
            result = EvalResult(
                dataset=dataset_config.name,
                problem=question,
                ground_truth=ground_truth,
                model_response=response,
                model_answer=model_answer,
                is_correct=is_correct,
                model_name=model_name,
                checkpoint_step=checkpoint_step,
            )
            results.append(result)

    # 计算准确率
    accuracy = sum(r.is_correct for r in results) / len(results) if results else 0.0
    print(f"\n✅ {dataset_config.name} Accuracy: {accuracy:.2%} ({sum(r.is_correct for r in results)}/{len(results)})")

    return results


def save_results(
    results: List[EvalResult],
    output_dir: Path,
    dataset_name: str,
):
    """
    保存评测结果到 jsonl

    Args:
        results: 评测结果列表
        output_dir: 输出目录
        dataset_name: 数据集名称
    """
    output_file = output_dir / f"{dataset_name}.jsonl"

    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(asdict(result), ensure_ascii=False) + '\n')

    print(f"📝 Results saved to: {output_file}")


def create_summary(
    all_results: Dict[str, List[EvalResult]],
    output_dir: Path,
):
    """
    创建汇总统计

    Args:
        all_results: 所有数据集的结果
        output_dir: 输出目录
    """
    summary = {}

    for dataset_name, results in all_results.items():
        if results:
            accuracy = sum(r.is_correct for r in results) / len(results)
            correct = sum(r.is_correct for r in results)
            total = len(results)

            summary[dataset_name] = {
                "accuracy": accuracy,
                "correct": correct,
                "total": total,
            }

    # 保存为 JSON
    summary_file = output_dir / "summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n📊 Summary saved to: {summary_file}")

    # 打印汇总
    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")
    for dataset_name, stats in summary.items():
        print(f"{dataset_name:20s}: {stats['accuracy']:.2%} ({stats['correct']}/{stats['total']})")
    print(f"{'='*60}\n")

    return summary


def create_summary_table(
    results_dir: Path,
    output_file: Path,
):
    """
    创建所有 checkpoint 的汇总表格

    Args:
        results_dir: 结果目录
        output_file: 输出文件
    """
    rows = []

    # 遍历所有模型的结果目录
    for model_dir in sorted(results_dir.iterdir()):
        if not model_dir.is_dir():
            continue

        summary_file = model_dir / "summary.json"
        if not summary_file.exists():
            continue

        # 读取 summary
        with open(summary_file, 'r', encoding='utf-8') as f:
            summary = json.load(f)

        # 提取模型信息
        model_name = model_dir.name

        # 创建行
        row = {"model": model_name}
        for dataset_name, stats in summary.items():
            row[f"{dataset_name}_accuracy"] = stats["accuracy"]
            row[f"{dataset_name}_correct"] = stats["correct"]
            row[f"{dataset_name}_total"] = stats["total"]

        rows.append(row)

    # 创建 DataFrame
    if rows:
        df = pd.DataFrame(rows)

        # 按模型名称排序
        df = df.sort_values("model")

        # 保存为 CSV
        df.to_csv(output_file, index=False)
        print(f"\n📊 Summary table saved to: {output_file}")

        # 打印表格
        print(f"\n{df.to_string(index=False)}\n")
    else:
        print("⚠️  No results found to create summary table")


def evaluate_checkpoint(
    checkpoint_path,
    output_root: Path,
    port: int = 8000,
    max_samples: Optional[int] = None,
    tensor_parallel_size: Optional[int] = None,
    verbose: bool = False,
    chat_template: Optional[str] = None,
    batch_size: int = 32,
):
    """
    评估单个 checkpoint

    Args:
        checkpoint_path: checkpoint 路径（Path 对象或 HF 模型名字符串）
        output_root: 输出根目录
        port: vLLM server 端口
        max_samples: 每个数据集的最大样本数
        tensor_parallel_size: Tensor parallel size
        verbose: 是否显示详细的 vLLM 启动日志
        chat_template: 自定义 chat template 文件路径 (.j2 Jinja2 模板)
        batch_size: 批量推理的批次大小
    """
    # 提取模型信息
    model_name, checkpoint_step = extract_checkpoint_info(checkpoint_path)

    # 创建输出目录
    output_dir = output_root / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'#'*60}")
    print(f"# Evaluating: {model_name}")
    if checkpoint_step:
        print(f"# Checkpoint Step: {checkpoint_step}")
    print(f"# Model Path: {checkpoint_path}")
    print(f"# Output: {output_dir}")
    print(f"{'#'*60}\n")

    # 启动 vLLM server
    server = VLLMServer(
        model_path=str(checkpoint_path),
        port=port,
        tensor_parallel_size=tensor_parallel_size,
        verbose=verbose,
        chat_template=chat_template,
    )

    if not server.start():
        print(f"❌ Failed to start server for {checkpoint_path}")
        return

    try:
        # 在所有数据集上评估
        all_results = {}

        for dataset_config in DATASETS:
            results = evaluate_on_dataset(
                server=server,
                dataset_config=dataset_config,
                model_name=model_name,
                checkpoint_step=checkpoint_step,
                max_samples=max_samples,
                batch_size=batch_size,
            )

            if results:
                # 保存结果
                save_results(results, output_dir, dataset_config.name)
                all_results[dataset_config.name] = results

        # 创建汇总
        if all_results:
            create_summary(all_results, output_dir)

    finally:
        # 停止 server
        server.stop()


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate all checkpoints in a directory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 评估目录中的所有 checkpoint
  python eval_checkpoints.py \\
      --checkpoint-dir /path/to/checkpoints \\
      --output-dir ./eval_results

  # 使用指定的 GPU 数量
  python eval_checkpoints.py \\
      --checkpoint-dir /path/to/checkpoints \\
      --output-dir ./eval_results \\
      --tensor-parallel-size 4

  # 使用自定义 chat template
  python eval_checkpoints.py \\
      --checkpoint-dir /path/to/checkpoints \\
      --output-dir ./eval_results \\
      --chat-template ./templates/qwen.j2

  # 测试模式（每个数据集只评估 10 个样本）
  python eval_checkpoints.py \\
      --checkpoint-dir /path/to/checkpoints \\
      --output-dir ./eval_results \\
      --max-samples 10

  # 详细日志模式（调试用）
  python eval_checkpoints.py \\
      --checkpoint-dir /path/to/checkpoints \\
      --output-dir ./eval_results \\
      --verbose

  # 断点续传（跳过已完成的 checkpoint）
  python eval_checkpoints.py \\
      --checkpoint-dir /path/to/checkpoints \\
      --output-dir ./eval_results \\
      --resume

  # 包含所有模型（不过滤 checkpoint- 路径）
  python eval_checkpoints.py \\
      --checkpoint-dir /path/to/checkpoints \\
      --output-dir ./eval_results \\
      --no-filter

  # 添加 HuggingFace 上的其他模型进行对比评测
  python eval_checkpoints.py \\
      --checkpoint-dir /path/to/checkpoints \\
      --output-dir ./eval_results \\
      --extra-models Qwen/Qwen2.5-Math-7B-Instruct deepseek-ai/DeepSeek-Math-7B-Instruct

  # 只评测 HuggingFace 模型（不搜索本地 checkpoint）
  python eval_checkpoints.py \\
      --checkpoint-dir /nonexistent \\
      --output-dir ./eval_results \\
      --extra-models Qwen/Qwen2.5-Math-7B-Instruct meta-llama/Llama-3.1-8B-Instruct
        """
    )

    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        required=True,
        help='包含 checkpoint 的根目录'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='./eval_results',
        help='评测结果输出目录 (默认: ./eval_results)'
    )

    parser.add_argument(
        '--port',
        type=int,
        default=8000,
        help='vLLM server 端口 (默认: 8000)'
    )

    parser.add_argument(
        '--tensor-parallel-size',
        type=int,
        default=None,
        help='Tensor parallel size (默认: 自动检测 GPU 数量)'
    )

    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='每个数据集的最大样本数（用于测试）'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='显示详细的 vLLM 启动日志'
    )

    parser.add_argument(
        '--chat-template',
        type=str,
        default=None,
        help='自定义 chat template 文件路径 (.j2 Jinja2 模板文件)'
    )

    parser.add_argument(
        '--resume',
        action='store_true',
        help='断点续传：跳过已完成评测的 checkpoint（检查是否存在完整的 summary.json）'
    )

    parser.add_argument(
        '--no-filter',
        action='store_true',
        help='不过滤 checkpoint 路径：评测所有找到的模型（默认只评测路径中包含 "checkpoint-" 的模型）'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='批量推理的批次大小 (默认: 32)'
    )

    parser.add_argument(
        '--extra-models',
        type=str,
        nargs='*',
        default=[],
        help='额外的 HuggingFace 模型列表，用于对比评测（例如: Qwen/Qwen2.5-Math-7B-Instruct deepseek-ai/DeepSeek-Math-7B-Instruct）'
    )

    args = parser.parse_args()

    # 检查 chat template 文件是否存在
    if args.chat_template and not Path(args.chat_template).exists():
        print(f"❌ Chat template file not found: {args.chat_template}")
        sys.exit(1)

    # 检查 math_verify
    if not USE_MATH_VERIFY:
        print("⚠️  math_verify not available, will use string comparison")
        print("   Install with: pip install math-verify")

    # 查找所有 checkpoint（递归搜索）
    print(f"\n{'='*60}")
    print("Checkpoint Discovery")
    print(f"{'='*60}")
    checkpoints = find_checkpoints(
        args.checkpoint_dir,
        filter_checkpoint_pattern=not args.no_filter
    )

    if not checkpoints:
        print(f"❌ No checkpoints found in {args.checkpoint_dir}")
        if not args.no_filter:
            print("   💡 Tip: Use --no-filter to search for all models (not just checkpoint-* directories)")

        # 如果没有找到 checkpoint 但有 extra_models，继续执行
        if not args.extra_models:
            sys.exit(1)
        else:
            print("   ℹ️  But extra models are provided, continuing...")

    print(f"\n✅ Found {len(checkpoints)} checkpoint(s):")
    for ckpt in checkpoints:
        print(f"   - {ckpt}")

    # 添加 extra models
    if args.extra_models:
        print(f"\n{'='*60}")
        print("Extra HuggingFace Models")
        print(f"{'='*60}")
        print(f"Adding {len(args.extra_models)} extra model(s) for evaluation:")
        for model in args.extra_models:
            print(f"   - {model}")
            checkpoints.append(model)  # 直接添加字符串
        print()
    else:
        print()

    # 创建输出目录
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    # 如果启用断点续传，过滤已完成的 checkpoint
    if args.resume:
        print(f"\n{'='*60}")
        print("Resume Mode: Checking for completed checkpoints")
        print(f"{'='*60}")

        checkpoints_to_eval = []
        skipped_count = 0

        for checkpoint_path in checkpoints:
            model_name, step = extract_checkpoint_info(checkpoint_path)
            if is_checkpoint_evaluated(checkpoint_path, output_root):
                print(f"⏭️  Skipping (already evaluated): {model_name}")
                skipped_count += 1
            else:
                checkpoints_to_eval.append(checkpoint_path)

        print(f"\n📊 Resume Summary:")
        print(f"   - Total checkpoints: {len(checkpoints)}")
        print(f"   - Already evaluated: {skipped_count}")
        print(f"   - To evaluate: {len(checkpoints_to_eval)}")
        print()

        if not checkpoints_to_eval:
            print("✅ All checkpoints already evaluated!")
            print(f"   Results directory: {output_root}")

            # 直接跳到创建汇总表格
            print(f"\n{'#'*60}")
            print("# Creating summary table for all checkpoints")
            print(f"{'#'*60}\n")

            summary_table_file = output_root / "summary_table.csv"
            create_summary_table(output_root, summary_table_file)

            print(f"\n{'#'*60}")
            print("# Evaluation complete!")
            print(f"# Results saved to: {output_root}")
            print(f"{'#'*60}\n")

            return

        checkpoints = checkpoints_to_eval
    else:
        print(f"💡 Tip: Use --resume to skip already evaluated checkpoints\n")

    # 评估每个 checkpoint
    print(f"{'='*60}")
    print(f"Starting Evaluation")
    print(f"{'='*60}\n")

    for idx, checkpoint_path in enumerate(checkpoints, 1):
        model_name, step = extract_checkpoint_info(checkpoint_path)
        print(f"\n{'#'*60}")
        print(f"# Checkpoint {idx}/{len(checkpoints)}: {model_name}")
        if step:
            print(f"# Step: {step}")
        print(f"{'#'*60}")

        try:
            evaluate_checkpoint(
                checkpoint_path=checkpoint_path,
                output_root=output_root,
                port=args.port,
                max_samples=args.max_samples,
                tensor_parallel_size=args.tensor_parallel_size,
                verbose=args.verbose,
                chat_template=args.chat_template,
                batch_size=args.batch_size,
            )
        except Exception as e:
            print(f"❌ Error evaluating {checkpoint_path}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # 创建汇总表格
    print(f"\n{'#'*60}")
    print("# Creating summary table for all checkpoints")
    print(f"{'#'*60}\n")

    summary_table_file = output_root / "summary_table.csv"
    create_summary_table(output_root, summary_table_file)

    print(f"\n{'#'*60}")
    print("# Evaluation complete!")
    print(f"# Results saved to: {output_root}")
    print(f"{'#'*60}\n")


if __name__ == "__main__":
    main()
