#!/usr/bin/env python3
"""
数据清洗和 SFT 数据准备脚本

功能:
1. 读取生成的 parquet 数据
2. 使用 math_verify 验证 original_answer 的正确性
3. 过滤出正确的样本
4. 生成两组 SFT 数据:
   - original: prompt + original_answer (ground truth)
   - sampled: prompt + response (optimal sampling 生成的答案)
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Union
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
from tqdm import tqdm

# 添加 verl 路径到 sys.path
VERL_PATH = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(VERL_PATH))

# 导入验证函数
try:
    from math_verify import parse, verify
    USE_MATH_VERIFY = True
except ImportError:
    print("Error: Could not import math_verify!")
    print("Please install: pip install math-verify")
    sys.exit(1)


def verify_answer(response: str, ground_truth: str) -> bool:
    """
    验证答案是否正确（使用官方 math_verify）

    Args:
        response: 模型输出的答案
        ground_truth: 正确答案

    Returns:
        bool: 答案是否正确
    """
    try:
        # 使用官方 math_verify API
        # 注意：parse 会自动处理各种格式（latex, expr 等）
        gold = parse(ground_truth)
        answer = parse(response)

        # 顺序很重要！verify(gold, answer)
        result = verify(gold, answer)
        return result
    except Exception as e:
        # 如果解析或验证失败，返回 False
        # print(f"Verification error: {e}")
        return False


def extract_prompt_text(prompt_list: Union[List[Dict], np.ndarray, str]) -> str:
    """
    从 prompt 列表中提取文本

    Args:
        prompt_list: prompt 列表，可能是：
            - numpy.ndarray 包含 list
            - list of dicts: [{'content': '...', 'role': 'user'}]
            - 或直接是字符串

    Returns:
        str: 提取的 prompt 文本（去除 "Question: " 等前缀）
    """
    # 处理 numpy.ndarray
    if isinstance(prompt_list, np.ndarray):
        prompt_list = prompt_list.tolist()

    # 提取文本
    text = ""
    # 处理 list
    if isinstance(prompt_list, list) and len(prompt_list) > 0:
        first_item = prompt_list[0]
        if isinstance(first_item, dict):
            text = first_item.get('content', '')
        else:
            text = str(first_item)
    # 处理字符串
    elif isinstance(prompt_list, str):
        text = prompt_list
    # 兜底
    else:
        text = str(prompt_list)

    # 去除常见前缀（保证与 teacher file 格式一致）
    # 数据集可能有 "Question: " 前缀，而 teacher file 没有
    prefixes_to_remove = ["Question: ", "Question:\n", "Question:\n\n"]
    for prefix in prefixes_to_remove:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
            break

    # 去除常见后缀（保证与 teacher file 格式一致）
    # 数据集可能有 "\n\nAnswer:" 后缀，而 teacher file 没有
    suffixes_to_remove = ["\n\nAnswer:", "\nAnswer:", "Answer:"]
    for suffix in suffixes_to_remove:
        if text.endswith(suffix):
            text = text[:-len(suffix)].strip()
            break

    return text


def create_sft_format(prompt: str, response: str, source: str = "DeepScaleR") -> Dict:
    """
    创建 SFT 格式的数据

    Args:
        prompt: 问题
        response: 答案
        source: 数据来源（默认: DeepScaleR）

    Returns:
        Dict: SFT 格式的数据，包含 source, messages, num_turns
    """
    return {
        "source": source,
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response}
        ],
        "num_turns": 1
    }


def normalize_prompt(prompt: str) -> str:
    """
    标准化 prompt 用于匹配

    Args:
        prompt: 原始 prompt

    Returns:
        str: 标准化后的 prompt（去除多余空格）
    """
    return ' '.join(prompt.split())


# 全局变量，用于多进程共享（避免重复序列化）
_global_teacher_dict = None
_global_verify_original = False
_global_verify_sampled = False


def _init_worker(teacher_dict, verify_original, verify_sampled):
    """
    初始化 worker 进程的全局变量

    Args:
        teacher_dict: teacher 响应字典
        verify_original: 是否验证 original_answer
        verify_sampled: 是否验证 sampled response
    """
    global _global_teacher_dict, _global_verify_original, _global_verify_sampled
    _global_teacher_dict = teacher_dict
    _global_verify_original = verify_original
    _global_verify_sampled = verify_sampled


def load_teacher_responses(teacher_file: str) -> Dict[str, str]:
    """
    加载 teacher 响应并建立索引

    Args:
        teacher_file: teacher JSONL 文件路径

    Returns:
        Dict[str, str]: {normalized_prompt: teacher_response}
    """
    print(f"\nLoading teacher responses from {teacher_file}...")
    teacher_dict = {}

    with open(teacher_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line)
                messages = data.get('messages', [])
                if len(messages) >= 2:
                    user_prompt = messages[0].get('content', '')
                    teacher_response = messages[1].get('content', '')

                    # 使用标准化的 prompt 作为 key
                    normalized_prompt = normalize_prompt(user_prompt)
                    teacher_dict[normalized_prompt] = teacher_response
            except Exception as e:
                print(f"Warning: Failed to parse line {line_num}: {e}")

    print(f"Loaded {len(teacher_dict)} teacher responses")
    return teacher_dict


def process_single_sample(
    row_data: Tuple[int, Dict, str]
) -> Dict:
    """
    处理单个样本（用于多进程）

    Args:
        row_data: (index, row_dict, normalized_prompt) 样本数据

    Returns:
        Dict: 处理结果，包含是否有效、SFT 数据等
    """
    # 使用全局变量（在 worker 初始化时设置）
    global _global_teacher_dict, _global_verify_original, _global_verify_sampled

    idx, row, normalized_prompt = row_data
    extra_info = row['extra_info']
    prompt_list = row['prompt']

    # 提取必要的字段
    original_answer = extra_info.get('original_answer', '')
    original_solution = extra_info.get('original_solution', '')
    response = extra_info.get('response', '')

    # 提取 prompt 文本
    prompt_text = extract_prompt_text(prompt_list)

    # 验证 original_answer 的正确性（可选）
    if _global_verify_original:
        is_original_correct = verify_answer(original_solution, original_answer)
        if not is_original_correct:
            return {
                'valid': False,
                'idx': idx
            }

    # 准备结果
    result = {
        'valid': True,
        'idx': idx,
        'row': row,
        'original_sft': None,
        'sampled_sft': None,
        'teacher_sft': None,
        'sampled_correct': False,
        'teacher_correct': False,
        'teacher_found': False
    }

    # 查找 teacher response（如果提供了 teacher_dict）
    # 使用预计算的 normalized_prompt，避免重复计算
    teacher_response = None
    if _global_teacher_dict is not None:
        teacher_response = _global_teacher_dict.get(normalized_prompt)
        result['teacher_found'] = teacher_response is not None

    # 处理采样数据
    if response:
        if _global_verify_sampled:
            # 验证采样答案的正确性
            is_sampled_correct = verify_answer(response, original_answer)
            if is_sampled_correct:
                # 只有采样答案正确时，才同时保留 original、sampled 和 teacher
                result['sampled_correct'] = True
                result['original_sft'] = create_sft_format(prompt_text, original_solution)
                result['sampled_sft'] = create_sft_format(prompt_text, response)

                # 如果有 teacher response，也添加
                if teacher_response:
                    # 验证 teacher 答案
                    is_teacher_correct = verify_answer(teacher_response, original_answer)
                    result['teacher_correct'] = is_teacher_correct
                    # 无论 teacher 是否正确，都保存（为了公平对比）
                    result['teacher_sft'] = create_sft_format(prompt_text, teacher_response)
            # 如果采样答案不正确，所有 sft 都保持为 None（公平对比）
        else:
            # 不验证，保留所有 response、original 和 teacher
            result['original_sft'] = create_sft_format(prompt_text, original_solution)
            result['sampled_sft'] = create_sft_format(prompt_text, response)

            if teacher_response:
                result['teacher_sft'] = create_sft_format(prompt_text, teacher_response)
    else:
        # 如果没有 response，只保留 original（如果不验证 sampled）
        if not _global_verify_sampled:
            result['original_sft'] = create_sft_format(prompt_text, original_solution)

            if teacher_response:
                result['teacher_sft'] = create_sft_format(prompt_text, teacher_response)

    return result


def filter_and_prepare_data(
    input_file: str,
    output_dir: str,
    verify_original: bool = True,
    verify_sampled: bool = False,
    num_workers: int = 8,
    teacher_file: str = None
) -> Tuple[int, int, int, int]:
    """
    过滤和准备 SFT 数据

    Args:
        input_file: 输入的 parquet 文件路径
        output_dir: 输出目录
        verify_original: 是否验证 original_answer 的正确性
        verify_sampled: 是否验证 sampled response 的正确性
        num_workers: 多进程工作进程数（默认: 8）
        teacher_file: teacher 模型响应的 JSONL 文件路径（可选）

    Returns:
        Tuple[int, int, int, int]: (总样本数, 过滤后的样本数, 采样答案正确的样本数, teacher匹配的样本数)
    """
    # 读取数据
    print(f"Reading data from {input_file}...")
    df = pd.read_parquet(input_file)
    print(f"Total samples: {len(df)}")

    # 加载 teacher 响应（如果提供）
    teacher_dict = None
    if teacher_file:
        teacher_dict = load_teacher_responses(teacher_file)

    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 准备输出文件
    input_path = Path(input_file)
    base_name = input_path.stem  # 例如: kl_symmetry

    filtered_parquet = output_path / f"{base_name}_filtered.parquet"

    # 为 SFT 数据创建目录结构（HuggingFace 标准格式）
    original_sft_dir = output_path / f"{base_name}_sft_original"
    sampled_sft_dir = output_path / f"{base_name}_sft_sampled"
    teacher_sft_dir = output_path / f"{base_name}_sft_teacher" if teacher_dict else None
    stats_json = output_path / f"{base_name}_stats.json"

    # 创建 SFT 数据目录
    original_sft_dir.mkdir(parents=True, exist_ok=True)
    sampled_sft_dir.mkdir(parents=True, exist_ok=True)
    if teacher_sft_dir:
        teacher_sft_dir.mkdir(parents=True, exist_ok=True)

    # 统计信息
    total_samples = len(df)
    valid_samples = 0
    sampled_correct = 0
    teacher_found = 0
    teacher_correct = 0

    # 存储过滤后的数据
    filtered_data = []
    original_sft_data = []
    sampled_sft_data = []
    teacher_sft_data = []

    # 准备数据：转换为 (index, row_dict, normalized_prompt) 格式
    # 优化：使用 to_dict('records') 替代 iterrows()，速度提升 10-100 倍
    print(f"\nPreparing data for {len(df)} samples...")

    # 1. 一次性转换所有行为字典列表（比 iterrows() 快得多）
    records = df.to_dict('records')

    # 2. 预计算所有 normalized prompts（批量处理）
    print("Pre-computing normalized prompts...")
    normalized_prompts = [
        normalize_prompt(extract_prompt_text(record['prompt']))
        for record in records
    ]

    # 3. 组合成最终数据格式
    row_data_list = list(zip(range(len(records)), records, normalized_prompts))

    print(f"Processing samples with {num_workers} workers...")

    # 使用进程池，通过 initializer 在每个 worker 中设置全局变量
    # 这样避免了每次调用都序列化 teacher_dict（可能很大）
    with ProcessPoolExecutor(
        max_workers=num_workers,
        initializer=_init_worker,
        initargs=(teacher_dict, verify_original, verify_sampled)
    ) as executor:
        # 使用 map 批量处理，自动管理 chunksize
        results = list(tqdm(
            executor.map(process_single_sample, row_data_list, chunksize=100),
            total=len(row_data_list),
            desc="Processing"
        ))

    # 处理结果
    print("\nCollecting results...")
    for result in results:
        if not result['valid']:
            continue

        valid_samples += 1

        # 保存过滤后的原始数据
        filtered_data.append(result['row'])

        # 保存 Original SFT（如果存在）
        if result['original_sft'] is not None:
            original_sft_data.append(result['original_sft'])

        # 保存 Sampled SFT（如果存在）
        if result['sampled_sft'] is not None:
            sampled_sft_data.append(result['sampled_sft'])

        # 保存 Teacher SFT（如果存在）
        if result['teacher_sft'] is not None:
            teacher_sft_data.append(result['teacher_sft'])

        # 统计正确的采样答案
        if result['sampled_correct']:
            sampled_correct += 1

        # 统计 teacher 匹配和正确率
        if result['teacher_found']:
            teacher_found += 1
            if result['teacher_correct']:
                teacher_correct += 1

    # 保存过滤后的 parquet 数据
    print(f"\nSaving filtered parquet to {filtered_parquet}...")
    filtered_df = pd.DataFrame(filtered_data)
    filtered_df.to_parquet(filtered_parquet, index=False)

    # 保存 SFT 数据（HuggingFace 标准目录格式 - 使用 parquet）
    # 保存为 train.parquet，这样 load_dataset() 可以自动识别
    print(f"\nSaving original SFT data to {original_sft_dir}...")
    original_dataset = Dataset.from_list(original_sft_data)
    original_parquet_file = original_sft_dir / "train.parquet"
    original_dataset.to_parquet(str(original_parquet_file))
    print(f"Saved {len(original_sft_data)} samples")
    print(f"Load with: load_dataset('{original_sft_dir}')")

    print(f"\nSaving sampled SFT data to {sampled_sft_dir}...")
    sampled_dataset = Dataset.from_list(sampled_sft_data)
    sampled_parquet_file = sampled_sft_dir / "train.parquet"
    sampled_dataset.to_parquet(str(sampled_parquet_file))
    print(f"Saved {len(sampled_sft_data)} samples")
    print(f"Load with: load_dataset('{sampled_sft_dir}')")

    # 保存 Teacher SFT 数据（如果有）
    if teacher_sft_dir and teacher_sft_data:
        print(f"\nSaving teacher SFT data to {teacher_sft_dir}...")
        teacher_dataset = Dataset.from_list(teacher_sft_data)
        teacher_parquet_file = teacher_sft_dir / "train.parquet"
        teacher_dataset.to_parquet(str(teacher_parquet_file))
        print(f"Saved {len(teacher_sft_data)} samples")
        print(f"Load with: load_dataset('{teacher_sft_dir}')")

    # 保存统计信息
    stats = {
        "input_file": str(input_file),
        "total_samples": total_samples,
        "valid_samples": valid_samples,
        "filtered_ratio": valid_samples / total_samples if total_samples > 0 else 0,
        "sampled_correct": sampled_correct if verify_sampled else "not_verified",
        "sampled_accuracy": sampled_correct / valid_samples if verify_sampled and valid_samples > 0 else "not_verified",
        "teacher_file": str(teacher_file) if teacher_file else None,
        "teacher_found": teacher_found if teacher_dict else "not_applicable",
        "teacher_match_rate": teacher_found / valid_samples if teacher_dict and valid_samples > 0 else "not_applicable",
        "teacher_correct": teacher_correct if teacher_dict and verify_sampled else "not_verified",
        "teacher_accuracy": teacher_correct / teacher_found if teacher_dict and verify_sampled and teacher_found > 0 else "not_verified",
        "output_files": {
            "filtered_parquet": str(filtered_parquet),
            "original_sft": str(original_sft_dir),
            "sampled_sft": str(sampled_sft_dir),
            "teacher_sft": str(teacher_sft_dir) if teacher_sft_dir else None
        }
    }

    print(f"Saving statistics to {stats_json}...")
    with open(stats_json, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    # 打印统计信息
    print("\n" + "="*60)
    print("Data Filtering and Preparation Complete!")
    print("="*60)
    print(f"Total samples: {total_samples}")
    print(f"Valid samples (original_answer correct): {valid_samples}")
    print(f"Filter ratio: {valid_samples/total_samples*100:.2f}%")
    if verify_sampled:
        print(f"\nSampled response verification (--verify-sampled enabled):")
        print(f"  - Total responses evaluated: {valid_samples}")
        print(f"  - Correct responses: {sampled_correct}")
        print(f"  - Accuracy: {sampled_correct/valid_samples*100:.2f}%")
        print(f"  - ⚠️  IMPORTANT: For fair comparison, both datasets contain")
        print(f"     ONLY the samples where sampled response is correct")
        print(f"     Original samples dropped: {valid_samples - sampled_correct}")
    if teacher_dict:
        print(f"\nTeacher model integration:")
        print(f"  - Teacher file: {teacher_file}")
        print(f"  - Prompts matched: {teacher_found}/{valid_samples} ({teacher_found/valid_samples*100:.2f}%)")
        if verify_sampled and teacher_found > 0:
            print(f"  - Teacher correct (among matched): {teacher_correct}/{teacher_found} ({teacher_correct/teacher_found*100:.2f}%)")
            print(f"  - ⚠️  Note: Teacher dataset also filtered by sampled correctness")
    print(f"\nOutput files:")
    print(f"  - Filtered parquet: {filtered_parquet}")
    print(f"  - Original SFT (ground truth): {original_sft_dir}/ ({len(original_sft_data)} samples)")
    if verify_sampled:
        print(f"  - Sampled SFT (optimal sampling): {sampled_sft_dir}/ ({len(sampled_sft_data)} samples)")
        print(f"    → Same samples as original for fair comparison")
    else:
        print(f"  - Sampled SFT (optimal sampling, all): {sampled_sft_dir}/ ({len(sampled_sft_data)} samples)")
    if teacher_sft_dir and teacher_sft_data:
        print(f"  - Teacher SFT (teacher model): {teacher_sft_dir}/ ({len(teacher_sft_data)} samples)")
        if verify_sampled:
            print(f"    → Same samples as original/sampled for fair comparison")
    print(f"  - Statistics: {stats_json}")
    print(f"\nLoad datasets with:")
    print(f"  from datasets import load_dataset")
    print(f"  dataset = load_dataset('{original_sft_dir}')")
    print(f"  dataset = load_dataset('{sampled_sft_dir}')")
    if teacher_sft_dir and teacher_sft_data:
        print(f"  dataset = load_dataset('{teacher_sft_dir}')")
    print("="*60)

    return total_samples, valid_samples, sampled_correct, teacher_found


def main():
    parser = argparse.ArgumentParser(
        description="Filter data and prepare SFT datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 基本使用 - 只验证 original_answer
  python filter_and_prepare_sft_data.py \\
      --input production/data/deepscaler_q14b_q4bb_new/kl_symmetry.parquet \\
      --output production/data/deepscaler_q14b_q4bb_new/filtered/

  # 同时验证 sampled response 的准确率
  python filter_and_prepare_sft_data.py \\
      --input production/data/deepscaler_q14b_q4bb_new/kl_symmetry.parquet \\
      --output production/data/deepscaler_q14b_q4bb_new/filtered/ \\
      --verify-sampled

  # 不验证 original_answer (假设都是正确的)
  python filter_and_prepare_sft_data.py \\
      --input production/data/deepscaler_q14b_q4bb_new/kl_symmetry.parquet \\
      --output production/data/deepscaler_q14b_q4bb_new/filtered/ \\
      --no-verify-original

  # 引入 teacher model 响应
  python filter_and_prepare_sft_data.py \\
      --input production/data/deepscaler_q14b_q4bb_new/kl_symmetry.parquet \\
      --output production/data/deepscaler_q14b_q4bb_new/filtered/ \\
      --verify-sampled \\
      --teacher-response production/data/deepscaler_q14b.jsonl
        """
    )

    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='输入的 parquet 文件路径'
    )

    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='输出目录'
    )

    parser.add_argument(
        '--verify-original',
        action='store_true',
        default=False,
        help='验证 original_answer 的正确性（默认: False，因为来自数据集）'
    )

    parser.add_argument(
        '--no-verify-original',
        action='store_false',
        dest='verify_original',
        help='不验证 original_answer（默认行为）'
    )

    parser.add_argument(
        '--verify-sampled',
        action='store_true',
        default=False,
        help='验证采样答案的正确性。启用后，只有正确的采样答案会被保存到 sampled SFT 数据集 (默认: False，保留所有采样答案)'
    )

    parser.add_argument(
        '--num-workers',
        type=int,
        default=8,
        help='多进程工作进程数（默认: 8）。设置为 1 禁用多进程'
    )

    parser.add_argument(
        '--teacher-response',
        type=str,
        default=None,
        help='Teacher 模型响应的 JSONL 文件路径（可选）。启用后将创建第三个 teacher SFT 数据集'
    )

    args = parser.parse_args()

    # 检查输入文件是否存在
    if not Path(args.input).exists():
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    # 检查 teacher 文件是否存在（如果提供）
    if args.teacher_response and not Path(args.teacher_response).exists():
        print(f"Error: Teacher response file not found: {args.teacher_response}")
        sys.exit(1)

    # 执行过滤和准备
    filter_and_prepare_data(
        input_file=args.input,
        output_dir=args.output,
        verify_original=args.verify_original,
        verify_sampled=args.verify_sampled,
        num_workers=args.num_workers,
        teacher_file=args.teacher_response
    )


if __name__ == "__main__":
    main()


# python3 production/filter_and_prepare_sft_data.py     --input production/data/deepscaler_q14b_q4bb_new/kl_symmetry.parquet     --output production/data/deepscaler_q14b_q4bb_new/filtered/  --verify-sampled