#!/usr/bin/env python3
"""
Clean and Filter SFT Data using math-verify

This script:
1. Loads the ground truth dataset (agentica-org/DeepScaleR-Preview-Dataset)
2. Reads three JSONL files (baseline, ws, os)
3. Extracts answers from generated_reasoning (from \\boxed{})
4. Uses math-verify to check if answers are correct
5. Filters samples where all three methods got the correct answer
6. Saves three separate datasets (one per method) for SFT training
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import argparse
from datasets import load_dataset
from tqdm import tqdm

from math_verify import parse, verify



def verify_answer(predicted: str, ground_truth: str) -> bool:
    """
    Verify if predicted answer matches ground truth using math-verify

    Args:
        predicted: Predicted answer
        ground_truth: Ground truth answer

    Returns:
        True if answers match, False otherwise
    """
    try:
        # Parse both answers
        gold_parsed = parse(ground_truth)
        pred_parsed = parse(predicted)

        # Verify (order matters: gold first, then answer)
        result = verify(gold_parsed, pred_parsed)
        return result
    except Exception as e:
        # If parsing or verification fails, return False
        return False


def extract_boxed_answer(text: str) -> Optional[str]:
    """
    Extract answer from \\boxed{} in the text

    Args:
        text: Text containing \\boxed{answer}

    Returns:
        Extracted answer or None if not found
    """
    if not text:
        return None

    # Pattern to match \boxed{...}
    # Handle nested braces
    pattern = r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
    matches = re.findall(pattern, text)

    if matches:
        # Return the last boxed answer
        return matches[-1].strip()

    return None


def load_jsonl(file_path: Path) -> List[Dict]:
    """Load JSONL file"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse line in {file_path}: {e}")
    return data


def save_jsonl(data: List[Dict], file_path: Path):
    """Save data to JSONL file"""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def create_sft_record(record: Dict, ground_truth_data: Dict, method: str) -> Dict:
    """
    Create SFT training record

    Args:
        record: Original record from generated data
        ground_truth_data: Ground truth from dataset
        method: Method name

    Returns:
        SFT record with source, messages (OpenAI format), and num_turns
    """
    # Create messages in OpenAI format
    messages = [
        {
            "role": "user",
            "content": ground_truth_data['problem']
        },
        {
            "role": "assistant",
            "content": record['generated_reasoning']
        }
    ]

    return {
        'source': f"DeepScaleR-{method}",  # Source identifier
        'messages': messages,  # OpenAI message format
        'num_turns': 1  # Single-turn conversation
    }


def main():
    parser = argparse.ArgumentParser(description='Clean and filter SFT data using math-verify')
    parser.add_argument('--raw_dir', type=str,
                        default='/diancpfs/user/guobin/verl/recipe/optimal_sampling/os_hf/outputs/raw',
                        help='Directory containing raw JSONL files')
    parser.add_argument('--output_dir', type=str,
                        default='/diancpfs/user/guobin/verl/recipe/optimal_sampling/os_hf/outputs/sft',
                        help='Output directory for cleaned SFT data')
    parser.add_argument('--baseline_file', type=str, default='q3_1.7b-baseline.jsonl')
    parser.add_argument('--ws_file', type=str, default='q3_1.7b-ws.jsonl')
    parser.add_argument('--os_file', type=str, default='q3_1.7b-os.jsonl')
    parser.add_argument('--dataset', type=str, default='agentica-org/DeepScaleR-Preview-Dataset')

    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)

    print("=" * 80)
    print("SFT Data Cleaning and Filtering (with math-verify)")
    print("=" * 80)
    print()

    # Load ground truth dataset
    print("📂 Loading ground truth dataset...")
    dataset = load_dataset(args.dataset, split='train')
    print(f"   Loaded {len(dataset)} problems from {args.dataset}")
    print()

    # Create index map for ground truth
    print("📊 Creating ground truth index...")
    ground_truth_map = {i: dataset[i] for i in range(len(dataset))}
    print(f"   Indexed {len(ground_truth_map)} problems")
    print()

    # Load generated data
    print("📂 Loading generated data...")
    baseline_data = load_jsonl(raw_dir / args.baseline_file)
    ws_data = load_jsonl(raw_dir / args.ws_file)
    os_data = load_jsonl(raw_dir / args.os_file)

    print(f"   Baseline: {len(baseline_data)} records")
    print(f"   WS: {len(ws_data)} records")
    print(f"   OS: {len(os_data)} records")
    print()

    # Index data by problem index
    print("📊 Indexing data by problem index...")
    baseline_map = {r['index']: r for r in baseline_data}
    ws_map = {r['index']: r for r in ws_data}
    os_map = {r['index']: r for r in os_data}

    # Find common indices
    common_indices = set(baseline_map.keys()) & set(ws_map.keys()) & set(os_map.keys())
    print(f"   Common indices: {len(common_indices)}")
    print()

    # Verify answers using math-verify
    print("🔍 Verifying answers with math-verify...")
    print()

    stats = {
        'total': len(common_indices),
        'baseline_correct': 0,
        'ws_correct': 0,
        'os_correct': 0,
        'all_correct': 0,
        'all_wrong': 0,
        'baseline_only': 0,
        'ws_only': 0,
        'os_only': 0,
        'no_answer_extracted': 0,
        'verification_errors': 0
    }

    # Results
    all_correct_indices = []
    baseline_sft = []
    ws_sft = []
    os_sft = []

    # Process with progress bar
    for idx in tqdm(sorted(common_indices), desc="Processing"):
        # Skip if index not in ground truth
        if idx not in ground_truth_map:
            continue

        baseline_rec = baseline_map[idx]
        ws_rec = ws_map[idx]
        os_rec = os_map[idx]
        ground_truth_data = ground_truth_map[idx]

        # Ground truth answer
        ground_truth_answer = ground_truth_data['answer']

        # Extract generated answers
        baseline_ans = extract_boxed_answer(baseline_rec.get('generated_reasoning', ''))
        ws_ans = extract_boxed_answer(ws_rec.get('generated_reasoning', ''))
        os_ans = extract_boxed_answer(os_rec.get('generated_reasoning', ''))

        if not baseline_ans or not ws_ans or not os_ans:
            stats['no_answer_extracted'] += 1
            continue

        # Verify answers using math-verify
        try:
            baseline_correct = verify_answer(baseline_ans, ground_truth_answer)
            ws_correct = verify_answer(ws_ans, ground_truth_answer)
            os_correct = verify_answer(os_ans, ground_truth_answer)
        except Exception as e:
            # If verification fails, skip this sample
            stats['verification_errors'] += 1
            continue

        # Update statistics
        if baseline_correct:
            stats['baseline_correct'] += 1
        if ws_correct:
            stats['ws_correct'] += 1
        if os_correct:
            stats['os_correct'] += 1

        # All three correct
        if baseline_correct and ws_correct and os_correct:
            stats['all_correct'] += 1
            all_correct_indices.append(idx)

            # Add to SFT datasets
            baseline_sft.append(create_sft_record(baseline_rec, ground_truth_data, 'baseline'))
            ws_sft.append(create_sft_record(ws_rec, ground_truth_data, 'warmstart'))
            os_sft.append(create_sft_record(os_rec, ground_truth_data, 'optimal_sampling'))

        # None correct
        if not baseline_correct and not ws_correct and not os_correct:
            stats['all_wrong'] += 1

        # Only one correct
        if baseline_correct and not ws_correct and not os_correct:
            stats['baseline_only'] += 1
        if ws_correct and not baseline_correct and not os_correct:
            stats['ws_only'] += 1
        if os_correct and not baseline_correct and not ws_correct:
            stats['os_only'] += 1

    # Print statistics
    print()
    print("=" * 80)
    print("Statistics")
    print("=" * 80)
    print()
    print(f"Total samples analyzed: {stats['total']}")
    print(f"Samples with no answer extracted: {stats['no_answer_extracted']}")
    print(f"Samples with verification errors: {stats['verification_errors']}")
    print()

    valid_samples = stats['total'] - stats['no_answer_extracted'] - stats['verification_errors']
    print("Correctness by method:")
    print(f"  Baseline correct: {stats['baseline_correct']} ({stats['baseline_correct']/valid_samples*100:.1f}%)")
    print(f"  WS correct: {stats['ws_correct']} ({stats['ws_correct']/valid_samples*100:.1f}%)")
    print(f"  OS correct: {stats['os_correct']} ({stats['os_correct']/valid_samples*100:.1f}%)")
    print()
    print("Agreement statistics:")
    print(f"  All three correct: {stats['all_correct']} ({stats['all_correct']/valid_samples*100:.1f}%)")
    print(f"  All three wrong: {stats['all_wrong']} ({stats['all_wrong']/valid_samples*100:.1f}%)")
    print(f"  Only baseline correct: {stats['baseline_only']} ({stats['baseline_only']/valid_samples*100:.1f}%)")
    print(f"  Only WS correct: {stats['ws_only']} ({stats['ws_only']/valid_samples*100:.1f}%)")
    print(f"  Only OS correct: {stats['os_only']} ({stats['os_only']/valid_samples*100:.1f}%)")
    print()

    # Save SFT datasets as parquet (Hugging Face datasets format)
    print("💾 Saving SFT datasets as parquet...")

    # Convert to datasets and save
    from datasets import Dataset

    # Create directories for each method
    baseline_dir = output_dir / 'baseline_sft'
    ws_dir = output_dir / 'warmstart_sft'
    os_dir = output_dir / 'optimal_sampling_sft'

    baseline_dir.mkdir(parents=True, exist_ok=True)
    ws_dir.mkdir(parents=True, exist_ok=True)
    os_dir.mkdir(parents=True, exist_ok=True)

    # Save as parquet
    if baseline_sft:
        baseline_dataset = Dataset.from_list(baseline_sft)
        baseline_dataset.to_parquet(baseline_dir / 'train.parquet')

    if ws_sft:
        ws_dataset = Dataset.from_list(ws_sft)
        ws_dataset.to_parquet(ws_dir / 'train.parquet')

    if os_sft:
        os_dataset = Dataset.from_list(os_sft)
        os_dataset.to_parquet(os_dir / 'train.parquet')

    print(f"   Baseline SFT: {len(baseline_sft)} samples -> {baseline_dir}")
    print(f"   WS SFT: {len(ws_sft)} samples -> {ws_dir}")
    print(f"   OS SFT: {len(os_sft)} samples -> {os_dir}")
    print()

    # Save metadata
    metadata = {
        'total_samples': stats['total'],
        'valid_samples': valid_samples,
        'all_correct_samples': stats['all_correct'],
        'statistics': stats,
        'dataset': args.dataset,
        'output_files': {
            'baseline': str(baseline_dir / 'train.parquet'),
            'warmstart': str(ws_dir / 'train.parquet'),
            'optimal_sampling': str(os_dir / 'train.parquet')
        }
    }

    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"✅ Metadata saved to: {output_dir / 'metadata.json'}")
    print()
    print("=" * 80)
    print("Cleaning Complete!")
    print("=" * 80)
    print()
    print("📦 You can now load the datasets with:")
    print(f"   from datasets import load_dataset")
    print(f"   dataset = load_dataset('{baseline_dir}', split='train')")
    print(f"   # or")
    print(f"   dataset = load_dataset('{ws_dir}', split='train')")
    print(f"   # or")
    print(f"   dataset = load_dataset('{os_dir}', split='train')")
    print()


if __name__ == '__main__':
    main()
