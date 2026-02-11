#!/usr/bin/env python3
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess datasets for VESPO experiments.

Training sets:
- agentica-org/DeepScaleR-Preview-Dataset
- BytedTsinghua-SIA/DAPO-Math-17k

Test sets:
- math-ai/amc23
- HuggingFaceH4/aime_2024
- math-ai/aime25
- HuggingFaceH4/MATH-500
- google/IFEval
"""

import argparse
import json
import os
from pathlib import Path

import datasets
from verl.utils.reward_score.math_reward import last_boxed_only_string, remove_boxed


def extract_solution(solution_str):
    """Extract the boxed solution from a solution string."""
    try:
        return remove_boxed(last_boxed_only_string(solution_str))
    except:
        # If extraction fails, return the original solution
        return solution_str


def normalize_answer(answer):
    """Normalize answer to string format."""
    if answer is None:
        return ""
    if isinstance(answer, list):
        # If list, take first element if single item, otherwise join with comma
        if len(answer) == 1:
            return str(answer[0])
        elif len(answer) > 1:
            return ", ".join(str(a) for a in answer)
        else:
            return ""
    return str(answer)


def process_deepscaler_train(output_dir):
    """
    Process agentica-org/DeepScaleR-Preview-Dataset training set.

    Structure: {problem, answer, solution}
    """
    print("\n" + "=" * 80)
    print("Processing DeepScaleR-Preview-Dataset (Training Set)")
    print("=" * 80)

    data_source = "agentica-org/DeepScaleR-Preview-Dataset"
    dataset = datasets.load_dataset(data_source, split="train")

    print(f"Loaded {len(dataset)} examples")

    instruction = "Let's think step by step and output the final answer within \\boxed{}."

    def process_fn(example, idx):
        question = example["problem"] + " " + instruction
        # answer field contains the ground truth
        answer = normalize_answer(example["answer"])

        data = {
            "id": f"deepscaler_{idx}",
            "data_source": data_source,
            "prompt": [{"role": "user", "content": question}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": answer},
            "extra_info": {
                "split": "train",
                "index": idx,
                "has_solution": bool(example.get("solution"))
            }
        }
        return data

    processed_dataset = dataset.map(function=process_fn, with_indices=True, remove_columns=dataset.column_names)

    # Save to output directory
    save_dir = Path(output_dir) / "deepscaler"
    save_dir.mkdir(parents=True, exist_ok=True)

    processed_dataset.to_parquet(str(save_dir / "train.parquet"))

    # Save example
    with open(save_dir / "train_example.json", "w") as f:
        json.dump(processed_dataset[0], f, indent=2, ensure_ascii=False)

    print(f"✓ Saved to {save_dir}")
    print(f"  - train.parquet: {len(processed_dataset)} examples")

    return processed_dataset


def process_dapo_math(output_dir):
    """
    Process BytedTsinghua-SIA/DAPO-Math-17k training set.

    This dataset is already in the correct format with fields:
    {data_source, prompt, ability, reward_model, extra_info}

    We just need to add an id field and save it.
    """
    print("\n" + "=" * 80)
    print("Processing BytedTsinghua-SIA/DAPO-Math-17k (Training Set)")
    print("=" * 80)

    data_source = "BytedTsinghua-SIA/DAPO-Math-17k"
    dataset = datasets.load_dataset(data_source, split="train")

    print(f"Loaded {len(dataset)} examples")

    def process_fn(example, idx):
        # The dataset is already in the correct format, just add id

        prompt = example["prompt"]
        prompt[0]['content'] = prompt[0]['content'].replace(
            "Solve the following math problem step by step. The last line of your response should be of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\n",
            ""
        ).replace(
            '\n\nRemember to put your answer on its own line after "Answer:".',
            ' Please reason step by step, and put your final answer within \\boxed{}.'
        )
        # print(prompt)
        data = {
            "id": f"dapo_math_{idx}",
            "data_source": example.get("data_source", data_source),
            "prompt": prompt,
            "ability": example.get("ability", "math"),
            "reward_model": example["reward_model"],
            "extra_info": example.get("extra_info", {})
        }
        # Update extra_info to include our index
        if isinstance(data["extra_info"], dict):
            data["extra_info"]["processed_index"] = idx
            data["extra_info"]["split"] = "train"

        return data

    processed_dataset = dataset.map(function=process_fn, with_indices=True, remove_columns=dataset.column_names)

    # Save to output directory
    save_dir = Path(output_dir) / "dapo_math"
    save_dir.mkdir(parents=True, exist_ok=True)

    # Cast to explicit features to avoid metadata issues
    from datasets import Features, Value
    explicit_features = Features({
        'id': Value('string'),
        'data_source': Value('string'),
        'prompt': [{'content': Value('string'), 'role': Value('string')}],
        'ability': Value('string'),
        'reward_model': {
            'style': Value('string'),
            'ground_truth': Value('string')
        },
        'extra_info': {
            'index': Value('string'),
            'processed_index': Value('int64'),
            'split': Value('string')
        }
    })

    processed_dataset = processed_dataset.cast(explicit_features)
    processed_dataset.to_parquet(str(save_dir / "train.parquet"))

    # Save example
    with open(save_dir / "train_example.json", "w") as f:
        json.dump(processed_dataset[0], f, indent=2, ensure_ascii=False)

    print(f"✓ Saved to {save_dir}")
    print(f"  - train.parquet: {len(processed_dataset)} examples")

    return processed_dataset


def process_amc23(output_dir):
    """
    Process math-ai/amc23 test set.

    Structure: {id, question, answer, url}
    """
    print("\n" + "=" * 80)
    print("Processing math-ai/amc23 (Test Set)")
    print("=" * 80)

    data_source = "math-ai/amc23"
    dataset = datasets.load_dataset(data_source, split="test")

    print(f"Loaded {len(dataset)} examples")

    instruction = r"Please reason step by step, and put your final answer within \boxed{}."

    def process_fn(example, idx):
        question = example["question"] + " " + instruction
        answer = normalize_answer(example["answer"])

        data = {
            "id": f"amc23_{idx}",
            "data_source": data_source,
            "prompt": [{"role": "user", "content": question}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": answer},
            "extra_info": {
                "split": "test",
                "index": idx,
                # "original_id": example.get("id"),
                "url": example.get("url")
            }
        }
        return data

    processed_dataset = dataset.map(function=process_fn, with_indices=True, remove_columns=dataset.column_names)

    save_dir = Path(output_dir) / "amc23"
    save_dir.mkdir(parents=True, exist_ok=True)

    processed_dataset.to_parquet(str(save_dir / "test.parquet"))

    with open(save_dir / "test_example.json", "w") as f:
        json.dump(processed_dataset[0], f, indent=2, ensure_ascii=False)

    print(f"✓ Saved to {save_dir}")
    print(f"  - test.parquet: {len(processed_dataset)} examples")

    return processed_dataset


def process_aime_2024(output_dir):
    """
    Process HuggingFaceH4/aime_2024 test set.

    Structure: {id, problem, solution, answer, url, year}
    """
    print("\n" + "=" * 80)
    print("Processing HuggingFaceH4/aime_2024 (Test Set)")
    print("=" * 80)

    data_source = "HuggingFaceH4/aime_2024"
    dataset = datasets.load_dataset(data_source, split="train")

    print(f"Loaded {len(dataset)} examples")

    instruction = "Let's think step by step and output the final answer within \\boxed{}."

    def process_fn(example, idx):
        question = example["problem"] + " " + instruction
        answer = normalize_answer(example["answer"])

        data = {
            "id": f"aime2024_{idx}",
            "data_source": data_source,
            "prompt": [{"role": "user", "content": question}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": answer},
            "extra_info": {
                "split": "test",
                "index": idx,
                # "original_id": example.get("id"),
                "url": example.get("url"),
                "year": example.get("year"),
                "has_solution": bool(example.get("solution"))
            }
        }
        return data

    processed_dataset = dataset.map(function=process_fn, with_indices=True, remove_columns=dataset.column_names)

    save_dir = Path(output_dir) / "aime_2024"
    save_dir.mkdir(parents=True, exist_ok=True)

    processed_dataset.to_parquet(str(save_dir / "test.parquet"))

    with open(save_dir / "test_example.json", "w") as f:
        json.dump(processed_dataset[0], f, indent=2, ensure_ascii=False)

    print(f"✓ Saved to {save_dir}")
    print(f"  - test.parquet: {len(processed_dataset)} examples")

    return processed_dataset


def process_aime25(output_dir):
    """
    Process math-ai/aime25 test set.

    Structure: {problem, answer, id}
    """
    print("\n" + "=" * 80)
    print("Processing math-ai/aime25 (Test Set)")
    print("=" * 80)

    data_source = "math-ai/aime25"
    dataset = datasets.load_dataset(data_source, split="test")

    print(f"Loaded {len(dataset)} examples")

    instruction = "Let's think step by step and output the final answer within \\boxed{}."

    def process_fn(example, idx):
        question = example["problem"] + " " + instruction
        answer = normalize_answer(example["answer"])

        data = {
            "id": f"aime25_{idx}",
            "data_source": data_source,
            "prompt": [{"role": "user", "content": question}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": answer},
            "extra_info": {
                "split": "test",
                "index": idx,
                # "original_id": example.get("id")
            }
        }
        return data

    processed_dataset = dataset.map(function=process_fn, with_indices=True, remove_columns=dataset.column_names)

    save_dir = Path(output_dir) / "aime25"
    save_dir.mkdir(parents=True, exist_ok=True)

    processed_dataset.to_parquet(str(save_dir / "test.parquet"))

    with open(save_dir / "test_example.json", "w") as f:
        json.dump(processed_dataset[0], f, indent=2, ensure_ascii=False)

    print(f"✓ Saved to {save_dir}")
    print(f"  - test.parquet: {len(processed_dataset)} examples")

    return processed_dataset


def process_math500(output_dir):
    """
    Process HuggingFaceH4/MATH-500 test set.

    Structure: {problem, solution, answer, subject, level, unique_id}
    """
    print("\n" + "=" * 80)
    print("Processing HuggingFaceH4/MATH-500 (Test Set)")
    print("=" * 80)

    data_source = "HuggingFaceH4/MATH-500"
    dataset = datasets.load_dataset(data_source, split="test")

    print(f"Loaded {len(dataset)} examples")

    instruction = "Let's think step by step and output the final answer within \\boxed{}."

    def process_fn(example, idx):
        question = example["problem"] + " " + instruction
        answer = normalize_answer(example["answer"])

        data = {
            "id": f"math500_{idx}",
            "data_source": data_source,
            "prompt": [{"role": "user", "content": question}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": answer},
            "extra_info": {
                "split": "test",
                "index": idx,
                "subject": example.get("subject"),
                "level": example.get("level"),
                "unique_id": example.get("unique_id"),
                "has_solution": bool(example.get("solution"))
            }
        }
        return data

    processed_dataset = dataset.map(function=process_fn, with_indices=True, remove_columns=dataset.column_names)

    save_dir = Path(output_dir) / "math500"
    save_dir.mkdir(parents=True, exist_ok=True)

    processed_dataset.to_parquet(str(save_dir / "test.parquet"))

    with open(save_dir / "test_example.json", "w") as f:
        json.dump(processed_dataset[0], f, indent=2, ensure_ascii=False)

    print(f"✓ Saved to {save_dir}")
    print(f"  - test.parquet: {len(processed_dataset)} examples")

    return processed_dataset


def process_ifeval(output_dir):
    """
    Process google/IFEval test set (instruction following evaluation).

    Structure: {key, prompt, instruction_id_list, kwargs}
    """
    print("\n" + "=" * 80)
    print("Processing google/IFEval (Test Set)")
    print("=" * 80)

    data_source = "google/IFEval"
    dataset = datasets.load_dataset(data_source, split="train")

    print(f"Loaded {len(dataset)} examples")

    def process_fn(example, idx):
        prompt_text = example["prompt"]

        data = {
            "id": f"ifeval_{idx}",
            "data_source": data_source,
            "prompt": [{"role": "user", "content": prompt_text}],
            "ability": "instruction_following",
            "reward_model": {
                "style": "ifeval",
                "instruction_id_list": example.get("instruction_id_list", []),
                "kwargs": example.get("kwargs", [])
            },
            "extra_info": {
                "split": "test",
                "index": idx,
                "original_key": example.get("key")
            }
        }
        return data

    processed_dataset = dataset.map(function=process_fn, with_indices=True, remove_columns=dataset.column_names)

    save_dir = Path(output_dir) / "ifeval"
    save_dir.mkdir(parents=True, exist_ok=True)

    processed_dataset.to_parquet(str(save_dir / "test.parquet"))

    with open(save_dir / "test_example.json", "w") as f:
        json.dump(processed_dataset[0], f, indent=2, ensure_ascii=False)

    print(f"✓ Saved to {save_dir}")
    print(f"  - test.parquet: {len(processed_dataset)} examples")

    return processed_dataset


def process_minervamath(output_dir):
    """
    Process math-ai/minervamath test set.

    Structure: {question, answer}
    """
    print("\n" + "=" * 80)
    print("Processing math-ai/minervamath (Test Set)")
    print("=" * 80)

    data_source = "math-ai/minervamath"
    dataset = datasets.load_dataset(data_source, split="test")

    print(f"Loaded {len(dataset)} examples")

    instruction = "Let's think step by step and output the final answer within \\boxed{}."

    def process_fn(example, idx):
        question = example["question"] + " " + instruction
        answer = normalize_answer(example["answer"])

        data = {
            "id": f"minervamath_{idx}",
            "data_source": data_source,
            "prompt": [{"role": "user", "content": question}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": answer},
            "extra_info": {
                "split": "test",
                "index": idx,
            }
        }
        return data

    processed_dataset = dataset.map(function=process_fn, with_indices=True, remove_columns=dataset.column_names)

    save_dir = Path(output_dir) / "minervamath"
    save_dir.mkdir(parents=True, exist_ok=True)

    processed_dataset.to_parquet(str(save_dir / "test.parquet"))

    with open(save_dir / "test_example.json", "w") as f:
        json.dump(processed_dataset[0], f, indent=2, ensure_ascii=False)

    print(f"✓ Saved to {save_dir}")
    print(f"  - test.parquet: {len(processed_dataset)} examples")

    return processed_dataset


def process_olympiadbench(output_dir):
    """
    Process math-ai/olympiadbench test set.

    Structure: {id, question, solution, final_answer, context, image_*, modality,
                difficulty, is_multiple_answer, unit, answer_type, error,
                question_type, subfield, subject, language}
    """
    print("\n" + "=" * 80)
    print("Processing math-ai/olympiadbench (Test Set)")
    print("=" * 80)

    data_source = "math-ai/olympiadbench"
    dataset = datasets.load_dataset(data_source, split="test")

    print(f"Loaded {len(dataset)} examples")

    instruction = "Let's think step by step and output the final answer within \\boxed{}."

    def process_fn(example, idx):
        # Combine question with context if available
        question = example["question"]
        if example.get("context"):
            question = example["context"] + "\n\n" + question

        question = question + " " + instruction

        # Use final_answer as ground truth
        answer = normalize_answer(example.get("final_answer", example.get("answer", "")))

        data = {
            "id": f"olympiadbench_{idx}",
            "data_source": data_source,
            "prompt": [{"role": "user", "content": question}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": answer},
            "extra_info": {
                "split": "test",
                "index": idx,
                # "original_id": example.get("id"),
                "difficulty": example.get("difficulty"),
                "subject": example.get("subject"),
                "subfield": example.get("subfield"),
                "question_type": example.get("question_type"),
                "modality": example.get("modality"),
                "is_multiple_answer": example.get("is_multiple_answer"),
                "answer_type": example.get("answer_type"),
                "unit": example.get("unit"),
                "language": example.get("language"),
                "has_solution": bool(example.get("solution")),
                "has_images": any([
                    example.get(f"image_{i}") for i in range(1, 6)
                ])
            }
        }
        return data

    processed_dataset = dataset.map(function=process_fn, with_indices=True, remove_columns=dataset.column_names)

    save_dir = Path(output_dir) / "olympiadbench"
    save_dir.mkdir(parents=True, exist_ok=True)

    processed_dataset.to_parquet(str(save_dir / "test.parquet"))

    with open(save_dir / "test_example.json", "w") as f:
        json.dump(processed_dataset[0], f, indent=2, ensure_ascii=False)

    print(f"✓ Saved to {save_dir}")
    print(f"  - test.parquet: {len(processed_dataset)} examples")

    return processed_dataset


def process_beyondaime(output_dir):
    """
    Process ByteDance-Seed/BeyondAIME test set.

    Structure: {problem, answer}
    """
    print("\n" + "=" * 80)
    print("Processing ByteDance-Seed/BeyondAIME (Test Set)")
    print("=" * 80)

    data_source = "ByteDance-Seed/BeyondAIME"
    dataset = datasets.load_dataset(data_source, split="test")

    print(f"Loaded {len(dataset)} examples")

    instruction = "Let's think step by step and output the final answer within \\boxed{}."

    def process_fn(example, idx):
        question = example["problem"] + " " + instruction
        answer = normalize_answer(example["answer"])

        data = {
            "id": f"beyondaime_{idx}",
            "data_source": data_source,
            "prompt": [{"role": "user", "content": question}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": answer},
            "extra_info": {
                "split": "test",
                "index": idx
            }
        }
        return data

    processed_dataset = dataset.map(function=process_fn, with_indices=True, remove_columns=dataset.column_names)

    save_dir = Path(output_dir) / "beyondaime"
    save_dir.mkdir(parents=True, exist_ok=True)

    processed_dataset.to_parquet(str(save_dir / "test.parquet"))

    with open(save_dir / "test_example.json", "w") as f:
        json.dump(processed_dataset[0], f, indent=2, ensure_ascii=False)

    print(f"✓ Saved to {save_dir}")
    print(f"  - test.parquet: {len(processed_dataset)} examples")

    return processed_dataset


def process_gpqa(output_dir):
    """
    Process Idavidrein/gpqa test set.

    Structure: {Question, Correct Answer, Incorrect Answer 1, Incorrect Answer 2,
                Incorrect Answer 3, subdomain}
    """
    print("\n" + "=" * 80)
    print("Processing Idavidrein/gpqa (Test Set)")
    print("=" * 80)

    data_source = "Idavidrein/gpqa"
    dataset = datasets.load_dataset(data_source, "gpqa_diamond", split="train")

    print(f"Loaded {len(dataset)} examples")

    instruction = "Please reason step by step, and put your final answer within \\boxed{}."

    def process_fn(example, idx):
        question = example["Question"] + " " + instruction
        answer = normalize_answer(example["Correct Answer"])

        data = {
            "id": f"gpqa_{idx}",
            "data_source": data_source,
            "prompt": [{"role": "user", "content": question}],
            "ability": "reasoning",
            "reward_model": {"style": "rule", "ground_truth": answer},
            "extra_info": {
                "split": "test",
                "index": idx,
                "subdomain": example.get("subdomain", example.get("Subdomain")),
            }
        }
        return data

    processed_dataset = dataset.map(function=process_fn, with_indices=True, remove_columns=dataset.column_names)

    save_dir = Path(output_dir) / "gpqa"
    save_dir.mkdir(parents=True, exist_ok=True)

    processed_dataset.to_parquet(str(save_dir / "test.parquet"))

    with open(save_dir / "test_example.json", "w") as f:
        json.dump(processed_dataset[0], f, indent=2, ensure_ascii=False)

    print(f"✓ Saved to {save_dir}")
    print(f"  - test.parquet: {len(processed_dataset)} examples")

    return processed_dataset


def process_eurus_code_datasets(output_dir):
    """
    Process code datasets from PRIME-RL/Eurus-2-RL-Data validation split.

    Extracts and saves 4 separate code datasets:
    - taco: 382 examples
    - codecontests: 377 examples
    - apps: 142 examples
    - codeforces: 123 examples

    These datasets are already preprocessed and ready to use.
    """
    print("\n" + "=" * 80)
    print("Processing PRIME-RL/Eurus-2-RL-Data Code Datasets")
    print("=" * 80)

    data_source = "PRIME-RL/Eurus-2-RL-Data"

    # Load validation split
    dataset = datasets.load_dataset(data_source, split="validation")

    print(f"Loaded {len(dataset)} total examples from validation split")

    # Filter for code examples only
    code_dataset = dataset.filter(lambda x: x.get("ability") == "code" or x.get("ability") == "Code")

    print(f"Found {len(code_dataset)} code examples")

    # Split by data source
    code_sources = ["taco", "codecontests", "apps", "codeforces"]
    results = {}

    for source_name in code_sources:
        print(f"\n--- Processing {source_name} ---")

        # Filter for this specific source
        source_dataset = code_dataset.filter(lambda x: x.get("data_source") == source_name)

        if len(source_dataset) == 0:
            print(f"  Warning: No examples found for {source_name}")
            continue

        print(f"  Found {len(source_dataset)} examples")

        # Add unique IDs
        def add_id(example, idx):
            example["id"] = f"{source_name}_{idx}"
            return example

        source_dataset = source_dataset.map(add_id, with_indices=True)

        # Save to separate directory
        save_dir = Path(output_dir) / source_name
        save_dir.mkdir(parents=True, exist_ok=True)

        source_dataset.to_parquet(str(save_dir / "test.parquet"))

        with open(save_dir / "test_example.json", "w") as f:
            json.dump(source_dataset[0], f, indent=2, ensure_ascii=False)

        print(f"  ✓ Saved to {save_dir}")
        print(f"    - test.parquet: {len(source_dataset)} examples")

        results[source_name] = source_dataset

    return results


def main():
    parser = argparse.ArgumentParser(description="Preprocess datasets for VESPO experiments")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../data",
        help="Output directory for processed datasets"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["deepscaler", "dapo_math", "amc23", "aime_2024", "aime25", "math500", "ifeval",
                 "minervamath", "olympiadbench", "beyondaime", "gpqa",
                 "taco", "codecontests", "apps", "codeforces", "eurus_code", "all"],
        default=["all"],
        help="Which datasets to process (default: all). Use 'eurus_code' to process all 4 code datasets at once."
    )

    args = parser.parse_args()

    # Resolve output directory
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("VESPO Dataset Preprocessing")
    print("=" * 80)
    print(f"Output directory: {output_dir}")
    print(f"Datasets to process: {args.datasets}")

    # Determine which datasets to process
    datasets_to_process = args.datasets
    if "all" in datasets_to_process:
        datasets_to_process = ["deepscaler", "dapo_math", "amc23", "aime_2024", "aime25", "math500",
                               "ifeval", "minervamath", "olympiadbench", "beyondaime", "gpqa", "eurus_code"]

    # Expand eurus_code to individual datasets
    if "eurus_code" in datasets_to_process:
        datasets_to_process.remove("eurus_code")
        datasets_to_process.extend(["taco", "codecontests", "apps", "codeforces"])

    # Process each dataset
    results = {}

    if "deepscaler" in datasets_to_process:
        try:
            results["deepscaler"] = process_deepscaler_train(output_dir)
        except Exception as e:
            print(f"✗ Error processing deepscaler: {e}")

    if "dapo_math" in datasets_to_process:
        try:
            results["dapo_math"] = process_dapo_math(output_dir)
        except Exception as e:
            print(f"✗ Error processing dapo_math: {e}")

    if "amc23" in datasets_to_process:
        try:
            results["amc23"] = process_amc23(output_dir)
        except Exception as e:
            print(f"✗ Error processing amc23: {e}")

    if "aime_2024" in datasets_to_process:
        try:
            results["aime_2024"] = process_aime_2024(output_dir)
        except Exception as e:
            print(f"✗ Error processing aime_2024: {e}")

    if "aime25" in datasets_to_process:
        try:
            results["aime25"] = process_aime25(output_dir)
        except Exception as e:
            print(f"✗ Error processing aime25: {e}")

    if "math500" in datasets_to_process:
        try:
            results["math500"] = process_math500(output_dir)
        except Exception as e:
            print(f"✗ Error processing math500: {e}")

    if "ifeval" in datasets_to_process:
        try:
            results["ifeval"] = process_ifeval(output_dir)
        except Exception as e:
            print(f"✗ Error processing ifeval: {e}")

    if "minervamath" in datasets_to_process:
        try:
            results["minervamath"] = process_minervamath(output_dir)
        except Exception as e:
            print(f"✗ Error processing minervamath: {e}")

    if "olympiadbench" in datasets_to_process:
        try:
            results["olympiadbench"] = process_olympiadbench(output_dir)
        except Exception as e:
            print(f"✗ Error processing olympiadbench: {e}")

    if "beyondaime" in datasets_to_process:
        try:
            results["beyondaime"] = process_beyondaime(output_dir)
        except Exception as e:
            print(f"✗ Error processing beyondaime: {e}")

    if "gpqa" in datasets_to_process:
        try:
            results["gpqa"] = process_gpqa(output_dir)
        except Exception as e:
            print(f"✗ Error processing gpqa: {e}")

    # Process code datasets from Eurus
    code_datasets_to_process = [d for d in datasets_to_process if d in ["taco", "codecontests", "apps", "codeforces"]]
    if code_datasets_to_process:
        try:
            print(f"\nProcessing {len(code_datasets_to_process)} code dataset(s) from Eurus...")
            eurus_results = process_eurus_code_datasets(output_dir)
            # Only keep the requested datasets
            for name in code_datasets_to_process:
                if name in eurus_results:
                    results[name] = eurus_results[name]
        except Exception as e:
            print(f"✗ Error processing Eurus code datasets: {e}")

    # Summary
    print("\n" + "=" * 80)
    print("Processing Summary")
    print("=" * 80)

    for name, dataset in results.items():
        if dataset is not None:
            print(f"✓ {name}: {len(dataset)} examples")

    print(f"\nAll processed datasets saved to: {output_dir}")
    print("\nDirectory structure:")
    print(f"  {output_dir}/")
    print(f"    ├── deepscaler/")
    print(f"    │   ├── train.parquet")
    print(f"    │   └── train_example.json")
    print(f"    ├── dapo_math/")
    print(f"    │   ├── train.parquet")
    print(f"    │   └── train_example.json")
    print(f"    ├── amc23/")
    print(f"    │   ├── test.parquet")
    print(f"    │   └── test_example.json")
    print(f"    ├── aime_2024/")
    print(f"    │   ├── test.parquet")
    print(f"    │   └── test_example.json")
    print(f"    ├── aime25/")
    print(f"    │   ├── test.parquet")
    print(f"    │   └── test_example.json")
    print(f"    ├── math500/")
    print(f"    │   ├── test.parquet")
    print(f"    │   └── test_example.json")
    print(f"    ├── minervamath/")
    print(f"    │   ├── test.parquet")
    print(f"    │   └── test_example.json")
    print(f"    ├── olympiadbench/")
    print(f"    │   ├── test.parquet")
    print(f"    │   └── test_example.json")
    print(f"    ├── beyondaime/")
    print(f"    │   ├── test.parquet")
    print(f"    │   └── test_example.json")
    print(f"    ├── gpqa/")
    print(f"    │   ├── test.parquet")
    print(f"    │   └── test_example.json")
    print(f"    ├── taco/")
    print(f"    │   ├── test.parquet")
    print(f"    │   └── test_example.json")
    print(f"    ├── codecontests/")
    print(f"    │   ├── test.parquet")
    print(f"    │   └── test_example.json")
    print(f"    ├── apps/")
    print(f"    │   ├── test.parquet")
    print(f"    │   └── test_example.json")
    print(f"    ├── codeforces/")
    print(f"    │   ├── test.parquet")
    print(f"    │   └── test_example.json")
    print(f"    └── ifeval/")
    print(f"        ├── test.parquet")
    print(f"        └── test_example.json")


if __name__ == "__main__":
    main()
