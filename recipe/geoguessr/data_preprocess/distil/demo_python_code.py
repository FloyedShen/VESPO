#!/usr/bin/env python3
"""
Distillation script with Python code execution as tool.

Features:
- Model can write Python code to process images
- Code is executed locally with proper sandboxing
- Supports PIL image operations
- Safe execution with timeout and resource limits
"""

import os
import sys
import json
import base64
from io import BytesIO
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FutureTimeoutError
import threading
import traceback

import datasets
from tqdm import tqdm
from PIL import Image
import requests
import numpy as np

# Add reward function to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from reward_function import geoguessr_reward_function


# ============================================================================
# Configuration
# ============================================================================

API_BASE_URL = "http://10.146.229.25:80/v1"
MODEL_NAME = "nginx"

# Simple system prompt
SYSTEM_PROMPT = "You are a helpful assistant."

# User prompt with Python code tool
USER_PROMPT_WITH_CODE = """<image>

Where was this photo taken? Analyze the image and predict the location.

Consider clues like: architecture, vegetation/terrain, text/language, road signs/markings, vehicles/traffic direction, climate, cultural elements, and landmarks.

You can use Python code to process the image. Available tools:

**python_code_tool**: Execute Python code to process the image
- The image is available as `image` variable (PIL.Image object)
- You can use PIL (pillow) operations
- Return the processed image by assigning to `result_image`
- Example:
<tool_call>
{{
  "name": "python_code_tool",
  "code": "# Rotate and crop\\nresult_image = image.rotate(90).crop((100, 100, 500, 500))"
}}
</tool_call>

When ready, output the final answer as: <answer>\\boxed{{latitude, longitude}}</answer>"""


# ============================================================================
# Python Code Execution Sandbox
# ============================================================================

def execute_python_code(code: str, image: Image.Image, timeout: int = 10) -> Tuple[Optional[Image.Image], str]:
    """
    Execute Python code to process an image in a sandboxed environment.

    Args:
        code: Python code to execute
        image: Input PIL Image
        timeout: Execution timeout in seconds

    Returns:
        (result_image, message) - Processed image and status message
    """
    # Prepare safe globals
    safe_globals = {
        '__builtins__': {
            'abs': abs,
            'all': all,
            'any': any,
            'bin': bin,
            'bool': bool,
            'dict': dict,
            'enumerate': enumerate,
            'float': float,
            'int': int,
            'len': len,
            'list': list,
            'map': map,
            'max': max,
            'min': min,
            'range': range,
            'round': round,
            'sorted': sorted,
            'str': str,
            'sum': sum,
            'tuple': tuple,
            'zip': zip,
        },
        'Image': Image,
        'np': np,
        'image': image.copy(),  # Work on a copy
        'result_image': None,
    }

    try:
        # Execute with timeout
        import signal

        def timeout_handler(signum, frame):
            raise TimeoutError("Code execution timeout")

        # Set timeout (only works on Unix)
        if hasattr(signal, 'SIGALRM'):
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout)

        try:
            exec(code, safe_globals)
        finally:
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0)

        # Get result
        result_image = safe_globals.get('result_image')

        if result_image is None:
            return None, "Error: No result_image returned. Please assign the result to 'result_image' variable."

        if not isinstance(result_image, Image.Image):
            return None, f"Error: result_image must be a PIL Image, got {type(result_image)}"

        return result_image, "Success: Image processed successfully"

    except TimeoutError:
        return None, f"Error: Code execution timeout (>{timeout}s)"
    except Exception as e:
        error_msg = f"Error: {type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        return None, error_msg


# ============================================================================
# Helper Functions
# ============================================================================

def encode_image_to_base64(image: Image.Image) -> str:
    """Encode PIL Image to base64 string."""
    buffered = BytesIO()
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image.save(buffered, format="JPEG", quality=95)
    img_bytes = buffered.getvalue()
    return base64.b64encode(img_bytes).decode('utf-8')


def create_initial_messages(image: Image.Image) -> List[Dict[str, Any]]:
    """Create initial chat messages with image."""
    image_base64 = encode_image_to_base64(image)

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": USER_PROMPT_WITH_CODE
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_base64}"
                    }
                }
            ]
        }
    ]

    return messages


def call_qwen_api(
    messages: List[Dict[str, Any]],
    api_base: str = API_BASE_URL,
    model: str = MODEL_NAME,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    timeout: int = 120
) -> Optional[Dict[str, Any]]:
    """Call Qwen3-VL-235B-Thinking API."""
    url = f"{api_base}/chat/completions"

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False
    }

    try:
        response = requests.post(
            url,
            json=payload,
            timeout=timeout,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] API call failed: {e}")
        return None


def extract_tool_call(text: str) -> Optional[Dict[str, Any]]:
    """Extract tool call from model output."""
    import re
    tool_call_match = re.findall(r'<tool_call>(.*?)</tool_call>', text, re.DOTALL)
    if not tool_call_match:
        return None

    try:
        tool_call = json.loads(tool_call_match[-1].strip())
        return tool_call
    except json.JSONDecodeError:
        return None


def extract_answer(text: str) -> Optional[str]:
    """Extract final answer from model output."""
    import re
    answer_match = re.findall(r'<answer>(.*?)</answer>', text, re.DOTALL)
    if answer_match:
        return answer_match[-1].strip()
    return None


def calculate_reward_score(
    response_text: str,
    ground_truth_lat: float,
    ground_truth_lon: float
) -> Dict[str, Any]:
    """Calculate reward score based on predicted coordinates."""
    ground_truth = {"lat": ground_truth_lat, "lon": ground_truth_lon}

    result = geoguessr_reward_function(
        data_source="gaea",
        solution_str=response_text,
        ground_truth=ground_truth,
        extra_info={
            "reward_type": "official",
            "return_dict": True,
            "verbose": False
        }
    )

    return result


def process_single_sample_with_code(
    sample: Dict[str, Any],
    sample_index: int,
    dataset_path: str,
    max_turns: int = 10,
    temperature: float = 0.7,
    max_tokens: int = 2048
) -> Optional[Dict[str, Any]]:
    """
    Process a single sample with Python code tool support.
    """
    # Initialize
    current_image = sample['image']
    messages = create_initial_messages(current_image)

    # Track conversation
    conversation_log = []
    tool_calls_log = []
    turn_count = 0

    # Multi-turn conversation
    while turn_count < max_turns:
        turn_count += 1

        # Call API
        api_response = call_qwen_api(
            messages,
            temperature=temperature,
            max_tokens=max_tokens
        )

        if api_response is None:
            return None

        # Extract response
        try:
            response_text = api_response["choices"][0]["message"]["content"]
        except (KeyError, IndexError):
            return None

        # Log conversation
        conversation_log.append({
            "turn": turn_count,
            "response": response_text,
            "usage": api_response.get("usage", {})
        })

        # Check for tool call (Python code)
        tool_call = extract_tool_call(response_text)

        if tool_call:
            tool_name = tool_call.get("name")

            if tool_name == "python_code_tool":
                code = tool_call.get("code", "")

                # Save tool call (code only)
                tool_calls_log.append({
                    "turn": turn_count,
                    "tool_name": "python_code_tool",
                    "code": code,
                    "success": False  # Will be updated
                })

                # Execute code
                result_image, message = execute_python_code(code, current_image)

                if result_image is not None:
                    current_image = result_image
                    tool_calls_log[-1]["success"] = True
                    tool_calls_log[-1]["message"] = message
                    observation = f"<tool_response>{message}\n<image></tool_response>"
                else:
                    tool_calls_log[-1]["message"] = message
                    observation = f"<tool_response>{message}</tool_response>"

                # Add to messages
                messages.append({
                    "role": "assistant",
                    "content": response_text
                })

                if result_image is not None:
                    # Add new image
                    image_base64 = encode_image_to_base64(current_image)
                    messages.append({
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": observation
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}"
                                }
                            }
                        ]
                    })
                else:
                    # Error message only
                    messages.append({
                        "role": "user",
                        "content": observation
                    })

                continue

            else:
                # Unknown tool
                messages.append({
                    "role": "assistant",
                    "content": response_text
                })
                messages.append({
                    "role": "user",
                    "content": f"Error: Unknown tool '{tool_name}'"
                })
                continue

        # Check for final answer
        answer = extract_answer(response_text)
        if answer:
            # Calculate reward
            reward_score = calculate_reward_score(
                answer,
                sample['lat'],
                sample['lon']
            )

            # Prepare sample data
            sample_data = {k: v for k, v in sample.items() if k != 'image'}

            return {
                "dataset_path": dataset_path,
                "sample_index": sample_index,
                "sample_data": sample_data,
                "conversation_log": conversation_log,
                "tool_calls_log": tool_calls_log,
                "final_response": answer,
                "reward_score": reward_score,
                "metadata": {
                    "total_turns": turn_count,
                    "num_tool_calls": len(tool_calls_log),
                    "parse_success": reward_score.get("parse_success", False),
                    "distance_km": reward_score.get("distance@km", None),
                    "score": reward_score.get("score", None)
                }
            }

        # No tool call and no answer
        messages.append({
            "role": "assistant",
            "content": response_text
        })
        messages.append({
            "role": "user",
            "content": "Please continue your analysis or provide your final answer."
        })

    # Max turns reached
    reward_score = calculate_reward_score(
        response_text,
        sample['lat'],
        sample['lon']
    )

    sample_data = {k: v for k, v in sample.items() if k != 'image'}

    return {
        "dataset_path": dataset_path,
        "sample_index": sample_index,
        "sample_data": sample_data,
        "conversation_log": conversation_log,
        "tool_calls_log": tool_calls_log,
        "final_response": response_text,
        "reward_score": reward_score,
        "metadata": {
            "total_turns": turn_count,
            "num_tool_calls": len(tool_calls_log),
            "parse_success": reward_score.get("parse_success", False),
            "distance_km": reward_score.get("distance@km", None),
            "score": reward_score.get("score", None)
        }
    }


def process_sample_wrapper(args):
    """Wrapper for concurrent processing."""
    sample, sample_index, dataset_path, max_turns, temperature, max_tokens = args
    return sample_index, process_single_sample_with_code(
        sample, sample_index, dataset_path, max_turns, temperature, max_tokens
    )


def run_distillation_with_python_code(
    dataset_path: str,
    output_dir: str,
    num_samples: int = 10,
    max_turns: int = 10,
    temperature: float = 0.7,
    max_tokens: int = 2048,
    max_workers: int = 4
):
    """Run distillation with Python code tool support."""
    print("=" * 80)
    print("GeoGuessr Trace Distillation (Python Code Tool)")
    print("=" * 80)

    # Load dataset
    print(f"Loading dataset from: {dataset_path}")
    ds = datasets.load_from_disk(dataset_path)
    print(f"Dataset loaded: {len(ds)} samples")

    if 'locatability_score' in ds.column_names:
        print(f"Sorting by difficulty (hardest first)...")
        ds = ds.sort('locatability_score', reverse=True)

    if num_samples < len(ds):
        ds = ds.select(range(num_samples))

    print(f"\n{'=' * 80}")
    print(f"Processing {len(ds)} samples with {max_workers} concurrent workers")
    print(f"Output directory: {output_dir}")
    print(f"{'=' * 80}\n")

    os.makedirs(output_dir, exist_ok=True)

    # Prepare tasks
    tasks = [
        (ds[i], i, dataset_path, max_turns, temperature, max_tokens)
        for i in range(len(ds))
    ]

    # Results storage
    results = {}
    success_count = 0
    failed_samples = []

    # Progress tracking
    pbar = tqdm(total=len(tasks), desc="Generating traces")
    lock = threading.Lock()

    # Process concurrently
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(process_sample_wrapper, task): task[1]
            for task in tasks
        }

        for future in as_completed(future_to_index):
            sample_index = future_to_index[future]

            try:
                idx, trace = future.result()

                with lock:
                    if trace is not None:
                        results[idx] = trace
                        success_count += 1

                        reward = trace['reward_score']
                        tqdm.write(
                            f"[Sample {idx}] SUCCESS - "
                            f"Tools: {trace['metadata']['num_tool_calls']}, "
                            f"Score: {reward.get('score', 0):.4f}"
                        )
                    else:
                        failed_samples.append(idx)
                        tqdm.write(f"[Sample {idx}] FAILED")

                    pbar.update(1)

            except Exception as e:
                with lock:
                    failed_samples.append(sample_index)
                    tqdm.write(f"[Sample {sample_index}] ERROR: {e}")
                    pbar.update(1)

    pbar.close()

    # Save results in order
    print(f"\nSaving traces...")
    for idx in sorted(results.keys()):
        trace = results[idx]
        output_path = os.path.join(output_dir, f"trace_{idx:05d}.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(trace, f, indent=2, ensure_ascii=False)

    # Summary
    total_tool_calls = sum(
        trace['metadata']['num_tool_calls']
        for trace in results.values()
    )
    parse_success = sum(
        1 for trace in results.values()
        if trace['metadata']['parse_success']
    )

    print(f"\n{'=' * 80}")
    print("Summary")
    print(f"{'=' * 80}")
    print(f"Total: {len(ds)}, Success: {success_count}, Failed: {len(failed_samples)}")
    print(f"Parse success: {parse_success}/{success_count}")
    print(f"Tool calls: {total_tool_calls} (avg: {total_tool_calls/success_count if success_count > 0 else 0:.2f})")
    print(f"Output: {output_dir}")
    print(f"{'=' * 80}\n")


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Distillation with Python code execution tool"
    )

    parser.add_argument("--dataset_path", type=str,
        default="/mnt/tidal-alsh-hilab/usr/shenguobin/geoguessr/processed/gaea_wlp/train")
    parser.add_argument("--output_dir", type=str,
        default="/mnt/tidal-alsh-hilab/usr/shenguobin/verl/recipe/geoguessr/data_preprocess/distil/traces_python_code")
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--max_turns", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--max_workers", type=int, default=4)

    args = parser.parse_args()

    run_distillation_with_python_code(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        max_turns=args.max_turns,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_workers=args.max_workers
    )


if __name__ == "__main__":
    main()
