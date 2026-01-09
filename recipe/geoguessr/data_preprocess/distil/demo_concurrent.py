#!/usr/bin/env python3
"""
Concurrent distillation script with simplified prompts and efficient storage.

Features:
- Simple system prompt: "You are a helpful assistant."
- Standard user prompt format (as provided)
- Tool calls saved as parameters only (no intermediate images)
- Concurrent API calls for speed
- Proper ordering maintained
"""

import os
import sys
import json
import base64
from io import BytesIO
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import datasets
from tqdm import tqdm
from PIL import Image
import requests

# Add reward function to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from reward_function import geoguessr_reward_function


# ============================================================================
# Configuration
# ============================================================================

API_BASE_URL = "http://10.146.229.25:80/v1"
MODEL_NAME = "nginx"

# Base system prompt (simple as requested)
SYSTEM_PROMPT_BASE = "You are a helpful assistant."

# System prompt with tools (when API doesn't support tools parameter)
SYSTEM_PROMPT_WITH_TOOLS = """You are a helpful assistant.

# Tools
You may call one or more functions to assist with the user query.
You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type":"function","function":{"name":"image_zoom_in_tool","description":"Zoom in on a specific region of an image by cropping it based on a bounding box (bbox) and an optional object label.","parameters":{"type":"object","properties":{"bbox_2d":{"type":"array","items":{"type":"number"},"minItems":4,"maxItems":4,"description":"The bounding box of the region to zoom in, as [x1, y1, x2, y2], where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner."},"label":{"type":"string","description":"The name or label of the object in the specified bounding box (optional)."}},"required":["bbox_2d"]}}}
{"type":"function","function":{"name":"image_rotate_tool","description":"Rotate the image by a specified angle to correct orientation or get a better view.","parameters":{"type":"object","properties":{"angle":{"type":"number","description":"Rotation angle in degrees. Positive values rotate counter-clockwise."}},"required":["angle"]}}}
</tools>

# How to call a tool
Return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>

**Example**:
<tool_call>
{"name": "image_zoom_in_tool", "arguments": {"bbox_2d": [10, 20, 100, 200], "label": "object description"}}
</tool_call>

# Tool Usage Guidelines
You should **actively use the available tools** to assist your analysis when working with images:
- Use tools to examine regions of interest in greater detail
- Multiple tool calls are encouraged to gather comprehensive information
- Consider using tools before providing your final answer to ensure accuracy

The final answer MUST BE put in \\boxed{}."""

# User prompt template (without tools - tools in system prompt)
USER_PROMPT_BASE = """<image>

Where was this photo taken? Analyze the image and predict the location.

Consider clues like: architecture, vegetation/terrain, text/language, road signs/markings, vehicles/traffic direction, climate, cultural elements, and landmarks.

Output final answer as: $\\boxed{{latitude, longitude}}$ (decimal degrees)."""

# OpenAI-style tool definitions (for API if supported)
TOOLS_DEFINITION = [
    {
        "type": "function",
        "function": {
            "name": "image_zoom_in_tool",
            "description": "Zoom into a specific region of the image for detailed analysis. Useful for examining text, signs, architectural details, or any small features.",
            "parameters": {
                "type": "object",
                "properties": {
                    "bbox_2d": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Bounding box coordinates [left, top, right, bottom] in pixels. For example: [100, 100, 500, 500]"
                    }
                },
                "required": ["bbox_2d"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "image_rotate_tool",
            "description": "Rotate the image to correct orientation or get a better viewing angle. Useful when the image is tilted or upside down.",
            "parameters": {
                "type": "object",
                "properties": {
                    "angle": {
                        "type": "number",
                        "description": "Rotation angle in degrees. Positive values rotate counter-clockwise. For example: 90, -90, 180"
                    }
                },
                "required": ["angle"]
            }
        }
    }
]


# ============================================================================
# Helper Functions
# ============================================================================

# Global variable to cache API tools support status
_API_SUPPORTS_TOOLS = None

def test_api_tools_support(api_base: str = API_BASE_URL, model: str = MODEL_NAME) -> bool:
    """
    Test if the API supports OpenAI-style tools parameter.
    Result is cached globally to avoid repeated tests.
    """
    global _API_SUPPORTS_TOOLS

    if _API_SUPPORTS_TOOLS is not None:
        return _API_SUPPORTS_TOOLS

    print("Testing API tools support...")

    try:
        response = requests.post(
            f"{api_base}/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": "test"}],
                "max_tokens": 10,
                "tools": [{
                    "type": "function",
                    "function": {
                        "name": "test_tool",
                        "description": "test",
                        "parameters": {"type": "object", "properties": {}}
                    }
                }]
            },
            timeout=10
        )

        # If status is 200, tools are supported
        if response.status_code == 200:
            _API_SUPPORTS_TOOLS = True
            print("✅ API supports OpenAI tools parameter")
            return True
        else:
            _API_SUPPORTS_TOOLS = False
            print("❌ API does not support tools parameter (will use prompt-based approach)")
            return False

    except Exception as e:
        _API_SUPPORTS_TOOLS = False
        print(f"❌ API tools test failed: {e} (will use prompt-based approach)")
        return False


def encode_image_to_base64(image: Image.Image) -> str:
    """Encode PIL Image to base64 string."""
    buffered = BytesIO()
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image.save(buffered, format="JPEG", quality=95)
    img_bytes = buffered.getvalue()
    return base64.b64encode(img_bytes).decode('utf-8')


def create_initial_messages(image: Image.Image, use_api_tools: bool = False) -> List[Dict[str, Any]]:
    """
    Create initial chat messages with image.
    Adapts prompt based on whether API supports tools parameter.
    """
    image_base64 = encode_image_to_base64(image)

    # Choose system prompt based on API support
    if use_api_tools:
        # API supports tools - use simple system prompt (tools passed via API)
        system_prompt = SYSTEM_PROMPT_BASE
    else:
        # API doesn't support tools - include tool descriptions in system prompt
        system_prompt = SYSTEM_PROMPT_WITH_TOOLS

    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": USER_PROMPT_BASE
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
    timeout: int = 120,
    tools: Optional[List[Dict[str, Any]]] = None
) -> Optional[Dict[str, Any]]:
    """Call Qwen3-VL-235B-Thinking API with optional tools."""
    url = f"{api_base}/chat/completions"

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False
    }

    # Add tools if provided
    if tools:
        payload["tools"] = tools

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
    """
    Extract final answer from model output.
    Supports multiple formats: <answer>...</answer> or \\boxed{...}
    """
    import re

    # Try <answer> tag first
    answer_match = re.findall(r'<answer>(.*?)</answer>', text, re.DOTALL)
    if answer_match:
        return answer_match[-1].strip()

    # Try \boxed{} format (LaTeX style)
    boxed_match = re.findall(r'\$?\$?\\boxed\{([^}]+)\}\$?\$?', text, re.DOTALL)
    if boxed_match:
        return boxed_match[-1].strip()

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


def process_single_sample(
    sample: Dict[str, Any],
    sample_index: int,
    dataset_path: str,
    max_turns: int = 10,
    temperature: float = 0.7,
    max_tokens: int = 2048
) -> Optional[Dict[str, Any]]:
    """
    Process a single sample with multi-turn tool support.
    Automatically detects API tools support and adapts accordingly.
    """
    # Test API tools support (cached after first call)
    use_api_tools = test_api_tools_support()

    # Initialize
    current_image = sample['image']
    messages = create_initial_messages(current_image, use_api_tools=use_api_tools)

    # Save initial prompt and tools definition (only once)
    initial_prompt = {
        "system": SYSTEM_PROMPT_WITH_TOOLS if not use_api_tools else SYSTEM_PROMPT_BASE,
        "user": USER_PROMPT_BASE,
        "tools": TOOLS_DEFINITION,  # Always save tools definition
        "tools_mode": "api_parameter" if use_api_tools else "prompt_based"  # Record which mode was used
    }

    # Track conversation but don't save intermediate images
    conversation_log = []
    tool_calls_log = []
    turn_count = 0

    # Multi-turn conversation
    while turn_count < max_turns:
        turn_count += 1

        # Call API (with or without tools parameter based on support)
        api_response = call_qwen_api(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=TOOLS_DEFINITION if use_api_tools else None
        )

        if api_response is None:
            return None

        # Extract response
        try:
            response_text = api_response["choices"][0]["message"]["content"]
        except (KeyError, IndexError):
            return None

        # Log this turn's interaction (not the full message history)
        conversation_log.append({
            "turn": turn_count,
            "assistant_response": response_text,
            "usage": api_response.get("usage", {})
        })

        # Check for tool call
        tool_call = extract_tool_call(response_text)

        if tool_call:
            # Save tool call parameters only (not the image)
            tool_calls_log.append({
                "turn": turn_count,
                "tool_name": tool_call["name"],
                "tool_arguments": tool_call["arguments"],
                "success": True  # Will be updated if execution fails
            })

            # Execute tool locally (but don't save result image)
            try:
                tool_name = tool_call["name"]
                args = tool_call["arguments"]

                if tool_name == "image_zoom_in_tool":
                    bbox = args["bbox_2d"]
                    current_image = current_image.crop(bbox)
                    observation = "<tool_response><image></tool_response>"

                elif tool_name == "image_rotate_tool":
                    angle = args["angle"]
                    current_image = current_image.rotate(angle, expand=True)
                    observation = "<tool_response><image></tool_response>"

                else:
                    observation = f"Error: Unknown tool '{tool_name}'"
                    tool_calls_log[-1]["success"] = False

            except Exception as e:
                observation = f"Error: Tool execution failed - {str(e)}"
                tool_calls_log[-1]["success"] = False

            # Add assistant response and new observation
            messages.append({
                "role": "assistant",
                "content": response_text
            })

            # Add new image observation
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

            continue

        # Check for final answer
        answer = extract_answer(response_text)
        if answer:
            # Calculate reward (pass full response_text for proper parsing)
            reward_score = calculate_reward_score(
                response_text,  # Pass full text with \boxed{} format
                sample['lat'],
                sample['lon']
            )

            # Prepare sample data (exclude image)
            sample_data = {k: v for k, v in sample.items() if k != 'image'}

            return {
                "dataset_path": dataset_path,
                "sample_index": sample_index,
                "sample_data": sample_data,
                "initial_prompt": initial_prompt,  # Save initial prompt once
                "conversation_log": conversation_log,
                "tool_calls_log": tool_calls_log,
                "final_response": response_text,  # Save full response
                "reward_score": reward_score,
                "metadata": {
                    "total_turns": turn_count,
                    "num_tool_calls": len(tool_calls_log),
                    "parse_success": reward_score.get("parse_success", False),
                    "distance_km": reward_score.get("distance@km", None),
                    "score": reward_score.get("score", None)
                }
            }

        # No tool call and no answer, prompt to continue
        messages.append({
            "role": "assistant",
            "content": response_text
        })
        messages.append({
            "role": "user",
            "content": "Please continue your analysis or provide your final answer using <answer>...</answer> tags."
        })

    # Reached max turns without answer
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
        "initial_prompt": initial_prompt,  # Save initial prompt once
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
    """Wrapper for multiprocessing."""
    sample, sample_index, dataset_path, max_turns, temperature, max_tokens = args
    return sample_index, process_single_sample(
        sample, sample_index, dataset_path, max_turns, temperature, max_tokens
    )


def run_concurrent_distillation(
    dataset_path: str,
    output_dir: str,
    num_samples: int = 10,
    max_turns: int = 10,
    temperature: float = 0.7,
    max_tokens: int = 2048,
    max_workers: int = 4
):
    """Run distillation with concurrent API calls."""
    print("=" * 80)
    print("Concurrent GeoGuessr Trace Distillation")
    print("=" * 80)

    # Load dataset
    print(f"Loading dataset from: {dataset_path}")
    ds = datasets.load_from_disk(dataset_path)
    print(f"Dataset loaded: {len(ds)} samples")

    # Handle sorting with caching for large datasets
    if 'locatability_score' in ds.column_names:
        # Generate cache file path based on dataset path
        import hashlib
        dataset_hash = hashlib.md5(dataset_path.encode()).hexdigest()[:8]
        cache_file = os.path.join(dataset_path, f".sorted_indices_{dataset_hash}.npy")

        try:
            # Try to load cached sorted indices
            if os.path.exists(cache_file):
                print(f"Loading cached sorted indices from: {cache_file}")
                import numpy as np
                sorted_indices = np.load(cache_file)
                print(f"Loaded {len(sorted_indices)} sorted indices from cache")

                # Select samples using cached indices
                if num_samples < len(sorted_indices):
                    ds = ds.select(sorted_indices[:num_samples].tolist())
                else:
                    ds = ds.select(sorted_indices.tolist())
            else:
                # First time: sort and cache indices
                print(f"Sorting {len(ds)} samples by difficulty (this may take a while on first run)...")
                import numpy as np

                # Extract scores and get sorted indices
                scores = np.array(ds['locatability_score'])
                sorted_indices = np.argsort(scores)[::-1]  # Descending order

                # Save indices to cache
                print(f"Saving sorted indices to cache: {cache_file}")
                np.save(cache_file, sorted_indices)
                print(f"Cache saved! Next run will be much faster.")

                # Select samples
                if num_samples < len(sorted_indices):
                    ds = ds.select(sorted_indices[:num_samples].tolist())
                else:
                    ds = ds.select(sorted_indices.tolist())

        except Exception as e:
            print(f"Warning: Failed to use cache ({e}), falling back to regular sort")
            ds = ds.sort('locatability_score', reverse=True)
            if num_samples < len(ds):
                ds = ds.select(range(num_samples))
    elif num_samples < len(ds):
        # Random selection if no locatability_score
        ds = ds.select(range(num_samples))

    print(f"\n{'=' * 80}")
    print(f"Processing {len(ds)} samples with {max_workers} concurrent workers")
    print(f"Output directory: {output_dir}")
    print(f"Max turns per sample: {max_turns}")
    print(f"{'=' * 80}\n")

    os.makedirs(output_dir, exist_ok=True)

    # Prepare tasks
    tasks = [
        (ds[i], i, dataset_path, max_turns, temperature, max_tokens)
        for i in range(len(ds))
    ]

    # Statistics
    results = {}  # sample_index -> trace
    success_count = 0
    failed_samples = []

    # Progress tracking
    pbar = tqdm(total=len(tasks), desc="Generating traces")
    lock = threading.Lock()

    # Process concurrently
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_index = {
            executor.submit(process_sample_wrapper, task): task[1]
            for task in tasks
        }

        # Collect results as they complete
        for future in as_completed(future_to_index):
            sample_index = future_to_index[future]

            try:
                idx, trace = future.result()

                with lock:
                    if trace is not None:
                        results[idx] = trace
                        success_count += 1

                        # Log progress
                        reward = trace['reward_score']
                        tqdm.write(
                            f"[Sample {idx}] SUCCESS - "
                            f"Turns: {trace['metadata']['num_tool_calls']}, "
                            f"Distance: {reward.get('distance@km', 'N/A'):.2f} km, "
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
    print(f"\nSaving traces in order...")
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

    if parse_success > 0:
        total_distance = sum(
            trace['reward_score'].get('distance@km', 0)
            for trace in results.values()
            if trace['metadata']['parse_success']
        )
        avg_distance = total_distance / parse_success
    else:
        avg_distance = 0

    print(f"\n{'=' * 80}")
    print("Distillation Summary")
    print(f"{'=' * 80}")
    print(f"Total samples: {len(ds)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {len(failed_samples)}")
    print(f"Parse success: {parse_success}/{success_count}")
    if parse_success > 0:
        print(f"Average distance: {avg_distance:.2f} km")
    print(f"Total tool calls: {total_tool_calls}")
    if success_count > 0:
        print(f"Average tool calls per sample: {total_tool_calls/success_count:.2f}")
    if failed_samples:
        print(f"Failed sample indices: {failed_samples}")
    print(f"Output directory: {output_dir}")
    print(f"{'=' * 80}\n")


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Concurrent distillation with efficient storage"
    )

    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/mnt/tidal-alsh-hilab/usr/shenguobin/geoguessr/processed/gaea_wlp/train",
        help="Path to dataset with locatability scores"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="/mnt/tidal-alsh-hilab/usr/shenguobin/verl/recipe/geoguessr/data_preprocess/distil/traces_concurrent",
        help="Directory to save generated traces"
    )

    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="Number of samples to process"
    )

    parser.add_argument(
        "--max_turns",
        type=int,
        default=10,
        help="Maximum turns per sample"
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature"
    )

    parser.add_argument(
        "--max_tokens",
        type=int,
        default=2048,
        help="Maximum tokens per turn"
    )

    parser.add_argument(
        "--max_workers",
        type=int,
        default=4,
        help="Number of concurrent workers"
    )

    args = parser.parse_args()

    run_concurrent_distillation(
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
