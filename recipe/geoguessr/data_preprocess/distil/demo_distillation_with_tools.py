#!/usr/bin/env python3
"""
Enhanced distillation script with tool support and reward scoring.

Features:
- Visual toolbox integration (zoom, rotate)
- Reward calculation based on coordinate accuracy
- Save dataset path + index instead of image
- Multi-turn conversations for tool calls
"""

import os
import sys
import json
import base64
from io import BytesIO
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

import datasets
from tqdm import tqdm
from PIL import Image
import requests

# Add reward function to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from reward_function import (
    geoguessr_reward_function,
    parse_coordinates_fallback,
    haversine_distance
)


# ============================================================================
# Configuration
# ============================================================================

API_BASE_URL = "http://10.146.229.25:80/v1"
MODEL_NAME = "nginx"

# System prompt with tool descriptions
SYSTEM_PROMPT = """You are an expert in geography and image analysis. Your task is to predict the geographical location (latitude and longitude) where a photo was taken.

Available tools:
1. **image_zoom_in_tool**: Zoom into a specific region of the image
   - Arguments: {"bbox_2d": [left, top, right, bottom]} (coordinates in pixels)
   - Use this to examine details like text on signs, architectural features, etc.

2. **image_rotate_tool**: Rotate the image for better viewing
   - Arguments: {"angle": degrees} (positive = counterclockwise)
   - Use this if the image orientation makes analysis difficult

To call a tool, use this format:
<tool_call>
{"name": "tool_name", "arguments": {...}}
</tool_call>

After the tool returns results, continue your analysis with the new image.

When you're ready to give your final answer, use:
<answer>
Based on my analysis... The location is \\boxed{latitude, longitude}
</answer>

Analyze the image carefully and look for clues such as:
- Architecture style and building materials
- Vegetation and landscape features
- Road signs, license plates, and text
- Climate indicators
- Cultural and regional markers
- Urban planning patterns

Think step by step and use tools when needed to gather more information."""

# User prompt template
USER_PROMPT_TEMPLATE = """<image>

Where was this photo taken? Analyze the image and predict the location.

Consider clues like: architecture, vegetation/terrain, text/language, road signs/markings, vehicles/traffic direction, climate, cultural elements, and landmarks.

Output the final answer as coordinates in $\\boxed{{latitude, longitude}}$ (decimal degrees)."""


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
                    "text": USER_PROMPT_TEMPLATE
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


def execute_tool(tool_call: Dict[str, Any], current_image: Image.Image) -> Tuple[Optional[Image.Image], str]:
    """
    Execute a tool call and return the new image and observation.

    Returns:
        (new_image, observation_text)
    """
    try:
        tool_name = tool_call["name"]
        args = tool_call["arguments"]

        if tool_name == "image_zoom_in_tool":
            bbox = args["bbox_2d"]
            # Validate bbox
            left, top, right, bottom = bbox
            if left >= right or top >= bottom:
                return None, f"Error: Invalid bbox coordinates {bbox}"

            # Crop image
            cropped = current_image.crop(bbox)
            return cropped, "<tool_response><image></tool_response>"

        elif tool_name == "image_rotate_tool":
            angle = args["angle"]
            rotated = current_image.rotate(angle, expand=True)
            return rotated, "<tool_response><image></tool_response>"

        else:
            return None, f"Error: Unknown tool '{tool_name}'"

    except Exception as e:
        return None, f"Error: Tool execution failed - {str(e)}"


def calculate_reward_score(
    response_text: str,
    ground_truth_lat: float,
    ground_truth_lon: float
) -> Dict[str, Any]:
    """
    Calculate reward score based on predicted coordinates.

    Returns:
        Dictionary with score, distance, and accuracy metrics
    """
    # Use the reward function from reward_function.py
    ground_truth = {"lat": ground_truth_lat, "lon": ground_truth_lon}

    result = geoguessr_reward_function(
        data_source="gaea",
        solution_str=response_text,
        ground_truth=ground_truth,
        extra_info={
            "reward_type": "official",  # Use official GeoGuessr scoring
            "return_dict": True,
            "verbose": False
        }
    )

    return result


def load_dataset_sorted_by_difficulty(
    dataset_path: str,
    max_samples: Optional[int] = None,
    ascending: bool = False
) -> datasets.Dataset:
    """Load dataset and sort by locatability_score (difficulty)."""
    print(f"Loading dataset from: {dataset_path}")
    ds = datasets.load_from_disk(dataset_path)

    print(f"Dataset loaded: {len(ds)} samples")

    if 'locatability_score' not in ds.column_names:
        print("[WARNING] No 'locatability_score' column found. Returning unsorted dataset.")
        return ds.select(range(min(max_samples or len(ds), len(ds))))

    # Sort by locatability_score
    print(f"Sorting by difficulty (ascending={ascending})...")
    ds = ds.sort('locatability_score', reverse=not ascending)

    # Print statistics
    scores = [s['locatability_score'] for s in ds.select(range(min(1000, len(ds))))]
    print(f"\nLocatability score statistics (first 1000 samples):")
    print(f"  Min: {min(scores):.4f}")
    print(f"  Max: {max(scores):.4f}")
    print(f"  Mean: {sum(scores)/len(scores):.4f}")

    # Limit samples if requested
    if max_samples is not None and max_samples < len(ds):
        print(f"Limiting to first {max_samples} samples")
        ds = ds.select(range(max_samples))

    return ds


def save_trace(
    output_path: str,
    dataset_path: str,
    sample_index: int,
    sample_data: Dict[str, Any],
    conversation_history: List[Dict[str, Any]],
    final_response: str,
    reward_score: Dict[str, Any],
    tool_calls: List[Dict[str, Any]]
):
    """
    Save the generated trace to a JSON file.

    Args:
        output_path: Path to save the trace
        dataset_path: Path to the source dataset
        sample_index: Index of the sample in the dataset
        sample_data: Sample metadata (without image)
        conversation_history: Full conversation including tool calls
        final_response: Final response text
        reward_score: Reward calculation results
        tool_calls: List of tool calls made
    """
    trace = {
        "dataset_path": dataset_path,
        "sample_index": sample_index,
        "sample_data": sample_data,
        "conversation_history": conversation_history,
        "final_response": final_response,
        "reward_score": reward_score,
        "tool_calls": tool_calls,
        "metadata": {
            "total_turns": len(conversation_history) // 2,  # Approximate
            "num_tool_calls": len(tool_calls),
            "parse_success": reward_score.get("parse_success", False),
            "distance_km": reward_score.get("distance@km", None),
            "score": reward_score.get("score", None)
        }
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(trace, f, indent=2, ensure_ascii=False)


# ============================================================================
# Main Distillation Function with Tool Support
# ============================================================================

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

    Returns:
        Trace dictionary or None if failed
    """
    # Initialize
    current_image = sample['image']
    messages = create_initial_messages(current_image)
    conversation_history = []
    tool_calls = []
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

        # Save to history
        conversation_history.append({
            "turn": turn_count,
            "messages": messages.copy(),
            "response": response_text,
            "usage": api_response.get("usage", {})
        })

        # Check if there's a tool call
        tool_call = extract_tool_call(response_text)

        if tool_call:
            # Execute tool
            new_image, observation = execute_tool(tool_call, current_image)

            if new_image is None:
                # Tool failed, add error message
                messages.append({
                    "role": "assistant",
                    "content": response_text
                })
                messages.append({
                    "role": "user",
                    "content": observation
                })
            else:
                # Tool succeeded, update image
                current_image = new_image
                tool_calls.append({
                    "turn": turn_count,
                    "tool_call": tool_call,
                    "success": True
                })

                # Add to messages
                messages.append({
                    "role": "assistant",
                    "content": response_text
                })

                # Add new image observation
                image_base64 = encode_image_to_base64(new_image)
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

        # Check if there's a final answer
        answer = extract_answer(response_text)
        if answer:
            # Calculate reward
            reward_score = calculate_reward_score(
                answer,
                sample['lat'],
                sample['lon']
            )

            # Prepare sample data (exclude image)
            sample_data = {k: v for k, v in sample.items() if k != 'image'}

            return {
                "dataset_path": dataset_path,
                "sample_index": sample_index,
                "sample_data": sample_data,
                "conversation_history": conversation_history,
                "final_response": answer,
                "reward_score": reward_score,
                "tool_calls": tool_calls
            }

        # No tool call and no answer, add to messages and continue
        messages.append({
            "role": "assistant",
            "content": response_text
        })
        messages.append({
            "role": "user",
            "content": "Please continue your analysis or provide your final answer using <answer>...</answer> tags."
        })

    # Reached max turns without answer
    # Try to calculate reward anyway
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
        "conversation_history": conversation_history,
        "final_response": response_text,
        "reward_score": reward_score,
        "tool_calls": tool_calls
    }


def run_distillation_with_tools(
    dataset_path: str,
    output_dir: str,
    num_samples: int = 10,
    max_turns: int = 10,
    temperature: float = 0.7,
    max_tokens: int = 2048
):
    """Run distillation with tool support."""
    print("=" * 80)
    print("GeoGuessr Trace Distillation (with Tool Support)")
    print("=" * 80)

    # Load dataset
    ds = load_dataset_sorted_by_difficulty(
        dataset_path,
        max_samples=num_samples,
        ascending=False  # Hard first
    )

    print(f"\n{'=' * 80}")
    print(f"Processing {len(ds)} samples (hardest first)")
    print(f"Output directory: {output_dir}")
    print(f"Max turns per sample: {max_turns}")
    print(f"{'=' * 80}\n")

    os.makedirs(output_dir, exist_ok=True)

    # Statistics
    success_count = 0
    failed_samples = []
    total_tool_calls = 0
    total_distance = 0
    parse_success_count = 0

    # Process samples
    for idx, sample in enumerate(tqdm(ds, desc="Generating traces")):
        print(f"\n[Sample {idx + 1}/{len(ds)}]")
        print(f"  Locatability score: {sample['locatability_score']:.4f}")
        print(f"  Ground truth: lat={sample['lat']:.4f}, lon={sample['lon']:.4f}")
        print(f"  Country: {sample.get('country', 'N/A')}")

        # Process sample
        trace = process_single_sample(
            sample,
            sample_index=idx,
            dataset_path=dataset_path,
            max_turns=max_turns,
            temperature=temperature,
            max_tokens=max_tokens
        )

        if trace is None:
            print(f"  [FAILED] Processing failed")
            failed_samples.append(idx)
            continue

        # Extract metrics
        reward = trace['reward_score']
        num_tools = len(trace['tool_calls'])
        total_tool_calls += num_tools

        print(f"  [SUCCESS] Generated trace")
        print(f"    Turns: {len(trace['conversation_history'])}")
        print(f"    Tool calls: {num_tools}")
        print(f"    Parse success: {reward.get('parse_success', False)}")
        if reward.get('parse_success'):
            parse_success_count += 1
            distance = reward.get('distance@km', 0)
            total_distance += distance
            print(f"    Distance: {distance:.2f} km")
            print(f"    Score: {reward.get('score', 0):.4f}")

        # Save trace
        output_path = os.path.join(output_dir, f"trace_{idx:05d}.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(trace, f, indent=2, ensure_ascii=False)
        print(f"    Saved to: {output_path}")

        success_count += 1

    # Summary
    print(f"\n{'=' * 80}")
    print("Distillation Summary")
    print(f"{'=' * 80}")
    print(f"Total samples: {len(ds)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {len(failed_samples)}")
    print(f"Parse success: {parse_success_count}/{success_count}")
    if parse_success_count > 0:
        print(f"Average distance: {total_distance/parse_success_count:.2f} km")
    print(f"Total tool calls: {total_tool_calls}")
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
        description="Enhanced distillation with tool support and reward scoring"
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
        default="/mnt/tidal-alsh-hilab/usr/shenguobin/verl/recipe/geoguessr/data_preprocess/distil/traces_with_tools",
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
        help="Maximum turns per sample (for tool calls)"
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

    args = parser.parse_args()

    run_distillation_with_tools(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        max_turns=args.max_turns,
        temperature=args.temperature,
        max_tokens=args.max_tokens
    )


if __name__ == "__main__":
    main()
