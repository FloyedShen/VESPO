#!/usr/bin/env python3
"""
Demo script for distilling high-quality think & tool traces from Qwen3-VL-235B-Thinking.

This script:
1. Loads the dataset with locatability scores
2. Sorts by difficulty (hardest first, lowest locatability_score)
3. Calls Qwen3-VL-235B-Thinking API to generate traces
4. Uses visual toolbox for tool calls
5. Saves the generated traces
"""

import os
import json
import base64
from io import BytesIO
from typing import Dict, List, Optional, Any
from pathlib import Path

import datasets
from tqdm import tqdm
from PIL import Image
import requests


# ============================================================================
# Configuration
# ============================================================================

API_BASE_URL = "http://10.146.229.25:80/v1"
MODEL_NAME = "nginx"

# System prompt for geolocation task
SYSTEM_PROMPT = """You are an expert in geography and image analysis. Your task is to predict the geographical location (latitude and longitude) where a photo was taken.

Analyze the image carefully and look for clues such as:
- Architecture style and building materials
- Vegetation and landscape features
- Road signs, license plates, and text
- Climate indicators
- Cultural and regional markers
- Urban planning patterns

Think step by step and use available tools to gather information. Finally, provide your answer in the format:
\\boxed{latitude, longitude}

For example: \\boxed{40.7128, -74.0060}"""

USER_PROMPT = "Where was this photo taken? Please predict the latitude and longitude."


# ============================================================================
# Helper Functions
# ============================================================================

def encode_image_to_base64(image: Image.Image) -> str:
    """
    Encode PIL Image to base64 string.

    Args:
        image: PIL Image object

    Returns:
        Base64 encoded string
    """
    buffered = BytesIO()
    # Convert to RGB if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image.save(buffered, format="JPEG", quality=95)
    img_bytes = buffered.getvalue()
    return base64.b64encode(img_bytes).decode('utf-8')


def create_chat_messages(image: Image.Image, system_prompt: str, user_prompt: str) -> List[Dict[str, Any]]:
    """
    Create OpenAI-style chat messages with image.

    Args:
        image: PIL Image object
        system_prompt: System instruction
        user_prompt: User question

    Returns:
        List of message dictionaries
    """
    # Encode image
    image_base64 = encode_image_to_base64(image)

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
                    "text": user_prompt
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
    """
    Call Qwen3-VL-235B-Thinking API.

    Args:
        messages: OpenAI-style chat messages
        api_base: API base URL
        model: Model name
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        timeout: Request timeout in seconds

    Returns:
        API response dictionary or None if failed
    """
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


def extract_response_content(api_response: Dict[str, Any]) -> Optional[str]:
    """
    Extract the assistant's response content from API response.

    Args:
        api_response: API response dictionary

    Returns:
        Response text or None if extraction failed
    """
    try:
        return api_response["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as e:
        print(f"[ERROR] Failed to extract response content: {e}")
        return None


def load_dataset_sorted_by_difficulty(
    dataset_path: str,
    max_samples: Optional[int] = None,
    ascending: bool = False
) -> datasets.Dataset:
    """
    Load dataset and sort by locatability_score (difficulty).

    Args:
        dataset_path: Path to HuggingFace dataset
        max_samples: Maximum number of samples to load (None = all)
        ascending: If True, sort ascending (easy first); if False, descending (hard first)

    Returns:
        Sorted dataset
    """
    print(f"Loading dataset from: {dataset_path}")
    ds = datasets.load_from_disk(dataset_path)

    print(f"Dataset loaded: {len(ds)} samples")
    print(f"Columns: {ds.column_names}")

    # Check if locatability_score exists
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
    sample_data: Dict[str, Any],
    api_response: Dict[str, Any],
    response_text: str
):
    """
    Save the generated trace to a JSON file.

    Args:
        output_path: Path to save the trace
        sample_data: Original sample data (without image)
        api_response: Full API response
        response_text: Extracted response text
    """
    trace = {
        "sample_data": sample_data,
        "api_response": api_response,
        "response_text": response_text
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(trace, f, indent=2, ensure_ascii=False)


# ============================================================================
# Main Distillation Function
# ============================================================================

def run_distillation_demo(
    dataset_path: str,
    output_dir: str,
    num_samples: int = 10,
    temperature: float = 0.7,
    max_tokens: int = 4096
):
    """
    Run distillation demo on a few samples.

    Args:
        dataset_path: Path to dataset with locatability scores
        output_dir: Directory to save generated traces
        num_samples: Number of samples to process
        temperature: Sampling temperature for API
        max_tokens: Maximum tokens to generate
    """
    print("=" * 80)
    print("GeoGuessr Trace Distillation Demo")
    print("=" * 80)

    # Load dataset sorted by difficulty (hardest first)
    ds = load_dataset_sorted_by_difficulty(
        dataset_path,
        max_samples=num_samples,
        ascending=False  # Hard first
    )

    print(f"\n{'=' * 80}")
    print(f"Processing {len(ds)} samples (hardest first)")
    print(f"Output directory: {output_dir}")
    print(f"{'=' * 80}\n")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Process samples
    success_count = 0
    failed_samples = []

    for idx, sample in enumerate(tqdm(ds, desc="Generating traces")):
        print(f"\n[Sample {idx + 1}/{len(ds)}]")
        print(f"  Locatability score: {sample['locatability_score']:.4f}")
        print(f"  Ground truth: lat={sample['lat']:.4f}, lon={sample['lon']:.4f}")
        print(f"  Country: {sample.get('country', 'N/A')}")

        # Create messages
        messages = create_chat_messages(
            sample['image'],
            SYSTEM_PROMPT,
            USER_PROMPT
        )

        # Call API
        print(f"  Calling API...")
        api_response = call_qwen_api(
            messages,
            temperature=temperature,
            max_tokens=max_tokens
        )

        if api_response is None:
            print(f"  [FAILED] API call failed")
            failed_samples.append(idx)
            continue

        # Extract response
        response_text = extract_response_content(api_response)
        if response_text is None:
            print(f"  [FAILED] Could not extract response")
            failed_samples.append(idx)
            continue

        print(f"  [SUCCESS] Generated response ({len(response_text)} chars)")
        print(f"  Response preview: {response_text[:150]}...")

        # Prepare sample data (exclude image for JSON serialization)
        sample_data = {k: v for k, v in sample.items() if k != 'image'}

        # Save trace
        output_path = os.path.join(output_dir, f"trace_{idx:05d}.json")
        save_trace(output_path, sample_data, api_response, response_text)
        print(f"  Saved to: {output_path}")

        success_count += 1

    # Summary
    print(f"\n{'=' * 80}")
    print("Distillation Summary")
    print(f"{'=' * 80}")
    print(f"Total samples: {len(ds)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {len(failed_samples)}")
    if failed_samples:
        print(f"Failed sample indices: {failed_samples}")
    print(f"Output directory: {output_dir}")
    print(f"{'=' * 80}\n")


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main entry point for the demo."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Demo script for distilling GeoGuessr traces from Qwen3-VL-235B-Thinking"
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
        default="/mnt/tidal-alsh-hilab/usr/shenguobin/verl/recipe/geoguessr/data_preprocess/distil/traces_demo",
        help="Directory to save generated traces"
    )

    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="Number of samples to process (default: 10 for demo)"
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)"
    )

    parser.add_argument(
        "--max_tokens",
        type=int,
        default=4096,
        help="Maximum tokens to generate (default: 4096)"
    )

    parser.add_argument(
        "--api_base",
        type=str,
        default=API_BASE_URL,
        help=f"API base URL (default: {API_BASE_URL})"
    )

    parser.add_argument(
        "--model",
        type=str,
        default=MODEL_NAME,
        help=f"Model name (default: {MODEL_NAME})"
    )

    args = parser.parse_args()

    # Run demo (API config will be passed via function params or read from args)
    run_distillation_demo(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        temperature=args.temperature,
        max_tokens=args.max_tokens
    )


if __name__ == "__main__":
    main()
