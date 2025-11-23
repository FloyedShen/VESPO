"""
Convert verl format datasets to tokenized format with vision preprocessing

This script takes the OUTPUT of convert_plain_rlvr.py (verl format with image bytes)
and adds:
1. Pre-processed vision features (pixel_values, image_grid_thw, etc.)
2. Pre-computed tokenization (as cache)
3. Optional prompt modification

This allows fast loading at training time while maintaining prompt flexibility.
"""

import os
import glob
import pickle
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import argparse
from datasets import load_dataset, Dataset as HFDataset
from PIL import Image
from tqdm import tqdm
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import io
import torch
import numpy as np
import gc

# Disable multiprocessing in various libraries to avoid process explosion
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')


def serialize_tensor(tensor: Any, compress_float32: bool = False) -> bytes:
    """
    Serialize tensor or numpy array to bytes using pickle

    Args:
        tensor: Tensor or numpy array to serialize
        compress_float32: If True, convert float32 to float16 to save 50% storage
    """
    if tensor is None:
        return pickle.dumps(None)

    if isinstance(tensor, torch.Tensor):
        # Convert to numpy for more efficient storage
        arr = tensor.cpu().numpy()
    elif isinstance(tensor, np.ndarray):
        arr = tensor
    else:
        return pickle.dumps(tensor)

    # Optionally compress float32 to float16 (50% size reduction)
    if compress_float32 and arr.dtype == np.float32:
        arr = arr.astype(np.float16)

    return pickle.dumps(arr)


def load_image(image_data: Any) -> Image.Image:
    """Load PIL Image from various formats"""
    if isinstance(image_data, Image.Image):
        return image_data
    elif isinstance(image_data, dict) and 'bytes' in image_data:
        return Image.open(io.BytesIO(image_data['bytes']))
    elif isinstance(image_data, bytes):
        return Image.open(io.BytesIO(image_data))
    else:
        raise ValueError(f"Unexpected image format: {type(image_data)}")


def process_single_sample(
    sample: Dict[str, Any],
    processor: Any,
    tokenizer: Any,
    custom_system_prompt: Optional[str] = None,
    custom_user_prompt: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Process a single verl format sample to add vision preprocessing and tokenization cache

    Input sample should have (from convert_plain_rlvr.py):
    - data_source, prompt (list of messages), images (bytes or list of bytes)
    - ability, reward_model, extra_info

    Returns sample with added fields:
    - vision_data: Pre-processed vision features
    - default_tokenization: Cached tokenization
    """
    # Get original prompt messages
    original_messages = sample.get('prompt', [])

    # Apply custom prompts if provided
    if custom_system_prompt is not None or custom_user_prompt is not None:
        messages = []

        # Handle system prompt
        if custom_system_prompt is not None:
            messages.append({"role": "system", "content": custom_system_prompt})
        else:
            # Keep original system prompt if exists
            for msg in original_messages:
                if msg.get("role") == "system":
                    messages.append(msg)
                    break

        # Handle user prompt
        if custom_user_prompt is not None:
            # Ensure <image> token is included
            if "<image>" not in custom_user_prompt:
                custom_user_prompt = "<image>\n\n" + custom_user_prompt
            messages.append({"role": "user", "content": custom_user_prompt})
        else:
            # Keep original user prompt
            for msg in original_messages:
                if msg.get("role") == "user":
                    messages.append(msg)
                    break
    else:
        # No custom prompts - use original
        messages = original_messages

    # Load image from bytes or PIL Image
    images_data = sample.get('images')
    if images_data is None:
        raise ValueError("Sample missing 'images' field")

    # Handle both single image and list of images
    if isinstance(images_data, list):
        if len(images_data) == 0:
            raise ValueError("Empty images list")
        image_data = images_data[0]  # Take first image
    else:
        image_data = images_data

    # Load image using helper function
    image = load_image(image_data)

    # Apply chat template to get raw prompt
    raw_prompt = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )

    # Process with processor (gets both text tokens and image features)
    model_inputs = processor(
        text=[raw_prompt],
        images=[image],
        return_tensors="pt"
    )

    # Extract vision preprocessing results
    vision_data = {}

    # Standard fields
    if "pixel_values" in model_inputs:
        # Compress pixel_values from float32 to float16 to save 50% storage
        vision_data["pixel_values"] = serialize_tensor(model_inputs["pixel_values"].squeeze(0), compress_float32=True)

    # Qwen2-VL specific fields
    if "image_grid_thw" in model_inputs:
        vision_data["image_grid_thw"] = serialize_tensor(model_inputs["image_grid_thw"])
    if "video_grid_thw" in model_inputs:
        vision_data["video_grid_thw"] = serialize_tensor(model_inputs["video_grid_thw"])
    if "second_per_grid_ts" in model_inputs:
        vision_data["second_per_grid_ts"] = serialize_tensor(model_inputs["second_per_grid_ts"])

    # Pre-compute default tokenization (as cache)
    default_tokenization = {
        "input_ids": serialize_tensor(model_inputs["input_ids"].squeeze(0)),
        "attention_mask": serialize_tensor(model_inputs["attention_mask"].squeeze(0)),
    }

    # Build output sample - keep all original fields and add new ones
    output_sample = dict(sample)  # Copy all original fields

    # Update 'prompt' field to reflect the messages used for tokenization
    # This ensures consistency between prompt field and tokenized content
    output_sample["prompt"] = messages

    # Remove 'images' field since we now have preprocessed vision_data
    if 'images' in output_sample:
        del output_sample['images']

    # Add vision preprocessing data
    output_sample["vision_data"] = vision_data

    # Add default tokenization cache
    output_sample["default_tokenization"] = default_tokenization

    return output_sample


def process_batch_samples(
    samples: List[Dict[str, Any]],
    processor: Any,
    tokenizer: Any,
    custom_system_prompt: Optional[str] = None,
    custom_user_prompt: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Process multiple samples in batch for GPU acceleration

    This function batches the vision preprocessing to utilize GPU more efficiently.
    It processes multiple images together, which is much faster than processing them one by one.
    """
    if len(samples) == 0:
        return []

    # Prepare batch data
    batch_messages = []
    batch_images = []
    batch_samples = []

    for sample in samples:
        # Get original prompt messages
        original_messages = sample.get('prompt', [])

        # Apply custom prompts if provided
        if custom_system_prompt is not None or custom_user_prompt is not None:
            messages = []

            # Handle system prompt
            if custom_system_prompt is not None:
                messages.append({"role": "system", "content": custom_system_prompt})
            else:
                for msg in original_messages:
                    if msg.get("role") == "system":
                        messages.append(msg)
                        break

            # Handle user prompt
            if custom_user_prompt is not None:
                if "<image>" not in custom_user_prompt:
                    custom_user_prompt = "<image>\n\n" + custom_user_prompt
                messages.append({"role": "user", "content": custom_user_prompt})
            else:
                for msg in original_messages:
                    if msg.get("role") == "user":
                        messages.append(msg)
                        break
        else:
            messages = original_messages

        # Load image
        images_data = sample.get('images')
        if images_data is None:
            continue  # Skip samples without images

        if isinstance(images_data, list):
            if len(images_data) == 0:
                continue
            image_data = images_data[0]
        else:
            image_data = images_data

        try:
            image = load_image(image_data)
            batch_messages.append(messages)
            batch_images.append(image)
            batch_samples.append(sample)
        except Exception as e:
            print(f"  Warning: Failed to load image for sample: {e}")
            continue

    if len(batch_images) == 0:
        return []

    # Batch process: Apply chat template for all samples
    batch_raw_prompts = []
    for messages in batch_messages:
        raw_prompt = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        batch_raw_prompts.append(raw_prompt)

    # Batch process: Vision preprocessing + tokenization for all samples at once
    # This is the key optimization - process all images together on GPU
    model_inputs = processor(
        text=batch_raw_prompts,
        images=batch_images,
        return_tensors="pt"
    )

    # Extract results for each sample
    # IMPORTANT: For Qwen2-VL, pixel_values is (total_patches, hidden_dim), not (batch_size, patches, hidden_dim)
    # We need to split it based on image_grid_thw
    output_samples = []

    # Calculate patches per image from image_grid_thw
    patches_per_image = []
    if "image_grid_thw" in model_inputs and "pixel_values" in model_inputs:
        grid_thw = model_inputs["image_grid_thw"]  # (batch_size, 3) where 3 = [t, h, w]
        for i in range(len(batch_samples)):
            t, h, w = grid_thw[i]
            num_patches = int(t * h * w)
            patches_per_image.append(num_patches)

        # Split pixel_values according to patches_per_image
        pixel_values_list = []
        start_idx = 0
        for num_patches in patches_per_image:
            end_idx = start_idx + num_patches
            pixel_values_list.append(model_inputs["pixel_values"][start_idx:end_idx])
            start_idx = end_idx
    else:
        pixel_values_list = [None] * len(batch_samples)

    for i, sample in enumerate(batch_samples):
        # Extract vision data for this sample
        vision_data = {}

        if pixel_values_list[i] is not None:
            # Compress pixel_values from float32 to float16 to save 50% storage
            vision_data["pixel_values"] = serialize_tensor(pixel_values_list[i], compress_float32=True)

        if "image_grid_thw" in model_inputs:
            # image_grid_thw is per-image with shape (batch_size, 3)
            if model_inputs["image_grid_thw"].shape[0] == len(batch_samples):
                vision_data["image_grid_thw"] = serialize_tensor(model_inputs["image_grid_thw"][i])
            else:
                # Fallback: if only one grid_thw for all images
                vision_data["image_grid_thw"] = serialize_tensor(model_inputs["image_grid_thw"])

        if "video_grid_thw" in model_inputs:
            vision_data["video_grid_thw"] = serialize_tensor(model_inputs["video_grid_thw"])

        if "second_per_grid_ts" in model_inputs:
            vision_data["second_per_grid_ts"] = serialize_tensor(model_inputs["second_per_grid_ts"])

        # Extract tokenization for this sample
        default_tokenization = {
            "input_ids": serialize_tensor(model_inputs["input_ids"][i]),
            "attention_mask": serialize_tensor(model_inputs["attention_mask"][i]),
        }

        # Build output sample
        output_sample = dict(sample)
        output_sample["prompt"] = batch_messages[i]

        if 'images' in output_sample:
            del output_sample['images']

        output_sample["vision_data"] = vision_data
        output_sample["default_tokenization"] = default_tokenization

        output_samples.append(output_sample)

    return output_samples


def process_batch(args: Tuple) -> List[Tuple[bool, Any, int]]:
    """
    Process a batch of samples (runs in worker process)

    NOTE: Each worker loads its own copy of the dataset, processor, and tokenizer
    """
    from transformers import AutoProcessor, AutoTokenizer

    input_file, batch_indices, model_name, custom_system_prompt, custom_user_prompt = args

    # Load dataset in worker process (avoids pickling large objects)
    dataset = load_dataset('parquet', data_files=input_file, split='train')

    # Load processor and tokenizer in worker
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    results = []
    for idx in batch_indices:
        try:
            # Load sample in worker process
            sample = dataset[idx]
            processed_sample = process_single_sample(
                sample=sample,
                processor=processor,
                tokenizer=tokenizer,
                custom_system_prompt=custom_system_prompt,
                custom_user_prompt=custom_user_prompt,
            )
            results.append((True, processed_sample, idx))
        except Exception as e:
            import traceback
            error_msg = f"{str(e)}\n{traceback.format_exc()}"
            results.append((False, error_msg, idx))

    return results


def find_parquet_files(input_path: str) -> List[str]:
    """Find all parquet files in input path (directory or single file)"""
    if os.path.isfile(input_path):
        if input_path.endswith('.parquet'):
            return [input_path]
        else:
            raise ValueError(f"Input file must be .parquet: {input_path}")
    elif os.path.isdir(input_path):
        parquet_files = glob.glob(os.path.join(input_path, "*.parquet"))
        if not parquet_files:
            raise ValueError(f"No .parquet files found in directory: {input_path}")
        return sorted(parquet_files)
    else:
        raise ValueError(f"Input path does not exist: {input_path}")


def convert_dataset(
    input_path: str,
    model_name: str,
    custom_system_prompt: Optional[str] = None,
    custom_user_prompt: Optional[str] = None,
    output_path: Optional[str] = None,
    max_samples: int = -1,
    n_workers: int = 1,
    batch_size: int = 8,
    skip_existing: bool = True,
    file_slice: Optional[str] = None,
    gpu_id: Optional[int] = None,
):
    """
    Convert verl format dataset to tokenized format with vision preprocessing

    Args:
        input_path: Path to input directory or .parquet file (output of convert_plain_rlvr.py)
        model_name: HuggingFace model name (e.g., "Qwen/Qwen2.5-VL-7B-Instruct")
        custom_system_prompt: Optional custom system prompt (None = keep original)
        custom_user_prompt: Optional custom user prompt (None = keep original)
        output_path: Output directory (None = use default)
        max_samples: Max samples per file (-1 for all)
        n_workers: Number of workers (1 for sequential, >1 for parallel)
        batch_size: Samples per batch
        skip_existing: Skip files that already exist
        file_slice: Optional file slice (e.g., "0::3", "1::3", "2::3") for manual parallelization
        gpu_id: Optional GPU ID to use (None = use default)
    """
    # Set GPU device if specified
    if gpu_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        print(f"Using GPU {gpu_id}")

    # Set PyTorch thread count to 1 for single-threaded operation
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    print(f"\n{'='*70}")
    print(f"CONVERT VERL FORMAT TO TOKENIZED FORMAT")
    print(f"{'='*70}")
    print(f"Input: {input_path}")
    print(f"Model: {model_name}")

    # Determine output path
    if output_path is None:
        # Use default: $GEOGUESSR_DIR/verl_data/plain_rlvr_{model_name_short}
        geoguessr_dir = os.environ.get('GEOGUESSR_DIR', '/mnt/tidal-alsh-hilab/usr/shenguobin/geoguessr')
        model_name_short = model_name.split('/')[-1].lower().replace('-', '_').replace('.', '_')
        output_path = os.path.join(geoguessr_dir, 'verl_data', f'plain_rlvr_{model_name_short}')
        print(f"Output (default): {output_path}")
    else:
        print(f"Output: {output_path}")

    # Create output directory
    os.makedirs(output_path, exist_ok=True)

    # Show prompt configuration
    if custom_system_prompt is not None:
        print(f"Custom system prompt: {custom_system_prompt[:80]}...")
    else:
        print(f"Custom system prompt: None (keep original)")

    if custom_user_prompt is not None:
        print(f"Custom user prompt: {custom_user_prompt[:80]}...")
    else:
        print(f"Custom user prompt: None (keep original)")

    print(f"Workers: {n_workers}, Batch size: {batch_size}")
    print(f"{'='*70}\n")

    # Find input parquet files
    input_files = find_parquet_files(input_path)
    total_files = len(input_files)

    # Apply file slice if provided (for manual parallelization)
    if file_slice is not None:
        try:
            # Parse slice notation: "start::step" or "start:stop:step"
            parts = file_slice.split(':')
            if len(parts) == 2:
                # Format: "start::step"
                start = int(parts[0]) if parts[0] else None
                stop = None
                step = int(parts[1]) if parts[1] else None
            elif len(parts) == 3:
                # Format: "start:stop:step"
                start = int(parts[0]) if parts[0] else None
                stop = int(parts[1]) if parts[1] else None
                step = int(parts[2]) if parts[2] else None
            else:
                raise ValueError(f"Invalid slice format: {file_slice}")

            # Apply slice
            input_files = input_files[start:stop:step]
            print(f"Found {total_files} total file(s), processing {len(input_files)} with slice [{file_slice}]")
        except Exception as e:
            raise ValueError(f"Invalid file_slice '{file_slice}': {e}")
    else:
        print(f"Found {len(input_files)} input file(s)")

    start_time = time.time()
    total_converted = 0
    total_errors = 0
    total_skipped = 0

    # Process each input file
    for file_idx, input_file in enumerate(input_files):
        file_name = os.path.basename(input_file)
        output_file = os.path.join(output_path, file_name.replace('.parquet', '_processed.parquet'))

        print(f"\n{'='*70}")
        print(f"[{file_idx+1}/{len(input_files)}] Processing: {file_name}")
        print(f"{'='*70}")

        # Check if output file exists
        if skip_existing and os.path.exists(output_file):
            file_size = os.path.getsize(output_file) / (1024 * 1024)
            print(f"⏭  Skipping (exists: {file_size:.2f} MB)")
            # Count samples in existing file
            try:
                existing_dataset = load_dataset('parquet', data_files=output_file, split='train')
                total_converted += len(existing_dataset)
                total_skipped += len(existing_dataset)
            except:
                pass
            continue

        # Load input dataset
        print(f"Loading dataset...")
        dataset = load_dataset('parquet', data_files=input_file, split='train')

        total_samples = len(dataset)
        if max_samples > 0:
            total_samples = min(max_samples, total_samples)
            print(f"Processing first {total_samples} samples out of {len(dataset)}")
        else:
            print(f"Processing all {total_samples} samples")

        # Process samples
        file_start = time.time()
        converted_samples, errors = process_samples_range(
            dataset=dataset,
            input_file=input_file,
            start_idx=0,
            end_idx=total_samples,
            model_name=model_name,
            custom_system_prompt=custom_system_prompt,
            custom_user_prompt=custom_user_prompt,
            n_workers=n_workers,
            batch_size=batch_size,
        )

        # Save output
        if converted_samples:
            print(f"\nSaving {len(converted_samples)} samples to {output_file}...")
            save_start = time.time()

            output_dataset = HFDataset.from_list(converted_samples)
            output_dataset.to_parquet(output_file)

            save_time = time.time() - save_start
            file_size = os.path.getsize(output_file) / (1024 * 1024)
            print(f"✓ Saved: {file_size:.2f} MB in {save_time:.2f}s")

            total_converted += len(converted_samples)

        total_errors += len(errors)

        file_elapsed = time.time() - file_start
        print(f"File time: {file_elapsed:.2f}s ({total_samples/file_elapsed:.1f} samples/s)")

    # Summary
    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"CONVERSION COMPLETE")
    print(f"{'='*70}")
    print(f"Total files: {len(input_files)}")
    print(f"Converted: {total_converted}")
    if total_skipped > 0:
        print(f"Skipped: {total_skipped}")
    print(f"Errors: {total_errors}")
    print(f"Time: {elapsed:.2f}s")
    print(f"Output directory: {output_path}")
    print(f"{'='*70}")


def process_samples_range(
    dataset: Any,
    input_file: str,
    start_idx: int,
    end_idx: int,
    model_name: str,
    custom_system_prompt: Optional[str],
    custom_user_prompt: Optional[str],
    n_workers: int,
    batch_size: int,
) -> Tuple[List[Dict], List[Tuple[int, str]]]:
    """Process samples with vision model"""
    total = end_idx - start_idx
    indices = list(range(start_idx, end_idx))

    converted_samples = []
    errors = []

    if n_workers > 1:
        # Parallel processing - only pass indices, not data!
        batches = []
        print(f"  Preparing {(total + batch_size - 1) // batch_size} batches...")
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i + batch_size]
            # Pass file path and indices, NOT the actual data
            batches.append((
                input_file,
                batch_indices,
                model_name,
                custom_system_prompt,
                custom_user_prompt
            ))

        print(f"  Processing {len(batches)} batches with {n_workers} workers...")

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(process_batch, args): i for i, args in enumerate(batches)}

            with tqdm(total=total, desc="Converting", unit="samples") as pbar:
                for future in as_completed(futures):
                    try:
                        batch_results = future.result()
                        for success, result, idx in batch_results:
                            if success:
                                converted_samples.append(result)
                            else:
                                errors.append((idx, result))
                            pbar.update(1)
                    except Exception as e:
                        batch_idx = futures[future]
                        print(f"\nBatch {batch_idx} failed: {e}")
                        import traceback
                        traceback.print_exc()
                        pbar.update(len(batches[batch_idx][1]))  # batch_indices is at index 1

    else:
        # Sequential processing with batching for GPU acceleration
        from transformers import AutoProcessor, AutoTokenizer

        print(f"  Loading processor and tokenizer...")
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        print(f"  Processing {total} samples in batches of {batch_size}...")

        # Process in batches for GPU acceleration
        with tqdm(total=total, desc="Converting", unit="samples") as pbar:
            for i in range(0, len(indices), batch_size):
                batch_indices = indices[i:i + batch_size]

                try:
                    # Load batch samples
                    batch_samples = [dataset[idx] for idx in batch_indices]

                    # Process batch with GPU acceleration
                    processed_batch = process_batch_samples(
                        samples=batch_samples,
                        processor=processor,
                        tokenizer=tokenizer,
                        custom_system_prompt=custom_system_prompt,
                        custom_user_prompt=custom_user_prompt,
                    )

                    converted_samples.extend(processed_batch)
                    pbar.update(len(processed_batch))

                    # Handle samples that failed in batch processing
                    if len(processed_batch) < len(batch_samples):
                        failed_count = len(batch_samples) - len(processed_batch)
                        for j in range(failed_count):
                            errors.append((batch_indices[j], "Failed in batch processing"))

                except Exception as e:
                    # If batch fails, fall back to single sample processing
                    import traceback
                    print(f"\n  Warning: Batch processing failed, falling back to single samples: {e}")

                    for idx in batch_indices:
                        try:
                            sample = dataset[idx]
                            processed_sample = process_single_sample(
                                sample=sample,
                                processor=processor,
                                tokenizer=tokenizer,
                                custom_system_prompt=custom_system_prompt,
                                custom_user_prompt=custom_user_prompt,
                            )
                            converted_samples.append(processed_sample)
                        except Exception as e2:
                            error_msg = f"{str(e2)}\n{traceback.format_exc()}"
                            errors.append((idx, error_msg))
                        pbar.update(1)

                # Periodic cleanup
                if (i + batch_size) % (batch_size * 10) == 0:
                    gc.collect()

    # Report errors
    if errors:
        print(f"\n⚠ {len(errors)} samples failed:")
        for idx, error in errors[:3]:
            print(f"  Sample {idx}: {error[:200]}")
        if len(errors) > 3:
            print(f"  ... and {len(errors) - 3} more errors")

    return converted_samples, errors


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert verl format to tokenized format with vision preprocessing"
    )
    parser.add_argument("input_path", help="Path to input directory or .parquet file")
    parser.add_argument("model_name", help="HuggingFace model name (e.g., Qwen/Qwen2.5-VL-7B-Instruct)")
    parser.add_argument("--system_prompt", default=None, help="Custom system prompt (None = keep original)")
    parser.add_argument("--user_prompt", default=None, help="Custom user prompt (None = keep original)")
    parser.add_argument("--output_path", default=None, help="Output directory (None = use default)")
    parser.add_argument("--max_samples", type=int, default=-1, help="Max samples per file (-1 for all)")
    parser.add_argument("--n_workers", type=int, default=1, help="Number of workers")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size per worker")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    parser.add_argument("--file_slice", default=None, help="File slice for manual parallelization (e.g., '0::3', '1::3', '2::3')")
    parser.add_argument("--gpu_id", type=int, default=None, help="GPU ID to use (None = use default)")

    args = parser.parse_args()

    convert_dataset(
        input_path=args.input_path,
        model_name=args.model_name,
        custom_system_prompt=args.system_prompt,
        custom_user_prompt=args.user_prompt,
        output_path=args.output_path,
        max_samples=args.max_samples,
        n_workers=args.n_workers,
        batch_size=args.batch_size,
        skip_existing=not args.overwrite,
        file_slice=args.file_slice,
        gpu_id=args.gpu_id,
    )

# python convert_plain_rlvr2token.py  $GEOGUESSR_DIR/verl_data/plain_rlvr  Qwen/Qwen2.5-VL-7B-Instruct  --n_workers -1  --batch_size 1000