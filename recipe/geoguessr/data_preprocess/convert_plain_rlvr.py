"""
Convert preprocessed GeoGuessr datasets to verl RLHF training format

This script converts the unified dataset format to verl-compatible parquet files
with proper prompt structure, reward configuration, and tool kwargs.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import argparse
from datasets import load_from_disk, load_dataset, Dataset as HFDataset, Sequence, Image as HFImage
from PIL import Image
from tqdm import tqdm
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import io


def build_system_prompt() -> str:
    """
    Build standard Qwen3-VL-8B-Thinking system prompt
    """
    return "You are a helpful assistant."


def build_user_prompt(sample: Dict[str, Any], include_address_hints: bool = False) -> str:
    """
    Build user prompt with detailed analysis instructions

    Args:
        sample: Dataset sample
        include_address_hints: Whether to include address information as hints
    """
    base_prompt = (
        "<image>\n\nWhere was this photo taken? Analyze the image and predict the location.\n\n"
        "Consider clues like: architecture, vegetation/terrain, text/language, road signs/markings, vehicles/traffic direction, climate, cultural elements, and landmarks.\n\n"
        "Output the final answer as coordinates in $\\boxed{latitude, longitude}$ (decimal degrees)."
    )

    if include_address_hints:
        # Add address hints if available (for training with hints)
        hints = []
        if sample.get('country'):
            hints.append(f"Country: {sample['country']}")
        if sample.get('city'):
            hints.append(f"City: {sample['city']}")

        if hints:
            base_prompt += "\n\n**Hints:**\n" + "\n".join(f"- {hint}" for hint in hints)

    return base_prompt


def convert_sample_to_verl_format(
    sample: Dict[str, Any],
    index: int,
    split: str,
    include_address_hints: bool = False,
    instruction_following: Optional[str] = None,
    encode_images: bool = True
) -> Dict[str, Any]:
    """
    Convert a single sample to verl RLHF format

    Args:
        sample: Original dataset sample
        index: Sample index
        split: Dataset split ('train' or 'test')
        include_address_hints: Whether to include address hints in prompt
        instruction_following: Additional instruction following prompt
        encode_images: If True, encode PIL Images to bytes in worker process (for parallel encoding)
    """
    # Extract ground truth coordinates - support both naming conventions
    if 'lat' in sample and 'lon' in sample:
        lat = float(sample['lat'])
        lon = float(sample['lon'])
    elif 'latitude' in sample and 'longitude' in sample:
        lat = float(sample['latitude'])
        lon = float(sample['longitude'])
    else:
        raise ValueError(f"Sample missing coordinate fields. Available keys: {list(sample.keys())}")

    ground_truth = {"lat": lat, "lon": lon}

    # Build prompt messages
    system_prompt = build_system_prompt()
    user_prompt = build_user_prompt(sample, include_address_hints)

    # Add instruction following if provided
    if instruction_following:
        user_prompt += "\n\n" + instruction_following

    prompt_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    # Process image - OPTIMIZED to avoid decode->encode cycle
    image = sample['image']

    # Check if image is already in bytes format (dict with 'bytes' or 'path' key)
    if isinstance(image, dict) and ('bytes' in image or 'path' in image):
        # Image is already encoded, use directly! No decode->encode needed
        images = [image]
    elif encode_images and isinstance(image, Image.Image):
        # Image is PIL, encode it in worker process
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG', quality=95, optimize=True)
        image_bytes = img_byte_arr.getvalue()
        images = [{'bytes': image_bytes, 'path': None}]
    else:
        # Keep as-is (for sequential processing or other formats)
        images = [image]

    # Build verl format data
    # Determine data source - auto-detect if not provided
    data_source = sample.get('source', 'GeoGuessr')
    if data_source == 'GeoGuessr':
        # Try to infer source from other fields
        if 'flickr_url' in sample:
            data_source = 'yfcc4k'
        elif 'img_id' in sample and 's3_label' in sample:
            data_source = 'im2gps3k'

    verl_sample = {
        "data_source": data_source,
        "prompt": prompt_messages,
        "images": images,
        "ability": "geolocation",
        "reward_model": {
            "style": "rule",
            "ground_truth": ground_truth
        },
        "extra_info": {
            "split": split,
            "index": index,
            "answer": ground_truth,
            "image_source": sample.get('image_source', sample.get('flickr_url', '')),
            "need_tools_kwargs": True,
            "tools_kwargs": {
                "calc_geoguessr_reward": {
                    "create_kwargs": {"ground_truth": ground_truth},
                }
            }
        }
    }

    # Add dataset-specific metadata to extra_info
    if sample.get('source') == 'gaea' or data_source == 'gaea':
        verl_sample['extra_info']['question_type'] = sample.get('question_type', '')
        verl_sample['extra_info']['subset'] = sample.get('subset', '')
    elif sample.get('source') == 'geochain' or data_source == 'geochain':
        verl_sample['extra_info']['locatability_score'] = sample.get('locatability_score', 0.0)
    elif sample.get('source') == 'osv5m' or data_source == 'osv5m':
        verl_sample['extra_info']['osv5m_country'] = sample.get('osv5m_country', '')
        verl_sample['extra_info']['land_cover'] = sample.get('land_cover')
        verl_sample['extra_info']['climate'] = sample.get('climate')
    elif data_source == 'yfcc4k':
        verl_sample['extra_info']['image_id'] = sample.get('image_id', '')
        verl_sample['extra_info']['timestamp'] = sample.get('timestamp', '')
        verl_sample['extra_info']['camera'] = sample.get('camera', '')
        verl_sample['extra_info']['license'] = sample.get('license', '')
    elif data_source == 'im2gps3k':
        verl_sample['extra_info']['img_id'] = sample.get('img_id', '')
        verl_sample['extra_info']['author'] = sample.get('author', '')
        verl_sample['extra_info']['s3_label'] = sample.get('s3_label')
        verl_sample['extra_info']['s16_label'] = sample.get('s16_label')
        verl_sample['extra_info']['s365_label'] = sample.get('s365_label')
        verl_sample['extra_info']['prob_indoor'] = sample.get('prob_indoor')
        verl_sample['extra_info']['prob_natural'] = sample.get('prob_natural')
        verl_sample['extra_info']['prob_urban'] = sample.get('prob_urban')

    # Add address information to extra_info for reference
    verl_sample['extra_info']['address'] = {
        'country': sample.get('country', ''),
        'city': sample.get('city', ''),
        'road': sample.get('road', ''),
    }

    return verl_sample


def process_batch(args: Tuple) -> List[Tuple[bool, Any, int]]:
    """
    Process a batch of samples in a worker process

    OPTIMIZED: Receives pre-loaded samples instead of reloading entire dataset

    Args:
        args: Tuple of (samples_batch, start_index, split, include_address_hints, instruction_following)

    Returns:
        List of (success, result_or_error, index) tuples
    """
    samples_batch, start_index, split, include_address_hints, instruction_following = args

    results = []
    for i, sample in enumerate(samples_batch):
        idx = start_index + i
        try:
            verl_sample = convert_sample_to_verl_format(
                sample=sample,
                index=idx,
                split=split,
                include_address_hints=include_address_hints,
                instruction_following=instruction_following,
                encode_images=True  # Encode in worker process for parallel encoding
            )
            results.append((True, verl_sample, idx))
        except Exception as e:
            results.append((False, str(e), idx))

    return results


def convert_dataset(
    input_path: str,
    output_path: str,
    split: str,
    include_address_hints: bool = False,
    instruction_following: Optional[str] = None,
    max_samples: int = -1,
    chunk_size: int = -1,
    n_workers: int = -1,
    batch_size: int = 100,
    skip_existing: bool = True,
    use_load_dataset: bool = False
):
    """
    Convert a preprocessed dataset to verl format and save as parquet

    Args:
        input_path: Path to the preprocessed dataset (HuggingFace Dataset format)
        output_path: Path to save the converted parquet file(s)
        split: Dataset split name ('train' or 'test')
        include_address_hints: Whether to include address hints in prompts
        instruction_following: Additional instruction to append to prompts
        max_samples: Maximum number of samples to convert (-1 for all)
        chunk_size: If > 0, split into multiple parquet files with this many samples each
        n_workers: Number of parallel workers (-1 for auto, 0 or 1 for sequential)
        batch_size: Number of samples per batch for parallel processing
        skip_existing: If True, skip chunks whose output files already exist
        use_load_dataset: If True, use load_dataset() instead of load_from_disk()
    """
    print(f"\n{'='*60}")
    print(f"Loading dataset from {input_path}...")
    print(f"{'='*60}")

    if use_load_dataset:
        dataset = load_dataset(input_path, split=split)
    else:
        dataset = load_from_disk(input_path)

    total_samples = len(dataset)
    if max_samples > 0:
        total_samples = min(max_samples, total_samples)
        print(f"Converting first {total_samples} samples out of {len(dataset)}...")
    else:
        print(f"Converting all {total_samples} samples...")

    # Determine number of workers
    if n_workers == -1:
        n_workers = min(os.cpu_count() or 1, 64)  # Cap at 64 workers for better parallelism
    use_parallel = n_workers > 1

    print(f"Processing mode: {'Parallel' if use_parallel else 'Sequential'}")
    if use_parallel:
        print(f"Workers: {n_workers}, Batch size: {batch_size}")

    start_time = time.time()

    # Determine if we need chunking
    use_chunking = chunk_size > 0

    if use_chunking:
        print(f"Chunking enabled: {chunk_size} samples per file")
        if skip_existing:
            print(f"Skip existing: Enabled (will skip chunks with existing output files)\n")
        else:
            print(f"Skip existing: Disabled (will overwrite existing files)\n")

        base_output_path = output_path.replace('.parquet', '')
        output_dir = os.path.dirname(base_output_path) or '.'
        os.makedirs(output_dir, exist_ok=True)

        output_files = []
        total_converted = 0
        total_errors = 0
        total_skipped = 0
        chunk_idx = 0

        # Calculate total chunks
        total_chunks = (total_samples + chunk_size - 1) // chunk_size

        for chunk_start in range(0, total_samples, chunk_size):
            chunk_end = min(chunk_start + chunk_size, total_samples)
            chunk_total = chunk_end - chunk_start
            chunk_output_path = f"{base_output_path}_chunk_{chunk_idx:04d}.parquet"

            # Check if chunk already exists
            if skip_existing and os.path.exists(chunk_output_path):
                file_size = os.path.getsize(chunk_output_path) / (1024 * 1024)
                print(f"⏭  Chunk {chunk_idx}/{total_chunks-1}: Skipping (file exists: {chunk_output_path}, {file_size:.2f} MB)")
                output_files.append(chunk_output_path)
                total_converted += chunk_total
                total_skipped += chunk_total
                chunk_idx += 1
                continue

            print(f"\n{'='*60}")
            print(f"Chunk {chunk_idx}/{total_chunks-1}: Processing samples {chunk_start} to {chunk_end-1} ({chunk_total} samples)")
            print(f"{'='*60}")

            chunk_time = time.time()

            try:
                # Process this chunk
                converted_samples, errors = process_samples_range(
                    input_path=input_path,
                    dataset=dataset,
                    start_idx=chunk_start,
                    end_idx=chunk_end,
                    split=split,
                    include_address_hints=include_address_hints,
                    instruction_following=instruction_following,
                    n_workers=n_workers,
                    batch_size=batch_size,
                    use_parallel=use_parallel
                )

                # Save chunk immediately
                if converted_samples:
                    print(f"\n{'='*50}")
                    print(f"Saving chunk {chunk_idx} to {chunk_output_path}...")
                    save_start = time.time()

                    # Create dataset from list (images already encoded as bytes)
                    print(f"  Creating HFDataset from {len(converted_samples)} samples...")
                    dataset_start = time.time()
                    chunk_dataset = HFDataset.from_list(converted_samples)
                    dataset_time = time.time() - dataset_start
                    print(f"  Dataset created in {dataset_time:.2f}s")

                    # Cast images column to proper type
                    print(f"  Casting images column...")
                    cast_start = time.time()
                    chunk_dataset = chunk_dataset.cast_column('images', Sequence(HFImage()))
                    cast_time = time.time() - cast_start
                    print(f"  Images cast in {cast_time:.2f}s")

                    # Save to parquet
                    print(f"  Writing to parquet...")
                    parquet_start = time.time()
                    chunk_dataset.to_parquet(chunk_output_path)
                    parquet_time = time.time() - parquet_start
                    print(f"  Parquet saved in {parquet_time:.2f}s")

                    save_time = time.time() - save_start

                    # Verify file was written
                    if os.path.exists(chunk_output_path):
                        file_size = os.path.getsize(chunk_output_path) / (1024 * 1024)  # MB
                        print(f"✓ Chunk {chunk_idx} saved successfully")
                        print(f"  Samples: {len(converted_samples)}, Size: {file_size:.2f} MB")
                        print(f"  Total save time: {save_time:.2f}s")
                        output_files.append(chunk_output_path)
                        total_converted += len(converted_samples)
                    else:
                        print(f"✗ ERROR: Failed to save chunk {chunk_idx}")

                    chunk_elapsed = time.time() - chunk_time
                    print(f"Chunk processing time: {chunk_elapsed:.2f}s ({chunk_total/chunk_elapsed:.1f} samples/s)")

                total_errors += len(errors)
                if errors:
                    print(f"Errors in chunk: {len(errors)}")

            except Exception as e:
                print(f"✗ ERROR processing chunk {chunk_idx}: {e}")
                import traceback
                traceback.print_exc()
                print(f"Continuing to next chunk...")

            chunk_idx += 1

        elapsed = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"CONVERSION COMPLETE")
        print(f"{'='*60}")
        print(f"Total chunks: {chunk_idx}")
        print(f"Total converted: {total_converted}/{total_samples}")
        if total_skipped > 0:
            print(f"Total skipped: {total_skipped} (files already existed)")
        print(f"Total errors: {total_errors}")
        print(f"Total time: {elapsed:.2f}s ({total_samples/elapsed:.1f} samples/s)")
        print(f"\nOutput files ({len(output_files)}):")
        for f in output_files:
            print(f"  ✓ {f}")

    else:
        # Single file mode
        print(f"Single file mode: {output_path}\n")

        converted_samples, errors = process_samples_range(
            input_path=input_path,
            dataset=dataset,
            start_idx=0,
            end_idx=total_samples,
            split=split,
            include_address_hints=include_address_hints,
            instruction_following=instruction_following,
            n_workers=n_workers,
            batch_size=batch_size,
            use_parallel=use_parallel
        )

        # Save as parquet
        print(f"\n{'='*50}")
        print(f"Saving {len(converted_samples)} samples to {output_path}...")
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        save_start = time.time()

        # Create dataset from list (images already encoded as bytes)
        print(f"  Creating HFDataset...")
        dataset_start = time.time()
        converted_dataset = HFDataset.from_list(converted_samples)
        dataset_time = time.time() - dataset_start
        print(f"  Dataset created in {dataset_time:.2f}s")

        # Cast images column to proper type
        print(f"  Casting images column...")
        cast_start = time.time()
        converted_dataset = converted_dataset.cast_column('images', Sequence(HFImage()))
        cast_time = time.time() - cast_start
        print(f"  Images cast in {cast_time:.2f}s")

        # Save to parquet
        print(f"  Writing to parquet...")
        parquet_start = time.time()
        converted_dataset.to_parquet(output_path)
        parquet_time = time.time() - parquet_start
        print(f"  Parquet saved in {parquet_time:.2f}s")

        save_time = time.time() - save_start

        # Verify
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path) / (1024 * 1024)
            print(f"✓ Saved successfully: {file_size:.2f} MB")
            print(f"  Total save time: {save_time:.2f}s")

        elapsed = time.time() - start_time
        print(f"\nTotal converted: {len(converted_samples)}/{total_samples}")
        print(f"Total errors: {len(errors)}")
        print(f"Total time: {elapsed:.2f}s ({total_samples/elapsed:.1f} samples/s)")


def process_samples_range(
    input_path: str,
    dataset: Any,
    start_idx: int,
    end_idx: int,
    split: str,
    include_address_hints: bool,
    instruction_following: Optional[str],
    n_workers: int,
    batch_size: int,
    use_parallel: bool
) -> Tuple[List[Dict], List[Tuple[int, str]]]:
    """
    Process a range of samples with optional parallel processing

    OPTIMIZED: Pre-loads samples and distributes to workers instead of
    letting each worker reload the entire dataset

    Returns:
        Tuple of (converted_samples, errors)
    """
    total = end_idx - start_idx
    indices = list(range(start_idx, end_idx))

    converted_samples = []
    errors = []

    if use_parallel:
        print(f"  Loading batch of {total} samples from dataset...")
        load_start = time.time()

        # OPTIMIZATION: Load samples without decoding images to avoid decode->encode cycle
        print(f"  Reading raw image bytes (skipping PIL decode) for {total} samples...")

        samples_to_process = []

        # Directly access Arrow data to avoid automatic PIL decoding
        # This is MUCH faster as we completely skip the decode->encode cycle
        for idx in tqdm(indices, desc="Loading samples (raw bytes)", leave=False):
            sample_dict = {}

            # Get the row from the dataset's underlying table
            try:
                # Use dataset.data to access Arrow table (handles sharding)
                arrow_batch = dataset.data.slice(idx, 1)

                for col_idx, col_name in enumerate(dataset.column_names):
                    if col_name == 'image':
                        # For image column, get raw struct (bytes + path)
                        # This avoids PIL decoding!
                        image_col = arrow_batch.column(col_idx)
                        image_struct = image_col[0].as_py()
                        sample_dict['image'] = image_struct
                    else:
                        # For other columns, get Python value
                        col_value = arrow_batch.column(col_idx)[0].as_py()
                        sample_dict[col_name] = col_value

            except Exception as e:
                # Fallback: if direct Arrow access fails, use regular dataset access
                # (This will decode images, but ensures we don't crash)
                print(f"\n  Warning: Arrow access failed for idx {idx}, falling back to regular access: {e}")
                sample_dict = dataset[idx]

            samples_to_process.append(sample_dict)

        load_time = time.time() - load_start
        print(f"  ✓ Samples loaded in {load_time:.2f}s ({total/load_time:.1f} samples/s)")
        print(f"  ✓ Images kept as raw bytes (no decoding needed!)")

        # Split samples into batches
        batches = []
        for i in range(0, len(samples_to_process), batch_size):
            batch_samples = samples_to_process[i:i + batch_size]
            batch_start_idx = indices[i]
            batches.append((batch_samples, batch_start_idx, split, include_address_hints, instruction_following))

        print(f"  Processing {len(batches)} batches with {n_workers} workers (encoding images in parallel)...")
        process_start = time.time()

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
                        pbar.update(len(batches[batch_idx][0]))

        process_time = time.time() - process_start
        print(f"  Processing completed in {process_time:.2f}s ({total/process_time:.1f} samples/s)")
    else:
        # Sequential processing
        for idx in tqdm(indices, desc="Converting", unit="samples"):
            try:
                sample = dataset[idx]
                verl_sample = convert_sample_to_verl_format(
                    sample=sample,
                    index=idx,
                    split=split,
                    include_address_hints=include_address_hints,
                    instruction_following=instruction_following,
                    encode_images=False  # No need to encode in sequential mode
                )
                converted_samples.append(verl_sample)
            except Exception as e:
                errors.append((idx, str(e)))

    # Report errors
    if errors:
        print(f"\n⚠ {len(errors)} samples failed:")
        for idx, error in errors[:5]:
            print(f"  Sample {idx}: {error}")
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more errors")

    return converted_samples, errors


def convert_all_datasets(
    geogusor_dir: str,
    output_dir: str,
    include_address_hints: bool = False,
    instruction_following: Optional[str] = None,
    max_samples_per_dataset: int = -1,
    chunk_size: int = -1,
    n_workers: int = -1,
    batch_size: int = 100,
    skip_existing: bool = True,
    datasets: Optional[List[str]] = None
):
    """
    Convert all preprocessed datasets to verl format

    Args:
        geogusor_dir: GEOGUESSR_DIR path
        output_dir: Output directory for parquet files
        include_address_hints: Whether to include address hints
        instruction_following: Additional instruction
        max_samples_per_dataset: Max samples per dataset (-1 for all)
        chunk_size: If > 0, split into multiple files with this many samples each
        n_workers: Number of parallel workers (-1 for auto)
        batch_size: Batch size for parallel processing
        skip_existing: If True, skip chunks whose output files already exist
        datasets: List of datasets to process (e.g., ['gaea', 'osv5m']). If None, process all.
    """
    processed_dir = os.path.join(geogusor_dir, "processed")
    output_dir = os.path.join(output_dir, "verl_data", "plain_rlvr")

    # Define all available datasets with their metadata
    # Format: (relative_path, output_name, split, use_load_dataset)
    all_datasets = {
        'gaea': [
            ('gaea/train', 'gaea_train.parquet', 'train', False),
            ('gaea/bench', 'gaea_bench.parquet', 'test', False),
        ],
        'geochain': [
            ('geochain/test', 'geochain_test.parquet', 'test', False),
            ('geochain/mini_test', 'geochain_mini_test.parquet', 'test', False),
        ],
        'osv5m': [
            ('osv5m/train', 'osv5m_train.parquet', 'train', False),
        ],
        'yfcc4k': [
            ('/diancpfs/user/guobin/geogussr/yfcc4k/yfcc4k_hf_dataset', 'yfcc4k_train.parquet', 'train', True),
        ],
        'im2gps3k': [
            ('/diancpfs/user/guobin/geogussr/im2gps3k/im2gps3k_hf_dataset', 'im2gps3k_train.parquet', 'train', False),
        ]
    }

    # If no datasets specified, use all
    if datasets is None or len(datasets) == 0:
        datasets = list(all_datasets.keys())

    # Validate dataset names
    invalid_datasets = [d for d in datasets if d not in all_datasets]
    if invalid_datasets:
        print(f"⚠ Warning: Unknown datasets will be skipped: {invalid_datasets}")
        datasets = [d for d in datasets if d in all_datasets]

    if not datasets:
        print("✗ Error: No valid datasets specified")
        return

    # Build the list of datasets to convert
    datasets_to_convert = []
    for dataset_name in datasets:
        for rel_path, output_name, split, use_load_dataset in all_datasets[dataset_name]:
            # For datasets with absolute paths (yfcc4k, im2gps3k), use path as-is
            if rel_path.startswith('/'):
                input_path = rel_path
            else:
                input_path = os.path.join(processed_dir, rel_path)
            output_path = os.path.join(output_dir, output_name)
            datasets_to_convert.append((input_path, output_path, split, dataset_name, use_load_dataset))

    print("\n" + "=" * 60)
    print("CONVERTING DATASETS TO VERL FORMAT")
    print("=" * 60)
    print(f"Selected datasets: {', '.join(datasets)}")
    print(f"Output directory: {output_dir}")
    if chunk_size > 0:
        print(f"Chunking: {chunk_size} samples per file")
    if n_workers != 1:
        print(f"Parallel processing: {n_workers if n_workers > 0 else 'auto'} workers")
    print("=" * 60)

    overall_start = time.time()
    successful = 0
    failed = 0

    for input_path, output_path, split, dataset_name, use_load_dataset in datasets_to_convert:
        if not os.path.exists(input_path):
            print(f"\n⊘ Skipping {input_path} (not found)")
            continue

        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_name}/{os.path.basename(input_path)}")
        print(f"{'='*60}")

        try:
            convert_dataset(
                input_path=input_path,
                output_path=output_path,
                split=split,
                include_address_hints=include_address_hints,
                instruction_following=instruction_following,
                max_samples=max_samples_per_dataset,
                chunk_size=chunk_size,
                n_workers=n_workers,
                batch_size=batch_size,
                skip_existing=skip_existing,
                use_load_dataset=use_load_dataset
            )
            successful += 1
        except Exception as e:
            print(f"\n✗ Error converting {input_path}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
            continue

    overall_elapsed = time.time() - overall_start
    print("\n" + "=" * 60)
    print("ALL DATASETS CONVERSION COMPLETE")
    print("=" * 60)
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total time: {overall_elapsed/60:.1f} minutes")
    print(f"Output directory: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert preprocessed GeoGuessr datasets to verl RLHF format"
    )
    parser.add_argument(
        "--geogusor_dir",
        default=None,
        help="GEOGUESSR_DIR path (defaults to $GEOGUESSR_DIR env var)"
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output directory for parquet files (defaults to $GEOGUESSR_DIR)"
    )
    parser.add_argument(
        "--include_address_hints",
        action="store_true",
        help="Include address information as hints in the prompt"
    )
    parser.add_argument(
        "--instruction_following",
        default=None,
        help="Additional instruction to append to all prompts"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=-1,
        help="Maximum samples per dataset (-1 for all)"
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=-1,
        help="If > 0, split into multiple parquet files with this many samples each"
    )
    parser.add_argument(
        "--n_workers",
        type=int,
        default=-1,
        help="Number of parallel workers (-1 for auto, 0 or 1 for sequential)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=100,
        help="Batch size for parallel processing (default: 100)"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing chunk files instead of skipping them"
    )
    parser.add_argument(
        "--datasets",
        nargs='+',
        choices=['gaea', 'geochain', 'osv5m', 'yfcc4k', 'im2gps3k'],
        default=None,
        help="Which datasets to process (default: all). Options: gaea, geochain, osv5m, yfcc4k, im2gps3k. Can specify multiple: --datasets gaea osv5m"
    )
    parser.add_argument(
        "--single_dataset",
        default=None,
        help="Convert only a single dataset (e.g., 'gaea/train')"
    )

    args = parser.parse_args()

    # Get GEOGUESSR_DIR
    geogusor_dir = args.geogusor_dir or os.environ.get('GEOGUESSR_DIR')
    if geogusor_dir is None:
        raise ValueError("Please set GEOGUESSR_DIR environment variable or use --geogusor_dir")

    output_dir = args.output_dir or geogusor_dir

    if args.single_dataset:
        # Convert single dataset
        input_path = os.path.join(geogusor_dir, "processed", args.single_dataset)
        dataset_name = args.single_dataset.replace("/", "_")
        output_path = os.path.join(output_dir, "verl_data", "plain_rlvr", f"{dataset_name}.parquet")
        split = "train" if "train" in args.single_dataset else "test"

        convert_dataset(
            input_path=input_path,
            output_path=output_path,
            split=split,
            include_address_hints=args.include_address_hints,
            instruction_following=args.instruction_following,
            max_samples=args.max_samples,
            chunk_size=args.chunk_size,
            n_workers=args.n_workers,
            batch_size=args.batch_size,
            skip_existing=not args.overwrite
        )
    else:
        # Convert all datasets (or selected datasets)
        convert_all_datasets(
            geogusor_dir=geogusor_dir,
            output_dir=output_dir,
            include_address_hints=args.include_address_hints,
            instruction_following=args.instruction_following,
            max_samples_per_dataset=args.max_samples,
            chunk_size=args.chunk_size,
            n_workers=args.n_workers,
            batch_size=args.batch_size,
            skip_existing=not args.overwrite,
            datasets=args.datasets
        )


# python convert_plain_rlvr.py     --chunk_size 100000     --n_workers -1     --batch_size 1000  --datasets osv5m  geochain  gaea