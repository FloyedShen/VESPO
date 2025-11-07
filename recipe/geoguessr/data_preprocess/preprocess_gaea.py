import os
import time
from datasets import load_dataset, Dataset, concatenate_datasets, load_from_disk
from PIL import Image
from typing import Dict, Any, Optional
import requests
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import multiprocessing as mp
import glob
import shutil
import json


def reverse_geocode(
        service_url: str,
        lat: float,
        lon: float,
        format: str = "json",
        addressdetails: int = 1,
        max_retries: int = -1,
        retry_delay: float = 1.0
) -> Optional[Dict[str, Any]]:
    """
    Call reverse geocoding service to get address info from coordinates
    Retries until successful or max_retries is reached

    Args:
        service_url: Base URL of the geocoding service
        lat: Latitude
        lon: Longitude
        format: Response format (default: json)
        addressdetails: Include address details (default: 1)
        max_retries: Maximum number of retry attempts (default: 5)
        retry_delay: Delay between retries in seconds (default: 2.0)
    """
    endpoint = f"{service_url.rstrip('/')}/reverse"
    params = {
        'lat': lat,
        'lon': lon,
        'format': format,
        'addressdetails': addressdetails,
        'accept-language': 'en'
    }

    attempt = 0
    while max_retries == -1 or attempt < max_retries:
        try:
            response = requests.get(endpoint, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            attempt += 1
            if max_retries != -1 and attempt >= max_retries:
                print(f"Request failed after {max_retries} attempts (lat={lat}, lon={lon}): {e}")
                return None

            print(f"Request failed (attempt {attempt}, lat={lat}, lon={lon}): {e}. Retrying in {retry_delay}s...")
            time.sleep(retry_delay)

    return None


def process_gaea_sample(sample: Dict[str, Any], service_url: str) -> Optional[Dict[str, Any]]:
    """
    Process a single GAEA sample

    Returns None if the sample should be discarded (e.g., corrupted image)
    """
    # Get image from file_name field (PIL Image)
    try:
        image = sample['file_name']
        if image is None:
            # Discard: image is None
            return None
        if not isinstance(image, Image.Image):
            # Discard: invalid image format
            return None
    except Exception as e:
        # Discard: cannot load image
        return None

    # Convert lat/lon to float
    try:
        lat = float(sample['lat'])
        lon = float(sample['lon'])
    except (ValueError, TypeError) as e:
        # Discard: invalid lat/lon values
        return None

    # Call reverse geocoding service (with infinite retry, should not fail)
    # If it does fail after many retries, we keep the sample with empty address fields
    geo_result = reverse_geocode(service_url, lat, lon)

    # Extract address information (use empty dict if geocoding failed)
    address = geo_result.get('address', {}) if geo_result is not None else {}

    # Build OpenAI format messages and convert to string
    messages = [
        {"role": "user", "content": sample['question']},
        {"role": "assistant", "content": sample['answer']}
    ]
    # Convert to string to avoid serialization issues
    messages_str = json.dumps(messages, ensure_ascii=False)

    # Build return dictionary
    processed_sample = {
        # Required fields
        'image': image,
        'lon': lon,
        'lat': lat,
        'image_source': sample.get('dataset', ''),
        'source': 'gaea',
        'messages': messages_str,  # Store as string

        # Nominatim address fields
        'road': address.get('road', '').strip().lower(),
        'suburb': address.get('suburb', '').strip().lower(),
        'ISO3166-2-lvl10': address.get('ISO3166-2-lvl10', '').strip().lower(),
        'city': address.get('city', '').strip().lower(),
        'postcode': address.get('postcode', '').strip().lower(),
        'country': address.get('country', '').strip().lower(),

        # GAEA custom fields
        'question_type': sample.get('question_type', ''),
        'subset': sample.get('subset', ''),
    }

    return processed_sample


def preprocess_gaea(
        dataset_name: str,
        dataset_split: str,
        service_url: str,
        output_dir: str = None,
        num_workers: int = None,
        chunk_size: int = 500,
        save_merged: bool = True,
        resume: bool = True,
        overwrite: bool = False
):
    """
    Process GAEA dataset in chunks with parallel processing and resume support

    Args:
        dataset_name: Dataset name ("ucf-crcv/GAEA-Train" or "ucf-crcv/GAEA-Bench")
        dataset_split: Dataset split name (usually "train" or "test")
        service_url: Reverse geocoding service URL
        output_dir: Output directory, defaults to $GEOGUESSR_DIR/processed/gaea/
        num_workers: Number of parallel workers, defaults to CPU count
        chunk_size: Size of each chunk
        save_merged: Whether to merge all chunks after processing, default True
        resume: Whether to resume from existing chunks (checkpoint), default True
        overwrite: Whether to overwrite existing data (True will delete all existing chunks), default False

    Returns:
        If save_merged=True: (merged_path, chunk_count)
        If save_merged=False: (chunks_dir, chunk_count)
    """
    # Get output directory
    if output_dir is None:
        geogusor_dir = os.environ.get('GEOGUESSR_DIR')
        if geogusor_dir is None:
            raise ValueError("Please set GEOGUESSR_DIR environment variable or pass output_dir parameter")
        output_dir = f"{geogusor_dir}/processed/gaea/"

    # Set number of workers
    if num_workers is None:
        num_workers = min(mp.cpu_count(), 8)

    print(f"Loading {dataset_name} dataset (split={dataset_split})...")
    gaea = load_dataset(dataset_name, split=dataset_split)
    print(f"Size of {dataset_name}/{dataset_split}: {len(gaea)}")

    # Create dataset-specific subdirectory
    dataset_short_name = "train" if "Train" in dataset_name else "bench"
    chunks_dir = os.path.join(output_dir, f"{dataset_short_name}_chunks")

    # Handle overwrite and resume logic
    existing_chunks = []
    start_chunk_idx = 0

    if os.path.exists(chunks_dir):
        existing_chunks = sorted(glob.glob(os.path.join(chunks_dir, "chunk_*")))

        if overwrite:
            # Delete all existing chunks and start over
            print(f"Detected overwrite=True, deleting {len(existing_chunks)} existing chunks...")
            shutil.rmtree(chunks_dir)
            existing_chunks = []
            os.makedirs(chunks_dir, exist_ok=True)
            print("Starting from scratch...")
        elif resume and existing_chunks:
            # Resume from last chunk
            start_chunk_idx = len(existing_chunks)
            print(f"Detected {len(existing_chunks)} existing chunks")
            print(f"Resuming from chunk {start_chunk_idx} (checkpoint resume)...")
        elif not resume and existing_chunks:
            # resume not allowed but has existing data, raise error
            raise ValueError(
                f"Detected {len(existing_chunks)} existing chunks but resume=False.\n"
                f"Please set resume=True to continue or overwrite=True to start over."
            )

    os.makedirs(chunks_dir, exist_ok=True)

    process_func = partial(process_gaea_sample, service_url=service_url)

    print(f"\nProcessing {dataset_name}/{dataset_split}...")

    # Start from checkpoint position
    chunk_count = start_chunk_idx
    start_sample_idx = start_chunk_idx * chunk_size

    if start_sample_idx > 0:
        print(f"Skipping first {start_sample_idx} samples (already processed)")

    # Process in chunks
    for start_idx in range(start_sample_idx, len(gaea), chunk_size):
        end_idx = min(start_idx + chunk_size, len(gaea))
        chunk = gaea.select(range(start_idx, end_idx))

        chunk_samples = []

        # First, safely collect samples from chunk (handle corrupted images during iteration)
        chunk_data = []
        print(f"Loading chunk data ({start_idx}-{end_idx})...")
        for i in tqdm(range(len(chunk)), desc="Loading samples"):
            try:
                sample = chunk[i]
                chunk_data.append(sample)
            except Exception as e:
                print(f"\nError loading sample {start_idx + i}: {e}")
                continue

        # Process current chunk in parallel
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(process_func, sample): i
                       for i, sample in enumerate(chunk_data)}

            for future in tqdm(as_completed(futures), total=len(futures),
                               desc=f"Processing chunk {chunk_count}"):
                try:
                    result = future.result()
                    if result is not None:
                        chunk_samples.append(result)
                except Exception as e:
                    print(f"\nError processing sample: {e}")

        # Save current chunk immediately
        if chunk_samples:
            chunk_dataset = Dataset.from_list(chunk_samples)
            chunk_path = os.path.join(chunks_dir, f"chunk_{chunk_count:04d}")
            chunk_dataset.save_to_disk(chunk_path)
            print(f"Saved chunk {chunk_count} ({len(chunk_samples)} samples)")
            chunk_count += 1

            # Release memory
            del chunk_samples
            del chunk_dataset

    print(f"\nTotal saved {chunk_count} chunks to: {chunks_dir}")

    # Merge all chunks if needed
    if save_merged and chunk_count > 0:
        print(f"\nMerging all {chunk_count} chunks...")
        merged_path = os.path.join(output_dir, dataset_short_name)

        # Load all chunks
        chunk_datasets = []
        for i in tqdm(range(chunk_count), desc="Loading chunks"):
            chunk_path = os.path.join(chunks_dir, f"chunk_{i:04d}")
            chunk_dataset = load_from_disk(chunk_path)
            chunk_datasets.append(chunk_dataset)

        # Merge all chunks
        print("Merging datasets...")
        full_dataset = concatenate_datasets(chunk_datasets)

        # Save merged dataset
        print(f"Saving full dataset to: {merged_path}")
        full_dataset.save_to_disk(merged_path)

        print(f"Full dataset saved, total {len(full_dataset)} samples")
        print(f"Can load directly with: load_from_disk('{merged_path}')")

        return merged_path, chunk_count

    return chunks_dir, chunk_count


def load_processed_gaea(
        dataset_type: str,  # 'train' or 'bench'
        output_dir: str = None,
        from_chunks: bool = False
) -> Dataset:
    """
    Load processed GAEA dataset

    Args:
        dataset_type: Dataset type ('train' or 'bench')
        output_dir: Output directory, defaults to $GEOGUESSR_DIR/processed/gaea/
        from_chunks: Whether to load from chunks (will auto load from chunks if no merged dataset)

    Returns:
        Dataset: Loaded dataset
    """
    # Get output directory
    if output_dir is None:
        geogusor_dir = os.environ.get('GEOGUESSR_DIR')
        if geogusor_dir is None:
            raise ValueError("Please set GEOGUESSR_DIR environment variable or pass output_dir parameter")
        output_dir = f"{geogusor_dir}/processed/gaea/"

    merged_path = os.path.join(output_dir, dataset_type)
    chunks_dir = os.path.join(output_dir, f"{dataset_type}_chunks")

    # Try to load merged dataset
    if not from_chunks and os.path.exists(merged_path):
        print(f"Loading full dataset from {merged_path}...")
        return load_from_disk(merged_path)

    # If no merged dataset or specified to load from chunks, load from chunks
    if os.path.exists(chunks_dir):
        print(f"Loading chunks from {chunks_dir}...")
        chunk_paths = sorted(glob.glob(os.path.join(chunks_dir, "chunk_*")))

        if not chunk_paths:
            raise FileNotFoundError(f"No chunks found in {chunks_dir}")

        print(f"Found {len(chunk_paths)} chunks")

        # Load all chunks
        chunk_datasets = []
        for chunk_path in tqdm(chunk_paths, desc="Loading chunks"):
            chunk_dataset = load_from_disk(chunk_path)
            chunk_datasets.append(chunk_dataset)

        # Merge all chunks
        print("Merging datasets...")
        full_dataset = concatenate_datasets(chunk_datasets)
        print(f"Loading complete, total {len(full_dataset)} samples")

        return full_dataset

    raise FileNotFoundError(
        f"No processed dataset found. Please run preprocess_gaea to process data first.\n"
        f"Search paths: {merged_path} or {chunks_dir}"
    )


# Usage example
if __name__ == "__main__":
    # Service URL
    service_url = "http://dsw-notebook-dsw-s9abjgcsvnls12nt68-8080.vpc-t4nptn1gzsxxk04su5mpc.instance-forward.dsw.ap-southeast-1.aliyuncs.com:8080"

    geogusor_dir = os.environ.get('GEOGUESSR_DIR')
    assert geogusor_dir is not None, "GEOGUESSR_DIR is required in environment variables."

    # Example 1: Process GAEA-Train (with checkpoint resume support)
    result_train, chunk_count_train = preprocess_gaea(
        dataset_name="ucf-crcv/GAEA-Train",
        dataset_split="train",
        service_url=service_url,
        num_workers=16,  # Adjust based on CPU cores
        chunk_size=50000,  # Adjust based on memory
        save_merged=True,  # Auto merge all chunks
        resume=True  # If interrupted, will auto resume from checkpoint (default True)
    )

    # Example 2: Process GAEA-Bench
    result_bench, chunk_count_bench = preprocess_gaea(
        dataset_name="ucf-crcv/GAEA-Bench",
        dataset_split="test",
        service_url=service_url,
        num_workers=16,
        chunk_size=50000,
        save_merged=True
    )

    # Example 3: If need to reprocess from scratch (delete existing data)
    # result_train, chunk_count_train = preprocess_gaea(
    #     dataset_name="ucf-crcv/GAEA-Train",
    #     dataset_split="train",
    #     service_url=service_url,
    #     num_workers=16,
    #     chunk_size=1000,
    #     save_merged=True,
    #     overwrite=True  # Delete all existing data and start from scratch
    # )

    print("\n" + "=" * 60)
    print("Processing complete!")
    print("=" * 60)
    print(f"\nGAEA-Train: {chunk_count_train} chunks -> {result_train}")
    print(f"GAEA-Bench: {chunk_count_bench} chunks -> {result_bench}")

    print("\nTo load data:")
    print("# Method 1: Use helper function (recommended)")
    print("from preprocess_gaea import load_processed_gaea")
    print("train_dataset = load_processed_gaea('train')")
    print("bench_dataset = load_processed_gaea('bench')")
    print()
    print("# Method 2: Direct load_from_disk")
    print("from datasets import load_from_disk")
    print(f"train_dataset = load_from_disk('{result_train}')")
    print(f"bench_dataset = load_from_disk('{result_bench}')")
    print()
    print("\nCheckpoint resume notes:")
    print("- If interrupted, rerunning the script will auto resume from where it stopped")
    print("- Set resume=False to raise error when existing data detected, prevent accidental overwrite")
    print("- Set overwrite=True to delete all existing data and start from scratch")
    print()
    print("\nmessages field notes:")
    print("- messages field is stored as JSON string format")
    print("- Need to deserialize when using: import json; messages = json.loads(sample['messages'])")
    print("- Format is OpenAI format: [{'role': 'user', 'content': '...'}, {'role': 'assistant', 'content': '...'}]")
