import os
import time
from pathlib import Path
from datasets import Dataset, concatenate_datasets, load_from_disk, Image as ImageFeature, Features, Value
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
import pandas as pd
import io


def diagnose_osv5m_images(dataset_split: str = 'train', geogusor_dir: str = None, num_samples: int = 10):
    """
    Diagnose OSV5M image path issues by checking a few samples

    Args:
        dataset_split: Dataset split name ('train' or 'test')
        geogusor_dir: GEOGUESSR_DIR environment variable path
        num_samples: Number of samples to check
    """
    if geogusor_dir is None:
        geogusor_dir = os.environ.get('GEOGUESSR_DIR')
        if geogusor_dir is None:
            raise ValueError("Please set GEOGUESSR_DIR environment variable")

    osv5m_root = Path(geogusor_dir) / 'osv5m' / 'osv5m'
    csv_path = osv5m_root / f'{dataset_split}.csv'
    images_dir = osv5m_root / 'images' / dataset_split

    print(f"CSV path: {csv_path}")
    print(f"CSV exists: {csv_path.exists()}")
    print(f"\nImages directory: {images_dir}")
    print(f"Images directory exists: {images_dir.exists()}")

    if images_dir.exists():
        # Check subdirectories
        subdirs = sorted([d for d in images_dir.iterdir() if d.is_dir()])
        print(f"\nFound {len(subdirs)} subdirectories in images directory")
        print(f"First few subdirs: {[d.name for d in subdirs[:10]]}")

        # Check total image count
        image_files = list(images_dir.glob("**/*.jpg"))
        print(f"Total .jpg files found: {len(image_files)}")
        if image_files:
            print(f"Sample image paths:")
            for img in image_files[:5]:
                print(f"  {img.relative_to(images_dir)}")

    if csv_path.exists():
        df = pd.read_csv(csv_path, nrows=num_samples)
        print(f"\nChecking {len(df)} samples from CSV:")
        print(f"CSV columns: {list(df.columns)}")

        found_count = 0
        not_found_count = 0

        for idx, row in df.iterrows():
            img_id = row['id']
            img_subdir = img_id[:2]
            expected_path = images_dir / img_subdir / f"{img_id}.jpg"
            exists = expected_path.exists()

            if exists:
                found_count += 1
            else:
                not_found_count += 1

            status = "✓" if exists else "✗"
            print(f"  {status} ID: {img_id}, Expected: {expected_path.relative_to(images_dir) if images_dir.exists() else expected_path}")

            if idx == 0 and not exists:
                # For first missing file, check what actually exists
                print(f"    Checking alternatives:")
                # Check if subdir exists
                if (images_dir / img_subdir).exists():
                    print(f"    Subdir '{img_subdir}' exists")
                    alt_files = list((images_dir / img_subdir).glob("*"))[:5]
                    print(f"    Files in subdir: {[f.name for f in alt_files]}")
                else:
                    print(f"    Subdir '{img_subdir}' does NOT exist")

                # Check if image exists without subdir
                alt_path = images_dir / f"{img_id}.jpg"
                if alt_path.exists():
                    print(f"    Found at: {alt_path.relative_to(images_dir)}")

        print(f"\nSummary: {found_count}/{len(df)} images found, {not_found_count} not found")
        print(f"Success rate: {found_count/len(df)*100:.2f}%")


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
        max_retries: Maximum number of retry attempts (default: -1 for infinite)
        retry_delay: Delay between retries in seconds (default: 1.0)
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


def process_osv5m_sample(image_path: Path, metadata: Dict[str, Any], service_url: str) -> Optional[Dict[str, Any]]:
    """
    Process a single OSV5M sample directly from image path and CSV metadata

    Args:
        image_path: Path to the image file
        metadata: Dictionary containing all metadata from CSV row
        service_url: Reverse geocoding service URL

    Returns None if the sample should be discarded (e.g., corrupted image or missing file)
    """
    # Check if image file exists
    # print(image_path)
    if not image_path.exists():
        # Discard: image file not found
        return ("skip", "image_not_found")

    # Load image directly from path and encode it in the worker process
    # This parallelizes the encoding work instead of doing it in the main thread
    try:
        image = Image.open(image_path).convert('RGB')

        # Encode image to JPEG bytes in the worker process
        # This is the expensive operation we want to parallelize
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG', quality=95, optimize=True)
        image_bytes = img_byte_arr.getvalue()

    except Exception as e:
        # Discard: cannot load image (corrupted or invalid format)
        return ("skip", "image_load_failed")

    # Convert lat/lon to float
    try:
        lat = float(metadata['latitude'])
        lon = float(metadata['longitude'])
    except (ValueError, TypeError, KeyError) as e:
        # Discard: invalid lat/lon values
        return ("skip", "invalid_latlon")

    # Call reverse geocoding service (with infinite retry, should not fail)
    # If it does fail after many retries, we keep the sample with empty address fields
    geo_result = reverse_geocode(service_url, lat, lon)

    # Extract address information (use empty dict if geocoding failed)
    address = geo_result.get('address', {}) if geo_result is not None else {}

    # Helper function to safely extract string values from metadata
    def get_string_field(field_name, default=''):
        value = metadata.get(field_name, default)
        # Handle NaN values from pandas (which are float type)
        if pd.isna(value):
            return default
        return str(value) if value is not None else default

    # Helper function to safely extract numeric values from metadata
    def get_numeric_field(field_name, default=None):
        value = metadata.get(field_name, default)
        # Handle NaN or None - return None instead of keeping NaN
        if pd.isna(value) or value is None:
            return default
        return value

    # Build return dictionary
    processed_sample = {
        # Required fields - use pre-encoded bytes instead of PIL Image
        # This way encoding happens in parallel in worker processes
        'image': {'bytes': image_bytes, 'path': None},
        'lon': lon,
        'lat': lat,
        'image_source': 'mapillary',  # OSV5M images come from Mapillary
        'source': 'osv5m',
        'messages': "{}",  # OSV5M doesn't have Q&A, use empty JSON string

        # Nominatim address fields (from reverse geocoding)
        'road': address.get('road', '').strip().lower(),
        'suburb': address.get('suburb', '').strip().lower(),
        'ISO3166-2-lvl10': address.get('ISO3166-2-lvl10', '').strip().lower(),
        'city': address.get('city', '').strip().lower(),
        'postcode': address.get('postcode', '').strip().lower(),
        'country': address.get('country', '').strip().lower(),

        # OSV5M original geographic fields - ensure they're strings
        'osv5m_country': get_string_field('country', ''),  # Original country code like 'FR'
        'osv5m_region': get_string_field('region', ''),
        'osv5m_sub_region': get_string_field('sub-region', ''),
        'osv5m_city': get_string_field('city', ''),

        # OSV5M environmental features (unique to this dataset) - keep as numeric
        'land_cover': get_numeric_field('land_cover', None),
        'road_index': get_numeric_field('road_index', None),
        'drive_side': get_numeric_field('drive_side', None),
        'climate': get_numeric_field('climate', None),
        'soil': get_numeric_field('soil', None),
        'dist_sea': get_numeric_field('dist_sea', None),

        # Optional metadata fields - ensure they're strings
        'captured_at': get_string_field('captured_at', ''),
        'sequence': get_string_field('sequence', ''),
        'thumb_original_url': get_string_field('thumb_original_url', ''),
    }

    return ("success", processed_sample)


def scan_subdir(subdir: Path) -> Dict[str, Path]:
    """
    Scan a single subdirectory and return ID->path mapping.
    Must be at module level to be pickle-able for multiprocessing.
    """
    local_map = {}
    for img_path in subdir.glob('*.jpg'):
        img_id = img_path.stem
        local_map[img_id] = img_path
    return local_map


def preprocess_osv5m(
        dataset_split: str,
        service_url: str,
        geogusor_dir: str = None,
        num_workers: int = None,
        chunk_size: int = 500,
        save_merged: bool = True,
        resume: bool = True,
        overwrite: bool = False
):
    """
    Process OSV5M dataset in chunks with parallel processing and resume support

    This version directly loads CSV files and processes images, without using OSV5MDataset class.

    Args:
        dataset_split: Dataset split name ('train' or 'test')
        service_url: Reverse geocoding service URL
        geogusor_dir: GEOGUESSR_DIR environment variable path, defaults to $GEOGUESSR_DIR
        num_workers: Number of parallel workers, defaults to CPU count
        chunk_size: Size of each chunk
        save_merged: Whether to merge all chunks after processing, default True
        resume: Whether to resume from existing chunks (checkpoint), default True
        overwrite: Whether to overwrite existing data (True will delete all existing chunks), default False

    Returns:
        If save_merged=True: (merged_path, chunk_count)
        If save_merged=False: (chunks_dir, chunk_count)
    """
    # Get data directory
    if geogusor_dir is None:
        geogusor_dir = os.environ.get('GEOGUESSR_DIR')
        if geogusor_dir is None:
            raise ValueError("Please set GEOGUESSR_DIR environment variable or pass geogusor_dir parameter")

    # Set number of workers (use more workers for faster processing)
    if num_workers is None:
        num_workers = min(mp.cpu_count(), 64)  # Allow up to 64 workers

    # Construct paths
    osv5m_root = Path(geogusor_dir) / 'osv5m' / 'osv5m'
    csv_path = osv5m_root / f'{dataset_split}.csv'
    images_dir = osv5m_root / 'images' / dataset_split

    if not csv_path.exists():
        raise ValueError(f"CSV file does not exist: {csv_path}")
    if not images_dir.exists():
        raise ValueError(f"Images directory does not exist: {images_dir}")

    print(f"Loading OSV5M CSV from {csv_path}...")

    # Load CSV with only the fields we need
    dtype_dict = {
        'id': str,
        'latitude': float,
        'longitude': float,
        'thumb_original_url': str,
        'country': str,
        'sequence': str,
        'captured_at': str,
        'region': str,
        'sub-region': str,
        'city': str,
        'land_cover': float,
        'road_index': float,
        'drive_side': float,
        'climate': float,
        'soil': float,
        'dist_sea': float,
    }

    df = pd.read_csv(csv_path, dtype=dtype_dict, low_memory=False)
    print(f"Loaded {len(df)} samples from CSV")

    # Build image path mapping: ID -> image_path (parallelized for speed)
    # This follows the same logic as osv5m_dataset.py
    print(f"Building image path mapping from {images_dir}...")

    subdirs = sorted([d for d in images_dir.iterdir() if d.is_dir()])
    print(f"Found {len(subdirs)} subdirectories, scanning with {num_workers} workers...")

    # Parallel scan all subdirectories (scan_subdir is defined at module level)
    image_path_map = {}
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(scan_subdir, subdir): subdir for subdir in subdirs}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Scanning directories"):
            local_map = future.result()
            image_path_map.update(local_map)

    print(f"Found {len(image_path_map)} image files")

    # Filter CSV to only include rows with existing images
    valid_ids = set(image_path_map.keys())
    csv_ids = set(df['id'].values)
    matched_ids = valid_ids & csv_ids

    print(f"CSV has {len(csv_ids)} IDs")
    print(f"Images have {len(valid_ids)} IDs")
    print(f"Matched {len(matched_ids)} IDs ({len(matched_ids)/len(csv_ids)*100:.2f}% of CSV)")

    if len(matched_ids) == 0:
        raise ValueError("No matching IDs found between CSV and images!")

    # Filter dataframe to only matched IDs
    df = df[df['id'].isin(matched_ids)].reset_index(drop=True)
    print(f"Processing {len(df)} samples with valid images")

    # Create output directory
    output_dir = f"{geogusor_dir}/processed/osv5m/"
    chunks_dir = os.path.join(output_dir, f"{dataset_split}_chunks")

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

    process_func = partial(process_osv5m_sample, service_url=service_url)

    print(f"\nProcessing OSV5M/{dataset_split}...")

    # Start from checkpoint position
    chunk_count = start_chunk_idx
    start_sample_idx = start_chunk_idx * chunk_size

    if start_sample_idx > 0:
        print(f"Skipping first {start_sample_idx} samples (already processed)")

    # Process in chunks
    for start_idx in range(start_sample_idx, len(df), chunk_size):
        end_idx = min(start_idx + chunk_size, len(df))
        chunk_df = df.iloc[start_idx:end_idx]

        # Prepare samples for processing
        chunk_data = []
        print(f"Loading chunk data ({start_idx}-{end_idx})...")
        for idx, row in tqdm(chunk_df.iterrows(), total=len(chunk_df), desc="Preparing samples"):
            try:
                # Get image path from the mapping built earlier
                img_id = row['id']
                image_path = image_path_map.get(img_id)

                if image_path is None:
                    # This should not happen since we filtered the dataframe
                    print(f"\nWarning: Image path not found for ID {img_id}")
                    continue

                # Convert row to dict
                metadata = row.to_dict()

                chunk_data.append((image_path, metadata))
            except Exception as e:
                print(f"\nError preparing sample at index {idx}: {e}")
                continue

        chunk_samples = []
        stats = {
            'total': 0,
            'success': 0,
            'image_not_found': 0,
            'image_load_failed': 0,
            'invalid_latlon': 0,
            'processing_error': 0
        }

        # Process current chunk in parallel
        print(f"\nProcessing {len(chunk_data)} samples with {num_workers} workers...")
        process_start = time.time()

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(process_func, image_path, metadata): i
                       for i, (image_path, metadata) in enumerate(chunk_data)}

            for future in tqdm(as_completed(futures), total=len(futures),
                               desc=f"Processing chunk {chunk_count}"):
                stats['total'] += 1
                try:
                    result = future.result()
                    if result is not None:
                        status, data = result
                        if status == "success":
                            chunk_samples.append(data)
                            stats['success'] += 1
                        elif status == "skip":
                            stats[data] += 1
                except Exception as e:
                    stats['processing_error'] += 1
                    print(f"\nError processing sample: {e}")

        process_time = time.time() - process_start
        print(f"\nProcessing completed in {process_time:.2f}s ({stats['total']/process_time:.2f} samples/s)")

        # Print statistics for this chunk
        print(f"\nChunk {chunk_count} statistics:")
        print(f"  Total samples processed: {stats['total']}")
        print(f"  Successfully processed: {stats['success']} ({stats['success']/stats['total']*100:.2f}%)")
        print(f"  Image not found: {stats['image_not_found']} ({stats['image_not_found']/stats['total']*100:.2f}%)")
        print(f"  Image load failed: {stats['image_load_failed']} ({stats['image_load_failed']/stats['total']*100:.2f}%)")
        print(f"  Invalid lat/lon: {stats['invalid_latlon']} ({stats['invalid_latlon']/stats['total']*100:.2f}%)")
        print(f"  Processing error: {stats['processing_error']} ({stats['processing_error']/stats['total']*100:.2f}%)")

        # Save current chunk immediately
        if chunk_samples:
            print(f"\nSaving chunk {chunk_count} with {len(chunk_samples)} samples...")
            save_start = time.time()

            # Create dataset from samples with explicit Image feature
            # This tells datasets that 'image' field contains image data in bytes format
            print(f"  Creating Dataset...")
            dataset_start = time.time()

            # Define features to ensure proper image handling
            features = Features({
                'image': ImageFeature(),
                'lon': Value('float64'),
                'lat': Value('float64'),
                'image_source': Value('string'),
                'source': Value('string'),
                'messages': Value('string'),
                'road': Value('string'),
                'suburb': Value('string'),
                'ISO3166-2-lvl10': Value('string'),
                'city': Value('string'),
                'postcode': Value('string'),
                'country': Value('string'),
                'osv5m_country': Value('string'),
                'osv5m_region': Value('string'),
                'osv5m_sub_region': Value('string'),
                'osv5m_city': Value('string'),
                'land_cover': Value('float64'),
                'road_index': Value('float64'),
                'drive_side': Value('float64'),
                'climate': Value('float64'),
                'soil': Value('float64'),
                'dist_sea': Value('float64'),
                'captured_at': Value('string'),
                'sequence': Value('string'),
                'thumb_original_url': Value('string'),
            })

            chunk_dataset = Dataset.from_list(chunk_samples, features=features)
            print(f"  Dataset created in {time.time() - dataset_start:.2f}s")

            # Save to disk
            print(f"  Writing to disk...")
            disk_start = time.time()
            chunk_path = os.path.join(chunks_dir, f"chunk_{chunk_count:04d}")
            chunk_dataset.save_to_disk(chunk_path)
            print(f"  Saved to disk in {time.time() - disk_start:.2f}s")

            print(f"Chunk {chunk_count} saved in {time.time() - save_start:.2f}s total")
            chunk_count += 1

            # Release memory
            del chunk_samples
            del chunk_data
            del chunk_dataset

    print(f"\nTotal saved {chunk_count} chunks to: {chunks_dir}")

    # Merge all chunks if needed
    if save_merged:
        # Use glob to find all actual chunks (not just the ones created in this run)
        all_chunk_paths = sorted(glob.glob(os.path.join(chunks_dir, "chunk_*")))

        if not all_chunk_paths:
            print(f"\nNo chunks found in {chunks_dir}, skipping merge")
            return chunks_dir, chunk_count

        print(f"\nMerging all {len(all_chunk_paths)} chunks...")
        merged_path = os.path.join(output_dir, dataset_split)

        # Load all chunks
        chunk_datasets = []
        for chunk_path in tqdm(all_chunk_paths, desc="Loading chunks"):
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


def load_processed_osv5m(
        dataset_split: str,
        geogusor_dir: str = None,
        from_chunks: bool = False
) -> Dataset:
    """
    Load processed OSV5M dataset

    Args:
        dataset_split: Dataset split ('train' or 'test')
        geogusor_dir: GEOGUESSR_DIR environment variable path, defaults to $GEOGUESSR_DIR
        from_chunks: Whether to load from chunks (will auto load from chunks if no merged dataset)

    Returns:
        Dataset: Loaded dataset
    """
    # Get data directory
    if geogusor_dir is None:
        geogusor_dir = os.environ.get('GEOGUESSR_DIR')
        if geogusor_dir is None:
            raise ValueError("Please set GEOGUESSR_DIR environment variable or pass geogusor_dir parameter")

    output_dir = f"{geogusor_dir}/processed/osv5m/"
    merged_path = os.path.join(output_dir, dataset_split)
    chunks_dir = os.path.join(output_dir, f"{dataset_split}_chunks")

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
        f"No processed dataset found. Please run preprocess_osv5m to process data first.\n"
        f"Search paths: {merged_path} or {chunks_dir}"
    )


# Usage example
if __name__ == "__main__":
    # Service URL
    service_url = "http://dsw-notebook-dsw-s9abjgcsvnls12nt68-8080.vpc-t4nptn1gzsxxk04su5mpc.instance-forward.dsw.ap-southeast-1.aliyuncs.com:8080"

    geogusor_dir = os.environ.get('GEOGUESSR_DIR')
    assert geogusor_dir is not None, "GEOGUESSR_DIR is required in environment variables."

    # Example 1: Process OSV5M train set (with checkpoint resume support)
    result_train, chunk_count_train = preprocess_osv5m(
        dataset_split='train',
        service_url=service_url,
        geogusor_dir=geogusor_dir,
        num_workers=64,  # Adjust based on CPU cores
        chunk_size=50000,  # Adjust based on memory
        save_merged=True,  # Auto merge all chunks
        resume=True  # If interrupted, will auto resume from checkpoint (default True)
    )

    # Example 2: Process OSV5M test set
    result_test, chunk_count_test = preprocess_osv5m(
        dataset_split='test',
        service_url=service_url,
        geogusor_dir=geogusor_dir,
        num_workers=64,
        chunk_size=50000,
        save_merged=True
    )

    # Example 3: If need to reprocess from scratch (delete existing data)
    # result_train, chunk_count_train = preprocess_osv5m(
    #     dataset_split='train',
    #     service_url=service_url,
    #     geogusor_dir=geogusor_dir,
    #     num_workers=16,
    #     chunk_size=50000,
    #     save_merged=True,
    #     overwrite=True  # Delete all existing data and start from scratch
    # )

    print("\n" + "=" * 60)
    print("Processing complete!")
    print("=" * 60)
    print(f"\nOSV5M-train: {chunk_count_train} chunks -> {result_train}")
    print(f"OSV5M-test: {chunk_count_test} chunks -> {result_test}")

    print("\nTo load data:")
    print("# Method 1: Use helper function (recommended)")
    print("from preprocess_osv5m import load_processed_osv5m")
    print("train_dataset = load_processed_osv5m('train')")
    print("test_dataset = load_processed_osv5m('test')")
    print()
    print("# Method 2: Direct load_from_disk")
    print("from datasets import load_from_disk")
    print(f"train_dataset = load_from_disk('{result_train}')")
    print(f"test_dataset = load_from_disk('{result_test}')")
    print()
    print("\nCheckpoint resume notes:")
    print("- If interrupted, rerunning the script will auto resume from where it stopped")
    print("- Set resume=False to raise error when existing data detected, prevent accidental overwrite")
    print("- Set overwrite=True to delete all existing data and start from scratch")
    print()
    print("\nOSV5My	W��:")
    print("- osv5m_country/region/sub_region/city: OSV5M�˄0�oW�")
    print("- land_cover, climate, soilI: ��y�OSV5MpnƄyr")
    print("- road/suburb/city/countryI: �Nominatim0�ք�0@�o")
