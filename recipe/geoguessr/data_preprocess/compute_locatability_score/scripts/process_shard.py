#!/usr/bin/env python3
"""
Single-GPU worker for computing locatability scores.

This script processes a shard of the dataset on a single GPU.
Multiple instances can run in parallel on different GPUs.
"""

import os
import sys
import json
import argparse
import pickle
from pathlib import Path
from typing import Optional

import torch
import datasets
from datasets import Dataset
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import SEMANTIC_CLASSES, CLASS_WEIGHTS, compute_locatability_score


class ImageDatasetWrapper(torch.utils.data.Dataset):
    """Wrapper for HuggingFace Dataset."""

    def __init__(self, hf_dataset: Dataset, start_idx: int = 0):
        self.dataset = hf_dataset
        self.start_idx = start_idx

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image']

        # Ensure RGB
        if isinstance(image, Image.Image) and image.mode != 'RGB':
            image = image.convert('RGB')

        return {
            'image': image,
            'original_size': image.size[::-1],  # (H, W)
            'index': self.start_idx + idx
        }


def custom_collate_fn(batch):
    """Handle None values in batch."""
    valid_items = [item for item in batch if item['image'] is not None]

    if not valid_items:
        return None

    return {
        'images': [item['image'] for item in valid_items],
        'original_sizes': [item['original_size'] for item in valid_items],
        'indices': [item['index'] for item in valid_items]
    }


def load_checkpoint(checkpoint_path: str) -> dict:
    """Load processing checkpoint."""
    if not os.path.exists(checkpoint_path):
        return {'processed_indices': set(), 'scores': {}, 'mappings': {}}

    with open(checkpoint_path, 'rb') as f:
        return pickle.load(f)


def save_checkpoint(checkpoint_path: str, data: dict):
    """Save processing checkpoint."""
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    with open(checkpoint_path, 'wb') as f:
        pickle.dump(data, f)


def process_shard(
    input_dataset_path: str,
    shard_id: int,
    total_shards: int,
    output_dir: str,
    gpu_id: int = 0,
    batch_size: int = 20,
    num_workers: int = 2,
    model_id: str = "facebook/mask2former-swin-large-ade-semantic",
    use_amp: bool = True,
    checkpoint_interval: int = 100,
    resume: bool = True
):
    """
    Process a shard of the dataset on a single GPU.

    Args:
        input_dataset_path: Path to input HuggingFace dataset
        shard_id: Shard ID (0-indexed)
        total_shards: Total number of shards
        output_dir: Directory to save output and checkpoints
        gpu_id: GPU device ID
        batch_size: Batch size for processing
        num_workers: Number of dataloader workers
        model_id: HuggingFace model ID
        use_amp: Use automatic mixed precision
        checkpoint_interval: Save checkpoint every N batches
        resume: Resume from checkpoint if exists
    """
    print(f"[Shard {shard_id}] Starting processing on GPU {gpu_id}")
    print(f"[Shard {shard_id}] Loading dataset from: {input_dataset_path}")

    # Setup device
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"[Shard {shard_id}] Using device: {device}")

    # Load full dataset
    full_dataset = datasets.load_from_disk(input_dataset_path)
    total_samples = len(full_dataset)
    print(f"[Shard {shard_id}] Full dataset size: {total_samples}")

    # Compute shard boundaries
    samples_per_shard = total_samples // total_shards
    start_idx = shard_id * samples_per_shard
    if shard_id == total_shards - 1:
        end_idx = total_samples
    else:
        end_idx = start_idx + samples_per_shard

    print(f"[Shard {shard_id}] Processing samples [{start_idx}, {end_idx}) ({end_idx - start_idx} samples)")

    # Extract shard
    shard_dataset = full_dataset.select(range(start_idx, end_idx))
    shard_size = len(shard_dataset)

    # Setup checkpoint
    checkpoint_path = os.path.join(output_dir, f"checkpoint_shard_{shard_id}.pkl")
    checkpoint_data = load_checkpoint(checkpoint_path) if resume else {
        'processed_indices': set(),
        'scores': {},
        'mappings': {}
    }

    already_processed = len(checkpoint_data['processed_indices'])
    if already_processed > 0:
        print(f"[Shard {shard_id}] Resuming from checkpoint: {already_processed}/{shard_size} already processed")

    # Load model
    print(f"[Shard {shard_id}] Loading model: {model_id}")
    try:
        processor = AutoImageProcessor.from_pretrained(model_id)
        model = Mask2FormerForUniversalSegmentation.from_pretrained(model_id)
    except OSError:
        print(f"[Shard {shard_id}] Failed to load {model_id}, trying fallback...")
        model_id = "facebook/maskformer-swin-large-ade-semantic"
        processor = AutoImageProcessor.from_pretrained(model_id)
        model = Mask2FormerForUniversalSegmentation.from_pretrained(model_id)

    model.to(device)
    model.eval()
    print(f"[Shard {shard_id}] Model loaded successfully")

    # Prepare weights
    weights_tensor = torch.tensor(CLASS_WEIGHTS, dtype=torch.float32, device=device)

    # Create dataloader
    wrapped_dataset = ImageDatasetWrapper(shard_dataset, start_idx=start_idx)
    dataloader = DataLoader(
        wrapped_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=custom_collate_fn,
        pin_memory=(device.type == 'cuda')
    )

    # Process batches
    print(f"[Shard {shard_id}] Processing {shard_size} samples in batches of {batch_size}")
    batch_count = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"GPU {gpu_id} Shard {shard_id}"):
            if batch is None:
                continue

            batch_images = batch['images']
            batch_sizes = batch['original_sizes']
            batch_indices = batch['indices']

            # Skip already processed
            to_process_mask = [idx not in checkpoint_data['processed_indices'] for idx in batch_indices]
            if not any(to_process_mask):
                continue

            # Filter batch
            batch_images = [img for img, keep in zip(batch_images, to_process_mask) if keep]
            batch_sizes = [size for size, keep in zip(batch_sizes, to_process_mask) if keep]
            batch_indices = [idx for idx, keep in zip(batch_indices, to_process_mask) if keep]

            if not batch_images:
                continue

            try:
                # Preprocess
                inputs = processor(images=batch_images, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}

                # Forward pass
                with autocast(enabled=(use_amp and device.type == 'cuda')):
                    outputs = model(**inputs)

                # Post-process
                semantic_maps = processor.post_process_semantic_segmentation(
                    outputs,
                    target_sizes=batch_sizes
                )

                # Compute scores
                for semantic_map, idx in zip(semantic_maps, batch_indices):
                    score, mapping = compute_locatability_score(
                        semantic_map,
                        weights_tensor,
                        SEMANTIC_CLASSES
                    )

                    checkpoint_data['scores'][idx] = score
                    checkpoint_data['mappings'][idx] = mapping
                    checkpoint_data['processed_indices'].add(idx)

            except Exception as e:
                print(f"\n[Shard {shard_id}] Error processing batch: {e}")
                # Fill with default values
                for idx in batch_indices:
                    if idx not in checkpoint_data['processed_indices']:
                        checkpoint_data['scores'][idx] = 0.0
                        checkpoint_data['mappings'][idx] = json.dumps({})
                        checkpoint_data['processed_indices'].add(idx)

            # Save checkpoint
            batch_count += 1
            if batch_count % checkpoint_interval == 0:
                save_checkpoint(checkpoint_path, checkpoint_data)
                print(f"[Shard {shard_id}] Checkpoint saved: {len(checkpoint_data['processed_indices'])}/{shard_size}")

    # Final checkpoint
    save_checkpoint(checkpoint_path, checkpoint_data)

    # Save final results
    output_path = os.path.join(output_dir, f"shard_{shard_id}.pkl")
    with open(output_path, 'wb') as f:
        pickle.dump({
            'shard_id': shard_id,
            'start_idx': start_idx,
            'end_idx': end_idx,
            'scores': checkpoint_data['scores'],
            'mappings': checkpoint_data['mappings']
        }, f)

    print(f"[Shard {shard_id}] ✓ Processing complete!")
    print(f"[Shard {shard_id}] Results saved to: {output_path}")

    # Statistics
    valid_scores = [s for s in checkpoint_data['scores'].values() if s > 0]
    if valid_scores:
        print(f"[Shard {shard_id}] Score statistics:")
        print(f"  Min: {min(valid_scores):.4f}")
        print(f"  Max: {max(valid_scores):.4f}")
        print(f"  Mean: {sum(valid_scores)/len(valid_scores):.4f}")


def main():
    parser = argparse.ArgumentParser(description="Process a dataset shard on single GPU")

    parser.add_argument("--input_dataset", type=str, required=True,
                      help="Path to input HuggingFace dataset")
    parser.add_argument("--shard_id", type=int, required=True,
                      help="Shard ID (0-indexed)")
    parser.add_argument("--total_shards", type=int, required=True,
                      help="Total number of shards")
    parser.add_argument("--output_dir", type=str, required=True,
                      help="Output directory for results")
    parser.add_argument("--gpu_id", type=int, default=0,
                      help="GPU device ID")
    parser.add_argument("--batch_size", type=int, default=20,
                      help="Batch size")
    parser.add_argument("--num_workers", type=int, default=2,
                      help="Number of dataloader workers")
    parser.add_argument("--model_id", type=str,
                      default="facebook/mask2former-swin-large-ade-semantic",
                      help="HuggingFace model ID")
    parser.add_argument("--no_amp", action="store_true",
                      help="Disable automatic mixed precision")
    parser.add_argument("--checkpoint_interval", type=int, default=100,
                      help="Save checkpoint every N batches")
    parser.add_argument("--no_resume", action="store_true",
                      help="Don't resume from checkpoint")

    args = parser.parse_args()

    process_shard(
        input_dataset_path=args.input_dataset,
        shard_id=args.shard_id,
        total_shards=args.total_shards,
        output_dir=args.output_dir,
        gpu_id=args.gpu_id,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        model_id=args.model_id,
        use_amp=not args.no_amp,
        checkpoint_interval=args.checkpoint_interval,
        resume=not args.no_resume
    )


if __name__ == "__main__":
    main()
