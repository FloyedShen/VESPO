#!/usr/bin/env python3
"""
Convert all checkpoints in ./checkpoints/*/actor to HuggingFace format.

This script:
1. Scans ./checkpoints/ for all */actor directories
2. Converts them to HuggingFace format using verl.model_merger
3. Saves to ./ckpt_hf/ with the same structure but without the 'actor' level
4. Supports incremental processing (skips already converted checkpoints)

Example:
    Input:  ./checkpoints/project/experiment/global_steps_100/actor
    Output: ./ckpt_hf/project/experiment/global_steps_100

Usage:
    python convert_checkpoints.py [--backend {fsdp,megatron}] [--checkpoints-dir CHECKPOINTS_DIR] [--output-dir OUTPUT_DIR] [--dry-run]
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple


def detect_backend(actor_path: Path) -> str:
    """
    Detect the backend type (fsdp or megatron) from the checkpoint structure.

    Args:
        actor_path: Path to the actor directory

    Returns:
        'fsdp' or 'megatron'
    """
    # Check for FSDP indicators
    fsdp_config = actor_path / "fsdp_config.json"
    if fsdp_config.exists():
        return "fsdp"

    # Check for Megatron indicators
    dist_ckpt = actor_path / "dist_ckpt"
    if dist_ckpt.exists() and dist_ckpt.is_dir():
        return "megatron"

    # Check for model files (FSDP pattern)
    model_files = list(actor_path.glob("model_world_size_*.pt"))
    if model_files:
        return "fsdp"

    # Default to fsdp if uncertain
    print(f"Warning: Cannot definitively determine backend for {actor_path}, defaulting to fsdp")
    return "fsdp"


def find_actor_checkpoints(checkpoints_dir: Path) -> List[Path]:
    """
    Find all actor checkpoint directories.

    Args:
        checkpoints_dir: Root checkpoints directory

    Returns:
        List of paths to actor directories
    """
    actor_paths = []

    # Find all directories named 'actor'
    for actor_path in checkpoints_dir.rglob("actor"):
        if actor_path.is_dir():
            # Verify it's a valid checkpoint (has huggingface subdir or checkpoint files)
            hf_dir = actor_path / "huggingface"
            dist_ckpt = actor_path / "dist_ckpt"
            model_files = list(actor_path.glob("model_world_size_*.pt"))

            if hf_dir.exists() or dist_ckpt.exists() or model_files:
                actor_paths.append(actor_path)

    return sorted(actor_paths)


def get_output_path(actor_path: Path, checkpoints_dir: Path, output_dir: Path) -> Path:
    """
    Calculate the output path by removing the 'actor' level.

    Args:
        actor_path: Path to the actor directory
        checkpoints_dir: Root checkpoints directory
        output_dir: Root output directory

    Returns:
        Output path for the converted checkpoint
    """
    # Get relative path from checkpoints_dir to actor_path
    rel_path = actor_path.relative_to(checkpoints_dir)

    # Remove 'actor' from the end
    parts = rel_path.parts[:-1]  # Remove 'actor'

    # Construct output path
    output_path = output_dir / Path(*parts) if parts else output_dir

    return output_path


def is_checkpoint_converted(output_path: Path) -> bool:
    """
    Check if a checkpoint has already been converted.

    Args:
        output_path: Path where the converted checkpoint would be

    Returns:
        True if already converted, False otherwise
    """
    if not output_path.exists():
        return False

    # Check for key HuggingFace model files
    config_json = output_path / "config.json"
    model_safetensors = output_path / "model.safetensors"
    model_bin = list(output_path.glob("pytorch_model*.bin"))

    # Consider it converted if config.json exists and at least one model file
    return config_json.exists() and (model_safetensors.exists() or len(model_bin) > 0)


def convert_checkpoint(
    actor_path: Path,
    output_path: Path,
    backend: str,
    additional_args: List[str] = None
) -> Tuple[bool, str]:
    """
    Convert a single checkpoint using verl.model_merger.

    Args:
        actor_path: Path to the actor directory
        output_path: Path to save the converted checkpoint
        backend: 'fsdp' or 'megatron'
        additional_args: Additional arguments to pass to model_merger

    Returns:
        Tuple of (success, message)
    """
    # Prepare command
    cmd = [
        sys.executable,
        "-m", "verl.model_merger",
        "merge",
        "--backend", backend,
        "--local_dir", str(actor_path),
        "--target_dir", str(output_path),
    ]

    # Add additional arguments if provided
    if additional_args:
        cmd.extend(additional_args)

    print(f"Running: {' '.join(cmd)}")

    try:
        # Run the conversion
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        error_msg = f"Error: {e.stderr}"
        return False, error_msg


def main():
    parser = argparse.ArgumentParser(
        description="Convert actor checkpoints to HuggingFace format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--backend",
        choices=["fsdp", "megatron", "auto"],
        default="auto",
        help="Backend type (auto will detect automatically)"
    )
    parser.add_argument(
        "--checkpoints-dir",
        type=Path,
        default=Path("./checkpoints"),
        help="Root directory containing checkpoints (default: ./checkpoints)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./ckpt_hf"),
        help="Output directory for converted checkpoints (default: ./ckpt_hf)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be converted without actually converting"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reconversion even if checkpoint already exists"
    )
    parser.add_argument(
        "--tie-word-embedding",
        action="store_true",
        help="Pass --tie-word-embedding to model_merger (Megatron only)"
    )
    parser.add_argument(
        "--is-value-model",
        action="store_true",
        help="Pass --is-value-model to model_merger (Megatron only)"
    )

    args = parser.parse_args()

    # Validate directories
    if not args.checkpoints_dir.exists():
        print(f"Error: Checkpoints directory does not exist: {args.checkpoints_dir}")
        return 1

    # Find all actor checkpoints
    print(f"Scanning {args.checkpoints_dir} for actor checkpoints...")
    actor_paths = find_actor_checkpoints(args.checkpoints_dir)

    if not actor_paths:
        print("No actor checkpoints found.")
        return 0

    print(f"Found {len(actor_paths)} actor checkpoint(s):")
    for path in actor_paths:
        rel_path = path.relative_to(args.checkpoints_dir)
        print(f"  - {rel_path}")
    print()

    # Prepare additional arguments
    additional_args = []
    if args.tie_word_embedding:
        additional_args.append("--tie-word-embedding")
    if args.is_value_model:
        additional_args.append("--is-value-model")

    # Process each checkpoint
    success_count = 0
    skip_count = 0
    error_count = 0

    for i, actor_path in enumerate(actor_paths, 1):
        rel_path = actor_path.relative_to(args.checkpoints_dir)
        output_path = get_output_path(actor_path, args.checkpoints_dir, args.output_dir)

        print(f"[{i}/{len(actor_paths)}] Processing: {rel_path}")
        print(f"  Output: {output_path.relative_to(Path.cwd()) if output_path.is_relative_to(Path.cwd()) else output_path}")

        # Check if already converted
        if not args.force and is_checkpoint_converted(output_path):
            print(f"  Status: SKIPPED (already converted)")
            skip_count += 1
            print()
            continue

        # Detect backend if auto
        backend = args.backend
        if backend == "auto":
            backend = detect_backend(actor_path)
            print(f"  Detected backend: {backend}")
        else:
            print(f"  Backend: {backend}")

        # Dry run
        if args.dry_run:
            print(f"  Status: WOULD CONVERT")
            print()
            continue

        # Create output directory
        output_path.mkdir(parents=True, exist_ok=True)

        # Convert
        success, message = convert_checkpoint(
            actor_path,
            output_path,
            backend,
            additional_args
        )

        if success:
            print(f"  Status: SUCCESS")
            success_count += 1
        else:
            print(f"  Status: FAILED")
            print(f"  {message}")
            error_count += 1

        print()

    # Summary
    print("=" * 60)
    print("Conversion Summary:")
    print(f"  Total found:      {len(actor_paths)}")
    print(f"  Successfully converted: {success_count}")
    print(f"  Skipped (already converted): {skip_count}")
    print(f"  Failed:           {error_count}")
    print("=" * 60)

    return 1 if error_count > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
