#!/usr/bin/env python3
"""
Quick script to check checkpoint status and progress.
"""

import sys
import json
from pathlib import Path


def check_checkpoint(output_dir: str):
    """Check checkpoint status."""
    checkpoint_path = Path(output_dir) / "checkpoint.json"

    if not checkpoint_path.exists():
        print(f"❌ No checkpoint found in: {output_dir}")
        print(f"   Expected: {checkpoint_path}")
        return

    try:
        with open(checkpoint_path) as f:
            data = json.load(f)

        processed = data.get('processed_indices', [])
        failed = data.get('failed_indices', [])
        timestamp = data.get('timestamp', 0)

        print("=" * 60)
        print(f"Checkpoint Status: {output_dir}")
        print("=" * 60)
        print(f"Processed samples: {len(processed)}")
        print(f"Failed samples: {len(failed)}")
        print(f"Total attempts: {len(processed) + len(failed)}")

        if timestamp:
            from datetime import datetime
            dt = datetime.fromtimestamp(timestamp)
            print(f"Last update: {dt.strftime('%Y-%m-%d %H:%M:%S')}")

        # Count actual trace files
        output_path = Path(output_dir)
        trace_files = list(output_path.glob("trace_*.json"))
        print(f"\nActual trace files: {len(trace_files)}")

        # Sample some processed indices
        if len(processed) > 0:
            sample_size = min(10, len(processed))
            sample_indices = sorted(processed)[:sample_size]
            print(f"\nFirst {sample_size} processed indices:")
            print(f"  {sample_indices}")

        # Show failed indices
        if len(failed) > 0:
            print(f"\nFailed indices: {sorted(failed)}")

        print("=" * 60)

    except Exception as e:
        print(f"❌ Error reading checkpoint: {e}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 check_checkpoint.py <output_dir>")
        print("\nExample:")
        print("  python3 check_checkpoint.py traces_production_1k")
        sys.exit(1)

    check_checkpoint(sys.argv[1])
