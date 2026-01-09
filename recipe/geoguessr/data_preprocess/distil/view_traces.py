#!/usr/bin/env python3
"""
Utility script to view and analyze generated traces.
"""

import json
import sys
import re
from pathlib import Path
from typing import Optional, Tuple


def parse_coordinates_from_text(text: str) -> Optional[Tuple[float, float]]:
    """Parse coordinates from model output."""
    # Try to find \boxed{lat, lon}
    boxed_match = re.search(r'\\boxed\{([^}]+)\}', text)
    if boxed_match:
        coords_str = boxed_match.group(1)
        parts = coords_str.split(',')
        if len(parts) == 2:
            try:
                lat = float(parts[0].strip())
                lon = float(parts[1].strip())
                if -90 <= lat <= 90 and -180 <= lon <= 180:
                    return (lat, lon)
            except ValueError:
                pass

    # Fallback patterns
    patterns = [
        r'lat(?:itude)?[:\s]+(-?\d+\.?\d*)[,\s]+lon(?:gitude)?[:\s]+(-?\d+\.?\d*)',
        r'\((-?\d+\.?\d*)[,\s]+(-?\d+\.?\d*)\)'
    ]

    for pattern in patterns:
        match = re.search(pattern, text.lower())
        if match:
            try:
                lat = float(match.group(1))
                lon = float(match.group(2))
                if -90 <= lat <= 90 and -180 <= lon <= 180:
                    return (lat, lon)
            except ValueError:
                pass

    return None


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance between two coordinates in km."""
    import math

    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))

    r = 6371.0  # Earth radius in km
    return c * r


def view_trace(trace_path: str, verbose: bool = False):
    """View and analyze a single trace."""
    with open(trace_path, 'r') as f:
        trace = json.load(f)

    sample = trace['sample_data']
    response = trace['response_text']

    print("=" * 80)
    print(f"Trace: {Path(trace_path).name}")
    print("=" * 80)

    # Sample info
    print("\n📍 Sample Information:")
    print(f"  Locatability Score: {sample['locatability_score']:.4f}")
    print(f"  Ground Truth: lat={sample['lat']:.4f}, lon={sample['lon']:.4f}")
    print(f"  Country: {sample.get('country', 'N/A')}")
    print(f"  City: {sample.get('city', 'N/A')}")
    print(f"  Image Source: {sample.get('image_source', 'N/A')}")

    # Token usage
    usage = trace['api_response']['usage']
    print(f"\n📊 Token Usage:")
    print(f"  Prompt: {usage['prompt_tokens']}")
    print(f"  Completion: {usage['completion_tokens']}")
    print(f"  Total: {usage['total_tokens']}")

    # Parse prediction
    pred_coords = parse_coordinates_from_text(response)
    print(f"\n🎯 Prediction:")
    if pred_coords:
        pred_lat, pred_lon = pred_coords
        print(f"  Predicted: lat={pred_lat:.4f}, lon={pred_lon:.4f}")

        # Calculate distance
        distance = haversine_distance(sample['lat'], sample['lon'], pred_lat, pred_lon)
        print(f"  Distance Error: {distance:.2f} km")

        # Accuracy thresholds
        thresholds = [1, 25, 200, 750, 2500]
        print(f"  Accuracy:")
        for t in thresholds:
            status = "✅" if distance <= t else "❌"
            print(f"    @{t:4d}km: {status}")
    else:
        print(f"  ❌ Failed to parse coordinates from response")

    # Response preview
    print(f"\n💭 Response ({len(response)} chars):")
    print("-" * 80)
    if verbose:
        print(response)
    else:
        # Show first 500 and last 300 chars
        if len(response) > 800:
            print(response[:500])
            print("\n... [truncated] ...\n")
            print(response[-300:])
        else:
            print(response)

    print("=" * 80)


def analyze_batch(traces_dir: str):
    """Analyze all traces in a directory."""
    traces_dir = Path(traces_dir)
    trace_files = sorted(traces_dir.glob("trace_*.json"))

    if not trace_files:
        print(f"No trace files found in {traces_dir}")
        return

    print("=" * 80)
    print(f"Batch Analysis: {len(trace_files)} traces")
    print("=" * 80)

    total_distance = 0
    parse_success = 0
    accuracy_counts = {1: 0, 25: 0, 200: 0, 750: 0, 2500: 0}
    total_tokens = 0

    for trace_file in trace_files:
        with open(trace_file, 'r') as f:
            trace = json.load(f)

        sample = trace['sample_data']
        response = trace['response_text']
        usage = trace['api_response']['usage']

        total_tokens += usage['total_tokens']

        # Parse coordinates
        pred_coords = parse_coordinates_from_text(response)
        if pred_coords:
            parse_success += 1
            pred_lat, pred_lon = pred_coords
            distance = haversine_distance(sample['lat'], sample['lon'], pred_lat, pred_lon)
            total_distance += distance

            # Check accuracy
            for threshold in accuracy_counts.keys():
                if distance <= threshold:
                    accuracy_counts[threshold] += 1

    # Print statistics
    print(f"\n📊 Statistics:")
    print(f"  Total traces: {len(trace_files)}")
    print(f"  Parse success rate: {parse_success}/{len(trace_files)} ({parse_success/len(trace_files)*100:.1f}%)")

    if parse_success > 0:
        print(f"  Average distance error: {total_distance/parse_success:.2f} km")
        print(f"\n🎯 Accuracy:")
        for threshold in [1, 25, 200, 750, 2500]:
            count = accuracy_counts[threshold]
            pct = count / len(trace_files) * 100
            print(f"    @{threshold:4d}km: {count:3d}/{len(trace_files)} ({pct:5.1f}%)")

    print(f"\n💰 Token Usage:")
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Average per trace: {total_tokens/len(trace_files):.0f}")

    print("=" * 80)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="View and analyze generated traces")

    parser.add_argument(
        "path",
        type=str,
        help="Path to trace file or directory"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show full response text"
    )

    parser.add_argument(
        "--batch",
        action="store_true",
        help="Analyze all traces in directory"
    )

    args = parser.parse_args()

    path = Path(args.path)

    if path.is_file():
        view_trace(str(path), verbose=args.verbose)
    elif path.is_dir():
        if args.batch:
            analyze_batch(str(path))
        else:
            # Show all traces
            trace_files = sorted(path.glob("trace_*.json"))
            for trace_file in trace_files:
                view_trace(str(trace_file), verbose=args.verbose)
                print("\n")
    else:
        print(f"Error: {path} does not exist")
        sys.exit(1)


if __name__ == "__main__":
    main()
