#!/usr/bin/env python3
"""
Enhanced utility to view and analyze traces with tool calls and reward scores.
"""

import json
import sys
from pathlib import Path
from typing import Optional


def view_trace_with_tools(trace_path: str, verbose: bool = False):
    """View and analyze a single trace with tool support."""
    with open(trace_path, 'r') as f:
        trace = json.load(f)

    print("=" * 80)
    print(f"Trace: {Path(trace_path).name}")
    print("=" * 80)

    # Dataset info
    print("\n📂 Dataset Information:")
    print(f"  Dataset path: {trace['dataset_path']}")
    print(f"  Sample index: {trace['sample_index']}")

    # Sample info
    sample = trace['sample_data']
    print(f"\n📍 Sample Information:")
    print(f"  Locatability Score: {sample['locatability_score']:.4f}")
    print(f"  Ground Truth: lat={sample['lat']:.4f}, lon={sample['lon']:.4f}")
    print(f"  Country: {sample.get('country', 'N/A')}")
    print(f"  City: {sample.get('city', 'N/A')}")
    print(f"  Image Source: {sample.get('image_source', 'N/A')}")

    # Conversation stats
    metadata = trace.get('metadata', {})
    print(f"\n💬 Conversation Statistics:")
    print(f"  Total turns: {len(trace['conversation_history'])}")
    print(f"  Tool calls: {metadata.get('num_tool_calls', 0)}")

    # Token usage
    total_tokens = sum(
        turn.get('usage', {}).get('total_tokens', 0)
        for turn in trace['conversation_history']
    )
    print(f"  Total tokens: {total_tokens}")

    # Tool calls details
    if trace['tool_calls']:
        print(f"\n🔧 Tool Calls:")
        for i, tool_call_info in enumerate(trace['tool_calls'], 1):
            tool_call = tool_call_info['tool_call']
            print(f"  {i}. Turn {tool_call_info['turn']}: {tool_call['name']}")
            print(f"     Arguments: {tool_call['arguments']}")
            print(f"     Success: {tool_call_info['success']}")

    # Reward score
    reward = trace['reward_score']
    print(f"\n🎯 Reward Score:")
    print(f"  Parse Success: {'✅' if reward.get('parse_success') else '❌'}")

    if reward.get('parse_success'):
        print(f"  Distance: {reward.get('distance@km', 0):.2f} km")
        print(f"  Score: {reward.get('score', 0):.4f}")
        print(f"  GeoGuessr Points: {reward.get('geoguessr@point', 0):.0f}/5000")

        # Accuracy metrics
        print(f"  Accuracy:")
        for threshold in [1, 25, 200, 750, 2500]:
            acc = reward.get(f'acc@{threshold}km', 0)
            status = "✅" if acc > 0 else "❌"
            print(f"    @{threshold:4d}km: {status}")

    # Final response
    final_response = trace['final_response']
    print(f"\n💭 Final Response ({len(final_response)} chars):")
    print("-" * 80)
    if verbose:
        print(final_response)
    else:
        # Show first 300 and last 200 chars
        if len(final_response) > 500:
            print(final_response[:300])
            print("\n... [truncated] ...\n")
            print(final_response[-200:])
        else:
            print(final_response)

    # Conversation history (if verbose)
    if verbose:
        print(f"\n📜 Conversation History:")
        print("-" * 80)
        for i, turn in enumerate(trace['conversation_history'], 1):
            print(f"\n--- Turn {i} ---")
            print(f"Response: {turn['response'][:200]}...")
            print(f"Tokens: {turn.get('usage', {}).get('total_tokens', 0)}")

    print("=" * 80)


def analyze_batch_with_tools(traces_dir: str):
    """Analyze all traces in a directory."""
    traces_dir = Path(traces_dir)
    trace_files = sorted(traces_dir.glob("trace_*.json"))

    if not trace_files:
        print(f"No trace files found in {traces_dir}")
        return

    print("=" * 80)
    print(f"Batch Analysis: {len(trace_files)} traces")
    print("=" * 80)

    # Statistics
    total_tokens = 0
    total_turns = 0
    total_tool_calls = 0
    total_distance = 0
    parse_success = 0
    accuracy_counts = {1: 0, 25: 0, 200: 0, 750: 0, 2500: 0}

    for trace_file in trace_files:
        with open(trace_file, 'r') as f:
            trace = json.load(f)

        # Count tokens
        tokens = sum(
            turn.get('usage', {}).get('total_tokens', 0)
            for turn in trace['conversation_history']
        )
        total_tokens += tokens

        # Count turns and tool calls
        total_turns += len(trace['conversation_history'])
        total_tool_calls += len(trace['tool_calls'])

        # Reward metrics
        reward = trace['reward_score']
        if reward.get('parse_success'):
            parse_success += 1
            distance = reward.get('distance@km', 0)
            total_distance += distance

            # Check accuracy
            for threshold in accuracy_counts.keys():
                if reward.get(f'acc@{threshold}km', 0) > 0:
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

    print(f"\n💬 Conversation:")
    print(f"  Total turns: {total_turns}")
    print(f"  Average turns per trace: {total_turns/len(trace_files):.1f}")
    print(f"  Total tool calls: {total_tool_calls}")
    print(f"  Average tool calls per trace: {total_tool_calls/len(trace_files):.2f}")
    print(f"  Tool usage rate: {total_tool_calls/total_turns*100:.1f}% of turns")

    print(f"\n💰 Token Usage:")
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Average per trace: {total_tokens/len(trace_files):.0f}")
    print(f"  Average per turn: {total_tokens/total_turns:.0f}")

    print("=" * 80)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="View and analyze traces with tool support")

    parser.add_argument(
        "path",
        type=str,
        help="Path to trace file or directory"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show full response text and conversation history"
    )

    parser.add_argument(
        "--batch",
        action="store_true",
        help="Analyze all traces in directory"
    )

    args = parser.parse_args()

    path = Path(args.path)

    if path.is_file():
        view_trace_with_tools(str(path), verbose=args.verbose)
    elif path.is_dir():
        if args.batch:
            analyze_batch_with_tools(str(path))
        else:
            # Show all traces
            trace_files = sorted(path.glob("trace_*.json"))
            for trace_file in trace_files:
                view_trace_with_tools(str(trace_file), verbose=args.verbose)
                print("\n")
    else:
        print(f"Error: {path} does not exist")
        sys.exit(1)


if __name__ == "__main__":
    main()
