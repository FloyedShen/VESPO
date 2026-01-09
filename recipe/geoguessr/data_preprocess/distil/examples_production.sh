#!/bin/bash
# Production distillation examples with checkpoint/resume support

# ============================================================================
# Basic Usage Examples
# ============================================================================

# Example 1: Generate 1000 samples with 4 workers (most common)
python3 distill_production.py \
    --dataset_path /mnt/tidal-alsh-hilab/usr/shenguobin/geoguessr/processed/gaea_wlp/train \
    --output_dir traces_production_1k \
    --num_samples 1000 \
    --max_workers 4

# Example 2: Generate with 8 workers for faster processing
python3 distill_production.py \
    --dataset_path /mnt/tidal-alsh-hilab/usr/shenguobin/geoguessr/processed/gaea_wlp/train \
    --output_dir traces_production_1k_fast \
    --num_samples 1000 \
    --max_workers 8

# Example 3: Start fresh (no resume)
python3 distill_production.py \
    --dataset_path /mnt/tidal-alsh-hilab/usr/shenguobin/geoguessr/processed/gaea_wlp/train \
    --output_dir traces_fresh \
    --num_samples 1000 \
    --max_workers 4 \
    --no_resume

# ============================================================================
# Sampling Strategy Examples
# ============================================================================

# Example 4: Sample hardest first (highest locatability_score)
python3 distill_production.py \
    --dataset_path /mnt/tidal-alsh-hilab/usr/shenguobin/geoguessr/processed/gaea_wlp/train \
    --output_dir traces_hardest \
    --num_samples 1000 \
    --max_workers 4 \
    --sampling_strategy hardest

# Example 5: Sample easiest first (lowest locatability_score)
python3 distill_production.py \
    --dataset_path /mnt/tidal-alsh-hilab/usr/shenguobin/geoguessr/processed/gaea_wlp/train \
    --output_dir traces_easiest \
    --num_samples 1000 \
    --max_workers 4 \
    --sampling_strategy easiest

# Example 6: Random sampling
python3 distill_production.py \
    --dataset_path /mnt/tidal-alsh-hilab/usr/shenguobin/geoguessr/processed/gaea_wlp/train \
    --output_dir traces_random \
    --num_samples 1000 \
    --max_workers 4 \
    --sampling_strategy random

# ============================================================================
# Resume Examples
# ============================================================================

# Example 7: Resume after interruption
# If you stopped at 500/1000, just run the same command again:
python3 distill_production.py \
    --dataset_path /mnt/tidal-alsh-hilab/usr/shenguobin/geoguessr/processed/gaea_wlp/train \
    --output_dir traces_production_1k \
    --num_samples 1000 \
    --max_workers 4
# It will automatically resume from sample 501

# Example 8: Check checkpoint status
cat traces_production_1k/checkpoint.json
# Shows: processed_indices, failed_indices, total_processed, etc.

# ============================================================================
# Large Scale Examples
# ============================================================================

# Example 9: Generate 10,000 samples (will take ~35 hours with 8 workers)
python3 distill_production.py \
    --dataset_path /mnt/tidal-alsh-hilab/usr/shenguobin/geoguessr/processed/gaea_wlp/train \
    --output_dir traces_10k \
    --num_samples 10000 \
    --max_workers 8

# Can be interrupted and resumed anytime!

# Example 10: Different datasets
# GAEA bench
python3 distill_production.py \
    --dataset_path /mnt/tidal-alsh-hilab/usr/shenguobin/geoguessr/processed/gaea_wlp/bench \
    --output_dir traces_gaea_bench \
    --num_samples 1000 \
    --max_workers 4

# ============================================================================
# Custom Parameters Examples
# ============================================================================

# Example 11: Higher temperature for more diversity
python3 distill_production.py \
    --dataset_path /mnt/tidal-alsh-hilab/usr/shenguobin/geoguessr/processed/gaea_wlp/train \
    --output_dir traces_diverse \
    --num_samples 1000 \
    --max_workers 4 \
    --temperature 0.9

# Example 12: More turns for complex reasoning
python3 distill_production.py \
    --dataset_path /mnt/tidal-alsh-hilab/usr/shenguobin/geoguessr/processed/gaea_wlp/train \
    --output_dir traces_complex \
    --num_samples 1000 \
    --max_workers 4 \
    --max_turns 15

# ============================================================================
# Monitoring Examples
# ============================================================================

# Monitor progress in real-time
watch -n 5 'python3 -c "import json; data=json.load(open(\"traces_production_1k/checkpoint.json\")); print(f\"Processed: {len(data[\"processed_indices\"])}, Failed: {len(data[\"failed_indices\"])}\")"'

# Count generated traces
ls traces_production_1k/trace_*.json | wc -l

# Check last 10 traces
ls -lt traces_production_1k/trace_*.json | head -10

# View statistics
python3 view_traces_enhanced.py traces_production_1k/ --batch
