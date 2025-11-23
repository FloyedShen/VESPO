#!/bin/bash
#
# Batch script to compute locatability scores for all datasets.
#
# This script processes GAEA and OSV5M datasets using multi-GPU parallelization.
#

# Note: Not using 'set -e' to allow processing to continue even if some datasets fail

# Configuration
#SOURCE_DIR="/mnt/tidal-alsh-hilab/dataset/diandian/user/geogussr/processed"
#OUTPUT_DIR="/mnt/tidal-alsh-hilab/usr/shenguobin/geoguessr/processed"

SOURCE_DIR="/diancpfs/user/guobin/geogussr/processed"
OUTPUT_DIR="/diancpfs/user/guobin/geogussr/processed"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Model configuration
MODEL_ID="facebook/mask2former-swin-large-ade-semantic"
BATCH_SIZE=256
NUM_WORKERS=16

# Processing options
USE_AMP=true
CHECKPOINT_INTERVAL=100
RESUME=true

# Color output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo "======================================================================"
echo "  Locatability Score Computation Pipeline"
echo "======================================================================"
echo ""
echo "Source directory: $SOURCE_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Model: $MODEL_ID"
echo "Batch size: $BATCH_SIZE"
echo ""

# Function to process a dataset
process_dataset() {
    local DATASET_NAME=$1
    local SPLIT=$2

    local INPUT_PATH="${SOURCE_DIR}/${DATASET_NAME}/${SPLIT}"
    local OUTPUT_PATH="${OUTPUT_DIR}/${DATASET_NAME}_wlp/${SPLIT}"

    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} Processing ${DATASET_NAME}/${SPLIT}..."
    echo "  Input:  $INPUT_PATH"
    echo "  Output: $OUTPUT_PATH"

    # Check if input exists
    if [ ! -d "$INPUT_PATH" ]; then
        echo -e "${RED}✗ Input dataset not found: $INPUT_PATH${NC}"
        return 1
    fi

    # Check if output already exists
    if [ -d "$OUTPUT_PATH" ] && [ ! "$FORCE_REPROCESS" = "true" ]; then
        echo -e "${YELLOW}⚠ Output already exists. Skipping...${NC}"
        echo "  Use --force to reprocess"
        return 0
    fi

    # Build command
    local CMD="python3 ${SCRIPT_DIR}/scripts/run_parallel.py \
        --input_dataset \"$INPUT_PATH\" \
        --output_dataset \"$OUTPUT_PATH\" \
        --batch_size $BATCH_SIZE \
        --num_workers $NUM_WORKERS \
        --model_id \"$MODEL_ID\" \
        --checkpoint_interval $CHECKPOINT_INTERVAL"

    if [ "$USE_AMP" != "true" ]; then
        CMD="$CMD --no_amp"
    fi

    if [ "$RESUME" != "true" ]; then
        CMD="$CMD --no_resume"
    fi

    # Execute
    echo ""
    echo "Command: $CMD"
    echo ""

    if eval $CMD; then
        echo -e "${GREEN}✓ Successfully processed ${DATASET_NAME}/${SPLIT}${NC}"
        return 0
    else
        echo -e "${RED}✗ Failed to process ${DATASET_NAME}/${SPLIT}${NC}"
        return 1
    fi
}

# Parse arguments
FORCE_REPROCESS=false
DATASETS_TO_PROCESS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --force)
            FORCE_REPROCESS=true
            shift
            ;;
        --gaea)
            DATASETS_TO_PROCESS+=("gaea/train" "gaea/bench")
            shift
            ;;
        --osv5m)
            DATASETS_TO_PROCESS+=("osv5m/train" "osv5m/test")
            shift
            ;;
        --all)
            DATASETS_TO_PROCESS+=("gaea/train" "gaea/bench" "osv5m/train" "osv5m/test")
            shift
            ;;
        --batch_size)
            BATCH_SIZE=$2
            shift 2
            ;;
        --no_amp)
            USE_AMP=false
            shift
            ;;
        --no_resume)
            RESUME=false
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --gaea              Process GAEA dataset (train + bench)"
            echo "  --osv5m             Process OSV5M dataset (train + test)"
            echo "  --all               Process all datasets"
            echo "  --force             Reprocess even if output exists"
            echo "  --batch_size N      Set batch size (default: 20)"
            echo "  --no_amp            Disable automatic mixed precision"
            echo "  --no_resume         Don't resume from checkpoint"
            echo "  --help              Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --gaea                    # Process only GAEA"
            echo "  $0 --osv5m --batch_size 32   # Process OSV5M with larger batch"
            echo "  $0 --all --force             # Reprocess everything"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Default to all if none specified
if [ ${#DATASETS_TO_PROCESS[@]} -eq 0 ]; then
    echo -e "${YELLOW}No datasets specified. Use --gaea, --osv5m, or --all${NC}"
    echo "Use --help for more information"
    exit 1
fi

# Process datasets
echo "======================================================================"
echo "  Starting processing..."
echo "======================================================================"
echo ""

TOTAL=${#DATASETS_TO_PROCESS[@]}
SUCCESS=0
FAILED=0
SKIPPED=0

# First, check all datasets and show status
echo "Checking datasets..."
echo ""
for DATASET_SPLIT in "${DATASETS_TO_PROCESS[@]}"; do
    IFS='/' read -r DATASET SPLIT <<< "$DATASET_SPLIT"
    INPUT_PATH="${SOURCE_DIR}/${DATASET}/${SPLIT}"
    OUTPUT_PATH="${OUTPUT_DIR}/${DATASET}_wlp/${SPLIT}"

    if [ ! -d "$INPUT_PATH" ]; then
        echo -e "${RED}✗${NC} ${DATASET}/${SPLIT} - Input not found: $INPUT_PATH"
    elif [ -d "$OUTPUT_PATH" ] && [ "$FORCE_REPROCESS" != "true" ]; then
        echo -e "${YELLOW}⊙${NC} ${DATASET}/${SPLIT} - Output exists, will skip (use --force to reprocess)"
    else
        echo -e "${GREEN}→${NC} ${DATASET}/${SPLIT} - Will process"
    fi
done

echo ""
read -p "Continue? [Y/n] " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]] && [[ ! -z $REPLY ]]; then
    echo "Aborted by user."
    exit 0
fi

echo ""
echo "======================================================================"
echo "  Processing..."
echo "======================================================================"
echo ""

for DATASET_SPLIT in "${DATASETS_TO_PROCESS[@]}"; do
    IFS='/' read -r DATASET SPLIT <<< "$DATASET_SPLIT"

    echo ""
    echo "----------------------------------------------------------------------"
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} Checking ${DATASET}/${SPLIT}..."

    OUTPUT_PATH="${OUTPUT_DIR}/${DATASET}_wlp/${SPLIT}"
    if [ -d "$OUTPUT_PATH" ] && [ "$FORCE_REPROCESS" != "true" ]; then
        echo -e "${YELLOW}⊙ Skipping ${DATASET}/${SPLIT} (output already exists)${NC}"
        ((SKIPPED++))
        echo "----------------------------------------------------------------------"
        echo ""
        continue
    fi

    echo -e "${GREEN}→ Processing ${DATASET}/${SPLIT}...${NC}"

    if process_dataset "$DATASET" "$SPLIT"; then
        echo -e "${GREEN}✓ Successfully completed ${DATASET}/${SPLIT}${NC}"
        ((SUCCESS++))
    else
        echo -e "${RED}✗ Failed to process ${DATASET}/${SPLIT}${NC}"
        ((FAILED++))
    fi

    echo "----------------------------------------------------------------------"
    echo ""
done

# Summary
echo ""
echo "======================================================================"
echo "  Processing Summary"
echo "======================================================================"
echo "  Total datasets: $TOTAL"
echo -e "  ${GREEN}Processed: $SUCCESS${NC}"
if [ $SKIPPED -gt 0 ]; then
    echo -e "  ${YELLOW}Skipped: $SKIPPED${NC} (already exist)"
fi
if [ $FAILED -gt 0 ]; then
    echo -e "  ${RED}Failed: $FAILED${NC}"
else
    echo "  Failed: 0"
fi
echo "======================================================================"
echo ""

if [ $FAILED -gt 0 ]; then
    echo -e "${RED}Some datasets failed to process. Check logs for details.${NC}"
    exit 1
else
    if [ $SUCCESS -gt 0 ]; then
        echo -e "${GREEN}✓ All requested datasets processed successfully!${NC}"
    fi
    if [ $SKIPPED -gt 0 ]; then
        echo -e "${YELLOW}Note: $SKIPPED dataset(s) were skipped because output already exists.${NC}"
        echo -e "${YELLOW}Use --force to reprocess them.${NC}"
    fi
    exit 0
fi
