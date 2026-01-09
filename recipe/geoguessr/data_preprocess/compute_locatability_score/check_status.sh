#!/bin/bash
#
# Quick status checker for datasets
#

# Configuration
SOURCE_DIR="/mnt/tidal-alsh-hilab/dataset/diandian/user/geogussr/processed"
OUTPUT_DIR="/mnt/tidal-alsh-hilab/usr/shenguobin/geoguessr/processed"

# Color output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "======================================================================"
echo "  Dataset Status Checker"
echo "======================================================================"
echo ""

check_dataset() {
    local DATASET=$1
    local SPLIT=$2

    local INPUT_PATH="${SOURCE_DIR}/${DATASET}/${SPLIT}"
    local OUTPUT_PATH="${OUTPUT_DIR}/${DATASET}_wlp/${SPLIT}"

    echo "Dataset: ${DATASET}/${SPLIT}"

    # Check input
    if [ -d "$INPUT_PATH" ]; then
        # Get sample count if possible
        if python3 -c "import datasets; ds = datasets.load_from_disk('$INPUT_PATH'); print(f'  Input:  ✓ Exists ({len(ds):,} samples)')" 2>/dev/null; then
            :
        else
            echo -e "  Input:  ${GREEN}✓ Exists${NC}"
        fi
    else
        echo -e "  Input:  ${RED}✗ Not found${NC}"
        echo "          Path: $INPUT_PATH"
    fi

    # Check output
    if [ -d "$OUTPUT_PATH" ]; then
        if python3 -c "import datasets; ds = datasets.load_from_disk('$OUTPUT_PATH'); has_score = 'locatability_score' in ds.column_names; print(f'  Output: ✓ Exists ({len(ds):,} samples) - Has scores: {has_score}')" 2>/dev/null; then
            :
        else
            echo -e "  Output: ${GREEN}✓ Exists${NC}"
        fi
        echo "          Path: $OUTPUT_PATH"
    else
        echo -e "  Output: ${YELLOW}⊙ Not found (needs processing)${NC}"
        echo "          Path: $OUTPUT_PATH"
    fi

    echo ""
}

echo "Checking GAEA datasets..."
echo "----------------------------------------------------------------------"
check_dataset "gaea" "train"
check_dataset "gaea" "bench"

echo "Checking OSV5M datasets..."
echo "----------------------------------------------------------------------"
check_dataset "osv5m" "train"
check_dataset "osv5m" "test"

echo "======================================================================"
echo "  Summary"
echo "======================================================================"
echo ""
echo "To process missing datasets:"
echo "  cd /mnt/tidal-alsh-hilab/usr/shenguobin/verl/recipe/geoguessr/data_preprocess/compute_locatability_score"
echo ""
echo "  ./run_batch.sh --gaea     # Process GAEA train + bench"
echo "  ./run_batch.sh --osv5m    # Process OSV5M train + test"
echo "  ./run_batch.sh --all      # Process all datasets"
echo ""
echo "To reprocess existing datasets:"
echo "  ./run_batch.sh --gaea --force"
echo ""
