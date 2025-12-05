#!/bin/bash
# ==============================================================================
# Evaluate PCA (No Finetune) for all k values sequentially
# ==============================================================================
#
# This script runs PCA (no fine-tuning) evaluation for all 4 PCA ranks
# (k=64, 128, 256, 512) sequentially.
#
# Usage:
#   bash scripts/eval_pca_no_finetune_all.sh
#
# ==============================================================================

set -e

PROJECT_DIR="/home/ghlee/ML_2025_SNUT"
cd "$PROJECT_DIR"

# Array of PCA ranks to test
PCA_RANKS=(64 128 256 512)

echo "=============================================="
echo "PCA (No Finetune) Evaluation (All Ranks)"
echo "=============================================="
echo "Mode: Evaluation only (no training)"
echo "PCA Ranks to evaluate: ${PCA_RANKS[@]}"
echo "=============================================="
echo ""

# Track start time
START_TIME=$(date +%s)
TOTAL_EXPERIMENTS=${#PCA_RANKS[@]}
CURRENT_EXPERIMENT=0

# Run evaluation for each PCA rank
for RANK_K in "${PCA_RANKS[@]}"; do
    CURRENT_EXPERIMENT=$((CURRENT_EXPERIMENT + 1))
    
    echo ""
    echo "=============================================="
    echo "[$CURRENT_EXPERIMENT/$TOTAL_EXPERIMENTS] Evaluating PCA (No Finetune) k=$RANK_K"
    echo "=============================================="
    echo "Time: $(date)"
    echo ""
    
    # Run the evaluation script
    bash scripts/eval_pca_no_finetune.sh "$RANK_K"
    
    # Check if evaluation was successful
    if [ $? -eq 0 ]; then
        echo ""
        echo "✓ Successfully completed PCA (No Finetune) evaluation for k=$RANK_K"
        echo ""
    else
        echo ""
        echo "✗ ERROR: Evaluation failed for k=$RANK_K"
        echo "Stopping batch evaluation."
        exit 1
    fi
    
    # Calculate elapsed time
    ELAPSED=$(($(date +%s) - START_TIME))
    HOURS=$((ELAPSED / 3600))
    MINUTES=$(((ELAPSED % 3600) / 60))
    SECONDS=$((ELAPSED % 60))
    
    echo "Elapsed time so far: ${HOURS}h ${MINUTES}m ${SECONDS}s"
    echo ""
done

# Calculate total time
TOTAL_ELAPSED=$(($(date +%s) - START_TIME))
TOTAL_HOURS=$((TOTAL_ELAPSED / 3600))
TOTAL_MINUTES=$(((TOTAL_ELAPSED % 3600) / 60))
TOTAL_SECONDS=$((TOTAL_ELAPSED % 60))

echo ""
echo "=============================================="
echo "All PCA (No Finetune) Evaluations Complete!"
echo "=============================================="
echo "Total experiments: $TOTAL_EXPERIMENTS"
echo "Total time: ${TOTAL_HOURS}h ${TOTAL_MINUTES}m ${TOTAL_SECONDS}s"
echo ""
echo "Results saved in:"
for RANK_K in "${PCA_RANKS[@]}"; do
    OUT_DIR="model_weights/gpt_pca_nofinetune_k${RANK_K}"
    echo "  - k=$RANK_K: $OUT_DIR"
done
echo "=============================================="


