#!/bin/bash
# ==============================================================================
# Train PCA + LoRA for all k values sequentially
# ==============================================================================
#
# This script runs PCA + LoRA fine-tuning for all 4 PCA ranks (k=64, 128, 256, 512)
# sequentially with LoRA rank=32.
#
# Usage:
#   bash scripts/train_pca_lora_all.sh
#
# ==============================================================================

set -e

PROJECT_DIR="/home/ghlee/ML_2025_SNUT"
cd "$PROJECT_DIR"

# Array of PCA ranks to test
PCA_RANKS=(64 128 256 512)

echo "=============================================="
echo "PCA + LoRA Training (All Ranks)"
echo "=============================================="
echo "LoRA Config:"
echo "  LoRA rank: 32"
echo "  LoRA alpha: 64"
echo "  LoRA dropout: 0.1"
echo ""
echo "PCA Ranks to train: ${PCA_RANKS[@]}"
echo "=============================================="
echo ""

# Track start time
START_TIME=$(date +%s)
TOTAL_EXPERIMENTS=${#PCA_RANKS[@]}
CURRENT_EXPERIMENT=0

# Run training for each PCA rank
for RANK_K in "${PCA_RANKS[@]}"; do
    CURRENT_EXPERIMENT=$((CURRENT_EXPERIMENT + 1))
    
    echo ""
    echo "=============================================="
    echo "[$CURRENT_EXPERIMENT/$TOTAL_EXPERIMENTS] Starting PCA + LoRA (k=$RANK_K)"
    echo "=============================================="
    echo "Time: $(date)"
    echo ""
    
    # Run the training script
    bash scripts/train_pca_lora.sh "$RANK_K"
    
    # Check if training was successful
    if [ $? -eq 0 ]; then
        echo ""
        echo "✓ Successfully completed PCA + LoRA (k=$RANK_K)"
        echo ""
    else
        echo ""
        echo "✗ ERROR: Training failed for k=$RANK_K"
        echo "Stopping batch training."
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
echo "All PCA + LoRA Training Complete!"
echo "=============================================="
echo "Total experiments: $TOTAL_EXPERIMENTS"
echo "Total time: ${TOTAL_HOURS}h ${TOTAL_MINUTES}m ${TOTAL_SECONDS}s"
echo ""
echo "Results:"
for RANK_K in "${PCA_RANKS[@]}"; do
    OUT_DIR="model_weights/gpt_pca_lora_k${RANK_K}"
    if [ -f "$OUT_DIR/ckpt.pt" ]; then
        echo "  ✓ k=$RANK_K: $OUT_DIR/ckpt.pt"
    else
        echo "  ✗ k=$RANK_K: Checkpoint not found"
    fi
done
echo "=============================================="

