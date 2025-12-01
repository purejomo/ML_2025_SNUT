#!/bin/bash
# ==============================================================================
# PCA Factorization Script
# ==============================================================================
#
# This script extracts token embeddings from a trained baseline model and
# performs SVD-based PCA factorization to create compressed embeddings.
#
# Usage:
#   bash scripts/run_pca_factorization.sh
#
# ==============================================================================

set -e

PROJECT_DIR="/home/ghlee/ML_2025_SNUT"
cd "$PROJECT_DIR"

# Baseline checkpoint path
CKPT_PATH="model_weights/gpt_baseline/ckpt.pt"

# Output directory for factorized matrices
OUTPUT_DIR="model_weights/pca_factorized"
mkdir -p "$OUTPUT_DIR"

# Different rank values to try
# You can modify these values based on your needs
RANK_VALUES=(64 128 256 384)

echo "=============================================="
echo "PCA Factorization of Token Embeddings"
echo "=============================================="
echo "Baseline checkpoint: $CKPT_PATH"
echo "Output directory: $OUTPUT_DIR"
echo "Rank values: ${RANK_VALUES[*]}"
echo "=============================================="

# Check if checkpoint exists
if [ ! -f "$CKPT_PATH" ]; then
    echo "Error: Checkpoint not found at $CKPT_PATH"
    echo "Please ensure baseline training is complete."
    exit 1
fi

# Run PCA factorization for each rank value
for RANK_K in "${RANK_VALUES[@]}"; do
    echo ""
    echo "----------------------------------------------"
    echo "Factorizing with rank k = $RANK_K"
    echo "----------------------------------------------"
    
    python util_factorization/pca_factorize_wte.py \
        --ckpt_path "$CKPT_PATH" \
        --rank_k "$RANK_K" \
        --out_wte_npy "${OUTPUT_DIR}/wte_pca_k${RANK_K}.npy" \
        --out_scale_npz "${OUTPUT_DIR}/scale_mats_pca_k${RANK_K}.npz" \
        --show_singular_values
    
    echo "Completed factorization for rank k = $RANK_K"
done

echo ""
echo "=============================================="
echo "PCA Factorization Complete!"
echo "=============================================="
echo "Output files saved to: $OUTPUT_DIR"
echo ""
echo "Next step - Train PCA-compressed model:"
echo "  bash scripts/train_pca_compressed.sh <rank_k>"
echo ""
echo "Example:"
echo "  bash scripts/train_pca_compressed.sh 128"
echo "=============================================="

