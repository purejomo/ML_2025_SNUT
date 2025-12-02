#!/bin/bash
# ==============================================================================
# PCA Factorization Script
# ==============================================================================
#
# Extract token embeddings from trained baseline model and perform
# SVD-based PCA factorization to create compressed embeddings.
#
# Generates factorized embeddings for ranks: 64, 128, 256, 512
#
# Usage:
#   bash scripts/run_pca_factorization.sh
#
# ==============================================================================

set -e

PROJECT_DIR="/home/ghlee/ML_2025_SNUT"
cd "$PROJECT_DIR"

CKPT_PATH="model_weights/gpt_baseline/ckpt.pt"
OUTPUT_DIR="model_weights/pca_factorized"

mkdir -p "$OUTPUT_DIR"

# Rank values for experiments
RANK_VALUES=(64 128 256 512)

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
    echo "Please train baseline first: bash scripts/train_baseline_gpt2.sh"
    exit 1
fi

# Run PCA factorization for each rank
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
echo "Output files:"
for RANK_K in "${RANK_VALUES[@]}"; do
    echo "  - ${OUTPUT_DIR}/wte_pca_k${RANK_K}.npy"
    echo "  - ${OUTPUT_DIR}/scale_mats_pca_k${RANK_K}.npz"
done
echo "=============================================="

