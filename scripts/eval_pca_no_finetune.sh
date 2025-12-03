#!/bin/bash
# ==============================================================================
# Experiments 6-9: PCA (No Finetune) Evaluation
# ==============================================================================
#
# Evaluate the model with PCA-factorized embeddings WITHOUT any fine-tuning.
# This measures how much information PCA preserves from the baseline.
#
# The script:
# 1. Loads baseline model
# 2. Replaces embedding with PCA-factorized version
# 3. Evaluates validation loss (no training)
# 4. Logs results to W&B
#
# Usage:
#   bash scripts/eval_pca_no_finetune.sh <rank_k>
#
# Examples:
#   bash scripts/eval_pca_no_finetune.sh 64
#   bash scripts/eval_pca_no_finetune.sh 128
#   bash scripts/eval_pca_no_finetune.sh 256
#   bash scripts/eval_pca_no_finetune.sh 512
#
# ==============================================================================

set -e

if [ -z "$1" ]; then
    echo "Usage: bash scripts/eval_pca_no_finetune.sh <rank_k>"
    echo "Available ranks: 64, 128, 256, 512"
    exit 1
fi

RANK_K=$1

PROJECT_DIR="/home/ghlee/ML_2025_SNUT"
cd "$PROJECT_DIR"

DATA_DIR="/home/ghlee/nanoGPT/data/wikitext103"
BASELINE_CKPT="model_weights/gpt_baseline"
WTE_NPY="model_weights/pca_factorized/wte_pca_k${RANK_K}.npy"
SCALE_NPZ="model_weights/pca_factorized/scale_mats_pca_k${RANK_K}.npz"
OUT_DIR="model_weights/gpt_pca_nofinetune_k${RANK_K}"

# Check files exist
if [ ! -f "$BASELINE_CKPT/ckpt.pt" ]; then
    echo "Error: Baseline checkpoint not found at $BASELINE_CKPT/ckpt.pt"
    exit 1
fi

if [ ! -f "$WTE_NPY" ] || [ ! -f "$SCALE_NPZ" ]; then
    echo "Error: PCA factorized files not found."
    echo "Please run: bash scripts/run_pca_factorization.sh"
    exit 1
fi

mkdir -p "$OUT_DIR"

echo "=============================================="
echo "Experiment: PCA No Finetune (k=$RANK_K)"
echo "=============================================="
echo "Config:"
echo "  Baseline: $BASELINE_CKPT"
echo "  PCA embedding: $WTE_NPY"
echo "  n_embd_wte: $RANK_K"
echo ""
echo "Mode: Evaluation only (no training)"
echo "=============================================="

# Run evaluation only (max_iters=0, eval_only)
python train.py \
    --out_dir "$OUT_DIR" \
    --dataset "$DATA_DIR" \
    \
    --init_from prev_run \
    --prev_run_ckpt "$BASELINE_CKPT" \
    \
    --n_layer 12 \
    --n_head 12 \
    --n_embd 768 \
    --n_embd_wte "$RANK_K" \
    --block_size 2048 \
    \
    --import_wte_npy "$WTE_NPY" \
    --import_scale_matrices_npz "$SCALE_NPZ" \
    \
    --dropout 0.0 \
    --activation_variant gelu \
    --norm_variant_attn layernorm \
    --norm_variant_output layernorm \
    --bias \
    \
    --use_abs_pos_embeddings \
    --no-use_rotary_embeddings \
    \
    --batch_size 4 \
    \
    --eval_only \
    --eval_iters 100 \
    \
    --device cuda \
    --dtype float16

echo ""
echo "=============================================="
echo "PCA No Finetune (k=$RANK_K) Evaluation Complete!"
echo "=============================================="

