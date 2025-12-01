#!/bin/bash
# ==============================================================================
# PCA-Compressed GPT-2 Training Script
# ==============================================================================
#
# This script trains a GPT-2 model initialized with PCA-factorized embeddings.
# The model uses a smaller embedding dimension (n_embd_wte) and projects to
# the full dimension (n_embd) via scale_up/scale_down matrices.
#
# Usage:
#   bash scripts/train_pca_compressed.sh <rank_k>
#
# Example:
#   bash scripts/train_pca_compressed.sh 128
#
# ==============================================================================

set -e

# Check argument
if [ -z "$1" ]; then
    echo "Usage: bash scripts/train_pca_compressed.sh <rank_k>"
    echo "Example: bash scripts/train_pca_compressed.sh 128"
    exit 1
fi

RANK_K=$1

PROJECT_DIR="/home/ghlee/ML_2025_SNUT"
cd "$PROJECT_DIR"

# Data directory
DATA_DIR="/home/ghlee/nanoGPT/data/wikitext103"

# PCA factorized matrices
WTE_NPY="model_weights/pca_factorized/wte_pca_k${RANK_K}.npy"
SCALE_NPZ="model_weights/pca_factorized/scale_mats_pca_k${RANK_K}.npz"

# Output directory
OUT_DIR="model_weights/gpt_pca_k${RANK_K}"

# Check if factorized files exist
if [ ! -f "$WTE_NPY" ]; then
    echo "Error: Factorized WTE not found at $WTE_NPY"
    echo "Please run: bash scripts/run_pca_factorization.sh"
    exit 1
fi

if [ ! -f "$SCALE_NPZ" ]; then
    echo "Error: Scale matrices not found at $SCALE_NPZ"
    echo "Please run: bash scripts/run_pca_factorization.sh"
    exit 1
fi

mkdir -p "$OUT_DIR"

echo "=============================================="
echo "Training PCA-Compressed GPT-2 Model"
echo "=============================================="
echo "Config:"
echo "  vocab_size: 50257 (from dataset)"
echo "  n_embd: 768"
echo "  n_embd_wte: $RANK_K (PCA rank)"
echo "  n_head: 12"
echo "  n_layer: 12"
echo "  block_size: 2048"
echo "  dropout: 0.1"
echo ""
echo "PCA Initialization:"
echo "  WTE: $WTE_NPY"
echo "  Scale matrices: $SCALE_NPZ"
echo ""
echo "Output directory: $OUT_DIR"
echo "=============================================="

# Training command
python train.py \
    --out_dir "$OUT_DIR" \
    --dataset "$DATA_DIR" \
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
    --dropout 0.1 \
    --activation_variant gelu \
    --norm_variant_attn layernorm \
    --norm_variant_output layernorm \
    --bias \
    \
    --use_abs_pos_embeddings \
    --no-use_rotary_embeddings \
    \
    --batch_size 4 \
    --gradient_accumulation_steps 16 \
    \
    --max_iters 10000 \
    --eval_interval 500 \
    --eval_iters 100 \
    \
    --learning_rate 3e-4 \
    --min_lr 3e-5 \
    --decay_lr \
    --warmup_iters 500 \
    --lr_decay_iters 10000 \
    \
    --optimizer adamw \
    --weight_decay 0.1 \
    --beta1 0.9 \
    --beta2 0.95 \
    --grad_clip 1.0 \
    \
    --device cuda \
    --dtype float16 \
    --compile \
    \
    --init_from scratch \
    \
    --wandb_log \
    --wandb_project "new-small-gpt" \
    --wandb_run_name "gpt-pca-k${RANK_K}"

echo ""
echo "=============================================="
echo "Training Complete!"
echo "=============================================="
echo "Checkpoint saved to: $OUT_DIR/ckpt.pt"
echo ""
echo "Model comparison:"
echo "  Baseline: model_weights/gpt_baseline/ckpt.pt"
echo "  PCA k=$RANK_K: $OUT_DIR/ckpt.pt"
echo "=============================================="

