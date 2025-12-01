#!/bin/bash
# ==============================================================================
# PCA-Compressed GPT-2 Training Script (Frozen Embeddings)
# ==============================================================================
#
# This script trains a GPT-2 model with FROZEN PCA-factorized embeddings.
# Only the transformer layers are trained; embeddings and scale matrices
# are kept fixed from the PCA initialization.
#
# This is useful for:
# - Faster training (fewer parameters to update)
# - Testing how well PCA captures the essential embedding structure
# - Comparison with full fine-tuning
#
# Usage:
#   bash scripts/train_pca_compressed_frozen.sh <rank_k>
#
# ==============================================================================

set -e

if [ -z "$1" ]; then
    echo "Usage: bash scripts/train_pca_compressed_frozen.sh <rank_k>"
    exit 1
fi

RANK_K=$1

PROJECT_DIR="/home/ghlee/ML_2025_SNUT"
cd "$PROJECT_DIR"

DATA_DIR="/home/ghlee/nanoGPT/data/wikitext103"
WTE_NPY="model_weights/pca_factorized/wte_pca_k${RANK_K}.npy"
SCALE_NPZ="model_weights/pca_factorized/scale_mats_pca_k${RANK_K}.npz"
OUT_DIR="model_weights/gpt_pca_k${RANK_K}_frozen"

if [ ! -f "$WTE_NPY" ] || [ ! -f "$SCALE_NPZ" ]; then
    echo "Error: Factorized files not found. Run: bash scripts/run_pca_factorization.sh"
    exit 1
fi

mkdir -p "$OUT_DIR"

echo "=============================================="
echo "Training PCA-Compressed GPT-2 (FROZEN Embeddings)"
echo "=============================================="
echo "n_embd_wte: $RANK_K (FROZEN)"
echo "=============================================="

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
    --import_wte_freeze \
    --import_scale_matrices_npz "$SCALE_NPZ" \
    --import_scale_matrices_freeze \
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
    --wandb_run_name "gpt-pca-k${RANK_K}-frozen"

echo ""
echo "=============================================="
echo "Training Complete! (Frozen Embeddings)"
echo "=============================================="

