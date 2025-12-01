#!/bin/bash
# ==============================================================================
# C. Random-Scratch: Random Low-rank Embedding으로 처음부터 학습
# ==============================================================================
#
# 이 스크립트는:
# 1. Low-rank embedding (n_embd_wte)을 RANDOM으로 초기화
# 2. 모델 전체를 처음부터 학습
#
# PCA 초기화(A)와 비교하기 위한 대조군입니다.
# 같은 차원(128d)에서 PCA vs Random 초기화의 효과를 측정합니다.
#
# Usage:
#   bash scripts/train_random_lowrank.sh <rank_k>
#
# Example:
#   bash scripts/train_random_lowrank.sh 128
#
# ==============================================================================

set -e

if [ -z "$1" ]; then
    echo "Usage: bash scripts/train_random_lowrank.sh <rank_k>"
    exit 1
fi

RANK_K=$1

PROJECT_DIR="/home/ghlee/ML_2025_SNUT"
cd "$PROJECT_DIR"

DATA_DIR="/home/ghlee/nanoGPT/data/wikitext103"

# Output directory
OUT_DIR="model_weights/gpt_random_lowrank_k${RANK_K}"

mkdir -p "$OUT_DIR"

echo "=============================================="
echo "C. Random-Scratch: Random Low-rank Embedding"
echo "=============================================="
echo "Config:"
echo "  vocab_size: 50257 (from dataset)"
echo "  n_embd: 768"
echo "  n_embd_wte: $RANK_K (RANDOM init, NOT PCA)"
echo "  n_head: 12"
echo "  n_layer: 12"
echo "  block_size: 2048"
echo ""
echo "Purpose:"
echo "  Control group to measure PCA initialization effect"
echo "  Compare with A (PCA-Scratch) at same dimension"
echo ""
echo "Output: $OUT_DIR"
echo "=============================================="

# NOTE: --import_wte_npy와 --import_scale_matrices_npz를 사용하지 않음
# → embedding이 random으로 초기화됨

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
    --gradient_accumulation_steps 4 \
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
    --wandb_run_name "gpt-random-lowrank-k${RANK_K}"

echo ""
echo "=============================================="
echo "C. Random-Scratch Complete!"
echo "=============================================="
echo "Checkpoint: $OUT_DIR/ckpt.pt"
echo ""
echo "Compare with A (PCA-Scratch):"
echo "  A > C → PCA initialization is effective!"
echo "  A ≈ C → Dimension reduction matters, not init"
echo "=============================================="

