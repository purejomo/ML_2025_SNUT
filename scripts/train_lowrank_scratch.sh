#!/bin/bash
# ==============================================================================
# Experiments 2-5: Low-rank Scratch Training
# ==============================================================================
#
# Train a GPT model with randomly initialized low-rank factorized embeddings.
# This measures the cost of dimensionality reduction without PCA initialization.
#
# Usage:
#   bash scripts/train_lowrank_scratch.sh <rank_k>
#
# Examples:
#   bash scripts/train_lowrank_scratch.sh 64
#   bash scripts/train_lowrank_scratch.sh 128
#   bash scripts/train_lowrank_scratch.sh 256
#   bash scripts/train_lowrank_scratch.sh 512
#
# ==============================================================================

set -e

if [ -z "$1" ]; then
    echo "Usage: bash scripts/train_lowrank_scratch.sh <rank_k>"
    echo "Available ranks: 64, 128, 256, 512"
    exit 1
fi

RANK_K=$1

PROJECT_DIR="/home/ghlee/ML_2025_SNUT"
cd "$PROJECT_DIR"

DATA_DIR="/home/ghlee/nanoGPT/data/wikitext103"
OUT_DIR="model_weights/gpt_lowrank_scratch_k${RANK_K}"

mkdir -p "$OUT_DIR"

echo "=============================================="
echo "Experiment: Low-rank Scratch (k=$RANK_K)"
echo "=============================================="
echo "Config:"
echo "  vocab_size: 50257"
echo "  n_embd: 768"
echo "  n_embd_wte: $RANK_K (random init)"
echo "  n_head: 12"
echo "  n_layer: 12"
echo ""
echo "Training: Full training from scratch"
echo "Output: $OUT_DIR"
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
    --wandb_run_name "lowrank-scratch-k${RANK_K}"

echo ""
echo "=============================================="
echo "Low-rank Scratch (k=$RANK_K) Complete!"
echo "=============================================="
echo "Checkpoint: $OUT_DIR/ckpt.pt"
echo "=============================================="

