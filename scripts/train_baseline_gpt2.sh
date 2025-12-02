#!/bin/bash
# ==============================================================================
# Experiment 1: Baseline Model Training
# ==============================================================================
#
# Train a standard GPT-2 model with full 768d embeddings.
# This serves as the upper bound reference for all other experiments.
#
# Usage:
#   bash scripts/train_baseline_gpt2.sh
#
# ==============================================================================

set -e

PROJECT_DIR="/home/ghlee/ML_2025_SNUT"
cd "$PROJECT_DIR"

DATA_DIR="/home/ghlee/nanoGPT/data/wikitext103"
OUT_DIR="model_weights/gpt_baseline"

mkdir -p "$OUT_DIR"

echo "=============================================="
echo "Experiment 1: Baseline Model Training"
echo "=============================================="
echo "Config:"
echo "  vocab_size: 50257 (from dataset)"
echo "  n_embd: 768 (full)"
echo "  n_head: 12"
echo "  n_layer: 12"
echo "  block_size: 2048"
echo "Output: $OUT_DIR"
echo "=============================================="

python train.py \
    --out_dir "$OUT_DIR" \
    --dataset "$DATA_DIR" \
    \
    --n_layer 12 \
    --n_head 12 \
    --n_embd 768 \
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
    --learning_rate 1e-4 \
    --min_lr 5e-6 \
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
    --wandb_run_name "baseline"

echo ""
echo "=============================================="
echo "Baseline Training Complete!"
echo "=============================================="
echo "Checkpoint: $OUT_DIR/ckpt.pt"
echo "=============================================="
