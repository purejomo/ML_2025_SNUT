#!/bin/bash
# ==============================================================================
# Experiments 10-13: PCA + LoRA Fine-tuning
# ==============================================================================
#
# Apply PCA factorization to baseline embeddings, then fine-tune using LoRA.
# LoRA is applied to: token embedding, attention (Q, K, V), FFN
#
# This measures whether LoRA fine-tuning can recover performance after
# PCA compression.
#
# Usage:
#   bash scripts/train_pca_lora.sh <rank_k>
#
# Examples:
#   bash scripts/train_pca_lora.sh 64
#   bash scripts/train_pca_lora.sh 128
#   bash scripts/train_pca_lora.sh 256
#   bash scripts/train_pca_lora.sh 512
#
# ==============================================================================

set -e

if [ -z "$1" ]; then
    echo "Usage: bash scripts/train_pca_lora.sh <rank_k>"
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
OUT_DIR="model_weights/gpt_pca_lora_k${RANK_K}"

# LoRA hyperparameters
LORA_RANK=16
LORA_ALPHA=32
LORA_DROPOUT=0.1

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
echo "Experiment: PCA + LoRA (k=$RANK_K)"
echo "=============================================="
echo "Config:"
echo "  Baseline: $BASELINE_CKPT"
echo "  PCA embedding: $WTE_NPY"
echo "  n_embd_wte: $RANK_K"
echo ""
echo "LoRA Config:"
echo "  LoRA rank: $LORA_RANK"
echo "  LoRA alpha: $LORA_ALPHA"
echo "  LoRA dropout: $LORA_DROPOUT"
echo "  Applied to: token embedding, attention, FFN"
echo ""
echo "Training: LoRA fine-tuning"
echo "Output: $OUT_DIR"
echo "=============================================="

# LoRA fine-tuning
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
    --use_lora \
    --lora_rank "$LORA_RANK" \
    --lora_alpha "$LORA_ALPHA" \
    --lora_dropout "$LORA_DROPOUT" \
    --lora_targets "wte,scale_up,scale_down,q_proj,k_proj,v_proj,c_proj,mlp_up,mlp_down" \
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
    --max_iters 2000 \
    --eval_interval 200 \
    --eval_iters 100 \
    \
    --learning_rate 1e-4 \
    --min_lr 1e-5 \
    --decay_lr \
    --warmup_iters 100 \
    --lr_decay_iters 2000 \
    \
    --optimizer adamw \
    --weight_decay 0.01 \
    --beta1 0.9 \
    --beta2 0.95 \
    --grad_clip 1.0 \
    \
    --device cuda \
    --dtype float16 \
    --compile \
    \
    --wandb_log \
    --wandb_project "new-small-gpt" \
    --wandb_run_name "pca-lora-k${RANK_K}"

echo ""
echo "=============================================="
echo "PCA + LoRA (k=$RANK_K) Complete!"
echo "=============================================="
echo "Checkpoint: $OUT_DIR/ckpt.pt"
echo "=============================================="

