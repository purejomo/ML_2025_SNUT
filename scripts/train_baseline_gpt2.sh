#!/bin/bash
# ==============================================================================
# Baseline GPT-2 Training Script for PCA Embedding Compression Pipeline
# ==============================================================================
#
# This script trains a baseline GPT-2 model on WikiText-103 dataset.
# After training, use pca_factorize_wte.py to extract and factorize embeddings.
#
# Model Config (based on GPT-2 style):
#   vocab_size: 50257 (tiktoken gpt2 - from wikitext103 dataset)
#   n_embd: 768
#   n_head: 12
#   n_layer: 12
#   block_size (n_ctx): 2048
#   dropout: 0.1 (attn_pdrop, embd_pdrop, resid_pdrop)
#   activation: gelu
#
# Usage:
#   bash scripts/train_baseline_gpt2.sh
#
# ==============================================================================

# Exit on error
set -e

# Project directory
PROJECT_DIR="/home/ghlee/ML_2025_SNUT"
cd "$PROJECT_DIR"

# Data directory (using nanoGPT's prepared wikitext103)
# vocab_size: 50257 (tiktoken gpt2)
DATA_DIR="/home/ghlee/nanoGPT/data/wikitext103"

# Output directory for baseline model
OUT_DIR="model_weights/gpt_baseline"

# Create output directory
mkdir -p "$OUT_DIR"

echo "=============================================="
echo "Training Baseline GPT-2 Model"
echo "=============================================="
echo "Config:"
echo "  vocab_size: 50257 (from dataset)"
echo "  n_embd: 768"
echo "  n_head: 12"
echo "  n_layer: 12"
echo "  block_size: 2048"
echo "  dropout: 0.1"
echo "Output directory: $OUT_DIR"
echo "Dataset: wikitext103"
echo "=============================================="

# Training command
# Model config follows the provided GPT-2 style config
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
    --wandb_run_name "gpt-baseline"

echo ""
echo "=============================================="
echo "Training Complete!"
echo "=============================================="
echo "Checkpoint saved to: $OUT_DIR/ckpt.pt"
echo ""
echo "Next step - Extract and factorize embeddings:"
echo "  python util_factorization/pca_factorize_wte.py \\"
echo "      --ckpt_path $OUT_DIR/ckpt.pt \\"
echo "      --rank_k 128 \\"
echo "      --out_wte_npy util_factorization/wte_pca_k128.npy \\"
echo "      --out_scale_npz util_factorization/scale_mats_pca_k128.npz"
echo "=============================================="

