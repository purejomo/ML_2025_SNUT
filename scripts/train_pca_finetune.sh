#!/bin/bash
# ==============================================================================
# B. PCA-Finetune: Baseline 가중치 유지 + Embedding만 PCA로 교체 + Fine-tune
# ==============================================================================
#
# 이 스크립트는:
# 1. Baseline checkpoint의 모든 가중치를 로드
# 2. Token embedding만 PCA-factorized 버전으로 교체
# 3. 짧은 fine-tuning 수행
#
# Baseline에서 학습된 attention/mlp 지식을 보존하면서
# embedding만 압축된 형태로 사용합니다.
#
# Usage:
#   bash scripts/train_pca_finetune.sh <rank_k>
#
# Example:
#   bash scripts/train_pca_finetune.sh 128
#
# ==============================================================================

set -e

if [ -z "$1" ]; then
    echo "Usage: bash scripts/train_pca_finetune.sh <rank_k>"
    exit 1
fi

RANK_K=$1

PROJECT_DIR="/home/ghlee/ML_2025_SNUT"
cd "$PROJECT_DIR"

DATA_DIR="/home/ghlee/nanoGPT/data/wikitext103"

# Baseline checkpoint (학습된 전체 가중치)
BASELINE_CKPT="model_weights/gpt_baseline"

# PCA factorized matrices
WTE_NPY="model_weights/pca_factorized/wte_pca_k${RANK_K}.npy"
SCALE_NPZ="model_weights/pca_factorized/scale_mats_pca_k${RANK_K}.npz"

# Output directory
OUT_DIR="model_weights/gpt_pca_finetune_k${RANK_K}"

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
echo "B. PCA-Finetune: Baseline + PCA Embedding"
echo "=============================================="
echo "Config:"
echo "  Baseline checkpoint: $BASELINE_CKPT"
echo "  n_embd: 768"
echo "  n_embd_wte: $RANK_K (PCA compressed)"
echo ""
echo "Strategy:"
echo "  - Load ALL weights from baseline"
echo "  - Replace embedding with PCA version"
echo "  - Fine-tune for fewer iterations"
echo ""
echo "Output: $OUT_DIR"
echo "=============================================="

# Fine-tuning: 더 적은 iteration, 낮은 learning rate
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
    --weight_decay 0.1 \
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
    --wandb_run_name "gpt-pca-finetune-k${RANK_K}"

echo ""
echo "=============================================="
echo "B. PCA-Finetune Complete!"
echo "=============================================="
echo "Checkpoint: $OUT_DIR/ckpt.pt"
echo "=============================================="

