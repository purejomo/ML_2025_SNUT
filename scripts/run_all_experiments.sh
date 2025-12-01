#!/bin/bash
# ==============================================================================
# Master Script: Run All PCA Embedding Compression Experiments
# ==============================================================================
#
# 이 스크립트는 4가지 실험을 순차적으로 실행합니다:
#
# D. Baseline         - Full embedding (768d), scratch 학습
# A. PCA-Scratch      - PCA embedding (128d), scratch 학습  
# B. PCA-Finetune     - PCA embedding (128d), baseline에서 fine-tune
# C. Random-Scratch   - Random embedding (128d), scratch 학습
#
# Usage:
#   bash scripts/run_all_experiments.sh [rank_k]
#
# Example:
#   bash scripts/run_all_experiments.sh 128
#
# ==============================================================================

set -e

RANK_K=${1:-128}  # Default rank = 128

PROJECT_DIR="/home/ghlee/ML_2025_SNUT"
cd "$PROJECT_DIR"

echo "=============================================="
echo "PCA Embedding Compression Experiments"
echo "=============================================="
echo "Rank K: $RANK_K"
echo ""
echo "Experiments:"
echo "  D. Baseline (768d)"
echo "  A. PCA-Scratch (${RANK_K}d)"
echo "  B. PCA-Finetune (${RANK_K}d)"
echo "  C. Random-Scratch (${RANK_K}d)"
echo "=============================================="
echo ""

# ==============================================================================
# D. Baseline (이미 학습된 경우 스킵)
# ==============================================================================
if [ -f "model_weights/gpt_baseline/ckpt.pt" ]; then
    echo "[D. Baseline] Already trained. Skipping..."
else
    echo "[D. Baseline] Training..."
    bash scripts/train_baseline_gpt2.sh
fi
echo ""

# ==============================================================================
# PCA Factorization (이미 생성된 경우 스킵)
# ==============================================================================
if [ -f "model_weights/pca_factorized/wte_pca_k${RANK_K}.npy" ]; then
    echo "[PCA Factorization] Already done for k=$RANK_K. Skipping..."
else
    echo "[PCA Factorization] Running..."
    bash scripts/run_pca_factorization.sh
fi
echo ""

# ==============================================================================
# A. PCA-Scratch
# ==============================================================================
if [ -f "model_weights/gpt_pca_k${RANK_K}/ckpt.pt" ]; then
    echo "[A. PCA-Scratch] Already trained. Skipping..."
else
    echo "[A. PCA-Scratch] Training..."
    bash scripts/train_pca_compressed.sh "$RANK_K"
fi
echo ""

# ==============================================================================
# B. PCA-Finetune
# ==============================================================================
if [ -f "model_weights/gpt_pca_finetune_k${RANK_K}/ckpt.pt" ]; then
    echo "[B. PCA-Finetune] Already trained. Skipping..."
else
    echo "[B. PCA-Finetune] Training..."
    bash scripts/train_pca_finetune.sh "$RANK_K"
fi
echo ""

# ==============================================================================
# C. Random-Scratch
# ==============================================================================
if [ -f "model_weights/gpt_random_lowrank_k${RANK_K}/ckpt.pt" ]; then
    echo "[C. Random-Scratch] Already trained. Skipping..."
else
    echo "[C. Random-Scratch] Training..."
    bash scripts/train_random_lowrank.sh "$RANK_K"
fi
echo ""

# ==============================================================================
# Summary
# ==============================================================================
echo "=============================================="
echo "All Experiments Complete!"
echo "=============================================="
echo ""
echo "Results saved to:"
echo "  D. model_weights/gpt_baseline/"
echo "  A. model_weights/gpt_pca_k${RANK_K}/"
echo "  B. model_weights/gpt_pca_finetune_k${RANK_K}/"
echo "  C. model_weights/gpt_random_lowrank_k${RANK_K}/"
echo ""
echo "Check W&B dashboard for training curves:"
echo "  Project: new-small-gpt"
echo ""
echo "Expected comparisons:"
echo "  A vs C → PCA initialization effect"
echo "  A vs D → Embedding compression cost"
echo "  B vs A → Fine-tune vs Scratch efficiency"
echo "=============================================="

