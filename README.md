# PCA-Based Token Embedding Compression for Efficient Language Models

This project was carried out as a term project for the 2025 fall semester EIE department at SNUT.

## Overview

This project explores whether token embeddings in GPT models can be compressed using PCA/SVD-based low-rank factorization without significant performance degradation. 
We compare multiple training strategies to understand:

1. **Low-rank factorization cost**: How much performance is lost when reducing embedding dimensions?
2. **PCA effectiveness**: Does PCA initialization preserve more information than random initialization?
3. **LoRA fine-tuning efficiency**: Can LoRA fine-tuning recover performance after PCA compression?

## Key Features

- **PCA/SVD-based factorization**: Extract and factorize token embeddings from trained models
- **Low-rank embedding**: Support for factorized token embeddings with configurable rank
- **LoRA fine-tuning**: Efficient fine-tuning with Low-Rank Adaptation
- **Comprehensive logging**: Track experiments with W&B (validation loss before/after fine-tuning)

## Experimental Setup

### 13 Experiments Overview

| # | Experiment | Embedding | Rank | Training | Description |
|---|------------|-----------|------|----------|-------------|
| 1 | **Baseline** | Full | 768 | Full training | Upper bound reference |
| 2-5 | **Low-rank Scratch** | Random low-rank | 64, 128, 256, 512 | Full training | Random init + full training |
| 6-9 | **PCA (no finetune)** | PCA factorized | 64, 128, 256, 512 | None | PCA only, no training |
| 10-13 | **PCA + LoRA** | PCA factorized | 64, 128, 256, 512 | LoRA finetune | PCA init + LoRA fine-tuning |

### Detailed Experiments

#### 1. Baseline (1 experiment)
- Standard GPT model with full 768d embeddings
- Full training from scratch
- Serves as upper bound for comparison

#### 2. Low-rank Scratch (4 experiments)
- Factorized token embedding with random initialization
- Ranks: 64, 128, 256, 512
- Full training from scratch
- Measures: Cost of dimensionality reduction

#### 3. PCA (no finetune) (4 experiments)
- Apply PCA factorization to baseline's token embedding
- Ranks: 64, 128, 256, 512
- **No training** - evaluation only
- Measures: How much information PCA preserves

#### 4. PCA + LoRA (4 experiments)
- Apply PCA factorization to baseline's token embedding
- LoRA fine-tuning on: token embedding, attention (Q, K, V), FFN
- Ranks: 64, 128, 256, 512
- Measures: Can LoRA recover performance after PCA compression?

### Research Questions

| Comparison | Question |
|------------|----------|
| **Low-rank vs Baseline** | What is the cost of dimensionality reduction? |
| **PCA vs Low-rank** | Does PCA initialization preserve more information than random? |
| **PCA+LoRA vs PCA** | How much does LoRA fine-tuning improve PCA-compressed models? |
| **PCA+LoRA vs Baseline** | Can PCA+LoRA achieve baseline performance with fewer parameters? |


## Installation

### Prerequisites

```bash
# Python 3.10+
# CUDA-compatible GPU (recommended)

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy transformers datasets tiktoken wandb tqdm tensorboard rich
```

### Dataset Setup

This project uses WikiText-103 dataset. Ensure the dataset is prepared:

```bash
# Dataset should be located at:
data/wikitext103/
# With files: train.bin, val.bin, meta.pkl
```

## Quick Start

### Individual Experiments

```bash
# 1. Train baseline model (768d)
bash scripts/train_baseline_gpt2.sh

# 2. Perform PCA factorization for all ranks
bash scripts/run_pca_factorization.sh

# 3. Train Low-rank Scratch models
bash scripts/train_lowrank_scratch.sh 64
bash scripts/train_lowrank_scratch.sh 128
bash scripts/train_lowrank_scratch.sh 256
bash scripts/train_lowrank_scratch.sh 512

# 4. Evaluate PCA (no finetune) - records val loss before LoRA
bash scripts/eval_pca_no_finetune.sh 64
bash scripts/eval_pca_no_finetune.sh 128
bash scripts/eval_pca_no_finetune.sh 256
bash scripts/eval_pca_no_finetune.sh 512

# 5. Train PCA + LoRA models
bash scripts/train_pca_lora.sh 64
bash scripts/train_pca_lora.sh 128
bash scripts/train_pca_lora.sh 256
bash scripts/train_pca_lora.sh 512
```

## Project Structure

### Scripts

| Script | Description |
|--------|-------------|
| `train_baseline_gpt2.sh` | Train baseline model (768d) |
| `run_pca_factorization.sh` | PCA factorization for all ranks |
| `train_lowrank_scratch.sh` | Low-rank scratch training |
| `eval_pca_no_finetune.sh` | Evaluate PCA without fine-tuning |
| `train_pca_lora.sh` | PCA + LoRA fine-tuning |

### Key Files

- **`model.py`**: GPT model with factorized embedding and LoRA support
- **`train.py`**: Training script with W&B logging
- **`util_factorization/pca_factorize_wte.py`**: PCA factorization script

## Technical Details

### Factorized Embedding Architecture

When `n_embd_wte < n_embd`, the model uses:
- Small embedding: `wte [vocab_size, n_embd_wte]` (e.g., [50257, 128])
- Scale-up layer: `scale_up [n_embd_wte, n_embd]` (e.g., [128, 768])
- Scale-down layer: `scale_down [n_embd_wte, n_embd]` (e.g., [128, 768])
- LM head: `lm_head [n_embd_wte, vocab_size]` (e.g., [128, 50257])

**Forward Pass**:
```python
tok_emb = wte(tokens)              # [batch, seq, 128]
tok_emb = scale_up(tok_emb)        # [batch, seq, 768]
# ... transformer blocks (768d) ...
x = scale_down(x)                  # [batch, seq, 128]
logits = lm_head(x)                # [batch, seq, vocab_size]
```

### LoRA Fine-tuning

LoRA (Low-Rank Adaptation) is applied to:
- **Token embedding**: wte, scale_up, scale_down
- **Attention**: Q, K, V projections
- **FFN**: Up and down projections

### PCA Factorization

```bash
python util_factorization/pca_factorize_wte.py \
    --ckpt_path model_weights/gpt_baseline/ckpt.pt \
    --rank_k 128 \
    --out_wte_npy model_weights/pca_factorized/wte_pca_k128.npy \
    --out_scale_npz model_weights/pca_factorized/scale_mats_pca_k128.npz
```

## Parameter Comparison

| Model | Embedding Params | Compression Ratio |
|-------|------------------|-------------------|
| Baseline (768d) | 38.60M | 1.0x |
| Rank 512 | 26.13M | 1.5x |
| Rank 256 | 13.06M | 3.0x |
| Rank 128 | 6.53M | **5.9x** |
| Rank 64 | 3.27M | **11.8x** |

## Results and Monitoring

### W&B Dashboard

- **Project**: `new-small-gpt`
- **Metrics tracked**:
  - Validation loss (before/after LoRA fine-tuning)
  - Training loss
  - Perplexity
  - `ln_f_cosine`

### Key Metrics to Compare

| Metric | Description |
|--------|-------------|
| `val/loss` | Validation loss |
| `val/loss_before_lora` | Val loss before LoRA (for PCA+LoRA experiments) |
| `val/loss_after_lora` | Val loss after LoRA fine-tuning |

## Output Structure

```
model_weights/
├── gpt_baseline/                    # 1. Baseline model
├── pca_factorized/                  # PCA factorization results
│   ├── wte_pca_k64.npy
│   ├── wte_pca_k128.npy
│   ├── wte_pca_k256.npy
│   ├── wte_pca_k512.npy
│   └── scale_mats_pca_k*.npz
├── gpt_lowrank_scratch_k64/         # 2. Low-rank Scratch (k=64)
├── gpt_lowrank_scratch_k128/        # 3. Low-rank Scratch (k=128)
├── gpt_lowrank_scratch_k256/        # 4. Low-rank Scratch (k=256)
├── gpt_lowrank_scratch_k512/        # 5. Low-rank Scratch (k=512)
├── gpt_pca_lora_k64/                # 10. PCA + LoRA (k=64)
├── gpt_pca_lora_k128/               # 11. PCA + LoRA (k=128)
├── gpt_pca_lora_k256/               # 12. PCA + LoRA (k=256)
└── gpt_pca_lora_k512/               # 13. PCA + LoRA (k=512)
```

Note: Experiments 6-9 (PCA no finetune) are evaluation-only and don't save checkpoints.
