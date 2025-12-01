# PCA-Based Token Embedding Compression for GPT Models

This repository implements a comprehensive experimental framework for analyzing the effectiveness of low-rank factorization of token embeddings in GPT models using Principal Component Analysis (PCA).

## Overview

This project explores whether token embeddings in GPT models can be compressed using PCA/SVD-based low-rank factorization without significant performance degradation. We compare four different training strategies to understand:

1. **PCA initialization effectiveness**: Does initializing with PCA-compressed embeddings help?
2. **Dimensionality reduction cost**: How much performance is lost when reducing embedding dimensions?
3. **Fine-tuning efficiency**: Can we preserve baseline knowledge while using compressed embeddings?

## Key Features

- **PCA/SVD-based factorization**: Extract and factorize token embeddings from trained models
- **Multiple training strategies**: Compare scratch training vs fine-tuning with compressed embeddings
- **Automatic experiment pipeline**: Run all experiments with a single command
- **Comprehensive logging**: Track experiments with W&B and TensorBoard

## Experimental Setup

### Four Comparison Models

| Model | Embedding | Dim | Transformer | Training | Purpose |
|------|-----------|-----|-------------|----------|---------|
| **D. Baseline** | Random init | 768 | Random init | Full training | Upper bound |
| **A. PCA-Scratch** | PCA init | 128 | Random init | Full training | PCA initialization effect |
| **B. PCA-Finetune** | PCA init | 128 | Baseline weights | Fine-tune | Efficient compression |
| **C. Random-Scratch** | Random init | 128 | Random init | Full training | Control group |

### Research Questions

- **A vs C**: Does PCA initialization provide benefits over random initialization at the same dimension?
- **A vs D**: What is the cost of dimensionality reduction (128d vs 768d)?
- **B vs A**: Is fine-tuning more efficient than training from scratch?
- **B vs D**: Can compressed embeddings + fine-tuning achieve baseline performance?

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
/home/ghlee/nanoGPT/data/wikitext103/
# With files: train.bin, val.bin, meta.pkl
```

## Quick Start

### Run All Experiments

```bash
# Run all 4 experiments with rank k=128
bash scripts/run_all_experiments.sh 128
```

### Individual Experiments

```bash
# 1. Train baseline model (768d)
bash scripts/train_baseline_gpt2.sh

# 2. Perform PCA factorization
bash scripts/run_pca_factorization.sh

# 3. Train PCA-Scratch model (A)
bash scripts/train_pca_compressed.sh 128

# 4. Train PCA-Finetune model (B)
bash scripts/train_pca_finetune.sh 128

# 5. Train Random-Scratch model (C)
CUDA_VISIBLE_DEVICES=1 bash scripts/train_random_lowrank.sh 128
```

## Project Structure

### Core Files

- **`model.py`**: GPT model implementation with factorized embedding support
  - Conditional `scale_up`/`scale_down` layers when `n_embd_wte` is set
  - Weight tying between `wte` and `lm_head` for factorized embeddings
  
- **`train.py`**: Training script with PCA embedding support
  - Fixed `ln_f_cosine` calculation for factorized embeddings (see Changes section)
  - Support for importing PCA-factorized embeddings and scale matrices

- **`util_factorization/pca_factorize_wte.py`**: PCA factorization script
  - Extracts token embeddings from checkpoints
  - Performs SVD-based low-rank factorization
  - Saves factorized matrices for model initialization

### Scripts

- **`scripts/train_baseline_gpt2.sh`**: Train full baseline model (D)
- **`scripts/run_pca_factorization.sh`**: Extract and factorize embeddings
- **`scripts/train_pca_compressed.sh`**: Train with PCA init from scratch (A)
- **`scripts/train_pca_finetune.sh`**: Fine-tune baseline with PCA embeddings (B)
- **`scripts/train_random_lowrank.sh`**: Train with random low-rank init (C)
- **`scripts/run_all_experiments.sh`**: Run all experiments automatically

## Key Changes and Additions

### 1. Factorized Embedding Support in `model.py`

**Conditional Architecture**:
- When `n_embd_wte` is set, the model uses:
  - Small embedding: `wte [vocab_size, n_embd_wte]` (e.g., [50257, 128])
  - Scale-up layer: `scale_up [n_embd_wte, n_embd]` (e.g., [128, 768])
  - Scale-down layer: `scale_down [n_embd_wte, n_embd]` (e.g., [128, 768])
  - LM head: `lm_head [n_embd_wte, vocab_size]` (e.g., [128, 50257])

**Forward Pass**:
```python
# Input: tokens
tok_emb = wte(tokens)              # [batch, seq, 128]
tok_emb = scale_up(tok_emb)        # [batch, seq, 768]
# ... transformer blocks (768d) ...
x = scale_down(x)                  # [batch, seq, 128]
logits = lm_head(x)                # [batch, seq, vocab_size]
```

### 2. PCA Factorization Script (`util_factorization/pca_factorize_wte.py`)

**Functionality**:
- Loads token embedding from trained checkpoint
- Performs SVD: `W [V, d] ≈ U_k @ diag(S_k) @ Vh_k`
- Saves:
  - `wte_pca_k{k}.npy`: Factorized embedding `[V, k]`
  - `scale_mats_pca_k{k}.npz`: Scale matrices `[k, d]`

**Usage**:
```bash
python util_factorization/pca_factorize_wte.py \
    --ckpt_path model_weights/gpt_baseline/ckpt.pt \
    --rank_k 128 \
    --out_wte_npy model_weights/pca_factorized/wte_pca_k128.npy \
    --out_scale_npz model_weights/pca_factorized/scale_mats_pca_k128.npz
```

### 3. Fixed `ln_f_cosine` Calculation in `train.py`

**Issue**: When `n_embd_wte` is set, `ln_f` outputs `[768]` but `lm_head.weight` is `[128]`, causing dimension mismatch.

**Solution**: Apply `scale_down` transformation to match the actual forward pass:

```python
# Before (error):
cos = F.cosine_similarity(ln_f_out[0], target_vecs, dim=-1)
#                         [768]        [128]      ❌ Dimension mismatch

# After (fixed):
if self.args.n_embd_wte is not None:
    ln_f_scaled = F.linear(ln_f_out[0], self.model.transformer.scale_down.weight.t())
    #            [768] → [128] via scale_down
    cos = F.cosine_similarity(ln_f_scaled, target_vecs, dim=-1)
    #                         [128]        [128]      ✅ OK
```

**Location**: `train.py` lines ~1004-1021 and ~1136-1153

### 4. CLI Arguments

**New arguments in `train_args.py`**:
- `--n_embd_wte`: Factorized embedding dimension
- `--n_embd_wte_scale_tying`: Weight tying between scale_up and scale_down
- `--import_wte_npy`: Path to factorized embedding `.npy` file
- `--import_scale_matrices_npz`: Path to scale matrices `.npz` file
- `--import_wte_freeze`: Freeze imported embedding
- `--import_scale_matrices_freeze`: Freeze imported scale matrices

## Parameter Comparison

| Model | Embedding Params | Compression Ratio |
|-------|------------------|-------------------|
| Baseline (768d) | 38.6M | 1.0x |
| PCA k=384 | 19.9M | 1.9x |
| PCA k=256 | 13.3M | 2.9x |
| PCA k=128 | 6.6M | **5.8x** |
| PCA k=64 | 3.3M | **11.7x** |

## Results and Analysis

### Expected Outcomes

- **A > C**: PCA initialization provides better starting point than random
- **A ≈ D**: 128d is sufficient, 768d has redundancy
- **B ≈ D**: Fine-tuning with compressed embeddings can match baseline
- **B < A**: Fine-tuning is more efficient than scratch training

### Monitoring

- **W&B Dashboard**: Project `new-small-gpt`
  - Runs: `gpt-baseline`, `gpt-pca-k128`, `gpt-pca-finetune-k128`, `gpt-random-lowrank-k128`
- **Metrics**: Validation loss, perplexity, `ln_f_cosine`, training time

## Output Structure

```
model_weights/
├── gpt_baseline/              # D. Baseline model
├── pca_factorized/            # PCA factorization results
│   ├── wte_pca_k64.npy
│   ├── wte_pca_k128.npy
│   ├── wte_pca_k256.npy
│   └── scale_mats_pca_k*.npz
├── gpt_pca_k128/              # A. PCA-Scratch
├── gpt_pca_finetune_k128/     # B. PCA-Finetune
└── gpt_random_lowrank_k128/   # C. Random-Scratch
```

## Troubleshooting

### "Checkpoint not found"
Ensure baseline training is complete before running PCA factorization.

### "Shape mismatch"
Verify `n_embd_wte` matches the rank used in PCA factorization.

### "CUDA out of memory"
Reduce `batch_size` or `block_size` in training scripts.

### GPU Selection
Use `CUDA_VISIBLE_DEVICES=1` to select specific GPU:
```bash
CUDA_VISIBLE_DEVICES=1 bash scripts/train_random_lowrank.sh 128
```

## Citation

If you use this code in your research, please cite:

```bibtex
@software{PCAEmbeddingCompression,
  author = {Your Name},
  title = {PCA-Based Token Embedding Compression for GPT Models},
  year = {2025},
  url = {https://github.com/your-repo}
}
```

## License

[Add your license information here]
