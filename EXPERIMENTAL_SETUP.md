# Experimental Setup

This document describes the experimental setup for the PCA-based token embedding compression experiments.

## Hardware & Software Environment

### PyTorch
- **Version**: 2.5.1
- **CUDA**: 12.1
- **cuDNN**: 9.1.0 (90100)

### GPU
- **Model**: NVIDIA GeForce RTX 3090
- **Quantity**: 2 GPUs
- **Memory per GPU**: 24,576 MiB (24 GB)
- **Driver Version**: 555.42.06

### Model Library
- **Base Implementation**: Custom GPT-2 implementation (based on nanoGPT)
- **Framework**: PyTorch (native implementation, no external model library)
- **Compilation**: PyTorch 2.0+ `torch.compile()` enabled for optimization

## Model Configuration

All experiments use the same base GPT-2 architecture with variations in embedding dimensions:

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Architecture** | GPT-2 | Transformer decoder |
| **n_layer** | 12 | Number of transformer blocks |
| **n_head** | 12 | Number of attention heads per block |
| **n_embd** | 768 | Hidden dimension (transformer blocks) |
| **n_embd_wte** | 768 / 64 / 128 / 256 / 512 | Token embedding dimension (varies by experiment) |
| **mlp_intermediate_size** | 3,072 | FFN intermediate dimension (mlp_expansion_factor × n_embd = 4 × 768) |
| **vocab_size** | 50,257 | Vocabulary size (from dataset) |
| **block_size** | 2,048 | Maximum sequence length |
| **activation** | GELU | Activation function |
| **norm_attn** | LayerNorm | Attention normalization |
| **norm_output** | LayerNorm | Output normalization |
| **bias** | True | Use bias in linear layers |
| **position_encoding** | Absolute positional embeddings | No rotary embeddings |
| **dropout** | 0.1 (training) / 0.0 (eval) | Dropout rate |

### Embedding Factorization Architecture

When `n_embd_wte < n_embd`, the model uses a factorized embedding architecture:

| Component | Shape | Description |
|-----------|-------|-------------|
| **wte** | `[vocab_size, n_embd_wte]` | Compressed token embedding |
| **scale_up** | `[n_embd_wte, n_embd]` | Upscale to transformer dimension |
| **scale_down** | `[n_embd, n_embd_wte]` | Downscale from transformer dimension |
| **lm_head** | `[n_embd_wte, vocab_size]` | Language modeling head |

**Forward Pass**:
```
tok_emb = wte(tokens)              # [batch, seq, n_embd_wte]
tok_emb = scale_up(tok_emb)        # [batch, seq, n_embd]
# ... transformer blocks (n_embd) ...
x = scale_down(x)                  # [batch, seq, n_embd_wte]
logits = lm_head(x)                # [batch, seq, vocab_size]
```

## LoRA Adapter Settings

LoRA (Low-Rank Adaptation) is applied for fine-tuning experiments (Experiments 10-13):

| Parameter | Value | Description |
|-----------|-------|-------------|
| **lora_rank** | 32 | Rank of low-rank matrices (A, B) |
| **lora_alpha** | 64 | Scaling factor (alpha/rank = 2.0) |
| **lora_dropout** | 0.1 | Dropout rate in LoRA layers |
| **lora_targets** | `wte,scale_up,scale_down,q_proj,k_proj,v_proj,c_proj,mlp_up,mlp_down` | Components with LoRA adapters |

### LoRA Implementation Details

- **LoRA Matrices**: For each target layer, LoRA adds:
  - `A`: `[rank, in_features]` - Down-projection matrix
  - `B`: `[out_features, rank]` - Up-projection matrix
  - Output: `original(x) + (x @ A^T @ B^T) * (alpha/rank)`

- **Initialization**:
  - `A`: Kaiming uniform initialization
  - `B`: Zero initialization (LoRA starts as identity)

- **Target Components**:
  - **wte**: Token embedding layer
  - **scale_up**: Embedding upscale layer
  - **scale_down**: Embedding downscale layer
  - **q_proj, k_proj, v_proj**: Attention query, key, value projections
  - **c_proj**: Attention output projection
  - **mlp_up, mlp_down**: Feed-forward network up/down projections

### LoRA Parameter Count

For a typical layer with shape `[768, 768]`:
- **Original parameters**: 768 × 768 = 589,824
- **LoRA parameters** (rank=32): 32 × (768 + 768) = 49,152
- **Reduction**: ~92% fewer trainable parameters

## Training Configuration

### Baseline Training (Experiment 1)

| Hyperparameter | Value |
|----------------|-------|
| **max_iters** | 10,000 |
| **eval_interval** | 500 |
| **eval_iters** | 100 |
| **batch_size** | 4 |
| **gradient_accumulation_steps** | 4 |
| **effective_batch_size** | 16 |
| **learning_rate** | 1e-4 |
| **min_lr** | 5e-6 |
| **warmup_iters** | 500 |
| **lr_decay_iters** | 10,000 |
| **optimizer** | AdamW |
| **weight_decay** | 0.1 |
| **beta1** | 0.9 |
| **beta2** | 0.95 |
| **grad_clip** | 1.0 |
| **dtype** | float16 |
| **compile** | True |

### Low-rank Scratch Training (Experiments 2-5)

Same as baseline, except:
- `n_embd_wte`: 64, 128, 256, or 512 (varies by experiment)
- Random initialization for factorized embeddings

### PCA + LoRA Fine-tuning (Experiments 10-13)

| Hyperparameter | Value |
|----------------|-------|
| **max_iters** | 5,000 |
| **eval_interval** | 200 |
| **eval_iters** | 100 |
| **batch_size** | 4 |
| **gradient_accumulation_steps** | 4 |
| **effective_batch_size** | 16 |
| **learning_rate** | 5e-5 |
| **min_lr** | 1e-6 |
| **warmup_iters** | 100 |
| **lr_decay_iters** | 5,000 |
| **optimizer** | AdamW |
| **weight_decay** | 0.01 |
| **beta1** | 0.9 |
| **beta2** | 0.95 |
| **grad_clip** | 1.0 |
| **dtype** | float16 |
| **compile** | True |
| **init_from** | prev_run (baseline checkpoint) |
| **n_embd_wte** | 64, 128, 256, or 512 (varies by experiment) |

### PCA (No Finetune) Evaluation (Experiments 6-9)

- **Mode**: Evaluation only (`eval_only=True`)
- **No training**: Load baseline, apply PCA factorization, evaluate
- **dropout**: 0.0 (inference mode)

## Dataset

- **Dataset**: WikiText-103
- **Path**: `/home/ghlee/nanoGPT/data/wikitext103`
- **Vocabulary**: 50,257 tokens (GPT-2 tokenizer)

## Experiment Summary

| Experiment | n_embd_wte | Training Method | LoRA Rank | Description |
|------------|------------|-----------------|-----------|-------------|
| 1. Baseline | 768 | Full training | - | Upper bound reference |
| 2-5. Low-rank Scratch | 64/128/256/512 | Full training | - | Random init + full training |
| 6-9. PCA (no finetune) | 64/128/256/512 | None (eval only) | - | PCA factorization only |
| 10-13. PCA + LoRA | 64/128/256/512 | LoRA fine-tuning | 32 | PCA init + LoRA fine-tuning |

## Logging & Monitoring

- **Weights & Biases (W&B)**: All experiments logged to project `new-small-gpt`
- **Run naming**:
  - Baseline: `baseline`
  - Low-rank Scratch: `lowrank-scratch-k{64,128,256,512}`
  - PCA (no finetune): `pca-nofinetune-k{64,128,256,512}`
  - PCA + LoRA: `pca-lora-k{64,128,256,512}-r32`
- **Metrics tracked**: Validation loss, training loss, best iteration, best tokens

## File Structure

```
model_weights/
├── gpt_baseline/                    # Experiment 1
├── gpt_lowrank_scratch_k{64,128,256,512}/  # Experiments 2-5
├── gpt_pca_nofinetune_k{64,128,256,512}/   # Experiments 6-9
├── gpt_pca_lora_k{64,128,256,512}/        # Experiments 10-13
└── pca_factorized/
    ├── wte_pca_k{64,128,256,512}.npy
    └── scale_mats_pca_k{64,128,256,512}.npz
```

