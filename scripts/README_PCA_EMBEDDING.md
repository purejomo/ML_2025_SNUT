# PCA-Based Embedding Compression Experiments

Token Embedding의 Low-rank Factorization 효과를 분석하는 실험 파이프라인입니다.

## 실험 구성 (4가지 비교군)

| 모델 | Embedding | Dim | Transformer | 학습 방식 | 목적 |
|------|-----------|-----|-------------|----------|------|
| **D. Baseline** | Random init | 768 | Random init | Full training | Upper bound |
| **A. PCA-Scratch** | PCA init | 128 | Random init | Full training | PCA 초기화 효과 |
| **B. PCA-Finetune** | PCA init | 128 | Baseline 유지 | Fine-tune | 효율적 압축 |
| **C. Random-Scratch** | Random init | 128 | Random init | Full training | 대조군 |

## Quick Start

### 전체 실험 한 번에 실행

```bash
bash scripts/run_all_experiments.sh 128
```

### 개별 실험 실행

```bash
# D. Baseline (768d, full training)
bash scripts/train_baseline_gpt2.sh

# PCA Factorization
bash scripts/run_pca_factorization.sh

# A. PCA-Scratch (128d, PCA init, full training)
bash scripts/train_pca_compressed.sh 128

# B. PCA-Finetune (128d, PCA init, fine-tune from baseline)
bash scripts/train_pca_finetune.sh 128

# C. Random-Scratch (128d, random init, full training)
bash scripts/train_random_lowrank.sh 128
```

## 실험이 보여주는 것

```
┌─────────────────────────────────────────────────────────┐
│                    Comparisons                          │
├─────────────────────────────────────────────────────────┤
│  A vs C  →  PCA 초기화 효과                              │
│            (같은 128d에서 PCA vs Random)                 │
│                                                         │
│  A vs D  →  차원 축소 비용                               │
│            (128d vs 768d)                               │
│                                                         │
│  B vs A  →  Fine-tune vs Scratch 효율성                 │
│            (Baseline 지식 활용 여부)                     │
│                                                         │
│  B vs D  →  실용적 압축 가능성                           │
│            (압축 + fine-tune으로 baseline 근접 가능?)    │
└─────────────────────────────────────────────────────────┘
```

## 파라미터 비교

| Model | Embedding Params | Compression | 
|-------|------------------|-------------|
| D. Baseline (768d) | 38.6M | 1.0x |
| A/B/C (128d) | 6.6M | **5.8x** |
| (64d) | 3.3M | **11.7x** |

## Scripts Reference

| Script | Model | Description |
|--------|-------|-------------|
| `train_baseline_gpt2.sh` | D | Full embedding baseline |
| `run_pca_factorization.sh` | - | SVD factorization |
| `train_pca_compressed.sh <k>` | A | PCA init + scratch training |
| `train_pca_finetune.sh <k>` | B | PCA init + fine-tune from baseline |
| `train_random_lowrank.sh <k>` | C | Random init + scratch training |
| `run_all_experiments.sh [k]` | All | Run all experiments |

## 결과 분석

### W&B Dashboard
- Project: `new-small-gpt`
- Runs: `gpt-baseline`, `gpt-pca-k128`, `gpt-pca-finetune-k128`, `gpt-random-lowrank-k128`

### 예상 결과 해석

| 결과 | 해석 |
|------|------|
| A ≈ D | 768d 중 128d만 실제로 필요! |
| A > C | PCA 초기화가 효과적 |
| A ≈ C | 차원 축소 자체가 중요, 초기화는 무관 |
| B ≈ D | 압축 + fine-tune으로 full 성능 달성 가능 |

## Output Directories

```
model_weights/
├── gpt_baseline/           # D. Baseline
├── pca_factorized/         # PCA matrices
│   ├── wte_pca_k64.npy
│   ├── wte_pca_k128.npy
│   └── scale_mats_pca_k*.npz
├── gpt_pca_k128/           # A. PCA-Scratch
├── gpt_pca_finetune_k128/  # B. PCA-Finetune
└── gpt_random_lowrank_k128/ # C. Random-Scratch
```

## Troubleshooting

### "Checkpoint not found"
Baseline 학습이 완료된 후 PCA factorization 및 B 실험을 실행하세요.

### "Shape mismatch"
`n_embd_wte` 값이 PCA factorization에서 사용한 rank와 일치하는지 확인하세요.

### Memory issues
`batch_size`를 줄이거나 `block_size`를 1024로 낮추세요.

