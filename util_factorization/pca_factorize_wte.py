#!/usr/bin/env python3
"""
PCA/SVD-based factorization of token embeddings from a trained checkpoint.

This script extracts the token embedding matrix from a trained GPT checkpoint,
performs SVD-based low-rank factorization, and saves the factorized matrices
for use in a compressed model.

Workflow:
1. Load a baseline checkpoint (e.g., out/baseline/ckpt.pt)
2. Extract the token embedding weight (transformer.wte.weight)
3. Perform SVD: W ≈ U_k @ diag(S_k) @ Vh_k
4. Save:
   - Small embedding: W_small = U_k @ diag(S_k)  [vocab_size, k]
   - Scale-up matrix: scale_up = Vh_k            [k, n_embd]

Example usage:
    python util_factorization/pca_factorize_wte.py \
        --ckpt_path out/baseline/ckpt.pt \
        --rank_k 128 \
        --out_wte_npy util_factorization/wte_pca_k128.npy \
        --out_scale_npz util_factorization/scale_mats_pca_k128.npz

Then use the factorized matrices in a new training run:
    python train.py \
        --out_dir out/pca_k128 \
        --n_embd 256 \
        --n_embd_wte 128 \
        --import_wte_npy util_factorization/wte_pca_k128.npy \
        --import_scale_matrices_npz util_factorization/scale_mats_pca_k128.npz
"""

import argparse
import os
import sys

import numpy as np
import torch


def load_wte_from_checkpoint(ckpt_path: str, device: str = 'cpu') -> torch.Tensor:
    """
    Load the token embedding weight from a checkpoint file.
    
    Args:
        ckpt_path: Path to the checkpoint file (.pt)
        device: Device to load the weights onto
    
    Returns:
        Token embedding weight tensor of shape [vocab_size, n_embd]
    """
    print(f"Loading checkpoint from: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    
    # Get the state dict (handle both compiled and non-compiled models)
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    # Try different possible key names for the token embedding
    possible_keys = [
        'transformer.wte.weight',
        '_orig_mod.transformer.wte.weight',  # For compiled models
        'wte.weight',
    ]
    
    wte_weight = None
    for key in possible_keys:
        if key in state_dict:
            wte_weight = state_dict[key]
            print(f"Found token embedding at key: '{key}'")
            break
    
    if wte_weight is None:
        # List available keys for debugging
        print("Available keys in checkpoint:")
        for k in state_dict.keys():
            if 'wte' in k.lower() or 'embed' in k.lower():
                print(f"  - {k}")
        raise KeyError("Could not find token embedding weight in checkpoint. "
                      f"Tried keys: {possible_keys}")
    
    return wte_weight


def svd_factorize(wte: torch.Tensor, rank_k: int) -> tuple:
    """
    Perform SVD-based factorization of the token embedding matrix.
    
    Given W of shape [V, d], compute:
        W ≈ U_k @ diag(S_k) @ Vh_k
    
    where:
        - U_k: [V, k] - left singular vectors (truncated)
        - S_k: [k]    - singular values (truncated)
        - Vh_k: [k, d] - right singular vectors (truncated)
    
    Returns:
        W_small: [V, k] = U_k @ diag(S_k) - the factorized small embedding
        scale_up: [k, d] = Vh_k - the projection matrix
        explained_variance_ratio: float - fraction of variance explained
    """
    print(f"Performing SVD on embedding matrix of shape {tuple(wte.shape)}...")
    
    # Perform full SVD
    # U: [V, min(V,d)], S: [min(V,d)], Vh: [min(V,d), d]
    U, S, Vh = torch.linalg.svd(wte, full_matrices=False)
    
    print(f"SVD complete. Singular values range: [{S.min().item():.6f}, {S.max().item():.6f}]")
    
    # Truncate to rank k
    U_k = U[:, :rank_k]      # [V, k]
    S_k = S[:rank_k]         # [k]
    Vh_k = Vh[:rank_k, :]    # [k, d]
    
    # Compute explained variance ratio
    total_variance = (S ** 2).sum()
    explained_variance = (S_k ** 2).sum()
    explained_variance_ratio = (explained_variance / total_variance).item()
    
    # Create the factorized embedding: W_small = U_k @ diag(S_k)
    W_small = U_k @ torch.diag(S_k)  # [V, k]
    
    # scale_up matrix is Vh_k
    scale_up = Vh_k  # [k, d]
    
    return W_small, scale_up, explained_variance_ratio, S


def compute_reconstruction_error(wte: torch.Tensor, W_small: torch.Tensor, 
                                  scale_up: torch.Tensor) -> dict:
    """
    Compute various reconstruction error metrics.
    
    Args:
        wte: Original embedding [V, d]
        W_small: Factorized embedding [V, k]
        scale_up: Projection matrix [k, d]
    
    Returns:
        Dictionary of error metrics
    """
    # Reconstruct the embedding
    W_reconstructed = W_small @ scale_up  # [V, d]
    
    # Compute errors
    diff = wte - W_reconstructed
    
    frobenius_error = torch.norm(diff, p='fro').item()
    frobenius_original = torch.norm(wte, p='fro').item()
    relative_error = frobenius_error / frobenius_original
    
    mse = (diff ** 2).mean().item()
    mae = diff.abs().mean().item()
    
    # Per-row (per-token) cosine similarity
    cos_sim = torch.nn.functional.cosine_similarity(wte, W_reconstructed, dim=1)
    mean_cos_sim = cos_sim.mean().item()
    min_cos_sim = cos_sim.min().item()
    
    return {
        'frobenius_error': frobenius_error,
        'frobenius_original': frobenius_original,
        'relative_error': relative_error,
        'mse': mse,
        'mae': mae,
        'mean_cosine_similarity': mean_cos_sim,
        'min_cosine_similarity': min_cos_sim,
    }


def save_factorized_matrices(W_small: torch.Tensor, scale_up: torch.Tensor,
                              out_wte_npy: str, out_scale_npz: str):
    """
    Save the factorized matrices to disk.
    
    Args:
        W_small: Factorized embedding [V, k]
        scale_up: Projection matrix [k, d]
        out_wte_npy: Output path for the embedding .npy file
        out_scale_npz: Output path for the scale matrices .npz file
    """
    # Ensure output directories exist
    os.makedirs(os.path.dirname(out_wte_npy) if os.path.dirname(out_wte_npy) else '.', exist_ok=True)
    os.makedirs(os.path.dirname(out_scale_npz) if os.path.dirname(out_scale_npz) else '.', exist_ok=True)
    
    # Convert to numpy
    W_small_np = W_small.cpu().numpy()
    scale_up_np = scale_up.cpu().numpy()
    
    # Save embedding
    np.save(out_wte_npy, W_small_np)
    print(f"Saved factorized embedding to: {out_wte_npy}")
    print(f"  Shape: {W_small_np.shape}")
    
    # Save scale matrices (scale_up and scale_down are the same for PCA initialization)
    # Note: scale_down = scale_up.T would be the transpose, but we save the same
    # matrix for both as a starting point - they can diverge during training
    np.savez(out_scale_npz, scale_up=scale_up_np, scale_down=scale_up_np)
    print(f"Saved scale matrices to: {out_scale_npz}")
    print(f"  scale_up shape: {scale_up_np.shape}")


def main():
    parser = argparse.ArgumentParser(
        description="PCA/SVD-based factorization of token embeddings from a trained checkpoint.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Factorize a baseline checkpoint with rank 128
  python util_factorization/pca_factorize_wte.py \\
      --ckpt_path out/baseline/ckpt.pt \\
      --rank_k 128 \\
      --out_wte_npy util_factorization/wte_pca_k128.npy \\
      --out_scale_npz util_factorization/scale_mats_pca_k128.npz

  # Then use in a new training run:
  python train.py \\
      --out_dir out/pca_k128 \\
      --n_embd 256 --n_embd_wte 128 \\
      --import_wte_npy util_factorization/wte_pca_k128.npy \\
      --import_scale_matrices_npz util_factorization/scale_mats_pca_k128.npz
        """
    )
    
    parser.add_argument('--ckpt_path', type=str, required=True,
                        help='Path to the baseline checkpoint (e.g., out/baseline/ckpt.pt)')
    parser.add_argument('--rank_k', type=int, required=True,
                        help='Target PCA rank (i.e., n_embd_wte for the compressed model)')
    parser.add_argument('--out_wte_npy', type=str, required=True,
                        help='Output path for the factorized embedding .npy file')
    parser.add_argument('--out_scale_npz', type=str, required=True,
                        help='Output path for the scale matrices .npz file')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device for loading and computing (default: cpu)')
    parser.add_argument('--show_singular_values', action='store_true',
                        help='Print the top singular values')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.ckpt_path):
        print(f"Error: Checkpoint file not found: {args.ckpt_path}")
        sys.exit(1)
    
    if args.rank_k <= 0:
        print(f"Error: rank_k must be positive, got {args.rank_k}")
        sys.exit(1)
    
    # Load token embedding from checkpoint
    wte = load_wte_from_checkpoint(args.ckpt_path, args.device)
    vocab_size, n_embd = wte.shape
    
    print(f"\nOriginal embedding shape: [{vocab_size}, {n_embd}]")
    print(f"Original parameters: {vocab_size * n_embd:,}")
    
    # Validate rank
    max_rank = min(vocab_size, n_embd)
    if args.rank_k > max_rank:
        print(f"Warning: rank_k ({args.rank_k}) exceeds maximum possible rank ({max_rank}). "
              f"Using {max_rank} instead.")
        args.rank_k = max_rank
    
    # Perform SVD factorization
    W_small, scale_up, explained_var_ratio, singular_values = svd_factorize(
        wte, args.rank_k
    )
    
    # Print summary
    print(f"\n{'='*60}")
    print("FACTORIZATION SUMMARY")
    print(f"{'='*60}")
    print(f"Target rank (k): {args.rank_k}")
    print(f"Factorized embedding shape: [{vocab_size}, {args.rank_k}]")
    print(f"Scale-up matrix shape: [{args.rank_k}, {n_embd}]")
    
    # Parameter counts
    original_params = vocab_size * n_embd
    factorized_params = vocab_size * args.rank_k + args.rank_k * n_embd
    compression_ratio = original_params / factorized_params
    
    print(f"\nParameter comparison:")
    print(f"  Original (wte):        {original_params:,}")
    print(f"  Factorized (wte+scale): {factorized_params:,}")
    print(f"  Compression ratio:     {compression_ratio:.2f}x")
    print(f"  Parameter reduction:   {(1 - 1/compression_ratio)*100:.1f}%")
    
    print(f"\nExplained variance ratio: {explained_var_ratio*100:.2f}%")
    
    # Show top singular values if requested
    if args.show_singular_values:
        print(f"\nTop {min(20, args.rank_k)} singular values:")
        for i, sv in enumerate(singular_values[:min(20, args.rank_k)]):
            print(f"  σ_{i+1}: {sv.item():.6f}")
    
    # Compute reconstruction error
    errors = compute_reconstruction_error(wte, W_small, scale_up)
    print(f"\nReconstruction quality:")
    print(f"  Relative Frobenius error: {errors['relative_error']*100:.4f}%")
    print(f"  MSE: {errors['mse']:.6e}")
    print(f"  MAE: {errors['mae']:.6e}")
    print(f"  Mean cosine similarity:  {errors['mean_cosine_similarity']:.6f}")
    print(f"  Min cosine similarity:   {errors['min_cosine_similarity']:.6f}")
    
    # Save the factorized matrices
    print(f"\n{'='*60}")
    print("SAVING MATRICES")
    print(f"{'='*60}")
    save_factorized_matrices(W_small, scale_up, args.out_wte_npy, args.out_scale_npz)
    
    print(f"\n{'='*60}")
    print("USAGE")
    print(f"{'='*60}")
    print("To use these factorized matrices in a new training run:")
    print(f"""
python train.py \\
    --out_dir out/pca_k{args.rank_k} \\
    --n_embd {n_embd} \\
    --n_embd_wte {args.rank_k} \\
    --import_wte_npy {args.out_wte_npy} \\
    --import_scale_matrices_npz {args.out_scale_npz} \\
    [... other training arguments ...]
""")
    
    print("Done!")


if __name__ == "__main__":
    main()

