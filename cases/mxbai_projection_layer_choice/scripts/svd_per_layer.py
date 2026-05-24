"""SVD each of the projection layers individually plus their composition.

Both layers in the 17M model have identity activation (see
`model_files/1_Dense/config.json` and `model_files/2_Dense/config.json`),
so the composition is mathematically a single linear map. This script
reports the singular spectrum of each piece so we can compare the
effective rank of the 2-layer pipeline against what a single linear
projection of the same output dim would have.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
from safetensors.torch import load_file as load_safetensors


def svd_summary(name: str, M: np.ndarray) -> dict:
    _, S, _ = np.linalg.svd(M, full_matrices=False)
    cum = np.cumsum(S**2) / np.sum(S**2)
    return {
        "matrix": name,
        "shape": list(M.shape),
        "rank_full": int(min(M.shape)),
        "singular_values_top": S[:20].tolist(),
        "singular_values_tail": S[-5:].tolist() if S.size > 25 else None,
        "rank_99_energy": int(np.searchsorted(cum, 0.99) + 1),
        "rank_95_energy": int(np.searchsorted(cum, 0.95) + 1),
        "rank_90_energy": int(np.searchsorted(cum, 0.90) + 1),
        "rank_50_energy": int(np.searchsorted(cum, 0.50) + 1),
        "effective_rank_participation_ratio": float((S.sum())**2 / (S**2).sum()),
        "frobenius_norm": float(np.linalg.norm(M, "fro")),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="model_files")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    w1 = load_safetensors(os.path.join(args.model_dir, "1_Dense", "model.safetensors"))
    w2 = load_safetensors(os.path.join(args.model_dir, "2_Dense", "model.safetensors"))
    W1 = next(iter(w1.values())).float().numpy()    # (512, 256)
    W2 = next(iter(w2.values())).float().numpy()    # (48, 512)
    W_eff = W2 @ W1                                  # (48, 256)

    out = {
        "1_Dense": svd_summary("W1 = 1_Dense.weight (in=256 -> hidden=512)", W1),
        "2_Dense": svd_summary("W2 = 2_Dense.weight (hidden=512 -> out=48)", W2),
        "composed": svd_summary("W_eff = W2 @ W1 (in=256 -> out=48)", W_eff),
        "rank_bound_note": (
            "The composed map W_eff is the equivalent single-linear "
            "projection. Its rank is bounded by min(rank W1, rank W2). "
            "Any single linear layer with the same input and output dims "
            "would also be at most rank min(in, out) = min(256, 48) = 48. "
            "So the 2-layer FFN with identity activation has identical "
            "EXPRESSIVITY to a single linear projection of the same shape."
        ),
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
