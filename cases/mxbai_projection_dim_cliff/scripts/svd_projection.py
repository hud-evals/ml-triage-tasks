"""SVD the composed projection matrix W_eff = W2 @ W1 of the released 17M
ColBERT model.

The model's two `Dense` modules (`1_Dense`: Linear(256 -> 512, identity)
and `2_Dense`: Linear(512 -> 48, identity)) compose to a single
linear map W_eff (48 x 256). Both have `activation_function:
torch.nn.modules.linear.Identity` per their respective `config.json`, so
W_eff = W2 @ W1 captures the full per-token projection (the backbone's
last_hidden_state is fed into W_eff to produce the 48-dim per-token
ColBERT embedding).

Writes the singular spectrum + rank-for-X-energy summaries to JSON.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
from safetensors.torch import load_file as load_safetensors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model-dir",
        default="model_files",
        help="dir containing 1_Dense/ and 2_Dense/ subdirs with safetensors",
    )
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    w1 = load_safetensors(os.path.join(args.model_dir, "1_Dense", "model.safetensors"))
    w2 = load_safetensors(os.path.join(args.model_dir, "2_Dense", "model.safetensors"))
    W1 = next(iter(w1.values())).float().numpy()   # (512, 256)
    W2 = next(iter(w2.values())).float().numpy()   # (48, 512)
    W_eff = W2 @ W1                                 # (48, 256)

    _, S, _ = np.linalg.svd(W_eff, full_matrices=False)
    cum_energy = np.cumsum(S**2) / np.sum(S**2)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "note": "W_eff = W2 @ W1, the composition of the two identity-activation Dense layers.",
            "W1_shape": list(W1.shape),
            "W2_shape": list(W2.shape),
            "W_eff_shape": list(W_eff.shape),
            "backbone_hidden_dim": int(W1.shape[1]),
            "projection_out_dim": int(W2.shape[0]),
            "singular_values": S.tolist(),
            "singular_value_normalized": (S / S.max()).tolist(),
            "cumulative_energy_top_k": cum_energy.tolist(),
            "rank_99_energy": int(np.searchsorted(cum_energy, 0.99) + 1),
            "rank_95_energy": int(np.searchsorted(cum_energy, 0.95) + 1),
            "rank_90_energy": int(np.searchsorted(cum_energy, 0.90) + 1),
            "rank_50_energy": int(np.searchsorted(cum_energy, 0.50) + 1),
            "effective_rank_participation_ratio": float((S.sum())**2 / (S**2).sum()),
        }, f, indent=2)


if __name__ == "__main__":
    main()
