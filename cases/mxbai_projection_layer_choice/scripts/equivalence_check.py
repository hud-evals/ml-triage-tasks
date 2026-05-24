"""Empirically verify that the 2-layer FFN projection (identity activation)
produces outputs identical to the collapsed single-linear projection
W_eff = W2 @ W1.

Encoding pipeline:
  hidden = backbone(x).last_hidden_state                # (B, T, 256)
  out_2layer = (hidden @ W1.T) @ W2.T                   # (B, T, 48)
  out_collapsed = hidden @ W_eff.T                       # (B, T, 48), W_eff = W2 @ W1

Both should be byte-identical up to floating-point associativity. Any
non-zero difference is numerical noise from a different order of FMAs.
This is the empirical complement to `svd_per_layer.py`'s rank argument.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
from safetensors.torch import load_file as load_safetensors
from transformers import AutoModel, AutoTokenizer


SAMPLE_TEXTS = [
    "What is the capital of France?",
    "Photosynthesis converts light energy into chemical energy.",
    "The Eiffel Tower was completed in 1889.",
    "Quantum mechanics describes the behavior of matter and energy at very small scales.",
    "MaxSim scoring in ColBERT computes the sum over query tokens of the max similarity to any document token.",
    "Bake at 350F for 25 minutes or until golden brown.",
    "The Pythagorean theorem relates the lengths of the sides of a right triangle.",
    "BGE-Reranker outputs a raw classification logit, not a probability.",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="model_files")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32  # use fp32 so any divergence is genuine, not casting

    tok = AutoTokenizer.from_pretrained(args.model_dir)
    backbone = AutoModel.from_pretrained(args.model_dir, torch_dtype=dtype).to(device).eval()

    w1 = load_safetensors(os.path.join(args.model_dir, "1_Dense", "model.safetensors"))
    w2 = load_safetensors(os.path.join(args.model_dir, "2_Dense", "model.safetensors"))
    W1 = next(iter(w1.values())).to(device).to(dtype)   # (512, 256)
    W2 = next(iter(w2.values())).to(device).to(dtype)   # (48, 512)
    W_eff = W2 @ W1                                       # (48, 256)

    enc = tok(SAMPLE_TEXTS, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
    with torch.no_grad():
        hidden = backbone(**enc).last_hidden_state       # (B, T, 256)
        out_2layer = (hidden @ W1.T) @ W2.T               # (B, T, 48)
        out_collapsed = hidden @ W_eff.T                  # (B, T, 48)

    mask = enc["attention_mask"].bool()
    out_2layer_np = out_2layer.cpu().numpy()
    out_collapsed_np = out_collapsed.cpu().numpy()
    mask_np = mask.cpu().numpy()

    per_text = []
    for b in range(len(SAMPLE_TEXTS)):
        valid = mask_np[b]
        a = out_2layer_np[b][valid]      # (T_valid, 48)
        c = out_collapsed_np[b][valid]
        diff = a - c
        per_text.append({
            "text_index": b,
            "n_tokens": int(valid.sum()),
            "max_abs_diff": float(np.abs(diff).max()),
            "mean_abs_diff": float(np.abs(diff).mean()),
            "max_abs_value_2layer": float(np.abs(a).max()),
            "relative_max_diff": float(np.abs(diff).max() / (np.abs(a).max() + 1e-9)),
            "cosine_per_token_min": float(
                (a * c).sum(-1).min() / (
                    np.linalg.norm(a, axis=-1).min() *
                    np.linalg.norm(c, axis=-1).min() + 1e-12
                )
            ),
        })

    overall = {
        "max_abs_diff_overall": max(p["max_abs_diff"] for p in per_text),
        "mean_abs_diff_overall": float(np.mean([p["mean_abs_diff"] for p in per_text])),
        "max_relative_diff_overall": max(p["relative_max_diff"] for p in per_text),
    }

    out = {
        "model": args.model_dir,
        "dtype": str(dtype),
        "device": device,
        "n_texts": len(SAMPLE_TEXTS),
        "embedding_dim": int(W_eff.shape[0]),
        "backbone_hidden_dim": int(W_eff.shape[1]),
        "comparison": (
            "out_2layer = (backbone_hidden @ W1.T) @ W2.T  vs "
            "out_collapsed = backbone_hidden @ W_eff.T   where W_eff = W2 @ W1"
        ),
        "expectation": (
            "Identical up to floating-point associativity (different FMA "
            "order in chained matmul vs single matmul)."
        ),
        "overall": overall,
        "per_text": per_text,
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
