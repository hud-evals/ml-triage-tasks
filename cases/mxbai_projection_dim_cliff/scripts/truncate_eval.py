"""Sweep post-hoc projection-dim truncation on pre-encoded ColBERT
embeddings. Three strategies per dim: naive slice, PCA top-K of the
per-token embedding distribution, random orthonormal rotation then slice.

Reads the encoded embeddings written by `encode_corpus.py`; writes a
single `truncation_ndcg.json` per output dir with per-(strategy, dim) NDCG@10.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import time

import numpy as np
import torch


def maxsim(q_t: torch.Tensor, d_ts: list[torch.Tensor]) -> np.ndarray:
    out = np.zeros(len(d_ts), dtype=np.float32)
    for i, d in enumerate(d_ts):
        sim = q_t @ d.T
        out[i] = float(sim.max(dim=1).values.sum().item())
    return out


def ndcg_at_k(ranked: list[str], rels: dict[str, int], k: int = 10) -> float:
    dcg = 0.0
    for i, did in enumerate(ranked[:k]):
        rel = rels.get(did, 0)
        if rel > 0:
            dcg += (2**rel - 1) / math.log2(i + 2)
    ideal = sorted(rels.values(), reverse=True)
    idcg = 0.0
    for i, rel in enumerate(ideal[:k]):
        if rel > 0:
            idcg += (2**rel - 1) / math.log2(i + 2)
    return dcg / idcg if idcg > 0 else 0.0


def renorm(embs: list[np.ndarray]) -> list[np.ndarray]:
    return [e / (np.linalg.norm(e, axis=-1, keepdims=True) + 1e-9) for e in embs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoded-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dims", default="48,40,32,24,16,8")
    ap.add_argument("--strategies", default="naive,pca,random")
    ap.add_argument(
        "--dataset",
        default="",
        help="dataset name to stamp into the output for provenance",
    )
    ap.add_argument(
        "--model",
        default="",
        help="model id to stamp into the output for provenance",
    )
    args = ap.parse_args()

    with open(os.path.join(args.encoded_dir, "doc_embeddings.pkl"), "rb") as f:
        d = pickle.load(f)
        doc_ids, doc_embs = d["ids"], d["embeddings"]
    with open(os.path.join(args.encoded_dir, "query_embeddings.pkl"), "rb") as f:
        q = pickle.load(f)
        query_ids, q_embs = q["ids"], q["embeddings"]
    with open(os.path.join(args.encoded_dir, "qrels.pkl"), "rb") as f:
        qrels = pickle.load(f)

    full_dim = doc_embs[0].shape[-1]
    print(f"  full_dim={full_dim} ndoc={len(doc_embs)} nq={len(q_embs)}")

    # Fit a PCA basis on the per-token embedding distribution
    sample = np.concatenate(doc_embs[:2000], axis=0)
    sample -= sample.mean(0, keepdims=True)
    _, pca_S, pca_Vt = np.linalg.svd(sample, full_matrices=False)
    pca_cum = (np.cumsum(pca_S**2) / np.sum(pca_S**2)).tolist()

    rng = np.random.default_rng(0)
    R, _ = np.linalg.qr(rng.normal(size=(full_dim, full_dim)).astype(np.float32))

    def truncate(embs, dim, strat):
        if strat == "naive":
            return [e[..., :dim] for e in embs]
        if strat == "pca":
            return [(e @ pca_Vt.T)[..., :dim] for e in embs]
        if strat == "random":
            return [(e @ R)[..., :dim] for e in embs]
        raise ValueError(strat)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dims = [int(x) for x in args.dims.split(",")]
    strategies = args.strategies.split(",")

    per_strategy: dict[str, dict[int, dict]] = {s: {} for s in strategies}
    for strat in strategies:
        for dim in dims:
            if dim > full_dim:
                continue
            t0 = time.time()
            d_trunc = renorm(truncate(doc_embs, dim, strat))
            q_trunc = renorm(truncate(q_embs, dim, strat))
            d_ts = [torch.from_numpy(e).to(device).float() for e in d_trunc]
            ndcgs = []
            for qi, qv in enumerate(q_trunc):
                q_t = torch.from_numpy(qv).to(device).float()
                scores = maxsim(q_t, d_ts)
                order = np.argsort(-scores)
                ranked = [doc_ids[i] for i in order]
                rels = qrels.get(query_ids[qi], {})
                ndcgs.append(ndcg_at_k(ranked, rels, k=10))
            avg = float(np.mean(ndcgs))
            per_strategy[strat][dim] = {
                "ndcg_at_10": avg,
                "n_queries": len(ndcgs),
                "elapsed_sec": time.time() - t0,
            }
            print(f"  {strat:6s} dim={dim:3d}  NDCG@10={avg:.4f}  ({time.time()-t0:.1f}s)")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "dataset": args.dataset,
            "model": args.model,
            "full_dim": int(full_dim),
            "n_queries": len(q_embs),
            "n_docs": len(doc_embs),
            "per_strategy": per_strategy,
            "pca_cumulative_energy_top_k": pca_cum,
        }, f, indent=2)


if __name__ == "__main__":
    main()
