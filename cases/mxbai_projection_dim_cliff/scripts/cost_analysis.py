"""Compute per-projection-dim storage + scoring cost for a ColBERT
deployment, given an encoded corpus and a list of candidate dims. Writes
a json with bytes/doc, bytes/query, and MaxSim ops per (query, doc) pair
at each dim. Used to translate truncation-NDCG curves into a deployment
budget conversation.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoded-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dims", default="48,40,32,24,16,8")
    args = ap.parse_args()

    with open(os.path.join(args.encoded_dir, "doc_embeddings.pkl"), "rb") as f:
        doc_embs = pickle.load(f)["embeddings"]
    with open(os.path.join(args.encoded_dir, "query_embeddings.pkl"), "rb") as f:
        q_embs = pickle.load(f)["embeddings"]

    avg_tokens_per_doc = float(np.mean([e.shape[0] for e in doc_embs]))
    avg_tokens_per_query = float(np.mean([e.shape[0] for e in q_embs]))

    dims = [int(x) for x in args.dims.split(",")]
    per_dim_cost: dict[int, dict] = {}
    for dim in dims:
        per_dim_cost[dim] = {
            "bytes_per_doc_fp16": int(round(avg_tokens_per_doc * dim * 2)),
            "bytes_per_doc_int8": int(round(avg_tokens_per_doc * dim * 1)),
            "bytes_per_query_fp16": int(round(avg_tokens_per_query * dim * 2)),
            "maxsim_ops_per_query_doc_pair": int(round(
                avg_tokens_per_query * avg_tokens_per_doc * dim
            )),
        }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "avg_tokens_per_doc": avg_tokens_per_doc,
            "avg_tokens_per_query": avg_tokens_per_query,
            "per_dim_cost": per_dim_cost,
            "scaling_note": (
                "MaxSim per (query, doc) pair = nq * nd * dim FMA ops. "
                "Storage scales linearly with dim. 48 -> 24 halves both."
            ),
        }, f, indent=2)


if __name__ == "__main__":
    main()
