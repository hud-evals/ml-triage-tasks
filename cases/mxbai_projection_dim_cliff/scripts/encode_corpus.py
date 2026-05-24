"""Encode a BEIR corpus + queries with the shipped 17M ColBERT model and
persist the per-token embeddings to disk for downstream evaluation /
truncation experiments.

This is the same encoding step `eval_beir.py` runs in-memory, factored out
so we can experiment on the resulting per-token embeddings without
re-encoding for each sweep cell (see `truncate_eval.py`).
"""
from __future__ import annotations

import argparse
import os
import pickle
import time

import numpy as np
import torch

from pylate import evaluation, models


QUERY_LEN = {
    "scifact": 48, "nq": 32, "fiqa": 32, "nfcorpus": 32, "trec-covid": 48,
    "msmarco": 32, "quora": 32, "hotpotqa": 32, "climate-fever": 64,
    "arguana": 64, "scidocs": 48, "dbpedia-entity": 32, "webis-touche2020": 32,
    "fever": 32,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mixedbread-ai/mxbai-edge-colbert-v0-17m")
    ap.add_argument("--dataset", default="scifact")
    ap.add_argument("--out", required=True, help="output dir for pickled embeddings")
    ap.add_argument("--doc-length", type=int, default=300)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    model = models.ColBERT(
        model_name_or_path=args.model,
        document_length=args.doc_length,
        query_length=QUERY_LEN.get(args.dataset, 32),
    )

    documents, queries, qrels = evaluation.load_beir(
        dataset_name=args.dataset,
        split="dev" if "msmarco" in args.dataset else "test",
    )

    t0 = time.time()
    doc_embs = model.encode(
        sentences=[d["text"] for d in documents],
        batch_size=64, is_query=False, show_progress_bar=True,
    )
    print(f"  encoded {len(doc_embs)} docs in {time.time()-t0:.1f}s")

    t0 = time.time()
    q_embs = model.encode(
        sentences=list(queries.values()),
        batch_size=32, is_query=True, show_progress_bar=True,
    )
    print(f"  encoded {len(q_embs)} queries in {time.time()-t0:.1f}s")

    with open(os.path.join(args.out, "doc_embeddings.pkl"), "wb") as f:
        pickle.dump({
            "ids": [d["id"] for d in documents],
            "embeddings": doc_embs,
        }, f)
    with open(os.path.join(args.out, "query_embeddings.pkl"), "wb") as f:
        pickle.dump({
            "ids": list(queries.keys()),
            "embeddings": q_embs,
        }, f)
    with open(os.path.join(args.out, "qrels.pkl"), "wb") as f:
        pickle.dump(qrels, f)
    print(f"  wrote {args.out}/")


if __name__ == "__main__":
    main()
