"""Tiny metrics helpers — kept for convenience. Not used by the canonical
eval path (eval.py recomputes top-K inline for speed) but useful for
analysis scripts under scratch/.
"""
from __future__ import annotations
import numpy as np


def topk_accuracy(preds: np.ndarray, labels: list[list[str]],
                  label_space: np.ndarray, k: int) -> float:
    """`preds` is an (N, K) array of indices into `label_space`. Returns
    the fraction of rows where preds[:k] intersects the label list."""
    hits = 0
    chosen = label_space[preds[:, :k]]
    for i, row in enumerate(chosen):
        if set(row.tolist()) & set(labels[i]):
            hits += 1
    return round(hits / len(labels), 4)


def mrr(preds: np.ndarray, labels: list[list[str]], label_space: np.ndarray) -> float:
    rr = 0.0
    for i in range(preds.shape[0]):
        row_names = label_space[preds[i]]
        for rank, name in enumerate(row_names, 1):
            if name in labels[i]:
                rr += 1.0 / rank
                break
    return round(rr / preds.shape[0], 4)
