"""Thin rapidfuzz wrapper used as a shared helper.

Not directly invoked by grader paths — eval.py / eval_v2.py call the
underlying rapidfuzz APIs themselves.
"""
from __future__ import annotations
from rapidfuzz import fuzz, process


def topk(query: str, choices: list[str], scorer: str, k: int = 3):
    fn = {"partial_ratio": fuzz.partial_ratio, "WRatio": fuzz.WRatio}[scorer]
    return process.extract(query, choices, scorer=fn, limit=k)
