"""Stratified accuracy helper — splits the eval subset on pluggable axes.

Supported axes (partial implementation; see TODOs):
  * lexical_overlap — buckets: name_contains_truth_city | no_overlap
  * name_length    — buckets: <= 10 | 11-25 | 26-40 | > 40  (TODO)
  * city_frequency — buckets: 1 | 2-5 | 6-50 | > 50         (TODO)

Only lexical_overlap is wired up today; the other axes are imported by
eval.py but currently raise NotImplementedError. Filed #stratify-axes.
"""
from __future__ import annotations
from typing import Iterable


def lexical_overlap_bucket(hotel_name: str, gt_cities: list[str]) -> str:
    lo = hotel_name.lower()
    if any(c.lower() in lo for c in gt_cities):
        return "name_contains_truth_city"
    return "no_overlap"


def name_length_bucket(hotel_name: str) -> str:
    raise NotImplementedError("wire up name_length axis — see #stratify-axes")


def city_frequency_bucket(city: str, city_counts: dict[str, int]) -> str:
    raise NotImplementedError("wire up city_frequency axis — see #stratify-axes")


def stratify(hotels: Iterable[str], gt: dict[str, list[str]], axis: str) -> dict[str, list[str]]:
    if axis == "lexical_overlap":
        out: dict[str, list[str]] = {"name_contains_truth_city": [], "no_overlap": []}
        for h in hotels:
            out[lexical_overlap_bucket(h, gt.get(h, []))].append(h)
        return out
    raise ValueError(f"unknown axis: {axis}")
