# PR #050 — Add WRatio scorer to fuzzy harness

**Author:** Priya  ·  **Merged:** 2025-10-05  ·  **Reviewer:** Mei

## Summary

Adds `configs/wratio.yaml` + a `runs/wratio/run1.json` artifact so the
fuzzy harness can be compared on WRatio head-to-head with partial_ratio.

## Numbers (3k eval subset)

WRatio: top_1 = 0.4223, top_2 = 0.4740, top_3 = 0.4803
Baseline (partial_ratio): top_1 = 0.4407, top_2 = 0.4913, top_3 = 0.4997

WRatio is marginally worse at every K. Keeping it in the harness as a
reference point but not recommending for production.

## Risk

None. Read-only addition of a config and a run JSON.
