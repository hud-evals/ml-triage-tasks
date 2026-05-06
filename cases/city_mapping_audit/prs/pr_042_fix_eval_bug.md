# PR #042 — Fix eval tie-break duplication warning

**Author:** Mei  ·  **Merged:** 2025-10-12  ·  **Reviewer:** Jordan

## Summary

`eval.py` was emitting a deprecation warning on numpy 2.x about argsort
stability on integer-scored ties. `eval_v2.py` does the same work via an
explicit group-by-score pass, which is warning-free and ~3% faster on the
full eval subset.

## Semantic change

None — the two scripts produce identical top-K on floating-point scorers
and are within ±0.001 top-1 on integer-scored fuzzy scorers (verified on
the 3k-hotel subset).

## Next

Deprecate `eval.py` once everyone migrates. Current run artifacts under
`runs/` still use `eval.py`.
