# PR #094 — Fix sentence-transformers dim assertion

**Author:** Priya  ·  **Merged:** 2025-09-29  ·  **Reviewer:** Mei

## Summary

`embed_local.py` had an `assert vecs.shape[1] == 384` that would fire
on any future SentenceTransformer variant. Relaxed to log a warning
instead.

## Impact

None on the canonical MiniLM path (dim is still 384). Prevents
future ablations from requiring a code change.
