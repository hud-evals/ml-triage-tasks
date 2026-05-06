# PR #118 — DRAFT: fuzzy-fallback routing sketch

**Author:** Priya  ·  **Status:** draft  ·  **Reviewer:** —

## Summary

Sketch of the post-ship fuzzy-fallback routing rule (per design doc
§fallback). Only for A/B. Not intended to merge as-is.

```python
def route(query: str) -> str:
    q = query.strip()
    if len(q) <= 12 and q.isascii():
        return "fuzzy"
    return "embedding"
```

## Targeted lift

+1 pp top-1 on the overlap bucket, no measurable change on the
no-overlap bucket.
