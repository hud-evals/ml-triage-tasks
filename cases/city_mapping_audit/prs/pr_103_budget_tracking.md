# PR #103 — Embed cost-tracking header in run JSONs

**Author:** Priya  ·  **Status:** draft  ·  **Reviewer:** —

## Summary

Adds an optional `cost_usd` field to run JSONs so we can track
cumulative OpenAI spend over time. Not yet merged.

## Rationale

Currently we're eyeballing monthly spend. Embedding a per-run cost
tag lets us plot spend vs delivered accuracy.

## Schema change

```json
{
  "method": "...",
  "eval_script": "...",
  "top_1": 0.xxxx,
  "cost_usd": 0.50
}
```

## Risk

None — additive, optional field.
