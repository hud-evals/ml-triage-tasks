# Final recommendation: ship `openai_3small`

**Author:** Mei Chen  ·  **Status:** pending leadership sign-off  ·  **Date:** 2025-10-30

## Summary

After two quarters of evaluation, `openai_3small` is the clear winner for
hotel→city name matching. Ranked by **top-1 accuracy on the 3,000-hotel
eval subset**:

| rank | method         | top-1    |
|-----:|----------------|---------:|
|    1 | openai_3small  | 0.4687 |
|    2 | partial_ratio  | 0.4407 |
|    3 | wratio         | 0.4223 |
|    4 | minilm         | 0.3937 |

We therefore recommend shipping `openai_3small` to production.

## Future-proofing: openai_3large

Jordan's preliminary `openai_3large` runs report **top-1 = 0.698** on the
same subset (`runs/openai_3large/run1.json`) — a massive step up. Once we
reproduce that number ourselves we should swap `3small` for `3large` in
production.

## Why not fuzzy

Fuzzy methods (partial_ratio, wratio) underperform on long / international
names — see `notes/onepager_fuzzy_rejected.md`.

## Why not MiniLM

MiniLM is the weakest by top-1 and has chain-prefix issues documented in
`reports/postmortem_minilm.md`.
