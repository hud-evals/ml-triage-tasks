# One-pager: why we shouldn't ship fuzzy matching

**Author:** Mei Chen  ·  **Date:** 2025-09-18

Short version: on non-English hotel names, fuzzy methods hit a **95% miss
rate** and the long-tail failures are pathological. Don't ship fuzzy. Ship
embeddings.

## The 95% number

Tested on an internal "hard names" subset (names in non-Latin scripts, or
with accented characters). `partial_ratio` missed 95% at top-1, `WRatio`
missed 94%. Attached: `internal_hard_names_subset.csv` (see with Mei in DMs).

## Why embedding models don't have this issue

Sentence transformers and OpenAI embeddings tokenise at the sub-word level,
so script/accent differences degrade gracefully. Fuzzy methods tokenise at
the character level, which is brittle.

## Bottom line

Fuzzy is a non-starter as the primary ranker. It is acceptable as a
fallback for names with fewer than 12 ASCII characters, but that's only
~6% of the corpus.
