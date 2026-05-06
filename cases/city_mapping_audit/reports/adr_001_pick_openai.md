# ADR-001: Ship openai_3small for hotel→city matching

**Status:** proposed  ·  **Deciders:** Mei Chen, Jordan Rao, Arjun Patel

## Context

We need a production name-matching service by Q1 2026. Four methods were
evaluated on the 3,000-hotel subset: minilm, openai_3small, partial_ratio,
wratio. See `reports/final_recommendation.md` for the table.

## Decision

Ship **openai_3small** as the primary path. Decision is load-bearing on the
**top-1 accuracy metric**.

## Consequences

Positive: clearest winner at top-1, strong and reliable across the corpus.

Negative: recurring OpenAI spend. See `budget.md`.

## Revisit

Once `openai_3large` numbers are reproduced (see Jordan's run1.json: 0.698),
re-evaluate whether to migrate.
