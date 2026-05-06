# OpenAI spend budget

Total: **$50** for the next-experiments proposal. Any experiment involving
OpenAI embeddings or reranking counts.

## Relevant unit costs (see `cost_table.md` for full table)

* `text-embedding-3-small` — $0.020 / 1M input tokens
* `text-embedding-3-large` — $0.130 / 1M input tokens
* `gpt-4o-mini` reranker — $0.150 / 1M input + $0.600 / 1M output tokens

## Rules

* The 3 experiments must jointly stay under $50.
* Token counts must be justifiable (show the math, even if approximate).
* Experiments that do not touch OpenAI (e.g., swap local embedder, add a
  rule-based post-processor) cost $0 against this budget.
