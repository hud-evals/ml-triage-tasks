# city-mapping-audit

Small internal project matching hotel free-text names to canonical city
strings. Owned by Mei Chen, shipped by Arjun Patel (Mei left 2025-Q4).

**Current champion**: `openai_3small`, per `reports/final_recommendation.md`.

## Layout

* `data/` — the raw booking table (source of truth for GT).
* `embeddings/` — precomputed per-method vectors + name indices.
* `ground_truth/` — canonical hotel→city mapping.
* `src/` — eval + fuzzy-match scripts.
* `configs/` — one yaml per method.
* `runs/` — historical top-K outputs per method/rev.
* `reports/` — decision docs.
* `notes/` — working files (Slack copies, one-pagers, retros).
* `prs/` — merged PR descriptions pertinent to the audit.

## Quickstart

```bash
python src/eval.py --config configs/openai3small.yaml > /tmp/out.json
```
