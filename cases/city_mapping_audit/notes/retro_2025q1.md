# Q1 2025 retro — hotel→city matching

**Authors:** Mei, Jordan  ·  **Date:** 2025-04-03

## What went well

- `openai_3small` shipped (see `reports/final_recommendation.md`).
- Stratified eval harness stood up (`src/eval.py`, `runs/stratified/`).
- OpenAI spend stayed ~40% under Q1 budget.

## What went poorly

- Ablations lagged. We intended four ablations this quarter; only one
  (`runs/minilm_l12_ablation.json`, "MiniLM-L12 at 34.5% top-1") got
  into the harness, and we haven't reproduced it since Jordan left.
- `openai_3large` runs stalled after the contractor handoff.

## Next quarter

- Reproduce `minilm_l12_ablation.json` numbers from scratch.
- Fix the `openai_3large` pipeline.
- Decide what to do about tie-break behavior on fuzzy scorers
  (see `notes/slack_eval_bugs_thread.md`).
