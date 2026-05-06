# Analytics dashboard — hotel→city matching

Frozen 2025-11-06 for ADR-001. All plots in `analytics/*.png`,
companion text / ASCII renderings in `analytics/*_notes.md`.

## Top-1 accuracy per method
See [top1_per_method_notes.md](top1_per_method_notes.md) and
`analytics/topk_per_method.png`.

Headline: openai_3small (0.469) > partial_ratio (0.441) > wratio
(0.422) > minilm (0.394). openai_3large reports 0.698 but is
unverified.

## Stratified: lexical_overlap
See [stratified_notes.md](stratified_notes.md) and
`analytics/stratified_topk.png`.

Headline: the average top-1 (~0.47) hides bimodal structure. Overlap-
positive hotels (47% of the subset) hit >0.85 top-1 with any method;
no-overlap hotels (53%) collapse to <0.12 top-1.

## Bucket sizes
See [bucket_sizes_notes.md](bucket_sizes_notes.md) and
`analytics/bucket_sizes.png`.

## The 3-large problem
See [suspect_3large_notes.md](suspect_3large_notes.md) and
`analytics/suspect_3large.png`.

The 22 pp step change over 3-small is almost certainly a row-order
mismatch in the .npy files rather than a real quality gain.

## Cost vs accuracy
See [cost_vs_accuracy_notes.md](cost_vs_accuracy_notes.md) and
`analytics/cost_vs_accuracy.png`.

Cheapest acceptable production stack: openai_3small primary +
partial_ratio fallback on short-ASCII overlap names.

## Quick references

- Canonical run JSONs: `runs/<method>/run1.json`.
- Canonical ground truth: `ground_truth/gt.json`.
- Eval harness: `src/eval.py --config configs/<method>.yaml`.
- Full eval log (reference): `logs/eval_full_run_2025-09-04.log`.
- Per-hotel predictions (reference): `runs/stratified/per_hotel_predictions.csv`.
