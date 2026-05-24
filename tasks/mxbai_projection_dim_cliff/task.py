"""Projection-dim deployment recommendation. The agent is dropped into the
researcher's working dir mid-decision: pylate package at the root, the
team's encode/truncate/svd scripts under scripts/, last-sprint's trained-
dim ablation under experiments/trained_dim_ablation/, this-sprint's
post-hoc truncation under experiments/truncation_post_hoc/, the released
17M model under model_files/. Researcher's question, delivered in the
prompt, is "can we ship a smaller-projection variant of the 17M model
without retraining, and what's the right dim/strategy?"

The hidden ground truth (not in the workspace, used by the grader):

  - The TRAINED-dim cliff in the last-sprint ablation lands between dim
    48 (0.5967) and dim 32 (0.5772) — a -3.3% absolute drop. The team
    has been treating this as the lower bound for ship-quality.

  - POST-HOC truncation on the shipped 48-dim model behaves DIFFERENTLY:
    PCA-32 retains 0.6753 vs full-dim 0.6922 (-1.7% absolute on scifact,
    half the trained-cliff drop). Naive-32 retains 0.6645 (-2.8%).
    Random-32 retains 0.6591. PCA wins at every dim < 48.

  - The actual POST-HOC cliff in our data lives between dim 16 and dim 8:
    PCA goes 0.591 -> 0.302 (-49% relative), naive 0.553 -> 0.282 (-49%),
    random 0.536 -> 0.181 (-66%). Every other adjacent step is single-
    digit % degradation.

  - The reason for the trained-vs-post-hoc divergence: the SVD of the
    projection matrix W_eff = W2 @ W1 (48 x 256) is nearly full-rank
    (rank-for-95%-energy = 45, eff rank 46.8), but the PCA of the
    per-token embedding distribution is heavily concentrated (top-1 axis
    holds 66.7% of variance, top-16 holds 83.1%). The model packs signal
    into a low-rank-effective subspace of its 48-dim output even though
    the projection matrix itself uses all 48 dims evenly. PCA truncation
    keeps that subspace; naive truncation throws away whichever dims
    happened to hold variance.

  - Storage at dim K with avg ~208 tokens per doc and fp16: 20017 bytes
    at K=48, 10008 at K=24. The customer's <10kB/doc constraint forces
    dim <= 24.

  - The correct call: PCA-truncate to 24 (within 10kB/doc, retains 0.6488
    vs 0.6922 = -4.34 NDCG points, within the <5-point budget). PCA-32
    is better but violates storage budget. Avoid naive truncation when
    PCA is essentially free at deploy time (one 48x48 matmul per token).
"""

from env import diagnose_research_study

PROMPT = (
    "engineering wants to ship a smaller-projection variant of the "
    "17M edge model without us re-training it. their constraint is "
    "under 10kb/doc at fp16, willing to lose under 5 ndcg points vs "
    "the shipped 48-dim. i need to send them a recommendation note "
    "today.\n\n"
    "context: last sprint we ran the trained-dim ablation — train a "
    "fresh 48 / 32 / 24 / 16 / etc projection head end-to-end at each "
    "dim, see where ndcg falls off. results in "
    "`experiments/trained_dim_ablation/results.json`. cliff there sits "
    "between dim 48 and dim 32 (4-subset nanobeir avg: 0.5967 -> "
    "0.5772). that's been my working number for 'how low can we go'.\n\n"
    "but trained-from-scratch is the wrong question for engineering's "
    "ask — we already SHIPPED 48-dim, they want to truncate without a "
    "retrain. so this morning i ran a post-hoc truncation pass on the "
    "released model: three strategies (naive slice, pca top-k of the "
    "per-token embedding distribution, random orthonormal rotation "
    "then slice), six dims (48 / 40 / 32 / 24 / 16 / 8), on beir/"
    "scifact (300 queries, 5183 docs). data lands in "
    "`experiments/truncation_post_hoc/truncation_ndcg.json`. also "
    "ran the svd of the composed projection matrix W2 @ W1 — "
    "`experiments/truncation_post_hoc/projection_svd.json`.\n\n"
    "i need the recommendation note: which dim, which truncation "
    "strategy, why that satisfies engineering's constraint, AND an "
    "honest treatment of how the post-hoc truncation curves relate to "
    "the trained-dim cliff (they're not the same curve — i want to "
    "make sure i'm not overclaiming or underclaiming what we can do).\n\n"
    "workspace layout:\n"
    "  - `pylate/` — pylate package; encoding goes through "
    "`pylate.models.ColBERT` (see `scripts/encode_corpus.py`)\n"
    "  - `examples/` — pylate's own reference scripts (the beir eval "
    "shape we built on)\n"
    "  - `scripts/encode_corpus.py`, `truncate_eval.py`, "
    "`svd_projection.py`, `cost_analysis.py` — the scripts i ran for "
    "this sprint's truncation experiment. encode dumps doc/query "
    "embeddings to a pickle dir; truncate_eval + cost_analysis both "
    "consume that dir and write into experiments/truncation_post_hoc/. "
    "(intermediate encoded pickles were huge so they're not in the "
    "workspace; re-run encode_corpus.py if you need them.)\n"
    "  - `model_files/` — the released 17M model. `1_Dense/` is "
    "Linear(256 -> 512, identity), `2_Dense/` is Linear(512 -> 48, "
    "identity); see their `config.json` files. composing them gives "
    "the effective W_eff that scripts/svd_projection.py decomposes.\n"
    "  - `experiments/trained_dim_ablation/results.json` — last "
    "sprint's trained-from-scratch curve\n"
    "  - `experiments/truncation_post_hoc/{truncation_ndcg,projection"
    "_svd,cost_analysis}.json` — this sprint's outputs\n\n"
    "write `recommendation.md` in this workspace root. concrete "
    "(dim, strategy) pick, justification from the numbers in the "
    "json files, byte budget worked through with `cost_analysis.json`, "
    "explicit comparison of trained-vs-post-hoc curves, and a call on "
    "where the post-hoc floor actually sits. no live encoding. work "
    "from what's on disk."
)

RUBRIC = {
    "truncation_curve_reading": (
        "A top answer reads `experiments/truncation_post_hoc/"
        "truncation_ndcg.json` and surfaces per-strategy NDCG: at "
        "dim=32, naive 0.6645 vs pca 0.6753 vs random 0.6591; at "
        "dim=24, naive 0.6352 vs pca 0.6488 vs random 0.6378; at "
        "dim=16, naive 0.5527 vs pca 0.5912 vs random 0.5360. "
        "Identifies that PCA consistently wins at every dim < 48 and "
        "the spread between strategies grows as dim shrinks. Lower "
        "scores for citing only one strategy or for missing the PCA win."
    ),
    "svd_pca_interpretation": (
        "A top answer reads `experiments/truncation_post_hoc/"
        "projection_svd.json` and surfaces the two distinct rank "
        "profiles: W_eff is nearly full-rank (rank-for-95%-energy = "
        "45 out of 48, effective rank ~46.8) while the per-token "
        "EMBEDDING PCA shows extreme concentration (top-1 axis = "
        "66.7% of variance, top-16 = 83.1%; this lives in the "
        "`pca_cumulative_energy_top_k` field of `truncation_ndcg.json`). "
        "Explains the discrepancy: the projection matrix uses all 48 "
        "dims evenly but the model's per-token outputs concentrate "
        "in a much lower-rank subspace, which is what makes PCA "
        "truncation work. Lower scores for conflating the two SVDs, "
        "ignoring one, or not noting the discrepancy."
    ),
    "trained_vs_post_hoc": (
        "A top answer compares last sprint's TRAINED cliff "
        "(`experiments/trained_dim_ablation/results.json`: 48 -> 32 "
        "drops 0.5967 -> 0.5772, -3.3% absolute) against this sprint's "
        "measured POST-HOC PCA truncation (48 -> 32 drops 0.6922 -> "
        "0.6753, -1.7% absolute) and notes that they're measuring "
        "different things. A model trained at K=k commits all k dims "
        "to discriminative features and bottlenecks below intrinsic "
        "dim. The shipped K=48 model has slack capacity, and PCA "
        "recovers most of it because the embeddings sit on a lower-"
        "rank manifold. Lower scores for treating the two cliffs as "
        "the same number, for ignoring the comparison, or for getting "
        "the direction of comparison backwards."
    ),
    "deployment_recommendation": (
        "A top answer makes a specific dim + strategy call that "
        "respects the stated constraints (under 10kB/doc fp16, <5 "
        "NDCG points lost). 48 ships at 20017 bytes/doc per "
        "`cost_analysis.json`; the constraint requires dim <= 24 "
        "(10008 bytes/doc fp16). At dim=24 PCA gives NDCG 0.6488 vs "
        "full 0.6922 — that's 4.34 NDCG points, right at the limit. "
        "A top answer picks PCA-24 (or naive-24 with caveats) and "
        "explains the trade-off explicitly. An acceptable weaker "
        "answer recommends dim=32 with int8 quantization (13344 -> "
        "6672 bytes after quant) — must call out that int8 risk "
        "isn't measured in this data. Lower scores for picking a dim "
        "that violates the storage constraint, for not justifying "
        "the strategy choice, or for recommending a dim without "
        "showing the NDCG/storage trade."
    ),
    "post_hoc_cliff_call": (
        "A top answer identifies where the POST-HOC cliff actually "
        "lives in the measured data: between dim 16 and dim 8 (PCA: "
        "0.591 -> 0.302, naive: 0.553 -> 0.282, random: 0.536 -> "
        "0.181 — all roughly halving). Below 16 the model can no "
        "longer represent the MaxSim-relevant signal in any strategy. "
        "Crucially this is DIFFERENT from where the trained cliff "
        "sits in the last-sprint results (between 48 and 32). Lower "
        "scores for conflating the two cliff locations."
    ),
    "cost_quantification": (
        "A top answer cites the actual byte counts at the recommended "
        "dim from `experiments/truncation_post_hoc/cost_analysis.json` "
        "and shows the storage save vs the 48-dim baseline (20017 "
        "bytes/doc -> ?). Lower scores for not engaging the cost side."
    ),
    "alternatives_considered": (
        "A top answer mentions ONE or more credible alternatives: "
        "(a) int8 quantization of per-token vectors (4x storage win, "
        "small quality loss in literature) — but not measured here; "
        "(b) product quantization / PLAID-style centroid encoding; "
        "(c) document-side late interaction with token pruning; "
        "(d) recommending engineering accept the 32M variant if the "
        "device can spare the params. Notes the workspace doesn't "
        "measure any of these — they're directions, not "
        "recommendations. Lower scores for treating an alternative "
        "as the primary recommendation or not considering any."
    ),
    "receipt_grounding": (
        "A top answer ties each claim to a specific source value: a "
        "NDCG cell in `truncation_ndcg.json`, an energy field in "
        "`projection_svd.json`, a per-dim row in `cost_analysis.json`, "
        "or a dim entry in `experiments/trained_dim_ablation/"
        "results.json`. Lower scores for vague citations."
    ),
}

AXIS_WEIGHTS = {
    "truncation_curve_reading":   0.17,
    "svd_pca_interpretation":     0.15,
    "trained_vs_post_hoc":        0.15,
    "deployment_recommendation":  0.17,
    "post_hoc_cliff_call":        0.12,
    "cost_quantification":        0.08,
    "alternatives_considered":    0.09,
    "receipt_grounding":          0.07,
}

HARD_CAPS = [
    {
        "name": "treats_trained_cliff_as_truncation_cliff",
        "cap": 0.35,
        "description": (
            "Trigger if the recommendation treats last sprint's "
            "TRAINED cliff (between dim 48 and dim 32) as the cliff "
            "they should expect from post-hoc truncation. The "
            "measured truncation curves directly refute this — "
            "post-hoc PCA at dim 32 loses only 1.7% absolute vs the "
            "trained 3.3%. The cliffs differ because they measure "
            "different things."
        ),
    },
    {
        "name": "recommends_dim_violating_storage_budget",
        "cap": 0.30,
        "description": (
            "Trigger if the final recommendation specifies a dim "
            "that violates the stated <10kB/doc fp16 budget. From "
            "`cost_analysis.json`: dim 48 = 20017 b/doc, dim 32 = "
            "13344 b/doc, dim 24 = 10008 b/doc. Only dims <= 24 "
            "satisfy the constraint at fp16."
        ),
    },
    {
        "name": "recommends_naive_without_engaging_pca",
        "cap": 0.45,
        "description": (
            "Trigger if the recommendation defaults to naive "
            "truncation as the deployment strategy without engaging "
            "the consistently-better PCA truncation results in "
            "`truncation_ndcg.json`. PCA wins at every dim < 48 by "
            "a widening margin as dim shrinks. The agent can "
            "justifiably recommend naive (if scoring code is "
            "immutable, etc.) but must engage WHY they're leaving "
            "PCA's gain on the table."
        ),
    },
    {
        "name": "misses_pca_vs_proj_svd_distinction",
        "cap": 0.50,
        "description": (
            "Trigger if the writeup conflates the SVD of the "
            "PROJECTION MATRIX (W_eff, nearly full-rank, eff rank "
            "~46.8) with the PCA of the EMBEDDING DISTRIBUTION (top-1 "
            "= 67% variance, top-16 = 83%). These are two different "
            "things and the gap between them is the central "
            "explanation for why PCA truncation works."
        ),
    },
    {
        "name": "no_specific_recommendation",
        "cap": 0.50,
        "description": (
            "Trigger if the writeup lists trade-offs without picking "
            "a final (dim, strategy) tuple to ship. The prompt "
            "explicitly asked for a single recommendation."
        ),
    },
    {
        "name": "post_hoc_cliff_wrong_location",
        "cap": 0.55,
        "description": (
            "Trigger if the writeup places the post-hoc cliff between "
            "any dim pair OTHER than 16 and 8. In the measured data, "
            "PCA goes 0.591 -> 0.302 across that boundary (-49%), "
            "while every other adjacent step is single-digit %."
        ),
    },
    {
        "name": "no_receipts",
        "cap": 0.50,
        "description": (
            "Trigger if the writeup cites essentially no concrete "
            "values from the json files in `experiments/`. Vague "
            "qualitative claims do not ground the recommendation."
        ),
    },
]

BONUS = {
    "value": 0.05,
    "description": (
        "Trigger if the writeup explicitly notes that the post-hoc "
        "cliff sits LOWER than the trained-cliff would suggest, AND "
        "draws the right inference for production: the shipped 48-dim "
        "model has redundant capacity beyond what a from-scratch K=k "
        "training would yield, so truncation is a strictly more "
        "permissive operation than re-training at K. This is the "
        "deeper lesson from juxtaposing the trained-dim ablation with "
        "the measured truncation curves."
    ),
}

task = diagnose_research_study.task(
    prompt=PROMPT,
    case="mxbai_projection_dim_cliff",
    rubric=RUBRIC,
    axis_weights=AXIS_WEIGHTS,
    axis_scale=4,
    hard_caps=HARD_CAPS,
    bonus=BONUS,
    report_filename="recommendation.md",
    anti_fake={"min_verified": 3},
)
task.slug = "mxbai_projection_dim_cliff"
task.metadata = {
    "category": "ml-researcher-edge-deployment-recommendation",
    "shape": "researcher_asks_for_help_with_eng_recommendation_during_sprint",
    "model": "mixedbread-ai/mxbai-edge-colbert-v0-17m",
    "dataset": "BeIR/scifact (300 queries, 5183 docs)",
    "key_measurements": {
        "ndcg_at_full_48": 0.6922,
        "ndcg_at_24_pca": 0.6488,
        "ndcg_at_24_naive": 0.6352,
        "ndcg_at_16_pca": 0.5912,
        "ndcg_at_8_pca": 0.3018,
        "proj_svd_rank95_energy": 45,
        "pca_top1_energy": 0.667,
        "pca_top16_energy": 0.831,
        "trained_dim32_ndcg": 0.5772,
        "trained_dim48_ndcg": 0.5967,
    },
}
