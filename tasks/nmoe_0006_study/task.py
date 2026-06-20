"""Reconstruct the 0006 super-4096 sparsity study from raw receipts."""

from env import diagnose_research_study

PROMPT = (
    "finished a sparsity ablation with a handful of runs across a few "
    "expert counts plus one aux-loss control. before i turn the 4096 "
    "setup into an internal note, i want to pressure-test what the "
    "runs actually show rather than what i'm already inclined to "
    "write. the checkout is at the root of this folder; the sweep "
    "outputs landed in `data/` — start with `AGENTS.md` if you need "
    "to orient on the repo.\n\n"
    "quick MoE refresher in case you need it: a dense block has one "
    "FFN every token hits. MoE replaces that with `E` expert FFNs and "
    "per token the router picks top-`K` of them (plus any shared) to "
    "run. active FFN width stays dense-equivalent, the pool grows "
    "with `E`. the balance metrics below are the primary signal for "
    "router health. tokens-per-expert at fixed total tokens is "
    "`total_tokens × K / E`.\n\n"
    "tell me which runs should i consider for actual comparison, what "
    "the metrics actually demonstrate, how the lower-E runs and the aux "
    "control fit in, and where the 4096 run data could be overclaiming. "
    "write it up as `study.md` here — include an anchor-checkpoint "
    "table, a controls comparison, and drop run ids, paths, and "
    "checkpoints in backticks so i can click through. no internet."
)

RUBRIC = {
    "scope_selection_and_framing": (
        "The bundle has 10 runs under `metrics/` with config metadata in "
        "`experiments.db`. Identify runs by `run_id` + `config_json` "
        "delta. A top answer recognizes the following four as the real "
        "comparison set and centers them:\n"
        "  - `run_1776990705_32` — bf16, steps=12000, E=4096, aux=0.0, "
        "bias=1e-4, shared=1 (the main 4096 subject; note status "
        "`completed_target` means target_loss hit early)\n"
        "  - `run_1776990754_32` — bf16, steps=2048, E=4096, aux=1e-4, "
        "bias=0, shared=1 (matched aux-on falsifier paired with the "
        "main subject's early window)\n"
        "  - `run_1776990812_32` — bf16, steps=2048, E=1024 (density "
        "control)\n"
        "  - `run_1776990865_32` — bf16, steps=2048, E=2048 (density "
        "control)\n"
        "A top answer explicitly excludes the other six runs as stale / "
        "superseded / out-of-scope (fp8 pre-dtype-switch baselines, "
        "short 1024-step bf16 first-passes, an aux=1e-3 probe that "
        "over-tuned, a n_shared=0 architecture probe, and an 8192-step "
        "aux variant that breaks matched-step comparability). The "
        "framing treats the 4096 runs as probing whether extreme "
        "sparsity breaks the routed system, not as a leaderboard or "
        "best-expert-count sweep. Lower scores for pulling stale runs "
        "into the comparison, missing any of the four anchors, or "
        "framing the note as a ranking / ablation sweep."
    ),
    "empirical_reconstruction": (
        "A top answer reconstructs the main empirical picture from the "
        "parquet metric files under `data/metrics/` and the run "
        "metadata in `data/experiments.db`: loss keeps improving at "
        "extreme sparsity while collapse indicators (routing entropy, "
        "expert utilization, dead-expert counts, or equivalent "
        "telemetry defined in `nmoe/metrics.py`) stay severe.\n\n"
        "GOLD NUMERIC ANCHORS for `run_1776990705_32` (the 4096 main):\n"
        "  step 100:  train/loss ≈ 7.26, router_agg/mean_cv ≈ 1583, "
        "router_agg/min_entropy ≈ 2.46, router_agg/dead_experts_count "
        "≈ 31k\n"
        "  step 1000: loss ≈ 4.07, mean_cv ≈ 1640, min_entropy ≈ 2.94\n"
        "  step 2000: loss ≈ 3.63, mean_cv ≈ 1627, min_entropy ≈ 3.26\n"
        "  step 4000: loss ≈ 3.37, mean_cv ≈ 1552 (loss has dropped "
        "by 2x+ while mean_cv stays pinned near 1500–1650)\n"
        "Use these tuples as reference: claims within ~20% of the gold "
        "value + correct direction earn high scores; wildly off values "
        "or wrong direction indicate fabrication or misreading and "
        "should score low. Lower scores for vague prose without "
        "numbers, wrong direction, or citing checkpoints that don't "
        "support the central divergence story."
    ),
    "aux_control_reasoning": (
        "A top answer treats the aux-on run (`run_1776990754_32`: "
        "aux_loss_alpha=1e-4, router_bias_update_rate=0, E=4096, 2048 "
        "steps) as a real falsifier of the 'balancing-pressure fixes "
        "it' story relative to the early window of the 4096 main run "
        "(`run_1776990705_32`) at matched step 2000.\n\n"
        "GOLD MATCHED-STEP COMPARISON (step≈2000, E=4096, bf16):\n"
        "  main 4096 (run_1776990705_32): mean_cv≈1627, "
        "min_entropy≈3.26, layer-by-layer max_load% L0..L11 = "
        "[14.3, 7.3, 12.9, 13.8, 13.8, 12.5, 13.6, 14.3, 12.1, 14.2, "
        "14.3, 14.2]\n"
        "  aux-on (run_1776990754_32): mean_cv≈941, min_entropy≈3.70, "
        "layer-by-layer max_load% L0..L11 = [4.6, 2.7, 3.3, 4.5, 5.2, "
        "4.0, 3.9, 3.9, 4.1, 4.2, 4.5, 5.3]\n"
        "That is: aux cuts mean_cv roughly in half (1627 → 941) and "
        "crushes every-layer max_load from a pinned-at-ceiling (~13-14% "
        "= 1/K ceiling) to a uniform ~3-5%. Bias-update alone does NOT "
        "produce this effect. This falsifies the 'bias-update suffices' "
        "framing and should be central to the note. Lower scores for "
        "mentioning aux but not drawing the falsification, or concluding "
        "aux does not matter (which inverts the truth)."
    ),
    "density_control_reasoning": (
        "A top answer uses the E=1024 and E=2048 runs comparatively to "
        "show that more tokens per expert (lower E at fixed token "
        "budget) attenuates but does NOT cleanly remove the regime — "
        "i.e., rejects the simple claim that lowering E alone is "
        "sufficient to remove hard saturation.\n\n"
        "GOLD NUMERIC ANCHORS at matched step 2000, layer-by-layer "
        "max_load% (L0..L11), bf16:\n"
        "  E=1024 (run_1776990812_32): [5.8, 10.3, 14.3, 14.3, 12.7, "
        "14.0, 14.2, 14.1, 12.9, 14.3, 13.7, 14.2]\n"
        "  E=2048 (run_1776990865_32): [14.3, 7.1, 7.6, 13.4, 14.3, "
        "14.3, 14.3, 13.9, 14.2, 14.2, 13.3, 14.3]\n"
        "  E=4096 (run_1776990705_32): [14.3, 7.3, 12.9, 13.8, 13.8, "
        "12.5, 13.6, 14.3, 12.1, 14.2, 14.3, 14.2]\n"
        "Critical reading: even at E=1024, 10 of 12 layers are still "
        "pinned near the 1/K=1/7≈14.3% ceiling. Only L0 (and a bit L1) "
        "come off the ceiling. That means lowering E helps the "
        "shallowest layers but does NOT resolve saturation in the bulk "
        "of the network. A report that frames E=1024 or E=2048 as a "
        "'sweet spot' or implies lowering E fixes the regime is "
        "reading only the L0 / aggregate number and missing the "
        "layer-level ceiling. Top answers make the tokens-per-expert "
        "framing explicit. Lower scores for treating controls as "
        "redundant or concluding density alone resolves the issue."
    ),
    "collapse_geometry_and_depth": (
        "The prompt does not explicitly demand a depth-geometry "
        "analysis, but the per-layer telemetry tells a clear story: "
        "collapse is NOT early-layer-first.\n\n"
        "GOLD NUMERIC ANCHORS — per-layer max_load% (L0..L11) at step "
        "2000 for the 4096 main run (run_1776990705_32):\n"
        "  [14.3, 7.3, 12.9, 13.8, 13.8, 12.5, 13.6, 14.3, 12.1, 14.2, "
        "14.3, 14.2]\n"
        "Read that distribution: L0 is at 14.3% (ceiling), L1 is the "
        "HEALTHIEST layer at 7.3%, and L2–L11 are mostly pinned near "
        "ceiling. The pattern is depth-nonuniform, not shallow-first. "
        "An early-layer-first story would predict L0<L1<...<L11 on "
        "max_load (shallow healthier, deep more collapsed). The data "
        "shows the opposite: L0 is already at ceiling while L1 is "
        "best. A top answer either (a) engages with the layerwise "
        "evidence and reaches this depth-nonuniform reading, or (b) "
        "explicitly flags that the depth question is open rather than "
        "silently assuming an early-layer-first story. Lower scores "
        "only for asserting early-layer-first collapse as fact. Not "
        "engaging with depth at all earns a middle score, not zero."
    ),
    "claim_boundary_and_calibration": (
        "A top answer is calibrated about what the runs actually "
        "establish versus what remains unclear, matching the hidden "
        "boundary in substance (whatever vocabulary the report uses): "
        "loss-improves-while-routing-collapses is real; aux-loss "
        "balancing alone does NOT rescue the regime; tokens-per-expert "
        "pressure contributes but is not the full story; lowering E "
        "alone is NOT sufficient to remove hard saturation; an "
        "early-layer-first collapse reading is wrong; and the precise "
        "mechanism behind the collapse is NOT closed by this bundle. "
        "The report need not use the words 'supported / falsified / "
        "open' — what matters is that its confidence-per-claim matches "
        "the above. Does NOT overclaim mechanism closure. Lower scores "
        "for treating everything as equally settled, or for declaring "
        "the mechanism solved."
    ),
    "receipt_grounding_and_corpus_use": (
        "A top answer cites concrete run ids, file paths, checkpoints, "
        "and config deltas, and clearly reflects broad reading across "
        "`nmoe/metrics.py`, `nmoe/train.py`, `configs/speedrun/"
        "small_moe_super.toml`, `data/experiments.db` metadata, and "
        "step-level parquet files under `data/metrics/<run_id>/`. "
        "Lower scores for vague citations, or for leaning on a single "
        "artifact type (e.g. only the config, only one run's metrics)."
    ),
}

AXIS_WEIGHTS = {
    # Prompt-addressed axes get the bulk of the weight.
    "scope_selection_and_framing":      0.18,
    "empirical_reconstruction":         0.18,
    "aux_control_reasoning":            0.15,
    "density_control_reasoning":        0.13,
    # Central to the new prompt ("where the 4096 reading would be
    # overclaiming") — bumped to reflect that.
    "claim_boundary_and_calibration":   0.19,
    # Not directly prompted; rewards expertise without penalizing absence.
    "collapse_geometry_and_depth":      0.09,
    "receipt_grounding_and_corpus_use": 0.08,
}

HARD_CAPS = [
    {
        "name": "aux_does_not_matter",
        "cap": 0.25,
        "description": (
            "Trigger if the report asserts that the aux-only control does "
            "not materially change the regime, or that balancing pressure "
            "is irrelevant to the collapse story."
        ),
    },
    {
        "name": "lowering_E_resolves",
        "cap": 0.35,
        "description": (
            "Trigger if the report concludes that lowering the routed "
            "expert count E alone is sufficient to remove hard saturation."
        ),
    },
    {
        "name": "early_layer_first",
        "cap": 0.40,
        "description": (
            "Trigger if the report asserts collapse is early-layer-first "
            "as its depth-geometry conclusion."
        ),
    },
    {
        "name": "mechanism_closed",
        "cap": 0.40,
        "description": (
            "Trigger if the report claims the precise underlying mechanism "
            "for the collapse regime is already identified and closed by "
            "the current bundle."
        ),
    },
    {
        "name": "stale_scope",
        "cap": 0.35,
        "description": (
            "Trigger if the report builds its central comparisons on "
            "stale or superseded nearby runs rather than the corrected "
            "baseline rerun and the matched control set."
        ),
    },
    {
        "name": "no_receipts",
        "cap": 0.50,
        "description": (
            "Trigger if the report gives essentially no concrete receipts "
            "— no run ids, no file paths, no checkpoints, no config "
            "references."
        ),
    },
]

BONUS = {
    "value": 0.05,
    "description": (
        "Trigger if the report explicitly makes the calibrated move that "
        "the receipts show something real is breaking but do not yet "
        "identify the exact damaged object or attractor mechanism."
    ),
}

task = diagnose_research_study(
    prompt=PROMPT,
    case="nmoe_0006_study",
    rubric=RUBRIC,
    axis_weights=AXIS_WEIGHTS,
    axis_scale=4,
    hard_caps=HARD_CAPS,
    bonus=BONUS,
    report_filename="study.md",
    anti_fake={"min_verified": 3},
)
task.slug = "nmoe_0006_study"
task.columns = {
    "category": "ml-research-reconstruction",
    "repo": "Noumena-Network/nmoe",
    "commit": "970a146433f9c649d09ddab36f675974f53dd905",
    "study_family": "0006_super_sparsity",
    "anchor_runs": [
        "run_1773434558_119414",
        "run_1773441395_7407",
        "run_1773437178_465",
        "run_1773455592_12991",
    ],
}
