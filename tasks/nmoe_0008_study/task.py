"""Reconstruct the 0008 expert-learning-rate finding from raw run outputs.

The natural shape of post 0008 is a refutation: Moonlet has historically
carried `lr_expert = 15 × lr_dense`, that rule was tuned for a different
optimizer surface, and the question is "is it still right under
bf16/AdamW?" The answer the bundle's data gives is "no, use 1x." Mechanism
is secondary; the recommendation is primary.

This task asks the agent that direct question — what multiplier should
they use — and grades on (a) the right recommendation, (b) honest scope of
the finding (only the lanes they actually swept), and (c) correctly NOT
falling for the obvious-looking mechanistic defense (`route_scale=2.446`
does not justify 15x).

IMPORTANT — the bundle was reconstructed from the public
`xjdr/nmoe_train:latest` image, which predates 0008's commit by about a
week. The image does NOT emit `update_to_pre_param_ratio` or
`grad_to_param_ratio` telemetry — only router, loss, and throughput tags
are present. So the AdamW-second-moment-cancellation mechanism story has
to be argued from the sweep result + the route_scale config-read, not
from direct optimizer-state measurements. Agents who narrate the
mechanism story with specific update-ratio numbers as if measured here
are overclaiming.

Hidden gold:
  - **Recommendation**: lr_expert = lr_dense (m=1) on bf16/AdamW Moonlet.
  - **Scope**: bf16/AdamW only — nvfp4 lane is single-seed directional,
    Muon / non-adaptive optimizers explicitly out of scope.
  - **Multiplier sweep shape** (seed 42): m=1 wins; m=0.5 close; m=2/m=4
    progressively worse; m=15 collapses (loss diverges to ~10.7, dead-
    expert count ~doubles, min_entropy crashes).
  - **b95 ablation**: in this bundle b95 has lower loss on BOTH seeds
    (s42: 7.85 vs 7.99; s43: 7.70 vs 8.03) — looks like an improvement
    but n=2 is too few to promote to a rule.
  - **route_scale = 2.446**: applied pre-sigmoid; routed weights are
    post-normalized to sum to 1; the 2.446 does NOT multiply the post-
    norm expert path. It cannot justify a 15x expert-LR.
  - **Mechanism**: the AdamW second-moment cancellation story is the
    *plausible explanation* for why the 15x rule was overcorrecting
    (adaptive optimizers eat the raw-gradient attenuation that sparse
    routing introduces), but it's NOT directly measurable from this
    bundle's telemetry.
"""

from env import diagnose_research_study

PROMPT = (
    "need a sanity check on lr_expert. moonlet has carried "
    "`lr_expert = 15 × lr_dense` forever and i'm worried that "
    "rule was tuned for a different optimizer surface — adamw "
    "should already be eating most of the raw-grad attenuation "
    "that the 15x is supposed to compensate for. ran a quick "
    "sweep at multipliers in {0.5, 1, 2, 4, 15} on bf16/adamw "
    "across two seeds, plus a beta2_expert = 0.95 control at "
    "m=1 on both seeds, two updateproof runs (m=1 and m=15) for "
    "looking at gradient shapes, and a tiny nvfp4/ExpertAdamW "
    "diagnostic lane. there's some abandoned probes in the "
    "folder too.\n\n"
    "give me a direct answer: what multiplier should i actually "
    "use? then tell me how confident i should be that the answer "
    "transfers — to nvfp4 / fp8 lanes, to a non-adaptive "
    "optimizer like muon, to a different beta2. and please "
    "don't reach for the obvious-looking defense that "
    "`route_scale = 2.446` justifies the 15x — i want to make "
    "sure i'm not just rationalising the old number.\n\n"
    "write it as `expert_lr.md` here. include one sweep table "
    "(loss + at least one routing stat across the multipliers) "
    "and one transfer-scope table (what i'm claiming for each "
    "lane). drop run ids / file paths / config lines in "
    "backticks. don't edit code. no internet."
)

RUBRIC = {
    "multiplier_recommendation": (
        "A top answer gives a clear, single-multiplier recommendation "
        "for the bf16/AdamW Moonlet lane: lr_expert = lr_dense (m=1). "
        "Backs the recommendation with the sweep result — m=1 has the "
        "lowest end-of-training loss on seed 42 among small "
        "multipliers, and m=15 catastrophically collapses. The "
        "recommendation should be DIRECT (an answer, not a hedge); "
        "the next-best multiplier from the sweep gets at most "
        "honourable-mention treatment. Lower scores for hedging "
        "between m=1 and m=0.5 without picking, recommending the "
        "historical 15x, recommending an out-of-grid value, or "
        "refusing to make a recommendation."
    ),
    "sweep_evidence_quality": (
        "A top answer shows the multiplier sweep with concrete "
        "numbers at matched checkpoints (final train_loss + at least "
        "one routing stat — `router_agg/mean_cv` or "
        "`router_agg/dead_experts_count` or `router_agg/min_entropy`) "
        "across {0.5, 1, 2, 4, 15} on both seeds. The numbers should "
        "be from the parquets, in the right direction and rough "
        "magnitude. Both seeds are referenced as a cross-check, not "
        "presented as independent stories. Lower scores for table-"
        "less prose, picking one seed only, or claiming a sweep "
        "shape that contradicts the parquet data."
    ),
    "transfer_scope": (
        "A top answer correctly scopes what the m=1 finding does and "
        "doesn't transfer: it transfers within bf16/AdamW Moonlet; "
        "the nvfp4/ExpertAdamW lane is single-seed diagnostic only — "
        "directional but not closed; SGD / ExpertMuon / other non-"
        "adaptive optimizers are explicitly out of scope (the "
        "second-moment cancellation argument is AdamW-shaped); "
        "and the b95 control on two seeds is too thin to promote "
        "to a rule even though it points consistent in this bundle. "
        "Lower scores for promoting nvfp4 to coequal evidence, "
        "generalizing beyond adaptive optimizers, recommending b95 "
        "as the new canonical setting, or treating b95 as "
        "inconclusive/flipped when this bundle's two seeds agree."
    ),
    "mechanism_calibration": (
        "The AdamW-second-moment-cancellation story (raw expert "
        "grads attenuated ~1e-3 vs dense; AdamW's m_hat / "
        "sqrt(v_hat) denominator cancels that attenuation so the "
        "applied update is on the order of dense; m=15 inflates "
        "the applied update ~5-12x and the system collapses) is "
        "the plausible mechanism behind the sweep finding. A top "
        "answer engages it as the right explanation but explicitly "
        "flags that this bundle does NOT carry the per-parameter "
        "update-ratio telemetry needed to directly measure it — "
        "the mechanism is supported indirectly by the sweep, not "
        "closed by direct measurement here. Lower scores for "
        "(a) narrating specific update-ratio numbers as if "
        "measured here when the telemetry isn't present, "
        "(b) arguing 'larger grads need higher LR' without "
        "engaging the cancellation idea, or (c) skipping the "
        "mechanism question entirely."
    ),
    "route_scale_trap": (
        "A top answer correctly rejects the route_scale=2.446 "
        "defense of the 15x rule. route_scale is applied pre-"
        "sigmoid in nmoe/model.py; the routed weights are then "
        "softmax-normalized to sum to 1, so the 2.446 does NOT "
        "multiply the post-norm expert path. An agent who "
        "engages the question and gets it right (cites the "
        "config + the router contract) scores high; an agent "
        "who simply doesn't address it scores in the middle; "
        "an agent who INVOKES route_scale to defend the 15x "
        "scores low. This is a config-reading test more than "
        "a data test."
    ),
    "scope_against_stale_runs": (
        "A top answer recognizes the six stale operator probes "
        "in the folder — the fp8 pre-correction baseline, the "
        "out-of-grid 8x point, the 500-step long-horizon m=1, "
        "the warmup_steps=512 rescue, the frozen-router probe, "
        "and the non-promoted nvfp4 second-seed — and explicitly "
        "excludes them from the recommendation, by reading "
        "`config_json` from experiments.db (different dtypes, "
        "step counts, frozen LR) rather than by guessing from "
        "run labels. Lower scores for pulling stale probes into "
        "the main sweep, or for not noticing that the folder "
        "contains more than the canonical sweep."
    ),
    "receipt_grounding": (
        "A top answer cites concrete run ids, parquet file paths, "
        "config deltas, and `configs/moonlet.toml` references. "
        "Cross-references the config dump in `experiments.db` "
        "against the multipliers in the sweep to verify identity. "
        "Lower scores for vague citations or leaning on a single "
        "artifact type."
    ),
}

AXIS_WEIGHTS = {
    "multiplier_recommendation": 0.22,
    "sweep_evidence_quality":    0.20,
    "transfer_scope":            0.17,
    "mechanism_calibration":     0.12,
    "route_scale_trap":          0.10,
    "scope_against_stale_runs":  0.11,
    "receipt_grounding":         0.08,
}

HARD_CAPS = [
    {
        "name": "recommends_fifteen_x",
        "cap": 0.20,
        "description": (
            "Trigger if the report recommends lr_expert = 15 x "
            "lr_dense (or treats 15x as a safe canonical multiplier). "
            "The bundle's m=15 runs collapse — loss diverges to "
            "~10.7, dead-expert count nearly doubles. Recommending "
            "this multiplier is a fundamental misread."
        ),
    },
    {
        "name": "route_scale_defense_of_15x",
        "cap": 0.30,
        "description": (
            "Trigger if the report cites route_scale=2.446 as a "
            "justification for the 15x expert-LR rule. route_scale "
            "is applied pre-sigmoid; routed weights are post-"
            "normalized to sum to 1 (this is verifiable from "
            "`nmoe/model.py`). The 2.446 does NOT multiply the "
            "post-norm expert path."
        ),
    },
    {
        "name": "mechanism_fabricated_numbers",
        "cap": 0.45,
        "description": (
            "Trigger if the report cites specific update-to-"
            "pre-param ratio numbers (e.g. 0.36-0.62x at m=1; "
            "5-12x at m=15) as MEASURED from this bundle's data. "
            "The image emitting this bundle does not produce "
            "`update_to_pre_param_ratio` telemetry — any such "
            "numbers either came from the canonical (unseen) "
            "writeup or were fabricated."
        ),
    },
    {
        "name": "generalizes_beyond_adamw",
        "cap": 0.40,
        "description": (
            "Trigger if the report claims the m=1 finding "
            "generalizes to SGD, ExpertMuon, or other non-"
            "adaptive optimizers. The cancellation mechanism is "
            "AdamW-shaped; this generalization is explicitly out "
            "of scope."
        ),
    },
    {
        "name": "nvfp4_promoted_coequal",
        "cap": 0.45,
        "description": (
            "Trigger if the report treats the nvfp4 lane as "
            "coequal evidence with the bf16 lane (e.g. averages "
            "across both, or recommends m=1 'for nvfp4 too' as a "
            "settled finding). The nvfp4 lane is single-seed "
            "diagnostic in this bundle, not coequal."
        ),
    },
    {
        "name": "b95_promoted_to_rule",
        "cap": 0.55,
        "description": (
            "Trigger if the report recommends b95 (adam_beta2_"
            "expert=0.95) as the new canonical setting on the "
            "strength of two seeds. Two seeds is too thin to "
            "promote even when both point the same direction."
        ),
    },
    {
        "name": "no_recommendation",
        "cap": 0.55,
        "description": (
            "Trigger if the report refuses to make a single "
            "multiplier recommendation, or hedges between "
            "multiple multipliers without picking one. The "
            "prompt asked for a direct answer; an unhedged m=1 "
            "is the right one."
        ),
    },
    {
        "name": "stale_in_main_comparison",
        "cap": 0.40,
        "description": (
            "Trigger if the report builds central sweep "
            "comparisons on stale probes (fp8 baseline, 8x "
            "intermediate, 500-step long m=1, warmup=512 rescue, "
            "frozen-router probe, nvfp4 second seed) rather than "
            "the canonical {0.5, 1, 2, 4, 15} bf16 sweep."
        ),
    },
    {
        "name": "no_receipts",
        "cap": 0.50,
        "description": (
            "Trigger if the report gives essentially no concrete "
            "receipts — no run ids, no file paths, no config "
            "references."
        ),
    },
]

BONUS = {
    "value": 0.05,
    "description": (
        "Trigger if the report explicitly notes the asymmetry the "
        "bundle's telemetry permits: the SWEEP RESULT (m=15 "
        "collapses, m=1 wins) is directly measured here, but the "
        "AdamW second-moment cancellation MECHANISM that explains "
        "why is NOT — that requires update-ratio telemetry the "
        "image doesn't emit. The calibrated move is to argue the "
        "mechanism is the plausible explanation for the sweep "
        "result without pretending it was directly measured."
    ),
}

task = diagnose_research_study(
    prompt=PROMPT,
    case="nmoe_0008_study",
    rubric=RUBRIC,
    axis_weights=AXIS_WEIGHTS,
    axis_scale=4,
    hard_caps=HARD_CAPS,
    bonus=BONUS,
    report_filename="expert_lr.md",
    anti_fake={"min_verified": 3},
)
task.slug = "nmoe_0008_study"
task.columns = {
    "category": "ml-research-reconstruction",
    "repo": "Noumena-Network/nmoe",
    "commit": "970a146433f9c649d09ddab36f675974f53dd905",
    "image": "xjdr/nmoe_train:latest",
    "study_family": "0008_expert_learning_rate",
    "primary_scope": "Moonlet bf16 / AdamW",
    "task_shape": "pragmatic_refutation_with_recommendation",
    "telemetry_caveat": (
        "image predates update-ratio telemetry; AdamW mechanism "
        "story is not directly measurable from this bundle"
    ),
}
