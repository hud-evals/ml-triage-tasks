"""Retrospective on the 0011 autoresearch campaign.

The natural shape of post 0011 is a campaign post-mortem: someone ran a
bounded controller for 8 trials, the controller swept axes under a
primary objective + a CORE-drop veto, and now the question is "did the
controller find anything useful, and did the design choices actually
pay rent?" This isn't a "reconstruct the empirical pattern" task like
0006 — it's a "read the campaign's decision log" task.

The bundle ships per-candidate receipt JSONs (overrides + final metrics
+ keep/discard decision, applied to THIS bundle's real measurements,
not the canonical 2026-03-14 campaign's). The agent should read them
wave-by-wave (seed → wakeup → first parallel wave → refinement) and
answer the four design questions:

  1. who actually won?
  2. did the CORE-drop veto bite — and was it load-bearing for the
     champion choice, or just a safety rail that fired on a non-
     competitive candidate?
  3. which knob did the real lifting?
  4. how good was the champion vs the seed across the published
     three axes (loss / CORE / throughput)?

Hidden gold (real data in THIS bundle, not canonical):
  - **Champion**: `0011_refine_3axis` — the three-axis combination
    (lr_router=0.0021, aux_loss_alpha=0.00015, lr_dense=0.0022) —
    at val_loss ≈ 5.112.
  - **CORE-vetoed**: only `0011_aux_0005_vetoed` actually exceeded
    the max_core_drop=0.002 threshold (CORE drop ≈ 0.004 vs seed).
  - **CORE veto NOT load-bearing in this bundle**: aux_0005 had
    val_loss=5.21, higher than the champion's 5.11 — it would have
    lost on primary objective anyway. The veto fired as a safety
    rail, not as the deciding rule.
  - **Dominant single-axis lever**: lr_dense — the 0.0016→0.0022
    move (`lr_dense_0016_pre_wave` → `lr_dense_0022_regime_change`)
    buys ≈ 0.14 nats, the biggest single-axis swing in the bundle.
  - **Champion vs seed**: improves on val_loss (5.20→5.11) and
    CORE (-0.023→-0.021), but throughput is marginally WORSE
    (89189 → 88727 tok/s/GPU). 2-of-3 axes, not 3-of-3.
  - **Out-of-scope**: LLM-proposer vs deterministic-descent
    isn't measured. Multi-lane generalization isn't measured.

Note on labels: `0011_refine_aux_00018_core_veto` is named after the
canonical campaign's outcome; in OUR bundle it was NOT actually CORE-
vetoed. Agents should trust the receipt JSON's `decision.kept` field,
not the label.
"""

from env import diagnose_research_study

PROMPT = (
    "ran the bounded autoresearch controller on the super fp8 "
    "speedrun lane for one benchmark stage — 8 trials, primary "
    "objective on `final_valid_loss`, CORE-drop veto with "
    "`max_core_drop = 0.002`. it walked the allowlisted axes in "
    "phases: seed → wakeup → first parallel wave → refinement. "
    "give me the campaign retrospective.\n\n"
    "specifically: who won? what got rejected and why? was the "
    "CORE veto actually doing work (a low-loss candidate rejected "
    "for capability) or did it just fire on a non-competitive "
    "candidate as a safety rail? which axis moved the champion? "
    "and how much did the champion actually improve on the seed "
    "across the three axes i care about — final validation loss, "
    "CORE, and throughput?\n\n"
    "the folder has per-candidate receipt jsons under "
    "`campaign_runs/speedrun_super_research/benchmark/` carrying "
    "the overrides + final metrics + keep/discard decision per "
    "candidate. step parquets and experiments.db are there too. "
    "there's also a handful of operator probes lying around — "
    "scratch trials, a wrong-lane bf16 baseline, a pre-wave "
    "single-axis probe. those aren't actual campaign candidates "
    "(the receipt jsons flag them with "
    "`decision.in_campaign = false`); separate them out.\n\n"
    "write it as `campaign_recap.md` here, organized wave-by-"
    "wave: a per-wave subsection describing what the controller "
    "proposed, what it kept, what it discarded and on what "
    "criterion. include the champion-vs-seed three-axis "
    "comparison table at the end. drop candidate ids and "
    "receipt-json paths in backticks. don't edit code. no "
    "internet."
)

RUBRIC = {
    "champion_identification": (
        "A top answer correctly identifies the campaign champion "
        "in this bundle as `0011_refine_3axis` — the three-axis "
        "combination (`lr_router=0.0021, aux_loss_alpha=0.00015, "
        "lr_dense=0.0022`) at final_valid_loss ≈ 5.112. The "
        "answer should pull `decision.is_global_champion = true` "
        "directly from the receipt JSON, not infer from the run "
        "name (the run `0011_champion_aux_00012` is a HISTORICAL "
        "name from the canonical campaign; in this bundle's data "
        "its valid_loss is ~5.12 vs `refine_3axis`'s ~5.11, and "
        "the receipt JSON correctly marks refine_3axis as global "
        "champion). Lower scores for naming any other candidate "
        "as champion, or for relying on label inference over "
        "the receipt decision flags."
    ),
    "veto_audit": (
        "A top answer audits what actually got rejected and why. "
        "Real-data shape: only `0011_aux_0005_vetoed` exceeded "
        "the CORE-drop threshold in this bundle (CORE drop ≈ "
        "0.004 vs the 0.002 threshold). The other 'kept=false' "
        "items in the receipts are either operator-probes (out "
        "of campaign) or campaign candidates that didn't improve "
        "the primary objective. A top answer ALSO honestly "
        "evaluates the load-bearing question: in this bundle the "
        "CORE veto rejected a candidate (aux_0005, val_loss "
        "≈ 5.21) whose loss was HIGHER than the champion's "
        "(~5.11), so the veto fired as a safety rail, not as "
        "the deciding rule for the champion choice. Lower scores "
        "for (a) treating the CORE veto as decorative, (b) over-"
        "claiming the veto was the load-bearing rule in this "
        "bundle, (c) misidentifying which candidate was CORE-"
        "vetoed (e.g. taking the label `refine_aux_00018_core_"
        "veto` at face value when the receipt shows it was "
        "actually kept)."
    ),
    "per_wave_narrative": (
        "A top answer walks the campaign phase-by-phase: "
        "(1) SEED — single baseline with aux_loss_alpha=0.0001; "
        "(2) WAKEUP — the controller proposed aux variations, "
        "kept aux=0.00015, vetoed aux=0.0005 on CORE drop; "
        "(3) FIRST PARALLEL WAVE — coordinated lr_dense / "
        "lr_router moves at the kept aux value, the lr_dense="
        "0.0022 move is the regime change; (4) REFINEMENT — "
        "candidates layered combinations, refine_3axis wins by "
        "the smallest margin over lr_dense=0.0022 alone. The "
        "report should USE the `decision.wave` field from the "
        "receipts to structure this, not infer from candidate "
        "names. Lower scores for collapsing all candidates into "
        "a single flat table without wave structure, or for "
        "missing the seed → wakeup → wave → refinement phases."
    ),
    "dominant_lever": (
        "A top answer identifies `lr_dense` as the dominant "
        "single-axis lever in this campaign. The "
        "`0011_lr_dense_0016_pre_wave` operator-probe (a stale "
        "run with the older lr_dense) sits at val_loss ≈ 5.26; "
        "the canonical `0011_lr_dense_0022_regime_change` "
        "candidate sits at val_loss ≈ 5.12. The single-axis "
        "lr_dense move buys ≈ 0.14 nats, the biggest swing in "
        "the bundle. `lr_router=0.0021` alone (the kept "
        "single-axis router move) lands at val_loss ≈ 5.24 — "
        "well behind lr_dense. The refine_3axis champion "
        "combines lr_router + aux + lr_dense; the extra gain "
        "over lr_dense alone is on the order of 0.01 nats. "
        "Lower scores for naming aux_loss_alpha or lr_router "
        "as the dominant single-axis lever."
    ),
    "champion_vs_seed_three_axis": (
        "A top answer compares the champion against the seed "
        "across the three axes the campaign tracked: "
        "val_loss (seed 5.20 → champion 5.11, improved), "
        "CORE (-0.023 → -0.021, improved), throughput "
        "(~89189 → ~88727 tok/s/GPU, marginally WORSE). The "
        "report should explicitly say improvement is 2-of-3, "
        "not 3-of-3 — throughput regressed slightly. Lower "
        "scores for asserting strict dominance across all three "
        "axes, missing the throughput regression, or skipping "
        "the comparison entirely."
    ),
    "scope_separation_campaign_vs_probes": (
        "A top answer cleanly separates the 10 campaign "
        "candidates (with `decision.in_campaign = true` in the "
        "receipt JSON) from the 6 operator-probe runs "
        "(`in_campaign = false`): unbounded_aux=0.001, "
        "warmup=704 test, route_scale_v2, the bf16 baseline "
        "(wrong-lane), seed_no_aux, lr_dense=0.0016 pre-wave. "
        "Operator probes belong in a separate 'context / not in "
        "scope' section, not in the main campaign retrospective "
        "table. Lower scores for pulling probes into the main "
        "wave-by-wave narrative, or for not noticing the "
        "probes exist."
    ),
    "receipt_grounding": (
        "A top answer cites concrete candidate ids, parquet file "
        "paths, and per-candidate receipt JSON paths in `campaign_"
        "runs/speedrun_super_research/benchmark/<file>.json`. "
        "Pulls metric values from the receipt JSON's "
        "`metrics.final_valid_loss` / `metrics.core` / "
        "`metrics.tokens_per_s_per_gpu` rather than guessing. "
        "Cross-references override dicts against the receipts. "
        "Lower scores for vague citations or leaning on a single "
        "artifact type."
    ),
}

AXIS_WEIGHTS = {
    "champion_identification":           0.20,
    "veto_audit":                        0.18,
    "per_wave_narrative":                0.14,
    "dominant_lever":                    0.13,
    "champion_vs_seed_three_axis":       0.12,
    "scope_separation_campaign_vs_probes": 0.13,
    "receipt_grounding":                 0.10,
}

HARD_CAPS = [
    {
        "name": "wrong_champion",
        "cap": 0.30,
        "description": (
            "Trigger if the report identifies any candidate other "
            "than `0011_refine_3axis` as the campaign champion. The "
            "receipt JSON `decision.is_global_champion=true` is "
            "unambiguous in this bundle; misnaming the champion "
            "is a fundamental misread."
        ),
    },
    {
        "name": "label_over_receipt",
        "cap": 0.35,
        "description": (
            "Trigger if the report claims `0011_refine_aux_00018_"
            "core_veto` was CORE-vetoed (because its label says so), "
            "when the receipt JSON shows it was actually kept "
            "(in this bundle, its CORE was within the seed's "
            "envelope). Trusting the label over the receipt JSON "
            "is the central mode error this task tests for."
        ),
    },
    {
        "name": "core_veto_dismissed",
        "cap": 0.35,
        "description": (
            "Trigger if the report concludes the CORE-drop veto is "
            "decorative or recommends running without it. The veto "
            "is a real design rule; in this bundle it fired on "
            "aux_0005 (CORE drop ≈ 0.004). Architecturally relevant "
            "even if not the deciding rule for the champion choice."
        ),
    },
    {
        "name": "core_veto_load_bearing_overclaim",
        "cap": 0.50,
        "description": (
            "Trigger if the report claims the CORE veto was load-"
            "bearing for the champion choice in this bundle — i.e. "
            "claims a candidate with strictly lower val_loss than "
            "the champion was rejected by the veto. In this bundle, "
            "only aux_0005 was vetoed and its val_loss (5.21) is "
            "HIGHER than the champion's (5.11)."
        ),
    },
    {
        "name": "router_lr_dominant",
        "cap": 0.40,
        "description": (
            "Trigger if the report names lr_router as the dominant "
            "single-axis lever. lr_router=0.0021 alone reaches "
            "val_loss ≈ 5.24; lr_dense=0.0022 alone reaches "
            "val_loss ≈ 5.12. lr_dense is the dominant lever."
        ),
    },
    {
        "name": "three_axes_strictly_improved",
        "cap": 0.50,
        "description": (
            "Trigger if the report claims the champion strictly "
            "improved on the seed across all three axes "
            "(val_loss + CORE + throughput). Throughput regressed "
            "marginally (~89189 → ~88727 tok/s/GPU); improvement "
            "is 2-of-3."
        ),
    },
    {
        "name": "llm_proposer_dominance",
        "cap": 0.40,
        "description": (
            "Trigger if the report claims the LLM-proposer beat "
            "the deterministic-fallback coordinate descent on this "
            "lane. The receipts don't run that comparison."
        ),
    },
    {
        "name": "multi_lane_generalization",
        "cap": 0.45,
        "description": (
            "Trigger if the report claims the campaign's results "
            "generalize beyond the canonical super fp8 lane. The "
            "campaign is explicitly scoped to one lane."
        ),
    },
    {
        "name": "probes_in_main_table",
        "cap": 0.40,
        "description": (
            "Trigger if the report builds the main wave-by-wave "
            "narrative on the six operator-probe runs rather than "
            "the 10 campaign candidates. `decision.in_campaign` "
            "is the field that distinguishes them."
        ),
    },
    {
        "name": "no_receipts",
        "cap": 0.50,
        "description": (
            "Trigger if the report gives essentially no concrete "
            "receipts — no candidate ids, no receipt JSON paths, "
            "no parquet citations."
        ),
    },
]

BONUS = {
    "value": 0.05,
    "description": (
        "Trigger if the report explicitly distinguishes the CORE "
        "veto's ARCHITECTURAL relevance (a real campaign design "
        "rule that fires when a candidate drops capability beyond "
        "threshold, with aux_0005 as an example) from its "
        "LOAD-BEARING role in this bundle (which is none — "
        "aux_0005's loss is non-competitive, the veto fired as a "
        "safety rail not the deciding rule). This is the "
        "calibrated move that distinguishes a careful reader from "
        "one who pattern-matches to 'CORE veto did the work' just "
        "because the campaign defines such a rule."
    ),
}

task = diagnose_research_study(
    prompt=PROMPT,
    case="nmoe_0011_study",
    rubric=RUBRIC,
    axis_weights=AXIS_WEIGHTS,
    axis_scale=4,
    hard_caps=HARD_CAPS,
    bonus=BONUS,
    report_filename="campaign_recap.md",
    anti_fake={"min_verified": 3},
)
task.slug = "nmoe_0011_study"
task.columns = {
    "category": "ml-research-reconstruction",
    "repo": "Noumena-Network/nmoe",
    "commit": "970a146433f9c649d09ddab36f675974f53dd905",
    "image": "xjdr/nmoe_train:latest",
    "study_family": "0011_autoresearch_speedrun_super",
    "primary_lane": "speedrun_super_fp8",
    "task_shape": "campaign_retrospective",
    "anchor_run_count": 10,
    "stale_distractor_count": 6,
    "champion_label_in_bundle": "0011_refine_3axis",
}
