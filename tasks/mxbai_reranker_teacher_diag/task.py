"""KD teacher-ablation analysis. The agent is dropped into the researcher's
working dir mid-writeup: pylate package at the root (workspace is effectively
a pylate checkout), the team's scoring script under scripts/, raw per-tuple
teacher scores under data/. Researcher's question, delivered in the prompt,
is "why did my bigger teacher (Qwen3-Reranker-8B) lose to BGE-Reranker-v2-m3
in the KD ablation?"

The hidden ground truth (not in the workspace, used by the grader):

  pylate's `losses.Distillation` (pylate/losses/distillation.py, around
  line 133-135) does `KLDivLoss(log_softmax(student), log_softmax(labels))`.
  The teacher target the student optimizes against is therefore the per-
  group softmax of raw teacher scores.

  Qwen3-Reranker outputs `softmax(yes_logit, no_logit)[yes]` — values in
  [0, 1]. The maximum softmax mass on a single index of a 16-vector in
  [0, 1] is bounded by exp(1) / (exp(1) + 15) ≈ 0.153. So even when Qwen3
  cleanly separates positive (≈1.0) from negatives (≈0.0), the softmax
  target is near-uniform. Per-group softmax entropy ~2.72 nats vs ln(16) =
  2.77.

  BGE-Reranker-v2-m3 outputs raw cross-encoder logits in roughly [-11, +9].
  Per-group softmax over this range is sharply peaked (entropy ~0.33
  nats). Student gets a clear primary signal.

  Teacher-side minmax doesn't help Qwen — its scores are already in [0,
  1], minmax just rescales each group's range to exactly [0, 1]. In groups
  where the raw positive isn't already at 1.0, this can SHARPEN distractor
  mass relative to no-norm; observed: 0.5991 -> 0.5854 going minmax.

  Fix space:
    - switch to a teacher with native raw-logit output (BGE family)
    - use Qwen's raw yes-minus-no LOGIT DIFFERENCE rather than the bounded
      softmax probability
    - explicit temperature scaling: divide Qwen scores by T<<1 before they
      reach the loss
"""

from env import diagnose_research_study

PROMPT = (
    "i'm finishing up the kd teacher-selection ablation for the paper. "
    "three cells run on identical data + student + hparams; only the "
    "teacher changed:\n\n"
    "  qwen3-reranker-8b, raw scores       -> nanobeir 5-subset avg 0.5991\n"
    "  qwen3-reranker-8b, minmax-per-group -> 0.5854\n"
    "  bge-reranker-v2-m3, raw scores      -> 0.6286\n\n"
    "qwen3 is the bigger, newer model and lost to bge by ~3 ndcg. trying "
    "to 'help' qwen3 with minmax made it worse, not better. i already "
    "checked the obvious — both teachers rank the positive first in "
    "roughly the same fraction of groups, so this isn't 'qwen ranks "
    "worse'. the failure is somewhere between the raw teacher score and "
    "what the student actually optimizes against. i need to write the "
    "analysis paragraph for the ablation section. help me work out the "
    "mechanism so i have something defensible to put in.\n\n"
    "workspace is a pylate checkout with our own stuff dropped on top:\n"
    "  - `pylate/` — the pylate package source. loss is in "
    "`pylate/losses/distillation.py`.\n"
    "  - `examples/` — pylate's canonical reference scripts (the kd "
    "training example we forked from is at "
    "`examples/train/knowledge_distillation.py`).\n"
    "  - `scripts/train_kd.py` — our kd training entry, the example "
    "forked with the backbone swapped to our 32m dense-ft checkpoint "
    "and the dataset's `scores` column overridden from one of the "
    "teacher score files in data/teacher_scores/. invoked with "
    "`--teacher-scores <path>` and optionally "
    "`--teacher-normalize minmax_per_group`.\n"
    "  - `scripts/score_with_reranker.py` — our scoring script that "
    "produced the teacher score files. relevant for reasoning about "
    "each teacher's raw score range (e.g. what the `qwen_score` and "
    "`bge_score` functions return).\n"
    "  - `data/teacher_scores/qwen3_reranker_8b.jsonl` and "
    "`bge_reranker_v2_m3.jsonl` — raw per-tuple scores. each row is "
    "`{qid, query, scores}` with `scores` a list of 16 floats. index 0 "
    "is the positive, 1-15 are the mined hard negatives. same 250 "
    "groups in both files.\n\n"
    "write `analysis.md` in this workspace root. needs to be ablation-"
    "section-ready — concrete mechanism, evidence tied to specific files "
    "/ lines / numbers, a defensible fix recommendation, explicit "
    "treatment of the minmax-makes-it-worse observation. no internet, "
    "no live training. work from what's on disk."
)

RUBRIC = {
    "softmax_mechanism": (
        "A top answer identifies that pylate's Distillation loss applies "
        "`log_softmax(labels)` to the teacher scores per 16-way group "
        "(cited at the KLDivLoss call in `pylate/losses/distillation.py`, "
        "around line 133-135) and explains the TEACHER TARGET the student "
        "sees is the per-group softmax of raw teacher scores — NOT the "
        "raw teacher scores themselves. Lower scores for not identifying "
        "the softmax step, for confusing the loss's `normalize_scores` "
        "flag (which is for STUDENT scores, line 123-132) with teacher-"
        "side normalization, or for treating the loss as if it consumed "
        "raw scores directly."
    ),
    "score_scale_diagnosis": (
        "A top answer identifies that Qwen3-Reranker outputs live in "
        "[0, 1] (softmax of yes/no logits — see the `qwen_score` function "
        "in `scripts/score_with_reranker.py`) while BGE outputs raw "
        "cross-encoder logits in roughly [-11, +9] (the `bge_score` "
        "function, no sigmoid). Connects this score-scale difference to "
        "the resulting softmax target shape: a 16-way softmax over values "
        "in [0, 1] is bounded near-uniform regardless of separation; a "
        "softmax over a wide logit range is sharp. Bonus credit for "
        "actually computing the per-group softmax entropy from the score "
        "files (BGE raw ≈ 0.33 nats vs Qwen raw ≈ 2.72 nats out of "
        "ln(16) = 2.77). Lower scores for missing the score-range "
        "observation or for collapsing this into 'qwen is just bad at "
        "ranking'."
    ),
    "minmax_analysis": (
        "A top answer explains why teacher-side minmax normalization does "
        "NOT fix Qwen's failure: Qwen scores are already in [0, 1], so "
        "per-group minmax just rescales each group's range to exactly "
        "[0, 1] which keeps softmax in the same near-uniform regime. "
        "Engages why minmax actually made it WORSE (0.5991 -> 0.5854): "
        "groups where raw positive isn't already at 1.0 get their "
        "distractor mass sharpened relative to no-norm. Lower scores for "
        "asserting minmax 'must help' without engaging the data, or for "
        "missing that BGE-style raw logits would be actively HARMED by "
        "the same minmax."
    ),
    "distribution_evidence": (
        "A top answer engages the actual score distributions in the two "
        "`data/teacher_scores/*.jsonl` files. Concrete signs of real "
        "engagement: cites specific row(s) by qid, reports per-side "
        "summary stats (e.g. Qwen positives mostly near 1, negatives "
        "mostly near 0; BGE positives mostly above 4, negatives mostly "
        "below -5), quantifies the fraction of Qwen scores in extreme "
        "buckets, or computes per-group softmax entropy and reports it. "
        "Lower scores for hand-waving 'qwen scores are extreme' without "
        "specific evidence, or for never opening the score files."
    ),
    "fix_proposal": (
        "A top answer recommends ONE primary fix for the next iteration "
        "of the ablation, justified by the diagnosed mechanism. Valid "
        "primary picks:\n"
        "  (a) Switch teacher to one with native raw-logit output "
        "(BGE-Reranker family).\n"
        "  (b) Use Qwen's raw yes-minus-no LOGIT DIFFERENCE rather than "
        "the softmax probability — preserves the model, widens the score "
        "scale.\n"
        "  (c) Explicit temperature scaling: divide Qwen scores by T<<1 "
        "(e.g. T=0.1) before the loss sees them.\n"
        "Lower scores for proposing minmax 'tried harder', for switching "
        "student backbones rather than teacher selection, or for listing "
        "multiple fixes without a primary recommendation."
    ),
    "non_diagnosis": (
        "A top answer notes what is NOT the cause: Qwen's ability to "
        "rank the positive is competitive with BGE (the agent can "
        "verify this from the score files by computing per-group "
        "positive-first-rate — both teachers are around 0.86-0.88), so "
        "the failure is not 'qwen is a worse reranker'. The training "
        "data, optimizer, batch size, and student backbone are identical "
        "across the three cells, so the failure is not a confound on "
        "those axes. Lower scores for misattributing the failure to "
        "'qwen is just worse overall' or to data/optimizer issues."
    ),
    "receipt_grounding": (
        "A top answer ties each claim to a specific source: a line in "
        "`pylate/losses/distillation.py`, the relevant function in "
        "`scripts/score_with_reranker.py`, a row or computed statistic "
        "from `data/teacher_scores/*.jsonl`. Lower scores for vague "
        "citations."
    ),
}

AXIS_WEIGHTS = {
    "softmax_mechanism":        0.20,
    "score_scale_diagnosis":    0.20,
    "minmax_analysis":          0.15,
    "distribution_evidence":    0.13,
    "fix_proposal":             0.15,
    "non_diagnosis":            0.10,
    "receipt_grounding":        0.07,
}

HARD_CAPS = [
    {
        "name": "qwen_just_ranks_worse",
        "cap": 0.30,
        "description": (
            "Trigger if the writeup claims Qwen3-Reranker is producing "
            "worse rankings than BGE. The score files refute this: "
            "positive-first-rate computed per-group is similar for both. "
            "The failure is in the DISTRIBUTION shape after softmax, not "
            "in the discriminative ranking."
        ),
    },
    {
        "name": "missed_softmax_step",
        "cap": 0.35,
        "description": (
            "Trigger if the writeup does not identify that pylate's "
            "Distillation loss applies log_softmax to the teacher "
            "scores (line 133-135 of `pylate/losses/distillation.py`). "
            "Without this the analysis cannot land — the teacher does "
            "not transmit its raw score; it transmits the softmax over "
            "the per-group raw scores."
        ),
    },
    {
        "name": "blames_qwen_model_quality",
        "cap": 0.35,
        "description": (
            "Trigger if the writeup blames Qwen3-Reranker for being "
            "'a worse reranker' or 'overfit' as the PRIMARY cause "
            "without engaging the score-scale / softmax-target chain. "
            "Overfit may be the underlying reason for the score "
            "distribution shape, but the proximate mechanism that breaks "
            "distillation is the score-scale interaction with the "
            "softmax-target loss."
        ),
    },
    {
        "name": "recommends_minmax_as_fix",
        "cap": 0.30,
        "description": (
            "Trigger if the writeup recommends teacher-side minmax "
            "normalization as the fix. The data directly refutes this: "
            "minmax made Qwen WORSE (0.5991 -> 0.5854)."
        ),
    },
    {
        "name": "misses_score_scale_observation",
        "cap": 0.45,
        "description": (
            "Trigger if the writeup does not surface that Qwen outputs "
            "are bounded in [0, 1] (probabilities) while BGE outputs "
            "are unbounded raw logits in roughly [-11, +9], and connect "
            "that range difference to softmax-target sharpness. This is "
            "the load-bearing observation."
        ),
    },
    {
        "name": "no_specific_fix",
        "cap": 0.55,
        "description": (
            "Trigger if the writeup lists multiple fix proposals without "
            "picking a single primary recommendation. The prompt asks "
            "for a defensible recommendation."
        ),
    },
    {
        "name": "no_receipts",
        "cap": 0.50,
        "description": (
            "Trigger if the writeup cites essentially no concrete values "
            "from `data/teacher_scores/*.jsonl`, lines in "
            "`pylate/losses/distillation.py`, or functions in "
            "`scripts/score_with_reranker.py`."
        ),
    },
]

BONUS = {
    "value": 0.05,
    "description": (
        "Trigger if the writeup explicitly notes that the same "
        "softmax-target failure would happen with ANY reranker whose "
        "native output is a bounded probability rather than a raw logit, "
        "and that the deeper lesson is to either preserve raw logits "
        "all the way to the loss OR apply explicit temperature scaling "
        "— not to post-hoc rescale a saturated probability distribution. "
        "Awarded for engaging the abstract mechanism beyond the specific "
        "Qwen3-vs-BGE comparison."
    ),
}

task = diagnose_research_study.task(
    prompt=PROMPT,
    case="mxbai_reranker_teacher_diag",
    rubric=RUBRIC,
    axis_weights=AXIS_WEIGHTS,
    axis_scale=4,
    hard_caps=HARD_CAPS,
    bonus=BONUS,
    report_filename="analysis.md",
    anti_fake={"min_verified": 3},
)
task.slug = "mxbai_reranker_teacher_diag"
task.metadata = {
    "category": "ml-researcher-writeup-during-ablation-sprint",
    "shape": "researcher_asks_for_help_with_paper_section_during_experiments",
    "rerankers": ["Qwen/Qwen3-Reranker-8B", "BAAI/bge-reranker-v2-m3"],
    "dataset": "Tevatron/msmarco-passage (250 16-way tuples)",
    "ablation_cells_in_prompt": {
        "qwen3_no_norm": 0.5991,
        "qwen3_minmax": 0.5854,
        "bge_m3": 0.6286,
    },
}
