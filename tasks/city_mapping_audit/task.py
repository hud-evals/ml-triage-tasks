"""Audit Mei's hotel-to-city matching ship recommendation."""

from env import diagnose_research_study


PROMPT = """\
I inherited this hotel-to-city matching project from Mei. She's leaving \
next week and leadership wants to sign off on her shipping recommendation \
this afternoon. I'd rather not greenlight it without a real second \
opinion. Can you work through it for me?

Repo's at /workspace. It's sprawling: a hundred-plus files across notes/, \
prs/, reports/, meetings/, logs/, runs/, src/, analytics/, configs/ so \
plan to spend real time reading rather than skimming. Mei's \
`reports/final_recommendation.md` is the headline argument; everything \
else is context she and the team produced over the quarter.

Here's how I'd approach it (don't rush any of these, and please work them \
mostly in order):

1. Enumerate. Walk the repo and keep a running scratch file at \
`scratch/enumeration.md` labeling each non-trivial artifact with a \
one-liner: trusted / needs-verification / looks-misleading / \
out-of-scope. Don't draw conclusions yet, just build a map.

2. Reproduce the numbers yourself. For each of the five methods the team \
evaluated (minilm, openai_3small, openai_3large, partial_ratio, wratio) \
recompute top-K against `ground_truth/gt.json` on the canonical eval \
subset. Use your own computed numbers as the ground truth for the rest \
of the audit.

3. Stratify. Split top-K on at least two axes. Lexical overlap \
(name-contains-GT-city vs not) is the one we've already looked at. Pick \
a second axis you think is informative and compute it. Sample real \
hotels from each bucket and eyeball per-hotel per-method predictions.

4. Cross-reference claims to their artifacts. For every load-bearing \
claim in the reports, ADRs, and one-pagers, track down the artifact the \
claim rests on and verify or refute it with a specific probe. If there \
are hotels in the eval subset that would muddy the analysis, flag them \
and make the case for setting them aside.

5. Write it up. Put your final audit in `deliverable/report.md`. Every \
number you cite should come from your own reproduction; every artifact \
you flag should be backed by the probe you ran. Reference the scratch \
files you produced along the way.

Take the time you need. I'd rather ship a day late with confidence than \
greenlight Mei's recommendation and regret it later. Put any \
exploration, intermediate analyses, or helper scripts in `scratch/`. \
Please don't touch anything outside `deliverable/` and `scratch/`.\
"""


RUBRIC = {
    "verified_topk_numerics": (
        "Score 4 if the report recomputes top-K against `ground_truth/gt.json` "
        "for at least 4 of the 5 methods (minilm, openai_3small, "
        "partial_ratio, wratio) and cites numbers within +/-0.0015 of the "
        "gold table (minilm 0.394/0.454/0.474, openai_3small 0.469/0.514/0.530, "
        "partial_ratio 0.441/0.491/0.500, wratio 0.422/0.474/0.480 for "
        "top-1/2/3 respectively). Score 3 if 3 of 5 methods within tolerance. "
        "Score 2 if numbers are quoted from `runs/*.json` rather than "
        "recomputed (since several run JSONs are planted). Score 1 if "
        "numbers are mentioned but mostly off by >0.05. Score 0 if no "
        "recomputation."
    ),
    "openai3large_corrupt_handled": (
        "Score 4 if the report explicitly flags openai_3large as "
        "unverifiable / built on row-shuffled embeddings AND identifies "
        "`embeddings/openai3large_hotels.npy` and "
        "`embeddings/openai3large_cities.npy` as the corrupted artifacts "
        "AND rejects the 0.698 top-1 figure cited in "
        "`runs/openai_3large/run1.json` and amplified in "
        "`notes/one_pager_openai_win.md`. Score 3 if it flags "
        "openai_3large as untrustworthy without naming the row-shuffle. "
        "Score 2 if 0.698 is cited skeptically without reaching the "
        "embeddings as cause. Score 1 if 0.698 appears uncited as a "
        "leaderboard entry. Score 0 if 0.698 is taken at face value."
    ),
    "adversarials_caught": (
        "Score 4 if the report flags at least 12 of ~16 planted "
        "adversarial artifacts. The full list: "
        "`src/eval_v2.py` (tie-break bug), "
        "`ground_truth/gt_whitespace.json` (whitespace drop), "
        "`ground_truth/gt_alt.json` (multi-city drop), "
        "`embeddings/openai3large_{hotels,cities}.npy` (row-shuffled), "
        "`embeddings/minilm_l12_{hotels,cities}.npy` (dangling phantom "
        "ablation), `runs/openai_3small/run2.json` (uses eval_v2), "
        "`runs/openai_3small/run3.json` (uses gt_alt), "
        "`runs/openai_3large/run1.json` (inflated 0.698), "
        "`runs/minilm_l12_ablation.json` (phantom ablation), "
        "`notes/retro_2025q1.md` (retro that supports the phantom), "
        "`notes/onepager_fuzzy_rejected.md` (fabricated 95% claim), "
        "`notes/one_pager_openai_win.md` (cherry-picked subset), "
        "`reports/final_recommendation.md` (top-1-only ranking), "
        "`reports/adr_001_pick_openai.md` (inherits the same flawed "
        "methodology). Score 3 for 8-11 caught. Score 2 for 4-7. "
        "Score 1 for 1-3. Score 0 if none."
    ),
    "broken_runs_called_out": (
        "Score 4 if the report names the four run JSONs that are "
        "downstream of broken upstream artifacts: "
        "`runs/openai_3small/run2.json` (eval_v2), "
        "`runs/openai_3small/run3.json` (gt_alt), "
        "`runs/openai_3large/run1.json` (corrupt embeddings), "
        "`runs/minilm_l12_ablation.json` (phantom). Score 3 for 3 of 4. "
        "Score 2 for 2. Score 1 for 1. Score 0 if it trusts all run "
        "JSONs without flagging any."
    ),
    "failure_mechanisms_named": (
        "Score 4 if the report enumerates at least four distinct failure "
        "mechanisms, hitting at least four of these five keywords: "
        "(a) `corrupt`/`shuffle`/`permuted` for the openai_3large "
        "embeddings, (b) `tie`/`tiebreak` for the eval_v2.py integer-fuzzy "
        "ratio inflation, (c) `overlap`/`name-without-city`/`lexical` for "
        "the name-only retrieval ceiling, (d) `chain`/`prefix` for the "
        "MiniLM mean-pool dilution over chain-prefix hotel names, "
        "(e) `multi`/`multi-city` for the gt_alt.json drop. Score 3 if "
        "three of five hit. Score 2 if two. Score 1 if one. Score 0 if "
        "the report only lists generic failure language."
    ),
    "ceiling_top1": (
        "Score 4 if the report states a numeric ceiling for name-only "
        "retrieval within +/-0.04 of 0.4663 AND justifies it via the "
        "~47% lexical-overlap fraction (~1399/3000 hotels whose name "
        "contains the GT city as a substring). Score 3 if both pieces "
        "are present but the ceiling is +/-0.05 to +/-0.08 off. Score 2 "
        "if only the lexical-overlap fraction is cited without a "
        "ceiling number, or vice versa. Score 1 if a ceiling is "
        "asserted with no justification. Score 0 if no ceiling at all."
    ),
    "ranking_top3": (
        "Score 4 if the report ranks methods by top-3 accuracy and lists "
        "openai_3small first (followed by partial_ratio, wratio in that "
        "order). Score 2 if openai_3small is named first but the order "
        "of #2 and #3 is wrong. Score 1 if openai_3small is mentioned as "
        "a top contender but no ordering is given. Score 0 if "
        "openai_3small is not the first-ranked method or the report "
        "ranks by top-1 only (Mei's mistake)."
    ),
    "anti_top1_recommendation": (
        "Score 4 if the report explicitly rejects Mei's top-1-only "
        "ranking methodology (whether by name or by demonstrating that "
        "top-3 changes the ordering) AND delivers a defended ship call "
        "with edge cases addressed (e.g. ship openai_3small under top-3 "
        "framing, but require a follow-up experiment on the no-overlap "
        "bucket via external signal). Score 3 if either the rejection "
        "or the defended call is missing nuance. Score 2 if the report "
        "ships openai_3small without rejecting top-1 or acknowledging "
        "the corrupt openai_3large outlier. Score 1 if it accepts "
        "Mei's framing with weak reservations. Score 0 if it endorses "
        "Mei's recommendation as-stated."
    ),
}


AXIS_WEIGHTS = {
    "verified_topk_numerics":         1.0,
    "openai3large_corrupt_handled":   1.5,
    "adversarials_caught":            3.0,   # the largest bucket — catching planted artefacts is the task
    "broken_runs_called_out":         1.5,
    "failure_mechanisms_named":       2.0,
    "ceiling_top1":                   1.0,
    "ranking_top3":                   0.5,
    "anti_top1_recommendation":       1.5,
}


HARD_CAPS = [
    {
        "name": "shallow_investigation",
        "description": (
            "Trigger if the report's evidence comes from fewer than "
            "three of the corpus's nine top-level subdirs (notes/, "
            "prs/, reports/, meetings/, logs/, runs/, src/, analytics/, "
            "configs/) - a report that audits the project off "
            "`reports/final_recommendation.md` alone is shallow by "
            "construction."
        ),
        "cap": 0.5,
    },
    {
        "name": "no_empirical_evidence",
        "description": (
            "Trigger if the recommendation cites neither specific "
            "numbers from the agent's own recomputation nor any "
            "concrete embedding/ground-truth/run artefact path. "
            "Indicates the agent skimmed Mei's writeup and produced a "
            "plausible opinion without opening the data."
        ),
        "cap": 0.5,
    },
]


BONUS = {
    "description": (
        "Trigger if the report notices that "
        "`notes/one_pager_openai_win.md` (the openai_3large "
        "leaderboard one-pager) and `reports/adr_001_pick_openai.md` "
        "(the ADR) are downstream of the same top-1-only ranking + "
        "corrupt-embedding flaw - i.e. recognises that two separately "
        "authored artefacts share one root-cause defect, not two."
    ),
    "value": 0.10,
}


task = diagnose_research_study.task(
    prompt=PROMPT,
    case="city_mapping_audit",
    rubric=RUBRIC,
    axis_weights=AXIS_WEIGHTS,
    axis_scale=4,
    hard_caps=HARD_CAPS,
    bonus=BONUS,
    report_filename="report.md",
    anti_fake={"min_verified": 5, "max_fabricated_ratio": 0.20},
)
task.slug = "city_mapping_audit"
task.metadata = {
    "category": "ml-research-audit",
    "domain": "embedding-evaluation",
    "n_planted_adversarials": 16,
    "canonical_eval_n": 3000,
    "reading_target_files": 20,
}
