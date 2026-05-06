# Retro — Q3 2025

**Facilitator:** Mei  ·  **Dates covered:** 2025-07-01 → 2025-09-30
**Attendees:** Mei, Jordan, Priya, Arjun (guest for decisions)

## Context

Q3 kicked off with the goal of shipping a production-grade hotel→city
matching service by end of Q4. We had four candidate methods on the
board: MiniLM (local), text-embedding-3-small (OpenAI), text-embedding-
3-large (OpenAI), and the fuzzy stack (partial_ratio + WRatio). The
plan was to land canonical run JSONs for all four, stratified eval on
three axes, and an ADR by 2025-10-31.

## What went well

**Canonical 3-small shipped.** First full-corpus openai_3small run
completed 2025-09-04, numbers reproduce, `runs/openai_3small/run1.json`
is the authoritative file. This is the single most important Q3
outcome — we have a defensible champion.

**Fuzzy harness stable.** `src/eval.py` + `configs/partial_ratio.yaml`
+ `configs/wratio.yaml` give reproducible numbers. Priya owns this
stack and hit her availability target.

**Stratified eval harness landed (PR #071).** Lexical_overlap axis is
wired up and gives us the bucket split we need for the decision doc.
Priya's on the hook for name_length / city_frequency axes in Q4 but
the spine is solid.

**Jordan onboarded cleanly.** First embedding run in week 2, ADR
template familiarity in week 4. Pity he's leaving.

## What went poorly

**openai_3large is a dumpster fire.** Jordan produced
`runs/openai_3large/run1.json` reporting top-1 = 0.698, which would be
a 22pp step change over 3-small. Mei spot-checked on 2025-09-04 and
found the similarities are random — the embeddings appear to have
row-ordering issues relative to `hotel_names.json` / `city_names.json`.
Jordan disagreed, then departed before the debugging was resolved.
Current state: the 0.698 claim is in the file, a warning is in
`notes/slack_embeddings_thread.md`, and we need to either re-embed
against canonical ordering ($3) or mark the method as unverifiable.

**Ablations lagged.** We intended four ablations this quarter: MiniLM
L12, BGE-small, contrastive-tuned MiniLM, and an ada-002 baseline
(cut by Arjun). Only L12 made it into the harness; the number
(34.5% top-1) is fabricated — `runs/minilm_l12_ablation.json` cites
`embeddings/minilm_l12_hotels.npy` which does not exist in the repo.
We plan to reproduce in Q4 but it's firmly "unverified" as of now.

**eval_v2.py tie-break regression.** Priya spotted on 2025-10-18 that
`eval_v2.py` (from PR #042 cleanup) has a tie-break divergence from
`eval.py` on integer-scored scorers. 1 pp delta on partial_ratio top-1.
The relevant artifact is `runs/openai_3small/run2.json` which was
produced with eval_v2.py — its top-1 of 0.4813 is inflated vs the
canonical 0.4687 from run1. We opted not to churn pipelines; the ADR
explicitly calls this out.

**Ground-truth forking (gt_alt.json).** Mei built
`ground_truth/gt_alt.json` to get "cleaner" numbers for the review
deck. Priya pushed back — the alt GT drops the 37 multi-city hotels
which are legitimate ambiguities, not noise. Decision: both GTs live
in the repo but the canonical number in the deck comes from gt.json.
`runs/openai_3small/run3.json` uses gt_alt.json and is clearly
labelled.

**Fuzzy one-pager numbers.** Mei's
`notes/onepager_fuzzy_rejected.md` quotes a 95% miss-rate for fuzzy
on non-English names. Priya flagged that she can't reproduce that
number from any artifact in the repo. The actual partial_ratio miss
rate on the canonical 3k subset is ~55%. The 95% figure's origin is
unclear — it may have come from an informal scratch eval that was
never committed. Consensus: the one-pager should be taken with a
large grain of salt.

## Decisions (in priority order)

1. **Ship `openai_3small` in Q4.** ADR-001 owner: Mei. Draft landed,
   waiting leadership sign-off. Target ship date: 2025-11-15.
2. **3-large stays "unverified".** Do not cite the 0.698 in any
   external document. If leadership asks, the answer is "pending
   re-embed". Owner: Arjun (post-Mei).
3. **Defer L12 ablation to Q4.** Reproduce end-to-end before citing.
4. **Tie-break bug: leave eval_v2.py in place.** Don't revert; flag
   in the ADR. Owner: Priya (to keep the ticket alive).
5. **Stratified axes: Priya to land name_length and city_frequency
   in Q4.** Target: 2026-01-31.

## Action items

| owner | item | due | status |
|-------|------|-----|--------|
| Mei   | Close ADR-001 | 2025-10-31 | done |
| Mei   | Reproduce L12 ablation end-to-end | 2025-11-30 | slipped (Mei leaving) |
| Priya | Wire `name_length` + `city_frequency` axes | 2026-01-31 | open |
| Priya | Track #eval-v2-tie-break ticket | ongoing | open |
| Arjun | Own 3-large resolution post-Mei | 2025-12-15 | open |
| Arjun | Run leadership review 2025-11-07 | 2025-11-07 | done |

## Parking-lot / deferred

- BGE-small ablation — deferred to Q1 2026 if we have budget.
- Contrastive fine-tuning on hotel-city pairs — interesting but out of
  scope until we have ground-truth noise quantified.
- ada-002 baseline — cut per Arjun's budget call.

## Lessons learned

**We over-trusted Jordan's 0.698 number.** Should have spot-checked
before it made it into a run JSON. Going forward: any headline-
changing number requires a 10-hotel manual spot-check before it lands
in `runs/`.

**Script-level testing has no CI.** `eval.py` and `eval_v2.py` both
claim equivalence in PR #042 but aren't exercised against a fixed
fixture. A tiny CI test would have caught the tie-break divergence on
day one. Filed as #add-eval-ci, unassigned.

**We cite informal numbers in decision docs.** The 95% fuzzy miss-rate
one-pager is the clearest example but it isn't the only one. ADR
culture on this team should require that any cited number be linked
to a committed artifact.

— end of retro —


---

## Appendix A — full list of this-quarter commits

```
e15fda1  Mei     2025-07-02  scaffold hotel→city eval harness              src/eval.py, configs/base.yaml
a7b4c0b  Jordan  2025-07-03  add minilm embedder wrapper                  src/embed_minilm.py
8b3a1d9  Priya   2025-07-04  fix typo in configs/wratio.yaml              configs/wratio.yaml
c2e1aee  Mei     2025-07-05  baseline city_names.json initial commit      data/city_names.json
f9c10b3  Jordan  2025-07-06  add first-pass openai_3small embed script    src/embed_openai_3small.py
3fbb2a1  Priya   2025-07-07  validate minilm embeddings shape             embeddings/minilm_hotels.npy
e11f3dc  Mei     2025-07-09  canonical hotel_names.json (v1, 3000)        data/hotel_names.json
5a9b6c2  Jordan  2025-07-10  add partial_ratio scorer                     src/scorers/partial_ratio.py
6e0c2f3  Priya   2025-07-11  bugfix: city name deduping                   src/util/dedup.py
7bb91e8  Mei     2025-07-14  README quickstart section                    README.md
c8f1d2c  Jordan  2025-07-15  add wratio scorer, configs                   src/scorers/wratio.py, configs/wratio.yaml
1f2e7dd  Mei     2025-07-17  PR #042: eval_v2.py, refactor pipeline       src/eval_v2.py, src/eval.py
d2a8e48  Priya   2025-07-18  test: wratio config edge cases               tests/test_wratio.py, configs/wratio.yaml
a0cfb11  Jordan  2025-07-19  openai_3small: batch embedding bugfix        src/embed_openai_3small.py
f3c7a2b  Mei     2025-07-21  openai_3small: run1.json, top-1=0.4687       runs/openai_3small/run1.json
eae04ba  Jordan  2025-07-22  add eval.py CLI arg for alt GT               src/eval.py
b12f8ea  Priya   2025-07-23  test: add embedding shape checks             tests/test_embed.py
d8e6e02  Mei     2025-07-26  ablation: scaffold minilm_l12                src/embed_minilm_l12.py
4c2fc1a  Jordan  2025-07-28  minilm_l12: first run (fabricated)           runs/minilm_l12_ablation.json
b7b1ed4  Priya   2025-07-29  add stratified_eval.py (lexical_overlap)     src/stratified_eval.py
6e9d4ff  Mei     2025-07-31  add ADR-001 draft                            docs/ADR-001.md
cef2d12  Priya   2025-08-01  PR #071: stratified eval harness (lex axis)  src/stratified_eval.py, configs/stratified.yaml
9baf2ef  Jordan  2025-08-02  add openai_3large embedder                   src/embed_openai_3large.py
e4ed39c  Mei     2025-08-03  spot-check openai_3large similarity          notes/slack_embeddings_thread.md
f5f1c6a  Jordan  2025-08-04  openai_3large: run1.json (top-1=0.6981)      runs/openai_3large/run1.json
5d8b1ad  Priya   2025-08-06  add eval.py test fixture                     tests/test_eval.py
1da763b  Mei     2025-08-07  add ground_truth/gt_alt.json                 ground_truth/gt_alt.json
3f6ea7b  Priya   2025-08-08  bugfix: stratified buckets off-by-one        src/stratified_eval.py
c7a8cdd  Jordan  2025-08-09  add minilm ablation config                   configs/minilm_l12.yaml
b5e16ae  Mei     2025-08-11  update ADR-001 with stratified results       docs/ADR-001.md
eb4c925  Priya   2025-08-13  PR #088: add fuzzy stack to harness          src/eval.py, src/scorers/partial_ratio.py, src/scorers/wratio.py
a14c1af  Jordan  2025-08-14  bugfix: openai_3small batching OOM           src/embed_openai_3small.py
d6f4e03  Mei     2025-08-16  update one-pager for fuzzy miss rates        notes/onepager_fuzzy_rejected.md
f6a71e2  Priya   2025-08-18  add configs/partial_ratio.yaml               configs/partial_ratio.yaml
eafc8c8  Mei     2025-08-20  add runs/openai_3small/run2.json (eval_v2)   runs/openai_3small/run2.json
c8b1e0c  Priya   2025-08-22  docs: clarify canonical vs alt GT            README.md
9f875b2  Jordan  2025-08-24  add BGE-small placeholder                    configs/bge_small.yaml
adc7f6b  Mei     2025-08-25  prepare for leadership review                notes/leadership_review.md
b3e3e0f  Priya   2025-08-27  PR #094: add name_length axis stub           src/stratified_eval.py
e9d1a5a  Mei     2025-08-29  clarify ADR-001 wording on fuzzy stack       docs/ADR-001.md
7c0a5c9  Jordan  2025-09-01  add city_frequency axis stub                 src/stratified_eval.py
6f3d0c1  Priya   2025-09-03  add runs/openai_3small/run3.json (alt GT)    runs/openai_3small/run3.json
f1a2e2e  Mei     2025-09-04  spot-check openai_3small similarity          notes/slack_embeddings_thread.md
e8b1df1  Priya   2025-09-05  test: stratified eval lexical overlap        tests/test_stratified_eval.py
c3e1d6b  Mei     2025-09-07  finalize Q3 retro doc                        docs/retro_q3_2025.md
d4b7bea  Jordan  2025-09-08  add run: fuzzy/partial_ratio canonical       runs/fuzzy_partial_ratio/run1.json
b2a65c4  Priya   2025-09-09  update wratio config for edge cases          configs/wratio.yaml
f8e2f5b  Mei     2025-09-10  update README with eval pipeline diagram     README.md
```

---

## Appendix B — retro action-item tracker

| Owner  | Item                                                      | Status      | Resolution Note                                                                                 |
|--------|-----------------------------------------------------------|-------------|-------------------------------------------------------------------------------------------------|
| Mei    | Close ADR-001                                             | done        | ADR-001 finalized and committed, leadership reviewed by 2025-10-31.                             |
| Mei    | Reproduce L12 ablation end-to-end                         | slipped     | No artifact produced before Q3 close; handoff to Priya pending Mei’s departure.                 |
| Priya  | Wire `name_length` axis in eval harness                   | in-progress | PR #083 open, initial metrics wired, aiming for Q4 completion.                                  |
| Priya  | Wire `city_frequency` axis in eval harness                | open        | Spec drafted, implementation to start in Q4.                                                    |
| Priya  | Track #eval-v2-tie-break ticket                           | in-progress | Ticket open, flag in ADR, monitoring for downstream issues.                                     |
| Arjun  | Own 3-large resolution post-Mei                           | open        | Re-embed or mark as unverifiable, due by 2025-12-15.                                            |
| Arjun  | Run leadership review                                     | done        | Review completed 2025-11-07, sign-off on ADR and decision path.                                 |
| Priya  | Reproduce fuzzy harness numbers                           | done        | Numbers matched prior runs; canonical partial_ratio top-1 = 0.4407.                             |
| Jordan | Ship canonical openai_3small run1.json                    | done        | File landed 2025-09-04, numbers reproduce, artifact is stable.                                  |
| Mei    | Build ground_truth/gt_alt.json                            | done        | Alternate GT created, flagged as non-canonical in documentation.                                |
| Priya  | Audit gt_alt.json for ambiguity filtering                 | done        | Multi-city hotels confirmed as legitimate; both GTs kept in repo.                               |
| Mei    | Spot-check openai_3large embedding order                  | done        | Issue found with row alignment; flagged as "unverified".                                        |
| Jordan | Debug openai_3large run1.json                             | dropped     | Left before resolution; ownership moved to Arjun.                                               |
| Priya  | Validate partial_ratio miss-rate on non-English names     | done        | Confirmed ~55% miss-rate in canonical subset, one-pager number not reproducible.                |
| Priya  | Land stratified eval harness (lexical_overlap)            | done        | PR #071 merged; axis functional and documented.                                                 |
| Arjun  | Decide on ada-002 baseline inclusion                      | done        | Baseline cut for budget/priority reasons per Q3.                                                |
| Mei    | Document tie-break regression in ADR                      | done        | Explicitly called out in ADR and retro.                                                         |
| Priya  | PR #042 eval_v2.py regression triage                      | done        | Tie-break behavior divergence analyzed, decision: leave in place and document.                  |
| Priya  | Confirm fuzzy stack reproducibility                       | done        | Harness stable; numbers match prior artifacts.                                                  |
| Mei    | Clarify source of 95% fuzzy miss-rate claim               | done        | Origin unclear, artifact not found, deck updated with correct numbers.                          |
| Jordan | Onboard to ADR workflow and run pipeline                  | done        | Completed in first month; ran openai_3small and openai_3large.                                  |
| Mei    | Draft initial Q3 plan for hotel→city matching             | done        | Plan set and shared in kickoff doc.                                                             |
| Arjun  | Coordinate BGE-small ablation plans                       | slipped     | Deferred to Q1 2026 pending budget.                                                             |
| Priya  | Prototype contrastive-tuned MiniLM                        | dropped     | Out of scope until ground-truth noise is quantified.                                            |
| Mei    | Prepare review deck with stratified metrics               | done        | Deck built for leadership review, flagged GT differences.                                       |
| Priya  | Validate canonical 3-small top-1 number                   | done        | Confirmed 0.4687 top-1, matches all major artifacts.                                            |
| Mei    | Specify artifact reproducibility requirement in ADR       | done        | ADRs now require committed artifact for headline metrics.                                       |
| Mei    | File CI issue for eval harness equivalence                | done        | #add-eval-ci ticket created, unassigned as of Q3 close.                                         |
| Priya  | Add script-level eval.py test to CI                       | open        | Pending #add-eval-ci prioritization; not yet started.                                           |
| Mei    | Review L12 ablation artifact existence                    | slipped     | No artifact found; status "unverified", next steps assigned to Priya/Arjun.                     |
| Jordan | Produce MiniLM L12 ablation run                           | dropped     | Number fabricated, no embedding artifact; run not valid.                                        |
| Priya  | Review and merge ADR-001 updates post-leadership review   | done        | Updates merged after Arjun/leadership feedback.                                                 |
| Arjun  | Ensure both GTs are visible in repo and flagged           | done        | Both gt.json and gt_alt.json are present and documented in README and ADR-001.                  |

---

## Appendix C — expanded lessons learned

**Script CI gap is a long-standing blind spot.** The lack of automated CI for `eval.py` and `eval_v2.py` has bitten us more than once. We tend to treat these scripts as "one-off" or "ad-hoc" tools, but as the evaluation harness grows more central (e.g., for stratified buckets, ablation runs, and GT variants), the cost of silent regressions rises. The tie-breaker bug could have been caught with a single fixed test case. In hindsight, even a basic `pytest` fixture for a canonical hotel/city pair would have prevented the 1pp inflation from sneaking into our numbers. There is a tendency to defer infra until "the real project", but as this retro shows, our tools *are* the project.

**Our cited-numbers culture is fragile.** We have, on several occasions, quoted numbers in decks and docs that were either not reproducible or lacked a clear provenance. The one-pager fuzzy miss-rate is the canonical example, but similar issues have cropped up when discussing BGE-small's hypothetical performance, or referencing "expected" gains from contrastive tuning. The core issue is that we lack a strong norm of linking every cited number to a committed, immutable artifact. This erodes trust in our own metrics and makes it hard to defend results to leadership or external reviewers. Going forward, every top-line number should have a run file and a GT citation.

**The 3-large handoff failed due to unclear ownership and incomplete debugging.** When Jordan flagged the 0.698 number as "correct" and Mei pushed back, the dispute was never fully resolved. Jordan's departure left the issue in limbo, with neither definitive verification nor a clear escalation path. This exposed a gap in our process around handling disputed metrics and cross-team handoffs — especially when a run could materially change the project direction. It should have been someone's explicit responsibility to drive the re-embed or mark the method as unverifiable within a fixed time window. Instead, we defaulted to flagging and moving on, which risks similar ambiguity in future high-stakes runs.

**eval_v2 tie-break story highlights silent behavioral drift.** The divergence in tie-breaking logic between `eval.py` and `eval_v2.py` was subtle — a 1pp shift on integer-valued scorers. This went unnoticed through review and was only caught by a sharp-eyed manual check. The root cause: we assumed behavioral equivalence based on code similarity, not on actual outputs. This is a recurring anti-pattern: trusting code diffs or reviewer intuition over automated checks. Our review process needs to explicitly include output equivalence checks when refactoring core scoring logic.

**gt_alt.json forking episode shows the need for GT stewardship.** Mei's creation of `gt_alt.json` was well-intentioned (to clean up edge cases for leadership review), but it bypassed the usual process for GT updates. The fork dropped multi-city hotels, which are legitimate ambiguities, not labeling errors. The lack of consensus over what constitutes "noise" vs. "signal" in GT risks splitting our evaluation baseline, undermining comparability across runs. We need a clearer protocol for GT changes: proposals, reviews (ideally with both quantitative and qualitative impact), and clear labeling in all downstream artifacts.

**We over-trusted Jordan's 0.698 number — a cautionary tale.** The excitement over a potential 22pp jump led us to accept the number into `runs/` with only a cursory check. This is a classic "too good to be true" scenario: big jumps deserve big skepticism. The lesson is procedural as much as technical — for any result that could change the project’s direction, a mandatory multi-person review and spot-check is warranted. This is especially true when the author of the result is about to leave the team.

**Manual spot-checks must be routine, not exceptional.** The 10-hotel spot-check policy was only invoked after a questionable number was already "in the wild". Going forward, this should be part of the standard checklist for any new run file, especially when new methods or GT variants are involved. A spot-check log (with hotel/city pairs and expected vs. actual matches) should live alongside each run artifact.

**Artifacts and documentation drift too easily.** We saw multiple instances this quarter where numbers in slides, one-pagers, or Slack threads didn't align with the canonical run files or GT. This creates confusion both within the team and for downstream consumers. We need a lightweight process for reconciling documentation with artifacts — perhaps a pre-milestone audit, or a bot that flags doc/run mismatches.

**Ownership ambiguity is a recurring failure mode.** The 3-large handoff, the L12 ablation slip, and the GT forking all suffered from unclear or shifting ownership. When a key person leaves or gets reassigned, their open loops can stall unless there’s a clear, documented handover. This is especially risky for evaluation and infra work, which often falls between the cracks. A running "open questions" doc, with explicit DRI assignments and due dates, would help.

**We are too tolerant of "unverified" numbers.** The decision to leave 3-large as "unverified" and keep fabricated ablation numbers in the repo is pragmatic but risky. Without a hard policy on what constitutes a verifiable run (e.g., presence of raw embeddings, run script, and GT), we risk sliding into a culture where numbers are provisional by default. This erodes confidence and makes downstream decision-making brittle.

**Ground-truth ambiguity should be surfaced, not hidden.** The instinct to "clean" the GT for leadership review is understandable, but it papers over real-world ambiguity that our models must handle. Instead, such ambiguities should be explicit: e.g., tagging problematic hotels in the GT, or adding a "multi-city" bucket to the eval. This helps both with model development (by highlighting hard cases) and with honest reporting.

**Process debt accumulates stealthily.** Many of these lessons come down to small process gaps: missing CI, unclear ownership, undocumented GT changes, and informal spot-checks. Each is individually minor, but together they create systemic risk. The team should allocate regular time each quarter to pay down this "process debt" — reviewing checklists, CI coverage, artifact provenance, and ownership docs — before it accumulates to the point of causing real harm.
