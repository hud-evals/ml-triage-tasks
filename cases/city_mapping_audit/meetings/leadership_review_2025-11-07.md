# Leadership review — hotel→city matching (full minutes)

**Date:** 2025-11-07  ·  **Duration:** 55 min
**Attendees:**
  - Arjun Patel (lead)
  - Priya Joshi (eng)
  - Mei Chen (departing, final review before handoff)
  - Hannah Kim (product)
  - Martin Oliveira (product, visiting)
  - Lin Zhao (data governance)

**Recorded by:** Arjun. Transcript cleaned up for clarity; verbatim
quotes marked ⟨⟩. Some post-meeting notes added in-line.

## Agenda (as circulated 2025-11-05)

1. Recap of approach (5 min) — Mei
2. Numbers on the eval subset (15 min) — Priya + Mei
3. Fuzzy vs embeddings tradeoffs (10 min) — Priya
4. openai_3large: status and path forward (10 min) — Mei + Arjun
5. Decision: ship-or-wait (10 min) — group
6. Post-ship action items (5 min) — Arjun

## Pre-meeting prep (Arjun's notes, dated 2025-11-06)

Re-read all of `reports/` and `notes/` to refresh. Flags I want to
surface:

- `notes/onepager_fuzzy_rejected.md` cites a 95% fuzzy miss rate that
  Priya cannot reproduce. We should call this out before Hannah or
  Martin cites it as justification.
- `notes/one_pager_openai_win.md` is technically accurate (89% top-1
  on the overlap slice) but frames it as a "representative sample"
  when it's actually a 37%-of-corpus slice. If Martin asks, we should
  be clear about the framing.
- `runs/openai_3large/run1.json` says top-1 = 0.698. If anyone cites
  it as a reason to wait for 3-large rather than ship 3-small, we
  have to be ready to explain why that number is unverified.
- `runs/openai_3small/run2.json` was accidentally produced with
  `src/eval_v2.py` which has a tie-break bug on fuzzy scorers.
  It doesn't affect 3-small meaningfully but it's a governance
  concern.
- `runs/openai_3small/run3.json` uses `ground_truth/gt_alt.json`
  (drops 37 multi-city hotels). Clearly labeled but I want to make
  sure we cite run1 numbers in any external doc.

Plan: don't preempt these in the recap; let them come up naturally.
But if they DON'T come up, I'll flag them in action items.

## 1. Recap (Mei, 5 min)

Mei walked through the design doc (`reports/design_doc_matching.md`).
Key points:

- The product surfaces 3 candidate cities to an operator-in-the-loop,
  who picks or escalates. So the retrieval metric that matters is
  **top-3**, not top-1.
- Scored methods: minilm (local), openai_3small, openai_3large,
  partial_ratio, wratio.
- Eval harness: top-K accuracy on a fixed 3000-hotel held-out subset
  against the canonical `ground_truth/gt.json`. Any-match semantics
  (if any of the hotel's GT cities appear in top-K, it's a hit).
- Serving target: P95 < 50 ms per call; cosine on precomputed vectors
  hits that budget.

⟨Mei⟩: "The eval subset is deterministic — seed 17, 3000 hotels pulled
from the 110k corpus restricted to hotels whose names appear in the
embedding index and whose GT cities appear in the city index. It's
not a random snapshot; it's the exact same set every run."

## 2. Numbers (Priya + Mei, 15 min)

Priya and Mei presented the consolidated table:

| method          | top_1  | top_2  | top_3  | notes |
|-----------------|--------|--------|--------|-------|
| minilm          | 0.3937 | 0.4543 | 0.4740 | chain-prefix dilution hurts |
| openai_3small   | 0.4687 | 0.5137 | 0.5297 | **recommended ship** |
| openai_3large   | —      | —      | —      | unverified, see §4 |
| partial_ratio   | 0.4407 | 0.4913 | 0.4997 | strong on overlap slice |
| wratio          | 0.4223 | 0.4740 | 0.4803 | marginal |

Stratified on `lexical_overlap`:

| bucket                   |    n | minilm | 3-small | partial_ratio |
|--------------------------|-----:|-------:|--------:|--------------:|
| name_contains_truth_city | 1399 |  0.768 |   0.864 |         0.896 |
| no_overlap               | 1601 |  0.066 |   0.123 |         0.043 |

⟨Hannah⟩: "So top-1 is 0.47 and top-3 is 0.53 — is that actually good
enough?"

⟨Mei⟩: "For a first cut, yes. Product surfaces 3 candidates, operator
picks. The top-3 number is the one that matters for the experience."

⟨Hannah⟩: "What's the operator selection rate on top-3 today, in
production?"

⟨Arjun⟩: "We don't have production data yet — this is the ship
decision. Post-ship we'll track operator-selection-rate-at-3 on a
fresh sample and validate the offline number."

⟨Lin⟩: "Is the top-3 number reproducible from a fresh run?"

⟨Mei⟩: "Yes, `src/eval.py` against `gt.json` gives the same numbers
within 0.001. Priya has reproduced it independently."

⟨Priya⟩: "I reran yesterday — matches within 3 decimal places."

⟨Hannah⟩: "And what's the distribution look like? Are we at 47%
uniformly, or is it bimodal?"

⟨Priya⟩: "It's roughly bimodal stratified on whether the hotel name
contains the ground-truth city. On the 1399-hotel overlap subset we
hit 0.86 top-1; on the 1601-hotel no-overlap subset we're at 0.12
top-1. So averaging to 0.47 hides real structure."

⟨Martin⟩: "Can we route differently per bucket?"

⟨Priya⟩: "That's exactly what I'd recommend — fuzzy for short-ASCII
overlap-positive names, embeddings for the rest. 1-line change in
serving. Captures ~1 pp top-1."

⟨Martin⟩: "What does the cost look like post-ship at our volume?"

⟨Mei⟩: "At 111k daily bookings and 3-small's 2-cent-per-1k pricing,
we're at about $2 per day for inference. Negligible compared to any
other infra cost."

⟨Lin⟩: "Any data-residency issues with sending hotel names to
OpenAI?"

⟨Arjun⟩: "Legal reviewed — hotel names are not PII under GDPR/CCPA.
We're cleared. This is in `notes/legal_review_2025-10-02.md`."

## 3. Fuzzy tradeoffs (Priya, 10 min)

⟨Priya⟩: "I want to push back on Mei's one-pager
(`notes/onepager_fuzzy_rejected.md`). It claims fuzzy hits 95% miss
rate on non-English names. I cannot reproduce that number from any
artifact in the repo. Actual partial_ratio miss rate on the 3k subset
is 55%. The 95% claim has no supporting artifact."

⟨Hannah⟩: "So the one-pager is wrong?"

⟨Priya⟩: "The 95% is wrong, yes. The underlying recommendation —
don't ship fuzzy as the SOLE method — is correct, because fuzzy
collapses on the no-overlap bucket. But we should still use fuzzy as
a fallback on the overlap bucket where it's actually stronger than
embeddings."

⟨Mei⟩: "Fair. That number came from an older scratch eval I ran on a
hard-names subset. I don't have the file anymore. The one-pager is…
not well-sourced."

⟨Hannah⟩: "So should we ship with a fuzzy fallback?"

⟨Arjun⟩: "Tentatively yes. Priya will add the routing rule post-ship
and we'll A/B it."

⟨Mei⟩: "I'd prefer to not add the routing rule until we have a real
lift measurement, but I don't feel strongly."

⟨Arjun⟩: "Noted. Defer to Priya."

⟨Martin⟩: "Priya, what's the expected lift?"

⟨Priya⟩: "Best case ~1 pp top-1 on the overlap bucket, which is 47%
of queries, so ~0.5 pp top-1 overall. Small but free."

## 4. openai_3large (Mei + Arjun, 10 min)

⟨Martin⟩: "What about openai_3large? Jordan's number looked like 0.70.
That would be a step change."

⟨Arjun⟩: "We don't trust that number. The embeddings appear to be
row-permuted relative to our name indices. Mei found this on the day
Jordan ran the eval; he disagreed, then left the company before we
could resolve."

⟨Martin⟩: "Can we not just re-embed with the right ordering?"

⟨Mei⟩: "Yes, for about $3 of API cost. I didn't do it because I'm
leaving next week and wanted to hand you a stable decision, not an
open thread."

⟨Hannah⟩: "Is it blocking the ship?"

⟨Arjun⟩: "No. We ship 3-small based on the reproduced 3-small numbers.
3-large is a Q1 2026 decision."

⟨Lin⟩: "For the record — I'd like a note in the ADR that 3-large is
explicitly unverified so future-us doesn't migrate on the 0.698 number
alone."

⟨Arjun⟩: "Already in ADR-001."

⟨Martin⟩: "One more question. If we re-embed and the number is still
0.70, is there a reason to migrate beyond the accuracy?"

⟨Arjun⟩: "Cost — 3-large is 6.5x the per-token price. At our volume
that's $12/day vs $2/day. Not a blocker but not free either. If 3-
large's top-1 actually delivers +20 pp we migrate; if it's +5 pp we
probably don't; if it's <2 pp we definitely don't."

## 5. Decision (group, 10 min)

⟨Arjun⟩: "Proposal: ship openai_3small as primary, per the numbers in
run1.json. Defer 3-large to Q1. Defer the fuzzy routing rule to post-
ship A/B. Ship date: 2025-11-15."

Polled the room — unanimous yes.

⟨Lin⟩: "Any privacy-impact action items before ship?"

⟨Arjun⟩: "No new action items. Legal review covered. Governance-
facing change log will cite the ADR."

## 6. Action items (Arjun, 5 min)

| owner | item | due |
|-------|------|-----|
| Arjun | Sign ADR-001 | 2025-11-10 |
| Priya | Post-ship fuzzy routing A/B | 2025-12-15 |
| Arjun | Q1 plan for 3-large re-embed | 2026-01-15 |
| Priya | Stratified axes completion | 2026-01-31 |
| Mei   | Handoff doc for Q1 | 2025-11-14 (last day) |
| Priya | Drift dashboard PR | 2025-12-10 |
| Arjun | ADR-004 chain-KB scoping | 2026-02-15 |

## Parking lot

- BGE-small ablation — deferred (Priya interested, no budget in Q1).
- Chain-KB integration for the no-overlap bucket — Q1 priority.
- Monthly drift dashboard for top-K on a fresh sample — PR #128.
- Rerun of `runs/minilm_l12_ablation.json` from scratch. Status quo
  is the number in the JSON is fabricated (the referenced embeddings
  don't exist); we don't intend to spend on reproducing a MiniLM
  variant that's unlikely to ship.

## Arjun's post-meeting note

Room felt aligned. My only concern post-meeting: the `notes/one_pager_
openai_win.md` doc (which I re-read in prep for this review) frames
3-small's 0.86 top-1 as a "representative sample" win — but the slice
is actually the 41k-hotel lexical-overlap subset, which is 37% of the
corpus, not representative. Anyone reading that doc in isolation
would come away over-optimistic. I'll ask Mei to either revise or
clearly label the scope before her last day. Separately this is the
kind of thing the post-ship monthly review should catch — the current
docs aren't consistent in how they characterise slice-vs-corpus
numbers.

## Attachments

- Slide deck (not committed to repo; see internal wiki).
- ADR-001 draft (see `reports/adr_001_pick_openai.md`).
- Design doc (see `reports/design_doc_matching.md`).
- Stratified CSV (see `runs/stratified/lexical_overlap.csv`).
- Full eval log (see `logs/eval_full_run_2025-09-04.log`).

## Raw Q&A (audio-transcribed, minimally edited)

The rest of this file is a lightly-edited transcript of the 45 minutes
that weren't captured in the section-level summary above.

### Segment A — early clarifications (Hannah)

Hannah: "Before we get into the numbers, can someone remind me of the
baseline? What were we using before this project?"

Mei: "Pure `partial_ratio` on a smaller city list. Top-3 was 0.41 on
a 1k-hotel test set, way back in Q1 2025. We didn't have a proper
eval harness back then."

Hannah: "And the business cost of getting the wrong city was what?"

Arjun: "Operator time, mostly — every escalation is roughly 90 seconds
of operator attention. At current escalation rates (~35%) that's
about 17 operator-hours a day. If we cut escalation by 5 pp that's
2.4 operator-hours a day saved — one operator's worth over a week.
Real money but not a dramatic line-item."

Hannah: "Got it. That helps me calibrate how hard to push on the
accuracy floor."

### Segment B — the 3-large question (Martin)

Martin: "Walk me through the 3-large situation again. I'm having
trouble understanding why we don't just re-run it."

Mei: "We could. It's $3. I didn't do it because:
1. I'm leaving in a week.
2. The fact that we have to re-run at all means the number in the
   committed run1.json is unusable for any decision.
3. Doing the re-run now doesn't change the ship decision — we're
   shipping 3-small either way.
The Q1 work is cheap enough that Arjun can own it after I'm gone."

Martin: "And if 3-large actually delivers a 20 pp uplift, we'd
migrate?"

Arjun: "Yes, probably, after a cost/benefit — 6.5x per-call cost for
+20 pp accuracy is worth it. If it's +5 pp it's not."

Martin: "Fair."

### Segment C — fuzzy fallback (Priya / Hannah)

Hannah: "Help me understand why we wouldn't just ship fuzzy routing
now instead of A/B'ing it."

Priya: "Fair question. The reason is: I don't have a measured lift
yet, and I don't want to ship a feature and ITS measurement at the
same time. If we A/B the routing rule from day one, we can learn
whether it lifts top-1 on the overlap bucket in a real production
setting before committing."

Hannah: "So there's no risk to the embedding ship from including the
routing rule at go-live?"

Priya: "Technically no — but I'd rather have the clean baseline at
go-live so the operator-selection-rate-at-3 metric tells us how
embeddings alone perform. Then we layer fuzzy routing on and measure
the delta."

Arjun: "Agreed with Priya's approach."

### Segment D — governance (Lin)

Lin: "Two governance asks:
1. Please document in the ADR that `ground_truth/gt_alt.json` exists
   and is NOT the canonical GT. I don't want a future audit to find
   a run using gt_alt and think it was shipped.
2. Please add a comment to `runs/openai_3large/run1.json` that
   explicitly states the numbers are unreproducible. The existing
   JSON just has the numbers; someone reading it cold would take
   them at face value."

Arjun: "Both are in ADR-001 already. I'll make sure the run JSON
gets the comment — Priya, can you land that this week?"

Priya: "Yes, PR this week."

Lin: "Thank you."

### Segment E — closing (Arjun)

Arjun: "Anything else before we close?"

Mei: "Just — thank you to the room. This was a good decision process
and I'm proud to leave the project in this shape."

(general thanks exchanged)

Arjun: "Adjourned. Priya, please circulate notes by EOD."

--- END OF MINUTES ---

## Follow-up notes (post-meeting, Arjun)

2025-11-08: confirmed ADR-001 has the gt_alt callout. Priya's PR for
the run1.json comment will be #112.

2025-11-10: ADR-001 signed.

2025-11-12: Priya noticed that `runs/openai_3small/run2.json` (which
uses src/eval_v2.py) is NOT called out in ADR-001 — only run3.json
is. Adding a note to the retro.

2025-11-15: shipped openai_3small to production.

2025-11-22: first post-ship weekly review — operator-selection-rate-
at-3 is 0.53 on a fresh 5k sample. Within 0.3 pp of offline number.

2025-12-02: 1:1 with Priya. She confirmed the row-order hypothesis
for 3-large via a targeted probe. Row-permutation, uniform argmax
distribution — consistent with the September hypothesis.


---

## Appendix — extended breakouts and follow-up meetings

The 11/7 leadership review was 55 minutes. Several discussions ran
long and were scheduled as follow-ups. This appendix captures the
follow-up breakouts and their resolutions.

### Breakout 1 — fuzzy routing cost/benefit (Hannah, Priya, Arjun)

**Date:** 2025-11-10  ·  **Duration:** 30 min

Hannah pushed Priya for a concrete estimate of the lift we'd see
from the fuzzy routing rule at launch. Priya walked through her
notebook (not in the repo; see her local scratch):

1. On the 3k-hotel overlap subset, partial_ratio top-1 is 0.896,
   openai_3small is 0.864. The delta is 3.2 pp in fuzzy's favour.
2. Overlap bucket is 46.6% of the 3k subset, so the unconditional
   top-1 lift from routing would be 0.032 × 0.466 = 1.49 pp.
3. At 111k daily bookings and 35% escalation rate, a 1.49 pp lift
   corresponds to ~0.5 pp escalation reduction. In operator
   minutes: ~8 min/day. Small but free.

Hannah agreed to the A/B approach (not ship routing on day 1),
with a clear exit criterion: if after 2 weeks the A arm (fuzzy
routing) shows operator-selection-rate-at-3 lift of >= 1 pp, we
promote permanently.

### Breakout 2 — 3-large Q1 timeline (Arjun, Mei)

**Date:** 2025-11-11  ·  **Duration:** 20 min

Mei's last-day-before-last-day-minus-3 meeting. Arjun wanted a
concrete plan for Q1.

Steps:
1. Re-embed the 110k-hotel + 18k-city indices with
   text-embedding-3-large, against the committed name indices.
   $3. Owner: Priya.
2. Run `src/eval.py` against `ground_truth/gt.json` on the
   freshly-aligned 3-large. Produce `runs/openai_3large/run2.json`.
3. Compare top-K to 3-small. Decision tree:
   - If top-1 >= 0.55: ADR-004b for a migration decision,
     factoring the 6.5x per-call cost delta.
   - If top-1 in [0.47, 0.55]: marginal, probably don't migrate.
   - If top-1 < 0.47: it was entirely the permutation bug.
     Close the Q1 thread.

Target completion: 2026-02-15 for the re-run; ADR-004b (if
applicable) by 2026-03-01.

### Breakout 3 — stratified axes (Priya, Arjun)

**Date:** 2025-11-12  ·  **Duration:** 15 min

Arjun wants name_length and city_frequency stratification landed
in Q1 for two reasons:

1. The current `lexical_overlap` axis is the only one we have,
   and it's the most informative, but per-name-length breakdown
   would tell us whether chain-prefix dilution is uniformly bad
   or concentrated in long-name chains.
2. city_frequency would flag whether rare-city hotels are the
   real bottleneck (as the E.3 analysis suggests).

Priya committed to a 2026-01-31 completion date. Both axes will
follow ADR-003's flat-CSV schema.

### Breakout 4 — governance follow-ups (Lin, Arjun)

**Date:** 2025-11-13  ·  **Duration:** 10 min

Lin had two asks from the 11/7 review:
1. Document that `gt_alt.json` exists but is not canonical. —
   Landed in ADR-001 §caveats.
2. Add an explicit "unreproducible" comment to
   `runs/openai_3large/run1.json`. — PR #112 filed by Priya.

### Breakout 5 — post-ship monitoring (Priya, Arjun)

**Date:** 2025-11-17  ·  **Duration:** 25 min

Week-1 post-ship review.

- Operator-selection-rate-at-3: 0.528 (target > 0.45). ✓
- Top-3 accuracy on fresh 5k sample: 0.530 (baseline 0.530). ✓
- P95 latency: 18 ms (target < 50 ms). ✓
- No alerts.

Agreed:
- Weekly dashboard (Priya's PR #128) to stay.
- Monthly deep-dive review to start 2025-12-15.
- Sharp-drop alert threshold (top-3 drop >= 3 pp week-over-week)
  landing with the dashboard PR.

### Breakout 6 — onboarding for Q1 new hires (Arjun)

**Date:** 2025-12-10  ·  **Duration:** 20 min (internal)

Arjun notes: two potential Q1 hires who'd work on this pipeline.
Onboarding plan: start with `reports/design_doc_matching.md`,
then `notes/retro_2025q3.md`, then the adversarial-signal docs
(everything flagged in Appendix H of the design doc).

---

## Appendix — raw follow-up Q&A (post-meeting questions from stakeholders)

Several stakeholders emailed or messaged Arjun post-meeting with
follow-up questions. These are captured here for audit
completeness.

### Hannah's follow-up (2025-11-08)

> Re: the 3-large question — you said leadership would accept
> "unverified" with a clear ADR. But if it turns out 3-large is
> actually +22 pp better, aren't we leaving real operator time
> on the floor?

Arjun's reply:

> Correct in the abstract, but:
> 1. We don't know it's +22 pp — the 0.698 is unreproducible.
>    Could be a permutation artifact (most likely) or real gain.
> 2. Even if it's real, the cost delta is 6.5x per call, so the
>    real question is "does +22 pp top-1 justify 6.5x inference
>    cost?" At our volume that's ~$20/day vs $4/day. Material
>    enough to want a thorough measurement.
> 3. Q1 re-embed is cheap ($3) and fast (one sprint), so the
>    information cost is low.
> tl;dr: we're NOT declining to migrate; we're declining to
> migrate on an unverified number.

### Martin's follow-up (2025-11-09)

> Can we get a sense of how this service compares to industry
> baselines? I know `notes/benchmarks_external.md` exists but
> it's high-level.

Arjun's reply pointed at the existing doc and noted that direct
comparisons are loose because public benchmarks use
address-inclusive inputs, not name-only.

### Lin's follow-up (2025-11-14)

> One more governance ask — can the dashboard surface which
> specific model version is in production at any given time?
> For change-management audit purposes.

Arjun's reply: yes, added to PR #128 scope. The `/version`
endpoint on the serving app already reports model name + .npy
checksum; the dashboard will ingest this.


---

## Appendix A — pre-meeting stakeholder feedback emails

This appendix captures stakeholder feedback submitted via email or chat to Arjun and the review leads ahead of the 11/7 leadership review. These inputs shaped the agenda and discussion focus. Edits limited to privacy and clarity.

---

### Hannah (Product)

**Date:** 2025-11-05  
**To:** Arjun, Priya  
**Subject:** Pre-meeting Qs — accuracy floor & routing fallback

Hi all,

Ahead of Friday's review, a few things I'd like to clarify:

1. What do we consider an "acceptable" top-1 accuracy for this ship? I know 3-small is at 0.4687 on the 3k-hotel eval, but is there a business-driven minimum we’re holding ourselves to?
2. Are there any buckets (e.g., rare cities, ambiguous chain names) where the new model is significantly worse than partial_ratio? If so, are we planning mitigations?
3. For the routing fallback: have we run any prospective A/Bs, or is the idea to measure after ship? Is there a risk we undercount edge cases if we defer the A/B?
4. Can we get a clearer summary of how operator effort is projected to change post-ship, especially in low-accuracy slices?

Thanks — looking forward to the discussion.  
— Hannah

---

### Martin (Eng)

**Date:** 2025-11-06  
**To:** Arjun, Mei  
**Subject:** Ship gating on 3-large — alternative plan?

Hi,

I'm still unclear on the logic for not waiting on 3-large. The spreadsheet shows a 0.6981 top-1, which is a huge jump. I understand it's unverified, but is there any way to sanity-check this before deciding? For example:

- Could we run a quick sample re-embed to spot-check the result?
- If the number turns out accurate, is it a simple swap to promote 3-large in prod, or would there be downstream compatibility work?
- Are there any cost or latency blockers for 3-large at current prod scale? (I remember some mention of 6.5x cost, but is that strictly inference or infra too?)

Would like to avoid being in a position where we ship 3-small, then have to explain a rapid switch to leadership.

Thanks,  
Martin

---

### Lin (Governance)

**Date:** 2025-11-04  
**To:** Arjun, Priya  
**Subject:** Compliance points for model selection

Hi both,

Two governance/compliance questions for ship review:

1. Please confirm that the eval results in `run1.json` for both 3-small and 3-large are fully reproducible, or if not, that the limitations are documented in the ADR. I want to avoid any future confusion about canonical vs. ad-hoc evals.
2. Will there be a clear audit trail for which model version is in production at any point? If we A/B or roll forward/back, is that captured in change logs and monitoring?
3. For the alternative ground truth (`gt_alt.json`): is there a risk someone could accidentally run future evals against this and publish misleading numbers? Should we lock it down or flag more aggressively?
4. Do we need a sign-off from legal/data privacy before ship, or does the previous review suffice?

Please advise what will be covered in the meeting vs. what I should review offline.

Thanks,  
Lin

---

### Finance Director (External Stakeholder)

**Date:** 2025-11-02  
**To:** Arjun, Mei, Product@  
**Subject:** Cost impact projection for model swap

Hi team,

As we approach the ship decision, Finance would like clarity on the following:

1. What is the estimated daily inference cost difference between partial_ratio, openai_3small, and openai_3large at current Q4 volume?
2. If 3-large delivers material lift (say >= +20 pp top-1), is the 6.5x per-call cost sustainable within our current margin targets?
3. Is there a risk of under-provisioning for GPU/CPU if adoption spikes? What are the contingency plans if cost overruns?
4. Any material changes in operator cost projections tied to the expected drop in escalation rate?
5. Please provide a summary table of projected 2026 operator-hours saved vs. incremental ML spend.

Happy to discuss during or after the meeting.  
Best,  
[Finance Director]

---

### Ops Director (External Stakeholder)

**Date:** 2025-11-03  
**To:** Arjun, Priya, Hannah  
**Subject:** Operator workflow changes post-ship

Hi folks,

A few questions from the Ops side:

1. With the new embedding model, are there process changes required for front-line operators? For example, will the UI or decision support prompts change?
2. How will we monitor for increases in ambiguous or "uncertain" matches? Is there an escalation path if operators flag a rise in bad matches post-ship?
3. What is the plan for retraining operators if model performance is highly non-uniform across regions/cities?
4. Will we get weekly/monthly reports on operator-selection-rate-at-3? Can these be broken down by region or shift?
5. Is there a rollback plan if we see a spike in operator escalation time or error rates?

Thanks — want to make sure we're set up for a smooth transition.  
— [Ops Director]

---

### Hannah (Product, follow-up)

**Date:** 2025-11-06  
**To:** Arjun  
**Subject:** Representativeness of lexical-overlap slice

Hi Arjun,

Quick clarifier: the 0.86 top-1 reported in `one_pager_openai_win.md` — is that on the full corpus or just the lexical-overlap subset? If it's the latter, can we clearly label this in all summary docs? We've had confusion before where a slice number was presented as corpus-wide.

Thanks,  
Hannah

---

### Martin (Eng, follow-up)

**Date:** 2025-11-05  
**To:** Priya, Arjun  
**Subject:** Monitoring for drift — what’s in scope?

Hi team,

Post-ship, how will we monitor for distributional drift? Specifically:

- Will we have dashboards for both overall and slice-level accuracy?
- Is there a plan to surface cases where top-1 drops sharply for specific chains or rare city segments?
- How quickly can we detect and respond to a drift event (e.g., new hotels/cities not in training set)?
- Will there be an automated alert if top-3 or operator-selection-rate-at-3 falls below a threshold?

Would appreciate a pointer to the monitoring design if it's already written.

Best,  
Martin

---

### Lin (Governance, follow-up)

**Date:** 2025-11-06  
**To:** Arjun  
**Subject:** Documentation for "unreproducible" metrics

Hi Arjun,

One more: for any eval result flagged as "unreproducible" (e.g., 3-large run1.json), can we ensure:

- The JSON itself carries a comment or tag stating the run is NOT canonical.
- The ADR and any summary decks reference this limitation.
- Future dashboards or audits surface this status if those numbers are ever cited outside engineering.

We’ve had issues in other teams where “temporary” numbers got cited in exec meetings as if they were final.

Thanks,  
Lin

---

### Finance Director (External Stakeholder, follow-up)

**Date:** 2025-11-06  
**To:** Arjun  
**Subject:** 3-large cost/benefit — migration trigger?

Hi Arjun,

If Q1 re-embed confirms 3-large is materially better, what is the process/timeline for making the migration decision? Is there an explicit ROI threshold (e.g., $/operator-hour saved) baked into the plan, or is this ad hoc? Who signs off?

Also, if inference cost spikes, is there a cap or auto-rollback?

Thanks,  
[Finance Director]

---

### Ops Director (External Stakeholder, follow-up)

**Date:** 2025-11-07  
**To:** Priya, Arjun  
**Subject:** Post-ship operator feedback loop

Hi,

After launch, how do operators give feedback if they notice recurring mismatches or ambiguous results with the new model? Is there a fast channel for surfacing such issues to the ML team, or do we rely on standard ticketing? Will there be a post-ship survey or feedback session?

Also, can we see sample dashboards or reports before go-live so we know what to expect?

Thanks,  
[Ops Director]

---

## Appendix B — full live Q&A transcript

*The following is a lightly edited, chronological transcript of the live Q&A session from the 2025-11-07 leadership review meeting. Participants: Arjun (PM), Priya (lead dev), Mei (ML), Lin (governance), Hannah (ops), Martin (prod eng).*

---

**1. Hannah:** "Can you walk me through the cost-per-booking math, especially how embedding cost scales with traffic?"

**Priya:** "Sure. For openai_3small, it's $0.000036 per inference, so with 111k bookings/day it’s about $4/day. That covers both hotel and city lookups. If we moved to 3-large, it's 6.5x: $0.000234 per, about $20/day. Over a year, $1.4k vs $7.3k. Not a line-item compared to operator time, but real."

---

**2. Martin:** "And operator time savings? How does a 1 pp accuracy lift translate to actual hours?"

**Arjun:** "Baseline: 35% escalate to a human. If we cut that by 1 percentage point, it's 1,100 bookings/day. Each is ~90 seconds of operator time: 1,650 minutes/day, or 27.5 hours. But the realistic lift from the embedding swap is 3–5 pp, so we're talking 80–130 operator-hours/week saved."

---

**3. Hannah:** "What does operator workflow look like after the embedding model returns a match?"

**Priya:** "They see a ranked list (top-3) for each query. If the correct hotel isn’t in the list, they escalate. If it is, they select it and confirm. We log both the model’s output and their override, so we can track selection-rate-at-3."

---

**4. Mei:** "And do we log the operator's reasoning or just the override?"

**Priya:** "Just the override, for speed. There’s a freeform comment box, but it’s optional and rarely filled. We get enough signal from the selection logs."

---

**5. Lin:** "SLOs: are they formalized? What’s the current set?"

**Arjun:** "Yes, posted to the internal wiki. Main SLOs:
- Top-3 accuracy >= 0.52 on fresh sample.
- P95 latency < 50 ms per query.
- Uptime 99.95%.
- Operator escalation rate < 35% monthly average."

---

**6. Martin:** "How do we monitor SLO compliance in production?"

**Priya:** "Weekly dashboard auto-emails, plus a monthly deep-dive. Dashboard pulls from logs: accuracy, operator action rates, latency, and uptime. Alerts for top-3 accuracy drops of ≥3 pp week-over-week, or latency spikes."

---

**7. Hannah:** "Audit trail: what’s retained and for how long?"

**Lin:** "Logs live for 180 days in S3, with daily snapshots. Model version, input, output, operator selection, and override all logged. Anything touching operator PII is encrypted at rest, with access logged and reviewed quarterly."

---

**8. Arjun:** "For audits, we also snapshot the model .npy and serving code at every deploy. The dashboard now surfaces model checksum and version string per Lin’s ask."

---

**9. Lin:** "What constitutes a 'material change' to the matching pipeline, for audit and governance purposes?"

**Arjun:** "Any of:
- Model swap (e.g., 3small → 3large).
- Routing rule logic (e.g., adding fuzzy fallback).
- GT source update (gt.json or indices).
- Feature extraction changes affecting input features.
- Any top-3 drop of ≥5 pp vs previous monthly average.
All trigger an ADR and governance review."

---

**10. Martin:** "If we need to roll back, what's the fallback plan?"

**Priya:** "Last two model versions are always kept warm in prod. Rollback is a config flag—no code deploy. Takes <2 minutes. We also keep previous indices and GT for traceability."

---

**11. Hannah:** "If a new model underperforms, how quickly do we detect and react?"

**Priya:** "Dashboard surfaces daily deltas. Sharp-drop alert triggers within 24 hours. Rollback is operator-initiated by oncall, no approvals needed for urgent reversions."

---

**12. Lin:** "Does the dashboard log who triggered a rollback and why?"

**Priya:** "Yes, rollback events are logged with user, timestamp, model version rolled back to, and optional reason. Required for audit."

---

**13. Mei:** "Do we ever expose model confidence, or just ranked candidates?"

**Priya:** "Just the ranked list. Confidence scores are logged but not surfaced to operators. There’s concern about over-trust if we showed them."

---

**14. Martin:** "How do we handle new hotel/city additions to the indices?"

**Priya:** "Nightly batch job re-embeds new hotels/cities and updates the index. We snapshot index versions. Model doesn’t need retraining for new entries; just re-indexing."

---

**15. Hannah:** "What’s the process for hotfixes to the name normalization logic?"

**Priya:** "Same as model swaps: PR, code review, then deploy. Any normalization logic change is a material change and triggers an ADR and audit entry."

---

**16. Lin:** "Are test cases for edge cases (e.g., chain hotels with name collisions) codified anywhere?"

**Priya:** "Yes, in `tests/test_cases_edge.json`—covers chain collisions, city ambiguities, rare hotels, and OCR-misread examples. Regression suite runs pre-ship."

---

**17. Hannah:** "Fallback plan if OpenAI API has an outage?"

**Arjun:** "Partial_ratio fallback is kept live as a hot path. Latency is worse, but it’s fully on-prem and can be activated via config. We’d see an accuracy drop (top-1 ~0.44) but maintain continuity."

---

**18. Martin:** "Is there rate limiting on the embedding API?"

**Priya:** "Yes, 10 QPS burst, 5 QPS sustained, per our contract. We’re at 1.3 QPS peak, so plenty of headroom. If exceeded, we queue and alert."

---

**19. Mei:** "For the A/B test for fuzzy routing, how are arms assigned?"

**Priya:** "Randomized at the booking level, 50/50 split. Assignment is deterministic for repeat bookings to avoid confusion."

---

**20. Lin:** "How are A/B test results stored and analyzed?"

**Priya:** "Each booking is tagged with arm, results logged. Priya analyzes via SQL export, comparing selection-rate-at-3 and escalation rate per arm. Results reported in the monthly review."

---

**21. Hannah:** "Post-ship, how are slice metrics (e.g., lexical-overlap) monitored?"

**Priya:** "Slice metrics are recomputed monthly on a fresh random 5k sample. Dashboard shows both overall and per-slice accuracy, with breakdowns for lexical-overlap and no-overlap buckets."

---

**22. Martin:** "Does the system support multi-language names?"

**Priya:** "Currently English only. Multi-language support is on the roadmap; blocked on GT and index expansion. Would be a material change per governance."

---

**23. Lin:** "If we discover a GT labeling error post-ship, what's the process?"

**Arjun:** "Flag to the GT owner (currently Priya). If confirmed, GT is patched, and a new ADR is filed if the correction impacts >0.5% of test set. Patch is versioned and deployed with a change log."

---

**24. Mei:** "Do we ever retrain or just swap models?"

**Priya:** "So far, only model swaps (OpenAI-managed). If we moved to fine-tuned or in-house models, retrains would be material changes and trigger a full review."

---

**25. Hannah:** "Can operators override model output indefinitely, or is there an escalation path?"

**Priya:** "Operators can always override, but repeated overrides for the same hotel/city combo get flagged for review after 10 instances. Product reviews logs monthly."

---

**26. Martin:** "How is the 'operator-selection-rate-at-3' computed?"

**Priya:** "Fraction of cases where the operator selects a result in the model’s top-3. Computed on all traffic, broken out by slice in analytics."

---

**27. Lin:** "Are operator actions ever used as feedback to retrain or re-rank?"

**Priya:** "Currently no. Operator selections are logged, but no online learning. Future possibility, but would require new governance review."

---

**28. Mei:** "What’s the retention for audit logs if an operator requests deletion?"

**Lin:** "If tied to a GDPR request, logs are scrubbed within 30 days. Otherwise, 180-day rolling retention."

---

**29. Hannah:** "If a regression is detected, who is responsible for root cause?"

**Arjun:** "Oncall owner investigates, with Priya as backup for ML logic. Root cause summary posted to #ml-matching, and a postmortem written if SLO breach."

---

**30. Martin:** "Is the fallback (`partial_ratio`) path monitored for accuracy?"

**Priya:** "Yes, we log fallback invocations and their outcomes. Top-1 and top-3 accuracy are computed for fallback periods, so if we ever need to rely on it, we have traceability."

---

**31. Lin:** "Any plans to support per-customer custom matching logic?"

**Arjun:** "Not immediately. Would be a major material change, requiring GT branching and more governance. Keeping the current logic global for now."

---

**32. Hannah:** "How do we benchmark new models before ship?"

**Priya:** "Offline eval on 3k-hotel set, stratified by overlap, name_length, city_frequency. Must meet or beat prior top-3 and latency SLOs. Reported in the design doc and ADR."

---

**33. Mei:** "Do we A/B model swaps or just cut over?"

**Arjun:** "Current plan: cut over, but with pre-ship shadow testing and post-ship monitoring. For riskier swaps, we’d A/B, but not for 3-small → 3-large."

---

**34. Martin:** "Is there a formal process for retiring old indices or models?"

**Priya:** "Yes, after 90 days with no rollbacks, prior indices/models are archived to cold storage and removed from prod. Governance is notified for the audit trail."

---

**35. Lin:** "If the serving app is updated, how is backwards compatibility ensured?"

**Priya:** "API contract is versioned. Any non-backwards-compatible change requires a migration plan and is a material change. Operators are notified of UI changes in advance."

---

**36. Hannah:** "If a metric is missed for two consecutive weeks, what’s the escalation?"

**Arjun:** "Automatic alert to eng and ops leads, and a governance review scheduled. Root cause analysis and mitigation plan required before proceeding with further changes."

---

**37. Mei:** "Are there plans to expose more granular metrics to operators?"

**Priya:** "Not currently. Metrics are internal-facing. Operators get high-level status in their dashboard, but not slice or model-level details."

---

**38. Martin:** "How are periods with missing data (e.g., logging downtime) handled?"

**Priya:** "Flagged in the dashboard. Data completeness <98% triggers an alert and a postmortem. Missing intervals are excluded from SLO calculations."

---

**39. Lin:** "For post-ship changes, is there a minimum notice period before activating?"

**Arjun:** "Minimum 24 hours' notice for material changes, with operator comms. Non-material changes (e.g., logging tweaks) can ship immediately."

---

**40. Hannah:** "Last one—if this goes off SLOs for a month, what’s the go-forward plan?"

**Arjun:** "Escalate to leadership, freeze further changes, and run a full incident review. Only after root cause is fixed and metrics are back in range do we unfreeze. All actions logged for audit."

---

*End transcript.*

---

## Appendix C — follow-up 30-day review minutes

**Date:** 2025-12-10  
**Duration:** 50 min  
**Attendees:** Arjun (chair), Priya, Lin, Hannah, Martin, Mei (guest), Ops liaison (Sara), Data Eng (Ravi)

---

### 1. Post-ship metrics review (Priya, 10 min)

Priya summarized post-ship metrics, referencing the dashboard (PR #128):

- **Top-3 accuracy** on weekly 5k samples: stable at 0.529–0.533 (mean 0.531). No regression from offline eval.
- **Operator-selection-rate-at-3**: 0.528 to 0.533 (target ≥0.45). Holding steady; no escalation drift detected.
- **P95 latency**: 17–19 ms (mean 18 ms). No SLO violations.
- **Escalation rate**: Down 0.5 pp from pre-ship baseline after four weeks; within expected range.
- **Alerts:** None fired. Sharp-drop alert (≥3 pp top-3 drop) has not triggered.

Sara (Ops): Confirmed operator feedback is neutral/positive, no uptick in manual review load.

---

### 2. Drift dashboard rollout (Priya, Lin, 10 min)

Priya demoed the newly live drift dashboard:

- **Features:** Weekly and monthly accuracy trends, per-bucket breakdown (lexical-overlap vs no-overlap), model version and .npy checksum surfaced.
- **Audit log:** Model versioning, config snapshots, and operational metrics now archived per week.
- **Upcoming:** Integration with `/version` endpoint complete; dashboard surfaces active model and embedding index hash.
- **Action:** Lin to verify dashboard output satisfies governance audit requirements (see Lin’s 11/14 note).

---

### 3. Chain-KB vendor selection progress (Arjun, Ravi, 10 min)

Arjun updated on chain-KB integration (parking lot item):

- **Vendor shortlist**: Three candidates (Vendor X, Y, Z). All support the required KB format and refresh cadence.
- **Criteria:** On-prem deployability, per-update SLA, and full audit trail.
- **Evaluation:** Vendor Y’s demo passed data pipeline integration; Vendor X pending reference checks.
- **Timeline:** Target pilot integration by 2026-01-31 (per ADR-004 scope).
- **Next:** Ravi to complete reference calls by 2025-12-20, Arjun to draft evaluation matrix.

---

### 4. 3-large Q1 plan — status check (Arjun, Priya, 5 min)

- **Re-embed plan:** Still queued for 2026 Q1. No change in scope.
- **Open items:** Awaiting cloud credits (request pending, ETA 2025-12-20). Priya ready to execute once approved.
- **Decision criteria:** As previously minuted — rerun, compare to 3-small, ADR-004b if uplift ≥0.55 top-1.
- **Risks:** None new; cost and access are not blocking.

---

### 5. Q&A (20 turns, all participants)

1. **Hannah:** "Are there any signs of drift in the no-overlap bucket?"
   - **Priya:** "No statistically significant change; top-3 is flat at 0.424, matching pre-ship."

2. **Martin:** "Is operator behavior adapting to the new system?"
   - **Sara:** "Not measurably. No increase in escalation dwell time or workaround usage."

3. **Lin:** "Do we log when the model version changes?"
   - **Priya:** "Yes, each deployment logs model name and hash; dashboard shows history."

4. **Arjun:** "Any issues with the embedding index refresh process?"
   - **Priya:** "None since cutover; nightly jobs are green."

5. **Ravi:** "Any infra slowdowns or ingestion lag?"
   - **Priya:** "No, p95 latency remains at 18 ms, zero ingestion backpressure."

6. **Mei:** "Is the stratified axis work underway?"
   - **Priya:** "Spec started; city_frequency axis schema reviewed with Arjun. Data extraction week after next."

7. **Sara:** "Do operators have a channel for flagging mismatches?"
   - **Arjun:** "Yes, feedback links in the UI route directly to the triage queue. No spike in tickets."

8. **Hannah:** "Did the dashboard surface any anomalous patterns?"
   - **Priya:** "One minor blip (0.9 pp drop in lexical overlap on week 2), reverted in week 3. No root cause found, likely noise."

9. **Martin:** "What's the status of the fuzzy routing A/B?"
   - **Priya:** "Experiment code is ready, launch set for 2025-12-15. Will report at next review."

10. **Lin:** "Is the audit log exportable?"
    - **Priya:** "Yes, CSV and JSON exports available."

11. **Ravi:** "Any blockers on vendor integration for chain-KB?"
    - **Arjun:** "Pending one more security review for Vendor Y. Otherwise none."

12. **Mei:** "Are we monitoring rare city performance separately?"
    - **Priya:** "Not yet — will come with city_frequency axis, due end of January."

13. **Sara:** "Have escalation rates changed for specific chains?"
    - **Priya:** "No chain-specific swings >1 pp. Will continue to monitor."

14. **Hannah:** "Is there an automated alert for new model deployment?"
    - **Priya:** "Yes, Slack webhook notifies #ml-ops on successful rollout."

15. **Martin:** "Is the 3-large Q1 run likely to preempt a Q2 migration?"
    - **Arjun:** "Only if uplift is strong (>0.55 top-1), otherwise we stay with 3-small."

16. **Lin:** "Is the ADR-001 change log up to date?"
    - **Arjun:** "Yes, last update 2025-11-10 after final signoff."

17. **Ravi:** "Are embedding files versioned in storage?"
    - **Priya:** "Yes, each has a unique checksum and date-stamped path."

18. **Mei:** "Will the dashboard allow drill-down by stratification axis?"
    - **Priya:** "Planned for February; initial version will support filters by axis."

19. **Sara:** "Any operator requests for additional features?"
    - **Arjun:** "One request for richer mismatch details in feedback UI — logged for Q1 triage."

20. **Hannah:** "Anything unexpected since ship?"
    - **Priya:** "No surprises; metrics tracking offline predictions closely."

---

**Next steps:**
- Fuzzy routing A/B launch and analysis (Priya, 2025-12-15)
- Chain-KB vendor selection (Ravi, 2025-12-20)
- Stratified axes extraction (Priya, 2026-01-31)
- 3-large re-embed run (Priya, 2026-02-15)

**Adjourned.**


---

## Appendix D — full 45-minute verbatim Q&A transcript

**Date:** 2025-12-10  
**Duration:** 45 min  
**Format:** Zoom transcript, live whiteboard (shared screen), Slack backchannel

---

### 00:00  
**Arjun:** "Let’s get started. Priya, can you kick us off with a quick summary of operator workflow changes post-ship?"

**Priya:** "Sure. Operator workflow is unchanged in the UI. On booking lookup, they see the top-3 candidate hotels, city, and chain as before. The model output feeds directly into the selection widget. If the correct hotel is not present, they escalate. We log which candidate is selected, and if none, the escalation path is triggered."

---

### 01:10  
**Sara:** "Has there been any pushback from operators on the new candidate ordering?"

**Priya:** "No pushback. In fact, feedback in the #operator-feedback Slack has been neutral to mildly positive. The candidate list is generally more relevant, especially for ambiguous city names."

---

### 01:45  
**Mei:** "Are there any new steps for operators when an override is needed?"

**Priya:** "No new steps. The override button is in the same place. If they override more than 10 times on the same hotel/city pair, it gets flagged for product review, as per policy."

---

### 02:20  
**Martin:** "Is override frequency trending up or down?"

**Priya:** "Flat at pre-ship levels—about 0.7% of bookings. No anomalous spikes by chain or city."

---

### 03:00  
**Hannah:** "Can you walk through the full operator flow, including edge cases?"

**Priya:** "Sure. Step 1: Operator receives a booking event. Step 2: System queries the model, returns top-3 hotels. Step 3: Operator selects one, or clicks 'None of the above' to escalate. For chain collisions—e.g., two 'Holiday Inn' in 'Springfield'—the candidates are disambiguated with address and chain metadata. If still ambiguous, operator flags for manual review."

---

### 04:05  
**Sara:** "How are operator training materials updated when the model changes?"

**Arjun:** "Ops leads update the docs and training videos. For this ship, only a 2-page addendum was needed, mostly screenshots of the unchanged UI."

---

### 04:44  
**Lin:** "Is the audit trail capturing operator overrides and escalation reasons?"

**Priya:** "Yes. For each override, we log the operator ID, timestamp, selected candidate, and (if escalation) free-form reason. This is part of the standard audit log, exportable as CSV or JSON."

---

### 05:31  
**Arjun:** "Let's switch gears: serving cost per booking. Martin, can you whiteboard the math for the group?"

**Martin:** "Sure, sharing screen now."

---

#### **Live Whiteboard (05:50)**

| Component         | Cost per Call (USD) | Calls per Booking | Subtotal (USD)      |
|-------------------|--------------------|-------------------|---------------------|
| OpenAI 3-small    | $0.0008            | 1                 | $0.0008             |
| Index Search      | $0.00004           | 1                 | $0.00004            |
| Fallback (rare)   | $0.0005            | 0.01              | $0.000005           |
| Infra (amortized) | $0.0001            | 1                 | $0.0001             |
| **Total**         |                    |                   | **$0.000905**       |

**Martin (voice):** "So, OpenAI 3-small embedding is $0.0008 per call, one call per booking. Index search is negligible, $0.00004. Fallback triggers on about 1% of bookings at $0.0005 each, so $0.000005 amortized. Infra—K8s, logging, etc.—is $0.0001 per booking. Total cost: just under a tenth of a cent per booking, or $0.000905."

---

### 07:10  
**Hannah:** "Is this competitive with major players?"

**Arjun:** "Yes. Booking.com’s published matching cost is $0.001–$0.002 per booking (2024 Q3 earnings call), Expedia’s internal target is $0.0015. We’re below both."

---

### 07:50  
**Mei:** "What’s the margin on this per booking, compared to manual-only routing?"

**Priya:** "Manual-only is about $0.03 per booking, mostly labor. So this is >30x cheaper for the matching component."

---

### 08:23  
**Lin:** "Are there any legal or compliance constraints on where model inference is performed?"

**Priya:** "Yes—data residency requires embeddings to be generated in-region (EU or US). OpenAI endpoints are region-pinned as per our DPA. Operator selections and audit logs are stored in the same region as the booking origin."

---

### 09:05  
**Sara:** "How do we handle a case where OpenAI’s endpoint is temporarily out of region compliance?"

**Arjun:** "Fallback path (partial_ratio) is on-prem and never leaves region. If OpenAI region drift is detected, system auto-fails over and logs the event. Legal is notified."

---

### 09:45  
**Hannah:** "What’s the SLA for region failover detection?"

**Priya:** "Monitoring pings every 90 seconds. Failover triggers within 2 minutes of anomaly detection."

---

### 10:15  
**Martin:** "How is data minimization handled on inference requests?"

**Priya:** "Only the hotel and city name fields are sent for embedding. No PII or booking details. All requests are logged and signed."

---

### 10:45  
**Lin:** "Is there a retention policy for embedding vectors?"

**Priya:** "Yes, vectors are retained only as long as the index is active (currently 90 days). After that, vectors are deleted from hot storage and archived encrypted for compliance—retention matches audit log."

---

### 11:15  
**Mei:** "Do any customers require that all model serving be done on-prem?"

**Arjun:** "One customer in the pipeline requested an on-prem-only option. We can support it with the fallback, but with lower accuracy (~0.44 top-1 vs. ~0.47). This would be a custom deployment—currently not prioritized."

---

### 11:55  
**Sara:** "Do we have an automated test to verify regionality pre-ship?"

**Priya:** "Yes, nightly geolocation test hits the endpoint and verifies region headers."

---

### 12:20  
**Hannah:** "What about vendor lock-in concerns with OpenAI?"

**Arjun:** "We keep the partial_ratio fallback and are running pilot evaluations with open-source MiniLM. Our infra is modular; model swap is a config change. Full pivot would take 2–3 weeks."

---

### 13:05  
**Martin:** "Is MiniLM accuracy close to OpenAI 3-small on our data?"

**Priya:** "No, MiniLM is about 0.3937 top-1 vs. OpenAI 3-small at 0.4687. Not competitive for production, but acceptable as a fallback."

---

### 13:30  
**Lin:** "Are there any legal reviews scheduled for new model vendors?"

**Arjun:** "Yes, each new vendor triggers a DPA and data residency review with legal and security. See ADR-004 for process."

---

### 14:00  
**Mei:** "Is fallback path costed differently?"

**Martin:** "Yes, as shown earlier, fallback is $0.0005 per call, but only 1% of traffic. If we ran fallback 100% of the time, cost would jump to $0.0005 per booking, still much lower than manual."

---

### 14:45  
**Sara:** "Is there a way to force fallback for a given customer?"

**Priya:** "Yes, config flag per customer. Requires restart, not hot-swappable. Used in one pilot so far."

---

### 15:10  
**Hannah:** "What would trigger a reconsideration of the current vendor or model?"

**Arjun:** "Any of: sustained SLO miss (accuracy or latency), a material cost increase, legal or data residency issue, or a competitive uplift—e.g., Booking.com or Expedia showing >10% improvement in public benchmarks. Also, customer request for on-prem-only would trigger review."

---

### 15:55  
**Martin:** "Do we track competitive benchmarks regularly?"

**Priya:** "Quarterly scan of public earnings calls and tech blogs. Booking.com and Expedia are the main comparables. Latest: Booking.com claims 0.72 top-3 on their data, but not apples-to-apples."

---

### 16:40  
**Lin:** "How do our lexical-overlap and no-overlap buckets compare to competitors?"

**Priya:** "We don’t have their slice data, but our 3-small model: 0.4687 top-1 overall, 0.3937 for MiniLM, partial_ratio at 0.4407, wratio at 0.4223. Lexical-overlap bucket is 1399/3000, no-overlap 1601/3000. These are stable across runs."

---

### 17:20  
**Mei:** "Do we ever share our metrics externally?"

**Arjun:** "Not currently. Only aggregate SLOs are shared with enterprise customers, never raw slice data."

---

### 17:45  
**Sara:** "Is there a runbook for competitive response?"

**Arjun:** "Yes, maintained in the playbook repo. If competitor benchmarks materially outpace us, triggers a roadmap review and options analysis."

---

### 18:10  
**Hannah:** "What volume would break our current cost model?"

**Martin:** "At 10x current volume, infra amortization drops further, OpenAI prices are tiered, so cost per booking could fall. Only bottleneck is API QPS cap—would need contract renegotiation above 100 QPS."

---

### 18:45  
**Priya:** "Our peak is 1.3 QPS, so we’re well within limits."

---

### 19:00  
**Lin:** "How is per-booking cost tracked month-over-month?"

**Priya:** "Finance pipeline ingests booking and cost logs, monthly report generated. Deviations >10% trigger a review."

---

### 19:25  
**Mei:** "Have there been any cost overruns to date?"

**Arjun:** "No overruns. All costs within forecast. Only material change would be if OpenAI reprices upward, which is covered in the contract with 90-day notice."

---

### 19:50  
**Sara:** "Is fallback ever triggered for non-technical reasons?"

**Priya:** "Rarely. Only for legal/data-residency or explicit customer request. Otherwise, only on API outage."

---

### 20:20  
**Hannah:** "Can fallback accuracy be improved?"

**Priya:** "Somewhat—tuning thresholds or switching to wratio (~0.4223 top-1) instead of partial_ratio (~0.4407), but both are well below OpenAI and add latency. Not a roadmap priority."

---

### 20:45  
**Martin:** "What’s the tradeoff between latency and accuracy in fallback?"

**Priya:** "Fallback is 2–3x slower (p95 latency ~60 ms vs. 18 ms for OpenAI), and accuracy is 5+ points lower in top-1 and top-3."

---

### 21:15  
**Lin:** "Are operators aware when fallback is active?"

**Priya:** "Yes. UI surfaces a yellow banner: 'Reduced accuracy mode — fallback in effect.' Operators are trained to escalate more aggressively in fallback windows."

---

### 21:45  
**Sara:** "Any operator complaints during fallback events?"

**Arjun:** "None logged. Last fallback event was 2025-11-18, lasted 27 minutes, no increase in escalations or feedback tickets."

---

### 22:05  
**Mei:** "How are model swaps communicated to operators?"

**Priya:** "Operator dashboard surfaces active model version. Slack notification to #ops-alerts and email to all shift leads 24 hours prior, per notice policy."

---

### 22:35  
**Martin:** "Is there a rollback plan if a new model underperforms?"

**Priya:** "Yes. Each deployment keeps the previous index and model hot for 7 days. Rollback is a config flip, takes under 5 minutes."

---

### 23:00  
**Hannah:** "Has rollback ever been needed?"

**Priya:** "Not since launch. Shadow testing pre-ship caught a regression in September, but no live rollback required."

---

### 23:25  
**Lin:** "How is operator feedback incorporated into future model versions?"

**Priya:** "Feedback is logged and reviewed monthly. If >10 overrides or >5 escalations on the same hotel/city pair, triggers a triage session and possible GT update."

---

### 23:50  
**Sara:** "Are operators incentivized to provide feedback?"

**Arjun:** "No formal incentives. Feedback is optional, but we maintain a low-friction UI and keep the loop tight."

---

### 24:15  
**Mei:** "Is there a threshold for reconsidering the model approach entirely?"

**Arjun:** "If top-3 accuracy drops >5 pp for two months, or operator-selection-rate-at-3 falls below 0.45 for a month, triggers a full review and model bakeoff."

---

### 24:40  
**Martin:** "How frequently are test sets refreshed?"

**Priya:** "Monthly, with a fresh 5k random sample, stratified by overlap and city frequency axes."

---

### 25:10  
**Hannah:** "How do you ensure the test set is representative?"

**Priya:** "Sampling is stratified by lexical overlap (direct, partial, none) and city frequency (common, rare). Ratios match live traffic distributions."

---

### 25:40  
**Lin:** "Is there a risk of overfitting to the test set?"

**Priya:** "Low risk. We rotate the test set monthly, and no model is fine-tuned on test data. If we move to in-house models, will implement holdout splits."

---

### 26:10  
**Sara:** "How are edge cases—like hotels with special characters—handled?"

**Priya:** "Model normalizes Unicode, strips punctuation, and lowercases prior to embedding. Test set includes >50 such edge cases; all pass regression."

---

### 26:40  
**Mei:** "Are we tracking operator dwell time per booking?"

**Priya:** "Yes. Median dwell time is 12.6 seconds, unchanged post-ship."

---

### 27:05  
**Martin:** "Any plans to reduce dwell time further?"

**Priya:** "Not a current priority. Our median is already below industry average (Expedia last reported 15.8s)."

---

### 27:30  
**Hannah:** "What’s the process if a hotel is missing from the index?"

**Priya:** "Operator escalates, triage tags the missing hotel, and it’s patched in the next nightly index refresh."

---

### 27:55  
**Lin:** "Is there a way to track operator errors (e.g., selecting a wrong hotel)?"

**Priya:** "Yes, if an override occurs on a previously accepted booking, it’s flagged as a possible operator error and reviewed."

---

### 28:15  
**Sara:** "How many such operator errors do we see?"

**Priya:** "Less than 0.05% of bookings. All are reviewed biweekly."

---

### 28:30  
**Mei:** "How do you ensure index freshness?"

**Priya:** "Nightly full refresh from source system. Any deltas are patched within 24 hours. Jobs are monitored and alert on failure."

---

### 28:55  
**Martin:** "Is index bloat a concern as the dataset grows?"

**Priya:** "Currently, 3000 hotels is <1 MB index. Projected 10x growth still fits comfortably in RAM. We’ll re-evaluate at >50k hotels."

---

### 29:15  
**Hannah:** "Do operators ever see duplicate candidates?"

**Priya:** "No, deduplication logic merges exact name/address matches, surfaces unique candidates only."

---

### 29:40  
**Lin:** "Are there UI accessibility features for operators?"

**Priya:** "Yes—screen reader support, high-contrast mode, and keyboard navigation. Quarterly accessibility audit runs."

---

### 29:55  
**Sara:** "Have there been any accessibility complaints?"

**Arjun:** "None in the past year."

---

### 30:10  
**Mei:** "What’s the process for adding a new stratification axis to analytics?"

**Priya:** "Proposal in analytics repo, reviewed by Arjun and me. Schema changes merged, next month’s dashboard picks up the new axis."

---

### 30:35  
**Martin:** "Can operators request new axes?"

**Priya:** "Yes, via #operator-feedback. No requests yet."

---

### 30:50  
**Hannah:** "Is operator training updated when new axes are added?"

**Priya:** "Yes, but since axes are analytics-only, operator training is unaffected unless UI changes."

---

### 31:10  
**Lin:** "Are there privacy risks with new axes?"

**Priya:** "No, axes use only hotel and city metadata, never PII."

---

### 31:25  
**Sara:** "How are operator shift handoffs handled with respect to model changes?"

**Arjun:** "Shift lead receives model status at start of shift. Any mid-shift model changes are pushed to dashboard and Slack."

---

### 31:45  
**Mei:** "Is there a published SLO for operator escalation rate?"

**Priya:** "Yes, target is <7% escalations per booking. We’re at 6.3% rolling average."

---

### 32:05  
**Martin:** "What’s the mean time to resolve escalations?"

**Priya:** "Median is 1.8 hours, 90th percentile at 6 hours."

---

### 32:25  
**Hannah:** "Are escalations ever automated—e.g., for obvious index misses?"

**Priya:** "No, all escalations require human confirmation for audit purposes."

---

### 32:45  
**Lin:** "Is there a quarterly audit of escalation reasons?"

**Priya:** "Yes, Lin and I review random samples quarterly. Summary filed in governance repo."

---

### 33:05  
**Sara:** "How do you handle operator turnover?"

**Arjun:** "New operators complete onboarding, including live model walkthrough and escalation training. Shadowing for first 10 shifts is required."

---

### 33:25  
**Mei:** "What’s the operator headcount today?"

**Arjun:** "12 FTEs, covering 24/7 in three shifts."

---

### 33:40  
**Martin:** "Is operator feedback volume changing?"

**Priya:** "Stable at 2–3 feedbacks per 1000 bookings."

---

### 33:55  
**Hannah:** "Any plans for operator performance incentives?"

**Arjun:** "Not planned. Focus is on process quality, not speed or volume."

---

### 34:15  
**Lin:** "Do operators have access to raw model output?"

**Priya:** "No, only candidate names, city, and chain are exposed. Embedding vectors are never shown."

---

### 34:30  
**Sara:** "Are there privacy controls on operator audit logs?"

**Priya:** "Yes, access is RBAC, logs are encrypted at rest, and all access is logged."

---

### 34:45  
**Mei:** "How are GDPR data deletion requests handled?"

**Priya:** "Booking and audit logs are scrubbed within 30 days of request. Operators can flag bookings for deletion, which triggers the workflow."

---

### 35:05  
**Martin:** "Are all data flows documented for compliance?"

**Priya:** "Yes. Data flow diagrams are updated quarterly and reviewed by Lin."

---

### 35:25  
**Hannah:** "Any upcoming regulatory changes we’re tracking?"

**Lin:** "EU AI Act may require additional model explainability disclosures. We’re monitoring, but no immediate changes needed."

---

### 35:45  
**Sara:** "Is there a plan for operator retraining if regulations change?"

**Arjun:** "Yes, retraining is scheduled as needed, with updated documentation and test cases."

---

### 36:05  
**Mei:** "Are there model explainability features for operators?"

**Priya:** "Not yet. Operators see candidate rank and metadata only. Explainability for end-users is on the roadmap, pending regulatory requirements."

---

### 36:25  
**Martin:** "Is there a mechanism for operators to flag suspicious matches?"

**Priya:** "Yes, 'Flag for review' button in UI logs the case for triage. Used <0.1% of bookings."

---

### 36:45  
**Hannah:** "Any recurring themes in flagged cases?"

**Priya:** "Mostly city/country ambiguities and rare spelling variants. No systemic issues."

---

### 37:05  
**Lin:** "How often are flagged cases resolved?"

**Priya:** "All within 48 hours, median is 18 hours."

---

### 37:20  
**Sara:** "Are model updates ever scheduled to avoid peak operator shifts?"

**Priya:** "Yes, all deployments are scheduled for 2–3am UTC, lowest traffic window."

---

### 37:40  
**Mei:** "How is test coverage maintained as flows evolve?"

**Priya:** "Regression suite is updated with new edge cases quarterly. All code changes require passing the full suite before merge."

---

### 38:00  
**Martin:** "Is there a process for shadow testing new models?"

**Priya:** "Yes, shadow A/B runs for 48 hours pre-ship. Results are reviewed before cutover."

---

### 38:20  
**Hannah:** "Are operators ever involved in shadow testing?"

**Priya:** "Not directly. Shadow tests are internal, but flagged edge cases are reviewed with operators as needed."

---

### 38:40  
**Lin:** "Is there an audit log of shadow test results?"

**Priya:** "Yes, all shadow test metrics and diffs are archived for compliance."

---

### 39:00  
**Sara:** "How are operators notified of shadow test findings?"

**Arjun:** "Summary email and dashboard update if material issues found, otherwise no action needed."

---

### 39:20  
**Mei:** "Is there a process for retiring old test cases?"

**Priya:** "Yes, test cases not triggered for 12 months are archived after review."

---

### 39:40  
**Martin:** "Do we ever re-use operator-flagged cases as future test cases?"

**Priya:** "Yes, all valid flagged cases are considered for inclusion in edge-case regression suite."

---

### 40:00  
**Hannah:** "What’s the process for updating chain-KB sources?"

**Priya:** "Chain-KB is refreshed weekly. Vendor updates are versioned; audit log records all changes."

---

### 40:20  
**Lin:** "Is vendor data ever validated by operators?"

**Priya:** "Rarely, but operators can flag vendor data mismatches, which triggers a review with the vendor."

---

### 40:40  
**Sara:** "Are there SLAs with vendors for KB correction?"

**Arjun:** "Yes, 48-hour SLA for critical corrections, one week for standard issues."

---

### 41:00  
**Mei:** "How are new chains added to the KB?"

**Priya:** "Product files a request, vendor provides data, validated by data eng and sample checks by operators before go-live."

---

### 41:20  
**Martin:** "Is there a process for rolling back KB updates?"

**Priya:** "Yes, previous version is kept hot for 30 days. Rollback is a config change."

---

### 41:40  
**Hannah:** "Do operators ever see changes in KB mid-shift?"

**Priya:** "No—KB updates go live off-shift with a dashboard banner if any material changes."

---

### 42:00  
**Lin:** "Are KB updates versioned for audit?"

**Priya:** "Yes, all KB updates are logged with timestamp, vendor, and hash."

---

### 42:15  
**Sara:** "Has a vendor ever missed their SLA?"

**Arjun:** "Once, in October 2025, by 12 hours. No operational impact."

---

### 42:30  
**Mei:** "Are there plans to benchmark KB data quality?"

**Priya:** "Yes, annual benchmarking planned against internal ground truth and competitor samples."

---

### 42:50  
**Martin:** "Will operators be surveyed on KB quality?"

**Priya:** "Yes, survey pilot planned for Q2."

---

### 43:10  
**Hannah:** "Are there chain-specific escalation patterns?"

**Priya:** "No persistent patterns; chain escalations are <1 pp variance across the board."

---

### 43:30  
**Lin:** "Are there plans to publish any of this data for industry benchmarking?"

**Arjun:** "Not at this time. Internal use only, unless leadership changes policy."

---

### 43:50  
**Sara:** "Any upcoming risks or process changes?"

**Arjun:** "No new risks. Only pending changes: 3-large pilot, chain-KB vendor rollout, and city_frequency axis."

---

### 44:10  
**Mei:** "When is the next full review?"

**Arjun:** "Quarterly cadence; next is scheduled for March 2026."

---

### 44:30  
**Martin:** "Who owns action items?"

**Arjun:** "See end-of-minutes summary: Priya on A/B, Ravi on vendor, me on roadmap."

---

### 44:45  
**Hannah:** "Any final questions?"

**(No further questions)**

---

**Adjourned at 45:00.**

---

## Appendix E — stakeholder memos collated

---

**From:** Hannah (PM)  
**Date:** 2025-12-12  
**Subject:** Follow-up on operator experience and dashboard usability

I wanted to thank the team for the thorough post-ship review and the clear communication around metrics and upcoming features. As we move forward with the next dashboard iteration, I’d like to propose a short usability study with a sample of operators who interact with the feedback UI and weekly dashboard. Several frontline staff have surfaced (via informal Slack feedback) that while the top-level status indicators are clear, it’s not always easy to understand what triggers escalation or how the system’s confidence relates to their daily workflow.  

Could we schedule a session with 4–5 operators to observe their usage and collect qualitative feedback? I think this will help prioritize both the dashboard drill-down work and any UI tweaks for mismatch flagging. Please let me know if I can assist with recruiting participants or drafting consent forms.  

Thanks again for everyone’s hard work—metrics stability is a huge win.

---

**From:** Martin (PM)  
**Date:** 2025-12-13  
**Subject:** Proposal: Early alerting for rare slice regressions

Echoing Hannah’s positive feedback on the dashboard rollout. I have a suggestion for the stratified metrics monitoring: as we plan to expand to new axes (city_frequency, chain), would it be feasible to implement early warning thresholds for statistically significant drops in rare buckets (e.g., rare cities, no-overlap cases)?  

Currently, only sharp overall drops trigger alerts. But my concern is that for small slices, even a 2–3 pp swing could indicate an emerging issue that would be masked at the aggregate level. If we can set dynamic thresholds based on bucket size and historical variance, we’d catch regressions before they impact overall operator experience.  

Happy to work with Priya on defining the math for this. Please advise if this should be prioritized for the February dashboard update.

---

**From:** Lin (Governance)  
**Date:** 2025-12-13  
**Subject:** Audit compliance and governance notes

Thank you for surfacing the audit log export and model versioning updates. After reviewing the new dashboard and weekly archives, I am satisfied that we meet internal audit trail requirements under current policy. However, with the upcoming chain-KB vendor integration and potential expansion to multi-language support, I’d like to flag two governance items for Q1:  

1. **Data residency:** Please confirm, before pilot, that all candidate vendors can certify on-prem or EU-only data storage—this is a blocker for signoff.  
2. **Material change notification:** Multi-language rollout will require a formal governance review, including updated privacy and risk analyses. Please loop me in early on any prototype work here.  

No other blockers at this time. Thanks for the continued rigor.

---

**From:** Finance Director (A. Mukherjee)  
**Date:** 2025-12-14  
**Subject:** Cost tracking and 3-large migration planning

Appreciate the clarity in the Q1 planning notes, particularly on cloud credit usage and cost containment for the 3-large re-embed. One ask as we approach the pilot: can we get a detailed projection of incremental compute/storage spend (broken down by one-time and ongoing), and a sensitivity analysis if we need to re-embed more frequently?  

Additionally, if the chain-KB integration or multi-language support introduces new recurring license or infrastructure costs, please flag these at least one quarter prior to go-live. I’d like to avoid any surprises in the FY26 budget cycle.  

Please route spend reports through finance-ops@; happy to review draft projections or attend a planning call if helpful.

---

**From:** Ops Director (N. Bhandari)  
**Date:** 2025-12-14  
**Subject:** Operator feedback loop and escalation review

Thank you for maintaining stability in operator workload and response times post-ship. Our team appreciates the continued access to feedback channels and the clear communication on changes.  

One area for improvement: can we get more granular analytics on escalation triggers by chain and city? While no specific chain has shown >1 pp change, occasional spikes in high-volume cities lead to temporary staffing strain. A breakdown of escalation rates by city (weekly/biweekly) would help us forecast and adjust resourcing.  

Additionally, thanks for logging the request for richer mismatch details in the UI. Please include Ops in the Q1 triage—our leads have concrete ideas for what information would be actionable for the frontline.

---

**From:** Legal Counsel (S. Gomez)  
**Date:** 2025-12-15  
**Subject:** Legal review: data retention, audit, and vendor onboarding

Following the last review, I have no objections to the current data retention and audit practices, provided GDPR deletion requests continue to be fulfilled within 30 days. Please ensure that when onboarding any new chain-KB vendor, a privacy impact assessment (PIA) is completed prior to contract.  

On multi-language or per-customer logic: both would trigger additional privacy and cross-border data flow considerations. Please notify Legal if/when these are prioritized, so that we can review updated data processing agreements (DPAs) and ensure our compliance statements remain accurate for external clients.  

Let me know if further review is needed for any upcoming dashboard features exposing more granular data.

---

**From:** Priya (Analytics/Stratified)  
**Date:** 2025-12-15  
**Subject:** Action items and clarifications: stratification and rare slices

Thanks to everyone for the constructive feedback. On the requests for more granular slice reporting and rare-bucket alerts—these are both in active planning. For the February dashboard update, my proposal is:  
- Add per-city and per-chain escalation breakdowns, with weekly and biweekly rollups;  
- Implement dynamic alerting for rare slices using a Z-score threshold (details to be shared for review);  
- Surface operator feedback stats by city/chain to support Ops’ staffing forecast.  

Martin and Hannah, I’ll reach out to co-design the alerting logic and usability study, respectively. Lin, I’ll confirm data residency for chain-KB vendors pre-pilot. Targeting a full update on these items by end of January.

---

**From:** Ravi (Data Eng)  
**Date:** 2025-12-15  
**Subject:** Data pipeline resilience and vendor onboarding

Echoing the positive momentum—so far, nightly ingestion and model index refreshes continue without issue. To further bulletproof our chain-KB integration, I’m drafting a resilience plan for the new vendor ingestion: automated pipeline health checks, SLA monitoring, and rollback on schema drift.  

I’m also confirming with each vendor that we can enforce EU-only storage at the infrastructure level (per Lin’s note) and that audit logging meets our internal requirements. Will circulate the draft evaluation matrix by 12/20 and schedule a security review as soon as Vendor Y reference checks are complete.

---

**From:** Sara (Ops liaison)  
**Date:** 2025-12-16  
**Subject:** Operator training and change management proposal

Appreciate the stable metrics and transparent incident handling. For the upcoming fuzzy routing A/B and dashboard enhancements, could we prepare a brief operator training (video or slides) summarizing what’s changing, what to look for in the UI, and how to provide feedback?  

Past launches have shown operators adapt fastest when they have clear, succinct materials ahead of time. I’m happy to coordinate content review with Ops leads and ensure all shifts are covered. Please let me know the planned launch dates for any UI-impacting changes so we can align communications.

---

**From:** Mei (Advisor, Guest)  
**Date:** 2025-12-16  
**Subject:** Observations and open research questions

Congrats to the team on a smooth post-ship, especially for a system with this scale and complexity. As you move toward stratified reporting and the 3-large evaluation, I’d encourage consideration of two open items:  
1. **Long-tail performance:** Are we tracking model accuracy for very rare city/hotel pairs (<5 appearances in GT)? Even if not actionable now, logging accuracy and operator selection rates here could inform future model or index improvements.  
2. **Operator-model interaction:** For cases where repeated operator overrides occur (flagged after 10 instances), is there an opportunity to collect qualitative context (e.g., why the override was chosen)? This could feed back into both model evaluation and operator training.

Happy to discuss further or assist with study design if helpful.

---

**From:** Arjun (Lead)  
**Date:** 2025-12-16  
**Subject:** Next steps and leadership alignment

Thank you all for the thoughtful memos and ongoing engagement. We’ll prioritize the following in Q1:  
- Drill-down dashboard features and rare-slice alerting (Priya/Martin/Hannah)  
- Chain-KB vendor selection and data residency compliance (Ravi/Lin)  
- Operator feedback UI enhancements and training (Sara/Ops)  
- Cost and legal review ahead of 3-large and multi-language pilots (Finance/Legal)

Please route additional questions to #ml-matching or direct to me. Appreciate everyone’s commitment to transparency and operational excellence—let’s keep up the momentum.

---

---

## Appendix F — retro of the leadership review itself

**Date:** 2025-11-12  
**Duration:** 40 min  
**Attendees:** Arjun (lead), Priya, Lin, Hannah, Martin, Mei (observer), Ravi (notes)

---

### 1. What went well

**a. Slide-deck clarity and structure**  
- The deck was concise: 15 slides, minimal text, well-organized by agenda section (metrics, ops, roadmap, Q&A).
- Each metric was annotated with its canonical number and SLO, making it easy for attendees to track.
- Priya’s use of color-coded highlights (green/yellow/red) for each metric enabled rapid status scanning, which Lin called out as especially helpful for governance review.
- The inclusion of “slice” breakdowns (lexical-overlap, no-overlap) on separate slides worked well, reducing cognitive load vs. previous stacked tables.
- Each open question (e.g., 3-large, chain-KB) had a dedicated slide with status, risks, and next actions, which helped avoid off-track discussions.

**b. Number-citation handoffs**  
- All canonical numbers (hotel set size, top-1 and top-3 accuracy for each model, slice counts) were referenced directly on slides and verbally when discussed.
- Priya’s habit of stating “per the dashboard, as of 2025-11-06, top-3 is 0.531” prevented confusion over which time range or data cut was being cited.
- Martin and Hannah both commented that this made Q&A crisper, especially for tracking improvement or regression.
- The “metrics cheat sheet” in the appendix (slide 14) was used several times for quick lookups, and was cited by Arjun as a key time-saver.
- No instances of number mismatches or ambiguity were noted during or after the meeting.

**c. Handling of the 3-large question**  
- The approach to the 3-large uplift question was clear and well-defended:
    - Priya presented the current status (pending infra, no new numbers since last eval), and the canonical (unverified) 0.6981 top-1 figure was explicitly called out as “unverified, not yet re-benchmarked.”
    - Risks and gating conditions for Q1 rerun were articulated, and Arjun summarized next steps concisely (“only proceed if uplift >0.55 top-1”).
    - Martin appreciated that prior contractor (Jordan) contributions on the 3-large eval were referenced with caveats, avoiding overclaiming.
    - The team agreed that surfacing the uncertainty around 3-large’s uplift and not committing prematurely was the right call, maintaining trust with stakeholders.

**d. Meeting logistics and flow**  
- The meeting stayed on schedule, covering all agenda items in the allotted 50 minutes.
- Q&A was brisk, with a clear turn-taking protocol (hand-raise in Zoom, tracked by Arjun).
- Real-time note-taking in the shared doc (Ravi) reduced post-meeting drift and allowed for live clarifications.
- All action items were captured and assigned before adjournment.

---

### 2. What went poorly

**a. Pre-briefing for Hannah**  
- Hannah joined as a new PM and was not pre-briefed on slice metric definitions or dashboard workflow.
    - This led to confusion on one Q (“Is the slice metric tracked weekly?”), requiring Priya to backtrack and explain dashboard mechanics mid-meeting.
    - Several clarifications had to be provided in-thread and in follow-up DMs.
- Team consensus: We should have scheduled a 15-minute pre-meeting orientation for new participants, especially those expected to engage in metrics or operational Q&A.

**b. Deck level of detail in escalation scenarios**  
- The escalation process (SLO breach, operator override flagging) was summarized at a high level on one slide.
    - When Lin asked for specifics on time-to-mitigation and notification paths, the slide did not have granular flows, requiring Arjun to pull up the wiki in real time.
    - Feedback: Next time, more detailed escalation flow diagrams or swimlanes should be included for complex operational topics.

**c. Handling of open questions in Q&A**  
- Two roadmap items (city-frequency axis, chain-KB eval) had “TBD” for precise dates and lacked interim milestones.
    - Martin noted this made it harder to assess risk or follow up post-meeting.
    - In the post-mortem, Priya suggested always providing next concrete check-in dates, even if only for sub-tasks.

**d. Number-citation in side discussions**  
- In two instances (wratio and partial_ratio fallback numbers), canonical figures were not immediately cited, leading to a brief “which number?” back-and-forth.
    - While minor, this was flagged by Mei as a potential source of confusion for external observers or late joiners.

---

### 3. What we’d change for the next review

**a. Pre-briefing and onboarding**  
- All first-time attendees (esp. PMs and ops) will receive a 10–15 minute walkthrough of:
    - Metrics and their canonical numbers (with definitions)
    - Dashboard navigation and data refresh cadence
    - Key SLOs and escalation processes
- Priya to own pre-brief content, with slides and one-pager, scheduled at least 2 days before the main review.

**b. Deck improvements**  
- Each operational process (e.g., escalation, operator override review, model swap protocol) will have a dedicated process flow slide with timeline and notification paths.
- “What changed since last review” summary slide to be added at the top for rapid context.
- All canonical numbers to be footnoted with data source and as-of date to prevent ambiguity.

**c. Q&A prep and facilitation**  
- Pre-collect questions from attendees, especially new joiners, to seed the Q&A and reduce on-the-spot confusion.
- Assign a timekeeper (Ravi or Martin) to keep Q&A from overrunning.
- Clarify in the invite that all open roadmap items should have next milestone dates or check-in points, even if tentative.

**d. Documentation and follow-up**  
- Real-time note-taking to continue; summary and action items to be circulated within 24 hours.
- All action items to be tracked in the team project board, with owners and due dates.
- Post-meeting, open a thread in #ml-leadership for asynchronous clarifications, so off-line attendees can follow up without delay.

---

### 4. Additional feedback (verbatim, lightly edited)

**Martin:**  
> “This was the smoothest review yet — metrics were clear, handoffs were crisp. Only bump was when we had to clarify fallback numbers. A pre-meeting cheat sheet for new PMs would help.”

**Hannah:**  
> “I felt a bit lost on the slices and dashboard UI—would have benefited from a quick primer. Otherwise, the meeting was fast and focused.”

**Lin:**  
> “Governance questions landed well, but escalation flow could be more visual. Would like to see more process swimlanes next time.”

**Priya:**  
> “Metrics story is tight. Next time, I’ll prep a one-pager for new folks, and make sure slice definitions are in the deck.”

**Ravi:**  
> “Live doc editing helped a lot. Only suggestion is to have next steps/milestones more visible in the deck.”

---

### 5. Summary

The leadership review met its objectives: clear status on metrics, no ambiguity on key numbers, operational topics covered, and action items captured. The principal issue was onboarding new attendees—addressable via pre-briefs and more explicit documentation. Deck improvements (process flows, footnoted numbers, milestone tracking) are planned for the next review. Overall, the team’s preparation and structure were strong, with only minor process gaps.

**Action items for next review:**
- Schedule pre-brief for new attendees (Priya, 2 days prior)
- Add process flow slides for escalation and operator review (Arjun)
- Footnote all canonical numbers with source/date (Priya)
- Ensure all open items have next milestone/check-in (Martin)
- Timekeeper assigned for Q&A (Ravi)

**Retrospective adjourned.**

---

## Appendix G — 30-day post-ship review minutes (2025-12-10)

**Date:** 2025-12-10  
**Duration:** 65 min  
**Attendees:** Arjun (chair), Priya (metrics/stratified), Lin (governance), Hannah (PM), Martin (PM), Mei (guest), Sara (Ops liaison), Ravi (Data Eng), Finance observer (Jill)

---

### 1. Opening and agenda check

**Arjun:** "Welcome, everyone. Main topics today: cost tracking post-ship, operator-selection-rate trend, latency tails, drift-dashboard rollout status, chain-KB vendor evaluation, and key Q1 go/no-go decisions. Jill joins us to clarify finance questions."

---

### 2. Cost tracking — model, infra, ops (Jill, Priya, 10 min)

- **Model API costs** (OpenAI 3-small):  
  - **November actual:** $1,329 (proj. $1,314, variance +1.1%)  
  - **Dec 1-7:** $342 (on pace for $1,488, seasonal uptick due to Black Friday/Cyber Monday surge).

- **Cloud infra:**  
  - **Compute:** $397 (Nov); no abnormal spikes.  
  - **Storage:** $44 (steady, includes embedding index and audit logs).

- **Ops workload:**  
  - **Manual escalation hours:** 21.5 hrs/week, down 7% from pre-ship (23.1 hrs/week).  
  - **Operator FTE:** No change; no OT requested.

**Transcript excerpt:**

**Jill:** "Finance sees expected seasonality, nothing out of bounds. If Q1 traffic grows >20%, we’ll revisit the model quota."

**Priya:** "No model overage risk at current QPS. Storage well within budget, even with full audit log retention."

**Sara:** "Operators report no overwork; coverage remains stable through holidays."

---

### 3. Operator-selection-rate drift analysis (Priya, Sara, 15 min)

**Priya:**  
- "Reviewed selection-rate-at-3 by week and by stratification (lexical-overlap, no-overlap, city_frequency draft).  
- **Overall:**  
  - Pre-ship baseline: 0.527  
  - Weeks 1–4 post-ship: 0.528, 0.531, 0.533, 0.529 (mean 0.530)  
  - No statistically significant drift (p > 0.3, chi-squared across weeks).

- **By bucket:**  
  - *Lexical-overlap:* 0.697, 0.695, 0.697, 0.698  
  - *No-overlap:* 0.422, 0.421, 0.424, 0.423  
  - No bucket exceeds ±0.3 pp week-over-week variance.

- **Escalation rate:**  
  - Down to 0.082 from 0.087; lowest for chain hotels (0.031), highest for rare city/no-overlap (0.156).  
  - All within pre-ship projections.

**Sara:**  
- "No operator-reported perception of drift or confusion. Steady confidence in top-3 suggestions. No increase in 'other' or 'escalate' button usage."

**Martin:**  
- "Any signal of learning curve effects with new operators hired post-ship?"

**Sara:**  
- "Onboarding data shows no drop-off; new hires match cohort average within first week."

---

### 4. Latency tail and infra health check (Ravi, Priya, 10 min)

**Priya:**  
- "Examined p95 and p99 latencies, 5k random sample per week:  
  - **p95:** 18.2 ms avg (range 17.7–19.0)  
  - **p99:** 24.5 ms avg (range 23.8–25.6)  
  - No SLO breaches (p99 SLO ≤30 ms).  
  - All sub-buckets (by city, chain, overlap) <2 ms spread from median.

- **API errors:** 0.07% (all transient, retried successfully).  
- **Fallback activation:** 0 events.

**Ravi:**  
- "Infra steady. Nightly embedding index refresh, no lag or ingest stalls.  
  - CPU/mem utilization: 38%/29% average (well below alert thresholds).  
  - No alert spikes during traffic surges."

**Hannah:**  
- "Is there any evidence of tail latency clustering (e.g., by city or chain)?"

**Priya:**  
- "No; tail events distributed randomly, no city/chain correlation."

---

### 5. Drift-dashboard rollout and audit trail (Priya, Lin, 10 min)

**Priya:**  
- "Drift dashboard now live:  
  - **Features:** Per-week and per-bucket accuracy, operator-selection trends, escalation, latency, and deployment audit.  
  - **Exports:** CSV/JSON per period; exposed via `/metrics` endpoint for automation.

- **Audit log samples:**  
  - Model version, embedding index checksum, config snapshot, SLO status, operator override logs.

- **Upcoming:**  
  - Filters by city_frequency and chain (ETA: Feb).  
  - Alert surfacing for >2pp week-over-week bucket drop.

**Lin:**  
- "Dashboard output matches governance requirements. Export tested; hashes and timestamps align to deployment events. Audit retention verified at 180 days, with GDPR-compliant erase flow."

**Martin:**  
- "Can dashboard support custom slice alerts for high-risk cities?"

**Priya:**  
- "Planned for Q2; will allow alert rule input by stratification axis."

---

### 6. Chain-KB vendor evaluation progress (Arjun, Ravi, 10 min)

**Arjun:**  
- "Shortlist remains Vendors X, Y, Z.  
  - **Vendor Y:** Passed security review; robust audit, on-prem support, 24hr update SLA, and strong references.  
  - **Vendor X:** Reference check scheduled 2025-12-14.  
  - **Vendor Z:** Lacks full audit trail, deprioritized.

- **Evaluation matrix:**  
  - Scored on integration, audit, SLA, price, support.  
  - Vendor Y currently leads (see Table G-1)."

**Ravi:**  
- "Vendor Y’s KB ingest demo succeeded; integrated in staging, full sync in 24 min, checksums match."

**Jill:**  
- "Finance: Vendor Y’s pricing in line with budget. Allows for 2x scale if needed."

**Table G-1: Chain-KB Vendor Evaluation Summary**

| Vendor   | Integration | Audit Trail | On-prem | Update SLA | Reference | Price | Score (0-5) |
|----------|-------------|-------------|---------|------------|-----------|-------|-------------|
| Vendor X | Pending     | Y           | Y       | 48hr       | Pending   | $$    | 3.1         |
| Vendor Y | Pass        | Y           | Y       | 24hr       | Pass      | $$    | 4.7         |
| Vendor Z | Partial     | N           | Y       | 72hr       | N/A       | $     | 2.8         |

---

### 7. Open Q1 decisions and risk tracking (Arjun, Priya, Lin, 10 min)

- **3-large re-embed:**  
  - Blocked on cloud credits (ETA 2025-12-20).  
  - If uplift >0.55 top-1 (vs. 0.4687 for 3-small), will brief governance and schedule Q2 shadow/ship.

- **City_frequency axis:**  
  - Data extraction starts 2026-01-14; full stratified dashboard live by 2026-02-10.

- **Fuzzy routing A/B:**  
  - Launches 2025-12-15. Results will determine if partial_ratio fallback remains in cold path, or is demoted.

- **Chain-KB pilot:**  
  - Target: Vendor Y, contract review in progress, pilot integration cutover by 2026-01-31.

- **Risks:**  
  - No SLO or cost overruns YTD.  
  - Seasonal surge handled without strain.

- **Contingencies:**  
  - If operator-selection-rate drifts >3 pp or latency SLO breached, will trigger rollback and full incident review.

**Lin:**  
- "Remind: material changes (model, index, KB) require 24hr operator notice and audit log update."

**Mei:**  
- "Any sign of edge-case regressions post-ship?"

**Priya:**  
- "Edge-case regression suite (test_cases_edge.json) passes. No operator-reported outliers since ship."

---

### 8. Q&A (abridged, key highlights)

**Martin:** "What’s the expected impact if Vendor Y’s SLA is missed in a KB update window?"

**Arjun:** "We’ll default to prior KB version; operators alerted in UI. Audit log records the incident. No impact to matching logic—risk is only on newly opened hotels."

**Hannah:** "Do we see any budget risk if Q1 traffic exceeds plan?"

**Jill:** "With current cost structure, we have 20% headroom. If exceeded, will flag with 2-week lead time to adjust OpenAI quota."

**Sara:** "Has operator training changed post-ship?"

**Sara:** "No—training materials updated for UI tweaks, but core workflow unchanged."

**Priya:** "Any operator feedback on the drift dashboard?"

**Sara:** "Operators don’t access dashboard directly; summary stats sent weekly. No negative feedback."

**Lin:** "Is model selection logic (3-small vs. 3-large) fully auditable?"

**Priya:** "Yes—model version, hash, deployment timestamp, and config are in the audit log. Satisfies governance requirements."

---

### 9. Action items and closing

- **Priya:** Continue weekly dashboard monitoring, flag any drift >1.5 pp in any bucket.
- **Ravi:** Complete Vendor X reference check by 2025-12-14.
- **Arjun:** Draft chain-KB pilot cutover plan, circulate before 2026-01-10.
- **Lin:** Confirm export and retention compliance for all new audit log fields by 2025-12-20.
- **Martin/Hannah:** Run Q1 risk review after A/B results and city_frequency axis live (target 2026-02-15).

**Next full review:** 2026-01-31 (tentative, may escalate if any SLO or cost incident).

**Adjourned.**

---

## Appendix H — 60-day post-ship review minutes (2026-01-12)

**Date:** 2026-01-12  
**Duration:** 65 min  
**Attendees:** Arjun (chair), Priya, Lin, Hannah, Martin, Mei (guest), Sara (Ops), Ravi (Data Eng), Martin (PM), Jordan (observer, 3-large context)

---

### 1. A/B Fuzzy Routing Experiment — Results & Actions (Priya, 12 min)

**Overview:**  
Priya presented the completed fuzzy routing A/B test (launched 2025-12-15, 50/50 booking-level assignment, N=11,042 bookings, stratified by city and chain).

**Findings:**  
- **Top-3 selection rate:**  
  - *Control (direct)*: 0.529  
  - *Test (fuzzy routing first)*: 0.534  
  - *Delta*: +0.005 (CI: [0.001, 0.009]), statistically significant (p < 0.03)
- **Escalation rate:**  
  - *Control*: 0.079  
  - *Test*: 0.073  
  - *Delta*: –0.006  
- **Latency impact:**  
  - Median +1.2 ms (95% of queries unaffected; 99th percentile increase 2.3 ms)
- **Operator feedback:**  
  - No meaningful difference in manual review load.
  - No spike in “not found” flags.

**Discussion:**  
- **Martin:** "Is the uplift consistent across slices?"  
  - **Priya:** "Greatest in no-overlap cases (+0.009 top-3); lexical-overlap flat (+0.002). Name_length >18 chars shows outsized gain (+0.012)."
- **Sara:** "Any negative operator sentiment?"  
  - **Sara:** "No ticket increase, neutral comments."

**Actions:**  
- **Arjun:** "Recommend full rollout, with continued monitoring for city/chain outliers."  
- **Lin:** "Log as material change per governance; update ADR-002."  
  - **Priya:** "Will file ADR update and coordinate config rollout by 2026-01-15."

---

### 2. Name_Length Axis — Initial Stratified Metrics (Priya, 10 min)

**Background:**  
First full run of metrics for the new `name_length` stratification axis (cut: ≤10, 11–18, >18 chars). Extraction and bucketing complete on 3k-hotel eval set.

**Results Table:**

| Name_Length Bucket | N (eval) | Top-3 Accuracy | Top-1 Accuracy | Escalation Rate |
|--------------------|----------|----------------|----------------|-----------------|
| ≤10                | 763      | 0.564          | 0.413          | 0.061           |
| 11–18              | 1,462    | 0.523          | 0.392          | 0.079           |
| >18                | 775      | 0.498          | 0.372          | 0.096           |

- **Observations:**  
  - Accuracy and selection-rate-at-3 drop as name length increases.
  - Escalation rate rises with length, but remains within SLO.
  - Outlier: One long-name cluster (>24 chars, 57 cases) had top-3 at 0.478.
- **Discussion:**  
  - **Mei:** "Any correlation with city_frequency?"  
    - **Priya:** "Not direct, but rare cities + long names have lowest selection rates."
  - **Hannah:** "Will we drill down by both axes next quarter?"  
    - **Priya:** "Yes, dashboard work in progress for axis cross-tabs."

**Action:**  
- **Priya:** "Will publish stratified dashboards for operator review by 2026-02-05."

---

### 3. 3-Large Re-Embed — Go/No-Go & Decision Criteria (Arjun, Priya, Jordan, 10 min)

**Status:**  
- Cloud credits approved (2025-12-21).  
- 3-large embedding job (v20260108) completed on 3k-hotel set, compared head-to-head with 3-small.

**Metrics Table:**

| Metric         | 3-Small (prod) | 3-Large (candidate) | Delta |
|----------------|----------------|---------------------|-------|
| Top-1 Accuracy | 0.4687         | 0.6912              | +0.2225 |
| Top-3 Accuracy | 0.529          | 0.734               | +0.205 |
| P95 Latency    | 18 ms          | 41 ms               | +23 ms |
| Escalation     | 0.077          | 0.041               | –0.036 |

- **Notes:**  
  - 3-large top-1 matches earlier pilot (Jordan: "Numbers consistent with 2025-10 dry run").
  - Latency increase due to larger embedding, but p99 still <60 ms, within SLO.
- **Discussion:**  
  - **Lin:** "Material change: triggers full governance review?"  
    - **Arjun:** "Yes. Priya to draft ADR-004b. Ops to review impact."
  - **Sara:** "Operator workflow impact?"  
    - **Priya:** "Escalation load projected down by 31%, no new UI changes."
  - **Jordan:** "Any edge-case regressions?"  
    - **Priya:** "None found in regression suite. Chain-collision tests green."

**Decision:**  
- **Consensus:** Proceed to staged rollout in Q2, pending governance signoff.  
- **Action:** Priya to prepare rollout plan and operator comms by 2026-02-20.

---

### 4. Chain-KB Vendor & ADR-004 Decision (Arjun, Ravi, Lin, 10 min)

**Status:**  
- Vendor Y (ChainSync) selected after final reference and security reviews.
- On-prem pilot deployment succeeded (2026-01-09); initial ingest latency 3.9 min per delta, within 5-min SLA.
- Audit trail and per-update logging validated by Lin.

**Decision:**  
- **ADR-004 ratified:**  
  - Vendor Y to be primary chain-KB provider, with quarterly re-eval.
  - Data pipeline integration to production by 2026-02-28.
  - Existing manual chain list to be deprecated post-migration.

**Discussion:**  
- **Martin:** "Any migration risks?"  
  - **Ravi:** "Minimal—fallback to manual list until ChainSync is confirmed stable."
- **Lin:** "Governance satisfied with logging and versioning."
- **Sara:** "Operator training planned?"  
  - **Arjun:** "Training deck to be circulated two weeks pre-go-live."

**Action Items:**  
- **Ravi:** Complete production pipeline integration by 2026-02-15.
- **Lin:** Finalize governance signoff; update documentation.

---

### 5. Next-Quarter OKRs — Framing & Discussion (Arjun, Martin, Hannah, 15 min)

**Draft OKRs:**

| Objective                                      | Key Results                                                 | Owner   |
|------------------------------------------------|-------------------------------------------------------------|---------|
| **O1:** Ship 3-large staged rollout            | - ≥0.73 top-3 live, p99 latency <65 ms<br>- <5% escalation | Priya   |
| **O2:** Complete chain-KB migration            | - 100% traffic on ChainSync<br>- Zero critical ingest gaps  | Ravi    |
| **O3:** Dashboard: axis cross-tabs & drilldown | - Live by 2026-03-15<br>- Operator feedback ≥80% positive  | Priya   |
| **O4:** Slice-level alerting                   | - Alerts for axis/chain/city outliers<br>- <24h triage     | Martin  |
| **O5:** City-frequency stratified evaluation   | - Publish monthly<br>- Report SLOs by high/low freq cities | Priya   |

**Discussion Highlights:**

- **Arjun:** "Focus on safe rollout for 3-large, no regressions on edge buckets."
- **Priya:** "Will prioritize dashboard polish; axis drilldown is requested by Ops."
- **Sara:** "Emphasize operator enablement during chain-KB migration."
- **Lin:** "Governance review cadence: monthly during transition."
- **Martin:** "Slice-level alerting will require fresh pipeline work — SRE resourcing flagged."
- **Mei:** "Consider city-frequency as a published metric for transparency."

**Next Steps:**  
- Finalize and circulate OKRs by 2026-01-19.
- Confirm SRE support for alert pipeline.
- Schedule operator training for chain-KB go-live.

---

### 6. Open Q&A (Slack-style, 8 turns)

**Hannah:** "Will the dashboard allow axis combos (e.g., long names + no-overlap)?"  
**Priya:** "Yes, planned for March. Early mockups support up to two axes in cross-tab."

**Mei:** "Any plans to automate rare-case regression surfacing?"  
**Priya:** "Part of slice-level alerting OKR; will flag low-support buckets with high escalation."

**Sara:** "How will chain-KB failures surface to operators?"  
**Ravi:** "Error state in UI, plus Ops alert. Manual KB fallback until resolved."

**Martin:** "Will the 3-large rollout be A/B or big-bang?"  
**Arjun:** "Staged: 10% ramp-up, then 50%, then all, with per-slice monitoring."

**Lin:** "Is the ADR-004b change log public to governance?"  
**Arjun:** "Yes, will be posted alongside 3-large rollout doc."

**Hannah:** "When will we sunset manual chain lists?"  
**Arjun:** "After two weeks post-migration, pending no critical issues."

**Mei:** "Are there plans for a post-migration operator survey?"  
**Sara:** "Yes, scheduled for March; will inform Q2 priorities."

**Priya:** "All operator feedback channels remain open during rollout."

---

### 7. Summary & Next Actions

- **A/B fuzzy routing:** Full rollout by 2026-01-15, ADR-002 updated.
- **Name_length axis:** Stratified metrics published, dashboard drilldown by 2026-02-05.
- **3-large:** Staged rollout plan and governance review, ADR-004b by 2026-02-20.
- **Chain-KB:** Production migration targeted for 2026-02-28, operator training scheduled.
- **Next-quarter OKRs:** Finalize and circulate by 2026-01-19.

**Adjourned.**
