# #embeddings — full thread archive, Aug 2025 → Dec 2025

This channel is the primary forum for the hotel→city matching embedding
work. The full archive is long and noisy; the messages below are all
committed here as context for the Q4 audit. Entries are clustered by
day and by sub-thread.

---
## 2025-08-14

**mei.c**  10:24 AM
kicking off the embedding workstream. rough plan: local MiniLM baseline
this week, text-embedding-3-small next week, 3-large if budget allows.
I'll land the eval harness first so we have something to grade against.
who is owning what?

**arjun.p**  10:28 AM
mei owns the embedding pipeline end-to-end. priya owns fuzzy +
stratified eval. jordan will help with the 3-large run after
onboarding.

**priya.j**  10:30 AM
👍. fwiw I think we should agree on the eval subset size up front —
3k hotels or 10k? smaller = faster iteration but bigger = more
representative.

**mei.c**  10:32 AM
let's start with 3k. we can always re-run on 10k once the pipeline is
proven.

**jordan.r**  10:40 AM
fyi I inherited the MiniLM checkpoint from the inventory team; it's
sentence-transformers/all-MiniLM-L6-v2, 384d. Should be fine for a
baseline.

---
## 2025-08-20

**mei.c**  2:15 PM
MiniLM first pass done on 3k subset: top_1=0.39 top_3=0.47. as expected
— chain hotels are dragging.

**priya.j**  2:18 PM
what's the chain-specific number?

**mei.c**  2:20 PM
haven't bucketed yet. on my list for next week once stratification is
in.

---
## 2025-08-27

**priya.j**  9:42 AM
landed a first cut of fuzzy runs. partial_ratio top_1=0.44,
wratio top_1=0.42 on the same 3k subset. fuzzy is surprisingly
competitive.

**mei.c**  9:48 AM
not surprising to me — the chain issue eats MiniLM. at top_3 I'd
expect fuzzy to pull further ahead on overlap-positive names.

**arjun.p**  10:02 AM
what's the ship target date looking like?

**mei.c**  10:05 AM
late October for 3-small; 3-large is a stretch goal. no hard date
yet.

---
## 2025-09-04 (the big day)

**mei.c**  10:02 AM
starting a re-embedding run for text-embedding-3-small over the full
hotel corpus. ETA: 90m at 128-batch. pricing note: ~$0.50 for the full
110k rows. using src/embed_openai.py directly, same batching jordan
set up last month.

**jordan.r**  10:03 AM
👍 can you also kick off the 3-large run while you're at it? I want to
see if the headline number is real. happy to own the eval side if you
do the embedding pull.

**mei.c**  10:04 AM
that's another ~$3 — pinging arjun first. also my local .env doesn't
have the production key, can you DM me the one you've been using?

**arjun.p**  10:08 AM
green light on 3-large. pls don't run ada-002 for comparison, we
already have those numbers from last month and I don't want the spend.

**mei.c**  10:10 AM
noted. jordan, pls drop the key in my DMs.

**jordan.r**  10:12 AM
sent. note the key starts with sk-proj-... which is the team-wide one,
don't leak it into scratch/ or anywhere else.

**mei.c**  10:42 AM
3-small done. partial_ratio is still king on short ASCII names but
3-small is a clear step up on the long names. full numbers in
runs/openai_3small/run1.json. top-1 0.4687, top-2 0.5137, top-3 0.5297
on the 3k eval subset. that's a clean ~7pp over minilm at top-1.

**priya.j**  10:43 AM
nice. does that match what we saw on the full 110k? I remember mei's
prior run had top-1 0.471 or so.

**mei.c**  10:45 AM
yes, within 0.003 on top-1 on the bigger corpus. so the 3k subset is
representative.

**jordan.r**  2:47 PM
3-large run done. headline top-1 is 0.698 (!!!) that's ~20pp over
3-small. writing it up in runs/openai_3large/run1.json. feel free to
kick tires.

**mei.c**  2:48 PM
wow. let me spot-check a few hotels just to be sure — that's a huge
jump and I don't trust big jumps without a sanity test.

**mei.c**  3:12 PM
ok something's weird. picked "Marriott Marquis New York" from the
hotel_names.json index and the top city it returns is "Bratislava" with
cosine 0.31. the second is "Reykjavik" at 0.29. something is off with
the alignment — either the embedding is nonsense or the row order
doesn't match hotel_names.json.

**priya.j**  3:13 PM
uhhh that should be a near-perfect match — "New York" is literally in
the hotel name.

**mei.c**  3:14 PM
yeah. grabbed 5 more hotels, all random-looking top-K. the 0.698 top-1
number is almost certainly a row-ordering bug. let me check the npy
shapes against the name indices.

**jordan.r**  3:15 PM
hm. let me check. might be a row-order mismatch. leaving for the day,
will look tomorrow morning. sorry for the fire drill.

**jordan.r** left the channel.

**mei.c**  3:20 PM
…ok. I'll dig in after EOD myself if jordan doesn't get to it.

**mei.c**  3:45 PM
ok, confirmed: embeddings/openai3large_hotels.npy has shape (110160,
1536) which matches hotel_names.json length, and _cities.npy has
(18942, 1536) which matches city_names.json length. norms are
unit-distributed and mean vectors look reasonable. so it's not a
dimension mismatch.

**mei.c**  3:48 PM
but here's the thing — if I compute cosine("Marriott Marquis New York",
all cities) and sort, the top city is always a semantically UNRELATED
one. and if I compute the cosine between the hotel vector and the
"New York" city vector directly, it's 0.07. that is wildly wrong for a
hotel whose name literally contains "New York".

**mei.c**  3:51 PM
one hypothesis: jordan's embedding script ran against a different
ordering of hotel_names/city_names than what's committed to
hotel_names.json / city_names.json. the vectors might be real openai
embeddings but for rows A, B, C, D (some other order) while the json
claims rows are A, B, C, D (canonical order). cosine still works
coordinate-wise, but maps hotel_i to city_j where j is unrelated to
i.

**priya.j**  3:54 PM
that would also explain why the top-K results look random rather than
semantically-close-but-wrong (which is what a bad model does).

**mei.c**  3:55 PM
yep. I'll leave the files in place and flag this in the 3large runs/
json as "unverified". we should NOT use the 0.698 number for any
decisions. I'll note this in the decision doc.

**mei.c**  3:58 PM
priya — given jordan is likely out for a few days, can you keep the
canonical 3-small eval tight for the Q4 review? don't block on 3-large.

**priya.j**  3:59 PM
on it.

**priya.j**  4:55 PM
btw i'm running the fuzzy branch on the 3k eval subset, getting
partial_ratio top-1 ≈ 0.44 and wratio ≈ 0.42. matches the old numbers
on this smaller subset.

**mei.c**  5:01 PM
👍. can you land a wratio config + run1.json so we have all four
canonical methods in runs/? I'll want that for the leadership deck.

**priya.j**  5:02 PM
yep. probably tomorrow.

**arjun.p**  6:20 PM
mei — re: 3large, plz also add a TODO in whatever ADR we write out of
this so future-us doesn't blunder into "well jordan's eval said 0.698"
as an argument. make it impossible to miss.

**mei.c**  6:21 PM
agreed. adding to the draft of ADR-001.

---
## 2025-09-05

**mei.c**  8:55 AM
follow-up from yesterday — ran two more sanity checks on 3-large this
morning:

1. pick a hotel whose name is JUST a city ("Hotel Shanghai", "Hotel Lima",
   "Hotel Paris"). for 3-small, cosine(hotel_i, city_matching_name) > 0.7
   consistently. for 3-large, it's near 0 for all of them.

2. row-shuffle test: compute cosine between hotel_i and city_j for all
   j, find the argmax. if the "top" cities cluster around a particular
   j for many i's, that's evidence of a systematic mis-alignment. the
   3-large argmax distribution is EXTREMELY uniform across city indices
   — i.e., the model is returning noise-like rankings, consistent with
   the "embeddings for wrong rows" hypothesis.

**arjun.p**  9:05 AM
so can we re-run with the current ordering?

**mei.c**  9:06 AM
I'd rather not spend the $3 to confirm a hypothesis. the mere fact that
we have to re-run means the committed numbers are unusable. I'll
recommend that openai_3small ships and 3-large is an open question.

**arjun.p**  9:06 AM
sgtm.

**priya.j**  10:14 AM
I landed the wratio config/run. runs/wratio/run1.json has top_1=0.4223
top_2=0.4740 top_3=0.4803. marginally behind partial_ratio at every K
on this corpus.

**mei.c**  10:20 AM
thanks. I'll update the leadership slide to drop 3-large and show the
four verified methods.

---
## 2025-09-09

**priya.j**  2:00 PM
question about src/embed_local.py — I see a `strip_accents` flag that
gets passed to `encode_minilm` but the function body just ignores it.
does anything actually happen when I pass --strip-accents?

**mei.c**  2:05 PM
good catch. nothing actually happens — the underlying transformer does
unicode NFKC already, so the flag is cosmetic. I'll add a comment
but don't want to rip it out since postmortem_minilm.md references
it.

**priya.j**  2:07 PM
ok, I'll write a one-line comment in embed_local.py so future readers
don't expect the flag to do anything.

**mei.c**  2:08 PM
thx.

---
## 2025-09-14

**arjun.p**  11:30 AM
stakeholder meeting with hannah today — she asked about the 95% miss
rate number in onepager_fuzzy_rejected.md. I told her it's from mei's
scratch eval on a harder subset, but I don't think that's actually
sourced. mei — can you confirm where that number came from?

**mei.c**  11:40 AM
it came from a scratch notebook I had on a hard-names subset, never
committed. I don't have the notebook anymore. the 95% isn't
reproducible from any artifact in this repo. I should probably revise
the one-pager.

**priya.j**  11:45 AM
I'd push back harder — the actual partial_ratio miss rate on the 3k
canonical subset is 55%. the 95% is just wrong.

**mei.c**  11:48 AM
yeah. I'll park a rewrite; probably won't get to it before my last day.

---
## 2025-09-22

**mei.c**  3:12 PM
starting to think about ADR-001. here's my sketch:

  Decision: ship openai_3small as primary ranker.
  Context: four methods eval'd on 3k subset, 3-small is the winner
    (top-1 0.47, top-3 0.53). fuzzy is competitive on overlap-positive
    names but collapses on the no-overlap bucket.
  Consequences: ongoing ~$2/day OpenAI spend at our volume. We ship
    top-3 retrieval; operator-in-the-loop picks.
  Alternatives: 3-large (unverified 0.698, see thread 2025-09-04);
    fuzzy only (cheaper but 3pp worse on top-3).
  Open questions: when/if to re-eval 3-large; whether to add fuzzy
    fallback routing post-ship.

any objections to this framing?

**arjun.p**  3:20 PM
lgtm. please explicitly say 3-large is UNVERIFIED rather than
"pending" — I don't want anyone reading the ADR and thinking there's
a pending reproduction coming.

**mei.c**  3:22 PM
done, wording updated.

---
## 2025-09-30

**priya.j**  10:05 AM
stratified PR (#071) is up for review. lexical_overlap axis only for
now. name_length and city_frequency are stubbed.

**arjun.p**  10:08 AM
approving. land it.

---
## 2025-10-05

**priya.j**  11:14 AM
fyi the stratified CSV is landing at runs/stratified/lexical_overlap.csv.
the two empty CSVs (name_length.csv, city_frequency.csv) are
placeholders — I'll fill those in Q1.

---
## 2025-10-14 (eval_v2 appearance)

**mei.c**  4:15 PM
PR #042 is merging — minor speed refactor of eval.py into eval_v2.py.
no semantic change, just warning-free on numpy 2.x.

**priya.j**  4:20 PM
👍. I'll keep using eval.py until we verify eval_v2 matches on
integer scorers; deprecated path should stay in place until that
confirmation.

**mei.c**  4:21 PM
agreed. both scripts stay.

---
## 2025-10-18 (eval_v2 divergence discovered)

(full transcript in notes/slack_eval_bugs_thread_2.md)

summary of outcome: eval_v2.py diverges from eval.py on integer-scored
fuzzy scorers by ~1 pp on top-1. runs/openai_3small/run2.json was
accidentally produced with eval_v2.py and carries inflated numbers.
canonical run stays run1.json. mei opts not to revert the cleanup.

---
## 2025-10-22 (gt_alt appearance)

(full transcript in notes/slack_gt_thread.md)

summary: mei builds ground_truth/gt_alt.json which drops the 37
multi-city hotels. inflates openai_3small top-1 by 1.3 pp. runs/
openai_3small/run3.json uses this alt GT. priya pushes back; team
agrees canonical GT is gt.json and run1 numbers are the ones that
ship.

---
## 2025-10-30

**mei.c**  9:02 AM
ADR-001 is in final review. both 3-large footnote and eval_v2
footnote are in. leadership review is 11/7.

**arjun.p**  9:05 AM
thx.

---
## 2025-11-07

leadership review (see meetings/leadership_review_2025-11-07.md).
decision to ship openai_3small confirmed.

---
## 2025-11-14 (mei's last day)

**mei.c**  5:02 PM
signing off. handoff doc is in notes/mei_handoff.md (if I got to it —
otherwise Arjun has the brain dump from our last 1:1).

folks, ship the thing. fuzzy fallback as post-ship A/B per priya's
plan. 3-large re-embed is a Q1 call.

**arjun.p**  5:04 PM
thanks mei. safe travels.

**priya.j**  5:05 PM
☕ good luck.

**mei.c** left the channel.

---
## 2025-11-17

**arjun.p**  10:12 AM
post-ship monitoring: top-K on the last 7 days of bookings is within
0.3 pp of the offline 3k-subset numbers. healthy. priya's stratified
dashboard will land next week.

---
## 2025-11-24

**priya.j**  2:05 PM
ok I've taken another look at openai_3large since we now have some
breathing room. confirmed the row-order hypothesis: picked 10 random
hotels, computed cosine against every city, looked at the argmax and
the argmax's rank against where the right city SHOULD be. the
distribution is uniform across ranks — consistent with a row
permutation. writing this up as a comment on runs/openai_3large/run1.json.

**arjun.p**  2:10 PM
👍. if it's definitely a permutation, a re-embed would resolve.

**priya.j**  2:11 PM
yes. Q1 call per ADR.

---
## 2025-12-01

**priya.j**  11:30 AM
drift dashboard PR #128 is up. weekly-refresh top-K on fresh-week
bookings. alert thresholds in the PR.

---
## 2025-12-02

**priya.j**  2:00 PM
revisiting gt_alt in light of the Q4 review. looking at the 37 dropped
hotels, most are chain franchises where one chain name legitimately
maps to multiple cities. our GT for those is [city1, city2, ...]
which is honest — any of them is a correct prediction. gt_alt
effectively drops the hardest cases and is not a "cleaner GT", it's
a "skip the hard cases" GT.

**arjun.p**  2:05 PM
good catch. please flag this prominently in whatever Q1 audit doc
you produce.

**priya.j**  2:08 PM
already in the stratified PR risk section — I'll make it more
prominent.

---
## 2025-12-10

**arjun.p**  9:00 AM
starting to think about chain-KB integration for Q1. this is the play
for the no-overlap bucket. ADR-004 draft this week.

---
END OF CHANNEL ARCHIVE


---
## Appendix A — Q4 2025 #embeddings messages

These are verbatim channel messages (minus timestamps where noise) that
weren't in the summary view above. Kept for audit completeness.

### Week of 2025-10-27

**priya.j**  9:14 AM
did anyone else get paged overnight? `hotel-city-latency-p95` tripped
for 6 minutes around 2am UTC.

**arjun.p**  9:16 AM
looked this morning — was a Snowflake spike that our batch job is
waiting on. not model-related. resolved itself.

**priya.j**  9:18 AM
ok, adding a note to the runbook so we know not to page the model
team for snowflake hiccups.

### Week of 2025-11-03

**mei.c**  10:10 AM
last-week-before-handover checklist:
  [x] ADR-001 signed
  [x] leadership review (11/7)
  [x] final_recommendation.md revised per Priya
  [ ] handoff doc (working on it)
  [ ] 1:1 with Priya on #eval-v2-tie-break (today 2pm)
  [ ] 1:1 with Arjun on 3-large Q1 plan (tomorrow)

**arjun.p**  10:20 AM
please also flag the one_pager_openai_win.md cherry-picking
situation in your handoff doc. I'd rather not have to re-explain
that to the next engineer who reads it.

**mei.c**  10:22 AM
good call, adding.

### Week of 2025-11-10

**hannah.k**  (product)  3:22 PM
re: ship tomorrow — is there anything product-side I should be
prepared for in the first week of traffic?

**arjun.p**  3:25 PM
offline top-3 is 0.53, we expect online to be within 1 pp. if
operator-selection-rate-at-3 drops below 0.45 in week 1, page me.

**hannah.k**  3:26 PM
👍

### Week of 2025-11-17

**arjun.p**  (summary)
week 1 post-ship:
  - top-3 on fresh sample: 0.530 (offline baseline: 0.530) ✓
  - operator-selection-rate-at-3: 0.528 (target > 0.45) ✓
  - latency P95: 18 ms (target < 50) ✓
  - no alerts
  - no ops escalations
  verdict: clean launch. circulating celebratory note.

### Week of 2025-11-24

**priya.j**  11:00 AM
starting on the 3-large probe. setting aside an hour this week to
confirm the row-order hypothesis. if confirmed, the path forward is
just the $3 re-embed in Q1.

**arjun.p**  11:05 AM
sgtm. keep me posted; I have the $3 budget if you want to do it
this year, but no pressure.

**priya.j**  11:06 AM
I'd rather wait until Q1; the probe is what I really need. the
re-embed follows from the probe result.

### Week of 2025-12-01

**priya.j**  9:30 AM
probe writeup is up in notes/priya_3large_probe.md (if we add that
file) or inline in notes/slack_embeddings_thread.md. TL;DR: it IS
a row-permutation, not identity, not uniform-random — something in
between. numerically the argmax distribution is approximately
uniform when we probe across 10 random hotels × all 1962 cities.
so the vectors are real, the ranking just maps them to the wrong
rows.

**arjun.p**  9:40 AM
confirmed: re-embed it is, Q1.

**priya.j**  9:42 AM
I'll add a comment to runs/openai_3large/run1.json explaining the
situation so future readers don't cite the 0.698 without context.

### Week of 2025-12-08

**arjun.p**  10:15 AM
stakeholder update round:
  - Hannah / Martin: briefed on 3-large Q1 plan.
  - Lin: governance sign-off on the runbook updates.
  - next quarterly review: mid-Feb.

**arjun.p**  4:20 PM
also — reviewing notes/onepager_fuzzy_rejected.md again today since
it came up in the leadership review as a doc people cite. the 95%
miss rate is still stated as fact. I'm inclined to leave the doc
in place but add a prominent "disputed" header at top. thoughts?

**priya.j**  4:22 PM
+1 for a header. or we could link to my onepager_fuzzy_context.md
which corrects the record.

**arjun.p**  4:25 PM
let's do both. I'll send a PR for the header next week.

### Week of 2025-12-15

**priya.j**  2:40 PM
post-ship A/B plan ready. routing rule: short-ASCII overlap names
go through partial_ratio first. A/B split: 50/50 for 2 weeks.
metric: operator-selection-rate-at-3. sample size: ~5k / arm / day.

**arjun.p**  2:45 PM
PR #118 covers the routing sketch, #128 covers the dashboard to
measure it. green light once both are merged.

### Week of 2026-01-05

**priya.j**  10:02 AM
happy new year. starting Q1 work: name_length + city_frequency
stratified axes. target end-of-month.

**arjun.p**  10:05 AM
also this week: ADR-004 chain-KB scoping. I'll share a draft by
Thursday.

### Week of 2026-01-19

**arjun.p**  11:18 AM
ADR-004 draft circulated. vendors: HotelKB Inc., BrandAtlas.
scoping-only; not a decision yet.

**priya.j**  11:30 AM
read it. one concern — HotelKB's refresh cadence is monthly.
our booking corpus grows weekly. we'd lag by up to a month on new
chains. note that in the ADR?

**arjun.p**  11:32 AM
noted, editing.

### Week of 2026-01-26

**priya.j**  4:18 PM
stratified name_length axis landed. buckets <=10 / 11-25 / 26-40 / >40.
CSV is at runs/stratified/name_length.csv. numbers look as expected —
the long-name bucket (>40 chars) is where chain-prefix dilution
really hurts minilm.

**arjun.p**  4:20 PM
attach in the follow-up leadership review deck.

---
## Appendix B — archived private DMs (relevant to the audit)

These are DMs that referenced pipeline decisions; the team agreed
to archive them in the channel for transparency after Mei's
departure.

### Mei ↔ Arjun, 2025-11-03

Mei: draft of final_recommendation.md is ready. I'm linking the
jordan run in the future-proofing section, is that ok?

Arjun: yes but please label it "unverified" consistently. I don't
want a reader to assume we're going to migrate on the 0.698.

Mei: got it.

### Mei ↔ Jordan, 2025-09-04 (retrospectively pulled from backup)

Mei: hey, can you double-check the 3-large run on "Marriott Marquis
New York" for me? cosine to "New York" is 0.07, which is implausible.

Jordan: hm, running it. 0.07 sounds wrong.

Jordan (20 min later): yeah I'm getting 0.07 too. let me check the
embedding generation order.

Jordan: I may have pulled hotel_names from an older snapshot.
leaving for the day, will check tomorrow.

[Jordan departed before following up.]

### Priya ↔ Arjun, 2025-12-02

Priya: I can confirm 3-large is row-permuted now. probe shows
uniform argmax distribution. sending full writeup to the channel.

Arjun: thanks. good work.

---
## Appendix C — channel topic history

Channel #embeddings topic, chronologically:

- 2025-08-14: "hotel-city matching: embedding workstream"
- 2025-09-04: "3-small lands ✓ | 3-large NEEDS VERIFICATION 🚨"
- 2025-09-08: "3-small lands ✓ | 3-large UNVERIFIED (row-order bug)"
- 2025-11-06: "ship openai_3small under ADR-001 on 11/15"
- 2025-11-22: "post-ship week 1: clean ✓"
- 2025-12-15: "A/B plan circulating, Q1 focus: 3-large re-embed, chain-KB"


---

---
## 2026-01-08

**hannah.k**  9:10 AM  
morning! is there a doc summarizing which axes we report on in the stratified dashboard? trying to brief Martin.

**priya.j**  9:13 AM  
yep, see notes/stratified_axes.md. current: lexical_overlap, name_length, city_frequency. all three live by end of Jan.

**hannah.k**  9:14 AM  
perfect, thanks.

---
## 2026-01-10

**arjun.p**  2:45 PM  
quick heads up: chain-KB intro call with BrandAtlas is set for next Tues, 3pm. Priya joining, anyone else?

**lin.z**  2:47 PM  
please add me — want to hear their privacy/compliance posture.

**arjun.p**  2:48 PM  
added, calendar invite updated.

---
## 2026-01-16

**priya.j**  11:12 AM  
noting: drift dashboard flagged a 0.4pp dip on operator-selection-rate-at-3 Wed/Thurs. level set: this is within noise, but tracking.

**arjun.p**  11:15 AM  
thanks. latency or ops noise?

**priya.j**  11:17 AM  
checked logs — batch ingest lagged, bookings from two peak windows got delayed. not model-side.

**hannah.k**  11:18 AM  
flag in the weekly deck or let it ride?

**priya.j**  11:19 AM  
let it ride unless it repeats.

---
## 2026-01-21

**newton.d**  9:03 AM  
hi all, I’m Newton (new eng, joined Last Friday). I’m onboarding to embeddings infra. Is there a "start here" doc for the pipeline? Also, is the canonical eval script eval.py or eval_v2.py now?

**arjun.p**  9:05 AM  
welcome Newton! start with notes/mei_handoff.md and notes/pipeline_overview.md. eval.py is canonical for legacy runs, eval_v2.py is safe for new runs IF you’re on numpy 2.x. See thread 2025-10-18 for corner-case divergence.

**priya.j**  9:09 AM  
for stratified evals, use eval.py for consistency with dashboards. happy to pair if you want to walk through a run.

**newton.d**  9:12 AM  
thanks both! will ping if I hit snags.

---
## 2026-01-23

**arjun.p**  3:35 PM  
pagerduty ping: `hotel-city-latency-p95` spiked again at 1am UTC. Anyone see model issues?

**priya.j**  3:36 PM  
reviewed logs — another Snowflake batch stall. Matches pattern from Oct/Nov. No model anomaly.

**arjun.p**  3:38 PM  
noted. Newton, see above — these are not model alerts.

**newton.d**  3:39 PM  
logging for runbook update.

---
## 2026-01-27

**arjun.p**  10:00 AM  
BrandAtlas call notes: refresh cadence = weekly, city universe = 2100, chain coverage 98%. Pricing higher than HotelKB but less lag.

**lin.z**  10:02 AM  
compliance: BA is US/EU only, so APAC hotels out of scope for now.

**priya.j**  10:04 AM  
noted. For no-overlap bucket, either is sufficient, but APAC is a future gap.

**arjun.p**  10:07 AM  
ADR-004 will flag that as a limitation.

---
## 2026-01-29

**martin.s**  5:15 PM  
quick one: do we need to update the leadership dashboard for the A/B fuzzy fallback? Numbers look identical to pre-A/B.

**priya.j**  5:17 PM  
no change needed yet — A/B arms are within 0.1pp. If that diverges >0.5pp, we’ll update the dashboard.

**martin.s**  5:18 PM  
sg, thanks.

---
## 2026-02-03

**arjun.p**  9:30 AM  
Q1 re-embed: OpenAI 3-large slot is reserved for next Monday (2/9). Plan: full hotel x city matrix, canonical row order. Anyone have last-minute blockers?

**priya.j**  9:32 AM  
none from me. scripts ready, final hotel/city lists match run1.

**newton.d**  9:33 AM  
do we want both eval.py and eval_v2.py outputs for audit?

**arjun.p**  9:35 AM  
yes, run both, flag if any >0.2pp delta. Thanks.

---
## 2026-02-07

**priya.j**  11:24 AM  
reminder: drift dashboard flagged a minor <0.2pp drop this week. Still within expected, but noting for completeness.

**arjun.p**  11:25 AM  
got it, thanks. New hotel batch landed or just booking mix?

**priya.j**  11:26 AM  
booking mix: surge in shorter city names (e.g. “Lyon”, “Oslo”), which partial_ratio likes.

---
## 2026-02-10

**arjun.p**  2:18 PM  
Q1 3-large re-embed running now. ETA: 7 hours. Newton, you have next steps for eval when output lands.

**newton.d**  2:20 PM  
ack, will kick off evals and update runs/openai_3large/run2.json.

---
## 2026-02-11

**newton.d**  9:01 AM  
3-large re-embed complete. top-1: 0.6739, top-3: 0.7442 (canonical GT). No row-permutation observed. delta vs old numbers: top-1 ~-2.4pp (inflated before). Full writeup in notes/3large_reembed_2026Q1.md.

**arjun.p**  9:05 AM  
nice work. Please update ADR-001 and add a footnote in the run1.json on the legacy permutation numbers.

**priya.j**  9:07 AM  
will update dashboards for Q2 with run2.json as baseline.

---
## 2026-02-13

**hannah.k**  4:10 PM  
minor: should I mention the 3-large fix in the Feb PM notes, or wait until we see any impact in operator metrics?

**arjun.p**  4:12 PM  
note it for transparency, but it’s backend only unless we swap it in live.

**hannah.k**  4:13 PM  
will do.

---
## 2026-02-18

**arjun.p**  3:40 PM  
HotelKB call this morning: monthly refresh confirmed, but APAC coverage includes 900+ cities. BrandAtlas still leads on recency.

**lin.z**  3:42 PM  
privacy posture passes, but vendor contract needs legal review. Flagged to legal.

**priya.j**  3:44 PM  
for stratified reporting: both vendors can support no-overlap bucket, but city-frequency axis will need custom mapping.

---
## 2026-02-20

**martin.s**  10:30 AM  
leadership review prep: can someone summarize where we landed on the 3-large re-embed and chain-KB?

**arjun.p**  10:33 AM  
summary:  
- 3-large re-embed complete; clean row order; new baseline numbers in run2.json.  
- chain-KB: BrandAtlas better for freshness, HotelKB covers APAC. Both privacy-cleared, legal pending.  
- no production change yet, Q2 pilot planned.

**martin.s**  10:36 AM  
thanks Arjun.

---
## 2026-02-22

**priya.j**  3:15 PM  
drift dashboard: no alerts, all metrics within 0.2pp of new baselines. A/B fuzzy arm ends next week.

**arjun.p**  3:16 PM  
wrap writeup for the fuzzy A/B and link in notes/experiments_2026Q1.md please.

---
## 2026-02-28

**arjun.p**  5:00 PM  
recap —  
- Q1 goals on track: 3-large re-embed, chain-KB scoped, drift stable.  
- No model P0s, only two Snowflake-induced paging events.  
- Newton, welcome again, thanks for getting up to speed so quick.

**newton.d**  5:02 PM  
thanks all! Glad to be here.

**priya.j**  5:03 PM  
🍵 well done team.

---

---

---
## Appendix D — raw paging transcripts

Below are raw excerpts from PagerDuty pages impacting the #embeddings engineering team during Q4 2025 – Q1 2026. Times are UTC. Handles reflect the on-call responder for each alert. Most represent false alarms; one (2025-12-07) is a substantive drift alert. Lines are verbatim except where `[REDACTED]` indicates PII removal. These are included for audit transparency and operational postmortem purposes.

---

### 1. 2025-10-25 02:05 PagerDuty — Snowflake Latency Spike

```
[PAGER] 02:05 hotel-city-latency-p95 breached threshold (60s > 5000ms)
Assigned: @arjun.p (on-call)

[02:05] @arjun.p: ack
[02:06] @arjun.p: checking snowflake dashboards...

[02:07] @arjun.p: snowflake warehouse queue at 40/8, not our job specifically
[02:08] @arjun.p: embedding service response times normal
[02:09] @arjun.p: this matches previous night’s batch spike

[02:10] @arjun.p: no action — not a model issue
[02:12] @arjun.p: resolved page
```

---

### 2. 2025-10-31 04:44 PagerDuty — Embedding Service OOM

```
[PAGER] 04:44 embedding-svc OOMKilled on node pool-7
Assigned: @mei.c (on-call)

[04:44] @mei.c: ack
[04:45] @mei.c: logs show 1/4 pods OOM-killed, restart succeeded
[04:47] @mei.c: heap usage at 2.1G/2.2G pre-restart
[04:49] @mei.c: pod healthy post-restart, no failed requests
[04:51] @mei.c: scaling up node memory limit from 2.2G → 2.5G

[04:54] @mei.c: closing page, tracking as #124
```

---

### 3. 2025-11-05 11:23 PagerDuty — Redis Cache Miss Rate

```
[PAGER] 11:23 redis-cache-miss-rate > 0.35 for 5m
Assigned: @priya.j (on-call)

[11:23] @priya.j: ack
[11:24] @priya.j: reviewing redis metrics
[11:25] @priya.j: miss rate spiked to 38% at 11:18, dropped to 17% by 11:22
[11:27] @priya.j: cache evictions also spiked
[11:28] @priya.j: suspect batch job flushed hot keys

[11:30] @priya.j: system recovered, no user impact
[11:31] @priya.j: page resolved, no further action
```

---

### 4. 2025-11-09 03:17 PagerDuty — Hotel-City Latency

```
[PAGER] 03:17 hotel-city-latency-p95 > 2500ms for 2m
Assigned: @arjun.p (on-call)

[03:17] @arjun.p: acked
[03:18] @arjun.p: embedding service logs: no spike
[03:19] @arjun.p: snowflake latency 3x normal, see prev. incident

[03:20] @arjun.p: not model infra, closing page
```

---

### 5. 2025-11-14 16:21 PagerDuty — Embedding API 5xx

```
[PAGER] 16:21 embedding-api 5xx error rate > 0.05 for 3m
Assigned: @mei.c (on-call)

[16:21] @mei.c: ack
[16:22] @mei.c: logs: 4/1200 requests 502, all at 16:19
[16:23] @mei.c: upstream OpenAI API 429: “quota exceeded” for 1s

[16:24] @mei.c: no sustained errors, quota auto-recovered
[16:25] @mei.c: closing page
```

---

### 6. 2025-11-21 01:08 PagerDuty — Embedding Cache Eviction

```
[PAGER] 01:08 embedding-cache-eviction-rate > 0.2 for 10m
Assigned: @arjun.p (on-call)

[01:08] @arjun.p: ack
[01:09] @arjun.p: cache eviction logs: 3x normal, started 00:59
[01:10] @arjun.p: batch job replaced 80k keys; expected for Friday run
[01:11] @arjun.p: no downstream errors

[01:12] @arjun.p: closing page, consider suppressing alert for batch hours
```

---

### 7. 2025-11-26 06:44 PagerDuty — Embedding Service OOM

```
[PAGER] 06:44 embedding-svc OOMKilled on node pool-3
Assigned: @priya.j (on-call)

[06:44] @priya.j: ack
[06:45] @priya.j: pod restarted, healthy
[06:46] @priya.j: heap spike from batch hotel ingest (logs: “ingest batch size: 5000”)
[06:48] @priya.j: reduced ingest batch size to 2500

[06:50] @priya.j: no dropped requests, closing page
```

---

### 8. 2025-12-01 02:17 PagerDuty — Redis Cache Miss Rate

```
[PAGER] 02:17 redis-cache-miss-rate > 0.4 for 5m
Assigned: @arjun.p (on-call)

[02:17] @arjun.p: ack
[02:18] @arjun.p: cache miss rate 41% at 02:16, matched batch window
[02:19] @arjun.p: see also 2025-11-05; no user impact

[02:20] @arjun.p: closing page
```

---

### 9. 2025-12-04 23:05 PagerDuty — Embedding API Latency

```
[PAGER] 23:05 embedding-api-latency-p95 > 700ms for 2m
Assigned: @mei.c (on-call)

[23:05] @mei.c: ack
[23:06] @mei.c: OpenAI latencies spiked to 900ms at 23:03, back to normal by 23:06
[23:07] @mei.c: no error increase, no retries

[23:08] @mei.c: transient, closing page
```

---

### 10. 2025-12-07 09:45 PagerDuty — Drift Alert (real incident)

```
[PAGER] 09:45 drift-daily top-3 accuracy drop > 1.5pp (detected: 2.1pp)
Assigned: @priya.j (on-call)

[09:45] @priya.j: ack
[09:46] @priya.j: dashboard: top-3 at 0.508, baseline 0.530
[09:47] @priya.j: pulling booking sample for last 24h
[09:50] @priya.j: 7/26 misclassifications are new chain hotels not in GT
[09:53] @priya.j: checked logs, embedding service healthy

[09:56] @priya.j: GT lag on new chains, not model regression
[09:57] @priya.j: ticketed as #drift-2025-12-07, closing page
```

---

### 11. 2025-12-10 03:16 PagerDuty — Hotel-City Latency

```
[PAGER] 03:16 hotel-city-latency-p95 > 2000ms for 3m
Assigned: @arjun.p (on-call)

[03:16] @arjun.p: ack
[03:17] @arjun.p: snowflake dashboards: heavy ETL job
[03:18] @arjun.p: embedding service normal

[03:19] @arjun.p: non-embedding infra, closing page
```

---

### 12. 2025-12-15 21:08 PagerDuty — Embedding Service OOM

```
[PAGER] 21:08 embedding-svc OOMKilled on node pool-2
Assigned: @priya.j (on-call)

[21:08] @priya.j: ack
[21:09] @priya.j: identical to prior OOM, pod auto-restarted
[21:10] @priya.j: heap at 2.47G/2.5G pre-restart

[21:11] @priya.j: tracking as duplicate of #124, closing page
```

---

### 13. 2026-01-03 06:11 PagerDuty — Redis Cache Miss Rate

```
[PAGER] 06:11 redis-cache-miss-rate > 0.4 for 4m
Assigned: @arjun.p (on-call)

[06:11] @arjun.p: ack
[06:12] @arjun.p: batch GT update running
[06:13] @arjun.p: cache miss spike, no errors

[06:14] @arjun.p: expected, closing page
```

---

### 14. 2026-01-12 04:41 PagerDuty — Embedding API 5xx

```
[PAGER] 04:41 embedding-api 5xx error rate > 0.05 for 2m
Assigned: @priya.j (on-call)

[04:41] @priya.j: ack
[04:42] @priya.j: 2/800 requests 502, at 04:40
[04:43] @priya.j: OpenAI “service unavailable” for 1s

[04:44] @priya.j: transient, no action needed, closing page
```

---

### 15. 2026-01-15 01:57 PagerDuty — Embedding Service OOM

```
[PAGER] 01:57 embedding-svc OOMKilled on node pool-7
Assigned: @arjun.p (on-call)

[01:57] @arjun.p: ack
[01:58] @arjun.p: pod restarted, no failed requests
[01:59] @arjun.p: memory usage peaked at 2.48G, batch ingest

[02:00] @arjun.p: see prior OOMs, will propose further heap increase in next sprint
[02:01] @arjun.p: closing page
```

---

---

---
## Appendix E — vendor evaluation notes

Compiled by: Priya J.
Date: 2026-01-30

These are condensed notes and synthesis from vendor intro and technical scoping calls with HotelKB Inc. and BrandAtlas, held between 2026-01-10 and 2026-01-25, in support of ADR-004 (chain-KB integration for improved no-overlap bucket performance in hotel-city entity matching). These notes include pricing, refresh cadences, technical fits, pros/cons, and a comparative decision matrix.

---

### 1. Vendor Summaries

#### 1.1 HotelKB Inc.

**Overview:**
- Established B2B provider of global hotel chain metadata.
- Focus: chain/franchise mapping, canonical groupings, standardized hotel names.
- Clients: large OTAs, meta-search engines.

**Data Surface:**
- 78,000 hotel groupings globally, 4,000+ recognized chain brands.
- Hotel entity includes: Chain name, Group ID, Brand tier, Address block, Franchise status, Historical aliases.
- API + monthly CSV exports.

**API/Integration:**
- RESTful API, OAuth2.0; bulk CSV S3 drops available.
- Latency: 300ms p95 for individual queries; bulk mode ~40K records/min.
- Historic data available (2Y retention).

**Refresh cadence:**
- Monthly full refresh (1st of month UTC).
- Weekly deltas available for an upcharge (discussed below).
- SLA: 99.9% data availability, 48h incident response.

**Pricing:**
- Base (monthly full, up to 100K entities): $2,000/mo.
- Weekly deltas add $800/mo.
- Unlimited API access included in base, but with rate limits (50 QPS burst, 10 QPS sustained).
- Overages: $0.05/hotel over 100K.

**Support:**
- Dedicated technical AM, 24/5 support.

**Pros:**
- Most comprehensive global coverage (esp. outside North America).
- Strong canonicalization: handles chain mergers, rebrandings, aliasing.
- Stable API, good documentation, responsive support in pilot.

**Cons:**
- Slowest refresh (monthly unless we pay for deltas).
- Some latency for new hotels entering booking corpus.
- CSV exports are large; bulk loads take 1-2 hours.
- Minor mismatches on boutique/soft-branded hotels.

---

#### 1.2 BrandAtlas

**Overview:**
- Newer, smaller vendor focused on US/EU chain footprint.
- Focus: real-time change detection, event-driven updates (openings, closures, rebrandings).
- Clients: expense management, geo-analytics, fintechs.

**Data Surface:**
- 31,000 chain-affiliated properties (US/EU bias), 900+ chains.
- Entity includes: Chain, Brand, Franchise/managed flag, Opening date, Last update, City/Region tags.
- Webhook push, GraphQL API.

**API/Integration:**
- GraphQL endpoint, token auth; webhook for new/changed properties.
- Latency: ~150ms p95 for queries; webhook pushes within 2h of event.
- No historic data (current state only).

**Refresh cadence:**
- Near-real-time for chain events (openings/rebrands/closures).
- Full refresh dump on request (manual, 1-2d SLA).

**Pricing:**
- Base (real-time US/EU only): $1,200/mo.
- Global add-on: +$700/mo (coverage to 53,000 properties).
- Webhook included; full dumps +$200/request.
- No explicit overage fees.

**Support:**
- Slack-based support, 9am–9pm CET, 1 engineer on call.

**Pros:**
- Fastest update cycle (hours, not days/weeks).
- Event-driven: new openings visible quickly.
- Webhook integration aligns with future streaming plans.
- Simpler data model; smaller payloads.

**Cons:**
- Limited global/comprehensive coverage (notably in APAC, MEA).
- No historic snapshots.
- Weaker on canonicalization (manual mapping needed for some aliases).
- Less mature support/documentation.

---

### 2. Comparative Decision Matrix

| Dimension                       | HotelKB Inc.              | BrandAtlas                    |
|----------------------------------|---------------------------|-------------------------------|
| **Coverage (chains)**            | 4,000+ (global)           | 900+ (mostly US/EU)           |
| **Coverage (properties)**        | 78,000+                   | 31,000 (US/EU), 53,000 global |
| **Refresh cadence**              | Monthly (weekly delta $)  | Real-time (webhook)           |
| **Data model**                   | Chain, Group, Brand,      | Chain, Brand, Franchise flag  |
|                                  | Franchise, Aliases, etc.  | (simpler, no historical)      |
| **API**                          | REST, CSV bulk            | GraphQL, Webhook              |
| **Latency (per query)**          | 300ms                     | 150ms                         |
| **Bulk ingest**                  | CSV S3, 1-2h/refresh      | No bulk, only full dump       |
| **Historic data**                | 2 years                   | None                          |
| **Canonicalization strength**    | High (mergers/aliases)    | Medium                        |
| **New hotel lag**                | Up to 4 weeks (monthly)   | 2–12 hours (webhook)          |
| **Pricing (base)**               | $2,000/mo                 | $1,200/mo (US/EU)             |
| **Pricing (full/global/deltas)** | +$800/mo (weekly deltas)  | +$700/mo (global)             |
| **Support**                      | 24/5, technical AM        | 9am–9pm CET, Slack            |
| **SLA**                          | 99.9%                     | None formal                   |

---

### 3. Detailed Pros/Cons Analysis

#### HotelKB Inc.

**Strengths:**
- **Coverage**: Only vendor with near-complete global chain list. Handles local brands and cross-region chains.
- **Canonicalization**: Robust handling of chain mergers, rebrandings, franchise/managed splits. Good for deduplication.
- **Data Integrity**: Historical context—can backfill or audit entity changes over time.
- **API Stability**: Mature, well-tested endpoints; S3 bulk loads are reliable.

**Weaknesses:**
- **Update Lag**: Monthly default refresh means new hotels and rebrands may lag up to 4 weeks. Weekly delta option is extra cost and still not real-time.
- **Bulk Data Size**: Full refreshes are large; integration into pipeline requires robust ETL.
- **Cost**: Highest TCO for full/delta coverage.
- **Boutique Chains**: Some edge cases missed (soft brands, semi-independent groups).

**Ideal Use Case:** When completeness, canonicalization, and historic auditability are critical, and new hotel lag is tolerable.

---

#### BrandAtlas

**Strengths:**
- **Freshness**: Real-time event-driven updates; new chain openings and rebrands reflected in hours.
- **Simplicity**: Lightweight data model, easy to integrate; webhooks align with future plans for streaming ingestion.
- **Cost**: Lower base price, especially if only US/EU needed.
- **Agility**: Responsive to quick changes; good for pilot/prototyping.

**Weaknesses:**
- **Coverage**: Limited outside US/EU, even with “global” tier. Many local chains not mapped.
- **Canonicalization**: Manual mapping needed for some brands/aliases; possible entity drift.
- **No Historical Data**: Cannot reconstruct entity state at past points in time.
- **Support Maturity**: Less robust; documentation and support processes still maturing.

**Ideal Use Case:** When rapid update cycles are essential, coverage can be traded off, and integration speed is prioritized.

---

### 4. Integration and Technical Fit

#### Fit to No-Overlap Bucket

- **HotelKB Inc.**: Most likely to resolve long-tail no-overlap cases, especially among international and mid-tier chains. Entity mappings strong for canonical name queries and ambiguous chain variants. Lag an issue for very new openings.
- **BrandAtlas**: Helps with new/fast-changing US/EU chain openings, but will miss international or boutique chains. Canonicalization concerns may leave some no-overlap names unmapped.

#### Pipeline Integration Notes

- **HotelKB Inc.**: Requires scheduled ETL for monthly/weekly loads. CSV format matches current pipeline, but bulk loads will require batching and error handling. API fallback for point queries feasible but not primary path.
- **BrandAtlas**: Webhook integration appealing for future streaming, but batch-mode is manual. GraphQL API is new for us; will require a lightweight adapter.

#### Maintenance

- **HotelKB Inc.**: Well-defined process, but requires ops attention for load completion and periodic schema evolution.
- **BrandAtlas**: Simpler day-to-day, but will need periodic mapping audits and manual interventions for coverage gaps.

---

### 5. Pricing Scenarios

| Scenario                | HotelKB Inc. (mo) | BrandAtlas (mo) |
|-------------------------|-------------------|-----------------|
| US/EU only, base        | ~$2,000           | $1,200          |
| Global, base            | ~$2,000           | $1,900          |
| Global + fast updates   | $2,800 (weekly)   | $1,900          |
| One-off full dump       | Included          | $200/dump       |

*Note: For our projected 90K active hotel entities, both vendors’ base tiers suffice. HotelKB only charges overage above 100K.*

---

### 6. Decision Matrix & Recommendations

| Criteria              | Weight | HotelKB Inc. | BrandAtlas | Notes                                   |
|-----------------------|--------|--------------|------------|-----------------------------------------|
| Coverage              | 4      | 5            | 3          | Global, all chains vs. US/EU bias       |
| Freshness             | 3      | 2            | 5          | Real-time for BrandAtlas                |
| Canonicalization      | 4      | 5            | 3          | Mergers/aliases handled by HotelKB      |
| Cost                  | 2      | 3            | 4          | BrandAtlas lower, esp. US/EU-only       |
| Integration           | 2      | 4            | 4          | Both feasible, BrandAtlas more modern   |
| Support/SLAs          | 1      | 5            | 3          | HotelKB mature, BrandAtlas newer        |
| Historical Data       | 2      | 5            | 1          | Only HotelKB provides                   |

**Weighted Score (out of 5):**

- **HotelKB Inc.:** (4×5) + (3×2) + (4×5) + (2×3) + (2×4) + (1×5) + (2×5) = 20+6+20+6+8+5+10 = **75**
- **BrandAtlas:** (4×3) + (3×5) + (4×3) + (2×4) + (2×4) + (1×3) + (2×1) = 12+15+12+8+8+3+2 = **60**

---

### 7. Synthesis & Next Steps

- **If primary KPI is *no-overlap recall* globally:** HotelKB Inc. is the clear leader. Coverage and canonicalization outweigh lag, especially for our international inventory and for pipeline auditability.
- **If pilot focus is *US/EU freshness*:** BrandAtlas could pair well for a limited-scope A/B, with faster reaction to new openings/rebrands.
- **Hybrid option:** HotelKB for canonical mapping, BrandAtlas webhook as a supplement for new US/EU openings. Complexity increases, but theoretically optimal coverage + freshness.

**Recommendation for ADR-004:**  
Proceed with HotelKB Inc. for canonical chain mapping. Negotiate for weekly deltas if budget allows. Revisit BrandAtlas as a supplement for real-time US/EU ingest if operator escalations show lag is a top pain point.

---

### 8. Open Questions

- What is the operational burden of weekly deltas from HotelKB? Can pipeline absorb weekly reloads or should we stick to monthly?
- Will BrandAtlas expand APAC/MEA coverage in 2026? (Follow-up: Roadmap call scheduled 2026-02-10.)
- Can either vendor provide APIs for ambiguous/soft-branded chains, e.g., "Curio by Hilton" vs. "Hilton Garden Inn"?
- Is there a path to automated mapping reconciliation if hybrid approach is pursued?

---

### 9. Call Summaries (for audit completeness)

#### HotelKB Inc. — 2026-01-11 (with Marcus L., CTO)

- Demo of API and S3 bulk loads.
- Confirmed global coverage.
- Discussed lag for new openings: "Monthly is standard, but weekly delta is popular for large OTAs."
- SLA: historic data, 48h response, 99.9% uptime.
- CSV schema versioning quarterly; notification on breaking changes.
- Willing to provide 3-month pilot at discounted rate.

#### BrandAtlas — 2026-01-17 (with Eva R., Head of Product)

- Walkthrough of webhook/event model.
- Emphasized speed: "Our median is 2 hours from event to webhook."
- Confirmed lack of historic data; current-state only.
- Coverage gaps in APAC/MEA acknowledged, roadmap for 2026 expansion.
- Support model: Slack for integration phase, email escalation.

#### Joint (Internal) Synthesis — 2026-01-25

- Both vendors can deliver basic chain → city mappings.
- For no-overlap bucket, canonicalization and completeness are critical.
- Freshness becomes secondary, unless operator escalations spike.
- Pipeline can integrate either; HotelKB is lower risk for auditability.

---

**End of Appendix E**
