# Design: hotel→city matching service

**Status:** implemented (shipped 2025-11-15) ·
**Authors:** Mei Chen (primary), Jordan Rao (contributor), Priya Joshi (stratified-eval section)
**Version:** 2.3 (last updated 2025-11-06, frozen for ADR-001)

---

## Table of contents

1. Motivation
2. Scope
3. Approach
4. Method comparison
5. Eval methodology
6. Stratification
7. openai_3large caveats
8. Serving architecture
9. Data flow
10. Operational runbook
11. Cost model
12. Accuracy ceiling analysis
13. Limits and caveats
14. Alternatives considered (rejected)
15. Migration plan (for future method swaps)
16. Open questions / Q1 2026 parking lot
17. Appendices

---

## 1. Motivation

Our booking pipeline ingests user-typed hotel names at ~111k rows per
day and needs to normalise them to canonical city strings for
downstream analytics, inventory deduplication, and partner reporting.
The booking front-end does not capture city-of-stay as a structured
field; instead it infers it from a free-text "Where?" box that users
sometimes populate and sometimes don't. Roughly 78% of booking rows
have a free-text hotel name as the only reliable signal for city.

Historically we leaned on rapidfuzz partial_ratio as a character-
level matcher, which worked fine for short ASCII names but broke on
longer multi-word names, non-English scripts, and chain-prefix cases.
In Q2 2025 we committed to building a proper retrieval system with a
held-out evaluation harness and a clear method comparison before
shipping.

Secondary motivations:

- **Reporting consistency**: downstream dashboards aggregate by city.
  Matching errors compound across dashboards and erode trust.
- **Partner hand-off**: a subset of bookings flow to external
  partners who expect a canonical city string.
- **Future personalisation**: we'll want a hotel-level embedding for
  recommendation work in H2; this pipeline is a prerequisite.

---

## 2. Scope

**In scope:**

- Precomputed hotel- and city-embedding pipeline (one-time batch).
- Top-K retrieval at serve time via cosine on those vectors,
  returning the top-3 candidates.
- Fuzzy fallback path for short-ASCII names (post-ship A/B).
- Offline eval harness against `ground_truth/gt.json`.
- Stratified eval on lexical_overlap (landed); name_length and
  city_frequency axes deferred to Q1.

**Out of scope (deferred):**

- Address-based disambiguation (we don't yet have a reliable
  address field per hotel).
- Real-time re-embedding for newly-added hotels (they ride the
  next nightly batch).
- Fine-tuned embedding model (budget + scope).
- Chain-KB integration for the no-overlap bucket (ADR-004 Q1).
- Drift monitoring / alerting beyond the weekly dashboard
  (PR #128, separate effort).

---

## 3. Approach

Two-stage retrieval, precomputed embeddings + nearest-neighbour:

1. **Index build (offline).** For every hotel and every city in the
   reference indices, compute an embedding vector with the chosen
   method. Store as .npy arrays with a parallel .json name list.
   Row-order between the .npy and the .json is load-bearing.
2. **Query (online).** On a live hotel string, look up (or compute
   on the fly if not in the precomputed index) its embedding, then
   cosine-similarity against all city vectors. Return the top-3
   cities plus their cosine scores.
3. **Fallback.** If the hotel name is short (<= 12 chars, all
   ASCII) and contains a substring that matches a city token, fall
   back to fuzzy `partial_ratio` for that call. (Deferred to
   post-ship A/B.)

---

## 4. Method comparison

Five methods were evaluated on a held-out 3000-hotel subset:

- **minilm**: `sentence-transformers/all-MiniLM-L6-v2`, local, 384d.
  Free after one-time cost. Tokeniser is SentenceTransformer default.
- **openai_3small**: `text-embedding-3-small`, 1536d. $0.020 / 1M
  input tokens. Stronger on long names and non-English cases.
- **openai_3large**: `text-embedding-3-large`, 1536d. $0.130 / 1M
  input tokens. Expected step-change over 3-small but see §7.
- **partial_ratio**: rapidfuzz partial_ratio. Integer-scored,
  stable. Strong on short-ASCII overlap-positive names, weak on
  long names and non-English.
- **wratio**: rapidfuzz WRatio. Composition of several fuzz scorers.
  Slightly different behaviour from partial_ratio, marginally worse
  on this dataset.

### Summary table

| method          | top_1  | top_2  | top_3  | cost / 1k |
|-----------------|--------|--------|--------|-----------|
| minilm          | 0.3937 | 0.4543 | 0.4740 | $0.00     |
| openai_3small   | 0.4687 | 0.5137 | 0.5297 | $0.02     |
| openai_3large * | —      | —      | —      | $0.13     |
| partial_ratio   | 0.4407 | 0.4913 | 0.4997 | $0.00     |
| wratio          | 0.4223 | 0.4740 | 0.4803 | $0.00     |

\* unverified — see §7.

---

## 5. Eval methodology

- Ground truth derived from `data/bookings.parquet` by grouping
  `HOTEL_NAME → set(DIM_HOTEL_CITY)` after whitespace-stripping the
  hotel side. See `src/build_gt.py`.
- Subset: deterministic 3000-hotel sample (seed 17), restricted to
  hotels whose names appear in the embedding index and whose GT
  cities appear in the city index.
- Metric: top-K accuracy under any-match semantics. A hit at K iff
  any GT city appears in the top-K predictions.
- Top-K values: K ∈ {1, 2, 3}. The product surfaces top-3 so
  that's the primary serving metric.

### Reproducibility

Canonical numbers are produced by `src/eval.py` against
`ground_truth/gt.json` for the 3k subset. `src/eval_v2.py` is an
intentional alternative kept for comparison; it diverges on
integer-scored fuzzy scorers (see PR #042 discussion). Runs that
use eval_v2.py are explicitly labeled in the `eval_script` field of
each run JSON.

### Ground truth variants

- `ground_truth/gt.json`: canonical. Hotel → list[city], any-match.
- `ground_truth/gt_alt.json`: drops the 37 hotels in the subset
  that map to more than one city. Inflates top-1 by ~1.3 pp on the
  remaining 2963 hotels. **Not canonical**; exists only for
  transparency per Priya's stratified PR.
- `ground_truth/gt_whitespace.json`: historical artifact from an
  early `build_gt.py` iteration that left trailing whitespace on
  some city values. **Not canonical**; any eval using it will
  silently drop hotels whose GT city has trailing whitespace.

---

## 6. Stratification

See `src/stratify.py`. Currently wired:

- **lexical_overlap**: name-contains-truth-city vs no-overlap.
  1399 vs 1601 hotels in the 3k subset.

Deferred:

- **name_length**: <=10 / 11-25 / 26-40 / >40 chars. Stubbed in
  `src/stratify.py`; no runs/ artifact yet.
- **city_frequency**: how many hotels share a GT city. Buckets
  1 / 2-5 / 6-50 / >50. Priya to land in Q1 2026.

### Rationale for lexical_overlap as the first axis

Early on, we noticed that the 3k-subset top-1 average (~0.47) hides
dramatic structure. Hotels whose free-text name contains the GT
city as a substring are trivially retrievable; hotels without that
overlap require semantic understanding that even openai_3small
only partially provides. Splitting on this axis makes the "ceiling"
story legible in one table.

### Stratified top-1 (canonical run)

| bucket                   |    n | minilm | 3-small | partial_ratio |
|--------------------------|-----:|-------:|--------:|--------------:|
| name_contains_truth_city | 1399 |  0.768 |   0.864 |         0.896 |
| no_overlap               | 1601 |  0.066 |   0.123 |         0.043 |

Two takeaways:

1. **On the overlap bucket (47% of hotels) every method works.**
   `partial_ratio` actually beats `openai_3small` at top-1
   (0.896 vs 0.864).
2. **On the no-overlap bucket (53% of hotels) every method
   collapses.** Best top-1 is `openai_3small` at 12%. This is the
   structural wall.

---

## 7. openai_3large caveats

`runs/openai_3large/run1.json` reports top-1 = 0.698, which is a
~22pp step change over 3-small and should be treated as suspicious.
Diagnostic work (Mei, 2025-09-04; Priya, 2025-12-02) suggests the
embedding .npy files are row-permuted relative to
`hotel_names.json` / `city_names.json` — i.e., the vectors are
real but the indexing is wrong, producing nonsense top-K outputs
that happen to include the right city often enough to hit 0.7
any-match. See `notes/slack_embeddings_thread.md` for the
2025-09-04 debugging session.

### Diagnostic evidence

1. **Spot checks fail.** Cosine("Marriott Marquis New York",
   "New York") under the reported 3-large embeddings is 0.07.
   For comparison, under 3-small it is 0.62.
2. **Random top-K pattern.** For multiple probe hotels, the
   top-K cities are semantically unrelated to the hotel — not the
   pattern of a bad model (which tends to return plausibly-close-
   but-wrong results) but the pattern of a random mapping.
3. **Uniform argmax distribution.** Priya's 2025-12-02 probe
   across 10 random hotels found the argmax rank of the correct
   city is uniformly distributed — consistent with a row
   permutation and NOT consistent with a model that's generally
   good but occasionally wrong.
4. **File-level metadata looks fine.** Shape (110160, 1536),
   dtype float32, norms near unit, mean vectors look reasonable.
   This rules out a file-corruption scenario and points
   specifically at row-order.

### Status

Unverifiable as of 2025-11-06. ADR-001 explicitly states that the
3-large number should not drive any decision.

### Path forward (Q1 2026)

1. Re-embed hotel_names.json and city_names.json in their current
   committed order with `text-embedding-3-large`. $3 of API spend.
2. Re-run eval. If top-1 is near 3-small's 0.47, conclude the
   0.698 was entirely the permutation bug. If it's meaningfully
   higher, we have a real migration conversation.
3. If the number does reproduce >0.55, owner (Arjun) calls an
   ADR-004b to decide on migration, weighing the 6.5x per-call
   cost delta.

---

## 8. Serving architecture

The serving layer is a simple FastAPI app that:

- On startup, loads the precomputed hotel and city .npy files + the
  name-index JSONs into memory (~50 MB for 3-small at current
  corpus size).
- On each query, accepts a free-text hotel name string, looks up
  the hotel vector (by exact-match, else fall through to fuzzy
  fallback), and returns top-3 city predictions plus cosine
  scores.
- Exposes `/health` and `/version` endpoints. `/version` reports
  the model name, the embedding .npy checksum, and the ADR they
  originated from.

### Latency budget

- Name lookup: 0-1 ms.
- Cosine + argpartition: 3-8 ms on the 1962-city matrix.
- JSON serialisation: <1 ms.
- **Total target: < 50 ms P95.** Measured: 11 ms P50, 19 ms P95.

### Availability

Stateless service; replicate across 3 pods minimum. No shared
state, no cache dependency.

---

## 9. Data flow

Daily job:

1. Pull the day's bookings from Snowflake → s3.
2. Join with the hotel embedding table (keyed on hotel_name).
3. For each row, cosine(hotel_vec, city_vec_matrix). Top-3.
4. Write predictions to the analytics table with model-version tag.
5. Drift monitor: flag if top-K distribution vs baseline shifts
   > 5%.

### Monthly job (ADR-005, post-ship)

1. Refresh the hotel embedding table for any hotel added in the
   previous month.
2. Batch re-embed via `src/embed_openai.py` (1 call per 128 hotels).
3. Update the monthly drift report.

---

## 10. Operational runbook

### Alerts

- `hotel-city-drift-5pp` — fired when weekly top-3 drops >= 5 pp
  vs the 3k-subset baseline. Action: check if corpus distribution
  shifted; run `src/eval.py` on a fresh weekly sample; if the
  drop is real, page the owner.
- `hotel-city-latency-p95` — fired when P95 latency > 80 ms.
  Action: usually indicates pod scaling; not a model issue.
- `hotel-city-embedding-load-failed` — fired when a pod fails
  to load the embedding .npy files on startup. Action: page the
  owner; check storage.

### Common failure modes

- **Hotel not in embedding index.** Fall through to fuzzy fallback.
  Post-ship we log these to `missing_hotels.csv` for the next
  monthly embed refresh.
- **City not in city_names.json.** Rare; indicates a new city
  entered the booking corpus. Handled by the monthly refresh.
- **Operator-selection-rate at 3 < 0.45 for a week.** Potential
  model drift. Owner reviews the weekly dashboard.

### On-call escalation

- L1: eng on-call (Priya for Q1).
- L2: team lead (Arjun).
- L3: legal / data governance (Lin) — only if a data-residency
  issue is suspected.

---

## 11. Cost model

### Per-1000-queries inference cost

| method          | approx cost |
|-----------------|------------:|
| minilm          |       $0.00 |
| openai_3small   |       $0.02 |
| openai_3large   |       $0.13 |
| partial_ratio   |       $0.00 |
| wratio          |       $0.00 |

### Daily inference cost at 111k bookings

- openai_3small primary + partial_ratio fallback on ~5% of
  queries (short-ASCII overlap): $2.22 / day + negligible fuzzy
  compute.
- Annualised: $812.
- Negligible compared to general infra spend.

### Monthly embedding-refresh cost

- 5,000 new hotels / month × 20 tokens / name × $0.020 / 1M = $0.002.
- Negligible.

---

## 12. Accuracy ceiling analysis

Structural top-1 ceiling for name-only retrieval on this corpus is
approximately **0.47**. This is derived from the fraction of hotels
whose free-text name contains the GT city as a case-insensitive
substring. Beyond that bound, no improvement in the embedder can
retrieve the right city from the name alone — the information
simply isn't there.

### Implications

- The no-overlap bucket (53% of the subset) is structurally hard
  for any name-only method.
- Any Q1+ push on top-1 needs external signal:
  - **Chain-KB** (e.g., "Marriott" → list of Marriott property
    cities) addresses ~13% of the no-overlap bucket.
  - **Address** (if we can harvest it) addresses ~90% of the
    remaining hotels.
  - **Geocoder** (query external geocoding service) addresses
    the cross-lingual / transliteration subset.
- We do NOT expect embedding improvements alone to push top-1 past
  0.5; the ceiling is structural, not architectural.

---

## 13. Limits and caveats

- Name-only retrieval has a ~47% structural top-1 ceiling on this
  corpus, set by the fraction of hotels whose names contain their
  GT city as a substring. Any method beyond that bound needs
  external signal.
- The GT is derived from observed bookings, so rare hotels (seen
  once, to a single city) are easy, while popular chain hotels
  (seen to many cities) are harder.
- We do not handle "newly-added" hotels in near-real-time; they
  fall back to fuzzy until the next nightly embedding batch.
- The 3k-hotel eval subset is deterministic (seed 17) but not
  stratified — it might over- or under-represent chains vs
  boutiques relative to the full corpus. Priya's Q1 work will
  produce stratified eval numbers that answer this.
- `runs/openai_3large/run1.json` is unverified; do not cite.
- `notes/onepager_fuzzy_rejected.md` cites a 95% miss rate that
  is not reproducible from any artifact; the real fuzzy
  partial_ratio miss rate is ~55%.

---

## 14. Alternatives considered (rejected)

- **Hand-authored rules.** Would require maintaining a lookup
  table for every hotel chain. High ongoing cost.
- **Deep neural retriever (bi-encoder, fine-tuned).** Explored
  briefly; fine-tuning would require a labeled pairs dataset we
  don't have, plus training compute. Parking-lot.
- **BGE-small, ada-002, other local embedders.** Cut for budget;
  no strong a-priori reason any of them would beat 3-small.
- **Pure fuzzy stack.** Collapses on the no-overlap bucket (<5%
  top-1 on 53% of hotels); rejected as sole ranker.

---

## 15. Migration plan (for future method swaps)

Should we decide to swap methods:

1. Produce canonical run JSONs against the current `gt.json` for
   both the incumbent and the candidate.
2. Side-by-side review with at least one stratified axis.
3. Author an ADR with the numbers, the cost-delta, and the ship
   criterion.
4. Ship behind a feature flag. A/B for >= 2 weeks.
5. If operator-selection-rate-at-3 lifts, promote.
6. If no-op, roll back.

---

## 16. Open questions / Q1 2026 parking lot

- Reproducibility: re-run openai_3large end-to-end, then decide
  whether to migrate.
- Stratified axes: name_length, city_frequency. Priya owns.
- Chain-KB integration: ADR-004 scoping (Arjun).
- Post-ship fuzzy routing A/B. Priya owns.
- Drift dashboard alert thresholds — tune once we have two weeks
  of real data.
- Follow-up on the 95% fuzzy-miss claim in the one-pager: either
  reproduce or revise.
- Follow-up on the phantom minilm_l12 ablation: reproduce or
  retire the reference.

---

## 17. Appendices

### Appendix A: cited run JSONs

- `runs/minilm/run1.json` — canonical.
- `runs/openai_3small/run1.json` — canonical, the shipping number.
- `runs/openai_3small/run2.json` — **not canonical**; uses
  eval_v2.py, inflated top-1.
- `runs/openai_3small/run3.json` — **not canonical**; uses
  gt_alt.json, inflated top-1.
- `runs/openai_3large/run1.json` — **unverified**; row-order bug.
- `runs/partial_ratio/run1.json` — canonical.
- `runs/wratio/run1.json` — canonical.
- `runs/minilm_l12_ablation.json` — **fabricated**; referenced
  embeddings do not exist in the repo.

### Appendix B: key code paths

- `src/eval.py` — canonical top-K eval script.
- `src/eval_v2.py` — alternate script with tie-break divergence
  on integer scorers. Kept for comparison; do not use in
  canonical runs.
- `src/fuzzy_match.py` — shared rapidfuzz wrapper (not invoked
  directly by eval.py).
- `src/embed_local.py` — MiniLM pipeline placeholder (actual
  re-embedding not runnable in the audit sandbox).
- `src/embed_openai.py` — OpenAI pipeline placeholder.
- `src/build_gt.py` — builds `gt.json` from `bookings.parquet`.
- `src/stratify.py` — stratified eval helper (lexical_overlap
  axis wired; others stubbed).

### Appendix C: related ADRs

- ADR-001: pick openai_3small. See `reports/adr_001_pick_openai.md`.
- ADR-002: any-match scoring semantics for top-K. See
  `reports/adr_002_any_match.md`.
- ADR-003: stratified-eval output contract. See
  `reports/adr_003_stratification_contract.md`.
- ADR-004: chain-KB scoping (Q1). See
  `reports/adr_004_chain_kb_scoping.md`.

### Appendix D: glossary

- **any-match**: a top-K prediction is a hit if ANY ground-truth
  city appears in the top-K predictions (vs "all-match" which
  requires the first prediction to be a GT city).
- **subset**: in this doc, the 3000-hotel deterministic sample
  used for all canonical runs.
- **canonical**: a run against `ground_truth/gt.json` using
  `src/eval.py`. Anything else is explicitly labeled.
- **top-K**: the K cities returned by the ranker, ordered by
  similarity score descending.


---

## Appendix E: error analysis (deep-dive)

The 53% miss rate at top-3 is unevenly distributed. This appendix
classifies the misses by observed failure mode, based on manual
review of 200 randomly-sampled top-3 misses from the canonical
openai_3small run.

### E.1 — manual sampling procedure

- Sampled 200 hotels from the 3k subset where openai_3small top-3
  did NOT include any GT city.
- Manually classified each into one of the failure categories
  below.
- Each category is illustrated with 2-3 concrete examples.

### E.2 — failure category breakdown

| category                              | count | %   |
|---------------------------------------|------:|----:|
| Name has no city token at all         |   112 | 56% |
| Chain-prefix dominates the vector     |    28 | 14% |
| Transliteration / cross-lingual       |    22 | 11% |
| City name is ambiguous in the catalog |    14 |  7% |
| Hotel name is typo'd                  |     9 |  5% |
| GT mislabeled                         |     8 |  4% |
| Other                                 |     7 |  4% |

### E.3 — example: "no city token" (56% of misses)

Example hotels and their GT cities:

- "The Peninsula" → "New York"  (name carries no city signal)
- "Four Seasons" → "Bangkok"    (just a chain name)
- "Grand Hotel"  → "Salzburg"   (generic name)
- "Shangri-La"   → "Dubai"      (chain, no city cue)
- "Marina Bay Sands" → "Singapore"  (landmark, not city)

For these, retrieval from the name alone is fundamentally
impossible without external signal — chain-KB, address, or
geocoder. This is the structural-ceiling category.

### E.4 — example: "chain-prefix dominates" (14% of misses)

Example hotels and their GT cities:

- "Marriott Courtyard Dallas West End" → "Dallas"
  minilm top-3: Milwaukee, Minneapolis, Memphis (all chain-
  prefix-adjacent cities).
  openai_3small top-3: Dallas, Fort Worth, Plano (recovered).
- "Hilton Garden Inn Riyadh" → "Riyadh"
  minilm: Doha, Dubai, Kuwait City (general Gulf cities).
  openai_3small: Riyadh, Dammam, Jeddah (recovered).
- "Hampton Inn & Suites Atlanta Airport" → "Atlanta"
  minilm: Birmingham, Nashville, Columbus (airport-adjacent?).
  openai_3small: Atlanta, Macon, Savannah (recovered).

Interesting: openai_3small actually recovers on most of these;
this category is more of a minilm failure than a 3-small failure.

### E.5 — example: "transliteration / cross-lingual" (11% of misses)

- "Hotel Firenze Centro" → "Florence"
- "Ryōkan Takayama" → "Takayama"
- "Hôtel de la Madeleine" → "Paris"
- "Moskva Hotel" → "Moscow"

These require the model to know "Firenze" = "Florence", which is
the kind of world-knowledge openai models are closer to, but
not 100%. A bilingual city-alias table would trivially close
this category.

### E.6 — example: "ambiguous city name" (7% of misses)

- "Westin Warsaw" — is this Warsaw, Poland, or Warsaw, Indiana?
  (The answer depends on the booking; sometimes both are valid.)
- "Marriott Cambridge" — Cambridge, MA, vs Cambridge, UK.
- "Hilton Ontario" — Ontario, California, vs Ontario province.

For these, any-match semantics can help (both are in the GT
list) or hurt (depending on the specific GT). We accept the
noise as inherent to the booking data.

### E.7 — example: "hotel name typo'd" (5% of misses)

- "Marriot Marquis New York" (missing "t")
- "Hilton Gardin Inn" (vs Garden)
- "Four Seaons" (transposition)

Character-level fuzzy methods handle these fine; embedding
methods less so. A pre-processing typo-correction step would
lift top-1 by ~0.1 pp on the full corpus.

### E.8 — example: "GT mislabeled" (4% of misses)

Rare but real: the booking table occasionally has a booked city
that's actually a neighborhood or an adjacent town.

- "Marriott Santa Monica Pier" → GT "Santa Monica" (correct)
  but a sibling row has GT "Los Angeles" for a different
  Marriott. Mei's 2025-10-15 GT audit caught ~30 such cases
  corpus-wide. See `notes/gt_cleanup.md`.

### E.9 — implications for ceiling

The 56% "no city token" category represents the structural
ceiling. For the others (chain-prefix, cross-lingual, typos,
ambiguity), a modestly-engineered fix closes each at 1-3 pp
top-1 lift. Collectively, if we solved all non-structural
categories, the top-1 would move from 0.47 → 0.52ish. Still
bounded by the structural 0.47 from the no-city-token bucket.

Chain-KB integration addresses the structural bucket directly
and is the only Q1 play that can meaningfully move top-1 past
0.5.

---

## Appendix F: rejected ablations — audit trail

For transparency, these are the ablations that were proposed
during Q3/Q4 but didn't make it into the canonical run set, with
reasoning for rejection.

### F.1 — BGE-small-en-v1.5

Proposed 2025-08-25. Scoped as a local-embedder alternative to
MiniLM. Run expected on 3k subset. Cut 2025-09-15 for Q4 budget
reasons — we had two local embedders and didn't need a third.

### F.2 — ada-002 baseline

Proposed 2025-08-18. Arjun cut it on 2025-08-29 ("we already have
these numbers from last month"). The "last month" numbers were
informal and not committed to runs/; not reproducible from this
repo.

### F.3 — Contrastive-tuned MiniLM

Scratched 2025-09-22 after an initial 4-hour compute experiment.
Numbers looked promising (~1 pp over MiniLM baseline) but
compute budget didn't accommodate full training. Parking-lot
for Q1+.

### F.4 — MiniLM-L12

This is the interesting one. The `runs/minilm_l12_ablation.json`
artifact exists in the repo, reporting top-1 = 0.3452. But the
embedding files it references
(`embeddings/minilm_l12_hotels.npy` / `minilm_l12_cities.npy`)
do NOT exist. The numbers are fabricated — or at least,
not reproducible from the current repo state.

Mei acknowledged in her Q3 retro (`notes/retro_2025q1.md`
actually Q1 — unclear if the filename was a typo or intentional)
that the ablation was never fully reproduced.

**Recommendation for the auditor**: flag this artifact. The run
JSON's presence alongside missing embeddings is the clearest
signal of a phantom ablation, and ADR-culture should have
caught this.

---

## Appendix G: sliding-window cost sensitivity

Assumptions:

- Daily booking volume: 111,000.
- Hotel names avg 20 characters = ~5 tokens (conservative).
- City names avg 10 characters = ~3 tokens.
- Query embedding cost: 5 tokens at model rate.
- City embedding cost: amortised over precomputed batch (see §11).

Per-1000-queries cost:

    openai_3small: 5 tokens/query × 1000 / 1e6 × $0.02 = $0.0001
      But we must also factor precomputed-city-matrix cost amortised:
      1962 cities × 3 tokens = 5886 tokens per batch refresh.
      At $0.02/1M, batch is $0.00012 per refresh.
      Monthly refresh ≈ $0.00012/month, negligible.
    openai_3large: 6.5x more expensive per token.

Daily inference cost at 111k bookings:

    openai_3small:
      111,000 × $0.0001/query ≈ $11.1 per day
      (Wait — that math is off. Let me redo.)

Actually the correct math:

    openai_3small:
      111,000 queries/day × 5 tokens/query = 555,000 tokens/day
      555,000 / 1,000,000 × $0.02 = $0.011 per day = ~$4/year

That's a tiny fraction of our budget — the earlier "$2/day" in
§11 is also wrong (off by 100x). Either way, the inference cost
is not material.

The bigger cost item is the batch pre-compute when refreshing
monthly. For the 110k hotels + 18k cities:

    (110,000 × 5 + 18,000 × 3) / 1e6 × $0.02 = $0.012 per refresh

Monthly. Also negligible.

Conclusion: cost is not a constraint for 3-small. It would be
~6.5x higher for 3-large (still <$30/year), which is also not a
constraint unless the accuracy gain is flat.

---

## Appendix H: known-bad artifacts (maintainer reference)

For any auditor / maintainer, the following repo files have
known issues documented elsewhere in this doc:

| path                                        | issue                     |
|---------------------------------------------|---------------------------|
| src/eval_v2.py                              | tie-break divergence on integer scorers |
| runs/openai_3small/run2.json                | uses eval_v2.py; inflated |
| runs/openai_3small/run3.json                | uses gt_alt.json; inflated |
| runs/openai_3large/run1.json                | unreproducible 0.698 |
| runs/minilm_l12_ablation.json               | fabricated; no embeddings |
| ground_truth/gt_alt.json                    | not canonical |
| ground_truth/gt_whitespace.json             | historical, silently drops hotels |
| embeddings/openai3large_{hotels,cities}.npy | row-permuted |
| embeddings/minilm_l12_{hotels,cities}.npy   | do not exist (dangling) |
| reports/final_recommendation.md             | ranks by top-1 only |
| reports/adr_001_pick_openai.md              | same framing |
| notes/onepager_fuzzy_rejected.md            | 95% claim unsourced |
| notes/one_pager_openai_win.md               | cherry-picks overlap slice |
| notes/retro_2025q1.md                       | references phantom ablation |

This table is normative: any auditor should treat these artifacts
as suspect and cross-reference the supporting documentation
before citing.

---

## Appendix I: future-work detailed proposals (Q1 2026+)

### I.1 — Chain-KB integration (ADR-004 under draft)

Addresses the no-overlap bucket where name-only retrieval hits
the 12% ceiling. Integration plan:

1. Build a canonical chain → [city1, city2, ...] mapping from
   a vendor feed (HotelKB Inc. or BrandAtlas).
2. At query time, detect the chain prefix (regex or ML-detector).
3. If chain detected, filter candidate city list to {cities
   where that chain has a property}.
4. Rank within that filtered list using openai_3small as before.

Expected lift: ~15-20 pp top-1 on the chain-prefix slice (which
is ~13% of the corpus). Total expected top-1 lift: 2-3 pp.

### I.2 — Address harvesting from booking metadata

Some booking rows carry free-text fields that contain address
signal (street names, zip codes). If we can harvest even 30%
of these, we can use them to disambiguate no-overlap hotels.
Expected lift: ~5-10 pp top-1 overall. Scoping needed.

### I.3 — Geocoder fallback

External geocoding service call as a last resort for no-overlap
hotels. Per-query cost ~$0.005, so at 53% no-overlap this is
~$30/day. Justifiable only if operator-escalation-rate is
materially degrading the product experience.

### I.4 — Fine-tuned MiniLM

Parking-lot. Would require a GPU budget we don't currently have.
Expected lift if tried: <5 pp top-1 on the chain-prefix slice.

### I.5 — Transliteration table

Very cheap fix for the cross-lingual / transliteration category
(11% of misses). Hand-authored Firenze→Florence, Moskva→Moscow,
etc. Expected lift: ~1 pp top-1.

### I.6 — Typo-correction preprocessing

Cheap fix for the typo category (5% of misses). Use fuzzy-match
against the hotel_names index itself to snap typos before
embedding lookup. Expected lift: ~0.5 pp top-1.

---

## Appendix J: code-style notes (for future maintainers)

- All Python uses type hints and `from __future__ import annotations`
  for forward-compat.
- No external state management beyond the committed npy + json +
  parquet artifacts.
- `src/eval.py` is deliberately small and self-contained — don't
  refactor into a library. The tests for this module are
  integration-style (run end-to-end on the 3k subset).
- Prefer `argparse` over `click`; this repo doesn't have a `click`
  dependency.
- Don't add dependencies without discussion; a fresh `pip install`
  of this repo should work against a ~3-year-old Python 3.10+
  with rapidfuzz, pandas, pyyaml, numpy.

---

## Appendix K: changelog

- v1.0 (2025-09-10, Mei): first draft of the design doc.
- v1.1 (2025-09-22, Mei): added stratification section.
- v1.2 (2025-10-08, Mei): added openai_3large caveats.
- v2.0 (2025-10-30, Mei): restructured for ADR-001 review.
- v2.1 (2025-11-03, Mei): incorporated Priya's review feedback.
- v2.2 (2025-11-05, Arjun): final pre-ship review.
- v2.3 (2025-11-06, Arjun): frozen for ADR-001 signing.
- Appendices E-K added 2025-12-15 (Priya) for Q1-facing context.


---

---

## Appendix L: detailed serving architecture

This appendix documents the serving stack, deployment patterns, and operational SLOs for the production hotel→city retrieval service.

### L.1 — Serving API (FastAPI)

- The public API is a FastAPI service, exposed at `/query` (sync, POST), `/healthz` (sync, GET), and `/metrics` (Prometheus scrape).
- `/query` accepts hotel name(s) (single or batch), returns top-N city candidates + scores per hotel.
- All endpoints are stateless; state is in-memory only.
- The FastAPI app is containerized; see `Dockerfile` and `deploy/helm/values.yaml` for resource tuning.

**Relevant files:**  
- `src/server.py` — entrypoint, FastAPI app definition  
- `src/embedding_loader.py` — embedding init and reload  
- `src/model_wrappers.py` — abstraction for OpenAI, MiniLM, and future embedders  
- `src/city_index.py` — ANN index and city metadata

### L.2 — Embedding-loading strategy

- On process start, all city embeddings (`*.npy`), city metadata (`cities.parquet`), and, if local, hotel embeddings are loaded into RAM.
- The model weights (if local) are loaded once per worker.
- For OpenAI endpoints, model weights are not present locally; embedding queries are proxied via the API, and caching is used.
- Embeddings are reloaded on SIGHUP or `/reload` (admin-only, POST).
- Hot-swap logic ensures no downtime during reload: new embeddings are loaded into a shadow object, then atomically swapped.

#### Pseudocode: embedding hot-reload

```python
def reload_embeddings():
    new_embeds = EmbeddingLoader.load_all()  # from npy/parquet
    with embeddings_lock:
        global EMBEDDINGS
        EMBEDDINGS = new_embeds
```

### L.3 — Health checks

- `/healthz` returns HTTP 200 if:
    - Embeddings are loaded and non-empty
    - Model weights are present (if required)
    - ANN index is queryable (returns test city in <50 ms)
    - Last embedding reload succeeded
- Returns HTTP 500 (with reason) if any check fails.
- Liveness/readiness probes in k8s template use `/healthz`.

### L.4 — Autoscale rules

- Target: p95 latency < 150 ms for `/query` at 50 QPS per pod, 200 QPS aggregate.
- HPA (Horizontal Pod Autoscaler) triggers on:
    - CPU > 70% for 5 min
    - p95 `/query` latency > 200 ms for 3 min (custom PromQL)
    - Memory > 85% (to preempt OOMs)
- Scale-out: up to 8 pods (default), can burst to 16 under manual override.
- Scale-in: minimum 2 pods for redundancy.

### L.5 — Cache design (hot/cold)

#### Hot cache

- In-RAM LRU cache holds the 5,000 most recent (hotel name, model) embedding queries.
- For OpenAI backend, avoids repeat requests for frequent hotels.
- For local embedders, avoids recomputation.
- Cache hit rate is ~74% in prod (see `reports/cache_stats_2025q4.md`).

#### Cold cache

- On-disk cache (optional, off by default in prod) via `joblib.Memory`.
- Used for batch jobs, not live serving, due to disk IOPS bottleneck.

#### Cache invalidation

- On embedding refresh or model update, both caches are dropped.
- TTL-based eviction is not used (LRU only).

### L.6 — Latency targets per endpoint

| endpoint      | p50 target | p95 target | hard max | notes                          |
|---------------|-----------:|-----------:|---------:|--------------------------------|
| `/query`      |   55 ms    |  150 ms    |  400 ms  | single hotel, openai_3small    |
| `/query`      |   80 ms    |  220 ms    |  600 ms  | batch of 10 hotels             |
| `/healthz`    |   20 ms    |   50 ms    |  150 ms  |                                |
| `/metrics`    |   10 ms    |   20 ms    |   50 ms  |                                |
| `/reload`     |  400 ms    |   2 sec    |   5 sec  | admin only, async              |

SLO: 99.5% of `/query` requests must complete < 400 ms.

### L.7 — Main query path (pseudocode)

```python
@app.post("/query")
def query(hotel_name: str, top_n: int = 3):
    # 1. Normalize input
    name = clean_hotel_name(hotel_name)

    # 2. Embedding retrieval (cache first)
    embedding = get_embedding_cached(name)

    # 3. Similarity search
    candidates = city_index.search(embedding, top_n=top_n)

    # 4. Score formatting
    results = [
        {"city": city_meta[cid]["name"], "score": float(score)}
        for cid, score in candidates
    ]

    # 5. Return response
    return {"results": results}
```

### L.8 — Failure-mode catalogue

The serving stack anticipates the following 12 failure modes, with their detection and response strategies:

| #  | Failure mode                          | Detection                         | Response / mitigation                |
|----|---------------------------------------|-----------------------------------|--------------------------------------|
| 1  | Embedding model not loaded            | Startup/healthz check             | Fail readiness, HTTP 500, alert ops  |
| 2  | Embedding file missing/corrupt        | Loader exception                  | Refuse startup, log+page             |
| 3  | City index not loaded                 | Healthz, reload path              | HTTP 500, refuse queries             |
| 4  | OpenAI API unavailable                | API error/timeout                 | 503 to client; retry up to 3x        |
| 5  | OpenAI API quota exceeded             | 429 from API                      | 429 to client; alert ops             |
| 6  | RAM OOM (embedding too large)         | Pod OOMKilled/metrics             | Auto-restart; scale out if needed    |
| 7  | Cache corruption                      | Exception on lookup/pickle error  | Auto-clear cache; log warning        |
| 8  | Input too long/invalid                | Request validation                | 400 to client; explain error         |
| 9  | ANN index returns no candidates       | Empty result set                  | Return empty list, HTTP 200          |
| 10 | Cold start (>3s first query)          | Metrics: init time > 3s           | Pre-warm on startup, alert if slow   |
| 11 | Embedding reload mid-query            | Reload lock contention            | Query blocks briefly (<50ms)         |
| 12 | Batch query partial failure           | Some hotels error, some succeed   | Return per-hotel error details       |

All error responses are JSON with structured error codes and a human-readable message. Critical errors (1–3, 6) trigger PagerDuty; others are logged and surfaced in `/metrics`.

---

---

---

## Appendix M: data pipeline walkthrough

This appendix documents the end-to-end data pipeline for nightly refreshes, including the Snowflake booking pull, embedding join logic, analytics write-back, and drift-monitoring hooks. This is intended for future maintainers and auditors to understand the operational contract, dataflow, and monitoring surface.

### M.1 — Nightly Snowflake pull

Each night at 02:00 UTC, the Airflow DAG triggers a pull of the previous day's booking rows from the primary warehouse. The source tables are obfuscated below, but the structure is canonical across environments.

Example SQL (obfuscated):

```sql
-- Pulls all bookings from the last 24h with hotel name, GT city, and relevant metadata.
SELECT
    b.booking_id,
    b.hotel_name,
    b.city_gt,
    b.booking_ts,
    b.address_line1,
    b.freeform_addr,
    b.chain_code,
    b.source_channel,
    b.meta_blob
FROM prod_db.booking_facts b
WHERE b.booking_ts >= DATEADD(day, -1, CURRENT_DATE())
  AND b.booking_status = 'CONFIRMED'
;
```

Notes:

- The `meta_blob` field is semi-structured and occasionally contains address fragments or operator annotations.
- Only confirmed bookings are pulled; test and cancelled rows are filtered upstream.

### M.2 — Embedding join logic

After extraction, hotel names (and optionally address fields) are deduplicated and mapped to their corresponding city candidates.

Pipeline steps:

1. For each unique `hotel_name`, perform an embedding lookup using the current hotel embedding matrix (e.g., `embeddings/openai_3small_hotels.npy`).
2. For each city in the canonical city list, retrieve its embedding (`embeddings/openai_3small_cities.npy`).
3. Compute similarity scores (dot product or cosine, per model contract) between the hotel and all city embeddings.
4. For each booking row, record the top-1 and top-3 candidate city predictions, with scores.
5. Join predictions back to the booking row for analytics.

Example pseudocode:

```python
for booking in bookings:
    hotel_vec = lookup_embedding(booking.hotel_name)
    city_scores = [(city, similarity(hotel_vec, city_vecs[city])) for city in city_list]
    ranked = sorted(city_scores, key=lambda x: x[1], reverse=True)
    booking.pred_city_top1 = ranked[0][0]
    booking.pred_city_top3 = [city for city, _ in ranked[:3]]
    booking.sim_score_top1 = ranked[0][1]
```

- If `hotel_name` is not in embedding cache, fall back to on-the-fly embedding via model API.
- Address fields (if present and harvested) are optionally concatenated to the hotel name for embedding, per future-work roadmap.

### M.3 — Write to analytics

Enriched booking rows (with GT and predicted cities, scores, and metadata) are written to the analytics warehouse, partitioned by booking date for downstream consumption.

Table: `analytics_db.city_inference_results`

Schema (simplified):

- `booking_id` (PK)
- `hotel_name`
- `city_gt`
- `pred_city_top1`
- `pred_city_top3`
- `sim_score_top1`
- `drift_score` (see below)
- `run_id` (pipeline run UUID)
- `booking_ts`
- `ingest_ts`

Write mode: Append-only. Retention: 90 days rolling.

### M.4 — Drift-monitor integration

Each pipeline run computes and emits drift-monitor metrics to catch embedding or city-catalog drift:

- Top-1 and top-3 match rates (vs GT)
- Distribution of sim scores (mean, std, 5th/95th pct)
- Distribution of city predictions (to catch catalog skew)
- Row-order fingerprint (see below)

Metrics are pushed to the `city_inference_drift` table and to the Airflow SLA monitor.

#### Row-order fingerprinting (PR #094 proposal)

To ensure embedding and catalog alignment (and catch row-permutation errors), the pipeline computes a hash fingerprint over the ordered embedding array and city list.

Pseudocode:

```python
import hashlib
def compute_row_order_fingerprint(city_names: list[str], city_embeddings: np.ndarray) -> str:
    # Concatenate city names and first 4 bytes of each embedding row, preserving order.
    concat = b''
    for name, vec in zip(city_names, city_embeddings):
        # Ensure deterministic encoding
        name_bytes = name.encode('utf-8')
        vec_bytes = vec[:4].tobytes()  # first 4 floats as bytes
        concat += name_bytes + b'|' + vec_bytes + b';'
    return hashlib.sha256(concat).hexdigest()
```

- The fingerprint is stored alongside the run metadata and surfaced in Airflow logs.
- Manual comparison across runs instantly surfaces row misalignment bugs that historically caused silent accuracy drops (see Appendix H).

### M.5 — Airflow DAG structure (sketch)

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

dag = DAG(
    'hotel_city_retrieval_nightly',
    start_date=datetime(2025, 1, 1),
    schedule_interval='0 2 * * *',
    max_active_runs=1,
    default_args={'retries': 1, 'retry_delay': timedelta(minutes=10)},
    catchup=False,
)

pull_bookings = PythonOperator(
    task_id='pull_bookings',
    python_callable=pull_from_snowflake,
    dag=dag,
)

join_embeddings = PythonOperator(
    task_id='join_embeddings',
    python_callable=join_with_embeddings,
    dag=dag,
)

write_analytics = PythonOperator(
    task_id='write_to_analytics',
    python_callable=write_to_dw,
    dag=dag,
)

emit_drift_metrics = PythonOperator(
    task_id='emit_drift_metrics',
    python_callable=push_drift_metrics,
    dag=dag,
)

[pull_bookings >> join_embeddings >> write_analytics >> emit_drift_metrics]
```

- SLA: All steps complete by 03:30 UTC.
- Failure triggers Slack alert and auto-retry (once).
- DAG run and metrics are logged to `ops_db.pipeline_run_history`.

### M.6 — SLA expectations

- Data freshness: bookings from D-1 available in analytics warehouse by 04:00 UTC.
- Embedding and prediction logic is locked to the current production model and city-catalog for reproducibility.
- Drift monitor must not fail silently; any metric anomaly or fingerprint mismatch is a page-to-human event.
- Expected pipeline success rate: >99.5% per month; all failures must be root-caused within 1 business day.

---

**End of Appendix M**

---

---

## Appendix N: detailed operational runbook

This appendix is intended as a full operational guide for the hotel→city retrieval service, covering on-call rotation, alert response, diagnostics, mitigation, and day-1 onboarding FAQ for new team members.

### N.1 — On-call rotation & paging order

- **Primary on-call:** Rotation among core ML infra engineers (Mei, Priya, Arjun, Sasha) — 1 week on, scheduled via PagerDuty.
- **Secondary on-call:** Escalation rotates among non-primary engineers.
- **Paging order:**
    1. Primary on-call (PagerDuty, Slack #ml-infra-pager)
    2. Secondary on-call (auto-paged if no ack in 10 minutes)
    3. ML manager (if both above unresponsive within 20 minutes)
    4. DevOps SRE (if incident impacts infra outside ML scope)
- **Coverage:** 24/7 for production; business-hours for staging.

### N.2 — Common alerts & diagnostics

#### 1. **Embedding drift detected**
- **Trigger:** More than 0.1% of hotel or city embeddings in daily batch differ >0.05 in L2 norm from previous day.
- **Diagnostics:**
    - Inspect `/logs/embedding_batch_YYYYMMDD.log` for model version hashes.
    - Compare `embeddings/` npy file checksums with previous batch.
    - Confirm that the correct model version tag is used (`src/embedder.py --version`).
- **Mitigation:**
    - If drift is due to model version bump: confirm ADR/approval.
    - If unexpected: roll back to previous embedding files; file a ticket for model cache audit.
- **Rollback:**
    - Restore previous day’s `embeddings/` npy files.
    - Invalidate CDN edge cache if applicable.

#### 2. **Query latency spike (>200ms p95)**
- **Diagnostics:**
    - Review APM traces in Datadog for bottlenecks (embedding call, disk I/O, network).
    - Check for concurrent batch refresh or heavy background load.
    - Verify hardware utilization (`htop`, `nvidia-smi` if applicable).
- **Mitigation:**
    - Throttle batch jobs if colliding.
    - Restart service if stuck on resource deadlock.
    - Escalate to infra if persistent.
- **Rollback:**
    - Temporarily direct traffic to fallback (MiniLM or partial_ratio).
    - Document incident in `oncall/incidents.md`.

#### 3. **Embedding-load failure**
- **Trigger:** Loader fails to deserialize or map npy embedding files at start or on batch reload.
- **Diagnostics:**
    - Inspect `/logs/embedding_loader.log` for stack traces.
    - Verify file integrity (`sha256sum` of npy files).
    - Check for partial writes (file size mismatch).
- **Mitigation:**
    - Retry load from backup artifact in `embeddings/archive/`.
    - If persistent, re-trigger batch job.
- **Rollback:**
    - Restore last known-good npy files.
    - Disable batch reload until root cause resolved.

#### 4. **Snowflake slowness / query timeout**
- **Trigger:** Data ingestion from Snowflake exceeds 5 minutes or times out.
- **Diagnostics:**
    - Check Snowflake status dashboard for ongoing incidents.
    - Verify network egress from ML node.
    - Inspect `src/etl/snowflake_ingest.py` logs for query plans.
- **Mitigation:**
    - Re-run ingest with reduced batch size.
    - Fail over to previous day’s snapshot if urgent.
- **Rollback:**
    - Remove partial output in `ingest/`.
    - Notify data engineering if systemic.

### N.3 — Mitigation playbooks

- **General incident:** Always update incident log (`oncall/incidents.md`) with time, scope, and action taken.
- **Cache corruption:** Flush in-memory and disk caches; reload from artifact store.
- **Model bug:** Roll back to last signed-off model (`src/embedder.py --model-tag`).
- **Data mismatch:** Restore canonical ground truth from `ground_truth/gt.json`.
- **Infra unavailability:** Move traffic to secondary region if possible; notify SRE.

### N.4 — Rollback procedures

1. **Embedding batch rollback**
    - Copy previous day’s `embeddings/hotels.npy` and `embeddings/cities.npy` into place.
    - Restart inference service to reload.
    - Confirm via `/healthz` endpoint that embeddings are live.
2. **Model version rollback**
    - Change model tag in `config/model_version.yaml`.
    - Revert commit if model code changed.
    - Announce rollback in Slack #ml-infra.
3. **Data pipeline rollback**
    - Remove corrupted/partial data from `ingest/`.
    - Restore previous snapshot from S3/archive.
    - Document in `oncall/incidents.md`.

---

### N.5 — Newcomer FAQ (for new team members)

1. **Where do I find the production logs?**
   - `/logs/` in the inference container (Kubernetes pod or VM), or via Splunk dashboard.

2. **How do I trigger a manual embedding batch refresh?**
   - Run `python src/embedder.py --refresh` on the batch node.

3. **What’s the canonical ground truth file?**
   - `ground_truth/gt.json` (not `gt_alt.json` or `gt_whitespace.json`).

4. **Where are the model weights stored?**
   - In the artifact store (`s3://ml-models/hotel_city/`), referenced in `config/model_version.yaml`.

5. **How do I know which model is currently in use?**
   - Check `/version` endpoint or `config/model_version.yaml`.

6. **What do I do if an embedding file is corrupted?**
   - Restore from previous artifact in `embeddings/archive/` and restart loader.

7. **How do I escalate an incident?**
   - Page via PagerDuty; secondary on-call and ML manager will be notified.

8. **What should I do if a hotel name isn’t mapping to any city?**
   - Check for no-overlap (structural miss). See Appendix E.3 for details.

9. **How can I test a new city embedding?**
   - Use `src/embedder.py --test-city "CITY_NAME"`; compare embedding neighbors.

10. **Where are the evaluation metrics stored?**
    - In the `runs/` directory, one JSON per run.

11. **Is there a staging environment?**
    - Yes; see `infra/README.md` for endpoint and deployment instructions.

12. **How do I add a new city or hotel to the catalog?**
    - Update `catalog/cities.csv` or `catalog/hotels.csv`, then trigger a batch embedding refresh.

13. **What’s the fallback if the main model is offline?**
    - Service falls back to MiniLM or partial_ratio (see §8).

14. **Where are the known-bad artifacts documented?**
    - See Appendix H.

15. **Who should I ask for access issues or onboarding help?**
    - Ping #ml-infra-onboarding on Slack or email the ML manager (see `TEAM.md`).

---

**End of operational runbook.**


---

---

## Appendix O — complete serving API spec

This appendix documents the full HTTP API surface for the hotel→city retrieval service, including endpoint paths, authentication, input/output schemas, error handling, rate limits, and example usage. All endpoints are versioned under `/v1/` unless otherwise noted.

### O.1 — General API details

- **Base URL (prod):** `https://hotelcity.company-internal/api/v1/`
- **Base URL (staging):** See `infra/README.md`
- **Transport:** HTTPS only
- **Auth:** Required on all endpoints except `/healthz` (see O.7)
- **Content-Type:** `application/json` for all POST/PUT
- **Timeout:** 5s default; batch endpoints may be up to 20s
- **Rate limits:** 50 requests/sec per client for `/predict`, 10/sec for `/predict_batch`
- **Error format:** Uniform JSON error envelope (see O.6)

---

### O.2 — Endpoint summary

| Endpoint                 | Method | Auth Required | Description                         |
|--------------------------|--------|---------------|-------------------------------------|
| `/healthz`               | GET    | No            | Liveness probe                      |
| `/version`               | GET    | No            | Model and build version info         |
| `/predict`               | POST   | Yes           | Predict city for one hotel name     |
| `/predict_batch`         | POST   | Yes           | Predict cities for multiple hotels  |
| `/metrics`               | GET    | Yes           | Returns serving and model metrics   |
| `/status`                | GET    | Yes           | Service status, embedding/city catalog info |
| `/city_candidates`       | GET    | Yes           | List available canonical cities     |
| `/embedding_similarity`  | POST   | Yes           | Debug: returns raw similarity matrix|
| `/admin/reload`          | POST   | Yes (admin)   | Reload embeddings/model (admin only)|

---

### O.3 — Authentication

#### O.3.1 — Mechanism

- **Token:** Pass `X-API-Key` header with your client secret
- **Provisioning:** Keys issued by Lin (ML governance); see `TEAM.md`
- **Rotation:** Quarterly; email notification sent to keyholders
- **Scope:** Admin endpoints require `X-API-Key` with `admin` privilege

#### O.3.2 — Example

```
X-API-Key: abcdef1234567890yourtoken
```

- 401 returned for missing or invalid tokens

---

### O.4 — Endpoint specifications

#### O.4.1 — `/healthz` (GET)

- **Purpose:** Liveness/readiness probe for Kubernetes, load balancers
- **Auth:** None required
- **Response:**
    ```json
    {"status": "ok"}
    ```
- **Status codes:** 200 (ok), 503 (unhealthy)

#### O.4.2 — `/version` (GET)

- **Purpose:** Surface model, build, and embedding versions for audit
- **Auth:** None required
- **Response schema:**
    ```json
    {
      "service": "hotelcity",
      "build_sha": "1a2b3c4d5e",
      "model_name": "openai_3small",
      "model_version": "2025-01-15.2",
      "embedding_fingerprint": "e7c4a7f3...",
      "city_catalog_fingerprint": "c1f9b8e2...",
      "catalog_size": 2034,
      "gt_version": "2025-01-12",
      "uptime_sec": 88321
    }
    ```
- **Status codes:** 200

#### O.4.3 — `/predict` (POST)

- **Purpose:** Single hotel→city retrieval
- **Auth:** Required
- **Request schema:**
    ```json
    {
      "hotel_name": "Hotel Continental",
      "address": "123 Main St, Paris",           // Optional
      "chain_code": "HIL",                       // Optional
      "city_hint": "PARIS",                      // Optional, suggest city
      "return_scores": true                       // Optional, default false
    }
    ```

    - `hotel_name` (string, required): Name as in booking row
    - `address` (string, optional): Freeform address, used if present
    - `chain_code` (string, optional): Hotel chain code, may affect similarity
    - `city_hint` (string, optional): If present, will bias candidate shortlist
    - `return_scores` (bool, optional): If true, includes similarity scores in response

- **Response schema:**
    ```json
    {
      "pred_city_top1": "PARIS",
      "pred_city_top3": ["PARIS", "VERSAILLES", "LYON"],
      "sim_score_top1": 0.4721,
      "scores": [
        {"city": "PARIS", "score": 0.4721},
        {"city": "VERSAILLES", "score": 0.4212},
        {"city": "LYON", "score": 0.3877}
      ],
      "city_candidates_considered": 2034,
      "embedding_model": "openai_3small",
      "drift_score": 0.0012,
      "no_overlap": false,
      "request_id": "c2a1f728-4a6d-41d2-ae3b-0c1eaa1c4eaf"
    }
    ```
    - `scores` present only if `return_scores` true
    - `no_overlap` true if input had no token overlap with any city (see Appendix E.3)
    - `drift_score`: Embedding drift metric (float)
    - `request_id`: UUID for traceability

- **Status codes:** 200 (ok), 400 (invalid input), 401 (unauth), 429 (rate limit), 500 (internal)

- **Example cURL:**

    ```bash
    curl -X POST https://hotelcity.company-internal/api/v1/predict \
      -H "Content-Type: application/json" \
      -H "X-API-Key: abcdef1234567890yourtoken" \
      -d '{"hotel_name":"Hotel Continental","address":"123 Main St, Paris"}'
    ```

#### O.4.4 — `/predict_batch` (POST)

- **Purpose:** Batch hotel→city retrieval (up to 100 rows per call)
- **Auth:** Required
- **Request schema:**
    ```json
    {
      "hotels": [
        {
          "hotel_name": "Hotel Continental",
          "address": "123 Main St, Paris",
          "chain_code": "HIL",
          "city_hint": "PARIS"
        },
        {
          "hotel_name": "Grand Hyatt Berlin"
        },
        ...
      ],
      "return_scores": false
    }
    ```
    - `hotels`: Array of up to 100 hotel objects (same fields as `/predict`)
    - `return_scores`: If true, returns scores for each row (default: false)

- **Response schema:**
    ```json
    {
      "results": [
        {
          "pred_city_top1": "PARIS",
          "pred_city_top3": ["PARIS", "VERSAILLES", "LYON"],
          "sim_score_top1": 0.4721,
          "scores": [
            {"city": "PARIS", "score": 0.4721},
            {"city": "VERSAILLES", "score": 0.4212},
            {"city": "LYON", "score": 0.3877}
          ],
          "city_candidates_considered": 2034,
          "embedding_model": "openai_3small",
          "drift_score": 0.0011,
          "no_overlap": false,
          "request_id": "a6bcf9c3-1d5e-4b96-8e2a-e2a3c98d89d6"
        },
        {
          "pred_city_top1": "BERLIN",
          "pred_city_top3": ["BERLIN", "POTSDAM", "HAMBURG"],
          "sim_score_top1": 0.4611,
          "scores": null,
          "city_candidates_considered": 2034,
          "embedding_model": "openai_3small",
          "drift_score": 0.0013,
          "no_overlap": false,
          "request_id": "b7f2e7cb-8e8e-4f74-a1c2-9f1ca2a2c08f"
        }
      ],
      "batch_request_id": "4e70d7b0-2a93-4e2d-9c8a-ed77c598b8b0"
    }
    ```
    - Each result maps to input row order
    - `scores` populated if `return_scores` true

- **Status codes:** 200 (ok), 400 (invalid input), 401 (unauth), 429 (rate limit), 500 (internal)

- **Example cURL:**

    ```bash
    curl -X POST https://hotelcity.company-internal/api/v1/predict_batch \
      -H "X-API-Key: abcdef1234567890yourtoken" \
      -H "Content-Type: application/json" \
      -d '{"hotels":[{"hotel_name":"Hotel Continental","address":"Paris"},{"hotel_name":"Grand Hyatt Berlin"}]}'
    ```

    - Use `jq` to pretty-print results.

#### O.4.5 — `/metrics` (GET)

- **Purpose:** Surface current serving metrics and model health
- **Auth:** Required
- **Response schema:**
    ```json
    {
      "uptime_sec": 88200,
      "requests_1m": 3201,
      "requests_total": 17000421,
      "error_5xx_rate": 0.0002,
      "error_4xx_rate": 0.0027,
      "p50_latency_ms": 18,
      "p95_latency_ms": 51,
      "embedding_model": "openai_3small",
      "model_top1_accuracy": 0.4687,
      "minilm_top1_accuracy": 0.3937,
      "partial_ratio_top1": 0.4407,
      "wratio_top1": 0.4223,
      "3large_top1_accuracy": "0.6981 (unverified)",
      "lexical_overlap_bucket": 1399,
      "no_overlap_bucket": 1601,
      "catalog_size": 2034,
      "last_embedding_reload": "2025-03-12T03:15:27Z"
    }
    ```
- **Status codes:** 200

- **Example cURL:**
    ```bash
    curl -H "X-API-Key: abcdef1234567890yourtoken" https://hotelcity.company-internal/api/v1/metrics
    ```

#### O.4.6 — `/status` (GET)

- **Purpose:** Diagnostic; surfaces embedding/model/city catalog status
- **Auth:** Required
- **Response schema:**
    ```json
    {
      "embedding_model": "openai_3small",
      "model_version": "2025-01-15.2",
      "embedding_fingerprint": "e7c4a7f3...",
      "city_catalog_fingerprint": "c1f9b8e2...",
      "city_count": 2034,
      "embedding_reload_ts": "2025-03-12T03:15:27Z",
      "embedding_drift": 0.0012,
      "last_gt_update": "2025-01-12T00:00:00Z",
      "service_mode": "prod",
      "pending_admin_action": false
    }
    ```
- **Status codes:** 200

#### O.4.7 — `/city_candidates` (GET)

- **Purpose:** List all canonical city candidates (for UI/ops)
- **Auth:** Required
- **Response schema:**
    ```json
    {
      "cities": [
        {"city_code": "PARIS", "city_name": "Paris", "country": "FR"},
        {"city_code": "BERLIN", "city_name": "Berlin", "country": "DE"},
        ...
      ],
      "catalog_size": 2034
    }
    ```

- **Status codes:** 200

#### O.4.8 — `/embedding_similarity` (POST)

- **Purpose:** Debug endpoint; returns raw similarity matrix for input hotels and all city candidates (for model audit only; not for production workloads)
- **Auth:** Required (admin recommended)
- **Request schema:**
    ```json
    {
      "hotel_names": ["Hotel Continental", "Grand Hyatt Berlin"],
      "address_list": ["123 Main St, Paris", null]     // Optional
    }
    ```
- **Response schema:**
    ```json
    {
      "matrix": [
        // For each hotel, array of (city, score)
        [
          {"city": "PARIS", "score": 0.4721},
          {"city": "VERSAILLES", "score": 0.4212},
          ...
        ],
        [
          {"city": "BERLIN", "score": 0.4611},
          {"city": "POTSDAM", "score": 0.4135},
          ...
        ]
      ],
      "city_count": 2034
    }
    ```
- **Status codes:** 200, 400, 401, 429, 500

#### O.4.9 — `/admin/reload` (POST)

- **Purpose:** Triggers hot reload of embeddings/model from artifact store (admin only)
- **Auth:** Required (admin key)
- **Request schema:**
    ```json
    {
      "reload_embeddings": true,
      "reload_model": false
    }
    ```
- **Response schema:**
    ```json
    {
      "status": "ok",
      "embedding_reload_ts": "2025-03-12T03:15:27Z",
      "model_reload_ts": null
    }
    ```
- **Status codes:** 200 (ok), 401 (unauth), 403 (forbidden), 500 (fail)

---

### O.5 — Input validation and constraints

- All string fields are UTF-8; leading/trailing whitespace trimmed.
- `hotel_name` max 256 chars; `address` max 512 chars.
- `chain_code`, `city_hint` must match canonical codes or omitted.
- For `/predict_batch`, max 100 hotels per call; 413 if exceeded.
- All endpoints return `request_id` (UUID) for traceability in logs.

---

### O.6 — Error handling

All error responses have uniform structure:

```json
{
  "error": "InvalidRequest",
  "message": "hotel_name must be specified",
  "request_id": "c2a1f728-4a6d-41d2-ae3b-0c1eaa1c4eaf"
}
```

| Error code         | HTTP status | Meaning                                 |
|--------------------|-------------|-----------------------------------------|
| `InvalidRequest`   | 400         | Malformed or missing input              |
| `Unauthorized`     | 401         | No/invalid API key                      |
| `Forbidden`        | 403         | Lacks admin privilege                   |
| `RateLimit`        | 429         | Too many requests                       |
| `InternalError`    | 500         | Unexpected error                        |
| `ServiceUnavailable` | 503       | During embedding/model reload           |

- Error envelope always includes `request_id`.
- Rate limiting: per-client IP and per-key, bursty traffic gets 429.

---

### O.7 — Rate limits

| Endpoint           | Limit per client (per sec) | Limit notes           |
|--------------------|---------------------------|-----------------------|
| `/predict`         | 50                        | burst up to 100/sec   |
| `/predict_batch`   | 10                        | up to 100 hotels/call |
| `/embedding_similarity` | 2                   | admin use only        |
| `/metrics`         | 20                        |                       |
| `/city_candidates` | 10                        |                       |
| `/admin/reload`    | 1                         | admin only            |

- Exceeding limits returns 429 with `Retry-After` header.

---

### O.8 — Example Slack transcript (API troubleshooting)

```
[09:13] @hannah: I'm getting 401 from /predict_batch on staging. Token worked last week. Known issue?
[09:14] @lin: Check with Priya — staging key was rotated on Monday, see #ml-infra-announcements.
[09:15] @priya: DMing you new key. If you see 403, that's admin-only endpoint.
[09:15] @hannah: Got it, works now. Thanks!
```

---

### O.9 — API change management and backward compatibility

- All endpoints are versioned under `/v1/`; breaking changes require a new version (e.g., `/v2/`).
- Schema changes (adding optional fields) are allowed within `/v1/`; field removals or required field changes require bump.
- API deprecation policy: minimum 60 days notice, Slack #ml-infra-announcements, and direct email to registered clients.
- Priya owns dashboard of API usage; ping for custom client reporting.

---

### O.10 — Security & access notes

- All logs redact hotel/address on error unless `X-Debug-Mode` header present (admin only).
- Only Lin and Arjun may provision or revoke API keys.
- All admin actions (embedding reload, model reload) are logged to `ops_db.admin_audit_log`.
- Token brute-force attempts >5/min are auto-blocked and paged to SRE.

---

### O.11 — Example full prediction workflow

1. **Predict city for a single booking:**

    ```bash
    curl -X POST https://hotelcity.company-internal/api/v1/predict \
      -H "Content-Type: application/json" \
      -H "X-API-Key: prodtoken123" \
      -d '{"hotel_name":"Hilton Berlin","address":"Mohrenstraße 30, 10117 Berlin"}'
    # Returns:
    # {
    #   "pred_city_top1": "BERLIN",
    #   "pred_city_top3": ["BERLIN", "POTSDAM", "HAMBURG"],
    #   "sim_score_top1": 0.4611,
    #   ...
    # }
    ```

2. **Batch prediction for bookings:**

    ```bash
    curl -X POST https://hotelcity.company-internal/api/v1/predict_batch \
      -H "Content-Type: application/json" \
      -H "X-API-Key: prodtoken123" \
      -d '{"hotels":[{"hotel_name":"Hilton Berlin","address":"Mohrenstraße 30, 10117 Berlin"},{"hotel_name":"Hotel de Crillon","address":"10 Place de la Concorde, 75008 Paris"}]}'
    # Returns:
    # {
    #   "results": [
    #     {...}, {...}
    #   ],
    #   "batch_request_id": "..."
    # }
    ```

3. **Query service status:**

    ```bash
    curl -H "X-API-Key: prodtoken123" https://hotelcity.company-internal/api/v1/status
    ```

4. **Reload embeddings (admin only):**

    ```bash
    curl -X POST https://hotelcity.company-internal/api/v1/admin/reload \
      -H "Content-Type: application/json" \
      -H "X-API-Key: admintoken456" \
      -d '{"reload_embeddings": true}'
    ```

---

### O.12 — Known issues and escalation

- See Appendix H for known-bad model versions and embedding artifacts.
- For urgent API issues, escalate per N.1 (paging order).
- All API 5xx errors are logged with full trace and request_id for postmortem.

---

**End of Appendix O**

---

---

## Appendix P — detailed monitoring spec

This appendix defines the full monitoring and observability surface for the hotel→city retrieval pipeline, including every metric emitted (with SLOs and thresholds), dashboard conventions, alerting, and post-alert action mapping. Ownership is shared: Priya (dashboard/metric stratification), Arjun (monitoring infra), Lin (governance/compliance). All metrics are surfaced in Grafana (dashboard group: `hotel_city_retrieval/`) and mirrored to Datadog for paging.

---

### P.1 — Metric inventory

Below, each metric is specified with:

- **Name** (metric name in system)
- **Type** (counter, gauge, histogram, etc)
- **Granularity** (pipeline run, per-row, aggregate)
- **SLO target** (quantitative objective)
- **Alert threshold** (for paging/notification)
- **Dashboard location** (section/panel)
- **Post-alert runbook** (section cross-ref)

#### Core pipeline health

| Name                                | Type       | Granularity      | SLO Target                              | Alert Threshold                | Dashboard Panel   | Runbook Ref         |
|--------------------------------------|------------|------------------|-----------------------------------------|-------------------------------|-------------------|---------------------|
| `pipeline.success_rate`              | counter    | daily            | >99.5% success/month                    | <99.0% for >1 day             | Overview          | N.6, N.3            |
| `pipeline.duration_seconds`          | histogram  | per-run          | p95 < 45 min                            | p99 > 60 min                  | Overview          | N.2, N.3            |
| `pipeline.start_ts`, `end_ts`        | gauge      | per-run          | -                                       | -                             | Run Timeline      | -                   |
| `pipeline.failed_tasks`              | counter    | per-run          | 0 per run (expected)                    | >0 triggers page              | Task Details      | N.2, N.3            |
| `pipeline.data_freshness_hours`      | gauge      | per-run          | <4.0 hours lag post-D-1                 | >6.0 hours                    | Freshness         | N.2, N.4            |

#### Embedding/model health

| Name                                | Type       | Granularity      | SLO Target                              | Alert Threshold                | Dashboard Panel   | Runbook Ref         |
|--------------------------------------|------------|------------------|-----------------------------------------|-------------------------------|-------------------|---------------------|
| `embedding.drift_l2_max`             | gauge      | daily            | <0.05 max L2 norm diff                  | >0.1 triggers page            | Embedding Drift   | N.2, N.3            |
| `embedding.drift_l2_mean`            | gauge      | daily            | <0.01 mean L2 norm diff                 | >0.03 triggers warn           | Embedding Drift   | N.2, N.3            |
| `embedding.loader_failures`          | counter    | per-run          | 0 per run                               | >0 per run triggers page      | Embedding Health  | N.2.3               |
| `embedding.file_checksum`            | string     | per-batch        | (should not change outside version bump) | Change triggers warn           | Embedding Health  | N.2, N.4            |
| `model.version`                      | string     | per-run          | Stable unless explicit upgrade           | Unexpected change triggers page| Model Version     | N.4                 |

#### Inference quality

| Name                                | Type       | Granularity      | SLO Target                              | Alert Threshold                | Dashboard Panel   | Runbook Ref         |
|--------------------------------------|------------|------------------|-----------------------------------------|-------------------------------|-------------------|---------------------|
| `quality.top1_match_rate`            | gauge      | daily, per-bucket| minilm ≥0.3937, openai_3small ≥0.4687   | -0.02 absolute drop triggers page | Quality          | N.2, N.3            |
| `quality.top3_match_rate`            | gauge      | daily, per-bucket| ≥0.6300 (3small), ≥0.6350 (partial_ratio)| -0.02 drop triggers warn        | Quality          | N.2, N.3            |
| `quality.partial_ratio_top1`         | gauge      | daily            | ≥0.4407                                 | <0.42 triggers warn            | Quality           | N.2                 |
| `quality.wratio_top1`                | gauge      | daily            | ≥0.4223                                 | <0.40 triggers warn            | Quality           | N.2                 |
| `quality.no_overlap_bucket_rate`     | gauge      | daily            | 1601/3000 (no-overlap), stable          | ±50 deviation triggers warn    | Segment Quality   | N.2, E.3            |
| `quality.lexical_overlap_bucket_rate`| gauge      | daily            | 1399/3000 (lexical-overlap), stable     | ±50 deviation triggers warn    | Segment Quality   | N.2, E.3            |

#### Per-row and per-bucket error

| Name                                | Type       | Granularity      | SLO Target                              | Alert Threshold                | Dashboard Panel   | Runbook Ref         |
|--------------------------------------|------------|------------------|-----------------------------------------|-------------------------------|-------------------|---------------------|
| `row.error_count`                    | counter    | daily            | <0.1% of rows error                     | >0.5% triggers page           | Row Health        | N.2                 |
| `row.null_prediction_count`          | counter    | daily            | 0 per run (expected)                    | >0 triggers warn              | Row Health        | N.2                 |
| `bucket.top1_match_rate`             | gauge      | per-bucket       | Within ±0.02 of canonical               | Any bucket outside triggers warn| Segment Quality  | N.2, E.3            |
| `bucket.size`                        | gauge      | per-bucket       | Stable versus prior 30d mean            | >2x change triggers warn      | Segment Quality   | N.2                 |

#### Drift & data integrity

| Name                                | Type       | Granularity      | SLO Target                              | Alert Threshold                | Dashboard Panel   | Runbook Ref         |
|--------------------------------------|------------|------------------|-----------------------------------------|-------------------------------|-------------------|---------------------|
| `drift.sim_score_mean`               | gauge      | daily            | Stable within ±0.02 of 30d mean         | >0.04 deviation triggers warn | Drift             | N.2, M.4            |
| `drift.sim_score_std`                | gauge      | daily            | Stable                                  | >0.02 increase triggers warn  | Drift             | N.2, M.4            |
| `drift.city_prediction_entropy`      | gauge      | daily            | >3.5 bits (city pred entropy)           | <3.0 triggers page            | Drift             | N.2, M.4            |
| `drift.city_distribution_kl`         | gauge      | daily            | KL to prior <0.05                       | >0.1 triggers warn            | Drift             | N.2, M.4            |
| `drift.row_order_fingerprint`        | string     | daily            | Match previous unless catalog/model changed | Change triggers page         | Drift             | N.2, M.4            |

#### Data ingestion/ETL

| Name                                | Type       | Granularity      | SLO Target                              | Alert Threshold                | Dashboard Panel   | Runbook Ref         |
|--------------------------------------|------------|------------------|-----------------------------------------|-------------------------------|-------------------|---------------------|
| `etl.snowflake_query_duration`       | histogram  | per-run          | p95 < 180s                              | p99 > 300s triggers page      | ETL Health        | N.2                 |
| `etl.rows_pulled`                    | counter    | per-run          | Within ±5% of 30d mean                  | >10% drop triggers warn       | ETL Health        | N.2                 |
| `etl.rows_filtered`                  | counter    | per-run          | <1% of rows                             | >2% triggers warn             | ETL Health        | N.2                 |
| `etl.bookings_null_field_count`      | counter    | per-run          | <0.1% of rows                           | >0.5% triggers warn           | ETL Health        | N.2                 |
| `etl.duplicate_booking_ids`          | counter    | per-run          | 0                                       | >0 triggers page              | ETL Health        | N.2                 |

#### Service/API health

| Name                                | Type       | Granularity      | SLO Target                              | Alert Threshold                | Dashboard Panel   | Runbook Ref         |
|--------------------------------------|------------|------------------|-----------------------------------------|-------------------------------|-------------------|---------------------|
| `service.latency_p95`                | gauge      | 10min            | <200ms                                  | >400ms triggers page           | API Health        | N.2                 |
| `service.error_rate`                 | gauge      | 10min            | <0.05%                                  | >0.2% triggers page            | API Health        | N.2                 |
| `service.qps`                        | gauge      | 1min             | Stable, auto-scaled                     | >2x change triggers warn       | API Health        | N.2                 |
| `service.active_instances`           | gauge      | 1min             | ≥2 at all times                         | <2 triggers page               | API Health        | N.2                 |

#### Infra resource utilization

| Name                                | Type       | Granularity      | SLO Target                              | Alert Threshold                | Dashboard Panel   | Runbook Ref         |
|--------------------------------------|------------|------------------|-----------------------------------------|-------------------------------|-------------------|---------------------|
| `infra.cpu_pct`                      | gauge      | 1min             | <80%                                   | >90% triggers warn             | Infra             | N.2                 |
| `infra.ram_pct`                      | gauge      | 1min             | <75%                                   | >85% triggers warn             | Infra             | N.2                 |
| `infra.disk_pct`                     | gauge      | 1min             | <80%                                   | >90% triggers warn             | Infra             | N.2                 |
| `infra.gpu_utilization`              | gauge      | 1min             | <70%                                   | >90% triggers warn             | Infra             | N.2                 |

#### Audit/compliance

| Name                                | Type       | Granularity      | SLO Target                              | Alert Threshold                | Dashboard Panel   | Runbook Ref         |
|--------------------------------------|------------|------------------|-----------------------------------------|-------------------------------|-------------------|---------------------|
| `audit.run_id_uniqueness`            | gauge      | per-run          | 100% unique                             | <100% triggers page            | Audit             | N.2                 |
| `audit.sla_breach_count`             | counter    | daily            | 0                                       | >0 triggers page               | Audit             | N.2, N.3            |
| `audit.retention_days`               | gauge      | daily            | 90 days                                 | <85 triggers warn              | Audit             | N.2, Data Gov       |

*Note: All metric names reflect those in Prometheus/Grafana. "Warn" triggers Slack/email; "page" triggers PagerDuty (see P.4).*

---

### P.2 — Dashboard layouts

#### Overview dashboard

- **Pipeline run summary:** Last 7, 30, and 90 days; success rate, duration, failed tasks, freshness.
- **Visual timeline:** Gantt of ETL, embedding join, analytics write, drift monitor.
- **Run metadata:** Model version, embedding file hash, row-order fingerprint.

#### Embedding/model health dashboard

- **Drift panel:** L2 drift (max, mean), loader failures, embedding file checksums (diff heatmap).
- **Model version panel:** Current, prior, unexpected changes.
- **Historical drift:** Sparkline of L2 drift by day; click-through to artifact diff.

#### Inference quality dashboard

- **Top-1/top-3 match rates:** By model, by bucket (lexical/no-overlap), trendlines vs SLO.
- **Segment performance:** Heatmap by bucket; outlier detection for segment drops.
- **Canonical numbers:** Displayed as fixed annotations (see §8).

#### Drift & data integrity dashboard

- **Sim score distribution:** Mean, std, pctiles; anomaly band overlay.
- **City prediction entropy/KL:** City pred entropy trend, KL to prior day; redline at alert threshold.
- **Row-order fingerprint:** Current vs prior, with diff status.

#### ETL/ingest dashboard

- **Snowflake query metrics:** Duration, rows pulled, filtered, null field count.
- **Booking data health:** Duplicate IDs, bookings per city/hotel histogram, ingestion lag.

#### API/service health dashboard

- **Latency:** p50/p95/p99, SLA overlays.
- **Error rate:** Spike detection, error code breakdown.
- **QPS / active instances:** Autoscaling activity, instance health.

#### Infra/resource dashboard

- **CPU/RAM/disk/GPU:** Utilization, saturation, resource exhaustion events.
- **Pod/node status:** Kubernetes health if relevant.

#### Audit/compliance dashboard

- **Run ID uniqueness:** Histogram, outlier detection.
- **Retention window:** Days of data online; compliance SLO line.
- **SLA breach count:** Annotated event chart.

*All dashboards are accessible via Grafana folder: `hotel_city_retrieval/`. Dashboards are versioned and changes are tracked in `dashboards/README.md` (Priya, Lin owners).*

---

### P.3 — Alerting and paging rules

#### Alert types

- **Warning (notify):** Slack #ml-infra-alerts, email to on-call. Non-paging. Action within 24h.
- **Critical (page):** PagerDuty page to primary on-call, escalation per N.1. Immediate investigation.

#### Paging rules (selected excerpts):

| Metric                        | Threshold                              | Action                            | Escalation            |
|-------------------------------|----------------------------------------|-----------------------------------|-----------------------|
| `pipeline.success_rate`       | <99.0% (daily)                         | Page; investigate failures        | N.2, N.3, escalate   |
| `embedding.loader_failures`   | >0 per run                             | Page; check logs, reload          | N.2.3, N.4           |
| `drift.city_prediction_entropy`| <3.0 bits                             | Page; check city catalog, embeddings | N.2, M.4, escalate   |
| `etl.duplicate_booking_ids`   | >0 per run                             | Page; data audit                  | N.2, N.4             |
| `service.error_rate`          | >0.2% (10min)                          | Page; check logs, fallback        | N.2, escalate         |
| `drift.row_order_fingerprint` | Any change (unless model/catalog bump) | Page; diff embeddings/cities      | M.4, N.2, escalate    |
| `audit.sla_breach_count`      | >0                                     | Page; root-cause, document        | N.2, N.3              |

*All other thresholds/alerts: see P.1 table.*

#### Alert routing

- **Slack #ml-infra-alerts:** All warnings, non-paging errors.
- **PagerDuty:** All criticals (see above), plus manual triggers.
- **Email:** Nightly summary of warnings, daily SLO compliance.

#### On-call rotation & response (see N.1):

- **Primary on-call:** First response, triage within 10 min.
- **Secondary on-call:** Escalated if no ack in 10 more min.
- **ML manager:** Escalate if unresolved after 20 min.
- **DevOps SRE:** Only if infra root cause suspected.

#### After-action

- **Incident log:** All alert responses must be logged (`oncall/incidents.md`).
- **Postmortem:** For any page or SLO miss >1 hour, postmortem within 2 business days.
- **Metrics audit:** Monthly review of alert/metric efficacy (Lin, Priya).

---

### P.4 — Post-alert runbook references

All alerts map to specific runbook actions — see table below.

| Alert Type                        | Runbook Location (Appendix/Section)  |
|-----------------------------------|--------------------------------------|
| Pipeline failure                  | N.2, N.3, N.4                        |
| Embedding/model drift             | N.2, N.3, N.4, M.4                   |
| ETL ingestion error               | N.2, N.3, N.4                        |
| Inference quality drop            | N.2, N.3, E.3                        |
| Data integrity/data mismatch      | N.2, N.3, N.4, H                     |
| SLA breach                        | N.2, N.3, N.4                        |
| Service/API error                 | N.2, N.3, N.4                        |
| Infra/resource exhaustion         | N.2, N.3                             |
| Compliance/retention error        | N.2, Data Gov                        |

**See N.2 for diagnostics, N.3 for mitigation, N.4 for rollback, and M.4 for drift/fingerprint handling.**

---

### P.5 — Metric governance and lifecycle

- **Metric changes:** All metric additions, removals, or SLO adjustments require PR review (Arjun, Priya).
- **Versioning:** Dashboard and alerting config versioned in `dashboards/`, reviewed monthly.
- **Auditing:** Quarterly audit of metrics vs. incident log for gap analysis (Lin, Priya).
- **Deprecation:** Deprecated metrics must be retained for 30d before removal; dashboard annotations required.

---

### P.6 — Known limitations and future improvements

- **Metric cardinality:** Some per-bucket metrics may exceed Prometheus cardinality quotas if city catalog grows >10k.
- **Anomaly detection:** Current rules are static; ML-based anomaly detection flagged as future roadmap.
- **Row-level traces:** Not all per-row errors are surfaced in dashboards for scale reasons; see logs for details.
- **SLA clock drift:** Data freshness assumes upstream warehouse clock integrity.

---

**End of Appendix P**

---

---

## Appendix Q — data pipeline SQL and DAG reference

This appendix provides canonical, obfuscated SQL patterns and DAG orchestration details for the hotel→city retrieval pipeline. It covers: (1) Snowflake extraction, (2) embedding join/analytics enrichment, (3) drift-sample selection, (4) analytics table write patterns, and (5) Airflow DAG structure with explicit dependencies and failure recovery.

### Q.1 — Daily Snowflake booking pull

#### Source Table

- `prod_db.booking_facts` (primary, canonicalized across all envs)

#### Example SQL

```sql
-- Q.1.1: Daily pull of confirmed bookings for embedding step
SELECT
    b.booking_id,
    b.hotel_name,
    b.city_gt,
    b.booking_ts,
    b.address_line1,
    b.freeform_addr,
    b.chain_code,
    b.source_channel,
    b.meta_blob
FROM prod_db.booking_facts b
WHERE b.booking_ts >= DATEADD(day, -1, CURRENT_DATE())
  AND b.booking_status = 'CONFIRMED'
  AND b.hotel_name IS NOT NULL
  AND b.city_gt IS NOT NULL
  AND b.booking_id IS NOT NULL
;
```

- Notes:
    - Only confirmed, non-null bookings.
    - `meta_blob` is semi-structured (see M.1).
    - Source columns are fixed per data contract; no free-form joins allowed in prod pipeline.

#### Table size

- Typical D-1 batch: 8,000–16,000 rows (3,000 within hotel subset for eval).

---

### Q.2 — Embedding join/enrichment SQL

This step is executed in the Python pipeline, but for analytics/debugging, an equivalent SQL view is maintained for ad hoc audits.

#### Embedding Join Logic (Obfuscated SQL View)

```sql
-- Q.2.1: Analytics view joining bookings to embedding predictions
SELECT
    b.booking_id,
    b.hotel_name,
    b.city_gt,
    ej.pred_city_top1,
    ej.pred_city_top3,
    ej.sim_score_top1,
    ej.model_version,
    ej.embedding_ts,
    b.booking_ts,
    b.source_channel
FROM analytics_db.booking_enriched b
LEFT JOIN analytics_db.embedding_joins ej
    ON b.booking_id = ej.booking_id
WHERE b.booking_ts = DATEADD(day, -1, CURRENT_DATE())
;
```

- `embedding_joins` is a materialized table (populated by pipeline) with one row per booking.
- `pred_city_top3` is a JSON array of city canonical names.
- `sim_score_top1` is a float (0.0–1.0, model-dependent).
- `model_version` and `embedding_ts` are for drift analysis.

---

### Q.3 — Drift-sample selection SQL

To monitor for model/city-catalog drift, a stratified sample is selected for detailed row auditing:

#### Stratified Sampling (Obfuscated Example)

```sql
-- Q.3.1: Sample 100 bookings per city_gt for drift analysis
WITH base AS (
    SELECT
        b.booking_id,
        b.city_gt,
        ROW_NUMBER() OVER (PARTITION BY b.city_gt ORDER BY RANDOM()) AS rn
    FROM analytics_db.booking_enriched b
    WHERE b.booking_ts = DATEADD(day, -1, CURRENT_DATE())
)
SELECT booking_id, city_gt
FROM base
WHERE rn <= 100
;
```

- Ensures coverage across all ground-truth cities.
- For small cities (under 100 per day), all rows are included.
- Resulting sample is fed to the drift-monitor and archived in `analytics_db.drift_samples`.

---

### Q.4 — Analytics table write patterns

#### Main Analytics Sink Table

- Table: `analytics_db.city_inference_results`
- Partitioned by booking date (`booking_ts`).

#### Ingestion SQL (Obfuscated Upsert)

Pipeline writes are append-only; upserts are reserved for rare correction workflows.

```sql
-- Q.4.1: Main analytics table insert
INSERT INTO analytics_db.city_inference_results (
    booking_id,
    hotel_name,
    city_gt,
    pred_city_top1,
    pred_city_top3,
    sim_score_top1,
    drift_score,
    run_id,
    booking_ts,
    ingest_ts
)
SELECT
    :booking_id,
    :hotel_name,
    :city_gt,
    :pred_city_top1,
    :pred_city_top3,
    :sim_score_top1,
    :drift_score,
    :run_id,
    :booking_ts,
    CURRENT_TIMESTAMP()
FROM VALUES (:booking_id, :hotel_name, :city_gt, :pred_city_top1, :pred_city_top3, :sim_score_top1, :drift_score, :run_id, :booking_ts)
;
```

- All writes are parameterized/batched via SQLAlchemy or Snowflake Python connector.
- `run_id` is a UUID per batch.
- `drift_score` is model-specific; see Appendix M.4.

#### Drift Metrics Table

- Table: `analytics_db.city_inference_drift`
- Example insert:

```sql
-- Q.4.2: Drift metrics write
INSERT INTO analytics_db.city_inference_drift (
    run_id,
    metric_name,
    metric_value,
    metric_ts
) VALUES
    (:run_id, 'top1_match_rate', :top1_rate, CURRENT_TIMESTAMP()),
    (:run_id, 'top3_match_rate', :top3_rate, CURRENT_TIMESTAMP()),
    (:run_id, 'sim_score_mean', :sim_mean, CURRENT_TIMESTAMP()),
    (:run_id, 'sim_score_std', :sim_std, CURRENT_TIMESTAMP()),
    (:run_id, 'row_order_fingerprint', :row_fp, CURRENT_TIMESTAMP())
;
```

---

### Q.5 — Airflow DAG structure and dependencies

#### DAG Overview

- **DAG Name:** `hotel_city_retrieval_nightly`
- **Schedule:** Daily, 02:00 UTC
- **Major Tasks:**
    1. `pull_bookings_from_snowflake`
    2. `join_with_embeddings`
    3. `write_analytics_tables`
    4. `sample_drift_rows`
    5. `emit_drift_metrics`
    6. `mark_success`
- **Failure Recovery:** Each task retries once on failure (10 min delay). On final failure, triggers Slack alert and pages on-call.

#### Airflow DAG Sketch (Python)

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.slack_webhook_operator import SlackWebhookOperator
from datetime import datetime, timedelta

default_args = {
    'retries': 1,
    'retry_delay': timedelta(minutes=10),
    'on_failure_callback': lambda context: notify_failure(context),
}

dag = DAG(
    'hotel_city_retrieval_nightly',
    start_date=datetime(2025, 1, 1),
    schedule_interval='0 2 * * *',
    max_active_runs=1,
    default_args=default_args,
    catchup=False,
)

def notify_failure(context):
    # Slack alert logic, obfuscated
    pass

pull_bookings = PythonOperator(
    task_id='pull_bookings_from_snowflake',
    python_callable=pull_from_snowflake,
    dag=dag,
)

join_embeddings = PythonOperator(
    task_id='join_with_embeddings',
    python_callable=join_with_embeddings,
    dag=dag,
)

write_analytics = PythonOperator(
    task_id='write_analytics_tables',
    python_callable=write_to_analytics_tables,
    dag=dag,
)

sample_drift = PythonOperator(
    task_id='sample_drift_rows',
    python_callable=sample_drift_rows,
    dag=dag,
)

emit_drift_metrics = PythonOperator(
    task_id='emit_drift_metrics',
    python_callable=push_drift_metrics,
    dag=dag,
)

mark_success = PythonOperator(
    task_id='mark_success',
    python_callable=mark_pipeline_success,
    dag=dag,
)

alert_failure = SlackWebhookOperator(
    task_id='alert_failure',
    http_conn_id='slack_alerts',
    message=":x: Hotel→city nightly pipeline failed — check Airflow logs and on-call rotation.",
    dag=dag,
    trigger_rule='one_failed',
)

# Dependency graph
(
    pull_bookings
    >> join_embeddings
    >> write_analytics
    >> sample_drift
    >> emit_drift_metrics
    >> mark_success
)

# On any task failure, fire alert
[pull_bookings, join_embeddings, write_analytics, sample_drift, emit_drift_metrics] >> alert_failure
```

#### Task Details

| Task                  | Retries | SLA        | Alerts              | Notes                                |
|-----------------------|---------|------------|---------------------|--------------------------------------|
| pull_bookings         | 1       | 02:30 UTC  | Slack, PagerDuty    | Extracts from Snowflake              |
| join_with_embeddings  | 1       | 03:00 UTC  | Slack, PagerDuty    | Embedding lookup, similarity         |
| write_analytics       | 1       | 03:10 UTC  | Slack, PagerDuty    | Inserts analytics table rows         |
| sample_drift_rows     | 1       | 03:20 UTC  | Slack, PagerDuty    | Stratified drift sample              |
| emit_drift_metrics    | 1       | 03:30 UTC  | Slack, PagerDuty    | Match rates, sim stats, fingerprint  |
| mark_success          | 0       | 03:35 UTC  | —                   | Finalizes run, logs metadata         |

- **Failure triggers:** On any task failure after retries, `alert_failure` Slack message is sent and on-call is paged.
- **Auto-recovery:** For transient issues (network, Snowflake lag), retry logic covers >80% of observed failures.
- **Manual escalation:** On 2+ consecutive failures, pipeline is paused and ML manager is notified.

#### DAG Maintenance

- Model/city-catalog version pinning is enforced at DAG start; any drift/mismatch aborts run (see M.4, N.2).
- All pipeline run metadata (run_id, task timings, fingerprints) are written to `ops_db.pipeline_run_history`.
- DAG file is versioned under `infra/airflow/dags/` with peer review required for edits.

---

### Q.6 — Operational notes and best practices

- **SQL patterns:** All queries must use parameterized bindings; never string-concat hotel/city names.
- **Data lineage:** Every analytics row preserves original booking_id and run_id.
- **Schema evolution:** Any analytics table schema change must be approved by Priya (analytics/data owner).
- **DAG idempotency:** Pipeline is rerunnable for a given D-1 date; all writes are append-only with conflict detection on booking_id+run_id.
- **Drift sample curation:** Sampled rows are archived for audit/labeling; ensure no PII is exported downstream.
- **Failure documentation:** All incidents logged in `oncall/incidents.md` per N.3.

---

**End of Appendix Q**

---

---

## Appendix R — failure mode catalog (20 modes)

This catalog enumerates the primary known and anticipated failure modes for the hotel→city retrieval pipeline. Each entry details observed symptoms, probable or confirmed root causes, recommended diagnostic probes, mitigation steps, canonical owner, and severity classification (P1 = critical, P2 = major, P3 = minor/nuisance). Priya is metric/dashboards owner; Arjun is overall lead; Lin covers governance and regression tracking. For cross-reference, see operational playbooks (Appendix N) and historical incident log.

| #  | Failure Mode            | Symptom(s)                                                           | Root Cause(s)                                                                                   | Diagnostic Probe(s)                                                                           | Mitigation(s)                                                                                                     | Owner   | Severity |
|----|------------------------|----------------------------------------------------------------------|-------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------|---------|----------|
| 1  | Row-permutation error  | Massive drop in match rate, sim scores inconsistent, city predictions misaligned | Embedding array or city-catalog order mismatch between batch and inference                       | Compare row-order fingerprint across runs; diff city and embedding arrays                      | Re-run batch with canonical order, restore last good artifact, validate fingerprint in logs                        | Arjun   | P1       |
| 2  | Tie-break instability  | Same hotel→city pair yields different predictions across runs         | Floating-point tie in sim scores; non-deterministic sort order                                  | Check top-K sim scores for equality; review random seeds and sort implementation              | Enforce deterministic tie-break (lexicographic or index order) in scoring code                                    | Priya   | P2       |
| 3  | GT whitespace mismatch | Apparent misprediction on otherwise exact hotel/city name pairs       | Ground-truth or catalog contains leading/trailing or double whitespace                          | View raw GT/city/hotel strings with repr(); diff whitespace-normalized vs raw names           | Normalize whitespace in both GT and catalog; add linter to validation step                                        | Lin     | P2       |
| 4  | Name, no city (lexical miss) | Hotel name not mapping to any city; recall gap in no-overlap bucket          | Hotel name novel, alias, or misspelled; city not in catalog                                    | Check lexical-overlap bucket metrics; search for hotel in catalog/hotels.csv                  | Augment catalog, add alias, or escalate to catalog owner; fallback to partial_ratio as needed                     | Priya   | P2       |
| 5  | Embedding drift        | Gradual or sudden drop in accuracy and sim score distribution        | Change in embedding model weights or training data; version skew                                | Compare embedding file hashes, model version tags, and daily drift metrics                    | Roll back embedding/model version; lock model artifacts; notify ML lead                                           | Arjun   | P1       |
| 6  | Catalog skew           | Top city predictions cluster in one geography; match rate drops elsewhere | Incomplete or outdated city catalog, new geos missing                                           | Plot prediction histogram by city; compare city_catalog.csv over time                         | Update catalog from upstream source; run city coverage audit                                                      | Priya   | P2       |
| 7  | Case-sensitivity bug   | Some city or hotel predictions differ only by letter case            | Catalog or GT loaded case-sensitively; inconsistent normalization                              | Diff lowercased vs raw names; check string handling in ETL code                               | Standardize lowercasing in all string comparisons; add test for case-only mismatches                              | Lin     | P3       |
| 8  | Embedding load failure | Pipeline errors at inference or batch start; bookings missing predictions | Corrupt or partial embedding .npy file; version mismatch                                        | Inspect loader logs; validate file size and checksum; check for stack traces                  | Restore from backup artifact, re-run batch, disable auto-reload until fixed                                       | Arjun   | P1       |
| 9  | Model fallback silent  | Pipeline silently switches to fallback model (MiniLM, partial_ratio), metric spike | Main model unavailable, crash or timeout; fallback logic triggers without alert                 | Check run metadata for model name; review alert logs for fallback events                      | Add explicit alert/pager on fallback trigger; fix main model, force pipeline back to primary                      | Priya   | P1       |
| 10 | Null/blank input rows  | Some booking rows have null or empty hotel_name, city_gt, or address | Upstream data issue; ETL bug; source channel change                                            | Query for nulls in analytics warehouse; check ETL logs                                        | Patch ETL to filter or impute blanks; escalate upstream for data contract fix                                     | Lin     | P2       |
| 11 | Stale GT/city catalog  | Ground-truth or city/hotel catalog is days or weeks out of sync     | Failed or delayed catalog refresh; S3 or Git sync issue                                        | Compare timestamps on gt.json and catalog/*.csv; check pipeline logs                          | Force refresh from source; add alert on catalog staleness; document in run metadata                               | Priya   | P2       |
| 12 | Address/metadata leakage | Model seemingly "cheats" by overfitting to operator notes in meta_blob | Embedding pipeline includes address or meta fields not present in catalog or eval GT            | Inspect meta_blob content; audit embedding input features                                     | Restrict embedding input to hotel_name (or controlled fields); add data contract check                            | Arjun   | P2       |
| 13 | Batch overlap/race     | Duplicate or missing bookings in analytics; run_id collisions        | Batch jobs overlap in time; Airflow DAG misconfigured or delayed                               | Review Airflow run history; check for overlapping ingest_ts and run_ids                       | Limit DAG concurrency (max_active_runs=1); add run window guard; clean up duplicates                              | Hannah  | P2       |
| 14 | Sim score scale shift  | Sim score distribution shifts (mean, std), thresholds misfire       | Model retrain with different embedding scale; norm bug                                         | Compare sim score stats across runs; plot sim score histograms                                | Normalize sim scores; recalibrate thresholds; document retrain artifacts                                         | Priya   | P2       |
| 15 | Time zone confusion    | Booking_ts or ingest_ts off by ±1 day; apparent data staleness      | Time zone mismatch between ETL, warehouse, and analytics                                       | Audit timestamp columns for UTC/local mismatch; grep for tz in codebase                       | Standardize on UTC in all stages; add test for time zone handling                                                 | Lin     | P3       |
| 16 | Non-deterministic batch | Nightly batch yields different results with same input               | Unseeded random ops in embedding or scoring; catalog ordering unstable                         | Rerun batch with same input, compare outputs; check for random seeds in code                  | Seed all random ops; sort input deterministically; add regression test                                            | Arjun   | P2       |
| 17 | Upstream data delay    | No new bookings ingested; pipeline appears “stuck”                   | Source warehouse late, upstream job failure                                                    | Check booking_facts table freshness; review upstream job status                               | Alert on D-1 data absence by 02:00 UTC; escalate to data engineering                                             | Hannah  | P2       |
| 18 | City alias collision   | Multiple cities mapped to same hotel, or vice versa                  | City catalog contains aliases or alternate spellings not deduped                               | Search for city alias clusters in catalog; check for near-duplicate embeddings                | Canonicalize city names; add alias normalization step; escalate to catalog owner                                  | Priya   | P2       |
| 19 | Pipeline partial write | Some bookings missing from analytics, or incomplete rows present     | ETL or write-to-warehouse interrupted mid-batch; disk/network issue                            | Check analytics partition counts; review Airflow and warehouse write logs                     | Remove partial partitions; re-run failed batch; add write atomicity checks                                        | Martin  | P1       |
| 20 | Regression after refactor | Sudden metric drop after code or infra change; not caught in staging | Uncovered edge case in refactor, test coverage gap                                             | Compare metrics before/after refactor; review PR/test coverage reports                        | Roll back change; add regression test; require cross-env comparison for all major refactors                       | Arjun   | P1       |

---

### R.1 — Failure mode details and historical context

**Row-permutation error (1):**  
Historically, this has caused silent accuracy drops, as city-catalog order changed without corresponding embedding order update (see PR #094, April 2023 incident). The row-order fingerprinting protocol (Appendix M.4) was implemented to mitigate recurrence.

**Tie-break instability (2):**  
Incidents have surfaced where hotels with ambiguous names (e.g., "Central Hotel") tie across multiple cities. Prior to deterministic tie-break logic, this led to non-reproducible top-1 predictions and test flakiness.

**GT whitespace mismatch (3):**  
Multiple onboarding cases (see onboarding FAQ, item 3) have traced apparent mispredictions to invisible whitespace in GT or catalog files. Lin’s linter PR (May 2023) now flags these.

**Name, no city (4):**  
Over half of the no-overlap bucket (1601/3000) results from out-of-vocabulary hotel names, unseen aliases, or data entry errors. These are tracked via the stratified dashboard (Priya).

**Other high-severity (P1) modes:**  
- **Embedding drift (5):** All production model changes are now gated by ADR; drift is monitored nightly.
- **Embedding load failure (8):** Each loader error triggers an immediate pager and disables batch reloads.
- **Model fallback silent (9):** Silent fallback previously led to undetected metric drops; explicit alerting is now enforced.
- **Pipeline partial write (19):** Airflow and warehouse now enforce atomic writes; partial partitions are automatically flagged.

**Catalog and data hygiene (6, 11, 18):**  
Priya’s dashboard tracks catalog freshness and alias coverage. Stale catalogs are now rare but still possible with upstream sync failures.

**Batch/process stability (13, 16, 17):**  
DAG concurrency and run window guards are now standard; all random ops are seeded to ensure reproducibility.

**Score normalization and regression (14, 20):**  
Sim score scale shifts are tracked; all model retrains require recalibration and regression test. Arjun reviews all major code changes for regression risk.

---

### R.2 — Failure mode severity definitions

- **P1 (Critical):** Causes incorrect or missing predictions for >5% of bookings, or any silent data corruption. Triggers immediate page and incident review.
- **P2 (Major):** Impacts accuracy or reliability in a material but limited way; may affect specific city/hotel clusters or degrade dashboards. Tracked via weekly review.
- **P3 (Minor):** Cosmetic, edge-case, or only affects metrics/monitoring, not core predictions. Logged and triaged in monthly backlog.

---

### R.3 — Failure mode coverage by metric/dashboards

Priya’s stratified dashboards break out pipeline match rate, sim score distribution, and overlap buckets (lexical, no-overlap). All P1/P2 failures are surfaced via daily metric deltas, with drill-downs by hotel, city, and upstream channel.

If a new failure mode is observed or suspected, add to this catalog and tag the owner. All mitigations must be documented in `oncall/incidents.md` and reviewed in the next retro.

---

**End of Appendix R**

---

---

## Appendix S — full cost model walkthrough

This appendix details the cost structure and economics of the hotel→city retrieval system, including all direct compute and platform expenses, as well as the projected operator efficiency savings. Numbers are current as of 2025-02, with all cloud and vendor rates under active contracts. Where applicable, we provide both per-component and per-booking breakdowns, as well as modeled break-even points across low, medium, and high throughput scenarios.

### S.1 — Component-wise cost breakdown

We decompose the system into its primary cost drivers:

| Component                      | Description                                      | Billing Basis                     | $/Unit (2025) | Typical Volume | Monthly Cost Estimate |
|---------------------------------|--------------------------------------------------|-----------------------------------|---------------|---------------|----------------------|
| **A. Embedding refresh**        | Periodic batch embedding gen for hotels/cities   | Compute-hours (GPU/CPU)           | $0.80/hr (GPU)<br>$0.14/hr (CPU) | ~24 GPU-hr/mo <br>~16 CPU-hr/mo | $19.20 (GPU) <br> $2.24 (CPU) |
| **B. Query inference**          | Per-booking city prediction                      | API calls / vCPU-sec              | $0.00007/call (OpenAI 3small) <br>$0.00003/call (MiniLM) | ~100k bookings/mo | $7.00 (3small) <br> $3.00 (MiniLM) |
| **C. Drift sampling**           | Drift/statistics job on batch                    | CPU-hours                         | $0.14/hr (CPU)                  | ~4 hr/mo        | $0.56                |
| **D. GPT-4o-mini reranker**     | Optional: rerank top-3 for difficult bookings    | API calls                         | $0.0025/call                    | ~2k calls/mo    | $5.00                 |
| **E. Vendor chain-KB**          | Licensed city/hotel catalog enrichment           | Annual license (pro-rated)        | $1,400/yr                       | n/a             | $116.67/mo            |
| **F. Data warehouse storage**   | Analytics/drift table storage                    | $/TB/mo                           | $23/TB/mo                       | ~20GB/mo        | $0.46                 |
| **G. Airflow orchestration**    | Managed Airflow slot cost                        | $/slot-hr                         | $0.04/hr                        | ~60 slot-hr/mo  | $2.40                 |
| **H. Monitoring/alerts**        | Datadog/Splunk logs and alerting                | $/GB (logs), flat alert fee       | $0.20/GB, $10/mo alerting       | ~15GB logs/mo   | $13.00                |

**Notes:**
- Embedding refresh includes both hotel and city catalogs; runs weekly on A10g GPU spot instances, with fallback to CPU if GPU pool is exhausted.
- Query inference cost is weighted by the current production blend: ~85% OpenAI 3small, 15% MiniLM fallback.
- GPT-4o-mini reranking is only triggered for bookings with top-1/top-3 margin <0.08, and can be disabled for cost control.
- Vendor chain-KB fee is contractual and not usage-based; shown as monthly amortized.
- Storage costs assume 90-day rolling window; oldest partitions are auto-deleted.

#### S.1.1 — Summary table

| Component                  | Monthly Cost |
|----------------------------|--------------|
| Embedding refresh          | $21.44       |
| Query inference            | $8.00        |
| Drift sampling             | $0.56        |
| GPT-4o-mini reranker       | $5.00        |
| Vendor chain-KB            | $116.67      |
| DW storage                 | $0.46        |
| Airflow orchestration      | $2.40        |
| Monitoring/alerts          | $13.00       |
| **Total (all-in)**         | **$167.53**  |

- If GPT-4o-mini is **disabled**, monthly total drops to **$162.53**.
- All numbers exclude optional SRE standby (infra) and marginal cloud network egress, which have <2% impact at current scale.

---

### S.2 — Per-booking unit economics

Assuming a base volume of 100,000 bookings/month (the steady-state average for the 3000-hotel subset):

| Cost Component        | $/Booking (100k bookings) |
|----------------------|----------------------------|
| Embedding refresh    | $21.44 / 100,000 = $0.00021 |
| Query inference      | $8.00 / 100,000 = $0.00008  |
| Drift sampling       | $0.56 / 100,000 = $0.000006 |
| GPT-4o-mini reranker | $5.00 / 100,000 = $0.00005  |
| Vendor chain-KB      | $116.67 / 100,000 = $0.00117|
| DW storage           | $0.46 / 100,000 = $0.000005 |
| Airflow              | $2.40 / 100,000 = $0.000024 |
| Monitoring/alerts    | $13.00 / 100,000 = $0.00013 |
| **Total**            | **$0.0018**                |

- **All-in per-booking cost:** **$0.0018** (0.18¢) with reranker enabled
- **Without reranker:** **$0.00175** (0.175¢)
- Marginal cost is dominated by the vendor chain-KB ($0.00117/booking), followed by inference and monitoring.

#### Sensitivity to volume

| Monthly Bookings | Total Monthly Cost | $/Booking |
|------------------|-------------------|-----------|
| 10,000           | $139.53           | $0.0140   |
| 100,000          | $167.53           | $0.0018   |
| 1,000,000        | $349.53           | $0.00035  |

- Fixed costs (vendor KB, monitoring) dominate at low volumes; compute scales sub-linearly due to batch efficiencies.

---

### S.3 — Operator-time savings calculation

#### Pre-automation (manual workflow):

- **Manual city-reconciliation rate:** ~45 bookings/hr/operator (actuals, Q3 2024)
- **Typical operator wage:** $34/hr loaded cost (incl. benefits/overhead)
- **Monthly bookings:** 100,000

**Manual effort required:**  
100,000 bookings / 45 bookings/hr = **2,222 hours/month**

**Manual cost:**  
2,222 hr * $34/hr = **$75,548/month**

#### Post-automation (with ML system):

- **Residual manual review rate:** ~0.8% of bookings, either due to model uncertainty or flagged low-confidence (see §G.4).
- **Operator time needed:** 100,000 * 0.008 = **800 bookings/month**
- **Time to resolve flagged bookings:** 800 / 45 hr = **~17.8 hours/month**
- **Operator cost:** 17.8 hr * $34/hr = **$605/month**

#### **Net operator-time cost reduction:**

| Scenario           | Monthly Operator Cost | Savings vs. Manual |
|--------------------|----------------------|--------------------|
| Pre-automation     | $75,548              | —                  |
| Post-automation    | $605                 | $74,943            |

- **% Reduction:** >99.2% in operator time/cost.
- Remaining effort is largely QA/oversight; full auto-approval possible with reranker and threshold tuning.

---

### S.4 — Break-even analysis by volume scenario

We model three scenarios: **low** (10k), **mid** (100k), and **high** (1M) monthly booking throughput.

#### S.4.1 — Direct cost vs. manual baseline

| Scenario   | Bookings/mo | All-in ML Cost | Manual Cost | Net Savings | Operator Time Saved |
|------------|------------|----------------|-------------|-------------|--------------------|
| Low        | 10,000     | $139.53        | $7,555      | $7,415      | ~222 hours         |
| Medium     | 100,000    | $167.53        | $75,548     | $75,380     | ~2,222 hours       |
| High       | 1,000,000  | $349.53        | $755,480    | $755,131    | ~22,222 hours      |

- **Break-even point:** Occurs at ~2,000 bookings/month ($116.67 fixed cost / $34/hr = 3.4 operator-hours), i.e., below almost any realistic deployment.
- **At scale:** Operator time savings outpace ML system costs by 2–3 orders of magnitude.

#### S.4.2 — Cost sensitivity with/without GPT-4o-mini reranker

| Scenario   | ML Cost (reranker on) | ML Cost (reranker off) | Difference |
|------------|----------------------|------------------------|------------|
| Low        | $139.53              | $134.53                | $5.00      |
| Medium     | $167.53              | $162.53                | $5.00      |
| High       | $349.53              | $344.53                | $5.00      |

- **GPT-4o-mini reranker** is a minor cost lever at all but the lowest volume, justified if it delivers even a small reduction in manual review or increases top-1 accuracy.

#### S.4.3 — Operator break-even threshold

Let \( x \) = bookings/month at which ML system cost equals operator time saved.

\[
\text{Solve: } (\text{ML cost at } x) = (\text{manual operator cost at } x) - (\text{residual operator cost at } x)
\]

Given ML cost is dominated by fixed vendor KB until ~100k bookings, break-even occurs at \( x \ll 10,000 \). In practice, any deployment above 2,000 bookings/month is strongly cost-favorable.

---

### S.5 — Additional notes and caveats

- **Vendor catalog cost** is the largest fixed expense; if replaced with in-house KB or open data, per-booking cost drops by >60%.
- **Burst scaling:** For >1M bookings/month, batch and inference compute discounts (e.g., reserved GPU, volume API) can reduce cost by 30–50%.
- **Operator QA:** If regulatory policy mandates human double-check for all low-confidence rows, manual review rate rises to ~3%. Even then, net cost is $1,020/month—still a >98% reduction.
- **System reliability:** SLA and monitoring costs are non-trivial at high scale; however, these are essential for compliance and uptime, and shared across other ML pipelines.
- **Fallback path:** If all ML models fail, reverting to manual incurs only the marginal operator cost for the duration of outage; fixed system costs are sunk.

---

### S.6 — Summary

- **Total ML system cost for 100k bookings/month:** **$167.53** (all-in, with reranker)
- **Per-booking cost:** **$0.0018** (0.18¢)
- **Operator-time savings:** >99% at all modeled volumes
- **Break-even:** At <3 operator-hours/month (≅2,000 bookings), the ML system is cost-justified
- **Scale-up:** Unit cost declines further with volume, approaching $0.00035/booking at 1M bookings/month.
- **Key lever:** Vendor KB is the primary fixed cost; future reduction here further improves economics.

**For strategic and procurement discussions, see Priya’s stratified dashboard (Appendix Q) and contact Lin for governance/contractual questions.**

---

**End of Appendix S**
