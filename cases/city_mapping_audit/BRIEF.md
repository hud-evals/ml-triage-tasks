# city-mapping-audit — orientation

Internal project: map free-text hotel names to canonical city strings.
Shipped to production in stages over the last two quarters; leadership
has asked for a second opinion on Mei Chen's final recommendation
(see `reports/final_recommendation.md`) before signing off.

## Team

- **Mei Chen** — lead, departed 2025-Q4. Authored most of `reports/`,
  `notes/one_pager_*`, `notes/retro_2025q1.md`.
- **Jordan Rao** — contractor on the `openai_3large` embeddings branch;
  departed 2025-Q3.
- **Priya Joshi** — junior engineer, current.
- **Arjun Patel** — engineering manager, current.

## Where things live

- `data/bookings.parquet` — raw booking observations; source of truth
  for any derived ground-truth mapping.
- `embeddings/` — per-method vectors + name indices.
- `ground_truth/` — derived hotel→city mappings.
- `src/` — eval + fuzzy-match helpers.
- `configs/` — one yaml per method.
- `runs/` — historical top-K outputs per method/rev.
- `reports/` — decision documents.
- `notes/` — working files (Slack copies, one-pagers, retros).
- `prs/` — merged PR descriptions pertinent to the pipeline.

That's all the context you should need. Dig in.

## Analytics snapshots

`analytics/dashboard.md` and its companion `_notes.md` files summarise top-K per method, stratified results by lexical overlap, the cost-vs-accuracy picture, and the suspect openai_3large number — each with a PNG under `analytics/` and the same content in text form. Worth reading before you form an opinion on the ship decision.


---

## Appendix A — project history timeline (2024-Q4 through 2025-Q4)

### 1. Baseline: Fuzzy-Only Matching (2024-Q4)

The project kicked off with a barebones fuzzy-matching approach, motivated by a need to rapidly prototype city-mapping on the full 3000-hotel subset. Initiated by Mei Chen, with oversight from Arjun Patel, the initial pipeline leveraged standard libraries: `fuzzywuzzy`'s `partial_ratio` and `wratio` scores were chosen for their no-nonsense API and tractable speed on commodity hardware. Hannah Kim and Martin Oliveira (product) pushed for quick wins to demonstrate the feasibility of automated mapping and reduce manual curation. The team established a ground-truth subset using a semi-random sample from `bookings.parquet`, which revealed early that fuzzy-only recall plateaued at sub-0.45 top-1 accuracy, with partial_ratio = 0.4407 and wratio = 0.4223, both notably sensitive to token order and punctuation noise. Mei documented known limitations in `notes/retrospective_baseline.md`: brittle to name variants, inconsistent on international properties, and a pain for anything non-Latin. Still, it provided a crucial benchmark and let Priya Joshi get her bearings on the codebase. This phase also set the precedent for storing all outputs in `runs/`, which paid dividends later.

### 2. Embedding Experiments: Early Forays (2025-Q1)

With the fuzzy baseline’s limitations laid bare, the team (now including Jordan Rao on contract) began exploring sentence-embedding models to better capture semantic similarity between hotel names and city strings. Using off-the-shelf MiniLM and SBERT variants, Priya ran the first batch of embedding evaluations against the 3000-hotel subset. Results were promising, if uneven — early configs improved robustness to token order and diacritics but were slow to compute on the full data and required non-trivial post-processing to align scores with fuzzy-match outputs. Arjun and Mei debated the right thresholding approach in `notes/one_pager_thresholds.md`. Hannah flagged concerns that recall gains wouldn’t justify compute unless accuracy jumped past 0.5. The team also faced their first “embedding drift” incident when a version mismatch in `embeddings/` caused a week of noisy evals (see `prs/2025-01-emb-fix.md`). These early experiments set the stage for more systematic analytics and a new round of model onboarding.

### 3. MiniLM Analytics Rollout (Internal Only, 2025-Q1)

The MiniLM model emerged as the most promising embedding baseline, pushing top-1 accuracy to 0.3937 — just shy of fuzzy but with clear wins on non-exact matches and international names. Mei and Priya collaborated to wire up a MiniLM-specific pipeline, storing intermediate vectors under `embeddings/minilm/`. The analytics snapshot (`analytics/dashboard.md`) was updated to include both fuzzy and MiniLM head-to-head, with stratified breakdowns by lexical overlap and booking volume. Internal-only access was enforced after Lin Zhao flagged privacy concerns related to vector storage formats. Martin and Hannah used these analytics to pitch the project’s progress in Q1 product review, highlighting the improved stability and broader coverage. The team also started formalizing method configs in `configs/minilm.yaml`, a move that later smoothed onboarding of additional methods. This phase marked the first time method-level evals were visible outside engineering, helping leadership visualize progress — even if MiniLM’s raw score didn’t blow fuzzy out of the water.

### 4. OpenAI Embedding Onboarding (2025-Q2)

With budget unlocked for cloud APIs, Arjun greenlit a controlled onboarding of OpenAI’s embedding endpoints. Jordan spearheaded adapter code to batch hotel and city name queries, wrangling rate limits and API quirks. Mei authored the first OpenAI-specific configs (`configs/openai_3small.yaml` and `openai_3large.yaml`), and Priya ran a full sweep on the validation set. The openai_3small model landed a top-1 accuracy of 0.4687 — a decent leap over both fuzzy and MiniLM. Leadership responded favorably, and Hannah began using these numbers in external product discussions. Lin worked with Priya to audit API data retention, ensuring compliance with internal governance policy. The team documented this phase heavily in `reports/openai_eval_summary.md`, including cost-per-1k inference calculations (which became a recurring talking point). OpenAI onboarding proved to be a turning point: it validated the bet on embeddings and set the bar for subsequent model experiments, but also raised questions about vendor lock-in and reproducibility.

### 5. OpenAI_3large: Arrival and Verification Issues (2025-Q2–Q3)

The much-anticipated openai_3large model arrived mid-2025-Q2, with initial results reported at an eyebrow-raising 0.6981 top-1 accuracy on the 3000-hotel subset. Jordan and Mei led the first round of analytics, but repeatability surfaced as a major issue. Multiple reruns yielded slightly different top-K lists owing to API versioning and a silent change in tokenization logic on OpenAI’s side (see `notes/openai_3large_anomalies.md`). Priya noticed inconsistencies between `runs/` outputs and dashboard summaries, prompting a deeper audit. Martin and Hannah fielded repeated questions from leadership about the reliability of these results for go/no-go decisions. Lin Zhao convened a one-off review to discuss whether the unverified 0.6981 could be used in official reporting. Ultimately, the team agreed to treat openai_3large as "directionally promising but not production-grade" until OpenAI stabilized their API, and annotated all references as such in both `analytics/` and `reports/`.

### 6. Leadership Review Buildup (2025-Q3)

As the project neared its decision point, demand for defensible analytics and transparent tradeoffs escalated. Martin and Hannah coordinated a series of reviews with product leadership, using the stratified outputs from `analytics/` and annotated dashboards. Arjun and Priya fielded technical deep-dives from stakeholders — “Why does openai_3small outperform everything else, and can we trust 3large?” Lin Zhao joined as official governance rep, steering the team toward more formal documentation and pushing for reproducibility guarantees. Mei Chen, still lead at this point, authored the initial draft of the “final recommendation” (see `reports/final_recommendation.md`), providing a clear-eyed summary of strengths, weaknesses, and open risks for each method. This period also saw a spike in Slack traffic and ad-hoc Zooms as the team worked to align on a story that would withstand both product scrutiny and governance review.

### 7. ADR Culture Adoption (2025-Q3)

In response to repeated confusion over model selection, run configs, and experiment interpretation, the team embraced an "Architecture Decision Record" (ADR) workflow. Mei and Arjun championed this change, creating a `notes/adr/` directory and mandating a record for any method onboarding, threshold tuning, or major config change. Priya drafted the first ADR for OpenAI embedding selection, while Lin Zhao contributed a governance ADR template. Martin and Hannah, from product, provided feedback on clarity and audience targeting. The ADR process quickly paid off: onboarding of new models became smoother, and future debates referenced documented tradeoffs and rationale instead of revisiting old Slack threads. This culture shift helped stabilize the project as departures loomed and institutional memory risk increased.

### 8. Departures and Transition (2025-Q3–Q4)

By mid-2025-Q3, both Jordan Rao (contractor, embeddings lead) and Mei Chen (project lead) announced their departures. Arjun Patel took on additional responsibility as interim lead, with Priya Joshi stepping up on both technical and documentation fronts. Hannah and Martin increased their involvement, focusing on maintaining continuity from the product side and ensuring handoff of institutional knowledge. Lin Zhao formalized a governance check-in cadence, reviewing all new configs and analytic snapshots. The team prioritized updating `reports/` and `notes/` to reflect the new staffing landscape, and all critical ADRs were reviewed one last time by Mei before her exit. Despite some loss of velocity, the transition period was notable for its focus on documentation, reproducibility, and clarity — setting the project up for stable maintenance and, eventually, sign-off.

### 9. Final Recommendation and Second Opinion (2025-Q4)

Following Mei Chen’s departure, leadership requested a second opinion on her “final recommendation” to validate the chosen mapping method and ensure confidence before sign-off. Arjun and Priya led a comprehensive re-audit, revisiting the canonical numbers (openai_3small top-1 = 0.4687, minilm = 0.3937, partial_ratio = 0.4407, wratio = 0.4223, openai_3large at unverified 0.6981) and confirming that all analytics snapshots matched the underlying `runs/`. Lin Zhao facilitated a governance review, confirming that data provenance and model selection were documented and reproducible. Martin and Hannah prepared a final product summary for leadership, incorporating both original recommendations and the second-opinion findings. The project, now fully documented and rigorously reviewed, was poised for official closure pending leadership approval, capping off a year-long cycle of experimentation, iteration, and process improvement.

### 10. Post-Sign-Off and Maintenance Planning (Late 2025-Q4)

With leadership sign-off imminent, the team shifted focus to maintenance and future-proofing. Priya took ownership of regression testing, ensuring that any new hotel or city string could be mapped using the locked-in pipeline with consistent outputs. Arjun and Lin established a quarterly review calendar for both analytics refreshes and compliance checks. Product (Hannah and Martin) began outlining a v2 wish list, including improved handling for ambiguous cases and potential multi-lingual expansion. All canonical numbers and configs were frozen, with a summary table added to `analytics/dashboard.md`. The project entered a new, steadier phase — fully transitioned from a high-volatility experiment to a stable, referenceable component within the larger bookings stack.

---

# Appendix B — detailed directory-by-directory walkthrough

This appendix provides a practical overview of each top-level directory in the hotel-to-city ML mapping project repo. Use this if you’re onboarding, debugging, or just lost in the guts. (Ping Arjun or Priya if you hit anything inconsistent with what’s below — we try to keep this up to date.)

---

## `data/`

**Key files:**
- `bookings.parquet`
- (occasionally: `*_sample.csv`, `legacy_hotels.json`)

**Purpose:**  
This is the landing zone for raw and minimally processed data. Chief among these is `bookings.parquet`, which is the canonical source for all hotel name and city string pairs. Most downstream artifacts—ground truth, embeddings, eval splits—are ultimately derived from here.

**Historical notes & gotchas:**  
- The 3000-hotel evaluation subset is always extracted from this parquet; don’t trust old CSVs that might be floating around.
- Early pilots used `legacy_hotels.json`, but this is now deprecated—ignore unless you’re debugging ancient bugs (Jordan’s 2024Q2 PRs).
- Watch for schema tweaks: column names have changed (see `notes/schema_change_2025q1.md` for the last big one).
---

## `embeddings/`

**Key files:**
- `openai_3small.vec`, `minilm.vec`, `openai_3large.vec` (and index files)
- `name_to_idx.json`
- `*_meta.yaml`

**Purpose:**  
Contains all generated embedding vectors for hotel names, organized by method. Each vector file matches a method (e.g., `openai_3small`, `minilm`) and has an accompanying mapping from raw hotel name to index. These are loaded by the matching and evaluation code in `src/`.

**Historical notes & gotchas:**  
- Only the 3000-hotel subset is embedded unless otherwise noted.
- Jordan’s `openai_3large` embeddings were only partially validated; see the flagged accuracy in analytics.
- Some embedding runs have mismatched indices due to name de-duplication bugs—check vector shapes before running batch jobs.
---

## `ground_truth/`

**Key files:**
- `hotel_to_city.json`
- `eval_split_*.json`
- `README.md` (sparse)

**Purpose:**  
Ground-truth mappings between hotel names and canonical city strings, derived from `data/bookings.parquet`. Also includes evaluation splits for benchmarking.

**Historical notes & gotchas:**  
- All numbers referenced in reporting (e.g., 0.4687 top-1 for `openai_3small`) use these mappings.
- Splits are deterministic unless you override the random seed.
- Mei’s team debated whether to "collapse" similar city names—current ground truth does **not** collapse.
---

## `src/`

**Key files:**
- `eval.py`, `fuzzy_match.py`, `embedding_match.py`
- `helpers.py`
- (sometimes: `notebook/` with old experiments)

**Purpose:**  
All core evaluation scripts and matching logic live here. This is where you’ll find the implementation for both embedding and fuzzy matching, as well as evaluation runners.

**Historical notes & gotchas:**  
- Some scripts assume relative paths from project root; use with care if running from subfolders.
- `fuzzy_match.py` contains the canonical `partial_ratio` (0.4407) and `wratio` (0.4223) implementations—see docstrings for method specifics.
- `notebook/` is a dumping ground for scratch code and is not productionized.
---

## `configs/`

**Key files:**
- `openai_3small.yaml`, `minilm.yaml`, `fuzzy.yaml`, etc.

**Purpose:**  
Method-specific YAML config files that define hyperparameters, thresholds, and any pre/post-processing rules for each matching approach.

**Historical notes & gotchas:**  
- Changing a config does not automatically re-trigger embedding generation—run the relevant script in `src/`.
- Some configs have legacy fields from earlier pipelines (e.g., `legacy_thresholds`).
- Always check the `notes/` for config rationale—Mei documented a few "gotcha" parameters.
---

## `runs/`

**Key files:**
- `openai_3small_topk.json`, `minilm_topk.json`, etc.
- Timestamped subfolders for historical runs.

**Purpose:**  
Houses the output of each method’s top-K city predictions for the hotel set, along with batch results from evaluation runs. Useful for diffing performance across method versions or parameter tweaks.

**Historical notes & gotchas:**  
- Only the 3000-hotel test set is covered unless noted.
- Some older runs used a different city canonicalization—stick to runs after `2025-01-14` for apples-to-apples comparisons.
- Periodically cleaned up (see `prs/cleanup_runs_*.md`), but not always timely.
---

## `reports/`

**Key files:**
- `final_recommendation.md`
- `method_comparison.md`
- `error_analysis_2025q2.md`

**Purpose:**  
Authoritative writeups and decision docs. Most important: Mei’s final recommendation (the one under review) and supporting analyses.

**Historical notes & gotchas:**  
- Some reports reference now-retired methods (e.g., “bert-tiny v1”).
- The `final_recommendation.md` is still the single source for sign-off, pending governance review.
- Always check report dates—some numbers (esp. openai_3large’s 0.6981) come with caveats.
---

## `notes/`

**Key files:**
- `one_pager_*`
- `retro_2025q1.md`
- `schema_change_2025q1.md`
- Slack export snippets

**Purpose:**  
Working notes, design discussions, and retrospectives. This is the team’s informal brain dump, including retros, one-pagers on method choices, and Slack export snippets for sticky issues.

**Historical notes & gotchas:**  
- Mei and Priya were diligent about summarizing big pivots here.
- Some notes predate major schema changes—check `schema_change_2025q1.md` for context.
- Not everything here is canonical; defer to `reports/` for “official” numbers.
---

## `prs/`

**Key files:**
- `pr_###.md` (merged PR summaries)
- `cleanup_runs_2025q1.md`
- `embedding_update_2025q2.md`

**Purpose:**  
Summaries of merged pull requests that affected the pipeline, with rationales and reviewer notes. Good for tracking why/when a major change shipped.

**Historical notes & gotchas:**  
- Not every PR made it here—Martin flagged a few missing summaries in late 2024.
- Useful for reconstructing how config or data assumptions changed, especially around embedding updates.
---

## `meetings/`

**Key files:**
- `2025-01-09_gov_review.md`
- `2025-03-17_retro.md`

**Purpose:**  
Meeting notes and transcripts, mainly governance/leadership reviews and team retros. Lin Zhao’s governance reviews are all captured here.

**Historical notes & gotchas:**  
- Attendance varied; some meetings are just action-point summaries.
- Check for action items that didn’t make it into `prs/` or `notes/`.
---

## `logs/`

**Key files:**
- `embedding_batch_*.log`
- `eval_run_*.log`

**Purpose:**  
Raw logs from batch jobs, especially embedding generation and evaluation runs. Useful when debugging pipeline or infra hiccups.

**Historical notes & gotchas:**  
- Log verbosity varies wildly by script and author (Jordan was especially chatty).
- Rotation isn’t automated—old logs may linger.
---

## `analytics/`

**Key files:**
- `dashboard.md`, `dashboard_notes.md`
- `openai_3small_vs_minilm.png`
- Various `*_breakdown.png`, `*_table.csv`

**Purpose:**  
Snapshot summaries of method performance, stratified breakdowns (e.g., by lexical overlap), and cost/accuracy tradeoff charts. Every number referenced in `reports/` should be traceable here.

**Historical notes & gotchas:**  
- The openai_3large numbers (0.6981 top-1) are presented with clear caveats—see both the text and PNGs.
- All analytics are based on the current 3000-hotel eval set.
- Hannah and Martin supervise updates here; ping them for dashboard questions.

---

---

## Appendix C — On-call and Handoff Guide

This appendix provides the operational framework for sustaining and supporting the hotel-to-city ML matching pipeline. It covers the shift rotation, escalation and paging order, alerting conventions, recurring team rituals, and a structured checklist for major personnel transitions, specifically referencing the handoff process following Mei Chen’s 2025-Q4 departure.

---

### 1. Shift Rotation and On-Call Coverage

#### 1.1. Primary and Secondary On-Call

- **Primary On-Call:** Priya Joshi  
  - Responsible for first response to all system alerts, including after-hours (see paging order).
  - Handles triage, mitigation, and initial incident documentation.

- **Secondary On-Call:** Arjun Patel  
  - Steps in if Priya is unavailable or if escalation is required (e.g., pipeline is down >2h, or customer SLAs risk breach).
  - Provides managerial oversight and can authorize system-level interventions.

#### 1.2. Rotation Schedule

- The on-call schedule alternates weekly between Priya and Arjun:
    - **Week 1:** Priya (Primary), Arjun (Secondary)
    - **Week 2:** Arjun (Primary), Priya (Secondary)
- Covering swaps are coordinated in the `#hotel-city-ops` Slack channel at least 3 days in advance.
- Holidays and time off: The non-scheduled engineer assumes both roles, with coverage requests to Hannah Kim (Product) if both are unavailable. Martin Oliveira (Product) is backup for non-engineering escalations.

**Table: On-Call Rotation Example**

| Week        | Primary On-Call | Secondary On-Call | Notes                       |
|-------------|-----------------|-------------------|-----------------------------|
| 2025-W49    | Priya           | Arjun             | Standard rotation           |
| 2025-W50    | Arjun           | Priya             | Standard rotation           |
| 2025-W51    | Priya           | Arjun             | Priya swap for Arjun PTO    |

---

### 2. Paging and Escalation Order

**Paging is initiated via Opsgenie or direct Slack ping (if urgent).**

#### 2.1. Escalation Ladder

1. **Primary On-Call (Priya/Arjun)**
2. **Secondary On-Call (Arjun/Priya)**
3. **Product Escalation (Hannah Kim)**
4. **Governance/Ethics (Lin Zhao)** — Only for issues involving data integrity, privacy, or compliance anomalies.
5. **Historical Context (Martin Oliveira)** — For product-scope clarifications or customer comms support.

#### 2.2. Non-Engineering Escalations

- **Product-side:** Hannah Kim owns comms for customer-facing incidents.
- **Governance:** Lin Zhao is notified for data drift or privacy alerts with user impact >100 bookings.

---

### 3. Common Alerts and Playbooks

#### 3.1. Alert Types

| Alert Name                | Trigger Condition                                 | Severity | Playbook Reference      |
|---------------------------|---------------------------------------------------|----------|------------------------|
| Drift Alert               | City mapping accuracy drops >0.02 vs. baseline    | High     | `playbooks/drift.md`   |
| Latency Degradation       | p95 inference latency >2s for >10min              | Medium   | `playbooks/latency.md` |
| Embedding Load Failure    | Failure to read vectors from `embeddings/`        | High     | `playbooks/embed.md`   |
| Data Pipeline Stale       | No new bookings in `data/bookings.parquet` >24h   | Medium   | `playbooks/data.md`    |
| Disk Utilization High     | `/data` partition >85%                            | Low      | `playbooks/disk.md`    |

#### 3.2. Alert Response Summary

**Drift Alert**
- Validate: Compare top-1 method accuracy to canonical numbers (0.4687 for openai_3small, etc.).
- Investigate: Check for recent data schema changes, or suspicious booking clusters.
- Mitigate: Roll back to last known-good model config in `configs/`.
- Document: File incident in `notes/incidents/`.

**Latency Degradation**
- Validate: Confirm via Grafana dashboards.
- Investigate: Correlate with concurrent jobs, embedding server load.
- Mitigate: Restart affected pods; escalate if unresolved.

**Embedding Load Failure**
- Validate: Attempt manual load from `embeddings/`.
- Investigate: Check for file corruption, permission issues, or incomplete sync.
- Mitigate: Re-run embedding generation job; escalate to Arjun if failure persists.

---

### 4. Weekly and Quarterly Rituals

#### 4.1. Weekly Rituals

| Day     | Ritual              | Attendees                      | Description                                                                                      |
|---------|---------------------|-------------------------------|--------------------------------------------------------------------------------------------------|
| Monday  | Stand-up            | Priya, Arjun, Hannah, Martin  | 30min sync. Review prior week’s incidents, alert summaries, and upcoming priorities.             |
| Tuesday | 1:1s                | Arjun ↔ Priya, Hannah ↔ Martin| 20min check-ins. Focus on personal blockers, capacity, and growth.                               |
| Thursday| Drift Review        | Priya, Arjun                  | 45min. Deep dive into latest drift metrics, compare to canonical numbers, flag for action.       |

- **All rituals are remote-first, Zoom links in recurring calendar invites.**
- **Ritual notes are stored in `notes/rituals/YYYYMMDD.md`.**

#### 4.2. Quarterly Rituals

| Ritual            | Timing         | Facilitator   | Description                                                                              |
|-------------------|---------------|---------------|------------------------------------------------------------------------------------------|
| Retro             | End of Quarter| Arjun         | Full team, including product. Review what went well, what didn’t, process and tech debt. |
| OKR Cycle         | Start of Q    | Hannah Kim    | Product and engineering align on new Objectives and Key Results, set targets.             |
| Roadmap Review    | Mid-Quarter   | Martin O.     | Update and revalidate technical and product roadmaps; flag resource risks.                |

---

### 5. Handoff Checklist (Mei Chen Departure, 2025-Q4)

This checklist was followed for the leadership-approved transition following Mei's departure. *Any future departures at the lead/architect level should use the same structure.*

#### 5.1. Code and Documentation

- [x] **All code pushed to `main` branch; feature branches merged or closed.**
- [x] **`reports/` directory up-to-date:** Final recommendation (`final_recommendation.md`) and all supporting docs complete.
- [x] **README.md updated:** At both project root and in each major subdirectory.
- [x] **Data lineage:** Explicit documentation in `ground_truth/README.md` for data derivation steps.

#### 5.2. Knowledge Transfer

- [x] **Shadowing:** Priya shadowed Mei for two weeks on both on-call and drift review rituals.
- [x] **Pairing:** Arjun and Priya completed joint review of all pipeline config files (`configs/`), with Mei present.
- [x] **Walkthrough Videos:** Mei recorded three screen-share walkarounds (see `notes/mei_walkthroughs/`) covering:
    - End-to-end inference run
    - Data refresh workflow
    - Alert triage and incident reporting

#### 5.3. Stakeholder Communication

- [x] **Intro email:** Sent to product (Hannah, Martin), governance (Lin), and leadership with coverage update.
- [x] **Customer comms:** Hannah drafted and sent external update (if needed) on team change, with Martin reviewing language.

#### 5.4. Access and Permissions

- [x] **Accounts:** Mei’s SSO and data warehouse access revoked post-handoff.
- [x] **PagerDuty/Opsgenie:** On-call schedules updated, notifications tested.
- [x] **Calendar:** Recurring standups and rituals reassigned to Arjun/Priya as facilitators.

#### 5.5. Open Risks and Follow-ups

- [x] **Unverified openai_3large number:** Jordan’s work on this branch left as-is; flagged in `analytics/_notes.md`.
- [x] **Data privacy review:** Lin Zhao briefed on Mei’s last audit findings, scheduled quarterly follow-up.
- [x] **Technical debt:** Documented in `notes/retro_2025q4.md` for tracking in next OKR cycle.

---

### 6. Additional Notes

- **Slack Channels:**  
  - `#hotel-city-ops` for urgent on-call and incident threads.
  - `#hotel-city-product` for roadmap and customer feature discussion.
  - `#hotel-city-gov` for all governance, privacy, and legal issues (Lin Zhao primary).
- **Incident Postmortems:**  
  - Required for any high-severity alert or >2h downtime.
  - Follow the template in `notes/incidents/postmortem_template.md`.

- **Canonical Metrics Reminder:**  
  - Always reference the stable numbers:  
    - `openai_3small` top-1 = **0.4687**
    - `minilm` = **0.3937**
    - `partial_ratio` = **0.4407**
    - `wratio` = **0.4223**
    - `openai_3large` = **0.6981** (unverified)
  - These are the standard for drift detection, regression checks, and reporting.

---

#### For any questions or deviations from this guide, contact Arjun Patel (engineering), Hannah Kim (product), or Lin Zhao (governance).

---

**End of Appendix C.**

---

# Appendix D — historical glossary and acronyms

| Term | Definition |
|------|------------|
| **3000-hotel subset** | The canonical evaluation set of 3,000 hotels, sampled for reproducibility and coverage. All major metrics are reported against this slice. |
| **any-match semantics** | A match approach where a hotel-city pair is considered correct if any token in the city name appears in the hotel string. Used as a baseline. |
| **Arjun** | Arjun Patel, current engineering manager. Typically referenced for sign-off or historical context on architecture decisions. |
| **batch scoring** | Running inference on large batches of hotel strings, as opposed to real-time single-instance queries. |
| **bookings.parquet** | Master data file containing all raw booking records; basis for truth-mapping and evaluation. |
| **canonical city string** | The standardized representation of a city, as maintained in ground truth and outputs. |
| **chain-KB** | Internal knowledge base (Confluence, mostly) tracking all chain→city mappings, hotel groupings, and edge cases. |
| **city-mapping-audit** | The internal codename for this project: hotel free-text name to canonical city mapping. |
| **config YAML** | The per-method configuration file living in `configs/`, controlling scoring and thresholds. |
| **confidence threshold** | Minimum similarity/score for a match to be accepted. Often method-specific. |
| **cross-encoder** | ML model class where hotel and city texts are jointly encoded for richer context, at the cost of inference speed. Not used in production, but piloted. |
| **data drift** | The phenomenon where incoming data distributions shift over time, degrading model performance. Monitored via the drift dashboard. |
| **decision doc** | A formal write-up justifying a method or parameter, mostly in `reports/`. Mei Chen authored most of these. |
| **distance bucket** | Evaluation stratification by spatial/geographic distance between predicted and true city. Used to triangulate error types. |
| **drift dashboard** | Internal monitoring tool displaying feature and output drift statistics, with red/yellow/green indicators. |
| **embedding** | Vector representation of text as used in ML methods (e.g., minilm, openai_3small). Drives similarity calculations. |
| **eval script** | Python code under `src/` for running method evaluations and metrics computation. |
| **exclusion list** | Hotel names manually blacklisted from training due to ambiguity or mapping errors. |
| **fuzzy matching** | String comparison technique allowing for minor character differences (e.g., Levenshtein ratio, partial_ratio, wratio). |
| **ground truth** | The official mapping of hotels to cities, derived from bookings.parquet and manual curation. Stored in `ground_truth/`. |
| **Hannah** | Hannah Kim, product manager. Owner of the ship decision and stakeholder sign-off. |
| **historical retro** | Team post-mortem document, usually filed quarterly (see `notes/retro_2025q1.md`). |
| **Jaccard score** | Lexical overlap metric: intersection over union of tokens between hotel and city string. Used for quick diagnostic buckets. |
| **Jordan** | Jordan Rao, ML contractor (2025-Q1 to Q3). Led openai_3large experiments and benchmarking. |
| **KNN retrieval** | Nearest neighbor search in embedding space; core to all vector-based matching methods. |
| **latency budget** | Maximum acceptable time (usually 95th percentile, p95) for a mapping request to complete. Tracked in dashboard. |
| **lexical overlap bucket** | Evaluation stratification based on how many city tokens appear in the hotel string. High/low/no overlap buckets are called out in analytics. |
| **Lin Zhao** | Internal governance and compliance lead. Signs off on data provenance, privacy, and auditability. |
| **manual override** | Hard-coded mapping or exclusion, typically made for VIP clients or persistent edge cases. |
| **mapping error** | Any case where the predicted city does not match the ground truth city for a given hotel. |
| **Martin Oliveira** | Product owner for city-to-hotel matching; works with Hannah on roadmap and go/no-go. |
| **Mei Chen** | Original project lead and primary author of most docs, including final recommendations. Departed 2025-Q4. |
| **method config** | The YAML or Python dict specifying the parameters for a given matching approach. |
| **minilm** | ML model for generating embeddings; performant, lightweight, and used as a reference method. |
| **no-overlap bucket** | Evaluation set where hotel and city strings share zero tokens. Most challenging for string-based methods. |
| **openai_3large** | Large language model embedding method, yielding best unverified results (top-1 = 0.6981). Jordan’s branch, not prod-verified. |
| **openai_3small** | Smaller embedding model; main production vector method. Top-1 score is 0.4687 on the 3k subset. |
| **p95 latency** | 95th percentile latency; used as a performance target for production readiness. |
| **partial_ratio** | Fuzzy matching metric (from FuzzyWuzzy), reports 0.4407 top-1 on the 3k set. |
| **pipeline** | End-to-end system from data ingest, preprocessing, matching, to output. |
| **post-processing** | Steps after initial match (e.g., filtering, manual overrides, additional heuristics). |
| **precision@1** | Fraction of correct top-1 matches; primary accuracy metric. |
| **Priya** | Priya Joshi, current junior engineer. Maintains data pipeline and runs weekly evals. |
| **production ready** | Methods or configs that have passed all latency, accuracy, and governance checks and can be deployed. |
| **pseudo-label** | Label generated by model inference rather than manual annotation, sometimes used to augment training. |
| **retrieval ceiling** | The theoretical best performance achievable by the retrieval component alone, with perfect downstream logic. |
| **retro** | Shorthand for retrospective; team discussions on what worked, what didn’t. |
| **run ID** | Unique identifier for each method run, tied to a config and timestamp. Indexed in `runs/`. |
| **seed set** | Initial sample of hotel names used to bootstrap ground truth or method evaluation. |
| **similarity score** | Numeric value representing how similar two strings or embeddings are. Higher means more likely a match. |
| **stopwords** | Common words (e.g., "the", "hotel", "inn") filtered out prior to token comparisons. |
| **string normalization** | Preprocessing step: removing punctuation, lowercasing, eliminating diacritics, etc. |
| **token overlap** | Number of shared tokens between hotel and city name; basis for buckets and some heuristics. |
| **top-1** | Whether the correct city is the top predicted output for a hotel name. |
| **wratio** | Weighted fuzzy string metric; performs better than partial_ratio on noisy inputs (0.4223 top-1 on 3k). |

*(Contact Priya or Arjun to add new entries or clarify legacy terms.)*

---

# Appendix E — FAQ for New Team Members

Welcome to the city-mapping-audit project! This FAQ compiles the most common questions new members (including recent joiners and interns) have raised. Many of these were originally answered by Mei Chen or Jordan Rao, with updates from Priya Joshi and Arjun Patel as of 2026. If you don't see your question here, or if something feels out of date, please ask in `#ml-mapping-core` or reach out to Priya directly.

---

## Table of Contents

1. [What does "end-to-end eval harness" mean in this project?](#1)
2. [How do I run the full evaluation pipeline from scratch?](#2)
3. [What’s the difference between a canonical run and an experimental run?](#3)
4. [Where do I store new evaluation outputs or artifacts?](#4)
5. [How do PR reviews work for this codebase?](#5)
6. [What’s the process for adding a new model or matching method?](#6)
7. [How do I know which documentation to trust vs. verify?](#7)
8. [What should I do if I notice an inconsistency in reported numbers?](#8)
9. [Who maintains the "source of truth" for hotel-to-city mappings?](#9)
10. [How do I update or regenerate the ground-truth mappings?](#10)
11. [How are embeddings managed and versioned?](#11)
12. [Where do I put exploratory analysis or ad hoc results?](#12)
13. [How are method configs managed?](#13)
14. [What is the process for updating configs?](#14)
15. [How are the analytics dashboards generated and updated?](#15)
16. [Where do I find historical performance data for each method?](#16)
17. [What is the role of governance in this project?](#17)
18. [How are decisions around model selection documented?](#18)
19. [If I want to propose a new evaluation metric, where do I start?](#19)
20. [Who reviews production deploys or major algorithmic changes?](#20)
21. [How do I handle sensitive or PII data in my outputs?](#21)
22. [What should I know about reproducibility requirements?](#22)
23. [How do I request compute or storage resources?](#23)
24. [What are the canonical accuracy numbers for each method?](#24)
25. [How do I escalate if I believe a metric or output is incorrect?](#25)
26. [What are the preferred communication channels for questions?](#26)
27. [How do I onboard to the codebase efficiently?](#27)
28. [What are the most common pitfalls for new contributors?](#28)
29. [Who do I talk to for product direction or requirements?](#29)
30. [What does the team expect from documentation in PRs?](#30)

---

### 1. What does "end-to-end eval harness" mean in this project? <a name="1"></a>

In our context, the "end-to-end eval harness" refers to the set of scripts and utilities (primarily under `src/` and some helpers in `tools/` if you have legacy clones) that allow you to:

- Take a set of hotel names (from `data/bookings.parquet` or a derived subset),
- Generate candidate city strings using a given matching method (e.g., `openai_3small`, `minilm`, fuzzy matchers),
- Score and evaluate those candidates against the canonical ground-truth city mappings,
- Produce summary metrics (like top-1 accuracy, precision, recall), and
- Output per-example results and aggregate analytics to the appropriate directories.

The harness is intended to let you reproduce our headline results (see `analytics/dashboard.md`) as well as try out new models, configs, or matching approaches. "End-to-end" means from raw hotel entries all the way through to summary analytics and generated output files.

---

### 2. How do I run the full evaluation pipeline from scratch? <a name="2"></a>

Assuming you have the repo checked out, your environment set up (see `notes/env_setup.md` for conda or venv details), and access to the necessary data files, here’s the canonical process:

1. **Ensure data availability.**
   - Check that `data/bookings.parquet` exists and is readable.
   - Confirm that `ground_truth/hotel_to_city.parquet` (or `.csv` in older runs) is present.

2. **(Re)generate embeddings if needed.**
   - If you’re using a new model, run the relevant script in `src/embeddings/` (see method-specific README).
   - For existing methods, embeddings should be available under `embeddings/` by method and batch.

3. **Configure your run:**
   - Copy or edit a config from `configs/` for your method.
   - Standard config keys: input path, model/matcher, output path, batch size, and optional overrides (see comments in exemplar YAMLs).

4. **Run the evaluation:**
   - The main entrypoint is typically `src/eval.py`:
     ```
     python src/eval.py --config configs/my_run.yaml
     ```
   - This will generate per-example outputs (default to `runs/my_run/`) and a summary metrics file.

5. **Generate analytics and dashboards:**
   - Use `src/analytics/summarize.py` to process results:
     ```
     python src/analytics/summarize.py --input runs/my_run/results.parquet --output analytics/my_run_summary.md
     ```
   - For PNGs/plots, check `analytics/plotting.md` (requires matplotlib/seaborn).

6. **Document your run:**
   - Add a brief description in `runs/README.md` or, for major runs, as a new entry in `analytics/dashboard.md`.

If you get stuck, refer to `notes/e2e_eval_walkthrough.md` (originally by Jordan Rao, updated by Priya). Note: if you’re running on a non-canonical subset or with local changes, be sure to clearly label your outputs and adjust output paths accordingly.

---

### 3. What’s the difference between a canonical run and an experimental run? <a name="3"></a>

**Canonical run:**  
- Uses the officially blessed data subset (currently the 3000-hotel evaluation set; see `data/3000_eval_subset.parquet`).
- Employs a production-configured method (matching the YAML in `configs/` with no ad hoc overrides).
- Outputs to the standard naming convention in `runs/` (e.g., `runs/openai_3small_v1/`).
- Is reproducible and, if necessary, can be independently verified by another team member.
- Is eligible for inclusion in `analytics/dashboard.md` and referenced in reports.

**Experimental run:**  
- Deviates from the above in any way: e.g., new models, altered configs, different data splits, parameter sweeps, or local hacks.
- Must **not** overwrite canonical outputs.
- Should be clearly labeled as experimental in directory names, config files, and documentation.
- Results are for exploration or hypothesis generation, not for reporting as headline numbers.
- If an experiment looks promising, you’ll need to re-run it in canonical fashion before inclusion in any decision document.

*TL;DR: Canonical = official, repeatable, and trustworthy. Experimental = early-stage, for learning or iteration. Keep them separate.*

---

### 4. Where do I store new evaluation outputs or artifacts? <a name="4"></a>

- **Canonical runs:**  
  - Store all outputs under `runs/`, using a clear, method-specific directory structure. E.g.:
    ```
    runs/openai_3small_v1/
    runs/minilm_v2/
    runs/partial_ratio_baseline/
    ```
  - Each run directory should contain:
    - The config used (`config.yaml`)
    - Per-example results (`results.parquet` or `.csv`)
    - Summary metrics (`metrics.json` or `.md`)
    - Any logs (`log.txt`)

- **Experimental runs:**  
  - Use `runs/exp_{yourname}/` or similar. E.g.:
    ```
    runs/exp_priya_20260612/
    runs/exp_arjun_baseline_sweep/
    ```
  - Make it easy for reviewers to identify experimental vs canonical runs.

- **New embeddings:**  
  - Place under `embeddings/{method_name}/`, with batch or version suffixes as needed.

- **Analytics:**  
  - All dashboard summaries, plots, and comparative tables live under `analytics/`.
  - If you’re producing a new summary, either append to `dashboard.md` or create a new file in `analytics/` and link it in the master dashboard.

- **Working notes or ad hoc files:**  
  - Use `notes/`, with clear filenames and (ideally) date and author in the file name. E.g., `notes/exp_priya_20260612.md`.

If in doubt, ping Arjun or Priya for directory hygiene review before pushing a large batch of outputs.

---

### 5. How do PR reviews work for this codebase? <a name="5"></a>

- **Branching:**  
  - Use feature branches (`feature/{yourname}_...` or `bugfix/{...}`).

- **PR Description:**  
  - Summarize what you changed, why, and any relevant context or decisions (see `prs/` for exemplars).
  - Include links to relevant config files, runs, or analytics outputs.
  - If the PR affects canonical results, say so explicitly.

- **Reviewers:**  
  - At least one code reviewer (Priya for ML/infra, Arjun for pipeline or integration).
  - For anything touching product requirements or UX, loop in Hannah Kim or Martin Oliveira.

- **Review process:**  
  - Expect feedback within 1-2 business days.
  - Address comments in new commits (don’t force-push over reviewed history unless specifically asked).
  - For contentious or architecture-level changes, request a short call—async is preferred, but some topics are easier live.

- **Merge:**  
  - Once approved, squash-merge unless otherwise instructed.
  - Add a summary note in `prs/` (copy-paste your PR description and reviewer comments if useful).

- **Post-merge:**  
  - If your change affects pipeline outputs, update relevant dashboards or documentation as needed.
  - Announce major merges in `#ml-mapping-core`.

---

### 6. What’s the process for adding a new model or matching method? <a name="6"></a>

1. **Propose in `#ml-mapping-core` or via a short doc in `notes/`.**
   - Outline the motivation, expected benefits, and any risks.

2. **Create a feature branch.**
   - Place new code in `src/` (or `src/embeddings/` if model is embedding-based).
   - Add a new config YAML in `configs/`.

3. **Run experimental evaluations.**
   - Use an experimental data split (`runs/exp_{yourname}/`) — do not overwrite canonical outputs.
   - Document findings in `notes/` and, if promising, summarize in `analytics/`.

4. **Solicit feedback and review.**
   - Ping Priya or Arjun for initial code review.
   - If the method involves novel model weights or external dependencies, check with Lin Zhao (governance) on licensing and compliance.

5. **Finalize and standardize.**
   - Move model/matcher to canonical configs if approved.
   - Regenerate outputs on the 3000-hotel subset.
   - Summarize results in `analytics/dashboard.md`.

6. **Document and announce.**
   - PR as per [5. How do PR reviews work](#5).
   - Announce in `#ml-mapping-core` and tag relevant stakeholders.

---

### 7. How do I know which documentation to trust vs. verify? <a name="7"></a>

- **Trust:**
  - `analytics/dashboard.md` — unless marked as "suspect" or "unverified".
  - `runs/` canonical outputs that match their config and have not been superseded.
  - Decision documents in `reports/`.
  - Ground-truth mappings in `ground_truth/` (as of the last canonical run).
  - PR descriptions in `prs/` (assume up-to-date unless flagged).

- **Verify:**
  - Any number or claim marked "suspect", "unverified", or flagged in Slack.
  - `openai_3large` reported results (see notes in `analytics/dashboard.md`—these are unverified as of last audit).
  - Notes or one-pagers that predate a major code or data revision.
  - Ad hoc analyses in `notes/` or Slack pastes (these are for discussion, not for reporting).
  - Any claim that seems inconsistent with canonical numbers (see next question).

When in doubt, trace back to the underlying outputs in `runs/` and, if needed, rerun the pipeline yourself on the canonical subset.

---

### 8. What should I do if I notice an inconsistency in reported numbers? <a name="8"></a>

1. **Double-check the output:**  
   - Confirm which run or method the number came from.
   - Verify the config and data subset match the canonical definitions.

2. **Check for version drift:**  
   - Has the code, data, or embeddings changed since the number was reported?
   - Is there a newer canonical run or override that supersedes the old number?

3. **Audit the pipeline:**  
   - Rerun the eval harness on the canonical 3000-hotel subset, using the method in question.
   - Compare the generated metrics to what was reported.

4. **Ask for help:**  
   - If the mismatch persists, ping Priya or Arjun and share your findings.
   - For governance or compliance concerns, loop in Lin Zhao.

5. **Escalate if necessary:**  
   - If the number is in a published report or has been used for a ship decision, flag in `#ml-mapping-core` and add a "suspect" note in the relevant doc (do not silently edit canonical dashboards).

6. **Document:**  
   - Add a note to `notes/` or as a comment in the dashboard, summarizing what you found and how to reproduce it.

This is a detail-oriented team. If you spot something off, you’re doing the right thing—flag it, document it, and we’ll fix or clarify as a group.

---

### 9. Who maintains the "source of truth" for hotel-to-city mappings? <a name="9"></a>

- The canonical mapping is stored in `ground_truth/hotel_to_city.parquet`.
- This file is typically generated via a deterministic pipeline from `data/bookings.parquet`, using the logic in `src/ground_truth/derive_mapping.py` or (for legacy support) `src/ground_truth/legacy_mapping.py`.
- Updates to the mapping must be reviewed by Priya and, for major changes, Arjun.
- Changes should be documented in `ground_truth/CHANGELOG.md`, with a rationale, date, and author.
- For compliance or privacy review, Lin Zhao signs off on the mapping process.

If you notice drift between the ground-truth file and the latest data, consult with Priya before regenerating.

---

### 10. How do I update or regenerate the ground-truth mappings? <a name="10"></a>

1. **Sync with Priya or Arjun** before starting—ground-truth changes are rare and must be justified.

2. **Run the mapping script:**
   ```
   python src/ground_truth/derive_mapping.py --input data/bookings.parquet --output ground_truth/hotel_to_city.parquet
   ```

3. **Validate the output:**
   - Run `src/ground_truth/validate_mapping.py` against the newly created file.
   - Check for missing entries, duplicates, or unexpected city strings.

4. **Document the change:**
   - Add a note in `ground_truth/CHANGELOG.md`.
   - Include the date, author, and reason for the update.

5. **Review and sign-off:**
   - PR the changes to the core team.
   - Await sign-off from both engineering (Priya or Arjun) and governance (Lin Zhao).

6. **Announce in `#ml-mapping-core`** once merged.

Do **not** update the ground-truth mappings ad hoc or for minor experiments—this is a controlled process.

---

### 11. How are embeddings managed and versioned? <a name="11"></a>

- **Directory structure:**  
  - `embeddings/{method_name}/{version_or_date}/`
  - Example: `embeddings/openai_3small/2025q4/`

- **Naming:**  
  - Include model name, any relevant hyperparameters, and the data subset in the directory or file name.

- **Version control:**  
  - Embeddings themselves are too large for git; instead, store metadata (YAML or JSON) in the same directory, including:
    - Model config and version
    - Date generated
    - Data subset used
    - Author

- **Canonical embeddings:**  
  - Should only be generated on the official evaluation subset.
  - If you generate new embeddings for a canonical method, coordinate with Priya before updating the embeddings directory.

- **Experimental embeddings:**  
  - Store under a date- or user-specific subfolder (e.g., `embeddings/minilm/exp_priya_20260612/`).

- **Retiring old embeddings:**  
  - Archive, don't delete. Add an `ARCHIVED.md` with rationale and retirement date.

- **Reproducibility:**  
  - Always document the exact code and config used to generate embeddings, to allow others to reproduce.

---

### 12. Where do I put exploratory analysis or ad hoc results? <a name="12"></a>

- **Short-term analyses:**  
  - Place in `notes/`, with your name and date in the filename. E.g., `notes/exp_priya_city_overlap_20260612.md`.

- **Larger experiments:**  
  - Create a subfolder in `runs/` (e.g., `runs/exp_arjun_202606/`).
  - Reference these results in your notes file.

- **Dashboards or plots:**  
  - If not for the main dashboard, put in `analytics/exp_{yourname}/` and link from your notes.

- **Slack pastes:**  
  - Copy significant findings into `notes/` for posterity.

- **Documentation:**  
  - If your ad hoc analysis leads to a proposed change or insight, summarize in a new markdown file and announce in `#ml-mapping-core`.

---

### 13. How are method configs managed? <a name="13"></a>

- **All canonical configs live in `configs/`.**
- **Each config file is a YAML, named by method and version.**  
  - Example: `configs/openai_3small_v1.yaml`

- **Config keys:**  
  - Input/output paths
  - Model or matcher name
  - Hyperparameters (as applicable)
  - Seed or randomness controls
  - Any method-specific overrides

- **Experimental configs:**  
  - Use `configs/exp_{yourname}_...yaml` or similar.

- **Config documentation:**  
  - Each config should have a header comment with author, date, and purpose.

- **Versioning:**  
  - Significant changes = new config file. Do not overwrite canonical configs unless correcting a typo or error (in which case, document in the file and in PR).

---

### 14. What is the process for updating configs? <a name="14"></a>

1. **Edit a copy of the existing config.**
2. **Make changes and document in the header.**
3. **Test the new config via an experimental run.**
4. **If the config is for a canonical method/run:**
   - PR the new config with a clear description.
   - Get review from Priya or Arjun.
   - If approved, move the config to the main `configs/` directory and update any references.

5. **If the config is for a one-off experiment:**
   - Name accordingly and store in `configs/exp_{yourname}_...yaml`.
   - No formal review needed unless results may be published or used for decisions.

---

### 15. How are the analytics dashboards generated and updated? <a name="15"></a>

- **Dashboards are markdown files in `analytics/`**, primarily `dashboard.md` and supporting `_notes.md` files.

- **Generation:**  
  - Use scripts in `src/analytics/` to aggregate per-run metrics and output markdown summaries.
  - Plots are generated via `src/analytics/plotting.py` (see in-file docstring).

- **Updating:**  
  - After a new canonical run, append or update the relevant section in `dashboard.md`.
  - For major changes, update the companion notes file (e.g., `dashboard_notes.md`).
  - If a number is "suspect" or "unverified", clearly mark it as such.

- **Historical dashboards:**  
  - Do not delete old data—append new runs, with date and config.

- **Roles:**  
  - Priya maintains the main dashboard; others may append, but coordinate to avoid merge conflicts.

---

### 16. Where do I find historical performance data for each method? <a name="16"></a>

- **Primary source:**  
  - `analytics/dashboard.md` — summary table of top-1 and other metrics per method/version.

- **Details per run:**  
  - `runs/{method}/metrics.json` or `.md` — per-run metrics.
  - `analytics/{method}_summary.md` — method-specific notes and breakdowns.

- **For deeper dives:**  
  - Check `analytics/_notes.md` for stratified and breakdown analyses.
  - For very old runs, see archived outputs in `runs/ARCHIVED/`.

---

### 17. What is the role of governance in this project? <a name="17"></a>

- **Lin Zhao** is the governance lead.
- Responsibilities:
  - Oversight on data handling (esp. PII or compliance risks).
  - Approval for new external model dependencies, licensing, or changes to ground-truth mapping logic.
  - Periodic audit of pipeline reproducibility and data lineage.
- For any compliance, privacy, or external review questions, consult Lin before proceeding.

---

### 18. How are decisions around model selection documented? <a name="18"></a>

- **Core documents:**  
  - `reports/final_recommendation.md` (Mei Chen’s summary, with historical context).
  - `reports/decision_{date}_{topic}.md` for interim or special decisions.
  - `prs/` for PR-level rationale and review comments.

- **Supporting analyses:**  
  - `analytics/dashboard.md` and companion notes files.
  - Working notes in `notes/` (esp. `notes/retro_2025q1.md`).

- **Process:**  
  - Major model selection decisions are discussed in `#ml-mapping-core`, summarized in `reports/`, and referenced in dashboards.

---

### 19. If I want to propose a new evaluation metric, where do I start? <a name="19"></a>

1. **Draft a short proposal** (1-2 paragraphs) — rationale, definition, and why current metrics are insufficient.
2. **Share in `#ml-mapping-core`** for initial feedback.
3. **Add a prototype implementation** (preferably in `src/metrics/`), with a test config in `configs/exp_{yourname}_...yaml`.
4. **Run on a small subset** and share results in `notes/`.
5. **If promising, PR the metric code and supporting docs**.
6. **If adopted, update `analytics/dashboard.md` to include the new metric.**

---

### 20. Who reviews production deploys or major algorithmic changes? <a name="20"></a>

- **Engineering:**  
  - Arjun (manager) must sign off on any production-impacting changes.
  - Priya reviews ML or pipeline changes.

- **Product:**  
  - Hannah Kim or Martin Oliveira reviews for product alignment.

- **Governance:**  
  - Lin Zhao for compliance/data lineage.

- **Process:**  
  - Major changes require sign-off from at least two of the above before merging/deploying.

---

### 21. How do I handle sensitive or PII data in my outputs? <a name="21"></a>

- **Never include full hotel or user-identifiable info in public outputs or docs.**
- Use anonymized IDs or hashed representations in all analytics and outputs.
- Do not copy raw `bookings.parquet` data outside the repo or team-approved storage.
- For ad hoc analysis, scrub outputs before sharing in Slack or email.
- If unsure, check with Lin Zhao before exporting or publishing any data.

---

### 22. What should I know about reproducibility requirements? <a name="22"></a>

- **All canonical runs must be fully reproducible.**
  - Code, config, data, and environment must be documented.
  - Use seeds for any stochastic process.
  - Document all steps in per-run `README.md` or summary file.
  - If you can’t reproduce a result, flag it.

- **For experiments:**  
  - Still document code and config.
  - For any result you want to share beyond the team, ensure at least one other reviewer can reproduce it.

- **Reproducibility is a must for any number that goes into `dashboard.md` or a report.**

---

### 23. How do I request compute or storage resources? <a name="23"></a>

- For small experiments, use your assigned devbox or the team-shared GCP instance.
- For larger runs (full embedding generation, grid searches, etc.), email Arjun with:
  - Purpose of the run
  - Expected runtime and storage
  - Any external dependencies
- For long-running or production-impacting jobs, coordinate with the infra team (see `notes/infra_contacts.md`).

---

### 24. What are the canonical accuracy numbers for each method? <a name="24"></a>

As of the last canonical run (see `analytics/dashboard.md`), top-1 accuracy on the 3000-hotel evaluation subset is:

| Method         | Top-1 Accuracy |
|----------------|---------------|
| openai_3small  | 0.4687        |
| minilm         | 0.3937        |
| partial_ratio  | 0.4407        |
| wratio         | 0.4223        |
| openai_3large  | 0.6981*       |

\*openai_3large result is reported but **unverified** as of last audit.

If you see numbers in a doc that differ from the above, double-check the run, config, and data subset before relying on them.

---

### 25. How do I escalate if I believe a metric or output is incorrect? <a name="25"></a>

- **First, document the issue:**  
  - Note the file, method, metric, and what seems off.
  - Attempt to reproduce the result on the canonical subset.

- **Next, share findings in `#ml-mapping-core`** with a concise summary.

- **If the issue is urgent** (e.g., affects a ship decision or public report), escalate directly to Arjun and Priya via Slack DM.

- **For governance or compliance concerns, loop in Lin Zhao immediately.**

- **Add a "suspect" or "pending" note in any doc/dashboard** referencing the disputed number; don’t delete or overwrite until resolved.

- **Expect a rapid team response**—we take correctness seriously.

---

### 26. What are the preferred communication channels for questions? <a name="26"></a>

- **Quick questions / status:**  
  - `#ml-mapping-core` Slack channel.

- **Detailed design or decisions:**  
  - PR descriptions, or markdown notes in `notes/`.

- **Product-level questions:**  
  - DM or mention Hannah Kim or Martin Oliveira.

- **Governance/compliance:**  
  - DM Lin Zhao or use `#ml-governance`.

- **Escalations:**  
  - Slack DM to Arjun and Priya.

---

### 27. How do I onboard to the codebase efficiently? <a name="27"></a>

- **Start with `README.md` and `notes/onboarding.md`.**
- **Clone the repo and set up your environment** (`notes/env_setup.md`).
- **Browse canonical configs in `configs/` and outputs in `runs/`.**
- **Walk through an end-to-end eval run** as described in this FAQ.
- **Review historical dashboards and reports** (see `analytics/` and `reports/`).
- **Ask for a pairing session with Priya** if you get stuck.

---

### 28. What are the most common pitfalls for new contributors? <a name="28"></a>

- Overwriting canonical outputs with experimental results.
- Running the pipeline on a non-canonical data subset but labeling results as official.
- Failing to document changes to configs or embeddings.
- Not marking "suspect" or "unverified" numbers in dashboards.
- Sharing raw or sensitive data outside approved channels.
- Neglecting to review the latest `analytics/dashboard.md` before reporting a metric.
- Forgetting to ping the right reviewers for PRs.

---

### 29. Who do I talk to for product direction or requirements? <a name="29"></a>

- **Hannah Kim** (primary product owner).
- **Martin Oliveira** (secondary, covers integration and UX).
- For any questions about matching granularity, coverage targets, or downstream impact, reach out to Hannah first.

---

### 30. What does the team expect from documentation in PRs? <a name="30"></a>

- **Clear summary of what changed and why.**
- **Links to relevant runs, configs, and analytics outputs.**
- **Note if canonical results are affected.**
- **Describe any new dependencies or data requirements.**
- **If applicable, include "before/after" metrics or screenshots.**
- **Tag reviewers and stakeholders explicitly.**
- **If the PR closes an issue or addresses a decision doc, reference it.**

See `prs/` for good examples, and over-communicate rather than under—future you (and new joiners) will thank you.

---

**Still have questions?**  
Check `notes/`, ask in `#ml-mapping-core`, or ping Priya/Arjun directly. And remember: if something seems off, flag it! This team values clarity and correctness over speed every time. Welcome aboard.
