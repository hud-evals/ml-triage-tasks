# Scenario starters

The env is a HUD v6 template host. Each template defines its own contract:
what the agent produces and how that artifact is graded. There is no
"the grader" — there can be as many templates as the work needs.

This directory has three starters covering different points in the
contract space. Pick the one closest to your task, copy it to
`tasks/<your_slug>/`, edit the TODOs, drop case data under
`cases/<your_slug>/`, and add an import line to `tasks/__init__.py`.

The starters are samples, not a closed taxonomy. If none of them fit,
write your own template from scratch — see the bare contract below.

## Available shapes

### `research_audit/` — prose deliverable, LLM judge

Reuses the shipped `diagnose_research_study` template.

- Agent's deliverable is a single prose file (typically `REPORT.md`).
- Grading: anti-fake citation gate → multi-axis LLM rubric scored
  0..N → optional hard caps and bonus.
- Best when the work is "audit / reconstruct / triangulate from a
  frozen bundle" and the deliverable is reasoning prose.
- This is the shape every live task under `tasks/` currently uses.

### `data_pipeline/` — structured artifact + det verifier + LLM rubric

Defines its own template inline.

- Agent's deliverables are a structured artifact (`output.parquet`)
  and a prose `report.md`.
- Grading: deterministic verifier on the artifact (schema match,
  row-count window, column fingerprint) + LLM rubric on the report,
  weighted (default 0.7 / 0.3).
- Best when the agent is building a small pipeline / tool whose
  output is mechanically checkable but the reasoning behind it
  warrants a rubric.

### `structured_output/` — structured artifact, deterministic only

Defines its own template inline.

- Agent's deliverable is one structured file (`output.json`).
- Grading: macro-F1 (or whatever scorer fits) against gold. No LLM.
- Best when grading is unambiguous (label match, set overlap, exact
  match) and an LLM judge would add noise rather than signal.
- Gold is hardcoded in the task module to avoid the case-mount
  visibility footgun (see the docstring in
  `structured_output/task.py`).

## Toolkit (`env.py` exports)

Custom templates import these from `env`:

| Helper | Purpose |
|---|---|
| `env` | The `hud.Environment` instance. Decorate templates with `@env.template(id=...)`. |
| `mount_case(case)` | Hard-copy `cases/<case>/` into `/workspace`, chown to the agent uid, return the original case root path. |
| `load_report(filename)` | Read a file from `/workspace`, populate `_SUBMISSION` for citation extraction. |
| `extract_citations(text)` | Pull identifier-style citations (PR numbers, SHAs, run-ids, file paths) out of text. |
| `bundle_index(case_root)` | Index a case bundle into a lookup of real shas / prs / run-ids / paths. |
| `verify_citation(cit, idx)` | Single-citation check against an index. |
| `anti_fake_gate(...)` | Combined gate — runs extract + verify, returns (passed, results). |
| `run_scaled_judge(report, rubric, axis_scale, hard_caps, bonus)` | Calls the LLM judge with the standard scaled-rubric protocol. |

Compose any subset of these. None are required.

## Bare template contract

A v6 task template is an async generator decorated with `@env.template(...)`
that yields a prompt, lets the agent work, then yields an
`EvaluationResult`:

```python
from hud.graders import EvaluationResult
from env import env, mount_case

@env.template(id="my_thing")
async def my_thing(prompt: str, case: str):
    mount_case(case)
    yield prompt
    # ... after the agent stops, inspect /workspace, score, and ...
    yield EvaluationResult(reward=0.42, content="...", info={...})
```

The agent's `bash` cwd is `/workspace`. Anything you want graded must
either land there (and you read it after the prompt yield) or land in
a path the template opens via `case_root`.

## Adding a new template from scratch

1. `cp -R _template/<closest_shape> tasks/<your_slug>`
2. Edit `tasks/<your_slug>/task.py`:
   * Rename the `@env.template(id=...)` to your slug.
   * Edit `PROMPT`, the rubric / gold, and the grading body.
   * Set `task.slug` and `task.columns`.
3. Drop case data under `cases/<your_slug>/`. Large binaries travel
   via Git LFS — see the env-root `.gitattributes`.
4. Add the task import to `tasks/__init__.py` and root `tasks.py`.
5. `uv run python tools/local_test.py --task <your_slug>` to smoke-test.
