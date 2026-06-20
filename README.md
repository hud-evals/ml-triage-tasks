# ml-triage-tasks

HUD environment template for long-horizon agent tasks. The env is a
HUD v6 template host: each template defines its own contract (what the agent
produces and how that artifact is graded), and authors write whatever
template their task needs. The shipped template `diagnose_research_study`
is one worked example for prose-deliverable research audits.

The taskset ships multiple audit tasks, all currently wired onto
`diagnose_research_study`:

| Task | What the agent does |
|---|---|
| `prime_rl_chunk_default_tradeoff` | Audit a flipped default in prime-rl's fused-LM-head chunk-size knob; recommend whether to pin it for an upcoming run. |
| `nmoe_0006_study` | Reconstruct a sparsity ablation from raw run receipts; separate supported claims from falsified ones. |
| `city_mapping_audit` | Second-opinion an ML teammate's hotel-to-city matching ship recommendation under planted adversarial artifacts. |

## Setup

```bash
git lfs install
git lfs pull           # streams cases/**/*.parquet, *.npy, prs/*.json, ci-logs/**, etc.
uv sync                # installs hud-python, openai
```

## Run a task locally

```bash
export HUD_API_KEY=...
uv run python tools/local_test.py --task prime_rl_chunk_default_tradeoff --model grok-4.20
```

`--list` enumerates available tasks. The container image is built once
via `hud build .` (or pulled if you've already deployed).

## Run a task N-up

```bash
uv run python tools/run_many.py --task prime_rl_chunk_default_tradeoff --n 5
```

Reports per-sample rewards plus mean/median/min/max across the group.

## Build + deploy + sync

```bash
hud build .                                      # local image
hud deploy .                                     # push to platform
hud sync tasks <taskset-name>                    # sync local tasks/ to a taskset
```

`hud sync tasks` discovers tasks through the root `tasks.py` entrypoint,
which imports each concrete `tasks/<slug>/task.py` row. Add new tasks to
both `tasks/__init__.py` and `tasks.py`, and set `task.slug`.

## Add a new task

Pick a starter shape from `_template/<shape>/`:

| Starter | Contract |
|---|---|
| `_template/research_audit/` | Prose `REPORT.md` + LLM rubric (uses `diagnose_research_study`). |
| `_template/data_pipeline/` | Structured artifact (`output.parquet`) + prose `report.md`, deterministic verifier weighted with LLM rubric. |
| `_template/structured_output/` | Single structured file (`output.json`), deterministic-only grading (macro-F1 against gold). |

```bash
cp -R _template/<shape> tasks/<your_slug>
# edit tasks/<your_slug>/task.py
# drop case data under cases/<your_slug>/   (LFS handles binaries)
```

Then add the import line in `tasks/__init__.py`:

```python
import tasks.<your_slug>  # noqa: F401
```

The starters are samples, not a closed taxonomy — write your own
template from scratch when none fits. See [`_template/README.md`](_template/README.md)
for the full toolkit (`mount_case`, `anti_fake_gate`, `run_scaled_judge`,
…) and the bare template contract.

## Templates

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

There is no "the grader". The env hosts as many templates as the work
needs — define them inline in `tasks/<slug>/task.py` or, for shared
shapes, in `env.py` next to `diagnose_research_study`.

## How `diagnose_research_study` works

This is the worked-example template shipped in `env.py` and used by the
live tasks. Per-task arguments live in `tasks/<slug>/task.py`:

| Arg | What it is |
|---|---|
| `prompt` | Free-form text the agent reads at task start. |
| `case` | Subdir name under `cases/`. Hard-copied into `/workspace` at template start, chowned to the env process uid for the bwrap workspace. |
| `rubric` | Dict of `axis_name -> ground-truth description`. The LLM judge scores each axis 0..`axis_scale`. |
| `axis_weights` | Per-axis weight. Reward = `sum(weight * score / axis_scale)` over axes, normalised. |
| `hard_caps` | List of `{name, description, cap}`. If the judge flags a cap as triggered, reward is clamped to `min(reward, cap)`. |
| `bonus` | Optional `{description, value}`. Added to reward if the judge marks it triggered. |
| `anti_fake` | `{min_verified, max_fabricated_ratio}`. The grader extracts identifier-style citations (PR numbers, SHAs, GHA run-ids, file paths) from the report and verifies them against the case bundle. Failing the gate floors reward to 0. |
| `report_filename` | What file the agent must write under `/workspace/`. Typically `REPORT.md`. |

Set `task.slug` (stable identifier) and `task.columns` (free-form
filterable fields) after constructing the task.

## Case data

Vendored case bundles live under `cases/<slug>/`. Large binaries
(parquet, .npy, `.db`, scraped `prs/*.json`, GHA `ci-logs/**`, vendored
`.git/objects/pack/*`) stream via Git LFS — see `.gitattributes`. The
Dockerfile copies the populated `cases/` tree into `/opt/ci_cases` at
build time; the env mounts the selected case into `/workspace` at
template start, so the agent never sees the case slug or the
`/opt/ci_cases` path.

## Layout

```
ml-triage-tasks/
├── env.py                    # toolkit + diagnose_research_study worked example
├── Dockerfile.hud
├── pyproject.toml / uv.lock
├── .gitattributes            # LFS patterns for cases/
├── tasks.py                  # explicit v6 taskset entrypoint
├── _template/                # template starters — see _template/README.md
│   ├── research_audit/task.py
│   ├── data_pipeline/task.py
│   └── structured_output/task.py
├── tasks/
│   ├── __init__.py
│   ├── prime_rl_chunk_default_tradeoff/task.py
│   ├── nmoe_0006_study/task.py
│   └── city_mapping_audit/task.py
├── cases/
│   ├── prime_rl_chunk_default_tradeoff/
│   ├── nmoe_0006_study/
│   └── city_mapping_audit/
└── tools/
    ├── local_test.py         # run one task locally
    ├── run_many.py           # run one task N-up
    └── parse_traces.py       # slice job telemetry into per-rollout summary
```
