"""Starter: structured artifact + deterministic-only grading.

Agent ships `output.json` (a list of `{id, label}` records). Reward
is macro-F1 against gold. No LLM judge, no anti-fake citation gate.

Gold is hardcoded in this module. `mount_case` copies the case's
children into /workspace verbatim, so any gold file living under
`cases/<slug>/` would leak to the agent. Other valid patterns: load
gold from a sibling directory outside `cases/` (e.g. a root-locked
`grading/` dir copied in separately by the Dockerfile), or have the
scenario delete the gold file from /workspace right after mounting.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from hud.graders import EvaluationResult

from env import env, mount_case

WORK = Path(os.environ.get("CI_WORK", "/workspace"))


# --- Per-task gold (edit for your case) ------------------------------------


# List of {id, label} ground-truth records. Replace with your real gold.
GOLD: list[dict[str, str]] = [
    # {"id": "rec_001", "label": "drop"},
    # {"id": "rec_002", "label": "add"},
]


# --- Scoring ---------------------------------------------------------------


def _score_against_gold(submitted: list[dict], gold: list[dict]) -> tuple[float, dict[str, Any]]:
    """Macro-F1 over labels. Submitted + gold are lists of {id, label}."""
    sub_by_id = {r.get("id"): str(r.get("label", "")) for r in submitted if isinstance(r, dict)}
    gold_by_id = {r["id"]: str(r["label"]) for r in gold if isinstance(r, dict) and "id" in r}

    labels = sorted({*sub_by_id.values(), *gold_by_id.values()})
    if not labels:
        return 0.0, {"error": "no labels in submission or gold"}

    per_label: dict[str, dict[str, float]] = {}
    f1s: list[float] = []
    for lbl in labels:
        tp = sum(1 for k, v in sub_by_id.items() if v == lbl and gold_by_id.get(k) == lbl)
        fp = sum(1 for k, v in sub_by_id.items() if v == lbl and gold_by_id.get(k) != lbl)
        fn = sum(1 for k, v in gold_by_id.items() if v == lbl and sub_by_id.get(k) != lbl)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per_label[lbl] = {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4)}
        f1s.append(f1)

    macro = sum(f1s) / len(f1s)
    return round(macro, 4), {
        "macro_f1": round(macro, 4),
        "n_submitted": len(sub_by_id),
        "n_gold": len(gold_by_id),
        "per_label": per_label,
    }


# --- Scenario --------------------------------------------------------------


PROMPT = """\
TODO: write the agent-facing prompt. Tell the agent to produce
`output.json` in the working directory as a JSON list of {id, label}.
"""


@env.template(id="my_structured_task")
async def my_structured_task(prompt: str, case: str):
    """Mount the case, yield prompt, grade against hardcoded gold on stop."""
    mount_case(case)
    yield prompt

    sub_path = WORK / "output.json"
    if not sub_path.is_file():
        yield EvaluationResult(reward=0.0, content="output.json missing",
                               info={"reason": "artifact_missing"})
        return
    try:
        submitted = json.loads(sub_path.read_text(encoding="utf-8"))
    except Exception as e:
        yield EvaluationResult(reward=0.0, content=f"output.json unreadable: {e}",
                               info={"reason": "artifact_unreadable"})
        return
    if not isinstance(submitted, list):
        yield EvaluationResult(reward=0.0, content="output.json must be a JSON list",
                               info={"reason": "artifact_shape"})
        return

    score, info = _score_against_gold(submitted, GOLD)
    print(f"[grade] macro_f1={score:.3f} {info}", file=sys.stderr)
    yield EvaluationResult(
        reward=score,
        content=f"macro_f1={score:.3f} (n_sub={info['n_submitted']}, n_gold={info['n_gold']})",
        info=info,
    )


# --- Bind the scenario into a task instance --------------------------------


task = my_structured_task(prompt=PROMPT, case="TODO_case_slug")
task.slug = "TODO_task_slug"
task.columns = {"category": "TODO"}
