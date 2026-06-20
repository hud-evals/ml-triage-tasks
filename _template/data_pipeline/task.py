"""Starter: structured artifact + deterministic verifier + LLM rubric.

Agent ships `output.parquet` + `report.md`. Reward is
`0.7 * det + 0.3 * llm`, where det checks schema, row-count window,
and a label-set fingerprint, and llm runs the scaled-rubric judge
over the prose.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from hud.graders import EvaluationResult, SubScore

from env import env, load_report, mount_case, run_scaled_judge

WORK = Path(os.environ.get("CI_WORK", "/workspace"))


# --- Per-task gold (edit for your case) -----------------------------------

EXPECTED_SCHEMA = {"id": "int64", "label": "string", "score": "double"}
EXPECTED_ROW_RANGE = (1000, 5000)            # inclusive
EXPECTED_LABEL_FINGERPRINT = "TODO_md5_hex"  # md5 of sorted unique labels


def _verify_parquet(path: Path) -> tuple[float, dict[str, Any]]:
    """Return (score in [0, 1], info dict) for a single parquet artifact.

    Three sub-checks: schema match, row-count in window, label-set
    fingerprint. Each contributes 1/3 of the deterministic score.
    """
    info: dict[str, Any] = {"path": str(path)}
    if not path.is_file():
        info["error"] = "artifact missing"
        return 0.0, info

    try:
        import pyarrow.parquet as pq  # type: ignore
    except ImportError as e:
        info["error"] = f"pyarrow not installed: {e}"
        return 0.0, info

    try:
        table = pq.read_table(path)
    except Exception as e:
        info["error"] = f"unreadable parquet: {e}"
        return 0.0, info

    # 1. schema match
    actual_schema = {f.name: str(f.type) for f in table.schema}
    schema_ok = all(
        actual_schema.get(col) == typ for col, typ in EXPECTED_SCHEMA.items()
    )
    info["schema_ok"] = schema_ok
    info["actual_schema"] = actual_schema

    # 2. row count window
    n_rows = table.num_rows
    info["n_rows"] = n_rows
    rows_ok = EXPECTED_ROW_RANGE[0] <= n_rows <= EXPECTED_ROW_RANGE[1]
    info["rows_ok"] = rows_ok

    # 3. label-set fingerprint (md5 of sorted unique labels)
    import hashlib
    if "label" in actual_schema:
        labels = sorted(set(str(v) for v in table.column("label").to_pylist()))
        digest = hashlib.md5("\n".join(labels).encode()).hexdigest()
        info["label_fingerprint"] = digest
        fingerprint_ok = digest == EXPECTED_LABEL_FINGERPRINT
    else:
        fingerprint_ok = False
        info["label_fingerprint"] = None
    info["fingerprint_ok"] = fingerprint_ok

    score = sum([schema_ok, rows_ok, fingerprint_ok]) / 3.0
    return score, info


# --- Scenario -------------------------------------------------------------


PROMPT = """\
TODO: write the agent-facing task prompt. Tell the agent to produce
`output.parquet` AND `report.md` in the working directory.
"""


# Rubric for the prose `report.md` (the LLM-judged half).
RUBRIC = {
    # "axis_name": "Score 4 if ... Score 0 if ...",
}


AXIS_WEIGHTS: dict[str, float] = {
    # "axis_name": 1.0,
}


DET_WEIGHT = 0.7
LLM_WEIGHT = 0.3


@env.template(id="my_pipeline_task")
async def my_pipeline_task(prompt: str, case: str):
    """Mount the case, yield prompt, grade artifact + report on stop."""
    mount_case(case)
    yield prompt

    # Deterministic side: the agent's parquet artifact.
    det_score, det_info = _verify_parquet(WORK / "output.parquet")
    print(f"[grade] det score={det_score:.3f} {det_info}", file=sys.stderr)

    # LLM rubric side: the agent's prose report.
    load_report("report.md")
    judge = run_scaled_judge(RUBRIC, axis_scale=4, hard_caps=None, bonus=None)
    if "_error" in judge:
        llm_score = 0.0
        llm_info: dict[str, Any] = {"error": judge["_error"]}
    else:
        axes = judge.get("axes") or {}
        total_w = sum(AXIS_WEIGHTS.values()) or 1.0
        s = 0.0
        per_axis: dict[str, Any] = {}
        for axis, w in AXIS_WEIGHTS.items():
            entry = axes.get(axis) or {}
            try:
                v = int(entry.get("score", 0))
            except (TypeError, ValueError):
                v = 0
            v = max(0, min(4, v))
            wn = w / total_w
            s += wn * (v / 4.0)
            per_axis[axis] = {"score": v, "why": str(entry.get("why", "")), "weight": wn}
        llm_score = max(0.0, min(1.0, s))
        llm_info = {"per_axis": per_axis}

    reward = DET_WEIGHT * det_score + LLM_WEIGHT * llm_score
    yield EvaluationResult(
        reward=reward,
        content=f"reward={reward:.3f} | det={det_score:.3f} llm={llm_score:.3f}",
        info={
            "weights": {"det": DET_WEIGHT, "llm": LLM_WEIGHT},
            "deterministic": det_info,
            "llm": llm_info,
        },
        subscores=[
            SubScore(name="deterministic", weight=DET_WEIGHT, value=det_score),
            SubScore(name="llm_rubric", weight=LLM_WEIGHT, value=llm_score),
        ],
    )


# --- Bind the scenario into a task instance --------------------------------


task = my_pipeline_task(
    prompt=PROMPT,
    case="TODO_case_slug",
)
task.slug = "TODO_task_slug"
task.columns = {"category": "TODO"}
