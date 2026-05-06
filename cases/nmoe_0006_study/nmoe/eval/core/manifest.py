from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


TaskType = Literal["multiple_choice", "schema", "language_modeling"]


@dataclass(frozen=True)
class CoreTask:
    label: str
    dataset_uri: str
    task_type: TaskType
    fewshot: int
    continuation_delimiter: str
    has_categories: bool


def load_core_tasks(tasks_file: Path) -> list[CoreTask]:
    try:
        import tomllib
    except Exception as e:  # pragma: no cover
        raise RuntimeError("tomllib unavailable; require Python 3.11+") from e

    if not tasks_file.exists():
        raise FileNotFoundError(str(tasks_file))

    with tasks_file.open("rb") as f:
        obj = tomllib.load(f)

    tasks_raw = obj.get("task")
    if not isinstance(tasks_raw, list) or not tasks_raw:
        raise ValueError(f"{tasks_file}: expected non-empty [[task]] list")

    out: list[CoreTask] = []
    seen: set[str] = set()
    for i, t in enumerate(tasks_raw):
        if not isinstance(t, dict):
            raise ValueError(f"{tasks_file}: task[{i}] must be a table")
        label = str(t.get("label") or "").strip()
        if not label:
            raise ValueError(f"{tasks_file}: task[{i}].label is required")
        if label in seen:
            raise ValueError(f"{tasks_file}: duplicate task label '{label}'")
        seen.add(label)

        dataset_uri = str(t.get("dataset_uri") or "").strip()
        if not dataset_uri:
            raise ValueError(f"{tasks_file}: task[{i}].dataset_uri is required")

        task_type = str(t.get("task_type") or "").strip()
        if task_type not in ("multiple_choice", "schema", "language_modeling"):
            raise ValueError(
                f"{tasks_file}: task[{i}].task_type must be one of "
                f"multiple_choice|schema|language_modeling (got {task_type!r})"
            )

        fewshot = int(t.get("fewshot") or 0)
        if fewshot < 0:
            raise ValueError(f"{tasks_file}: task[{i}].fewshot must be >= 0 (got {fewshot})")

        continuation_delimiter = t.get("continuation_delimiter")
        if continuation_delimiter is None:
            continuation_delimiter = " "
        continuation_delimiter = str(continuation_delimiter)

        has_categories = bool(t.get("has_categories", False))

        out.append(
            CoreTask(
                label=label,
                dataset_uri=dataset_uri,
                task_type=task_type,  # type: ignore[arg-type]
                fewshot=fewshot,
                continuation_delimiter=continuation_delimiter,
                has_categories=has_categories,
            )
        )

    return out

