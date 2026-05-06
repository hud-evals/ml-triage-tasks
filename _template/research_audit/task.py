"""Starter task. Copy this directory to tasks/<your_slug>/, fill in the
TODOs, drop case data under cases/<your_slug>/, then add an import line
to tasks/__init__.py.
"""

from env import diagnose_research_study


PROMPT = """\
TODO: write the agent-facing task prompt.
"""


RUBRIC: dict[str, str] = {
    # "axis_name": "Score 4 if ... Score 2 if ... Score 0 if ...",
}


AXIS_WEIGHTS: dict[str, float] = {
    # "axis_name": 1.0,
}


HARD_CAPS: list[dict] = [
    # {"name": "...", "description": "trigger if ...", "cap": 0.5},
]


BONUS: dict | None = None
# BONUS = {"description": "trigger if ...", "value": 0.15}


ANTI_FAKE = {
    "min_verified": 5,
    "max_fabricated_ratio": 0.20,
}


task = diagnose_research_study.task(
    prompt=PROMPT,
    case="TODO_case_slug",
    rubric=RUBRIC,
    axis_weights=AXIS_WEIGHTS,
    axis_scale=4,
    hard_caps=HARD_CAPS,
    bonus=BONUS,
    anti_fake=ANTI_FAKE,
    report_filename="REPORT.md",
)
task.slug = "TODO_task_slug"
task.metadata = {"category": "TODO"}
