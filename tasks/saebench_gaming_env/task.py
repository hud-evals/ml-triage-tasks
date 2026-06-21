from __future__ import annotations

from hud.graders import EvaluationResult
from env import env, mount_case

PROMPT = """\
You are in a toy SAEBench gaming environment.

For now, simply write any file you want in the workspace. This is a placeholder task.
"""

@env.template(id="saebench_gaming_env")
async def saebench_gaming_env(prompt: str, case: str):
    mount_case(case)
    yield prompt

    yield EvaluationResult(
        reward=0.5,
        content="dummy saebench_gaming_env reward=0.5",
        info={"status": "dummy_task_working"},
    )

task = saebench_gaming_env(prompt=PROMPT, case="saebench_gaming_env")
task.slug = "saebench_gaming_env"
task.columns = {"category": "mech-interp", "project": "saebench-gaming"}
