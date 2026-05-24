"""Task collection for the ml-triage-tasks environment.

Each task module binds a scenario from `env.py` (currently
`diagnose_research_study`) with task-specific prompt / rubric /
axis_weights / hard_caps / bonus / case args. Adding a new task:

    cp -R _template tasks/<your_slug>
    # edit tasks/<your_slug>/task.py, drop case data under cases/<your_slug>/
    # then add an import line below.

Importing this package triggers each task module's top-level
`scenario.task(...)` call, which is how the HUD platform discovers tasks
during `hud sync tasks <taskset>`.
"""

import tasks.prime_rl_chunk_default_tradeoff  # noqa: F401
import tasks.nmoe_0006_study  # noqa: F401
import tasks.city_mapping_audit  # noqa: F401
import tasks.mxbai_reranker_teacher_diag  # noqa: F401
import tasks.mxbai_projection_dim_cliff  # noqa: F401
import tasks.mxbai_projection_layer_choice  # noqa: F401
