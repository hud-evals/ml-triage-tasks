"""Task collection for the ml-triage-tasks environment.

Each task module binds a v6 task template from `env.py` (currently
`diagnose_research_study`) with task-specific prompt / rubric /
axis_weights / hard_caps / bonus / case args. Adding a new task:

    cp -R _template tasks/<your_slug>
    # edit tasks/<your_slug>/task.py, drop case data under cases/<your_slug>/
    # then add an import line below.

Importing this package triggers each task module's top-level
task factory call, which is how the HUD platform discovers tasks
during `hud sync tasks <taskset>`.
"""

import tasks.city_mapping_audit.task  # noqa: F401
import tasks.mxbai_projection_dim_cliff.task  # noqa: F401
import tasks.mxbai_projection_layer_choice.task  # noqa: F401
import tasks.mxbai_reranker_teacher_diag.task  # noqa: F401
import tasks.nmoe_0006_study.task  # noqa: F401
import tasks.nmoe_0008_study.task  # noqa: F401
import tasks.nmoe_0011_study.task  # noqa: F401
import tasks.prime_rl_chunk_default_tradeoff.task  # noqa: F401
import tasks.wafer_cold_start.task  # noqa: F401
import tasks.wafer_kimi_delta_attention.task  # noqa: F401
import tasks.wafer_nvfp4_silu_audit.task  # noqa: F401
