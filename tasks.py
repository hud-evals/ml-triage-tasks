"""Concrete HUD v6 task rows for local eval, task listing, and sync."""

from tasks.city_mapping_audit.task import task as _city_mapping_audit
from tasks.mxbai_projection_dim_cliff.task import task as _mxbai_projection_dim_cliff
from tasks.mxbai_projection_layer_choice.task import task as _mxbai_projection_layer_choice
from tasks.mxbai_reranker_teacher_diag.task import task as _mxbai_reranker_teacher_diag
from tasks.nmoe_0006_study.task import task as _nmoe_0006_study
from tasks.nmoe_0008_study.task import task as _nmoe_0008_study
from tasks.nmoe_0011_study.task import task as _nmoe_0011_study
from tasks.prime_rl_chunk_default_tradeoff.task import task as _prime_rl_chunk_default_tradeoff
from tasks.wafer_cold_start.task import task as _wafer_cold_start
from tasks.wafer_kimi_delta_attention.task import task as _wafer_kimi_delta_attention
from tasks.wafer_nvfp4_silu_audit.task import task as _wafer_nvfp4_silu_audit

TASKS = [
    _prime_rl_chunk_default_tradeoff,
    _nmoe_0006_study,
    _city_mapping_audit,
    _mxbai_reranker_teacher_diag,
    _mxbai_projection_dim_cliff,
    _mxbai_projection_layer_choice,
    _nmoe_0008_study,
    _nmoe_0011_study,
    _wafer_cold_start,
    _wafer_kimi_delta_attention,
    _wafer_nvfp4_silu_audit,
]
