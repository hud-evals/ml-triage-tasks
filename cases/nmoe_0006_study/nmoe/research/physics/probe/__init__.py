"""
Probing infrastructure for model internals.

Two-tier system:

## Native probes (primary signal, tied to paper claims)
- atlas: weights-only geometry (gate weights, expert similarity)
- trajectory: router geometry on data (margin, boundary-span, flip rate)
- mhc: multi-head chart stream analysis

These native probes emit `ProbeEvent` records (JSONL) via `ProbeWriter`.

## Generic probes (opt-in microscope; legacy schema)
- capture: generic hook-based capture (writes one nested JSON dict per step)
- analysis: utilities for the legacy capture log format

We keep the generic probes for exploratory debugging, but they are not the
paper's primary measurement path.
"""
from nmoe.research.physics.probe.schema import (
    ProbeEvent,
    ProbeWriter,
    TensorStats,
    QUANTILES,
    load_probe_events,
    filter_events,
)

from nmoe.research.physics.probe.atlas import AtlasProbe, AtlasConfig, compare_atlases
from nmoe.research.physics.probe.trajectory import (
    TrajectoryProbe, TrajectoryConfig, TrajectoryProbeWithHooks,
    compute_rgb_components,
)
from nmoe.research.physics.probe.mhc import MHCProbe, MHCConfig, compute_mixing_stability

# Mental model validation probes
from nmoe.research.physics.probe.overlap import (
    OverlapProbe, OverlapConfig, measure_overlap_compatibility,
)
from nmoe.research.physics.probe.gain import (
    GainProbe, GainConfig, measure_gain_per_layer, detect_late_band,
)

# Keep generic probes for future use
from nmoe.research.physics.probe.capture import Probe, ProbeConfig
from nmoe.research.physics.probe.analysis import load_probe_log, plot_metric_over_training

__all__ = [
    # Schema
    "ProbeEvent",
    "ProbeWriter",
    "TensorStats",
    "QUANTILES",
    "load_probe_events",
    "filter_events",
    # Native probes
    "AtlasProbe",
    "AtlasConfig",
    "compare_atlases",
    "TrajectoryProbe",
    "TrajectoryConfig",
    "TrajectoryProbeWithHooks",
    "compute_rgb_components",
    "MHCProbe",
    "MHCConfig",
    "compute_mixing_stability",
    # Mental model validation probes
    "OverlapProbe",
    "OverlapConfig",
    "measure_overlap_compatibility",
    "GainProbe",
    "GainConfig",
    "measure_gain_per_layer",
    "detect_late_band",
    # Generic probes (kept for flexibility)
    "Probe",
    "ProbeConfig",
    "load_probe_log",
    "plot_metric_over_training",
]
