"""Physics of Language Models: synthetic playground for mechanistic research.

This is research-only code (not part of the production training surface).
"""

from nmoe.research.physics.data.generators import depo, brevo, mano, SyntheticMix, Sample
from nmoe.research.physics.eval.verifiers import verify_depo, verify_brevo, verify_mano, verify_sample, evaluate_batch
from nmoe.research.physics.probe.capture import Probe, ProbeConfig

__all__ = [
  "depo", "brevo", "mano", "SyntheticMix", "Sample",
  "verify_depo", "verify_brevo", "verify_mano", "verify_sample", "evaluate_batch",
  "Probe", "ProbeConfig",
]
