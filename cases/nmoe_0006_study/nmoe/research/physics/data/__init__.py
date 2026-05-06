"""
Synthetic data generators.

Each generator produces (tokens, labels, metadata) where:
- tokens: input sequence (list of ints)
- labels: which tokens to compute loss on (0=ignore, 1=predict)
- metadata: task-specific info for analysis (difficulty, structure, etc.)
"""
from nmoe.research.physics.data.generators import depo, brevo, mano, SyntheticMix, Sample

__all__ = ["depo", "brevo", "mano", "SyntheticMix", "Sample"]
