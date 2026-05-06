"""
Evaluation and verification for synthetic tasks.

Each verifier takes model output and returns (correct: bool, details: dict).
"""
from nmoe.research.physics.eval.verifiers import verify_depo, verify_brevo, verify_mano, verify_sample, evaluate_batch, VerifyResult

__all__ = ["verify_depo", "verify_brevo", "verify_mano", "verify_sample", "evaluate_batch", "VerifyResult"]
