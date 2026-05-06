"""
mHC (Multi-Head Charts) probes: Stream-level analysis.

For mHC(N=4) architecture with factored residual streams:
- Content stream: carries base model identity
- Routing-control stream: boundary-span motion (hinge acts here)
- Tail absorber stream: catches high-ρ events
- Scratch/adaptation stream: absorbs rapid non-stationary updates

Probes measure:
- Stream-wise ||x_s|| quantiles and cos(x_s, x_total)
- Mixing matrix diagnostics (doubly-stochastic error, condition)
- Attribution: routing-control variance explained per stream
- Tail regime: ρ_p99 explained by which stream
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

from nmoe.research.physics.probe.schema import (
    ProbeEvent, ProbeWriter, TensorStats, QUANTILES,
)


# Stream semantic labels
STREAM_CONTENT = 0
STREAM_ROUTING = 1
STREAM_TAIL = 2
STREAM_SCRATCH = 3

STREAM_NAMES = {
    STREAM_CONTENT: "content",
    STREAM_ROUTING: "routing",
    STREAM_TAIL: "tail",
    STREAM_SCRATCH: "scratch",
}


@dataclass
class MHCConfig:
    """Configuration for mHC probing."""
    log_path: Path | str
    log_every: int = 100
    n_streams: int = 4
    layers: list[int] | None = None
    compute_mixing: bool = True
    compute_attribution: bool = True
    quantiles: tuple[float, ...] = QUANTILES


class MHCProbe:
    """
    Probe mHC stream dynamics.

    Usage:
        probe = MHCProbe(MHCConfig(log_path="mhc.jsonl"))

        # At each step, provide stream-wise tensors:
        probe.log_streams(
            step=step,
            layer_idx=layer_idx,
            streams=[x_content, x_routing, x_tail, x_scratch],  # [N, D] each
            x_total=x_total,  # [N, D] combined
            mixing_matrix=M,  # [n_streams, n_streams] if available
            gate_weight=W,
            gate_bias=bias,
            gamma=gamma,
        )
    """

    def __init__(self, config: MHCConfig):
        self.config = config
        self.writer = ProbeWriter(config.log_path)

    def should_log(self, step: int) -> bool:
        return step % self.config.log_every == 0

    def close(self):
        self.writer.close()

    def log_streams(
        self,
        step: int,
        layer_idx: int,
        streams: list[torch.Tensor],  # List of [N, D] tensors, one per stream
        x_total: torch.Tensor,  # [N, D] combined residual
        mixing_matrix: torch.Tensor | None = None,  # [n_streams, n_streams]
        gate_weight: torch.Tensor | None = None,  # For attribution
        gate_bias: torch.Tensor | None = None,
        gamma: torch.Tensor | None = None,
        eps: float = 1e-6,
    ):
        """Log mHC stream-level geometry."""
        n_streams = len(streams)
        N, D = x_total.shape

        # === Per-stream norms ===
        stream_norms = []
        for s, x_s in enumerate(streams):
            norm_s = x_s.norm(dim=-1)  # [N]
            stream_norms.append(norm_s)

            stream_name = STREAM_NAMES.get(s, f"stream_{s}")
            self.writer.write(ProbeEvent(
                step=step,
                probe="mhc",
                layer=layer_idx,
                module=stream_name,
                tensor="norm",
                stats=TensorStats.from_tensor(norm_s, self.config.quantiles),
            ))

        # === Per-stream cosine with total ===
        x_total_norm = x_total.norm(dim=-1, keepdim=True) + 1e-8
        for s, x_s in enumerate(streams):
            x_s_norm = x_s.norm(dim=-1, keepdim=True) + 1e-8
            cos_s = (x_s * x_total).sum(dim=-1) / (x_s_norm.squeeze() * x_total_norm.squeeze())

            stream_name = STREAM_NAMES.get(s, f"stream_{s}")
            self.writer.write(ProbeEvent(
                step=step,
                probe="mhc",
                layer=layer_idx,
                module=stream_name,
                tensor="cos_total",
                stats=TensorStats.from_tensor(cos_s, self.config.quantiles),
            ))

        # === Stream energy fractions ===
        total_energy = x_total.pow(2).sum(dim=-1)  # [N]
        energy_fracs = []
        for s, x_s in enumerate(streams):
            energy_s = x_s.pow(2).sum(dim=-1)  # [N]
            frac_s = energy_s / (total_energy + 1e-8)
            energy_fracs.append(frac_s.mean().item())

        self.writer.write(ProbeEvent(
            step=step,
            probe="mhc",
            layer=layer_idx,
            module="global",
            tensor="energy_fractions",
            stats={f"stream_{s}": frac for s, frac in enumerate(energy_fracs)},
        ))

        # === z_norm CV per stream (coefficient of variation) ===
        for s, x_s in enumerate(streams):
            norm_s = stream_norms[s]
            cv = norm_s.std() / (norm_s.mean() + 1e-8)

            stream_name = STREAM_NAMES.get(s, f"stream_{s}")
            self.writer.write(ProbeEvent(
                step=step,
                probe="mhc",
                layer=layer_idx,
                module=stream_name,
                tensor="norm_cv",
                stats={"cv": cv.item(), "mean": norm_s.mean().item(), "std": norm_s.std().item()},
            ))

        # === Mixing matrix diagnostics ===
        if self.config.compute_mixing and mixing_matrix is not None:
            M = mixing_matrix.float()  # [n_streams, n_streams]

            # Doubly-stochastic error
            row_sums = M.sum(dim=1)
            col_sums = M.sum(dim=0)
            ds_error_row = (row_sums - 1).abs().mean().item()
            ds_error_col = (col_sums - 1).abs().mean().item()

            # Singular values and condition
            try:
                _, S, _ = torch.linalg.svd(M)
                s_max = S[0].item()
                s_min = S[-1].item()
                condition = s_max / (s_min + 1e-8)
            except Exception:
                s_max, s_min, condition = 0, 0, 0

            self.writer.write(ProbeEvent(
                step=step,
                probe="mhc",
                layer=layer_idx,
                module="mixing",
                tensor="matrix_diagnostics",
                stats={
                    "ds_error_row": ds_error_row,
                    "ds_error_col": ds_error_col,
                    "row_sums": row_sums.tolist(),
                    "col_sums": col_sums.tolist(),
                    "s_max": s_max,
                    "s_min": s_min,
                    "condition": condition,
                },
            ))

        # === Attribution: routing-control variance explained per stream ===
        if self.config.compute_attribution and gate_weight is not None:
            self._log_attribution(
                step, layer_idx, streams, x_total,
                gate_weight, gate_bias, gamma, eps
            )

    def _log_attribution(
        self,
        step: int,
        layer_idx: int,
        streams: list[torch.Tensor],
        x_total: torch.Tensor,
        gate_weight: torch.Tensor,
        gate_bias: torch.Tensor | None,
        gamma: torch.Tensor | None,
        eps: float,
    ):
        """
        Compute routing-control attribution per stream.

        Key question: which stream explains boundary-normal motion?
        """
        N, D = x_total.shape
        n_experts = gate_weight.shape[0]

        if gamma is None:
            gamma = torch.ones(D, device=x_total.device)

        # Compute gate input from total
        y_total = self._rmsnorm(x_total, gamma, eps)

        # Get top-1/top-2 for boundary normal
        if gate_bias is None:
            gate_bias = torch.zeros(n_experts, device=x_total.device)

        scores = self._gate_scores(y_total, gate_weight, gate_bias)
        topk = scores.topk(k=min(2, n_experts), dim=-1)
        top1 = topk.indices[:, 0]
        top2 = topk.indices[:, 1]

        # Boundary normal: n = w_{e1} - w_{e2}
        W = gate_weight.float()
        n = W[top1] - W[top2]  # [N, D]
        n_norm = n.norm(dim=-1, keepdim=True) + 1e-8

        # For each stream, compute how much of the boundary-normal projection it carries
        # This is: (n · y_s) / ||n|| where y_s = contribution of stream s to y_total
        attributions = []
        for s, x_s in enumerate(streams):
            # Stream contribution to gate input (approximate: assume linear contribution)
            # y_s ≈ γ * x_s / RMS(x_total)  (first-order approximation)
            rms_total = torch.sqrt(x_total.pow(2).mean(dim=-1, keepdim=True) + eps)
            y_s = gamma * x_s / rms_total

            # Boundary-normal projection
            n_dot_ys = (n * y_s).sum(dim=-1)  # [N]
            proj_magnitude = n_dot_ys.abs() / n_norm.squeeze()

            attributions.append({
                "mean": proj_magnitude.mean().item(),
                "std": proj_magnitude.std().item(),
                "p99": proj_magnitude.quantile(0.99).item() if N > 10 else proj_magnitude.max().item(),
            })

        # Total boundary-normal component
        n_dot_y = (n * y_total).sum(dim=-1)
        total_bn = n_dot_y.abs() / n_norm.squeeze()

        # Variance explained: how much of total_bn is explained by each stream
        var_explained = []
        for s, attr in enumerate(attributions):
            # Simple proxy: ratio of stream's mean to total mean
            ve = attr["mean"] / (total_bn.mean().item() + 1e-8)
            var_explained.append(ve)

        self.writer.write(ProbeEvent(
            step=step,
            probe="mhc",
            layer=layer_idx,
            module="attribution",
            tensor="boundary_normal",
            stats={
                "total_mean": total_bn.mean().item(),
                "total_p99": total_bn.quantile(0.99).item() if N > 10 else total_bn.max().item(),
                **{f"stream_{s}_mean": attr["mean"] for s, attr in enumerate(attributions)},
                **{f"stream_{s}_var_explained": ve for s, ve in enumerate(var_explained)},
            },
        ))

        # === ρ attribution: which stream explains tail events? ===
        # For each token, compute ρ_s = ||x_s|| / ||x_total||
        # Report which stream has highest ρ_s for high-ρ tokens
        rho_total = x_total.norm(dim=-1)
        rho_threshold = rho_total.quantile(0.9) if N > 10 else rho_total.median()
        tail_mask = rho_total > rho_threshold

        if tail_mask.sum() > 0:
            tail_attribution = {}
            for s, x_s in enumerate(streams):
                rho_s = x_s.norm(dim=-1)
                # For tail tokens, what fraction of norm comes from this stream?
                frac_s = rho_s[tail_mask] / (rho_total[tail_mask] + 1e-8)
                tail_attribution[f"stream_{s}_tail_frac"] = frac_s.mean().item()

            self.writer.write(ProbeEvent(
                step=step,
                probe="mhc",
                layer=layer_idx,
                module="attribution",
                tensor="tail",
                stats={
                    "tail_threshold": rho_threshold.item(),
                    "n_tail_tokens": int(tail_mask.sum().item()),
                    **tail_attribution,
                },
            ))

    def _rmsnorm(self, x: torch.Tensor, gamma: torch.Tensor, eps: float) -> torch.Tensor:
        """Apply RMSNorm."""
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
        return gamma * x / rms

    def _gate_scores(
        self,
        y: torch.Tensor,
        W: torch.Tensor,
        bias: torch.Tensor,
    ) -> torch.Tensor:
        """Compute gate scores."""
        import torch.nn.functional as F
        logits = F.linear(y.float(), W.float(), None)
        scores = logits.sigmoid() + bias.unsqueeze(0)
        return scores


# -----------------------------------------------------------------------------
# Utility: compute mixing stability over time
# -----------------------------------------------------------------------------

def compute_mixing_stability(
    mixing_history: list[torch.Tensor],
) -> dict:
    """
    Analyze mixing matrix stability over training.

    Args:
        mixing_history: List of [n_streams, n_streams] mixing matrices

    Returns:
        Dict with stability metrics
    """
    if len(mixing_history) < 2:
        return {"stable": True, "n_samples": len(mixing_history)}

    # Compute pairwise differences
    diffs = []
    for i in range(1, len(mixing_history)):
        diff = (mixing_history[i] - mixing_history[i-1]).abs().mean().item()
        diffs.append(diff)

    return {
        "mean_diff": sum(diffs) / len(diffs),
        "max_diff": max(diffs),
        "n_samples": len(mixing_history),
        "stable": max(diffs) < 0.1,  # Threshold for stability
    }


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    print("mHC probe: stream-level analysis for multi-head charts")
    print("Usage: Instantiate MHCProbe and call log_streams()")
