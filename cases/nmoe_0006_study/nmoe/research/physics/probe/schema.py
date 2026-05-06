"""
Unified probe event schema.

All probes emit events with this structure for consistent downstream analysis.
Events are written to JSONL, one event per line.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


# Standard quantiles for distribution logging
QUANTILES = (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)


@dataclass
class TensorStats:
    """Statistics for a tensor, including quantile spectrum."""
    mean: float
    std: float
    min: float
    max: float
    norm: float
    numel: int
    quantiles: dict[str, float] = field(default_factory=dict)  # "p01", "p05", etc.

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_tensor(cls, x, quantiles: tuple[float, ...] = QUANTILES) -> "TensorStats":
        """Compute stats from a torch tensor."""
        import torch
        x = x.detach().float().flatten()
        if x.numel() == 0:
            return cls(0.0, 0.0, 0.0, 0.0, 0.0, 0)

        q_dict = {}
        if quantiles and x.numel() > 0:
            q_vals = torch.quantile(x, torch.tensor(quantiles, device=x.device))
            for i, q in enumerate(quantiles):
                q_dict[f"p{int(q*100):02d}"] = q_vals[i].item()

        return cls(
            mean=x.mean().item(),
            std=x.std().item(),
            min=x.min().item(),
            max=x.max().item(),
            norm=x.norm().item(),
            numel=x.numel(),
            quantiles=q_dict,
        )


@dataclass
class ProbeEvent:
    """
    Unified probe event.

    All probes emit this structure. Fields:
    - step: training step
    - probe: probe type ("atlas", "trajectory", "deep_linear", "mhc")
    - layer: layer index (-1 for global)
    - module: module name/type (e.g., "router", "gate", "mlp")
    - tensor: tensor name (e.g., "weight", "margin", "grad")
    - stats: TensorStats or raw dict
    - extra: probe-specific metadata
    """
    step: int
    probe: str
    layer: int
    module: str
    tensor: str
    stats: TensorStats | dict
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {
            "step": self.step,
            "probe": self.probe,
            "layer": self.layer,
            "module": self.module,
            "tensor": self.tensor,
            "stats": self.stats.to_dict() if isinstance(self.stats, TensorStats) else self.stats,
        }
        if self.extra:
            d["extra"] = self.extra
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


class ProbeWriter:
    """
    Write probe events to JSONL.

    Usage:
        with ProbeWriter(Path("probe.jsonl")) as writer:
            writer.write(event)
    """

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = None

    def open(self):
        self._file = open(self.path, "a")

    def close(self):
        if self._file:
            self._file.close()
            self._file = None

    def write(self, event: ProbeEvent):
        if self._file is None:
            self.open()
        self._file.write(event.to_json() + "\n")
        self._file.flush()

    def __enter__(self) -> "ProbeWriter":
        self.open()
        return self

    def __exit__(self, *args):
        self.close()


def load_probe_events(path: Path | str) -> list[ProbeEvent]:
    """Load all events from a probe JSONL file."""
    events = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                d = json.loads(line)
                stats = d["stats"]
                if isinstance(stats, dict) and "mean" in stats:
                    stats = TensorStats(**stats)
                events.append(ProbeEvent(
                    step=d["step"],
                    probe=d["probe"],
                    layer=d["layer"],
                    module=d["module"],
                    tensor=d["tensor"],
                    stats=stats,
                    extra=d.get("extra", {}),
                ))
    return events


def filter_events(
    events: list[ProbeEvent],
    probe: str | None = None,
    layer: int | None = None,
    module: str | None = None,
    tensor: str | None = None,
) -> list[ProbeEvent]:
    """Filter events by criteria."""
    result = events
    if probe is not None:
        result = [e for e in result if e.probe == probe]
    if layer is not None:
        result = [e for e in result if e.layer == layer]
    if module is not None:
        result = [e for e in result if e.module == module]
    if tensor is not None:
        result = [e for e in result if e.tensor == tensor]
    return result
