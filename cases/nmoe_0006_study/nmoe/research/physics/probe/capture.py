"""
Probe: Capture model internals during forward/backward passes.

Design principles:
- Log distributions, not just scalars (quantiles capture shape)
- Per-layer granularity
- Minimal overhead when disabled
- JSONL output for streaming analysis
"""
from __future__ import annotations

import json
import math
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn


# Standard quantiles for distribution logging
QUANTILES = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]


@dataclass
class ProbeConfig:
    """Configuration for probing."""
    log_path: Path | str | None = None
    log_every: int = 100  # Log every N steps
    log_activations: bool = True
    log_gradients: bool = True
    log_weights: bool = False  # Usually too expensive
    log_attention: bool = True
    log_router: bool = True  # MoE routing stats
    quantiles: list[float] = field(default_factory=lambda: QUANTILES)
    enabled: bool = True


class Probe:
    """
    Capture and log model internals.

    Usage:
        probe = Probe(ProbeConfig(log_path="probe.jsonl"))

        # During training:
        probe.step_start(step)
        with probe.capture(model):
            loss = model(x)
            loss.backward()
        probe.step_end(loss.item())

        # Or simpler:
        probe.log_step(step, model, loss)
    """

    def __init__(self, config: ProbeConfig):
        self.config = config
        self.current_step: int = 0
        self.captures: dict = {}
        self._handles: list = []
        self._file = None

        if config.log_path:
            path = Path(config.log_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._file = open(path, "a")

    def close(self):
        if self._file:
            self._file.close()
            self._file = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def should_log(self, step: int) -> bool:
        return self.config.enabled and step % self.config.log_every == 0

    def step_start(self, step: int):
        """Call at start of step."""
        self.current_step = step
        self.captures = {"step": step, "layers": {}}

    def step_end(self, loss: float | None = None, metrics: dict | None = None):
        """Call at end of step, writes log entry."""
        if loss is not None:
            self.captures["loss"] = loss
        if metrics:
            self.captures["metrics"] = metrics

        if self._file and self.captures:
            self._file.write(json.dumps(self.captures) + "\n")
            self._file.flush()

        self.captures = {}

    @contextmanager
    def capture(self, model: nn.Module):
        """Context manager to capture activations/gradients during forward/backward."""
        if not self.should_log(self.current_step):
            yield
            return

        self._install_hooks(model)
        try:
            yield
        finally:
            self._remove_hooks()

    def log_step(
        self,
        step: int,
        model: nn.Module,
        loss: float | None = None,
        inputs: torch.Tensor | None = None,
        metrics: dict | None = None,
    ):
        """Convenience method: capture everything for this step."""
        if not self.should_log(step):
            return

        self.step_start(step)

        # Log weight statistics (if enabled)
        if self.config.log_weights:
            self._log_weights(model)

        # Log MoE routing statistics (if enabled and model has MoE)
        if self.config.log_router:
            self._log_router_stats(model)

        self.step_end(loss, metrics)

    # -------------------------------------------------------------------------
    # Internal: Statistics
    # -------------------------------------------------------------------------

    def _stats(self, x: torch.Tensor, prefix: str = "") -> dict:
        """Compute statistics for a tensor."""
        x = x.detach().float().flatten()
        if x.numel() == 0:
            return {}

        stats = {
            f"{prefix}mean": x.mean().item(),
            f"{prefix}std": x.std().item(),
            f"{prefix}min": x.min().item(),
            f"{prefix}max": x.max().item(),
        }

        # Quantiles
        if len(self.config.quantiles) > 0:
            q = torch.quantile(x, torch.tensor(self.config.quantiles, device=x.device))
            for i, qval in enumerate(self.config.quantiles):
                stats[f"{prefix}q{int(qval*100):02d}"] = q[i].item()

        return stats

    def _norm_stats(self, x: torch.Tensor, dim: int = -1, prefix: str = "") -> dict:
        """Compute statistics over norms."""
        norms = x.detach().float().norm(dim=dim).flatten()
        return self._stats(norms, prefix)

    # -------------------------------------------------------------------------
    # Internal: Hooks
    # -------------------------------------------------------------------------

    def _install_hooks(self, model: nn.Module):
        """Install forward/backward hooks on relevant layers."""
        for name, module in model.named_modules():
            # Detect layer index from name (e.g., "blocks.5" or "layers.5")
            layer_idx = self._extract_layer_idx(name)

            # Hook attention
            if self.config.log_attention and self._is_attention(name, module):
                self._handles.append(
                    module.register_forward_hook(
                        self._make_activation_hook(f"L{layer_idx}/attn")
                    )
                )

            # Hook MLP/MoE
            if self.config.log_activations and self._is_mlp(name, module):
                self._handles.append(
                    module.register_forward_hook(
                        self._make_activation_hook(f"L{layer_idx}/mlp")
                    )
                )

            # Hook router specifically
            if self.config.log_router and self._is_router(name, module):
                self._handles.append(
                    module.register_forward_hook(
                        self._make_router_hook(f"L{layer_idx}/router")
                    )
                )

    def _remove_hooks(self):
        """Remove all hooks."""
        for h in self._handles:
            h.remove()
        self._handles = []

    def _make_activation_hook(self, key: str) -> Callable:
        """Create a hook that logs activation statistics."""
        def hook(module, inputs, outputs):
            if isinstance(outputs, tuple):
                out = outputs[0]
            else:
                out = outputs

            if out is not None and isinstance(out, torch.Tensor):
                layer_key = key.split("/")[0]
                if layer_key not in self.captures["layers"]:
                    self.captures["layers"][layer_key] = {}

                subkey = key.split("/")[1] if "/" in key else "out"
                self.captures["layers"][layer_key][subkey] = self._stats(out)
                self.captures["layers"][layer_key][f"{subkey}_norm"] = self._norm_stats(out)

        return hook

    def _make_router_hook(self, key: str) -> Callable:
        """Create a hook that logs router/gating statistics."""
        def hook(module, inputs, outputs):
            # Try to extract gate logits/scores from various router implementations
            layer_key = key.split("/")[0]
            if layer_key not in self.captures["layers"]:
                self.captures["layers"][layer_key] = {}

            router_stats = {}

            # Handle different output formats
            if isinstance(outputs, tuple) and len(outputs) >= 2:
                # Common format: (gates, expert_ids) or (gates, expert_ids, ...)
                gates, expert_ids = outputs[0], outputs[1]

                if isinstance(gates, torch.Tensor):
                    router_stats["gate"] = self._stats(gates)

                if isinstance(expert_ids, torch.Tensor):
                    # Expert selection entropy
                    flat_ids = expert_ids.detach().flatten()
                    if flat_ids.numel() > 0:
                        n_experts = flat_ids.max().item() + 1
                        counts = torch.bincount(flat_ids, minlength=n_experts).float()
                        probs = counts / counts.sum()
                        entropy = -(probs * (probs + 1e-10).log()).sum().item()
                        router_stats["expert_entropy"] = entropy
                        router_stats["expert_counts"] = counts.tolist()

            self.captures["layers"][layer_key]["router"] = router_stats

        return hook

    # -------------------------------------------------------------------------
    # Internal: Module detection
    # -------------------------------------------------------------------------

    def _extract_layer_idx(self, name: str) -> int:
        """Extract layer index from module name."""
        import re
        match = re.search(r"(?:blocks|layers)\.(\d+)", name)
        return int(match.group(1)) if match else -1

    def _is_attention(self, name: str, module: nn.Module) -> bool:
        """Check if module is an attention layer."""
        return any(x in name.lower() for x in ["attn", "attention", "self_attn"])

    def _is_mlp(self, name: str, module: nn.Module) -> bool:
        """Check if module is an MLP/FFN layer."""
        return any(x in name.lower() for x in ["mlp", "ffn", "feed_forward"])

    def _is_router(self, name: str, module: nn.Module) -> bool:
        """Check if module is a router/gate."""
        return any(x in name.lower() for x in ["router", "gate", "moe.gate"])

    # -------------------------------------------------------------------------
    # Weight/Router logging (called explicitly, not via hooks)
    # -------------------------------------------------------------------------

    def _log_weights(self, model: nn.Module):
        """Log weight statistics for all layers."""
        for name, param in model.named_parameters():
            if param.requires_grad and param.numel() > 0:
                layer_idx = self._extract_layer_idx(name)
                layer_key = f"L{layer_idx}"

                if layer_key not in self.captures["layers"]:
                    self.captures["layers"][layer_key] = {}

                # Simplify parameter name
                short_name = name.split(".")[-1]
                self.captures["layers"][layer_key][f"w_{short_name}"] = self._stats(param)

                if param.grad is not None:
                    self.captures["layers"][layer_key][f"g_{short_name}"] = self._stats(param.grad)

    def _log_router_stats(self, model: nn.Module):
        """Log MoE router statistics."""
        for name, module in model.named_modules():
            if not self._is_router(name, module):
                continue

            layer_idx = self._extract_layer_idx(name)
            layer_key = f"L{layer_idx}"

            if layer_key not in self.captures["layers"]:
                self.captures["layers"][layer_key] = {}

            # Log gate weight geometry
            if hasattr(module, "weight"):
                W = module.weight.data
                self.captures["layers"][layer_key]["gate_weight"] = self._stats(W)
                self.captures["layers"][layer_key]["gate_weight_norm"] = self._norm_stats(W, dim=1)

            # Log bias if present
            if hasattr(module, "e_score_correction_bias"):
                bias = module.e_score_correction_bias.data
                self.captures["layers"][layer_key]["gate_bias"] = self._stats(bias)


# -----------------------------------------------------------------------------
# CLI for testing
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile

    print("=== Testing Probe ===")

    # Create a simple model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList([
                nn.ModuleDict({
                    "attn": nn.Linear(64, 64),
                    "mlp": nn.Linear(64, 64),
                })
                for _ in range(3)
            ])

        def forward(self, x):
            for block in self.blocks:
                x = block["attn"](x) + x
                x = block["mlp"](x) + x
            return x.mean()

    model = SimpleModel()
    x = torch.randn(2, 10, 64)

    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        config = ProbeConfig(log_path=f.name, log_every=1)

        with Probe(config) as probe:
            probe.step_start(0)
            with probe.capture(model):
                loss = model(x)
            probe.step_end(loss.item())

        # Read back
        with open(f.name) as f:
            data = json.loads(f.read().strip())

        print(f"Logged step: {data['step']}")
        print(f"Loss: {data['loss']:.4f}")
        print(f"Layers captured: {list(data['layers'].keys())}")
        print(f"L0 keys: {list(data['layers'].get('L0', {}).keys())}")
