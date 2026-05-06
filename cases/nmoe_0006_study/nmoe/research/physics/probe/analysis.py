"""
Analysis utilities for probe logs.

Load JSONL logs, compute statistics, generate plots.
All plots go to files (not interactive) for reproducibility.
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass
class ProbeEntry:
    """A single probe log entry."""
    step: int
    loss: float | None
    layers: dict
    metrics: dict | None

    @classmethod
    def from_dict(cls, d: dict) -> "ProbeEntry":
        return cls(
            step=d.get("step", 0),
            loss=d.get("loss"),
            layers=d.get("layers", {}),
            metrics=d.get("metrics"),
        )


def load_probe_log(path: Path | str) -> list[ProbeEntry]:
    """Load all entries from a probe log file."""
    path = Path(path)
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(ProbeEntry.from_dict(json.loads(line)))
    return entries


def stream_probe_log(path: Path | str) -> Iterator[ProbeEntry]:
    """Stream entries from a probe log file."""
    path = Path(path)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield ProbeEntry.from_dict(json.loads(line))


def extract_metric(
    entries: list[ProbeEntry],
    layer: str,
    metric: str,
) -> tuple[list[int], list[float]]:
    """
    Extract a specific metric over training steps.

    Args:
        entries: Probe log entries
        layer: Layer key (e.g., "L0", "L5")
        metric: Metric path within layer (e.g., "mlp.mean", "router.expert_entropy")

    Returns:
        (steps, values) tuples
    """
    steps = []
    values = []

    for entry in entries:
        if layer not in entry.layers:
            continue

        layer_data = entry.layers[layer]

        # Navigate nested keys
        parts = metric.split(".")
        val = layer_data
        for part in parts:
            if isinstance(val, dict) and part in val:
                val = val[part]
            else:
                val = None
                break

        if val is not None and isinstance(val, (int, float)):
            steps.append(entry.step)
            values.append(float(val))

    return steps, values


def extract_all_layers(
    entries: list[ProbeEntry],
    metric: str,
) -> dict[str, tuple[list[int], list[float]]]:
    """Extract a metric for all layers."""
    # Find all layer keys
    layer_keys = set()
    for entry in entries:
        layer_keys.update(entry.layers.keys())

    result = {}
    for layer in sorted(layer_keys, key=lambda x: int(x[1:]) if x[1:].isdigit() else 0):
        steps, values = extract_metric(entries, layer, metric)
        if values:
            result[layer] = (steps, values)

    return result


def summarize_log(entries: list[ProbeEntry]) -> dict:
    """Generate summary statistics from a probe log."""
    if not entries:
        return {}

    summary = {
        "n_entries": len(entries),
        "step_range": (entries[0].step, entries[-1].step),
        "layers": set(),
        "metrics_per_layer": defaultdict(set),
    }

    for entry in entries:
        for layer, data in entry.layers.items():
            summary["layers"].add(layer)
            _collect_keys(data, "", summary["metrics_per_layer"][layer])

    summary["layers"] = sorted(summary["layers"])
    summary["metrics_per_layer"] = {
        k: sorted(v) for k, v in summary["metrics_per_layer"].items()
    }

    return summary


def _collect_keys(d: dict, prefix: str, keys: set):
    """Recursively collect all keys in a nested dict."""
    for k, v in d.items():
        full_key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            _collect_keys(v, full_key, keys)
        else:
            keys.add(full_key)


# -----------------------------------------------------------------------------
# Plotting (optional, requires matplotlib)
# -----------------------------------------------------------------------------

def plot_metric_over_training(
    entries: list[ProbeEntry],
    metric: str,
    output: Path | str,
    layers: list[str] | None = None,
    title: str | None = None,
):
    """
    Plot a metric over training steps for multiple layers.

    Args:
        entries: Probe log entries
        metric: Metric path (e.g., "mlp.mean", "router.expert_entropy")
        output: Output path for plot
        layers: Which layers to plot (None = all)
        title: Plot title
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot")
        return

    all_data = extract_all_layers(entries, metric)

    if layers:
        all_data = {k: v for k, v in all_data.items() if k in layers}

    if not all_data:
        print(f"No data found for metric: {metric}")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    for layer, (steps, values) in all_data.items():
        ax.plot(steps, values, label=layer, alpha=0.8)

    ax.set_xlabel("Step")
    ax.set_ylabel(metric)
    ax.set_title(title or f"{metric} over training")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output}")


def plot_layer_heatmap(
    entries: list[ProbeEntry],
    metric: str,
    output: Path | str,
    title: str | None = None,
):
    """
    Plot metric as heatmap (step x layer).

    Useful for seeing how something evolves across both time and depth.
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib/numpy not available, skipping plot")
        return

    all_data = extract_all_layers(entries, metric)

    if not all_data:
        print(f"No data found for metric: {metric}")
        return

    # Build matrix
    layers = sorted(all_data.keys(), key=lambda x: int(x[1:]) if x[1:].isdigit() else 0)
    steps = sorted(set(s for layer in layers for s in all_data[layer][0]))

    step_to_idx = {s: i for i, s in enumerate(steps)}
    matrix = np.full((len(layers), len(steps)), np.nan)

    for i, layer in enumerate(layers):
        layer_steps, layer_vals = all_data[layer]
        for s, v in zip(layer_steps, layer_vals):
            matrix[i, step_to_idx[s]] = v

    fig, ax = plt.subplots(figsize=(12, 6))

    im = ax.imshow(matrix, aspect="auto", cmap="viridis", interpolation="nearest")
    ax.set_xlabel("Step")
    ax.set_ylabel("Layer")

    # Tick labels
    n_xticks = min(10, len(steps))
    xtick_idx = np.linspace(0, len(steps) - 1, n_xticks, dtype=int)
    ax.set_xticks(xtick_idx)
    ax.set_xticklabels([steps[i] for i in xtick_idx])

    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels(layers)

    ax.set_title(title or f"{metric} heatmap")
    fig.colorbar(im, ax=ax, label=metric)

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Analyze probe logs")
    parser.add_argument("log_path", type=Path, help="Path to probe.jsonl")
    parser.add_argument("--summary", action="store_true", help="Print summary")
    parser.add_argument("--plot", type=str, help="Metric to plot")
    parser.add_argument("--heatmap", type=str, help="Metric to plot as heatmap")
    parser.add_argument("--output", type=Path, default=Path("plots"), help="Output directory")
    parser.add_argument("--layers", type=str, help="Comma-separated layers to include")

    args = parser.parse_args()

    entries = load_probe_log(args.log_path)

    if args.summary:
        summary = summarize_log(entries)
        print(f"Entries: {summary['n_entries']}")
        print(f"Steps: {summary['step_range'][0]} - {summary['step_range'][1]}")
        print(f"Layers: {summary['layers']}")
        print("\nMetrics per layer:")
        for layer, metrics in summary["metrics_per_layer"].items():
            print(f"  {layer}: {metrics[:5]}{'...' if len(metrics) > 5 else ''}")

    if args.plot:
        layers = args.layers.split(",") if args.layers else None
        plot_metric_over_training(
            entries,
            args.plot,
            args.output / f"{args.plot.replace('.', '_')}.png",
            layers=layers,
        )

    if args.heatmap:
        plot_layer_heatmap(
            entries,
            args.heatmap,
            args.output / f"{args.heatmap.replace('.', '_')}_heatmap.png",
        )


if __name__ == "__main__":
    main()
