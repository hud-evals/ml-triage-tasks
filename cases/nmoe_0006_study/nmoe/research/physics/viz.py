"""
Visualization utilities for architecture evaluation.

Produces publication-ready figures using seaborn for:
- LogitLens KL divergence curves
- Layer-wise heatmaps (Engram-style)
- Comparative bar charts
- Training dynamics plots
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

# Lazy import to avoid issues in headless environments
_PLOTTING_IMPORTED = False
plt = None
sns = None


def _ensure_plotting():
    global _PLOTTING_IMPORTED, plt, sns
    if not _PLOTTING_IMPORTED:
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive backend
        import matplotlib.pyplot as _plt
        import seaborn as _sns

        plt = _plt
        sns = _sns
        _PLOTTING_IMPORTED = True

        # Publication-quality seaborn style
        sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
        sns.set_palette("husl")

        # Additional matplotlib tweaks for publication quality
        plt.rcParams.update({
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.1,
            "axes.linewidth": 1.2,
            "grid.linewidth": 0.8,
            "lines.linewidth": 2.5,
            "lines.markersize": 8,
        })


@dataclass
class LogitLensResult:
    """LogitLens KL divergence results for one model variant."""
    name: str
    kl_by_layer: list[float]  # KL(p_final || p_layer) for each layer
    color: str | None = None
    linestyle: str = "-"


def plot_logitlens_curves(
    results: Sequence[LogitLensResult],
    *,
    title: str = "Prediction Convergence by Layer",
    xlabel: str = "Layer",
    ylabel: str = "KL Divergence to Final Output",
    output_path: Path | str | None = None,
    figsize: tuple[float, float] = (10, 6),
    log_scale: bool = False,
) -> None:
    """
    Plot LogitLens KL divergence curves comparing multiple model variants.

    This reproduces the style of Figure 5 in the Engram paper, showing
    how quickly each model's representations converge to prediction-ready.
    """
    _ensure_plotting()

    fig, ax = plt.subplots(figsize=figsize)

    # Use seaborn color palette
    palette = sns.color_palette("husl", len(results))

    for i, r in enumerate(results):
        color = r.color if r.color else palette[i]
        layers = list(range(len(r.kl_by_layer)))
        ax.plot(
            layers,
            r.kl_by_layer,
            label=r.name,
            color=color,
            linestyle=r.linestyle,
            marker="o",
            markeredgecolor="white",
            markeredgewidth=1.5,
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=15)
    ax.legend(loc="upper right", frameon=True, fancybox=True, shadow=True)
    ax.set_xticks(range(max(len(r.kl_by_layer) for r in results)))
    ax.set_xticklabels([f"L{i}" for i in range(max(len(r.kl_by_layer) for r in results))])

    if log_scale:
        ax.set_yscale("log")

    # Add subtle gradient background
    ax.set_facecolor("#fafafa")

    sns.despine(ax=ax, left=False, bottom=False)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, facecolor="white", edgecolor="none")
        print(f"Saved LogitLens plot to {output_path}")

    plt.close(fig)
    return fig


def plot_logitlens_heatmap(
    results: Sequence[LogitLensResult],
    *,
    title: str = "LogitLens KL Divergence",
    output_path: Path | str | None = None,
    figsize: tuple[float, float] = (12, 5),
    cmap: str = "YlOrRd",
    annot: bool = True,
) -> None:
    """
    Plot LogitLens results as a heatmap (Engram paper Figure 5 style).

    Rows = model variants, Columns = layers
    Color intensity = KL divergence (lighter = lower = better convergence)
    """
    _ensure_plotting()

    n_variants = len(results)
    n_layers = max(len(r.kl_by_layer) for r in results)

    # Build data matrix
    data = np.full((n_variants, n_layers), np.nan)
    for i, r in enumerate(results):
        for j, kl in enumerate(r.kl_by_layer):
            data[i, j] = kl

    fig, ax = plt.subplots(figsize=figsize)

    # Create heatmap with seaborn
    hm = sns.heatmap(
        data,
        ax=ax,
        cmap=cmap,
        annot=annot,
        fmt=".2f",
        cbar_kws={"label": "KL Divergence", "shrink": 0.8},
        linewidths=0.5,
        linecolor="white",
        square=False,
        annot_kws={"size": 9, "weight": "bold"},
    )

    ax.set_yticklabels([r.name for r in results], rotation=0, fontweight="bold")
    ax.set_xticklabels([f"L{i}" for i in range(n_layers)], fontweight="bold")
    ax.set_xlabel("Layer", fontweight="bold")
    ax.set_title(title, pad=15, fontweight="bold")

    # Add border
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(2)
        spine.set_edgecolor("#333333")

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, facecolor="white", edgecolor="none")
        print(f"Saved LogitLens heatmap to {output_path}")

    plt.close(fig)
    return fig


@dataclass
class GateProfileResult:
    """Gate activation profile across layers for one model."""
    name: str
    gate_by_layer: list[float]  # gate_mean for each layer
    color: str | None = None


def plot_gate_profiles(
    results: Sequence[GateProfileResult],
    *,
    title: str = "Gate Activation by Layer",
    output_path: Path | str | None = None,
    figsize: tuple[float, float] = (10, 6),
) -> None:
    """
    Plot gate activation profiles across layers.

    Shows whether gating learns depth-wise selectivity (open early, closed late).
    """
    _ensure_plotting()

    fig, ax = plt.subplots(figsize=figsize)

    # Prepare data for seaborn
    import pandas as pd
    rows = []
    for r in results:
        for layer, gate in enumerate(r.gate_by_layer):
            rows.append({"Model": r.name, "Layer": f"L{layer}", "Gate": gate})
    df = pd.DataFrame(rows)

    # Create grouped bar plot
    sns.barplot(
        data=df,
        x="Layer",
        y="Gate",
        hue="Model",
        ax=ax,
        palette="husl",
        edgecolor="white",
        linewidth=1.5,
    )

    ax.set_xlabel("Layer", fontweight="bold")
    ax.set_ylabel("Gate Activation", fontweight="bold")
    ax.set_title(title, pad=15, fontweight="bold")
    ax.set_ylim(0, max(0.35, df["Gate"].max() * 1.1))

    # Add threshold line
    ax.axhline(y=0.25, color="#666666", linestyle="--", alpha=0.7, linewidth=1.5, label="Init (~0.25)")

    ax.legend(loc="upper right", frameon=True, fancybox=True, shadow=True)

    sns.despine(ax=ax)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, facecolor="white", edgecolor="none")
        print(f"Saved gate profile plot to {output_path}")

    plt.close(fig)
    return fig


@dataclass
class TaskComparisonResult:
    """Task-wise accuracy comparison for one model."""
    name: str
    task_acc: dict[str, float]  # task_name -> accuracy
    color: str | None = None


def plot_task_comparison(
    results: Sequence[TaskComparisonResult],
    *,
    title: str = "Model Accuracy Comparison",
    output_path: Path | str | None = None,
    figsize: tuple[float, float] = (12, 7),
) -> None:
    """
    Grouped bar chart comparing task-wise accuracy across models.
    """
    _ensure_plotting()

    # Prepare data for seaborn
    import pandas as pd
    rows = []
    for r in results:
        for task, acc in r.task_acc.items():
            rows.append({"Model": r.name, "Task": task, "Accuracy": acc})
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=figsize)

    # Create bar plot
    sns.barplot(
        data=df,
        x="Model",
        y="Accuracy",
        hue="Task" if len(df["Task"].unique()) > 1 else None,
        ax=ax,
        palette="viridis",
        edgecolor="white",
        linewidth=2,
    )

    ax.set_xlabel("Model", fontweight="bold")
    ax.set_ylabel("Accuracy", fontweight="bold")
    ax.set_title(title, pad=15, fontweight="bold")
    ax.set_ylim(0, 1.0)

    # Rotate x labels for readability
    plt.xticks(rotation=30, ha="right")

    # Add value labels on bars
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", padding=3, fontsize=9, fontweight="bold")

    if len(df["Task"].unique()) > 1:
        ax.legend(loc="upper right", frameon=True, fancybox=True, shadow=True)

    sns.despine(ax=ax)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, facecolor="white", edgecolor="none")
        print(f"Saved task comparison plot to {output_path}")

    plt.close(fig)
    return fig


def plot_training_curves(
    log_paths: dict[str, Path | str],
    *,
    metric: str = "loss",
    title: str | None = None,
    output_path: Path | str | None = None,
    figsize: tuple[float, float] = (12, 7),
    smoothing: float = 0.9,
) -> None:
    """
    Plot training curves from JSONL log files.
    """
    _ensure_plotting()
    import pandas as pd

    fig, ax = plt.subplots(figsize=figsize)

    all_data = []
    for name, path in log_paths.items():
        steps = []
        values = []

        with open(path, "r") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    if "step" in rec and metric in rec:
                        steps.append(rec["step"])
                        values.append(rec[metric])
                except json.JSONDecodeError:
                    continue

        if not steps:
            continue

        # Apply exponential smoothing
        if smoothing > 0:
            smoothed = []
            val = values[0]
            for v in values:
                val = smoothing * val + (1 - smoothing) * v
                smoothed.append(val)
            values = smoothed

        for s, v in zip(steps, values):
            all_data.append({"Model": name, "Step": s, metric: v})

    if not all_data:
        print("No data to plot")
        return

    df = pd.DataFrame(all_data)

    # Plot with seaborn
    sns.lineplot(
        data=df,
        x="Step",
        y=metric,
        hue="Model",
        ax=ax,
        palette="husl",
        linewidth=2.5,
    )

    ax.set_xlabel("Training Step", fontweight="bold")
    ax.set_ylabel(metric.replace("_", " ").title(), fontweight="bold")
    ax.set_title(title or f"Training Curve: {metric}", pad=15, fontweight="bold")

    ax.legend(loc="upper right", frameon=True, fancybox=True, shadow=True)

    sns.despine(ax=ax)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, facecolor="white", edgecolor="none")
        print(f"Saved training curves to {output_path}")

    plt.close(fig)
    return fig


def plot_scale_sweep(
    results: dict[int, dict[str, float]],
    *,
    title: str = "Scale Sweep: Accuracy vs Model Dimension",
    output_path: Path | str | None = None,
    figsize: tuple[float, float] = (10, 6),
) -> None:
    """
    Plot accuracy vs model dimension for scale sweep experiments.
    """
    _ensure_plotting()
    import pandas as pd

    rows = []
    for dim, variants in results.items():
        for variant, acc in variants.items():
            rows.append({"Dimension": dim, "Model": variant, "Accuracy": acc})

    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=figsize)

    sns.lineplot(
        data=df,
        x="Dimension",
        y="Accuracy",
        hue="Model",
        ax=ax,
        palette="husl",
        marker="o",
        markersize=10,
        markeredgecolor="white",
        markeredgewidth=2,
        linewidth=3,
    )

    ax.set_xlabel("Model Dimension", fontweight="bold")
    ax.set_ylabel("Accuracy", fontweight="bold")
    ax.set_title(title, pad=15, fontweight="bold")
    ax.set_xscale("log", base=2)

    ax.legend(loc="best", frameon=True, fancybox=True, shadow=True)

    sns.despine(ax=ax)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, facecolor="white", edgecolor="none")
        print(f"Saved scale sweep plot to {output_path}")

    plt.close(fig)
    return fig


def plot_convergence_comparison(
    results: Sequence[LogitLensResult],
    *,
    title: str = "Prediction Convergence: Early vs Late Layers",
    output_path: Path | str | None = None,
    figsize: tuple[float, float] = (10, 8),
) -> None:
    """
    Two-panel plot showing early-layer KL (L0-L2) vs late-layer KL (L3-L5).

    This highlights whether memory/Canon accelerates EARLY convergence as Engram claims.
    """
    _ensure_plotting()
    import pandas as pd

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)

    # Prepare data
    n_layers = max(len(r.kl_by_layer) for r in results)
    mid = n_layers // 2

    early_data = []
    late_data = []
    for r in results:
        for i, kl in enumerate(r.kl_by_layer):
            if i < mid:
                early_data.append({"Model": r.name, "Layer": f"L{i}", "KL": kl})
            else:
                late_data.append({"Model": r.name, "Layer": f"L{i}", "KL": kl})

    # Early layers
    df_early = pd.DataFrame(early_data)
    sns.barplot(
        data=df_early,
        x="Model",
        y="KL",
        hue="Layer",
        ax=axes[0],
        palette="YlOrRd",
        edgecolor="white",
        linewidth=1.5,
    )
    axes[0].set_title("Early Layers (L0-L2)", fontweight="bold")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("KL Divergence", fontweight="bold")
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].legend(title="Layer", loc="upper right")

    # Late layers
    df_late = pd.DataFrame(late_data)
    sns.barplot(
        data=df_late,
        x="Model",
        y="KL",
        hue="Layer",
        ax=axes[1],
        palette="YlGnBu",
        edgecolor="white",
        linewidth=1.5,
    )
    axes[1].set_title("Late Layers (L3-L5)", fontweight="bold")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("")
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].legend(title="Layer", loc="upper right")

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)

    sns.despine()
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, facecolor="white", edgecolor="none")
        print(f"Saved convergence comparison to {output_path}")

    plt.close(fig)
    return fig


# ----------------------------- CLI for quick viz -----------------------------

def main():
    """Quick CLI for generating plots from saved results."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate architecture evaluation plots")
    parser.add_argument("--results", type=str, required=True, help="Path to results.json")
    parser.add_argument("--output", type=str, default="./plots", help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.results) as f:
        data = json.load(f)

    # LogitLens plots
    logitlens_results = [
        LogitLensResult(name=k, kl_by_layer=v["logitlens_kl"])
        for k, v in data.items()
        if v.get("logitlens_kl")
    ]
    if logitlens_results:
        plot_logitlens_curves(logitlens_results, output_path=output_dir / "logitlens_curves.pdf")
        plot_logitlens_heatmap(logitlens_results, output_path=output_dir / "logitlens_heatmap.pdf")
        plot_convergence_comparison(logitlens_results, output_path=output_dir / "convergence_comparison.pdf")

    # Gate profiles
    gate_results = [
        GateProfileResult(name=k, gate_by_layer=v["gate_profile"])
        for k, v in data.items()
        if v.get("gate_profile")
    ]
    if gate_results:
        plot_gate_profiles(gate_results, output_path=output_dir / "gate_profiles.pdf")

    # Task comparison
    task_results = [
        TaskComparisonResult(name=k, task_acc=v["task_acc"])
        for k, v in data.items()
    ]
    if task_results:
        plot_task_comparison(task_results, output_path=output_dir / "task_comparison.pdf")

    print(f"All plots saved to {output_dir}")


if __name__ == "__main__":
    main()
