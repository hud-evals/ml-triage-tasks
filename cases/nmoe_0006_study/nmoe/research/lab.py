"""nmoe research lab - frontier-quality research interface.

Usage:
    from nmoe.research import lab

    # Run physics experiment
    run = lab.physics('stage1', steps=2000)

    # Analyze
    run.summary()
    run.loss_curve()
    run.layer_ce()

    # Compare
    lab.compare([run1, run2], metric='loss')
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    import matplotlib.pyplot as plt
    import pandas as pd

# =============================================================================
# ROSEPINE DAWN PALETTE
# =============================================================================

ROSEPINE = {
    'base': '#faf4ed',
    'surface': '#fffaf3',
    'overlay': '#f2e9e1',
    'muted': '#9893a5',
    'subtle': '#797593',
    'text': '#575279',
    'love': '#b4637a',
    'gold': '#ea9d34',
    'rose': '#d7827e',
    'pine': '#286983',
    'foam': '#56949f',
    'iris': '#907aa9',
}

PALETTE = [
    ROSEPINE['pine'],
    ROSEPINE['love'],
    ROSEPINE['gold'],
    ROSEPINE['foam'],
    ROSEPINE['iris'],
    ROSEPINE['rose'],
]

_THEME_APPLIED = False


def _ensure_plotting():
    """Import plotting deps and apply theme. Raises clear error if missing."""
    global _THEME_APPLIED

    try:
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import seaborn as sns
    except ImportError as e:
        raise ImportError(
            f"Plotting requires matplotlib, pandas, seaborn. Missing: {e.name}\n"
            f"Install with: pip install matplotlib pandas seaborn"
        ) from e

    if not _THEME_APPLIED:
        plt.rcParams.update({
            'figure.facecolor': ROSEPINE['base'],
            'figure.figsize': (10, 6),
            'figure.dpi': 100,
            'axes.facecolor': ROSEPINE['surface'],
            'axes.edgecolor': ROSEPINE['overlay'],
            'axes.labelcolor': ROSEPINE['text'],
            'axes.titlecolor': ROSEPINE['pine'],
            'axes.titleweight': 'bold',
            'axes.titlesize': 14,
            'axes.labelsize': 11,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'axes.grid': True,
            'axes.axisbelow': True,
            'grid.color': ROSEPINE['overlay'],
            'grid.linewidth': 0.8,
            'grid.alpha': 0.7,
            'xtick.color': ROSEPINE['subtle'],
            'ytick.color': ROSEPINE['subtle'],
            'text.color': ROSEPINE['text'],
            'legend.facecolor': ROSEPINE['surface'],
            'legend.edgecolor': ROSEPINE['overlay'],
            'lines.linewidth': 2,
            'lines.markersize': 6,
            'savefig.facecolor': ROSEPINE['base'],
            'savefig.bbox': 'tight',
            'savefig.dpi': 150,
        })
        sns.set_palette(PALETTE)
        _THEME_APPLIED = True

    return plt, pd, sns, np


# =============================================================================
# PHYSICS RUN
# =============================================================================

@dataclass
class PhysicsRun:
    """A completed physics ablation run."""

    path: Path
    name: str
    created: datetime
    _summary_cache: dict | None = field(default=None, repr=False)
    _logs_cache: dict = field(default_factory=dict, repr=False)

    def __repr__(self):
        return f"PhysicsRun({self.name}, {self.created.strftime('%Y-%m-%d %H:%M')})"

    # -------------------------------------------------------------------------
    # Data loading
    # -------------------------------------------------------------------------

    def _load_summary(self) -> dict[str, dict]:
        """Load summary.json. Returns {variant_key: {metric: value}}."""
        if self._summary_cache is not None:
            return self._summary_cache

        p = self.path / 'analysis' / 'summary.json'
        if not p.exists():
            return {}

        with open(p) as f:
            self._summary_cache = json.load(f)
        return self._summary_cache

    def _load_logs(self, variant: str) -> list[dict]:
        """Load training logs for a variant."""
        if variant in self._logs_cache:
            return self._logs_cache[variant]

        p = self.path / 'runs' / variant / 'train.jsonl'
        if not p.exists():
            return []

        rows = []
        with open(p) as f:
            for line in f:
                rows.append(json.loads(line))

        self._logs_cache[variant] = rows
        return rows

    def _load_layer_ce(self) -> dict:
        """Load per-layer CE data."""
        p = self.path / 'analysis' / 'layer_ce_valid.json'
        if not p.exists():
            return {}
        with open(p) as f:
            return json.load(f)

    def _load_slices(self) -> dict:
        """Load per-slice metrics."""
        p = self.path / 'analysis' / 'slices_valid.json'
        if not p.exists():
            return {}
        with open(p) as f:
            return json.load(f)

    def _load_cka(self) -> dict:
        """Load CKA data, filtering out _meta."""
        p = self.path / 'analysis' / 'cka_valid.json'
        if not p.exists():
            return {}
        with open(p) as f:
            data = json.load(f)
        # Filter out _meta
        return {k: v for k, v in data.items() if k != '_meta'}

    @property
    def variants(self) -> list[str]:
        """List variants in this run."""
        runs_dir = self.path / 'runs'
        if not runs_dir.exists():
            return []
        return sorted([d.name for d in runs_dir.iterdir() if d.is_dir()])

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------

    def summary(self) -> pd.DataFrame:
        """Return summary as DataFrame with variant as column."""
        plt, pd, _, _ = _ensure_plotting()

        data = self._load_summary()
        if not data:
            print("No summary data found")
            return pd.DataFrame()

        # Convert {variant: {metric: val}} to list of dicts with variant column
        rows = []
        for variant, metrics in data.items():
            row = {'variant': _short_variant(variant), **metrics}
            rows.append(row)

        df = pd.DataFrame(rows)

        # Reorder columns
        cols = ['variant']
        for c in ['loss', 'token_acc', 'exact_match', 'dp_kl']:
            if c in df.columns:
                cols.append(c)
        # Add any remaining columns
        cols += [c for c in df.columns if c not in cols]
        df = df[cols]

        return df

    def table(self, metrics: list[str] | None = None) -> pd.DataFrame:
        """Get raw metrics as DataFrame."""
        _, pd, _, _ = _ensure_plotting()
        data = self._load_summary()
        rows = [{'variant': k, **v} for k, v in data.items()]
        df = pd.DataFrame(rows)
        if metrics:
            cols = ['variant'] + [c for c in metrics if c in df.columns]
            df = df[cols]
        return df

    # -------------------------------------------------------------------------
    # Visualizations
    # -------------------------------------------------------------------------

    def loss_curve(self, variants: list[str] | None = None):
        """Plot training loss curves."""
        plt, pd, _, _ = _ensure_plotting()

        variants = variants or self.variants
        fig, ax = plt.subplots()

        for i, variant in enumerate(variants):
            logs = self._load_logs(variant)
            if not logs:
                continue
            steps = [r.get('step', i) for i, r in enumerate(logs)]
            losses = [r.get('loss', float('nan')) for r in logs]
            label = _short_variant(variant)
            ax.plot(steps, losses, label=label, color=PALETTE[i % len(PALETTE)])

        ax.set_xlabel('Step')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss')
        ax.legend()
        return fig

    def layer_ce(self, task: str | None = None):
        """Plot per-layer CE (LogitScope visualization)."""
        plt, _, _, _ = _ensure_plotting()

        data = self._load_layer_ce()
        if not data:
            print("No layer_ce data (run with --layer-ce)")
            return None

        fig, ax = plt.subplots()
        i = 0

        for variant, tasks in data.items():
            for task_name, metrics in tasks.items():
                if task and task not in task_name:
                    continue
                ce = metrics.get('ce_by_layer', [])
                if not ce:
                    continue
                label = _short_variant(variant)
                if len(tasks) > 1:
                    label += f" / {task_name}"
                ax.plot(ce, label=label, color=PALETTE[i % len(PALETTE)])
                i += 1

        ax.set_xlabel('Layer')
        ax.set_ylabel('CE to Ground Truth')
        ax.set_title('LogitScope: Per-Layer CE')
        ax.legend(fontsize=8)
        return fig

    def gate_heatmap(self, variant: str | None = None):
        """Plot Engram gate activation heatmap."""
        plt, pd, sns, _ = _ensure_plotting()
        import numpy as np

        slices = self._load_slices()
        if not slices:
            print("No slice data (run with --slice-metrics)")
            return None

        # Find engram variant if not specified
        if variant is None:
            variant = next((v for v in slices if 'engram' in v.lower()), None)
        if variant is None or variant not in slices:
            print(f"No gate data. Available: {list(slices.keys())}")
            return None

        gate_data = {}
        for task, metrics in slices[variant].items():
            if 'gate_mean_by_layer' in metrics:
                gate_data[task] = metrics['gate_mean_by_layer']

        if not gate_data:
            print("No gate_mean_by_layer data")
            return None

        df = pd.DataFrame(gate_data)
        fig, ax = plt.subplots(figsize=(12, max(4, len(df.columns) * 0.5)))
        sns.heatmap(df.T, ax=ax, cmap='YlOrRd', cbar_kws={'label': 'Gate Mean'})
        ax.set_xlabel('Layer')
        ax.set_ylabel('Task')
        ax.set_title(f'Engram Gate: {_short_variant(variant)}')
        return fig

    def cka_heatmap(self, variant: str | None = None, task: str | None = None):
        """Plot CKA layer alignment heatmap."""
        plt, _, sns, _ = _ensure_plotting()
        import numpy as np

        data = self._load_cka()
        if not data:
            print("No CKA data (run with --cka)")
            return None

        # Get first non-meta variant
        variants = list(data.keys())
        if not variants:
            print("No variants in CKA data")
            return None

        variant = variant or variants[0]
        if variant not in data:
            print(f"Variant not found. Available: {variants}")
            return None

        tasks = data[variant]
        if not tasks:
            print("No tasks in CKA data")
            return None

        task = task or list(tasks.keys())[0]
        if task not in tasks:
            print(f"Task not found. Available: {list(tasks.keys())}")
            return None

        matrix = np.array(tasks[task])
        fig, ax = plt.subplots(figsize=(8, 8))
        sns.heatmap(matrix, ax=ax, cmap='viridis', square=True, cbar_kws={'label': 'CKA'})
        ax.set_xlabel('Baseline Layer')
        ax.set_ylabel(f'{_short_variant(variant)} Layer')
        ax.set_title(f'CKA: {task}')
        return fig


def _short_variant(v: str) -> str:
    """Shorten variant string for labels."""
    parts = []
    for kv in v.split(','):
        if '=' in kv:
            k, val = kv.split('=', 1)
            if k in ('memory', 'residual', 'precond', 'attn') and val not in ('none', 'vanilla', 'global', 'fixed'):
                parts.append(val)
            elif k == 'canon_set':
                parts.append(f'canon-{val}')
        else:
            parts.append(kv)
    return '+'.join(parts) if parts else v[:20]


# =============================================================================
# EXPERIMENT LAUNCHERS
# =============================================================================

# Shorthand variant names -> full specs
VARIANT_SHORTHANDS = {
    'baseline': 'width=fixed,residual=vanilla',
    'canon': 'width=fixed,residual=canon,canon_set=ABCD',
    'canon-AB': 'width=fixed,residual=canon,canon_set=AB',
    'canon-A': 'width=fixed,residual=canon,canon_set=A',
    'engram': 'width=fixed,residual=vanilla,memory=engram',
    'canon+engram': 'width=fixed,residual=canon,canon_set=ABCD,memory=engram',
    'ple': 'width=fixed,residual=vanilla,memory=ple_ngrammer',
}

def _expand_variant(v: str) -> str:
    """Expand shorthand variant names to full specs."""
    return VARIANT_SHORTHANDS.get(v, v)


def physics(
    name: str | None = None,
    tasks: list[str] | None = None,
    variants: list[str] | None = None,
    seeds: list[int] | int = 0,
    steps: int = 2000,
    dim: int = 256,
    n_layers: int = 6,
    seq_len: int = 2048,
    mlp_type: str = 'swiglu',
    slice_metrics: bool = True,
    layer_ce: bool = False,
    dry_run: bool = False,
    resume: bool = True,
) -> 'PhysicsExperiment | None':
    """Run physics architecture ablation with multi-seed and multi-variant support.

    Args:
        name: Experiment name (default: auto-generated timestamp)
        tasks: Task specs (e.g., ['depo_v2:1.0'] or ['depo_v2:0.5', 'brevo:0.5'])
        variants: Variant specs (e.g., ['baseline', 'canon', 'canon+engram'])
        seeds: Single seed or list of seeds for statistical significance
        steps: Training steps per run
        dim: Model dimension
        n_layers: Number of layers
        seq_len: Sequence length
        mlp_type: MLP type (swiglu required for Canon-D)
        slice_metrics: Enable per-slice metrics
        layer_ce: Enable per-layer CE analysis
        dry_run: Print commands without running
        resume: Skip completed runs when re-running (default True)

    Returns:
        PhysicsExperiment object with aggregated results across seeds/variants

    Example:
        >>> exp = lab.physics(
        ...     name="canon_ablation",
        ...     tasks=["depo_v2:1.0"],
        ...     variants=["baseline", "canon"],
        ...     seeds=[0, 1, 2],
        ...     steps=2000,
        ... )
        >>> exp.summary()  # Shows mean ± std across seeds
        >>> exp.compare("baseline", "canon")  # Statistical comparison

        # Resume an interrupted experiment:
        >>> exp = lab.physics(name="canon_ablation", ...)  # Skips completed runs
    """
    # Normalize inputs
    if isinstance(seeds, int):
        seeds = [seeds]
    if variants is None:
        variants = ['baseline']
    if tasks is None:
        tasks = ['depo_v2:1.0']

    # Generate experiment name
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    name = name or f'exp_{timestamp}'
    base_path = Path('/data/physics') / name
    base_path.mkdir(parents=True, exist_ok=True)

    # Expand variant shorthands
    expanded_variants = [_expand_variant(v) for v in variants]
    variant_names = variants  # Keep original names for readability

    # Check for completed runs (for resume)
    completed_runs: set[tuple[str, int]] = set()
    if resume:
        for variant in variant_names:
            for seed in seeds:
                run_path = base_path / f"{variant}_s{seed}"
                summary = run_path / 'analysis' / 'summary.json'
                if summary.exists():
                    completed_runs.add((variant, seed))

    # Calculate total runs
    total_runs = len(variants) * len(seeds)
    pending_runs = total_runs - len(completed_runs)

    print(f"{'─' * 70}")
    print(f"  Experiment: {name}")
    print(f"  Variants:   {variant_names}")
    print(f"  Seeds:      {seeds}")
    print(f"  Total runs: {total_runs}")
    if completed_runs:
        print(f"  Completed:  {len(completed_runs)} (resuming)")
        print(f"  Pending:    {pending_runs}")
    print(f"{'─' * 70}")

    if dry_run:
        for variant, variant_spec in zip(variant_names, expanded_variants):
            for seed in seeds:
                run_name = f"{variant}_s{seed}"
                output = base_path / run_name
                cmd = _build_physics_cmd(
                    output=str(output), steps=steps, seed=seed, dim=dim,
                    n_layers=n_layers, seq_len=seq_len, mlp_type=mlp_type,
                    tasks=tasks, variant_spec=variant_spec,
                    slice_metrics=slice_metrics, layer_ce=layer_ce,
                )
                print(f"[DRY RUN] {run_name}: {' '.join(cmd)}")
        return None

    # Import physics training API
    from nmoe.research.physics.arch_ablations import run_single_variant

    # Try to import tqdm for notebook-native progress
    try:
        from tqdm.auto import tqdm
        HAS_TQDM = True
    except ImportError:
        HAS_TQDM = False

    # Run all combinations
    runs: dict[str, dict[int, Path]] = {v: {} for v in variant_names}

    # Pre-populate with completed runs
    for variant, seed in completed_runs:
        runs[variant][seed] = base_path / f"{variant}_s{seed}"

    run_idx = 0
    for variant, variant_spec in zip(variant_names, expanded_variants):
        for seed in seeds:
            run_idx += 1
            run_name = f"{variant}_s{seed}"
            output = base_path / run_name

            # Skip completed runs
            if (variant, seed) in completed_runs:
                print(f"\n[{run_idx}/{total_runs}] {run_name} - SKIPPED (complete)")
                continue

            # Run with tqdm progress or simple output
            if HAS_TQDM:
                pbar = tqdm(total=steps, desc=f"[{run_idx}/{total_runs}] {run_name}", unit="step",
                            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {postfix}]')

                def progress_callback(step: int, total: int, loss: float, metrics: dict):
                    acc = metrics.get('answer_token_acc', 0)
                    pbar.set_postfix(loss=f"{loss:.3f}", acc=f"{acc:.3f}")
                    pbar.update(1)

                try:
                    run_single_variant(
                        output=output,
                        variant_spec=variant_spec,
                        steps=steps,
                        seed=seed,
                        dim=dim,
                        n_layers=n_layers,
                        seq_len=seq_len,
                        mlp_type=mlp_type,
                        tasks=tasks,
                        slice_metrics=slice_metrics,
                        layer_ce=layer_ce,
                        progress_callback=progress_callback,
                    )
                    runs[variant][seed] = output
                except Exception as e:
                    pbar.close()
                    print(f"  FAILED: {str(e)[:200]}")
                    continue
                finally:
                    pbar.close()
            else:
                # Fallback: simple progress prints
                print(f"\n[{run_idx}/{total_runs}] Running {run_name}...")

                last_print_step = [0]  # mutable for closure
                def progress_callback(step: int, total: int, loss: float, metrics: dict):
                    # Print every 100 steps
                    if step - last_print_step[0] >= 100 or step == total:
                        acc = metrics.get('answer_token_acc', 0)
                        print(f"  step {step}: loss={loss:.3f}, acc={acc:.3f}")
                        last_print_step[0] = step

                try:
                    run_single_variant(
                        output=output,
                        variant_spec=variant_spec,
                        steps=steps,
                        seed=seed,
                        dim=dim,
                        n_layers=n_layers,
                        seq_len=seq_len,
                        mlp_type=mlp_type,
                        tasks=tasks,
                        slice_metrics=slice_metrics,
                        layer_ce=layer_ce,
                        progress_callback=progress_callback,
                    )
                    runs[variant][seed] = output
                    print(f"  Done: {output}")
                except Exception as e:
                    print(f"  FAILED: {str(e)[:200]}")
                    continue

    # Save experiment metadata
    meta = {
        'name': name,
        'variants': variant_names,
        'seeds': seeds,
        'steps': steps,
        'dim': dim,
        'n_layers': n_layers,
        'tasks': tasks,
        'runs': {v: {s: str(p) for s, p in seeds_dict.items()} for v, seeds_dict in runs.items()},
        'created': datetime.now().isoformat(),
    }
    (base_path / 'experiment.json').write_text(json.dumps(meta, indent=2))

    print(f"\n{'─' * 70}")
    print(f"  Experiment complete: {base_path}")
    print(f"{'─' * 70}")

    return PhysicsExperiment(path=base_path, meta=meta)


def _build_physics_cmd(
    output: str, steps: int, seed: int, dim: int, n_layers: int,
    seq_len: int, mlp_type: str, tasks: list[str], variant_spec: str,
    slice_metrics: bool, layer_ce: bool,
) -> list[str]:
    """Build command for a single physics run."""
    cmd = [
        sys.executable, '-m', 'nmoe.research.physics.arch_ablations',
        '--output', output,
        '--steps', str(steps),
        '--seed', str(seed),
        '--init-seed', str(seed),  # Same init seed for reproducibility
        '--dim', str(dim),
        '--n-layers', str(n_layers),
        '--seq-len', str(seq_len),
        '--mlp-type', mlp_type,
        '--tasks', *tasks,
        '--variant', variant_spec,
    ]
    if slice_metrics:
        cmd.extend(['--slice-metrics', '--slice-metrics-n', '512'])
    if layer_ce:
        cmd.extend(['--layer-ce', '--layer-ce-n', '256'])
    return cmd


@dataclass
class PhysicsExperiment:
    """Multi-seed, multi-variant physics experiment with aggregation."""
    path: Path
    meta: dict

    @property
    def name(self) -> str:
        return self.meta['name']

    @property
    def variants(self) -> list[str]:
        return self.meta['variants']

    @property
    def seeds(self) -> list[int]:
        return self.meta['seeds']

    def _load_run_metrics(self, variant: str, seed: int) -> dict | None:
        """Load metrics for a single run."""
        variant_runs = self.meta['runs'].get(variant, {})
        # Handle both int keys (in-memory) and string keys (from JSON)
        run_path_str = variant_runs.get(seed) or variant_runs.get(str(seed), '')
        run_path = Path(run_path_str) if run_path_str else Path()
        summary_path = run_path / 'analysis' / 'summary.json'
        if not summary_path.exists():
            # Try runs.json format
            runs_json = run_path / 'runs.json'
            if runs_json.exists():
                data = json.loads(runs_json.read_text())
                if data:
                    return list(data.values())[0].get('final', {})
        else:
            data = json.loads(summary_path.read_text())
            if data:
                return list(data.values())[0]
        return None

    def summary(self) -> None:
        """Print summary with mean ± std across seeds."""
        _, _, _, np = _ensure_plotting()

        print(f"\n{'═' * 70}")
        print(f"  {self.name} - Summary (n={len(self.seeds)} seeds)")
        print(f"{'═' * 70}")

        rows = []
        for variant in self.variants:
            metrics_list = []
            for seed in self.seeds:
                m = self._load_run_metrics(variant, seed)
                if m and 'valid' in m:
                    metrics_list.append(m['valid'])

            if not metrics_list:
                continue

            # Aggregate
            loss_vals = [m['loss'] for m in metrics_list if 'loss' in m]
            acc_vals = [m['answer_token_acc'] for m in metrics_list if 'answer_token_acc' in m]

            rows.append({
                'variant': variant,
                'loss': f"{np.mean(loss_vals):.4f} ± {np.std(loss_vals):.4f}" if loss_vals else "N/A",
                'accuracy': f"{np.mean(acc_vals):.4f} ± {np.std(acc_vals):.4f}" if acc_vals else "N/A",
                '_loss_mean': np.mean(loss_vals) if loss_vals else float('inf'),
                '_acc_mean': np.mean(acc_vals) if acc_vals else 0,
            })

        if not rows:
            print("  No results found")
            return

        # Print table
        print(f"\n  {'Variant':<20} {'Loss':<20} {'Accuracy':<20}")
        print(f"  {'─' * 60}")
        for row in rows:
            print(f"  {row['variant']:<20} {row['loss']:<20} {row['accuracy']:<20}")

        # Print deltas if multiple variants
        if len(rows) >= 2:
            baseline = rows[0]
            print(f"\n  Δ vs {baseline['variant']}:")
            for row in rows[1:]:
                loss_delta = (row['_loss_mean'] - baseline['_loss_mean']) / baseline['_loss_mean'] * 100
                acc_delta = (row['_acc_mean'] - baseline['_acc_mean']) / baseline['_acc_mean'] * 100
                print(f"    {row['variant']}: loss {loss_delta:+.1f}%, acc {acc_delta:+.1f}%")

        print(f"{'═' * 70}\n")

    def compare(self, baseline: str, treatment: str) -> dict:
        """Statistical comparison between two variants."""
        _, _, _, np = _ensure_plotting()

        b_metrics = [self._load_run_metrics(baseline, s) for s in self.seeds]
        t_metrics = [self._load_run_metrics(treatment, s) for s in self.seeds]

        b_loss = [m['valid']['loss'] for m in b_metrics if m and 'valid' in m]
        t_loss = [m['valid']['loss'] for m in t_metrics if m and 'valid' in m]
        b_acc = [m['valid']['answer_token_acc'] for m in b_metrics if m and 'valid' in m]
        t_acc = [m['valid']['answer_token_acc'] for m in t_metrics if m and 'valid' in m]

        result = {
            'baseline': baseline,
            'treatment': treatment,
            'n_seeds': len(self.seeds),
            'loss': {
                'baseline': f"{np.mean(b_loss):.4f} ± {np.std(b_loss):.4f}",
                'treatment': f"{np.mean(t_loss):.4f} ± {np.std(t_loss):.4f}",
                'delta_pct': (np.mean(t_loss) - np.mean(b_loss)) / np.mean(b_loss) * 100,
            },
            'accuracy': {
                'baseline': f"{np.mean(b_acc):.4f} ± {np.std(b_acc):.4f}",
                'treatment': f"{np.mean(t_acc):.4f} ± {np.std(t_acc):.4f}",
                'delta_pct': (np.mean(t_acc) - np.mean(b_acc)) / np.mean(b_acc) * 100,
            },
        }

        print(f"\n{'─' * 60}")
        print(f"  {baseline} vs {treatment} (n={len(self.seeds)})")
        print(f"{'─' * 60}")
        print(f"  Loss:     {result['loss']['baseline']} → {result['loss']['treatment']} ({result['loss']['delta_pct']:+.1f}%)")
        print(f"  Accuracy: {result['accuracy']['baseline']} → {result['accuracy']['treatment']} ({result['accuracy']['delta_pct']:+.1f}%)")
        print(f"{'─' * 60}\n")

        return result

    def get_run(self, variant: str, seed: int = 0) -> PhysicsRun:
        """Get a specific run for detailed analysis."""
        run_path = Path(self.meta['runs'][variant][str(seed)])
        return load_physics(run_path)


def miniseries(
    depths: str = '8,9,10,11,12,13,14,15,16,17,18,19,20',
    output: str | None = None,
    base_config: str = 'configs/speedrun/small_dense_sdpa.toml',
    target_dn: float = 8.0,
    warmup_ratio: float = 0.0,
    warmdown_ratio: float = 0.4,
    dry_run: bool = False,
) -> Path | None:
    """Run the v2 dense miniseries (nanochat #420 method, depth dial).

    This is intentionally dense-first. MoE miniseries variants are out of scope
    here until the dense track is clean and fully understood.
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output = output or f'/data/miniseries/m420v2_dense_{timestamp}'

    cmd = [
        sys.executable, '-m', 'nmoe.research.miniseries',
        '--base-config', base_config,
        '--depths', depths,
        '--target-dn', str(target_dn),
        '--warmup-ratio', str(warmup_ratio),
        '--warmdown-ratio', str(warmdown_ratio),
        '--out-dir', output,
    ]
    if dry_run:
        cmd.append('--dry-run')

    print(f"$ {' '.join(cmd)}")

    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"Miniseries experiment failed with code {result.returncode}")

    return Path(output)


# =============================================================================
# DISCOVERY
# =============================================================================

def info() -> None:
    """Show available tasks and variant syntax for physics experiments.

    Example:
        >>> lab.info()
        === Available Tasks ===
        depo      - Dependency parsing (DEPO)
        brevo     - Multi-hop composition (mHC)
        ...
    """
    print("""
═══════════════════════════════════════════════════════════════════════════════
                         nmoe Physics Harness
═══════════════════════════════════════════════════════════════════════════════

TASKS (use in tasks=['depo:0.5', 'brevo:0.5'])
─────────────────────────────────────────────────────────────────────────────────
  depo           Dependency parsing (PhysicsLM4 DEPO)
  depo_v2        Dependency parsing v2 (improved)
  brevo          Multi-hop composition (mHC) - A→B, B→C ⇒ A→C
  mano           Another compositional task
  lano_cfg       Language modeling with CFG structure
  ngram          N-gram prediction
  ngram_polysemy N-gram with polysemous tokens
  ngram_mixed    Mixed n-gram task
  ngram_scrambled Scrambled n-gram

VARIANTS (use in variants=['baseline', 'canon'])
─────────────────────────────────────────────────────────────────────────────────
  baseline                    residual=vanilla (standard transformer)
  canon                       residual=canon,canon_set=ABCD (Canon layers)
  canon-AB                    residual=canon,canon_set=AB (partial Canon)
  engram                      memory=engram (Engram memory)
  canon+engram                Canon + Engram combined

  Custom: "width=fixed,residual=vanilla,precond=none,memory=none,attn=global"

  residual: vanilla | altup | mhc | canon (shorthand)
  precond:  none | canon
  memory:   none | engram | ple_ngrammer
  attn:     global | local:W | converged:W | mixed:G1Ly:W

EXAMPLE
─────────────────────────────────────────────────────────────────────────────────
  # Canon vs baseline on mHC task
  run = lab.physics(
      output="my_mhc_test",
      tasks=["brevo:1.0"],
      variants=["baseline", "canon"],
      steps=2000,
      n_layers=24,
      dim=256,
  )

  # Then analyze
  lab.quick_compare(
      lab.load_physics("my_mhc_test_baseline"),
      lab.load_physics("my_mhc_test_canon")
  )
═══════════════════════════════════════════════════════════════════════════════
""")


# =============================================================================
# RUN MANAGEMENT
# =============================================================================

def load_physics(path: str | Path) -> PhysicsRun:
    """Load a physics run from disk.

    Can be called with:
      - Full path: load_physics("/data/physics/my_run")
      - Just name: load_physics("my_run")  # looks in /data/physics/
    """
    path = Path(path)
    if not path.exists():
        # Try under default physics dir
        default_path = Path('/data/physics') / path.name
        if default_path.exists():
            path = default_path
        else:
            raise FileNotFoundError(f"Run not found: {path} (also tried {default_path})")

    created = datetime.fromtimestamp(path.stat().st_mtime)

    return PhysicsRun(
        path=path,
        name=path.name,
        created=created,
    )


def list_physics(base: str = '/data/physics') -> list[PhysicsRun]:
    """List all physics runs."""
    base_path = Path(base)
    if not base_path.exists():
        return []

    runs = []
    for d in sorted(base_path.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
        if d.is_dir() and (d / 'analysis' / 'summary.json').exists():
            try:
                runs.append(load_physics(d))
            except Exception:
                pass
    return runs


def latest_physics(base: str = '/data/physics') -> PhysicsRun | None:
    """Get the most recent physics run."""
    runs = list_physics(base)
    return runs[0] if runs else None


# =============================================================================
# EXPERIMENT CATALOG
# =============================================================================

def list_experiments(base: str = '/data/physics') -> list['PhysicsExperiment']:
    """List all multi-seed/multi-variant experiments.

    Returns experiments (directories with experiment.json) sorted by creation time,
    most recent first.

    Example:
        >>> exps = lab.list_experiments()
        >>> for e in exps:
        ...     print(f"{e.name}: {len(e.variants)} variants x {len(e.seeds)} seeds")
    """
    base_path = Path(base)
    if not base_path.exists():
        return []

    experiments = []
    for d in base_path.iterdir():
        exp_json = d / 'experiment.json'
        if d.is_dir() and exp_json.exists():
            try:
                meta = json.loads(exp_json.read_text())
                experiments.append(PhysicsExperiment(path=d, meta=meta))
            except Exception:
                pass

    # Sort by creation time
    experiments.sort(key=lambda e: e.meta.get('created', ''), reverse=True)
    return experiments


def load_experiment(name: str, base: str = '/data/physics') -> 'PhysicsExperiment':
    """Load an experiment by name.

    Args:
        name: Experiment name (directory name under base)
        base: Base directory for physics experiments

    Returns:
        PhysicsExperiment object

    Example:
        >>> exp = lab.load_experiment("my_ablation")
        >>> exp.summary()
    """
    exp_path = Path(base) / name
    exp_json = exp_path / 'experiment.json'

    if not exp_json.exists():
        raise FileNotFoundError(f"Experiment not found: {name} (looked for {exp_json})")

    meta = json.loads(exp_json.read_text())
    return PhysicsExperiment(path=exp_path, meta=meta)


def latest_experiment(base: str = '/data/physics') -> 'PhysicsExperiment | None':
    """Get the most recent experiment."""
    experiments = list_experiments(base)
    return experiments[0] if experiments else None


def catalog(base: str = '/data/physics') -> None:
    """Print a summary catalog of all experiments.

    Shows name, variants, seeds, steps, and creation date for each experiment.

    Example:
        >>> lab.catalog()
        ╭──────────────────────────────────────────────────────────────────────╮
        │  Physics Experiment Catalog                                          │
        ├──────────────────────────────────────────────────────────────────────┤
        │  Name                 Variants    Seeds   Steps   Created           │
        │  test_multiseed       2           2       50      2026-01-23 22:24  │
        │  canon_vs_baseline    3           5       2000    2026-01-22 18:30  │
        ╰──────────────────────────────────────────────────────────────────────╯
    """
    experiments = list_experiments(base)

    if not experiments:
        print("\n  No experiments found in", base)
        return

    print(f"\n{'═' * 75}")
    print(f"  Physics Experiment Catalog ({len(experiments)} experiments)")
    print(f"{'═' * 75}")
    print(f"  {'Name':<25} {'Variants':>10} {'Seeds':>8} {'Steps':>8}   {'Created':<20}")
    print(f"  {'─' * 70}")

    for exp in experiments:
        created = exp.meta.get('created', 'unknown')
        if len(created) > 16:
            created = created[:16].replace('T', ' ')
        print(f"  {exp.name:<25} {len(exp.variants):>10} {len(exp.seeds):>8} {exp.meta.get('steps', '?'):>8}   {created:<20}")

    print(f"{'═' * 75}\n")


# =============================================================================
# COMPARISON
# =============================================================================

def compare(runs: list[PhysicsRun], metric: str = 'loss'):
    """Compare multiple physics runs on a metric."""
    _, pd, _, _ = _ensure_plotting()

    rows = []
    for run in runs:
        data = run._load_summary()
        for variant, metrics in data.items():
            rows.append({
                'run': run.name,
                'variant': _short_variant(variant),
                metric: metrics.get(metric),
            })

    df = pd.DataFrame(rows)
    if df.empty:
        print("No data to compare")
        return df

    pivot = df.pivot(index='variant', columns='run', values=metric)
    return pivot


def quick_compare(baseline: PhysicsRun, treatment: PhysicsRun, split: str = 'valid') -> None:
    """Quick side-by-side comparison of baseline vs treatment.

    The most common research workflow: compare two runs and see the delta.

    Args:
        baseline: The baseline/control run
        treatment: The treatment/experimental run
        split: Which split to compare ('train' or 'valid')

    Example:
        >>> baseline = lab.load_physics("e1_baseline")
        >>> canon = lab.load_physics("e1_canon")
        >>> lab.quick_compare(baseline, canon)
        ╭─────────────────────────────────────────────────────────────╮
        │  Baseline: e1_baseline                                      │
        │  Treatment: e1_canon                                        │
        │─────────────────────────────────────────────────────────────│
        │  Metric          Baseline    Treatment    Δ (%)             │
        │  loss            1.6244      0.9448       -41.8%            │
        │  answer_acc      0.2693      0.5918       +119.7%           │
        ╰─────────────────────────────────────────────────────────────╯
    """
    b_data = baseline._load_summary()
    t_data = treatment._load_summary()

    # Get first variant's metrics for each
    b_metrics = list(b_data.values())[0].get(split, {})
    t_metrics = list(t_data.values())[0].get(split, {})

    print(f"\n{'─' * 65}")
    print(f"  Baseline:  {baseline.name}")
    print(f"  Treatment: {treatment.name}")
    print(f"  Split:     {split}")
    print(f"{'─' * 65}")
    print(f"  {'Metric':<20} {'Baseline':>12} {'Treatment':>12} {'Δ':>10}")
    print(f"{'─' * 65}")

    for key in ['loss', 'answer_token_acc', 'answer_exact_match']:
        if key not in b_metrics or key not in t_metrics:
            continue
        b_val = b_metrics[key]
        t_val = t_metrics[key]

        # Calculate delta (always treatment - baseline, so negative loss = good)
        if b_val != 0:
            delta = (t_val - b_val) / b_val * 100
            delta_str = f"{delta:+.1f}%" if delta != 0 else "0%"
        else:
            delta_str = "N/A"

        # Shorter display name
        display_name = key.replace('answer_token_', '').replace('answer_exact_', 'exact_')
        print(f"  {display_name:<20} {b_val:>12.4f} {t_val:>12.4f} {delta_str:>10}")

    print(f"{'─' * 65}\n")


def compare_curves(runs: list[PhysicsRun], metric: str = 'loss'):
    """Plot loss curves from multiple runs."""
    plt, _, _, _ = _ensure_plotting()

    fig, ax = plt.subplots(figsize=(12, 6))

    color_idx = 0
    for run in runs:
        for variant in run.variants:
            logs = run._load_logs(variant)
            if not logs:
                continue
            steps = [r.get('step', i) for i, r in enumerate(logs)]
            values = [r.get(metric, float('nan')) for r in logs]
            label = f"{run.name[:15]} / {_short_variant(variant)}"
            ax.plot(steps, values, label=label, color=PALETTE[color_idx % len(PALETTE)])
            color_idx += 1

    ax.set_xlabel('Step')
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_title(f'{metric.replace("_", " ").title()} Comparison')
    ax.legend(fontsize=8, loc='upper right')
    return fig


# =============================================================================
# MODEL INSPECTION
# =============================================================================

def load_model(checkpoint_dir: str | Path, device: str = 'cuda'):
    """Load a model from checkpoint directory.

    Args:
        checkpoint_dir: Path to checkpoint (should contain model.pt and config.toml)
        device: Device to load to

    Returns:
        (model, config) tuple
    """
    import tomllib
    from nmoe.config import Config, upgrade_cfg_dict
    from nmoe.model import Transformer

    checkpoint_dir = Path(checkpoint_dir)

    # Find config
    config_path = checkpoint_dir / 'config.toml'
    if not config_path.exists():
        config_path = checkpoint_dir.parent / 'config.toml'
    if not config_path.exists():
        raise FileNotFoundError(f"No config.toml found at {checkpoint_dir}")

    with open(config_path, 'rb') as f:
        cfg = Config(**upgrade_cfg_dict(tomllib.load(f)))

    # Build model
    model = Transformer(cfg).to(device)

    # Load weights
    model_path = checkpoint_dir / 'model.pt'
    if model_path.exists():
        state = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state, strict=False)

    model.eval()
    return model, cfg


def probe_routing(model, x: torch.Tensor) -> pd.DataFrame:
    """Probe routing decisions for a batch.

    Args:
        model: Transformer model
        x: Input tensor [batch, seq_len]

    Returns:
        DataFrame with per-layer routing stats
    """
    _, pd, _, _ = _ensure_plotting()

    stats = []
    hooks = []

    def make_hook(layer_idx):
        def hook(module, input, _output):
            # module is MoE, which has router.gate
            inp = input[0] if isinstance(input, tuple) else input
            with torch.no_grad():
                if hasattr(module, 'router') and hasattr(module.router, 'gate'):
                    gate = module.router.gate
                    n_experts = module.router.n_experts
                    topk = module.router.topk
                    # Flatten batch and seq
                    flat = inp.view(-1, inp.size(-1))
                    scores = torch.sigmoid(gate(flat).float())
                    _, indices = scores.topk(topk, dim=-1)
                    counts = torch.bincount(indices.view(-1), minlength=n_experts).float()
                    stats.append({
                        'layer': layer_idx,
                        'entropy': -(scores * scores.log().nan_to_num()).sum(-1).mean().item(),
                        'max_prob': scores.max(-1).values.mean().item(),
                        'load_cv': (counts.std() / counts.mean()).item() if counts.mean() > 0 else 0,
                        'dead_experts': (counts == 0).sum().item(),
                    })
        return hook

    # Attach hooks to MoE layers
    for i, layer in enumerate(model.layers):
        if hasattr(layer, 'ffn') and hasattr(layer.ffn, 'router'):
            h = layer.ffn.register_forward_hook(make_hook(i))
            hooks.append(h)

    # Forward pass
    with torch.no_grad():
        cos, sin = model.rope(x)
        h = model.embed(x)
        for layer in model.layers:
            h = layer(h, cos, sin)

    # Remove hooks
    for h in hooks:
        h.remove()

    return pd.DataFrame(stats)


def gradient_flow(model):
    """Plot gradient flow through model.

    Call after loss.backward() to visualize gradients.
    """
    plt, _, _, _ = _ensure_plotting()

    norms = []
    names = []

    for name, p in model.named_parameters():
        if p.grad is not None and 'weight' in name:
            norms.append(p.grad.norm().item())
            # Shorten name
            parts = name.replace('.weight', '').split('.')
            names.append('.'.join(parts[-2:]) if len(parts) > 2 else parts[-1])

    if not norms:
        print("No gradients found. Run loss.backward() first.")
        return None

    fig, ax = plt.subplots(figsize=(max(12, len(norms) // 4), 5))
    ax.bar(range(len(norms)), norms, color=PALETTE[0])
    ax.set_xticks(range(len(norms)))
    ax.set_xticklabels(names, rotation=90, fontsize=7)
    ax.set_ylabel('Gradient Norm')
    ax.set_yscale('log')
    ax.set_title('Gradient Flow')
    plt.tight_layout()
    return fig


def memory_stats() -> dict:
    """Get current GPU memory stats."""
    if not torch.cuda.is_available():
        return {}
    return {
        'allocated_gb': torch.cuda.memory_allocated() / 1e9,
        'reserved_gb': torch.cuda.memory_reserved() / 1e9,
        'peak_gb': torch.cuda.max_memory_allocated() / 1e9,
    }
