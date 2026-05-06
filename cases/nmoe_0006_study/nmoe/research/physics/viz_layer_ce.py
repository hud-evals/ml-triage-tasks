"""
Layer-wise CE visualization (zero-dependency).

Reads `analysis/layer_ce_valid.json` produced by `arch_ablations.py` and writes
publication-friendly SVG plots of per-layer cross-entropy to ground-truth.

Key metrics visualized:
  - CE by layer curve: shows where useful computation happens
  - frac_late: fraction of CE reduction in last third of layers
  - layer_contributions: CE drop per layer

Run:
  python -m nmoe.research.physics.viz_layer_ce --runs /tmp/arch_ablations
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _lerp(a: float, b: float, t: float) -> float:
  return a + (b - a) * t


def _clamp(x: float, lo: float, hi: float) -> float:
  return lo if x < lo else hi if x > hi else x


def _rgb_hex(rgb: tuple[int, int, int]) -> str:
  r, g, b = rgb
  return f"#{r:02x}{g:02x}{b:02x}"


def _lerp_rgb(c0: tuple[int, int, int], c1: tuple[int, int, int], t: float) -> tuple[int, int, int]:
  t = _clamp(float(t), 0.0, 1.0)
  return (
    int(round(_lerp(c0[0], c1[0], t))),
    int(round(_lerp(c0[1], c1[1], t))),
    int(round(_lerp(c0[2], c1[2], t))),
  )


def _escape(s: str) -> str:
  return (
    s.replace("&", "&amp;")
    .replace("<", "&lt;")
    .replace(">", "&gt;")
    .replace('"', "&quot;")
    .replace("'", "&apos;")
  )


def _safe_name(s: str) -> str:
  out: list[str] = []
  for ch in s:
    if ch.isalnum() or ch in ("-", "_", "."):
      out.append(ch)
    else:
      out.append("_")
  return "".join(out)


# Color palette for multiple variants (husl-like)
COLORS = [
  "#e41a1c",  # red
  "#377eb8",  # blue
  "#4daf4a",  # green
  "#984ea3",  # purple
  "#ff7f00",  # orange
  "#a65628",  # brown
  "#f781bf",  # pink
  "#999999",  # gray
]


def _write_svg_line_chart(
  *,
  out_path: Path,
  title: str,
  subtitle: str,
  x_labels: list[str],
  y_label: str,
  series: list[tuple[str, list[float], str]],  # (name, values, color)
  y_min: float,
  y_max: float,
) -> None:
  """Write an SVG line chart with multiple series."""
  n_points = len(x_labels)
  if n_points == 0:
    return

  # Chart dimensions
  left = 80
  right = 180
  top = 70
  bottom = 50
  chart_w = 500
  chart_h = 300
  width = left + chart_w + right
  height = top + chart_h + bottom

  parts: list[str] = []
  parts.append('<?xml version="1.0" encoding="UTF-8"?>')
  parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">')
  parts.append('<rect x="0" y="0" width="100%" height="100%" fill="white"/>')

  # Title
  parts.append(f'<text x="{left}" y="26" font-size="16" font-family="sans-serif" font-weight="bold">{_escape(title)}</text>')
  parts.append(f'<text x="{left}" y="46" font-size="11" font-family="sans-serif" fill="#666666">{_escape(subtitle)}</text>')

  # Chart area background
  parts.append(f'<rect x="{left}" y="{top}" width="{chart_w}" height="{chart_h}" fill="#fafafa" stroke="#cccccc" stroke-width="1"/>')

  # Y-axis grid lines and labels
  n_y_ticks = 5
  for i in range(n_y_ticks + 1):
    y_val = y_min + (y_max - y_min) * i / n_y_ticks
    y_pos = top + chart_h - (chart_h * i / n_y_ticks)
    parts.append(f'<line x1="{left}" y1="{y_pos}" x2="{left + chart_w}" y2="{y_pos}" stroke="#e0e0e0" stroke-width="1"/>')
    parts.append(f'<text x="{left - 8}" y="{y_pos + 4}" font-size="10" font-family="sans-serif" text-anchor="end">{y_val:.1f}</text>')

  # Y-axis label
  y_label_x = 20
  y_label_y = top + chart_h / 2
  parts.append(f'<text x="{y_label_x}" y="{y_label_y}" font-size="11" font-family="sans-serif" font-weight="bold" text-anchor="middle" transform="rotate(-90 {y_label_x} {y_label_y})">{_escape(y_label)}</text>')

  # X-axis labels
  for i, label in enumerate(x_labels):
    x_pos = left + (chart_w * i / max(1, n_points - 1))
    parts.append(f'<text x="{x_pos}" y="{top + chart_h + 20}" font-size="10" font-family="sans-serif" text-anchor="middle">{_escape(label)}</text>')

  # Draw lines
  for name, values, color in series:
    points = []
    for i, v in enumerate(values):
      x = left + (chart_w * i / max(1, n_points - 1))
      y = top + chart_h - (chart_h * (v - y_min) / max(0.001, y_max - y_min))
      y = max(top, min(top + chart_h, y))
      points.append(f"{x:.1f},{y:.1f}")

    if len(points) >= 2:
      parts.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="2.5"/>')

    # Draw markers
    for i, v in enumerate(values):
      x = left + (chart_w * i / max(1, n_points - 1))
      y = top + chart_h - (chart_h * (v - y_min) / max(0.001, y_max - y_min))
      y = max(top, min(top + chart_h, y))
      parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}" stroke="white" stroke-width="1.5"/>')

  # Legend
  legend_x = left + chart_w + 15
  legend_y = top + 10
  for i, (name, _, color) in enumerate(series):
    y = legend_y + i * 20
    parts.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 20}" y2="{y}" stroke="{color}" stroke-width="2.5"/>')
    parts.append(f'<circle cx="{legend_x + 10}" cy="{y}" r="3" fill="{color}"/>')
    # Truncate long names
    display_name = name if len(name) <= 25 else name[:22] + "..."
    parts.append(f'<text x="{legend_x + 28}" y="{y + 4}" font-size="9" font-family="sans-serif">{_escape(display_name)}</text>')

  parts.append("</svg>")
  out_path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def _write_svg_bar_chart(
  *,
  out_path: Path,
  title: str,
  subtitle: str,
  labels: list[str],
  values: list[float],
  y_label: str,
  highlight_threshold: float | None = None,
) -> None:
  """Write an SVG bar chart."""
  n_bars = len(labels)
  if n_bars == 0:
    return

  # Chart dimensions
  left = 80
  right = 40
  top = 70
  bottom = 80
  bar_w = 50
  gap = 15
  chart_w = n_bars * bar_w + (n_bars - 1) * gap
  chart_h = 250
  width = left + chart_w + right
  height = top + chart_h + bottom

  y_max = max(values) * 1.1 if values else 1.0
  y_min = 0.0

  parts: list[str] = []
  parts.append('<?xml version="1.0" encoding="UTF-8"?>')
  parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">')
  parts.append('<rect x="0" y="0" width="100%" height="100%" fill="white"/>')

  # Title
  parts.append(f'<text x="{left}" y="26" font-size="16" font-family="sans-serif" font-weight="bold">{_escape(title)}</text>')
  parts.append(f'<text x="{left}" y="46" font-size="11" font-family="sans-serif" fill="#666666">{_escape(subtitle)}</text>')

  # Y-axis
  n_y_ticks = 5
  for i in range(n_y_ticks + 1):
    y_val = y_min + (y_max - y_min) * i / n_y_ticks
    y_pos = top + chart_h - (chart_h * i / n_y_ticks)
    parts.append(f'<line x1="{left}" y1="{y_pos}" x2="{left + chart_w}" y2="{y_pos}" stroke="#e0e0e0" stroke-width="1"/>')
    parts.append(f'<text x="{left - 8}" y="{y_pos + 4}" font-size="10" font-family="sans-serif" text-anchor="end">{y_val:.2f}</text>')

  # Y-axis label
  y_label_x = 20
  y_label_y = top + chart_h / 2
  parts.append(f'<text x="{y_label_x}" y="{y_label_y}" font-size="11" font-family="sans-serif" font-weight="bold" text-anchor="middle" transform="rotate(-90 {y_label_x} {y_label_y})">{_escape(y_label)}</text>')

  # Threshold line
  if highlight_threshold is not None and y_min <= highlight_threshold <= y_max:
    y_thresh = top + chart_h - (chart_h * (highlight_threshold - y_min) / max(0.001, y_max - y_min))
    parts.append(f'<line x1="{left}" y1="{y_thresh}" x2="{left + chart_w}" y2="{y_thresh}" stroke="#e41a1c" stroke-width="1.5" stroke-dasharray="5,3"/>')
    parts.append(f'<text x="{left + chart_w + 5}" y="{y_thresh + 4}" font-size="9" font-family="sans-serif" fill="#e41a1c">1/3</text>')

  # Bars
  for i, (label, val) in enumerate(zip(labels, values)):
    x = left + i * (bar_w + gap)
    bar_h = chart_h * val / max(0.001, y_max - y_min)
    y = top + chart_h - bar_h

    color = COLORS[i % len(COLORS)]
    parts.append(f'<rect x="{x}" y="{y}" width="{bar_w}" height="{bar_h}" fill="{color}" stroke="white" stroke-width="1"/>')

    # Value label
    parts.append(f'<text x="{x + bar_w / 2}" y="{y - 5}" font-size="9" font-family="sans-serif" text-anchor="middle" font-weight="bold">{val:.2f}</text>')

    # X-axis label (rotated for long names)
    display_label = label if len(label) <= 15 else label[:12] + "..."
    lx = x + bar_w / 2
    ly = top + chart_h + 15
    parts.append(f'<text x="{lx}" y="{ly}" font-size="9" font-family="sans-serif" text-anchor="start" transform="rotate(45 {lx} {ly})">{_escape(display_label)}</text>')

  parts.append("</svg>")
  out_path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
  p = argparse.ArgumentParser(description="Render per-layer CE plots as SVG.")
  p.add_argument("--runs", type=Path, required=True, help="Output directory produced by arch_ablations.py")
  p.add_argument("--out", type=Path, default=None, help="Output directory for SVGs (default: <runs>/analysis/viz)")
  p.add_argument("--task", type=str, default=None, help="Only render a specific task")
  p.add_argument("--summary", action="store_true", help="Print frac_late summary table to stdout")
  args = p.parse_args(argv)

  runs_dir = args.runs
  ce_path = runs_dir / "analysis" / "layer_ce_valid.json"
  if not ce_path.is_file():
    raise SystemExit(f"Missing {ce_path}. Run arch_ablations.py with --layer-ce.")

  raw = json.loads(ce_path.read_text(encoding="utf-8"))
  if not isinstance(raw, dict) or not raw:
    raise SystemExit("layer_ce_valid.json is empty")

  variants = sorted(raw.keys())
  if not variants:
    raise SystemExit("layer_ce_valid.json has no variants")

  out_dir = args.out if args.out is not None else (runs_dir / "analysis" / "viz")
  out_dir.mkdir(parents=True, exist_ok=True)

  # Identify tasks from first variant
  any_tasks = raw.get(variants[0]) or {}
  tasks = sorted(any_tasks.keys())
  if args.task is not None:
    tasks = [t for t in tasks if t == args.task]
  if not tasks:
    raise SystemExit("no tasks selected")

  summaries: list[dict] = []

  # Generate per-task CE curve plots
  for task in tasks:
    series = []
    for i, v in enumerate(variants):
      vdata = raw.get(v) or {}
      tdata = vdata.get(task)
      if tdata is None:
        continue
      ce_by_layer = tdata.get("ce_by_layer", [])
      if not ce_by_layer:
        continue
      color = COLORS[i % len(COLORS)]
      series.append((v, ce_by_layer, color))

      # Collect summary
      summaries.append({
        "variant": v,
        "task": task,
        "ce_L0": ce_by_layer[0] if ce_by_layer else 0.0,
        "ce_final": ce_by_layer[-1] if ce_by_layer else 0.0,
        "frac_late": tdata.get("frac_late", 0.0),
      })

    if series:
      all_values = [v for _, vals, _ in series for v in vals]
      y_min = 0.0
      y_max = max(all_values) * 1.05 if all_values else 10.0

      n_layers = max(len(vals) for _, vals, _ in series)
      x_labels = [f"L{i}" for i in range(n_layers)]

      out_path = out_dir / f"layer_ce_{task}.svg"
      _write_svg_line_chart(
        out_path=out_path,
        title=f"Per-Layer CE to Ground Truth ({task})",
        subtitle="Lower is better; steeper descent = more useful layer",
        x_labels=x_labels,
        y_label="Cross-Entropy",
        series=series,
        y_min=y_min,
        y_max=y_max,
      )

  # Generate frac_late comparison bar chart per task
  for task in tasks:
    labels = []
    values = []
    for v in variants:
      vdata = raw.get(v) or {}
      tdata = vdata.get(task)
      if tdata is None:
        continue
      frac_late = tdata.get("frac_late", 0.0)
      labels.append(v)
      values.append(frac_late)

    if labels:
      out_path = out_dir / f"frac_late_{task}.svg"
      _write_svg_bar_chart(
        out_path=out_path,
        title=f"Late-Layer Contribution ({task})",
        subtitle="frac_late = fraction of CE reduction in last 1/3 of layers",
        labels=labels,
        values=values,
        y_label="frac_late",
        highlight_threshold=1.0 / 3.0,  # Expected if work is uniform
      )

  # Print summary table if requested
  if bool(args.summary) and summaries:
    print("\n=== Layer CE Summary ===")
    print(f"{'variant':<50} {'task':<15} {'CE@L0':>8} {'CE@Lf':>8} {'frac_late':>10}")
    print("-" * 95)
    for s in summaries:
      print(f"{s['variant']:<50} {s['task']:<15} {s['ce_L0']:>8.2f} {s['ce_final']:>8.2f} {s['frac_late']:>10.3f}")
    print()

  print(str(out_dir))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
