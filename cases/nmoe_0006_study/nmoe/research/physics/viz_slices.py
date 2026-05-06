"""
Slice-metric visualization (zero-dependency).

Reads `analysis/slices_valid.json` produced by `arch_ablations.py --slice-metrics`
and renders publication-friendly SVGs for difficulty curves:
  - Depo: accuracy vs hops
  - Mano: accuracy vs depth
  - ngram_polysemy: accuracy by mode (A/B)

Run:
  python -m nmoe.research.physics.viz_slices --runs /tmp/arch_ablations
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


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


def _clamp01(x: float) -> float:
  return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def _lerp(a: float, b: float, t: float) -> float:
  return a + (b - a) * t


def _rgb(r: float, g: float, b: float) -> str:
  r = int(max(0, min(255, round(r))))
  g = int(max(0, min(255, round(g))))
  b = int(max(0, min(255, round(b))))
  return f"#{r:02x}{g:02x}{b:02x}"


def _heat_color(v: float) -> str:
  # White -> deep blue (0..1), publication-friendly.
  v = _clamp01(float(v))
  r0, g0, b0 = 255.0, 255.0, 255.0
  r1, g1, b1 = 8.0, 48.0, 107.0  # #08306b
  return _rgb(_lerp(r0, r1, v), _lerp(g0, g1, v), _lerp(b0, b1, v))


def _write_svg_heatmap(
  *,
  out_path: Path,
  title: str,
  subtitle: str,
  row_labels: list[str],
  col_labels: list[str],
  values: list[list[float]],  # rows x cols, assumed 0..1
) -> None:
  if not row_labels or not col_labels:
    return
  if len(values) != len(row_labels) or any(len(r) != len(col_labels) for r in values):
    raise ValueError("heatmap values must be rows x cols aligned to labels")

  left = 260
  right = 90
  top = 70
  bottom = 40
  cell = 18
  grid_w = cell * len(col_labels)
  grid_h = cell * len(row_labels)
  width = left + grid_w + right
  height = top + grid_h + bottom

  parts: list[str] = []
  parts.append('<?xml version="1.0" encoding="UTF-8"?>')
  parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">')
  parts.append('<rect x="0" y="0" width="100%" height="100%" fill="white"/>')
  parts.append(f'<text x="{left}" y="26" font-size="16" font-family="sans-serif" font-weight="bold">{_escape(title)}</text>')
  parts.append(f'<text x="{left}" y="46" font-size="11" font-family="sans-serif" fill="#666666">{_escape(subtitle)}</text>')

  # Column labels.
  for j, lab in enumerate(col_labels):
    x = left + j * cell + cell / 2
    parts.append(f'<text x="{x:.1f}" y="{top - 8}" font-size="10" font-family="sans-serif" text-anchor="middle">{_escape(lab)}</text>')

  # Rows.
  for i, rlab in enumerate(row_labels):
    y = top + i * cell
    # Row label (truncate for sanity).
    disp = rlab if len(rlab) <= 42 else rlab[:39] + "..."
    parts.append(f'<text x="{left - 8}" y="{y + cell*0.7:.1f}" font-size="9" font-family="sans-serif" text-anchor="end">{_escape(disp)}</text>')
    for j, v in enumerate(values[i]):
      x = left + j * cell
      parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{cell:.1f}" height="{cell:.1f}" fill="{_heat_color(float(v))}" stroke="#dddddd" stroke-width="0.5"/>')

  # Legend.
  leg_x = left + grid_w + 25
  leg_y = top
  leg_h = 120
  leg_w = 16
  # Gradient steps.
  steps = 40
  for s in range(steps):
    t0 = s / steps
    y0 = leg_y + leg_h * t0
    parts.append(f'<rect x="{leg_x}" y="{y0:.1f}" width="{leg_w}" height="{leg_h/steps:.1f}" fill="{_heat_color(1.0 - t0)}" stroke="none"/>')
  parts.append(f'<rect x="{leg_x}" y="{leg_y}" width="{leg_w}" height="{leg_h}" fill="none" stroke="#888888" stroke-width="1"/>')
  parts.append(f'<text x="{leg_x + leg_w + 8}" y="{leg_y + 9}" font-size="9" font-family="sans-serif">1.0</text>')
  parts.append(f'<text x="{leg_x + leg_w + 8}" y="{leg_y + leg_h}" font-size="9" font-family="sans-serif">0.0</text>')
  parts.append(f'<text x="{leg_x}" y="{leg_y + leg_h + 22}" font-size="9" font-family="sans-serif" fill="#333333">gate value</text>')

  parts.append("</svg>")
  out_path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def _write_svg_line_chart(
  *,
  out_path: Path,
  title: str,
  subtitle: str,
  x_vals: list[int],
  x_label: str,
  y_label: str,
  series: list[tuple[str, list[float], str]],  # (name, y_values, color)
  y_min: float,
  y_max: float,
) -> None:
  if not x_vals or not series:
    return

  left = 80
  right = 200
  top = 70
  bottom = 60
  chart_w = 520
  chart_h = 300
  width = left + chart_w + right
  height = top + chart_h + bottom

  parts: list[str] = []
  parts.append('<?xml version="1.0" encoding="UTF-8"?>')
  parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">')
  parts.append('<rect x="0" y="0" width="100%" height="100%" fill="white"/>')
  parts.append(f'<text x="{left}" y="26" font-size="16" font-family="sans-serif" font-weight="bold">{_escape(title)}</text>')
  parts.append(f'<text x="{left}" y="46" font-size="11" font-family="sans-serif" fill="#666666">{_escape(subtitle)}</text>')
  parts.append(f'<rect x="{left}" y="{top}" width="{chart_w}" height="{chart_h}" fill="#fafafa" stroke="#cccccc" stroke-width="1"/>')

  # X coords for integer x values.
  x_min = min(x_vals)
  x_max = max(x_vals)
  def x_pos(x: int) -> float:
    if x_max == x_min:
      return float(left + chart_w / 2)
    return float(left + chart_w * (x - x_min) / (x_max - x_min))

  # Y axis.
  n_y_ticks = 5
  for i in range(n_y_ticks + 1):
    y_val = y_min + (y_max - y_min) * i / n_y_ticks
    y = top + chart_h - (chart_h * i / n_y_ticks)
    parts.append(f'<line x1="{left}" y1="{y}" x2="{left + chart_w}" y2="{y}" stroke="#e0e0e0" stroke-width="1"/>')
    parts.append(f'<text x="{left - 8}" y="{y + 4}" font-size="10" font-family="sans-serif" text-anchor="end">{y_val:.2f}</text>')

  # X ticks.
  for xv in x_vals:
    x = x_pos(xv)
    parts.append(f'<text x="{x:.1f}" y="{top + chart_h + 20}" font-size="10" font-family="sans-serif" text-anchor="middle">{xv}</text>')

  # Axis labels.
  parts.append(f'<text x="{left + chart_w/2:.1f}" y="{top + chart_h + 45}" font-size="11" font-family="sans-serif" font-weight="bold" text-anchor="middle">{_escape(x_label)}</text>')
  y_label_x = 20
  y_label_y = top + chart_h / 2
  parts.append(f'<text x="{y_label_x}" y="{y_label_y}" font-size="11" font-family="sans-serif" font-weight="bold" text-anchor="middle" transform="rotate(-90 {y_label_x} {y_label_y})">{_escape(y_label)}</text>')

  def y_pos(v: float) -> float:
    if y_max <= y_min:
      return float(top + chart_h)
    t = (v - y_min) / max(1e-12, (y_max - y_min))
    t = 0.0 if t < 0.0 else 1.0 if t > 1.0 else t
    return float(top + chart_h - chart_h * t)

  for name, ys, color in series:
    pts = []
    for xv, yv in zip(x_vals, ys):
      pts.append(f"{x_pos(xv):.1f},{y_pos(float(yv)):.1f}")
    if len(pts) >= 2:
      parts.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{color}" stroke-width="2.5"/>')
    for xv, yv in zip(x_vals, ys):
      parts.append(f'<circle cx="{x_pos(xv):.1f}" cy="{y_pos(float(yv)):.1f}" r="4" fill="{color}" stroke="white" stroke-width="1.5"/>')

  # Legend.
  legend_x = left + chart_w + 15
  legend_y = top + 10
  for i, (name, _, color) in enumerate(series):
    y = legend_y + i * 20
    parts.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 20}" y2="{y}" stroke="{color}" stroke-width="2.5"/>')
    parts.append(f'<circle cx="{legend_x + 10}" cy="{y}" r="3" fill="{color}"/>')
    display_name = name if len(name) <= 28 else name[:25] + "..."
    parts.append(f'<text x="{legend_x + 28}" y="{y + 4}" font-size="9" font-family="sans-serif">{_escape(display_name)}</text>')

  parts.append("</svg>")
  out_path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def _write_svg_bar_chart(
  *,
  out_path: Path,
  title: str,
  subtitle: str,
  labels: list[str],
  values_by_variant: list[tuple[str, list[float]]],  # (variant, values)
  y_label: str,
) -> None:
  if not labels or not values_by_variant:
    return

  left = 90
  right = 200
  top = 70
  bottom = 80
  group_w = 60
  gap = 18
  chart_w = len(labels) * group_w + (len(labels) - 1) * gap
  chart_h = 250
  width = left + chart_w + right
  height = top + chart_h + bottom

  all_vals = [v for _, vs in values_by_variant for v in vs]
  y_max = max(all_vals) * 1.1 if all_vals else 1.0
  y_min = 0.0

  parts: list[str] = []
  parts.append('<?xml version="1.0" encoding="UTF-8"?>')
  parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">')
  parts.append('<rect x="0" y="0" width="100%" height="100%" fill="white"/>')
  parts.append(f'<text x="{left}" y="26" font-size="16" font-family="sans-serif" font-weight="bold">{_escape(title)}</text>')
  parts.append(f'<text x="{left}" y="46" font-size="11" font-family="sans-serif" fill="#666666">{_escape(subtitle)}</text>')
  parts.append(f'<rect x="{left}" y="{top}" width="{chart_w}" height="{chart_h}" fill="#fafafa" stroke="#cccccc" stroke-width="1"/>')

  # Y ticks.
  n_y_ticks = 5
  for i in range(n_y_ticks + 1):
    y_val = y_min + (y_max - y_min) * i / n_y_ticks
    y = top + chart_h - (chart_h * i / n_y_ticks)
    parts.append(f'<line x1="{left}" y1="{y}" x2="{left + chart_w}" y2="{y}" stroke="#e0e0e0" stroke-width="1"/>')
    parts.append(f'<text x="{left - 8}" y="{y + 4}" font-size="10" font-family="sans-serif" text-anchor="end">{y_val:.2f}</text>')

  # Y label.
  y_label_x = 20
  y_label_y = top + chart_h / 2
  parts.append(f'<text x="{y_label_x}" y="{y_label_y}" font-size="11" font-family="sans-serif" font-weight="bold" text-anchor="middle" transform="rotate(-90 {y_label_x} {y_label_y})">{_escape(y_label)}</text>')

  def y_pos(v: float) -> float:
    if y_max <= y_min:
      return float(top + chart_h)
    t = (v - y_min) / max(1e-12, (y_max - y_min))
    t = 0.0 if t < 0.0 else 1.0 if t > 1.0 else t
    return float(top + chart_h - chart_h * t)

  # Bars: grouped by label.
  n_series = len(values_by_variant)
  bar_w = max(6, int((group_w - 10) / max(1, n_series)))
  for li, lab in enumerate(labels):
    x0 = left + li * (group_w + gap)
    parts.append(f'<text x="{x0 + group_w/2:.1f}" y="{top + chart_h + 20}" font-size="10" font-family="sans-serif" text-anchor="middle">{_escape(lab)}</text>')
    for si, (variant, values) in enumerate(values_by_variant):
      v = float(values[li])
      x = x0 + 5 + si * bar_w
      h = chart_h * v / max(1e-12, y_max - y_min)
      y = top + chart_h - h
      color = COLORS[si % len(COLORS)]
      parts.append(f'<rect x="{x}" y="{y:.1f}" width="{bar_w-1}" height="{h:.1f}" fill="{color}" stroke="white" stroke-width="1"/>')

  # Legend.
  legend_x = left + chart_w + 15
  legend_y = top + 10
  for i, (variant, _) in enumerate(values_by_variant):
    y = legend_y + i * 20
    color = COLORS[i % len(COLORS)]
    parts.append(f'<rect x="{legend_x}" y="{y-7}" width="12" height="12" fill="{color}"/>')
    display_name = variant if len(variant) <= 28 else variant[:25] + "..."
    parts.append(f'<text x="{legend_x + 18}" y="{y + 4}" font-size="9" font-family="sans-serif">{_escape(display_name)}</text>')

  parts.append("</svg>")
  out_path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def _parse_int_suffix(key: str, prefix: str) -> int | None:
  if not key.startswith(prefix):
    return None
  try:
    return int(key[len(prefix):])
  except ValueError:
    return None


def main(argv: list[str] | None = None) -> int:
  p = argparse.ArgumentParser(description="Render slice-metric difficulty curves as SVG.")
  p.add_argument("--runs", type=Path, required=True, help="Output directory produced by arch_ablations.py")
  p.add_argument("--out", type=Path, default=None, help="Output directory for SVGs (default: <runs>/analysis/viz)")
  p.add_argument("--metric", type=str, default="answer_exact_match", choices=["answer_exact_match", "answer_token_acc"])
  p.add_argument("--task", type=str, default=None, help="Only render a specific task")
  args = p.parse_args(argv)

  runs_dir = args.runs
  path = runs_dir / "analysis" / "slices_valid.json"
  if not path.is_file():
    raise SystemExit(f"Missing {path}. Run arch_ablations.py with --slice-metrics.")

  raw = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(raw, dict) or not raw:
    raise SystemExit("slices_valid.json is empty")

  out_dir = args.out if args.out is not None else (runs_dir / "analysis" / "viz")
  out_dir.mkdir(parents=True, exist_ok=True)

  variants = sorted(raw.keys())
  tasks = sorted({t for v in variants for t in (raw.get(v) or {}).keys()})
  if args.task is not None:
    tasks = [t for t in tasks if t == args.task]
  if not tasks:
    raise SystemExit("no tasks selected")

  for task in tasks:
    # Collect all slice keys for this task.
    slice_keys = sorted({k for v in variants for k in ((raw.get(v) or {}).get(task) or {}).keys()})
    if not slice_keys:
      continue

    # Numeric-x charts (hops/depth).
    hops = [_parse_int_suffix(k, "hops=") for k in slice_keys]
    depths = [_parse_int_suffix(k, "depth=") for k in slice_keys]
    if any(h is not None for h in hops):
      xs = sorted({h for h in hops if h is not None})
      series = []
      for i, v in enumerate(variants):
        vtask = (raw.get(v) or {}).get(task) or {}
        ys = []
        for h in xs:
          rec = vtask.get(f"hops={h}") or {}
          ys.append(float(rec.get(args.metric, 0.0)))
        series.append((v, ys, COLORS[i % len(COLORS)]))
      _write_svg_line_chart(
        out_path=out_dir / f"slices_{_safe_name(task)}_hops_{args.metric}.svg",
        title=f"{task}: {args.metric} vs hops",
        subtitle="Difficulty slice curve (Depo-style)",
        x_vals=xs,
        x_label="hops",
        y_label=args.metric,
        series=series,
        y_min=0.0,
        y_max=1.0,
      )
      continue

    if any(d is not None for d in depths):
      xs = sorted({d for d in depths if d is not None})
      series = []
      for i, v in enumerate(variants):
        vtask = (raw.get(v) or {}).get(task) or {}
        ys = []
        for d in xs:
          rec = vtask.get(f"depth={d}") or {}
          ys.append(float(rec.get(args.metric, 0.0)))
        series.append((v, ys, COLORS[i % len(COLORS)]))
      _write_svg_line_chart(
        out_path=out_dir / f"slices_{_safe_name(task)}_depth_{args.metric}.svg",
        title=f"{task}: {args.metric} vs depth",
        subtitle="Difficulty slice curve (Mano-style)",
        x_vals=xs,
        x_label="depth",
        y_label=args.metric,
        series=series,
        y_min=0.0,
        y_max=1.0,
      )
      continue

    # Categorical chart (mode=A/B).
    modes = [k for k in slice_keys if k.startswith("mode=")]
    if modes:
      labels = sorted(modes)
      values_by_variant = []
      for v in variants:
        vtask = (raw.get(v) or {}).get(task) or {}
        vals = [float((vtask.get(m) or {}).get(args.metric, 0.0)) for m in labels]
        values_by_variant.append((v, vals))
      _write_svg_bar_chart(
        out_path=out_dir / f"slices_{_safe_name(task)}_mode_{args.metric}.svg",
        title=f"{task}: {args.metric} by mode",
        subtitle="Polysemy slice bars (ngram_polysemy-style)",
        labels=labels,
        values_by_variant=values_by_variant,
        y_label=args.metric,
      )

  # Stack-gate heatmap(s) (if present in slices).
  alpha_order = ["alphaA", "alphaB", "alphaC", "alphaD", "alphaE", "alphaH"]
  for v in variants:
    # Determine which α-columns are present for this variant (schema may evolve).
    present: list[str] = []
    for k in alpha_order:
      found = False
      for task in tasks:
        vtask = (raw.get(v) or {}).get(task) or {}
        for _sk, rec in vtask.items():
          sg = (rec or {}).get("stack_gate") or None
          if sg and k in sg:
            found = True
            break
        if found:
          break
      if found:
        present.append(k)

    if not present:
      continue

    rows: list[str] = []
    vals: list[list[float]] = []
    for task in tasks:
      vtask = (raw.get(v) or {}).get(task) or {}
      for sk, rec in sorted(vtask.items(), key=lambda kv: kv[0]):
        sg = (rec or {}).get("stack_gate") or None
        if not sg:
          continue
        rows.append(f"{task}:{sk}")
        vals.append([float(sg.get(k, 0.0)) for k in present])
    if rows:
      _write_svg_heatmap(
        out_path=out_dir / f"stack_gate_heatmap_{_safe_name(v)}.svg",
        title="Stack gate heatmap",
        subtitle=f"{v} (rows=task:slice, cols={','.join(present)})",
        row_labels=rows,
        col_labels=present,
        values=vals,
      )

  # Memory gate heatmap(s) (Engram-style layerwise means).
  for v in variants:
    n_layers = None
    for task in tasks:
      vtask = (raw.get(v) or {}).get(task) or {}
      for _sk, rec in vtask.items():
        g = (rec or {}).get("gate_mean_by_layer")
        if isinstance(g, list) and g:
          n_layers = len(g)
          break
      if n_layers is not None:
        break
    if n_layers is None:
      continue

    rows: list[str] = []
    vals: list[list[float]] = []
    for task in tasks:
      vtask = (raw.get(v) or {}).get(task) or {}
      for sk, rec in sorted(vtask.items(), key=lambda kv: kv[0]):
        g = (rec or {}).get("gate_mean_by_layer")
        if not isinstance(g, list) or len(g) != n_layers:
          continue
        rows.append(f"{task}:{sk}")
        vals.append([float(x) for x in g])
    if rows:
      _write_svg_heatmap(
        out_path=out_dir / f"mem_gate_heatmap_{_safe_name(v)}.svg",
        title="Memory gate heatmap",
        subtitle=f"{v} (rows=task:slice, cols=layers)",
        row_labels=rows,
        col_labels=[f"L{i}" for i in range(n_layers)],
        values=vals,
      )

  # Per-slice layer-CE curves (LogitScope-style): CE(unembed(h_l), y) by layer.
  # This is the key diagnostic for distinguishing "early correct then overwritten late"
  # from "never gets the right signal" on a specific slice (e.g., ngram_polysemy/mode=B).
  for v in variants:
    for task in tasks:
      vtask = (raw.get(v) or {}).get(task) or {}
      # Require at least two slices to make the plot informative.
      entries: list[tuple[str, list[float]]] = []
      n_layers = None
      for sk, rec in sorted(vtask.items(), key=lambda kv: kv[0]):
        lc = (rec or {}).get("layer_ce") or None
        if not isinstance(lc, dict):
          continue
        ce = lc.get("ce_by_layer")
        if not isinstance(ce, list) or not ce:
          continue
        if n_layers is None:
          n_layers = len(ce)
        if len(ce) != n_layers:
          continue
        entries.append((sk, [float(x) for x in ce]))

      if n_layers is None or len(entries) < 2:
        continue

      x_vals = list(range(int(n_layers)))
      y_min = min(min(ys) for _sk, ys in entries)
      y_max = max(max(ys) for _sk, ys in entries)
      pad = 0.05 * (y_max - y_min) if y_max > y_min else 0.5
      y_min -= pad
      y_max += pad

      series = []
      for i, (sk, ys) in enumerate(entries):
        series.append((sk, ys, COLORS[i % len(COLORS)]))

      _write_svg_line_chart(
        out_path=out_dir / f"slices_layer_ce_{_safe_name(task)}_{_safe_name(v)}.svg",
        title=f"{task}: per-layer CE by slice",
        subtitle=f"{v} (CE to ground-truth; lower is better)",
        x_vals=x_vals,
        x_label="layer",
        y_label="CE",
        series=series,
        y_min=float(y_min),
        y_max=float(y_max),
      )

  # Converged-attention heatmaps (if present).
  for v in variants:
    n_layers = None
    for task in tasks:
      vtask = (raw.get(v) or {}).get(task) or {}
      for _sk, rec in vtask.items():
        g = (rec or {}).get("attn_local_gate_mean_by_layer")
        if isinstance(g, list) and g:
          n_layers = len(g)
          break
      if n_layers is not None:
        break
    if n_layers is None:
      continue

    # Local-vs-global router (local fraction).
    rows: list[str] = []
    vals: list[list[float]] = []
    for task in tasks:
      vtask = (raw.get(v) or {}).get(task) or {}
      for sk, rec in sorted(vtask.items(), key=lambda kv: kv[0]):
        g = (rec or {}).get("attn_local_gate_mean_by_layer")
        if not isinstance(g, list) or len(g) != n_layers:
          continue
        rows.append(f"{task}:{sk}")
        vals.append([float(x) for x in g])
    if rows:
      _write_svg_heatmap(
        out_path=out_dir / f"attn_converged_local_gate_heatmap_{_safe_name(v)}.svg",
        title="Converged attention: local gate heatmap",
        subtitle=f"{v} (rows=task:slice, cols=layers; value=local fraction)",
        row_labels=rows,
        col_labels=[f"L{i}" for i in range(n_layers)],
        values=vals,
      )

    # Learned window fraction (w/T).
    rows = []
    vals = []
    for task in tasks:
      vtask = (raw.get(v) or {}).get(task) or {}
      for sk, rec in sorted(vtask.items(), key=lambda kv: kv[0]):
        g = (rec or {}).get("attn_window_frac_mean_by_layer")
        if not isinstance(g, list) or len(g) != n_layers:
          continue
        rows.append(f"{task}:{sk}")
        vals.append([float(x) for x in g])
    if rows:
      _write_svg_heatmap(
        out_path=out_dir / f"attn_converged_window_frac_heatmap_{_safe_name(v)}.svg",
        title="Converged attention: window fraction heatmap",
        subtitle=f"{v} (rows=task:slice, cols=layers; value=window/T)",
        row_labels=rows,
        col_labels=[f"L{i}" for i in range(n_layers)],
        values=vals,
      )

  print(str(out_dir))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
