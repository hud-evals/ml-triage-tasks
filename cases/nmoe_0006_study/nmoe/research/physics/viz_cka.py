"""
CKA visualization (zero-dependency).

Reads `analysis/cka_valid.json` produced by `arch_ablations.py` and writes
publication-friendly SVG heatmaps of layer alignment (Engram-style):

  - Rows: variant layers
  - Cols: baseline layers
  - Values: linear CKA similarity (higher is better)

Run:
  python -m nmoe.research.physics.viz_cka --runs /tmp/arch_ablations
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


def _color_cka(v: float, vmin: float, vmax: float) -> str:
  # White (low) -> dark blue (high).
  if vmax <= vmin:
    t = 0.0
  else:
    t = (float(v) - float(vmin)) / (float(vmax) - float(vmin))
  c_lo = (247, 251, 255)
  c_hi = (8, 48, 107)
  return _rgb_hex(_lerp_rgb(c_lo, c_hi, t))


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


def _write_svg_heatmap(
  *,
  out_path: Path,
  title: str,
  subtitle: str,
  row_labels: list[str],
  col_labels: list[str],
  values: list[list[float]],
  vmin: float,
  vmax: float,
  mark_row_argmax: bool,
) -> None:
  rows = len(row_labels)
  cols = len(col_labels)
  if rows == 0 or cols == 0:
    raise ValueError("empty heatmap")
  if len(values) != rows or any(len(r) != cols for r in values):
    raise ValueError("heatmap shape mismatch")

  cell_w = 22
  cell_h = 22
  left = 260
  top = 70
  right = 40
  bottom = 40
  width = left + cols * cell_w + right
  height = top + rows * cell_h + bottom

  parts: list[str] = []
  parts.append('<?xml version="1.0" encoding="UTF-8"?>')
  parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">')
  parts.append('<rect x="0" y="0" width="100%" height="100%" fill="white"/>')
  parts.append(f'<text x="{left}" y="26" font-size="16" font-family="monospace">{_escape(title)}</text>')
  parts.append(f'<text x="{left}" y="46" font-size="12" font-family="monospace">{_escape(subtitle)}</text>')

  for j, lab in enumerate(col_labels):
    x = left + j * cell_w + cell_w / 2
    parts.append(
      f'<text x="{x:.1f}" y="{top - 10}" font-size="10" font-family="monospace" text-anchor="middle">{_escape(lab)}</text>'
    )

  argmax_js: list[int] = []
  if mark_row_argmax:
    for i in range(rows):
      best_j = 0
      best_v = values[i][0]
      for j in range(1, cols):
        if values[i][j] > best_v:
          best_v = values[i][j]
          best_j = j
      argmax_js.append(best_j)

  for i, rlab in enumerate(row_labels):
    y = top + i * cell_h + cell_h * 0.72
    parts.append(
      f'<text x="{left - 8}" y="{y:.1f}" font-size="10" font-family="monospace" text-anchor="end">{_escape(rlab)}</text>'
    )
    for j in range(cols):
      x = left + j * cell_w
      y0 = top + i * cell_h
      c = _color_cka(values[i][j], vmin=vmin, vmax=vmax)
      parts.append(f'<rect x="{x}" y="{y0}" width="{cell_w}" height="{cell_h}" fill="{c}" stroke="#e0e0e0" stroke-width="1"/>')

    if mark_row_argmax:
      j = argmax_js[i]
      cx = left + j * cell_w + cell_w / 2
      cy = top + i * cell_h + cell_h / 2
      parts.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="3.5" fill="none" stroke="#111111" stroke-width="1.5"/>')

  parts.append("</svg>")
  out_path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def _default_baseline_key(candidates: list[str]) -> str | None:
  target = "width=fixed,residual=vanilla,memory=none,attn=global"
  for k in candidates:
    if k == target:
      return k
  return None


def _compute_depth_shift(cka_matrix: list[list[float]]) -> dict:
  """Compute depth shift metrics from a CKA matrix."""
  n_layers = len(cka_matrix)
  argmax_by_layer = []
  shifts = []

  for i, row in enumerate(cka_matrix):
    j_max = max(range(len(row)), key=lambda j: row[j])
    argmax_by_layer.append(j_max)
    shifts.append(j_max - i)

  third = max(1, n_layers // 3)
  early_shifts = shifts[:third]
  mid_shifts = shifts[third:2 * third]
  late_shifts = shifts[2 * third:]

  return {
    "argmax_by_layer": argmax_by_layer,
    "depth_shift": sum(shifts) / len(shifts) if shifts else 0.0,
    "early_shift": sum(early_shifts) / len(early_shifts) if early_shifts else 0.0,
    "mid_shift": sum(mid_shifts) / len(mid_shifts) if mid_shifts else 0.0,
    "late_shift": sum(late_shifts) / len(late_shifts) if late_shifts else 0.0,
    "max_cka": max(max(row) for row in cka_matrix) if cka_matrix else 0.0,
  }


def main(argv: list[str] | None = None) -> int:
  p = argparse.ArgumentParser(description="Render linear CKA matrices as SVG heatmaps.")
  p.add_argument("--runs", type=Path, required=True, help="Output directory produced by arch_ablations.py")
  p.add_argument("--out", type=Path, default=None, help="Output directory for SVGs (default: <runs>/analysis/viz)")
  p.add_argument("--baseline", type=str, default=None, help="Baseline variant key (default: from cka_valid.json or fixed/vanilla/none/global)")
  p.add_argument("--task", type=str, default=None, help="Only render a specific task")
  p.add_argument("--variant", type=str, default=None, help="Only render a specific variant key")
  p.add_argument("--mark-max", action="store_true", help="Mark argmax baseline layer per variant layer")
  p.add_argument("--summary", action="store_true", help="Print depth_shift summary table to stdout")
  args = p.parse_args(argv)

  runs_dir = args.runs
  cka_path = runs_dir / "analysis" / "cka_valid.json"
  if not cka_path.is_file():
    raise SystemExit(f"Missing {cka_path}. Run arch_ablations.py with --cka.")

  raw = json.loads(cka_path.read_text(encoding="utf-8"))
  if not isinstance(raw, dict) or not raw:
    raise SystemExit("cka_valid.json is empty")

  meta = raw.get("_meta") or {}
  if not isinstance(meta, dict):
    meta = {}

  variants = sorted([k for k in raw.keys() if k != "_meta"])
  if not variants:
    raise SystemExit("cka_valid.json has no variants")

  baseline = args.baseline or meta.get("baseline") or _default_baseline_key(variants)
  if baseline is None:
    baseline = variants[0]

  out_dir = args.out if args.out is not None else (runs_dir / "analysis" / "viz")
  out_dir.mkdir(parents=True, exist_ok=True)

  # Identify tasks from first variant.
  any_tasks = raw.get(variants[0]) or {}
  if not isinstance(any_tasks, dict) or not any_tasks:
    raise SystemExit("cka_valid.json has no tasks")

  tasks = sorted(any_tasks.keys())
  if args.task is not None:
    tasks = [t for t in tasks if t == args.task]
  if not tasks:
    raise SystemExit("no tasks selected")

  selected_variants = variants
  if args.variant is not None:
    if args.variant not in raw:
      raise SystemExit(f"variant {args.variant!r} not present in cka_valid.json")
    selected_variants = [args.variant]

  # Collect depth_shift summaries
  summaries: list[dict] = []

  for v in selected_variants:
    vdata = raw.get(v) or {}
    if not isinstance(vdata, dict):
      continue
    for task in tasks:
      mat = vdata.get(task)
      if mat is None:
        continue
      if not isinstance(mat, list) or not mat or not isinstance(mat[0], list):
        continue
      values = [[float(x) for x in row] for row in mat]
      rows = len(values)
      cols = len(values[0])
      row_labels = [f"L{i}" for i in range(rows)]
      col_labels = [f"L{i}" for i in range(cols)]

      flat = [x for r in values for x in r]
      vmin = min(flat)
      vmax = max(flat)

      name = _safe_name(v)
      out_path = out_dir / f"cka_{task}__{name}.svg"
      _write_svg_heatmap(
        out_path=out_path,
        title=f"Linear CKA layer alignment ({task})",
        subtitle=f"rows={v}  cols={baseline}  min={vmin:.4g}  max={vmax:.4g}  (higher is better)",
        row_labels=row_labels,
        col_labels=col_labels,
        values=values,
        vmin=vmin,
        vmax=vmax,
        mark_row_argmax=bool(args.mark_max),
      )

      # Compute depth_shift summary
      ds = _compute_depth_shift(values)
      summaries.append({
        "variant": v,
        "task": task,
        "depth_shift": ds["depth_shift"],
        "early_shift": ds["early_shift"],
        "mid_shift": ds["mid_shift"],
        "late_shift": ds["late_shift"],
        "max_cka": ds["max_cka"],
        "argmax": ds["argmax_by_layer"],
      })

  # Print summary table if requested
  if bool(args.summary) and summaries:
    print("\n=== CKA Depth Shift Summary ===")
    print(f"{'variant':<50} {'task':<15} {'shift':>8} {'early':>8} {'mid':>8} {'late':>8} {'max_cka':>8}")
    print("-" * 115)
    for s in summaries:
      print(f"{s['variant']:<50} {s['task']:<15} {s['depth_shift']:>8.2f} {s['early_shift']:>8.2f} {s['mid_shift']:>8.2f} {s['late_shift']:>8.2f} {s['max_cka']:>8.3f}")
    print()

  print(str(out_dir))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())

