"""
LogitLens visualization (zero-dependency).

Reads `analysis/logitlens_valid.json` produced by `arch_ablations.py` and writes
publication-friendly SVG heatmaps:
  - KL(p_final || p_layer) by layer (lower is better)
  - Delta vs baseline by layer (negative is better)

Run:
  python -m nmoe.research.physics.viz_logitlens --runs /tmp/arch_ablations
"""

from __future__ import annotations

import argparse
import json
import math
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


def _color_kl(v: float, vmin: float, vmax: float) -> str:
  # Dark blue (good/low) -> white (bad/high).
  if vmax <= vmin:
    t = 0.0
  else:
    t = (float(v) - float(vmin)) / (float(vmax) - float(vmin))
  c_good = (8, 48, 107)
  c_bad = (247, 251, 255)
  return _rgb_hex(_lerp_rgb(c_good, c_bad, t))


def _color_delta(d: float, maxabs: float) -> str:
  # Diverging: blue (negative/better) <-> white <-> red (positive/worse)
  if maxabs <= 0:
    t = 0.0
  else:
    t = float(d) / float(maxabs)
  t = _clamp(t, -1.0, 1.0)
  white = (247, 247, 247)
  blue = (33, 102, 172)
  red = (178, 24, 43)
  if t < 0:
    return _rgb_hex(_lerp_rgb(white, blue, abs(t)))
  return _rgb_hex(_lerp_rgb(white, red, t))


def _escape(s: str) -> str:
  return (
    s.replace("&", "&amp;")
    .replace("<", "&lt;")
    .replace(">", "&gt;")
    .replace('"', "&quot;")
    .replace("'", "&apos;")
  )


def _write_svg_heatmap(
  *,
  out_path: Path,
  title: str,
  row_labels: list[str],
  col_labels: list[str],
  values: list[list[float]],
  color_fn,
  legend: str,
) -> None:
  rows = len(row_labels)
  cols = len(col_labels)
  if rows == 0 or cols == 0:
    raise ValueError("empty heatmap")
  if len(values) != rows or any(len(r) != cols for r in values):
    raise ValueError("heatmap shape mismatch")

  cell_w = 26
  cell_h = 18
  left = 260
  top = 60
  right = 40
  bottom = 40
  width = left + cols * cell_w + right
  height = top + rows * cell_h + bottom

  parts: list[str] = []
  parts.append('<?xml version="1.0" encoding="UTF-8"?>')
  parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">')
  parts.append('<rect x="0" y="0" width="100%" height="100%" fill="white"/>')
  parts.append(f'<text x="{left}" y="26" font-size="16" font-family="monospace">{_escape(title)}</text>')
  parts.append(f'<text x="{left}" y="46" font-size="12" font-family="monospace">{_escape(legend)}</text>')

  # Column labels
  for j, lab in enumerate(col_labels):
    x = left + j * cell_w + cell_w / 2
    parts.append(
      f'<text x="{x:.1f}" y="{top - 8}" font-size="10" font-family="monospace" text-anchor="middle">{_escape(lab)}</text>'
    )

  # Rows + cells
  for i, rlab in enumerate(row_labels):
    y = top + i * cell_h + cell_h * 0.72
    parts.append(
      f'<text x="{left - 8}" y="{y:.1f}" font-size="10" font-family="monospace" text-anchor="end">{_escape(rlab)}</text>'
    )
    for j in range(cols):
      x = left + j * cell_w
      y0 = top + i * cell_h
      c = color_fn(values[i][j])
      parts.append(f'<rect x="{x}" y="{y0}" width="{cell_w}" height="{cell_h}" fill="{c}" stroke="#e0e0e0" stroke-width="1"/>')

  parts.append("</svg>")
  out_path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def _baseline_key(candidates: list[str]) -> str | None:
  target = "width=fixed,residual=vanilla,memory=none,attn=global"
  for k in candidates:
    if k == target:
      return k
  return None


def main(argv: list[str] | None = None) -> int:
  p = argparse.ArgumentParser(description="Render LogitLens KL curves as SVG heatmaps.")
  p.add_argument("--runs", type=Path, required=True, help="Output directory produced by arch_ablations.py")
  p.add_argument("--out", type=Path, default=None, help="Output directory for SVGs (default: <runs>/analysis/viz)")
  p.add_argument("--baseline", type=str, default=None, help="Baseline variant key (default: fixed/vanilla/none/global)")
  args = p.parse_args(argv)

  runs_dir = args.runs
  log_path = runs_dir / "analysis" / "logitlens_valid.json"
  if not log_path.is_file():
    raise SystemExit(f"Missing {log_path}. Run arch_ablations.py with --logitlens.")

  data = json.loads(log_path.read_text(encoding="utf-8"))
  if not isinstance(data, dict) or not data:
    raise SystemExit("logitlens_valid.json is empty")

  out_dir = args.out if args.out is not None else (runs_dir / "analysis" / "viz")
  out_dir.mkdir(parents=True, exist_ok=True)

  variants = sorted(data.keys())
  baseline = args.baseline or _baseline_key(variants) or variants[0]
  if baseline not in data:
    raise SystemExit(f"Baseline {baseline!r} not present in logitlens data")

  # Tasks present in (baseline) data define the set we render.
  tasks = sorted((data.get(baseline) or {}).keys())
  if not tasks:
    raise SystemExit(f"Baseline {baseline!r} has no tasks logged")

  for task in tasks:
    # Build KL matrix [variant, layer]
    curves: dict[str, list[float]] = {}
    for v in variants:
      td = data.get(v) or {}
      if task in td:
        curves[v] = td[task]

    if baseline not in curves:
      continue

    n_layers = len(curves[baseline])
    col_labels = [f"L{l}" for l in range(n_layers)]

    # KL heatmap
    rows = sorted(curves.keys())
    mat = [list(map(float, curves[v])) for v in rows]
    flat = [x for r in mat for x in r]
    vmin = min(flat)
    vmax = max(flat)

    def color_fn_kl(x: float) -> str:
      return _color_kl(x, vmin, vmax)

    _write_svg_heatmap(
      out_path=out_dir / f"logitlens_{task}.svg",
      title=f"LogitLens KL-to-final ({task})",
      row_labels=rows,
      col_labels=col_labels,
      values=mat,
      color_fn=color_fn_kl,
      legend=f"KL(p_final || p_layer)  min={vmin:.4g}  max={vmax:.4g}  (lower is better)",
    )

    # Delta heatmap vs baseline
    base = curves[baseline]
    dmat = [[float(curves[v][l]) - float(base[l]) for l in range(n_layers)] for v in rows]
    dflat = [x for r in dmat for x in r]
    maxabs = max(1e-12, max(abs(x) for x in dflat))

    def color_fn_delta(x: float) -> str:
      return _color_delta(x, maxabs=maxabs)

    _write_svg_heatmap(
      out_path=out_dir / f"logitlens_{task}_delta.svg",
      title=f"LogitLens ΔKL vs baseline ({task})",
      row_labels=rows,
      col_labels=col_labels,
      values=dmat,
      color_fn=color_fn_delta,
      legend=f"ΔKL = KL_variant - KL_baseline, baseline={baseline}  maxabs={maxabs:.4g}  (negative is better)",
    )

  print(str(out_dir))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())

