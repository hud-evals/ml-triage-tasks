"""
mHC reproduction visualization (zero-dependency).

Reads artifacts produced by `mhc_repro.py` and writes paper-friendly SVG plots:
  - loss vs step
  - grad_norm vs step
  - composite gain (max over layers) vs step (log10)

Run:
  python -m nmoe.research.physics.viz_mhc --input /tmp/mhc_repro
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


COLORS = {
  "baseline": "#377eb8",  # blue
  "hc": "#e41a1c",        # red
  "mhc": "#4daf4a",       # green
}


def _escape(s: str) -> str:
  return (
    s.replace("&", "&amp;")
    .replace("<", "&lt;")
    .replace(">", "&gt;")
    .replace('"', "&quot;")
    .replace("'", "&apos;")
  )


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

  x_min = min(x_vals)
  x_max = max(x_vals)
  def x_pos(x: int) -> float:
    if x_max == x_min:
      return float(left + chart_w / 2)
    return float(left + chart_w * (x - x_min) / (x_max - x_min))

  # Y ticks.
  n_y_ticks = 5
  for i in range(n_y_ticks + 1):
    y_val = y_min + (y_max - y_min) * i / n_y_ticks
    y = top + chart_h - (chart_h * i / n_y_ticks)
    parts.append(f'<line x1="{left}" y1="{y}" x2="{left + chart_w}" y2="{y}" stroke="#e0e0e0" stroke-width="1"/>')
    parts.append(f'<text x="{left - 8}" y="{y + 4}" font-size="10" font-family="sans-serif" text-anchor="end">{y_val:.2f}</text>')

  # X ticks (sparse).
  for xv in x_vals[:: max(1, len(x_vals) // 8)]:
    x = x_pos(xv)
    parts.append(f'<text x="{x:.1f}" y="{top + chart_h + 20}" font-size="10" font-family="sans-serif" text-anchor="middle">{xv}</text>')

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

  # Legend.
  legend_x = left + chart_w + 15
  legend_y = top + 10
  for i, (name, _, color) in enumerate(series):
    y = legend_y + i * 20
    parts.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 20}" y2="{y}" stroke="{color}" stroke-width="2.5"/>')
    parts.append(f'<text x="{legend_x + 28}" y="{y + 4}" font-size="10" font-family="sans-serif">{_escape(name)}</text>')

  parts.append("</svg>")
  out_path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict]:
  rows: list[dict] = []
  for line in path.read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if not line:
      continue
    rows.append(json.loads(line))
  return rows


def _max_list(x) -> float:
  if not isinstance(x, list) or not x:
    return float("nan")
  return float(max(float(v) for v in x))


def main(argv: list[str] | None = None) -> int:
  p = argparse.ArgumentParser(description="Render mHC reproduction plots as SVG.")
  p.add_argument("--input", type=Path, required=True, help="Output dir passed to mhc_repro.py")
  p.add_argument("--out", type=Path, default=None, help="Output dir for SVGs (default: <input>/analysis/viz)")
  args = p.parse_args(argv)

  root = args.input
  runs_path = root / "runs.json"
  if not runs_path.is_file():
    raise SystemExit(f"Missing {runs_path}. Run mhc_repro.py first.")

  runs = json.loads(runs_path.read_text(encoding="utf-8"))
  out_dir = args.out if args.out is not None else (root / "analysis" / "viz")
  out_dir.mkdir(parents=True, exist_ok=True)

  # Load all logs.
  traces: dict[str, list[dict]] = {}
  for name, meta in runs.items():
    log_path = Path(meta["train_log"])
    if log_path.is_file():
      traces[name] = _read_jsonl(log_path)

  # Common x axis: steps where "loss" is logged.
  step_sets = []
  for rows in traces.values():
    step_sets.append([int(r["step"]) for r in rows if "loss" in r])
  if not step_sets:
    raise SystemExit("No loss rows found in logs.")
  steps = sorted(set(step_sets[0]))

  def series_for(key: str, fn) -> tuple[str, list[float], str] | None:
    rows = traces.get(key) or []
    by_step = {int(r["step"]): r for r in rows if "loss" in r}
    ys = []
    for s in steps:
      r = by_step.get(int(s))
      ys.append(fn(r) if r is not None else float("nan"))
    color = COLORS.get(key, "#000000")
    return (key, ys, color)

  # Loss
  loss_series = []
  for k in ("baseline", "hc", "mhc"):
    if k in traces:
      loss_series.append(series_for(k, lambda r: float(r.get("loss", float("nan")))))
  loss_series = [s for s in loss_series if s is not None]
  all_loss = [v for _, ys, _ in loss_series for v in ys if not math.isnan(v)]
  _write_svg_line_chart(
    out_path=out_dir / "mhc_loss.svg",
    title="mHC Repro: Train Loss",
    subtitle="Lower is better",
    x_vals=steps,
    x_label="step",
    y_label="loss",
    series=loss_series,  # type: ignore[arg-type]
    y_min=min(all_loss) * 0.95 if all_loss else 0.0,
    y_max=max(all_loss) * 1.05 if all_loss else 1.0,
  )

  # Grad norm
  grad_series = []
  for k in ("baseline", "hc", "mhc"):
    if k in traces:
      grad_series.append(series_for(k, lambda r: float(r.get("grad_norm", float("nan")))))
  grad_series = [s for s in grad_series if s is not None]
  all_grad = [v for _, ys, _ in grad_series for v in ys if not math.isnan(v)]
  _write_svg_line_chart(
    out_path=out_dir / "mhc_grad_norm.svg",
    title="mHC Repro: Gradient Norm",
    subtitle="Stability diagnostic (lower variance is better)",
    x_vals=steps,
    x_label="step",
    y_label="grad_norm",
    series=grad_series,  # type: ignore[arg-type]
    y_min=0.0,
    y_max=max(all_grad) * 1.05 if all_grad else 1.0,
  )

  # Composite gain (log10 of max over fwd/bwd and layers).
  comp_series = []
  for k in ("hc", "mhc"):
    if k in traces:
      def comp_fn(r) -> float:
        if r is None:
          return float("nan")
        cf = _max_list(r.get("mhc_comp_fwd"))
        cb = _max_list(r.get("mhc_comp_bwd"))
        m = max(cf, cb)
        return math.log10(max(1e-12, m))
      comp_series.append(series_for(k, comp_fn))
  comp_series = [s for s in comp_series if s is not None]
  all_comp = [v for _, ys, _ in comp_series for v in ys if not math.isnan(v)]
  if comp_series:
    _write_svg_line_chart(
      out_path=out_dir / "mhc_comp_gain_log10.svg",
      title="mHC Repro: Composite Gain (log10)",
      subtitle="max over layers of amax row/col sum of Π H_res (lower is better)",
      x_vals=steps,
      x_label="step",
      y_label="log10(comp_gain)",
      series=comp_series,  # type: ignore[arg-type]
      y_min=min(all_comp) * 0.95 if all_comp else -1.0,
      y_max=max(all_comp) * 1.05 if all_comp else 1.0,
    )

  print(str(out_dir))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())

