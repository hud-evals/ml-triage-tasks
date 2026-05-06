"""
Analyze artifacts produced by `python -m nmoe.research.physics.mhc_repro`.

Terminal-first, no plotting deps: emits JSON + TSV suitable for paper plots.

Run:
  python -m nmoe.research.physics.mhc_repro_analysis --input /tmp/mhc_repro
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _write_tsv(path: Path, rows: list[dict], keys: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("\t".join(keys) + "\n")
        for r in rows:
            f.write("\t".join(str(r.get(k, "")) for k in keys) + "\n")


def _summarize_run(train_rows: list[dict]) -> dict:
    loss_rows = [r for r in train_rows if "loss" in r]
    valid_rows = [r for r in train_rows if "valid_loss" in r]

    out: dict = {}
    if loss_rows:
        out["steps"] = int(loss_rows[-1]["step"])
        out["loss_final"] = float(loss_rows[-1]["loss"])
        out["loss_min"] = float(min(r["loss"] for r in loss_rows))
        out["grad_norm_final"] = float(loss_rows[-1].get("grad_norm", float("nan")))

        # Track the worst composite gain across sublayers at each logged step.
        comp_fwd_max = []
        comp_bwd_max = []
        for r in loss_rows:
            cf = r.get("mhc_comp_fwd") or []
            cb = r.get("mhc_comp_bwd") or []
            comp_fwd_max.append(max(cf) if cf else float("nan"))
            comp_bwd_max.append(max(cb) if cb else float("nan"))
        out["comp_fwd_max_final"] = float(comp_fwd_max[-1])
        out["comp_bwd_max_final"] = float(comp_bwd_max[-1])
        out["comp_fwd_max_peak"] = float(max(comp_fwd_max))
        out["comp_bwd_max_peak"] = float(max(comp_bwd_max))

    if valid_rows:
        out["valid_loss_final"] = float(valid_rows[-1]["valid_loss"])
        out["valid_loss_min"] = float(min(r["valid_loss"] for r in valid_rows))

    return out


def main() -> int:
    p = argparse.ArgumentParser(description="Analyze mHC repro artifacts")
    p.add_argument("--input", type=Path, required=True, help="Output dir passed to mhc_repro")
    args = p.parse_args()

    root = args.input
    runs_path = root / "runs.json"
    if not runs_path.exists():
        raise SystemExit(f"Missing {runs_path}. Run `python -m nmoe.research.physics.mhc_repro --output ...` first.")

    runs = json.loads(runs_path.read_text(encoding="utf-8"))
    out_dir = root / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, dict] = {}

    for name, meta in runs.items():
        log_path = Path(meta["train_log"])
        train_rows = _read_jsonl(log_path)
        summary[name] = _summarize_run(train_rows)

        # Per-run TSV for quick plotting.
        keys = [
            "step",
            "loss",
            "grad_norm",
            "cuda_max_mem_mib",
        ]
        _write_tsv(out_dir / f"{name}_train.tsv", train_rows, keys)

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
