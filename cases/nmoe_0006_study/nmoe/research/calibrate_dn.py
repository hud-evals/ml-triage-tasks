"""DN (tokens/param) calibration runner for #420-style experiments.

This script exists to *measure* the right D:N ratio for our stack instead of
assuming it. It runs a single depth at multiple target_dn values and records
final val_bpb and CORE.

Example:
  python -m nmoe.research.calibrate_dn \\
    --depth 14 \\
    --target-dn 8,12 \\
    --out-dir /data/miniseries/dn_calib_d14
"""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
import time
from pathlib import Path

from nmoe.research.miniseries import (
  _default_nproc,
  _default_out_dir,
  _latest_metric_value,
  _load_toml,
  _pick_master_addr,
  _pick_master_port,
  _write_flat_toml,
  build_dense_depth_config,
)


def _parse_csv_floats(s: str) -> list[float]:
  out: list[float] = []
  for part in str(s).split(","):
    part = part.strip()
    if not part:
      continue
    out.append(float(part))
  if not out:
    raise ValueError("expected a non-empty comma list of floats")
  return out


def _dn_slug(dn: float) -> str:
  # Prefer "8" over "8.0" for readability.
  if math.isfinite(dn) and abs(dn - round(dn)) < 1e-9:
    return str(int(round(dn)))
  return str(dn).replace(".", "p")


def main() -> None:
  ap = argparse.ArgumentParser("nmoe.research.calibrate_dn")
  ap.add_argument(
    "--base-config",
    default="configs/research/420_dense_base.toml",
    help="base TOML config (must have seq_len=2048,batch_size=256)",
  )
  ap.add_argument("--depth", type=int, default=14, help="depth dial value")
  ap.add_argument("--target-dn", default="8,12", help="comma list of tokens/param ratios to try")
  ap.add_argument("--warmup-ratio", default=0.0, type=float)
  ap.add_argument("--warmdown-ratio", default=0.4, type=float)
  ap.add_argument("--use-muon", action="store_true", help="enable Muon for 2D weights")
  ap.add_argument("--out-dir", default=None, help="output directory for generated configs")
  ap.add_argument("--nproc-per-node", default=None, type=int, help="torchrun nproc_per_node (default: env or 8)")
  ap.add_argument("--dry-run", action="store_true", help="write configs and print commands, but do not launch")
  args = ap.parse_args()

  base_path = Path(args.base_config)
  base = _load_toml(base_path)

  out_dir = Path(args.out_dir) if args.out_dir else _default_out_dir()
  out_dir.mkdir(parents=True, exist_ok=True)
  results_path = out_dir / "results.csv"

  nproc = int(args.nproc_per_node) if args.nproc_per_node is not None else _default_nproc()
  series_id = out_dir.name
  master_addr = _pick_master_addr()
  master_port = _pick_master_port()
  repo_root = Path(__file__).resolve().parents[2]
  quack_path = repo_root / "third_party" / "quack"
  metrics_dir = Path(str(base.get("metrics_dir") or "/data/metrics"))

  if not results_path.exists():
    results_path.write_text(
      "depth,target_dn,use_muon,model_dim,num_params,num_iterations,tokens_trained,param_data_ratio,val_bpb,core_score,train_time_sec\n",
      encoding="utf-8",
    )

  dns = _parse_csv_floats(args.target_dn)
  d = int(args.depth)
  use_muon = bool(args.use_muon)

  print(f"[calibrate_dn] base={base_path}")
  print(f"[calibrate_dn] out_dir={out_dir}")
  print(f"[calibrate_dn] depth={d} use_muon={use_muon}")
  print(f"[calibrate_dn] target_dn={dns}")
  print(f"[calibrate_dn] nproc_per_node={nproc}")
  print(f"[calibrate_dn] master_addr={master_addr} master_port={master_port}")
  print(f"[calibrate_dn] quack_path={quack_path}")
  print("")

  for dn in dns:
    cfg, meta = build_dense_depth_config(
      base=base,
      depth=d,
      target_dn=float(dn),
      warmup_ratio=float(args.warmup_ratio),
      warmdown_ratio=float(args.warmdown_ratio),
    )
    cfg["use_muon"] = bool(use_muon)

    dn_slug = _dn_slug(float(dn))
    muon_slug = "muon1" if use_muon else "muon0"
    cfg_path = out_dir / f"d{d:02d}_dn{dn_slug}_{muon_slug}.toml"
    _write_flat_toml(cfg_path, cfg)

    run_id = f"{cfg['experiment_id']}__{series_id}__d{d:02d}__dn{dn_slug}__{muon_slug}"
    cmd = [
      sys.executable,
      "-m",
      "torch.distributed.run",
      f"--nproc_per_node={nproc}",
      f"--master_addr={master_addr}",
      f"--master_port={master_port}",
      "-m",
      "nmoe.train",
      str(cfg_path),
    ]

    print(
      f"[calibrate_dn] dn={float(dn):g} dim={meta['dim']} "
      f"params_total={meta['params_total']:,} steps={meta['steps']:,} tokens={meta['total_tokens']:,}"
    )
    print(f"[calibrate_dn] run_id={run_id}")
    print(f"[calibrate_dn] config={cfg_path}")
    print(f"[calibrate_dn] launch: {' '.join(cmd)}")
    print("")

    if args.dry_run:
      continue

    env = os.environ.copy()
    env["NMOE_RUN"] = run_id
    if quack_path.exists():
      env["PYTHONPATH"] = str(quack_path) + os.pathsep + env.get("PYTHONPATH", "")
    t0 = time.time()
    proc = subprocess.run(cmd, env=env, check=False)
    train_time_s = time.time() - t0
    if proc.returncode != 0:
      raise SystemExit(proc.returncode)

    val_bpb = _latest_metric_value(metrics_dir=metrics_dir, run_id=run_id, tag="valid/bpb")
    core = _latest_metric_value(metrics_dir=metrics_dir, run_id=run_id, tag="eval/CORE")
    ratio = float(meta["total_tokens"]) / float(meta["params_total"])

    row = ",".join([
      str(int(meta["depth"])),
      str(float(dn)),
      ("1" if use_muon else "0"),
      str(int(meta["dim"])),
      str(int(meta["params_total"])),
      str(int(meta["steps"])),
      str(int(meta["total_tokens"])),
      f"{ratio:.4f}",
      (f"{val_bpb:.6f}" if val_bpb is not None else ""),
      (f"{core:.6f}" if core is not None else ""),
      f"{train_time_s:.1f}",
    ])
    with results_path.open("a", encoding="utf-8") as f:
      f.write(row + "\n")
    print(f"[calibrate_dn] wrote results.csv row: dn={float(dn):g} val_bpb={val_bpb} core={core} time_s={train_time_s:.1f}")


if __name__ == "__main__":
  main()
