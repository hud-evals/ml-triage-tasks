"""Karpathy-style miniseries runner (v2, dense-first).

This file intentionally supersedes earlier v0 "miniseries" experiments in this
repo. v2 is designed to be:
  - faithful to the *method* in nanochat #420 (one dial, coherent family)
  - conservative about assumptions (no bespoke param-math modules)
  - minimal surface area (no hidden policies; the contract is explicit)

v2 contract (dense depth dial):
  Dial:
    - depth d
    - dim = 64*d
    - n_layers = d
    - n_heads chosen to approximate head_dim≈128 (nanochat base_train.py)
    - inter_dim = 4*dim

  Invariants:
    - seq_len=2048
    - tokens_per_step=524,288 (batch_size=256 global)

  Horizon (DN method):
    steps = floor(target_dn * params_total / tokens_per_step)

  Metrics:
    - prefer valid/bpb and eval/CORE (not raw valid loss)
"""

from __future__ import annotations

import argparse
import math
import os
import socket
import subprocess
import sys
import time
import tomllib
from pathlib import Path
from typing import Any

TOKENS_PER_STEP_CANON = 524_288
SEQ_LEN_CANON = 2048
HEAD_DIM_TARGET = 128


def _parse_csv_ints(s: str) -> list[int]:
  out: list[int] = []
  for part in str(s).split(","):
    part = part.strip()
    if not part:
      continue
    out.append(int(part))
  if not out:
    raise ValueError("expected a non-empty comma list of ints")
  return out


def _find_num_heads(*, model_dim: int, target_head_dim: int) -> int:
  """Find num_heads dividing model_dim with head_dim closest to target.

  Mirrors nanochat/scripts/base_train.py: find_num_heads().
  """
  md = int(model_dim)
  hd = int(target_head_dim)
  if md <= 0:
    raise ValueError(f"model_dim must be > 0 (got {md})")
  if hd <= 0:
    raise ValueError(f"target_head_dim must be > 0 (got {hd})")
  ideal = max(1, int(round(float(md) / float(hd))))
  for offset in range(md):
    for candidate in (ideal + offset, ideal - offset):
      if candidate > 0 and (md % candidate) == 0:
        return int(candidate)
  return 1


def _toml_quote(s: str) -> str:
  s = str(s)
  s = s.replace("\\", "\\\\").replace("\"", "\\\"")
  return f"\"{s}\""


def _toml_value(v: Any) -> str:
  if isinstance(v, bool):
    return "true" if v else "false"
  if isinstance(v, int):
    return str(v)
  if isinstance(v, float):
    if not math.isfinite(v):
      raise ValueError("refusing to write non-finite float")
    # Avoid scientific notation; keep it readable.
    s = f"{v:.12f}".rstrip("0").rstrip(".")
    return s if s else "0.0"
  if isinstance(v, str):
    return _toml_quote(v)
  if isinstance(v, list):
    return "[" + ", ".join(_toml_value(x) for x in v) + "]"
  raise TypeError(f"unsupported TOML value type: {type(v).__name__}")


def _write_flat_toml(path: Path, d: dict[str, Any]) -> None:
  lines: list[str] = []
  for k in sorted(d.keys()):
    v = d[k]
    if v is None:
      continue
    lines.append(f"{k} = {_toml_value(v)}")
  path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_toml(path: Path) -> dict[str, Any]:
  return tomllib.loads(path.read_text(encoding="utf-8"))


def dense_sdpa_params_total(*, vocab_size: int, depth: int) -> int:
  """Exact total param count for nmoe's dense SDPA stack under the v2 contract.

  Assumes:
    dim = 64*depth
    n_layers = depth
    inter_dim = 4*dim
    attention projections are square (dim->dim), bias-free
    MLP is SwiGLU with (w1,w3: dim->inter_dim) and (w2: inter_dim->dim), bias-free
    RMSNorm has a learned weight vector of length dim
    embeddings + lm_head are untied (both have vocab_size*dim params)
  """
  d = int(depth)
  if d <= 0:
    raise ValueError(f"depth must be > 0 (got {d})")
  V = int(vocab_size)
  if V <= 0:
    raise ValueError(f"vocab_size must be > 0 (got {V})")
  D = 64 * d
  L = d

  # Embedding + lm_head (untied)
  emb = 2 * V * D

  # Attention matrices: wq,wk,wv,wo each D x D
  attn = 4 * L * D * D

  # SwiGLU MLP matrices: w1,w3 each D x (4D), w2 (4D) x D => 12 * D^2 per layer
  mlp = 12 * L * D * D

  # RMSNorm weights: attn_norm + ffn_norm per layer, plus final norm.
  norm = (2 * L + 1) * D

  return int(emb + attn + mlp + norm)


def steps_for_dn(*, params_total: int, tokens_per_step: int, target_dn: float) -> int:
  P = int(params_total)
  tps = int(tokens_per_step)
  dn = float(target_dn)
  if P <= 0:
    raise ValueError(f"params_total must be > 0 (got {P})")
  if tps <= 0:
    raise ValueError(f"tokens_per_step must be > 0 (got {tps})")
  if dn <= 0:
    raise ValueError(f"target_dn must be > 0 (got {dn})")
  return max(1, int(math.floor((dn * float(P)) / float(tps))))


def wsd_schedule_from_horizon_steps(
  *,
  horizon_steps: int,
  tokens_per_step: int,
  warmup_ratio: float,
  warmdown_ratio: float,
) -> tuple[int, int, int]:
  """Compute WSD schedule fields from horizon steps (warmup/warmdown are ratios)."""
  steps = int(horizon_steps)
  tps = int(tokens_per_step)
  if steps <= 0:
    raise ValueError(f"horizon_steps must be > 0 (got {steps})")
  if tps <= 0:
    raise ValueError(f"tokens_per_step must be > 0 (got {tps})")
  wu = float(warmup_ratio)
  wd = float(warmdown_ratio)
  if not (0.0 <= wu < 1.0):
    raise ValueError(f"warmup_ratio must be in [0,1) (got {wu})")
  if not (0.0 < wd < 1.0):
    raise ValueError(f"warmdown_ratio must be in (0,1) (got {wd})")
  if wu + wd >= 1.0:
    raise ValueError(f"warmup_ratio + warmdown_ratio must be < 1 (got {wu} + {wd})")

  warmup_steps = max(1, int(math.floor(float(steps) * wu)))
  warmdown_steps = max(1, int(math.floor(float(steps) * wd)))

  if warmup_steps + warmdown_steps >= steps:
    warmup_steps = max(1, steps - warmdown_steps)

  hold_steps = max(0, steps - warmdown_steps)
  hold_tokens = int(hold_steps) * tps
  decay_tokens = int(warmdown_steps) * tps
  return int(warmup_steps), int(hold_tokens), int(decay_tokens)


def _require_int(cfg: dict[str, Any], key: str) -> int:
  if key not in cfg:
    raise KeyError(f"missing required config key: {key}")
  v = int(cfg[key])
  if v <= 0:
    raise ValueError(f"{key} must be > 0 (got {v})")
  return v


def build_dense_depth_config(
  *,
  base: dict[str, Any],
  depth: int,
  target_dn: float,
  warmup_ratio: float,
  warmdown_ratio: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
  """Return (config_dict, derived_metadata)."""
  cfg = dict(base)

  batch_size = _require_int(cfg, "batch_size")
  seq_len = _require_int(cfg, "seq_len")
  tokens_per_step = batch_size * seq_len
  if seq_len != SEQ_LEN_CANON or tokens_per_step != TOKENS_PER_STEP_CANON:
    raise ValueError(
      "miniseries v2 requires canonical throughput invariants:\n"
      f"  expected seq_len={SEQ_LEN_CANON}, tokens_per_step={TOKENS_PER_STEP_CANON:,}\n"
      f"  got seq_len={seq_len}, tokens_per_step={tokens_per_step:,}\n"
      "Fix your base config to use seq_len=2048 and batch_size=256 (global)."
    )

  d = int(depth)
  if d <= 0:
    raise ValueError(f"depth must be > 0 (got {d})")
  dim = 64 * d
  num_heads = _find_num_heads(model_dim=int(dim), target_head_dim=int(HEAD_DIM_TARGET))

  # Dial (depth) + architecture invariants.
  cfg["attn"] = "sdpa"
  cfg["dim"] = int(dim)
  cfg["n_layers"] = int(d)
  cfg["n_heads"] = int(num_heads)
  cfg["inter_dim"] = int(4 * dim)
  cfg["n_dense_layers"] = int(d)  # dense-only
  cfg["max_position_embeddings"] = int(SEQ_LEN_CANON)
  cfg["rope_theta"] = float(10000.0)
  cfg["resume"] = False

  # Ensure MoE knobs are not accidentally inherited from a base config.
  for k in ("n_routed_experts", "n_activated_experts", "n_shared_experts", "moe_inter_dim"):
    if k in cfg:
      cfg.pop(k)

  # Depth-aware LR scaling: mimic nanochat's ∝1/sqrt(dmodel) transfer.
  # We scale relative to the base config's dim (commonly 768).
  try:
    base_dim = int(base.get("dim") or 768)
    if base_dim > 0 and "lr_dense" in base:
      lr0 = float(base["lr_dense"])
      lr = lr0 * (float(dim) / float(base_dim)) ** -0.5
      cfg["lr_dense"] = float(lr)
      cfg["lr_router"] = float(lr)
      cfg["lr_expert"] = float(lr)
  except Exception:
    pass

  vocab_size = _require_int(cfg, "vocab_size")
  params_total = dense_sdpa_params_total(vocab_size=vocab_size, depth=d)
  horizon_steps = steps_for_dn(params_total=params_total, tokens_per_step=tokens_per_step, target_dn=target_dn)
  warmup_steps, hold_tokens, decay_tokens = wsd_schedule_from_horizon_steps(
    horizon_steps=horizon_steps,
    tokens_per_step=tokens_per_step,
    warmup_ratio=warmup_ratio,
    warmdown_ratio=warmdown_ratio,
  )

  cfg["steps"] = int(horizon_steps)
  cfg["warmup_steps"] = int(warmup_steps)
  cfg["hold_tokens"] = int(hold_tokens)
  cfg["decay_tokens"] = int(decay_tokens)

  run_tag = f"m420v2_dense_d{d:02d}"
  cfg["preset"] = run_tag
  cfg["experiment_id"] = run_tag

  meta = {
    "dial": "depth",
    "depth": d,
    "dim": dim,
    "layers": d,
    "heads": int(num_heads),
    "params_total": int(params_total),
    "tokens_per_step": int(tokens_per_step),
    "steps": int(horizon_steps),
    "total_tokens": int(tokens_per_step) * int(horizon_steps),
  }
  return cfg, meta


def _default_out_dir() -> Path:
  ts = time.strftime("%Y%m%d_%H%M%S")
  return Path("/data/miniseries") / f"m420v2_dense_{ts}"


def _default_nproc() -> int:
  for k in ("NPROC_PER_NODE", "NPROC"):
    v = os.getenv(k)
    if v:
      try:
        return int(v)
      except Exception:
        pass
  return 8


def _pick_master_addr() -> str:
  return os.getenv("NMOE_MASTER_ADDR", "127.0.0.1")


def _pick_master_port() -> int:
  v = os.getenv("NMOE_MASTER_PORT", "")
  if v:
    return int(v)
  # Best-effort: pick an ephemeral localhost port.
  # This avoids hangs in environments where --standalone tries to resolve an
  # unresolvable hostname (common in some k8s setups).
  s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  try:
    s.bind(("127.0.0.1", 0))
    return int(s.getsockname()[1])
  finally:
    s.close()


def _latest_metric_value(*, metrics_dir: Path, run_id: str, tag: str) -> float | None:
  """Read latest (by step) value for tag from parquet step files."""
  run_dir = metrics_dir / str(run_id)
  if not run_dir.exists():
    return None
  glob = str(run_dir / "step_*.parquet")
  try:
    import duckdb  # nmoe depends on duckdb via nmoe.metrics
    con = duckdb.connect(database=":memory:")
    row = con.execute(
      "SELECT value FROM read_parquet(?, union_by_name=true, filename=false) WHERE tag = ? ORDER BY step DESC LIMIT 1",
      [glob, str(tag)],
    ).fetchone()
    return float(row[0]) if row else None
  except Exception:
    return None


def main() -> None:
  ap = argparse.ArgumentParser("nmoe.research.miniseries")
  ap.add_argument(
    "--base-config",
    default="configs/research/420_dense_base.toml",
    help="base TOML config (must have seq_len=2048,batch_size=256)",
  )
  ap.add_argument(
    "--depths",
    default="8,9,10,11,12,13,14,15,16,17,18,19,20",
    help="comma list of depths (dial values)",
  )
  ap.add_argument("--target-dn", default=8.0, type=float, help="tokens/param ratio (DN)")
  ap.add_argument("--warmup-ratio", default=0.0, type=float)
  ap.add_argument("--warmdown-ratio", default=0.4, type=float)
  ap.add_argument("--out-dir", default=None, help="output directory for generated configs")
  ap.add_argument("--nproc-per-node", default=None, type=int, help="torchrun nproc_per_node (default: env or 8)")
  ap.add_argument("--dry-run", action="store_true", help="write configs and print commands, but do not launch")
  args = ap.parse_args()

  depths = _parse_csv_ints(args.depths)
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
      "depth,model_dim,num_params,num_scaling_params,num_iterations,tokens_trained,param_data_ratio,val_bpb,core_score,train_time_sec\n",
      encoding="utf-8",
    )

  print(f"[miniseries] base={base_path}")
  print(f"[miniseries] out_dir={out_dir}")
  print(f"[miniseries] nproc_per_node={nproc}")
  print(f"[miniseries] master_addr={master_addr} master_port={master_port}")
  print(f"[miniseries] quack_path={quack_path}")
  print(f"[miniseries] target_dn={float(args.target_dn)} warmup_ratio={float(args.warmup_ratio)} warmdown_ratio={float(args.warmdown_ratio)}")
  print("")

  for d in depths:
    cfg, meta = build_dense_depth_config(
      base=base,
      depth=int(d),
      target_dn=float(args.target_dn),
      warmup_ratio=float(args.warmup_ratio),
      warmdown_ratio=float(args.warmdown_ratio),
    )
    cfg_path = out_dir / f"d{int(d):02d}.toml"
    _write_flat_toml(cfg_path, cfg)

    run_id = f"{cfg['experiment_id']}__{series_id}__d{int(d):02d}"
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
      f"[miniseries] d={meta['depth']:>2} dim={meta['dim']:>4} "
      f"params_total={meta['params_total']:,} steps={meta['steps']:,} tokens={meta['total_tokens']:,}"
    )
    print(f"[miniseries] run_id={run_id}")
    print(f"[miniseries] config={cfg_path}")
    print(f"[miniseries] launch: {' '.join(cmd)}")
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

    # Summarize results from parquet metrics (rank0 only metrics).
    val_bpb = _latest_metric_value(metrics_dir=metrics_dir, run_id=run_id, tag="valid/bpb")
    core = _latest_metric_value(metrics_dir=metrics_dir, run_id=run_id, tag="eval/CORE")
    ratio = float(meta["total_tokens"]) / float(meta["params_total"])

    row = ",".join([
      str(int(meta["depth"])),
      str(int(meta["dim"])),
      str(int(meta["params_total"])),
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
    print(f"[miniseries] wrote results.csv row: d={meta['depth']} val_bpb={val_bpb} core={core} time_s={train_time_s:.1f}")


if __name__ == "__main__":
  main()
