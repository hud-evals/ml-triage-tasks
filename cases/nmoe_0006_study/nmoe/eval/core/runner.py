from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import tiktoken

from nmoe.eval.chunked_logits import argmax_vocab_chunked, forward_hidden, logsumexp_vocab_chunked, target_logits
from nmoe.eval.core.bundle import CoreBundle
from nmoe.eval.core.manifest import CoreTask, load_core_tasks
from nmoe.eval.core.prompt import (
    batch_sequences_lm,
    batch_sequences_mc,
    batch_sequences_schema,
    crop_to_max_len,
    render_prompts_lm,
    render_prompts_mc,
    render_prompts_schema,
    stack_sequences,
)
from nmoe.metrics import start_metrics, stop_metrics


EVAL_BUNDLE_URL = "https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip"


@dataclass(frozen=True)
class TaskResult:
    label: str
    accuracy: float
    centered: float
    n: int


def _centered(accuracy: float, random_baseline_percent: float) -> float:
    b = 0.01 * float(random_baseline_percent)
    if b >= 1.0:
        return 0.0
    return (float(accuracy) - b) / (1.0 - b)


def _init_dist_if_needed() -> None:
    if not (dist.is_available() and torch.cuda.is_available()):
        return
    if dist.is_initialized():
        return
    if "RANK" not in os.environ:
        return
    dist.init_process_group(backend="nccl")


def _rank_world() -> tuple[int, int]:
    if dist.is_available() and dist.is_initialized():
        return int(dist.get_rank()), int(dist.get_world_size())
    return 0, 1


def _all_reduce_sum_i64(x: int, device: torch.device) -> int:
    if not (dist.is_available() and dist.is_initialized()):
        return int(x)
    t = torch.tensor([int(x)], device=device, dtype=torch.long)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return int(t.item())


def _broadcast_i64(x: int, device: torch.device, *, src: int = 0) -> int:
    """Broadcast a single int64 from src to all ranks (no-op if not distributed)."""
    if not (dist.is_available() and dist.is_initialized()):
        return int(x)
    t = torch.tensor([int(x)], device=device, dtype=torch.long)
    dist.broadcast(t, src=src)
    return int(t.item())


def _sample_fewshot(data: list[dict], *, idx: int, k: int) -> list[dict]:
    if k <= 0:
        return []
    rng = random.Random(1234 + int(idx))
    available = [i for i in range(len(data)) if i != idx]
    if len(available) < k:
        return []
    fewshot_idx = rng.sample(available, k)
    return [data[i] for i in fewshot_idx]


@torch.no_grad()
def _mean_nll_by_sequence(
    *,
    hidden: torch.Tensor,          # [B,T,D]
    input_ids: torch.Tensor,       # [B,T]
    start_idxs: list[int],
    end_idxs: list[int],
    lm_head_weight: torch.Tensor,  # [V,D]
    logits_scale: float,
    vocab_chunk_size: int,
) -> list[float]:
    device = hidden.device

    hs: list[torch.Tensor] = []
    tgts: list[torch.Tensor] = []
    lengths: list[int] = []
    for i, (si, ei) in enumerate(zip(start_idxs, end_idxs)):
        si = int(si)
        ei = int(ei)
        if si <= 0 or ei <= si:
            lengths.append(0)
            continue
        pos = torch.arange(si - 1, ei - 1, device=device, dtype=torch.long)  # predicts [si:ei)
        tgt = torch.arange(si, ei, device=device, dtype=torch.long)
        hs.append(hidden[i].index_select(0, pos).to(dtype=torch.float32))
        tgts.append(input_ids[i].index_select(0, tgt))
        lengths.append(int(ei - si))

    if not hs:
        return [float("inf")] * len(start_idxs)

    h_all = torch.cat(hs, dim=0)  # [N,D]
    t_all = torch.cat(tgts, dim=0)  # [N]
    lse = logsumexp_vocab_chunked(h_all, lm_head_weight, logits_scale=logits_scale, chunk_size=vocab_chunk_size)
    tl = target_logits(h_all, lm_head_weight, t_all, logits_scale=logits_scale)
    nll = lse - tl  # [N]

    out: list[float] = []
    off = 0
    for L in lengths:
        if L <= 0:
            out.append(float("inf"))
            continue
        seg = nll.narrow(0, off, L)
        out.append(float(seg.mean().item()))
        off += L
    return out


@torch.no_grad()
def _lm_exact_match(
    *,
    hidden: torch.Tensor,          # [T,D]
    input_ids: torch.Tensor,       # [T]
    start_idx: int,
    end_idx: int,
    lm_head_weight: torch.Tensor,  # [V,D]
    logits_scale: float,
    vocab_chunk_size: int,
) -> bool:
    device = hidden.device
    si = int(start_idx)
    ei = int(end_idx)
    if si <= 0 or ei <= si:
        return False
    pos = torch.arange(si - 1, ei - 1, device=device, dtype=torch.long)  # predicts [si:ei)
    tgt_pos = torch.arange(si, ei, device=device, dtype=torch.long)
    h = hidden.index_select(0, pos).to(dtype=torch.float32)
    pred = argmax_vocab_chunked(h, lm_head_weight, logits_scale=logits_scale, chunk_size=vocab_chunk_size)
    tgt = input_ids.index_select(0, tgt_pos)
    return bool(torch.all(pred == tgt).item())


@torch.no_grad()
def _evaluate_example(
    *,
    idx: int,
    model: torch.nn.Module,
    enc: tiktoken.Encoding,
    data: list[dict],
    task: CoreTask,
    max_seq_len: int,
    bos_id: int,
    vocab_chunk_size: int,
) -> bool:
    item = data[idx]
    fewshot = _sample_fewshot(data, idx=idx, k=task.fewshot)

    if task.task_type == "multiple_choice":
        prompts = render_prompts_mc(item, task.continuation_delimiter, fewshot_examples=fewshot)
        tokens, start_idxs, end_idxs = batch_sequences_mc(enc, prompts, bos_id=bos_id)
    elif task.task_type == "schema":
        prompts = render_prompts_schema(item, task.continuation_delimiter, fewshot_examples=fewshot)
        tokens, start_idxs, end_idxs = batch_sequences_schema(enc, prompts, bos_id=bos_id)
    elif task.task_type == "language_modeling":
        prompt_without, prompt_with = render_prompts_lm(item, task.continuation_delimiter, fewshot_examples=fewshot)
        tokens, start_idxs, end_idxs = batch_sequences_lm(enc, prompt_without, prompt_with, bos_id=bos_id)
    else:
        raise ValueError(f"unsupported task_type: {task.task_type}")

    tokens, start_idxs, end_idxs = crop_to_max_len(tokens, start_idxs, end_idxs, max_len=max_seq_len)

    pad_token_id = int(bos_id)  # matches nanochat: BOS-as-pad is safe for eval
    input_ids = stack_sequences(tokens, pad_token_id).to(device=next(model.parameters()).device)

    hidden = forward_hidden(model, input_ids)  # [B,T,D]

    lm_head = getattr(model, "lm_head", None)
    if lm_head is None or not hasattr(lm_head, "weight"):
        raise TypeError("model must expose lm_head.weight")
    W = lm_head.weight.detach()
    logits_scale = float(getattr(model, "fp4_logits_gain", getattr(model, "logits_scale_factor", 1.0)))

    if task.task_type in ("multiple_choice", "schema"):
        mean_nll = _mean_nll_by_sequence(
            hidden=hidden,
            input_ids=input_ids,
            start_idxs=start_idxs,
            end_idxs=end_idxs,
            lm_head_weight=W,
            logits_scale=logits_scale,
            vocab_chunk_size=vocab_chunk_size,
        )
        pred_idx = int(min(range(len(mean_nll)), key=lambda i: mean_nll[i]))
        return pred_idx == int(item["gold"])

    # language_modeling: batch size is 1 by construction
    return _lm_exact_match(
        hidden=hidden[0],
        input_ids=input_ids[0],
        start_idx=start_idxs[0],
        end_idx=end_idxs[0],
        lm_head_weight=W,
        logits_scale=logits_scale,
        vocab_chunk_size=vocab_chunk_size,
    )


@torch.no_grad()
def evaluate_task(
    *,
    model: torch.nn.Module,
    enc: tiktoken.Encoding,
    data: list[dict],
    task: CoreTask,
    max_per_task: int,
    max_time_s: float,
    vocab_chunk_size: int,
    max_seq_len: int,
    bos_id: int,
) -> TaskResult:
    # Shuffle deterministically (matches nanochat/scripts/base_eval.py).
    rng = random.Random(1337)
    data = list(data)
    rng.shuffle(data)
    if max_per_task > 0:
        data = data[: int(max_per_task)]

    device = next(model.parameters()).device
    rank, world = _rank_world()

    t0 = time.time()
    local_correct = 0
    local_total = 0
    for idx in range(rank, len(data), world):
        if max_time_s > 0 and (time.time() - t0) > max_time_s:
            break
        try:
            ok = _evaluate_example(
                idx=idx,
                model=model,
                enc=enc,
                data=data,
                task=task,
                max_seq_len=max_seq_len,
                bos_id=bos_id,
                vocab_chunk_size=vocab_chunk_size,
            )
        except Exception:
            ok = False
        local_correct += int(ok)
        local_total += 1

    correct = _all_reduce_sum_i64(local_correct, device)
    total = _all_reduce_sum_i64(local_total, device)
    acc = (correct / total) if total > 0 else 0.0
    return TaskResult(label=task.label, accuracy=float(acc), centered=0.0, n=int(total))


def evaluate_task_live(
    *,
    model: torch.nn.Module,
    enc: tiktoken.Encoding,
    data: list[dict],
    task: CoreTask,
    max_per_task: int,
    max_time_s: float,
    vocab_chunk_size: int,
    max_seq_len: int,
    bos_id: int,
) -> TaskResult:
    """Evaluate a task on a live distributed model in lockstep.

    RDEP/EP models require all ranks to execute the same forwards with the same
    shapes and in the same order. This runner broadcasts the current example
    index from rank0; all ranks evaluate it; only rank0 accumulates correctness.
    """
    # Shuffle deterministically (matches nanochat/scripts/base_eval.py).
    rng = random.Random(1337)
    data = list(data)
    rng.shuffle(data)
    if max_per_task > 0:
        data = data[: int(max_per_task)]

    device = next(model.parameters()).device
    rank, _world = _rank_world()

    t0 = time.time()
    correct = 0
    total = 0
    for idx in range(len(data)):
        if rank == 0 and max_time_s > 0 and (time.time() - t0) > max_time_s:
            idx = -1
        idx = _broadcast_i64(int(idx), device, src=0)
        if idx < 0:
            break

        ok = False
        try:
            ok = _evaluate_example(
                idx=int(idx),
                model=model,
                enc=enc,
                data=data,
                task=task,
                max_seq_len=max_seq_len,
                bos_id=bos_id,
                vocab_chunk_size=vocab_chunk_size,
            )
        except Exception:
            ok = False

        if rank == 0:
            correct += int(ok)
            total += 1

    acc = (correct / total) if total > 0 else 0.0
    return TaskResult(label=task.label, accuracy=float(acc), centered=0.0, n=int(total))


def run_core_live(
    *,
    step: int,
    cfg_dict: dict[str, Any],
    cfg,
    model: torch.nn.Module,
    run_id: str,
    tasks_file: str = "configs/eval/core.toml",
    bundle_dir: str = "",
    max_per_task: int = -1,
    max_time_s: float = 0.0,
    vocab_chunk_size: int = 4096,
) -> dict[str, Any]:
    """Run CORE evaluation on the live model and log results.

    This is the correct integration path for RDEP/EP training: no snapshots of
    sharded weights, and all ranks evaluate in lockstep.
    """
    _init_dist_if_needed()
    rank, _world = _rank_world()

    device = next(model.parameters()).device
    was_training = bool(model.training)
    model.eval()
    try:
        enc = tiktoken.get_encoding(str(getattr(cfg, "tokenizer", "o200k_harmony")))
        bos_id = int(getattr(cfg, "eos_token_id", 0))
        max_seq_len = int(getattr(cfg, "seq_len", 0) or 0)

        bdir = Path(bundle_dir) if bundle_dir else _default_bundle_dir(cfg_dict)
        bundle = CoreBundle(bdir)
        bundle.require()

        tasks = load_core_tasks(Path(tasks_file))
        baselines = bundle.load_random_baselines()

        # Metrics go under {metrics_dir}/eval/{run_id}/step_XXXXXXXX.parquet (rank-0 only).
        metrics_dir = str(getattr(cfg, "metrics_dir", "/data/metrics"))
        metrics_ctx = start_metrics(run_id=str(run_id), metrics_dir=str(Path(metrics_dir) / "eval")) if rank == 0 else None

        out_root = Path(metrics_dir) / "eval" / str(run_id) / f"step_{int(step):07d}"
        if rank == 0:
            out_root.mkdir(parents=True, exist_ok=True)

        results: list[TaskResult] = []
        t_global0 = time.time()
        for task in tasks:
            data = bundle.load_jsonl(task.dataset_uri)
            res = evaluate_task_live(
                model=model,
                enc=enc,
                data=data,
                task=task,
                max_per_task=int(max_per_task),
                max_time_s=float(max_time_s),
                vocab_chunk_size=int(vocab_chunk_size),
                max_seq_len=max_seq_len,
                bos_id=bos_id,
            )
            rb = float(baselines.get(task.label, 0.0))
            results.append(
                TaskResult(
                    label=task.label,
                    accuracy=res.accuracy,
                    centered=_centered(res.accuracy, rb),
                    n=res.n,
                )
            )
            if rank == 0:
                print(f"[core] {task.label}: acc={res.accuracy:.4f} centered={results[-1].centered:.4f} n={res.n}")

        core_metric = sum(r.centered for r in results) / max(1, len(results))
        elapsed_s = time.time() - t_global0
        summary: dict[str, Any] = {
            "suite": "core",
            "CORE": float(core_metric),
            "tasks": [
                {"task": r.label, "acc": r.accuracy, "centered": r.centered, "n": r.n}
                for r in results
            ],
            "elapsed_s": float(elapsed_s),
            "bundle_dir": str(bdir),
            "bundle_url": EVAL_BUNDLE_URL,
        }

        if rank == 0:
            (out_root / "core_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

        if rank == 0:
            try:
                if metrics_ctx is not None and metrics_ctx.writer is not None:
                    items = [("eval/CORE", float(core_metric))]
                    for r in results:
                        items.append((f"eval/core/{r.label}/acc", float(r.accuracy)))
                        items.append((f"eval/core/{r.label}/centered", float(r.centered)))
                        items.append((f"eval/core/{r.label}/n", float(r.n)))
                    metrics_ctx.writer.insert_many(step=int(step), items=items)
            except Exception:
                pass
            finally:
                stop_metrics(metrics_ctx)
        else:
            stop_metrics(metrics_ctx)

        if rank == 0:
            print(f"[core] CORE={core_metric:.6f} ({len(results)} tasks, {elapsed_s:.1f}s)")
        return summary
    finally:
        if was_training:
            model.train()

def _load_snapshot(path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    obj = torch.load(str(path), map_location="cpu")
    cfg_dict = obj.get("config", {}) or {}
    model_state = obj.get("model_state", {}) or {}
    meta = {k: obj.get(k) for k in ("run_id", "step")}
    return cfg_dict, model_state, meta


def _default_bundle_dir(cfg_dict: dict[str, Any]) -> Path:
    data_root = str(cfg_dict.get("data_root") or "/data")
    return Path(data_root) / "eval" / "eval_bundle"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser("nmoe.eval.core")
    ap.add_argument("--snapshot", required=True, help="Path to eval snapshot (.pt) with config+model_state")
    ap.add_argument("--tasks-file", default="configs/eval/core.toml")
    ap.add_argument("--bundle-dir", default="", help="Path to eval_bundle dir (defaults to {data_root}/eval/eval_bundle)")
    ap.add_argument("--max-per-task", type=int, default=-1)
    ap.add_argument("--max-time-s", type=float, default=0.0)
    ap.add_argument("--vocab-chunk-size", type=int, default=4096)
    args = ap.parse_args(argv)

    _init_dist_if_needed()
    rank, _world = _rank_world()

    snap = Path(args.snapshot)
    cfg_dict, model_state, meta = _load_snapshot(snap)

    from nmoe.config import Config, upgrade_cfg_dict
    from nmoe.model import Transformer

    cfg = Config(**upgrade_cfg_dict(cfg_dict))
    # Eval buffers: bound batch_size for RDEP allocations; does not change params.
    # Must be >= the largest eval batch we will forward (max choices in MC tasks).
    cfg.batch_size = 16

    # Snapshot-based eval is not EP-correct under distributed MoE (experts sharded).
    if _world > 1 and int(getattr(cfg, "n_layers", 0) or 0) > int(getattr(cfg, "n_dense_layers", 0) or 0):
        raise RuntimeError(
            f"CORE snapshot eval is not supported for MoE under distributed execution (world_size={_world}). "
            "Run eval_tasks=core with eval_mode=inline (live lockstep), or evaluate the snapshot with world_size=1."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Transformer(cfg).to(device=device).eval()
    model.load_state_dict(model_state, strict=False)

    enc = tiktoken.get_encoding(str(getattr(cfg, "tokenizer", "o200k_harmony")))
    bos_id = int(getattr(cfg, "eos_token_id", 0))
    max_seq_len = int(getattr(cfg, "seq_len", 0) or 0)

    bundle_dir = Path(args.bundle_dir) if args.bundle_dir else _default_bundle_dir(cfg_dict)
    bundle = CoreBundle(bundle_dir)
    try:
        bundle.require()
    except FileNotFoundError as e:
        out_dir = str(Path(str(cfg_dict.get("data_root") or "/data")) / "eval")
        raise RuntimeError(
            f"CORE eval bundle missing at {bundle_dir}. "
            f"Prepare it with: python -m nmoe.eval.core.prep --out-dir {out_dir}"
        ) from e

    tasks = load_core_tasks(Path(args.tasks_file))
    baselines = bundle.load_random_baselines()

    # Metrics go under {metrics_dir}/eval/{run_id}/step_XXXXXXXX.parquet (rank-0 only).
    run_id = str(meta.get("run_id") or os.getenv("NMOE_RUN") or cfg.experiment_id or "eval")
    metrics_dir = str(cfg_dict.get("metrics_dir") or "/data/metrics")
    metrics_ctx = start_metrics(run_id=run_id, metrics_dir=str(Path(metrics_dir) / "eval"))

    max_per_task = int(args.max_per_task)
    max_time_s = float(args.max_time_s)
    vocab_chunk_size = int(args.vocab_chunk_size)

    results: list[TaskResult] = []
    t_global0 = time.time()
    for task in tasks:
        data = bundle.load_jsonl(task.dataset_uri)
        res = evaluate_task(
            model=model,
            enc=enc,
            data=data,
            task=task,
            max_per_task=max_per_task,
            max_time_s=max_time_s,
            vocab_chunk_size=vocab_chunk_size,
            max_seq_len=max_seq_len,
            bos_id=bos_id,
        )
        rb = float(baselines.get(task.label, 0.0))
        results.append(
            TaskResult(
                label=task.label,
                accuracy=res.accuracy,
                centered=_centered(res.accuracy, rb),
                n=res.n,
            )
        )
        if rank == 0:
            print(f"[core] {task.label}: acc={res.accuracy:.4f} centered={results[-1].centered:.4f} n={res.n}")

    core_metric = sum(r.centered for r in results) / max(1, len(results))

    elapsed_s = time.time() - t_global0
    summary = {
        "suite": "core",
        "CORE": float(core_metric),
        "tasks": [
            {"task": r.label, "acc": r.accuracy, "centered": r.centered, "n": r.n}
            for r in results
        ],
        "elapsed_s": float(elapsed_s),
        "bundle_dir": str(bundle_dir),
        "bundle_url": EVAL_BUNDLE_URL,
    }
    if rank == 0:
        out_path = snap.parent / "core_summary.json"
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Persist to DuckDB (rank-local file).
    try:
        if metrics_ctx is not None and metrics_ctx.writer is not None:
            step = int(meta.get("step") or 0)
            items = [("eval/CORE", float(core_metric))]
            for r in results:
                items.append((f"eval/core/{r.label}/acc", float(r.accuracy)))
                items.append((f"eval/core/{r.label}/centered", float(r.centered)))
                items.append((f"eval/core/{r.label}/n", float(r.n)))
            metrics_ctx.writer.insert_many(step=step, items=items)
    except Exception:
        pass
    finally:
        stop_metrics(metrics_ctx)

    if rank == 0:
        print(f"[core] CORE={core_metric:.6f} ({len(results)} tasks, {elapsed_s:.1f}s)")


if __name__ == "__main__":  # pragma: no cover
    main()
