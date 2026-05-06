"""
Command-line interface for nmoe dataset prep (HF → shards) and validation helpers.

Golden paths:
  - Prep a single dataset:
      python -m nmoe.data.cli prep --source hf --dataset HuggingFaceFW/fineweb-edu --output /data/fineweb_edu --name fineweb_edu

  - Prep all HuggingFace sources referenced by a mixture+flow:
      python -m nmoe.data.cli prep-mixture --config configs/mixtures/olmo3_1025.toml --flow dev --stage pretrain

  - Inspect/verify:
      python -m nmoe.data.cli info /data/fineweb_edu/manifest.json
      python -m nmoe.data.cli verify /data/fineweb_edu/manifest.json
"""

from __future__ import annotations

import argparse
import glob
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from nmoe.config import load_toml

log = logging.getLogger("nmoe.data.cli")


def _read_toml(path: Path) -> Dict[str, Any]:
    obj = load_toml(path)
    if not isinstance(obj, dict):
        raise ValueError("invalid TOML root (expected table)")
    return obj


def _parse_tokens(s: str) -> int:
    """Parse token count string like '100M', '1B', '1.5T'."""
    s = s.strip().upper()
    multipliers = {"K": 1_000, "M": 1_000_000, "B": 1_000_000_000, "T": 1_000_000_000_000}
    for suffix, mult in multipliers.items():
        if s.endswith(suffix):
            return int(float(s[:-1]) * mult)
    return int(s)


def _paths_from_globs(patterns: Sequence[str]) -> list[str]:
    out: list[str] = []
    for pat in patterns:
        out.extend(sorted(glob.glob(pat)))
    return out


def _shard_paths(paths: list[str], *, num_workers: int, worker_index: int) -> list[str]:
    if num_workers <= 1:
        return paths
    if worker_index < 0 or worker_index >= num_workers:
        raise ValueError(f"worker_index out of range: {worker_index} (num_workers={num_workers})")
    return [p for i, p in enumerate(paths) if (i % num_workers) == worker_index]


def _autodetect_flow_name(*, mixture_toml: Path, flow_profiles_toml: Path) -> str:
    """Infer a flow name from a mixture TOML and a flow-profiles TOML."""
    mix = _read_toml(mixture_toml)
    mixtures = mix.get("mixtures")
    if not isinstance(mixtures, dict) or not mixtures:
        raise ValueError(f"{mixture_toml} is missing a [mixtures] table")
    mixture_ids = set(mixtures.keys())

    fp = _read_toml(flow_profiles_toml)
    flows = fp.get("flow")
    if not isinstance(flows, dict) or not flows:
        raise ValueError(f"{flow_profiles_toml} is missing a [flow] table")

    candidates: list[str] = []
    for flow_name, flow_cfg in flows.items():
        if not isinstance(flow_cfg, dict):
            continue
        ref = flow_cfg.get("mixture")
        if not isinstance(ref, dict):
            continue
        if any(m in mixture_ids for m in ref.values()):
            candidates.append(str(flow_name))

    if not candidates:
        raise ValueError(
            f"Could not infer flow for {mixture_toml} from {flow_profiles_toml}; "
            "pass --flow explicitly."
        )
    if len(candidates) == 1:
        return candidates[0]

    for c in candidates:
        if c == mixture_toml.stem:
            return c

    raise ValueError(
        "Multiple flows match this mixture; pass --flow explicitly. "
        f"Candidates: {', '.join(sorted(candidates))}"
    )


def _iter_docs_limited(source, *, limit_docs: int | None) -> Iterable:
    if limit_docs is None:
        yield from source
        return
    n = 0
    for doc in source:
        if n >= limit_docs:
            break
        yield doc
        n += 1


def cmd_prep(args: argparse.Namespace) -> int:
    from .sources import (
        ArrowSource,
        HuggingFaceSource,
        HfHubParquetSource,
        JSONLSource,
        JSONLZstSource,
        TextFileSource,
    )
    from .prep import PrepConfig, PrepPipeline, ParallelPrepPipeline

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_type = str(args.source).lower()

    if source_type in ("hf", "huggingface"):
        if not args.dataset:
            log.error("--dataset is required for --source hf")
            return 1
        src = HuggingFaceSource(
            dataset=args.dataset,
            split=args.split,
            subset=args.subset,
            text_field=args.text_field,
            id_field=args.id_field,
            streaming=bool(args.streaming),
            data_files=args.data_files or None,
        )
        if args.num_workers > 1:
            src = src.shard(num_shards=args.num_workers, index=args.worker_index)
    elif source_type in ("hub_parquet", "hf_parquet"):
        if not args.dataset:
            log.error("--dataset is required for --source hub_parquet")
            return 1
        src = HfHubParquetSource(
            repo_id=args.dataset,
            split=args.split,
            text_field=args.text_field,
            id_field=args.id_field,
            data_files=args.data_files or None,
            worker_index=int(args.worker_index),
            num_workers=int(args.num_workers),
        )
    elif source_type == "jsonl":
        if not args.paths:
            log.error("--paths is required for --source jsonl")
            return 1
        paths = _paths_from_globs(args.paths)
        paths = _shard_paths(paths, num_workers=args.num_workers, worker_index=args.worker_index)
        if not paths:
            log.error("No input files found for this worker")
            return 1
        src = JSONLSource(paths=paths, text_field=args.text_field, id_field=args.id_field)
    elif source_type == "jsonl_zst":
        if not args.paths:
            log.error("--paths is required for --source jsonl_zst")
            return 1
        paths = _paths_from_globs(args.paths)
        paths = _shard_paths(paths, num_workers=args.num_workers, worker_index=args.worker_index)
        if not paths:
            log.error("No input files found for this worker")
            return 1
        src = JSONLZstSource(paths=paths, text_field=args.text_field, id_field=args.id_field)
    elif source_type in ("arrow", "parquet"):
        if not args.paths:
            log.error("--paths is required for --source arrow/parquet")
            return 1
        paths = _paths_from_globs(args.paths)
        paths = _shard_paths(paths, num_workers=args.num_workers, worker_index=args.worker_index)
        if not paths:
            log.error("No input files found for this worker")
            return 1
        src = ArrowSource(paths=paths, text_field=args.text_field, id_field=args.id_field)
    elif source_type == "text":
        if not args.paths:
            log.error("--paths is required for --source text")
            return 1
        paths = _paths_from_globs(args.paths)
        paths = _shard_paths(paths, num_workers=args.num_workers, worker_index=args.worker_index)
        if not paths:
            log.error("No input files found for this worker")
            return 1
        src = TextFileSource(paths=[Path(p) for p in paths])
    else:
        log.error(f"Unknown --source: {args.source}")
        return 1

    max_tokens_total = _parse_tokens(args.max_tokens_total) if args.max_tokens_total else None
    if max_tokens_total is not None and args.num_workers > 1:
        max_tokens_total = max(1, max_tokens_total // args.num_workers)

    cfg = PrepConfig(
        output_dir=output_dir,
        dataset_name=args.name,
        version=args.version,
        tokenizer=args.tokenizer,
        vocab_size=int(args.vocab_size),
        eos_token_id=int(args.eos_token_id),
        num_shards=int(args.num_shards),
        tokens_per_shard=int(args.tokens_per_shard),
        num_workers=int(args.workers),
        batch_size=int(args.batch_size),
        min_tokens=int(args.min_tokens),
        max_tokens=int(args.max_tokens),
        max_tokens_total=max_tokens_total,
    )

    pipeline = ParallelPrepPipeline(src, cfg) if args.parallel else PrepPipeline(src, cfg)
    manifest = pipeline.run()
    log.info(f"Complete: docs={manifest.total_documents:,} tokens={manifest.total_tokens:,} shards={manifest.num_shards}")
    log.info(f"Manifest: {output_dir / 'manifest.json'}")
    return 0


def cmd_prep_mixture(args: argparse.Namespace) -> int:
    from .mixture import resolve_plan
    from .prep import PrepConfig, PrepPipeline, ParallelPrepPipeline
    from .sources import HuggingFaceSource

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    mixture_path = Path(args.config)
    flow_path = Path(args.flow_profiles)
    if not mixture_path.exists():
        raise SystemExit(f"Mixture file not found: {mixture_path}")
    if not flow_path.exists():
        raise SystemExit(f"Flow profiles file not found: {flow_path}")

    flow = args.flow
    if not flow:
        flow = _autodetect_flow_name(mixture_toml=mixture_path, flow_profiles_toml=flow_path)

    output_root = Path(args.output_root) if args.output_root else (Path("/data/flows") / str(flow))

    plan = resolve_plan(
        mixture_toml=mixture_path,
        flow_profiles_toml=flow_path,
        flow_section=f"flow.{flow}",
        seq_len=int(args.seq_len),
        active_params_b=float(args.active_params_b) if args.active_params_b is not None else None,
    )

    stages = [s for s in plan.stages if s.stage_id == args.stage] if args.stage else list(plan.stages)
    if not stages:
        raise SystemExit(f"No stages found (requested={args.stage!r})")

    sources_to_prep: list[tuple[str, Any]] = []
    for st in stages:
        for sp in st.sources:
            if sp.hf is None:
                continue
            sources_to_prep.append((st.stage_id, sp))
    if not sources_to_prep:
        raise SystemExit("No sources with HuggingFace definitions found in requested stages")

    max_tokens_total = _parse_tokens(args.max_tokens) if args.max_tokens else None
    if max_tokens_total is None:
        flow_tokens_b = sum(s.total_tokens_b for s in stages)
        max_tokens_total = int(flow_tokens_b * 1_000_000_000) if flow_tokens_b > 0 else None

    source_quotas: dict[str, int] = {}
    if max_tokens_total is not None:
        total_target = sum(int(sp.target_tokens) for _, sp in sources_to_prep)
        for _stage_id, sp in sources_to_prep:
            quota = int(max_tokens_total * int(sp.target_tokens) / total_target) if total_target > 0 else 0
            source_quotas[str(sp.id)] = quota

    splits = [s.strip() for s in (args.splits.split(",") if args.splits else ["train"]) if s.strip()]
    if not splits:
        raise SystemExit("splits must be non-empty")

    for stage_id, sp in sources_to_prep:
        hf = sp.hf
        for split in splits:
            if split == "valid":
                if not hf.valid_split:
                    continue
                hf_split = hf.valid_split
            else:
                hf_split = hf.split

            out_dir = output_root / stage_id / str(sp.id) / split
            if args.num_workers > 1:
                out_dir = out_dir / f"worker_{args.worker_index:03d}"
            manifest_path = out_dir / "manifest.json"

            if manifest_path.exists() and not args.force:
                log.info(f"Skipping {stage_id}/{sp.id}/{split} (already exists)")
                continue

            quota = source_quotas.get(str(sp.id))
            if quota is not None and args.num_workers > 1:
                quota = max(1, quota // int(args.num_workers))

            src = HuggingFaceSource(
                dataset=hf.dataset,
                split=hf_split,
                subset=hf.subset,
                text_field=hf.text_field,
                id_field=hf.id_field,
                streaming=True,
                data_files=hf.data_files,
            )
            if args.num_workers > 1:
                src = src.shard(num_shards=int(args.num_workers), index=int(args.worker_index))

            limit_docs = int(args.limit_docs) if args.limit_docs else None
            if limit_docs is not None:
                from .sources import DataSource

                class _LimitedSource(DataSource):
                    def __init__(self, upstream: DataSource, limit: int):
                        self._upstream = upstream
                        self._limit = limit

                    @property
                    def name(self) -> str:
                        return self._upstream.name

                    def estimate_size(self) -> int | None:
                        return None

                    def __iter__(self):
                        yield from _iter_docs_limited(self._upstream, limit_docs=self._limit)

                src = _LimitedSource(src, limit_docs)

            cfg = PrepConfig(
                output_dir=out_dir,
                dataset_name=str(sp.id),
                version=args.version,
                tokenizer=args.tokenizer,
                vocab_size=int(args.vocab_size),
                eos_token_id=int(args.eos_token_id),
                num_shards=int(args.num_shards),
                tokens_per_shard=int(args.tokens_per_shard),
                num_workers=int(args.workers),
                batch_size=int(args.batch_size),
                min_tokens=int(args.min_tokens),
                max_tokens=int(args.max_tokens_per_doc),
                max_tokens_total=quota,
            )

            if args.dry_run:
                log.info(f"[dry-run] would write {out_dir} (quota={quota})")
                continue

            out_dir.mkdir(parents=True, exist_ok=True)
            pipeline = ParallelPrepPipeline(src, cfg) if args.parallel else PrepPipeline(src, cfg)
            manifest = pipeline.run()
            log.info(
                f"Wrote {stage_id}/{sp.id}/{split}: docs={manifest.total_documents:,} tokens={manifest.total_tokens:,} shards={manifest.num_shards}"
            )

    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    from .sinks import verify_manifest

    path = Path(args.manifest)
    ok, errors = verify_manifest(path, check_checksums=bool(args.checksums))
    if ok:
        print("OK")
        return 0
    print(f"FAILED ({len(errors)} errors)")
    for e in errors:
        print(f"- {e}")
    return 1


def cmd_info(args: argparse.Namespace) -> int:
    from .sinks import load_manifest

    m = load_manifest(Path(args.manifest))
    print(f"dataset={m.dataset}")
    print(f"version={m.version}")
    print(f"tokenizer={m.tokenizer}")
    print(f"vocab_size={m.vocab_size}")
    print(f"eos_token_id={m.eos_token_id}")
    print(f"dtype={m.dtype}")
    print(f"created_at={m.created_at}")
    print(f"total_documents={m.total_documents}")
    print(f"total_tokens={m.total_tokens}")
    print(f"num_shards={m.num_shards}")
    return 0


def cmd_regenerate_index(args: argparse.Namespace) -> int:
    from .index import regenerate_index_from_shard

    idx_path = regenerate_index_from_shard(Path(args.shard), eos_token_id=int(args.eos_token_id))
    print(idx_path)
    return 0


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser("nmoe.data")
    sp = p.add_subparsers(dest="cmd", required=True)

    p_prep = sp.add_parser("prep", help="Prep a dataset into .npy shards + manifest.json")
    p_prep.add_argument("--source", default="hf", help="hf|hub_parquet|jsonl|jsonl_zst|arrow|parquet|text")
    p_prep.add_argument("--dataset", help="HF dataset id (for --source hf)")
    p_prep.add_argument("--subset", help="HF subset/config name")
    p_prep.add_argument("--split", default="train", help="HF split (default: train)")
    p_prep.add_argument("--text-field", default="text")
    p_prep.add_argument("--id-field", default=None)
    p_prep.add_argument("--data-files", action="append", default=None, help="HF data_files pattern(s)")
    p_prep.add_argument("--streaming", action="store_true", default=True, help="Use HF streaming (default)")
    p_prep.add_argument("--no-streaming", dest="streaming", action="store_false", help="Disable HF streaming")
    p_prep.add_argument("--paths", nargs="*", help="Input file globs (for file sources)")
    p_prep.add_argument("--num-workers", type=int, default=1, help="Input sharding workers")
    p_prep.add_argument("--worker-index", type=int, default=0, help="Worker index in [0,num-workers)")
    p_prep.add_argument("--output", required=True)
    p_prep.add_argument("--name", required=True, help="Logical dataset name in manifest")
    p_prep.add_argument("--version", default="v1")
    p_prep.add_argument("--tokenizer", default="o200k_harmony")
    p_prep.add_argument("--vocab-size", type=int, default=201088)
    p_prep.add_argument("--eos-token-id", type=int, default=199999)
    p_prep.add_argument("--num-shards", type=int, default=1024)
    p_prep.add_argument("--tokens-per-shard", type=int, default=500_000_000)
    p_prep.add_argument("--workers", type=int, default=8, help="Tokenization workers")
    p_prep.add_argument("--batch-size", type=int, default=1000)
    p_prep.add_argument("--min-tokens", type=int, default=10)
    p_prep.add_argument("--max-tokens", type=int, default=1_000_000)
    p_prep.add_argument("--max-tokens-total", default=None, help="Stop after this many tokens (e.g. 100M)")
    p_prep.add_argument("--parallel", action="store_true", help="Use multiprocess pipeline")
    p_prep.set_defaults(fn=cmd_prep)

    p_mix = sp.add_parser("prep-mixture", help="Prep all HF sources referenced by a mixture+flow")
    p_mix.add_argument("--config", required=True, help="Mixture TOML")
    p_mix.add_argument("--flow-profiles", default="configs/flow_profiles.toml", help="Flow profiles TOML")
    p_mix.add_argument("--flow", default=None, help="Flow name (defaults to autodetect)")
    p_mix.add_argument("--stage", default=None, help="Stage id (e.g. pretrain)")
    p_mix.add_argument("--splits", default="train", help="Comma-separated: train,valid")
    p_mix.add_argument("--output-root", default=None, help="Output root (default: /data/flows/<flow>)")
    p_mix.add_argument("--max-tokens", default=None, help="Override total token budget for this run (e.g. 1B)")
    p_mix.add_argument("--seq-len", type=int, default=4096, help="Plan seq_len for token budgeting")
    p_mix.add_argument("--active-params-b", default=None, help="Active params (B) for tokens_b_ratio flows")
    p_mix.add_argument("--num-workers", type=int, default=1, help="Input sharding workers")
    p_mix.add_argument("--worker-index", type=int, default=0, help="Worker index in [0,num-workers)")
    p_mix.add_argument("--force", action="store_true", help="Rebuild even if manifest exists")
    p_mix.add_argument("--dry-run", action="store_true")
    p_mix.add_argument("--limit-docs", default=None, help="Limit docs per source (for smoke tests)")
    p_mix.add_argument("--version", default="v1")
    p_mix.add_argument("--tokenizer", default="o200k_harmony")
    p_mix.add_argument("--vocab-size", type=int, default=201088)
    p_mix.add_argument("--eos-token-id", type=int, default=199999)
    p_mix.add_argument("--num-shards", type=int, default=1024)
    p_mix.add_argument("--tokens-per-shard", type=int, default=500_000_000)
    p_mix.add_argument("--workers", type=int, default=8)
    p_mix.add_argument("--batch-size", type=int, default=1000)
    p_mix.add_argument("--min-tokens", type=int, default=10)
    p_mix.add_argument("--max-tokens-per-doc", type=int, default=1_000_000)
    p_mix.add_argument("--parallel", action="store_true")
    p_mix.set_defaults(fn=cmd_prep_mixture)

    p_verify = sp.add_parser("verify", help="Verify manifest and shard checksums")
    p_verify.add_argument("manifest")
    p_verify.add_argument("--checksums", action="store_true", help="Verify shard checksums (slower)")
    p_verify.set_defaults(fn=cmd_verify)

    p_info = sp.add_parser("info", help="Print manifest summary")
    p_info.add_argument("manifest")
    p_info.set_defaults(fn=cmd_info)

    p_idx = sp.add_parser("regenerate-index", help="Regenerate .idx from a .npy shard")
    p_idx.add_argument("shard")
    p_idx.add_argument("--eos-token-id", type=int, default=199999)
    p_idx.set_defaults(fn=cmd_regenerate_index)

    ns = p.parse_args(argv)
    return int(ns.fn(ns))


if __name__ == "__main__":
    raise SystemExit(main())
