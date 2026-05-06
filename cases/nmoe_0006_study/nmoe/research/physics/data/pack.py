"""
Pack synthetic samples into .npy shards for nmoe training.

Streams SyntheticMix → pads to seq_len+1 (no truncation) → ShardedWriter → .npy + .idx + manifest.json

Usage:
    python -m nmoe.research.physics.data.pack \
        --output /data/physics/depo-mano-v1 \
        --dataset depo-mano \
        --tasks depo:1.0:n_entities=100,max_hops=8 mano:1.0:depth=5 \
        --n-train 100000 \
        --n-valid 1000 \
        --seq-len 512 \
        --seed 42
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Iterator

import numpy as np

from nmoe.data.sinks import ShardedWriter
from nmoe.research.physics.data.generators import (
    Sample, SyntheticMix, depo, brevo, mano,
    BOS, EOS, ANSWER_START,
)


# Synthetic vocab config
# All synthetic tokens must be < SYNTHETIC_VOCAB_SIZE
SYNTHETIC_VOCAB_SIZE = 10240
SYNTHETIC_EOS = EOS  # 9998


def stream_samples(mix: SyntheticMix, n: int, rng: random.Random) -> Iterator[Sample]:
    """Generate exactly n samples from mix."""
    gen = mix.stream(rng)
    for _ in range(n):
        yield next(gen)


def parse_task_spec(spec: str) -> tuple[str, float, dict]:
    """Parse task spec like 'depo:1.0:n_entities=100,max_hops=8'."""
    parts = spec.split(":")
    task_name = parts[0]
    weight = float(parts[1]) if len(parts) > 1 else 1.0
    kwargs = {}
    if len(parts) > 2 and parts[2]:
        for kv in parts[2].split(","):
            k, v = kv.split("=")
            # Try to parse as int, fall back to string
            try:
                kwargs[k] = int(v)
            except ValueError:
                kwargs[k] = v
    return task_name, weight, kwargs


def pack_split(
    output_dir: Path,
    dataset: str,
    version: str,
    mix: SyntheticMix,
    n_samples: int,
    seq_len: int,
    seed: int,
    split: str,
    tokens_per_shard: int = 100_000_000,
) -> dict:
    """Pack one split (train or valid) to shards."""
    split_dir = output_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)

    source_info = {
        "generator": "nmoe.research.physics.data.pack",
        "split": split,
        "n_samples": n_samples,
        "seq_len": seq_len,
        "seed": seed,
        "tasks": [
            {"name": name, "weight": weight, "kwargs": kwargs}
            for name, weight, kwargs in mix.tasks
        ],
    }

    writer = ShardedWriter(
        output_dir=split_dir,
        dataset=dataset,
        version=version,
        eos_token_id=SYNTHETIC_EOS,
        vocab_size=SYNTHETIC_VOCAB_SIZE,
        tokenizer="synthetic",
        tokens_per_shard=tokens_per_shard,
        source_info=source_info,
    )

    for i, sample in enumerate(stream_samples(mix, n_samples, rng)):
        # Pad to seq_len + 1 (for input/target alignment); refuse to truncate.
        tokens = pad_or_truncate(sample.tokens, seq_len + 1, pad_token=EOS)

        # Validate all tokens are in vocab
        max_tok = max(tokens)
        if max_tok >= SYNTHETIC_VOCAB_SIZE:
            raise ValueError(
                f"Token {max_tok} >= vocab_size {SYNTHETIC_VOCAB_SIZE} "
                f"in sample {i} ({sample.task})"
            )

        # Write without appending EOS (we handle padding ourselves)
        writer.add_document(tokens, append_eos=False)

        if (i + 1) % 10000 == 0:
            print(f"  {split}: {i + 1}/{n_samples} samples", file=sys.stderr)

    manifest = writer.finalize()
    print(
        f"  {split}: {manifest.total_documents} docs, "
        f"{manifest.total_tokens:,} tokens, "
        f"{manifest.num_shards} shards",
        file=sys.stderr,
    )
    return manifest.to_dict()


def main():
    parser = argparse.ArgumentParser(
        description="Pack synthetic samples into .npy shards",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Depo-only dataset
    python -m nmoe.research.physics.data.pack \\
        --output /data/physics/depo-v1 \\
        --dataset depo \\
        --tasks "depo:1.0:n_entities=100,max_hops=8" \\
        --n-train 100000 --n-valid 1000 --seq-len 512

    # Ngram (Markov order-2) dataset
    python -m nmoe.research.physics.data.pack \\
        --output /data/physics/ngram-v1 \\
        --dataset ngram \\
        --tasks "ngram:1.0:n_symbols=512,n_steps=128,table_seed=0" \\
        --n-train 100000 --n-valid 1000 --seq-len 256

    # Mixed Depo + Mano
    python -m nmoe.research.physics.data.pack \\
        --output /data/physics/depo-mano-v1 \\
        --dataset depo-mano \\
        --tasks "depo:1.0:n_entities=100,max_hops=8" "mano:1.0:depth=5,ops=asm" \\
        --n-train 100000 --n-valid 1000 --seq-len 512
        """,
    )
    parser.add_argument("--output", type=Path, required=True, help="Output directory")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--version", type=str, default="v1", help="Dataset version")
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        required=True,
        help="Task specs: 'name:weight:k1=v1,k2=v2'",
    )
    parser.add_argument("--n-train", type=int, required=True, help="Number of training samples")
    parser.add_argument("--n-valid", type=int, required=True, help="Number of validation samples")
    parser.add_argument("--seq-len", type=int, default=512, help="Sequence length (default: 512)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--tokens-per-shard",
        type=int,
        default=100_000_000,
        help="Tokens per shard (default: 100M)",
    )

    args = parser.parse_args()

    # Build task mix
    mix = SyntheticMix(seed=args.seed)
    for spec in args.tasks:
        name, weight, kwargs = parse_task_spec(spec)
        mix.add(name, weight, **kwargs)

    print(f"Packing {args.dataset} to {args.output}", file=sys.stderr)
    print(f"Tasks: {args.tasks}", file=sys.stderr)
    print(f"Train: {args.n_train}, Valid: {args.n_valid}, SeqLen: {args.seq_len}", file=sys.stderr)

    args.output.mkdir(parents=True, exist_ok=True)

    # Pack train split (use seed for train, seed+1000000 for valid for separation)
    train_manifest = pack_split(
        output_dir=args.output,
        dataset=args.dataset,
        version=args.version,
        mix=mix,
        n_samples=args.n_train,
        seq_len=args.seq_len,
        seed=args.seed,
        split="train",
        tokens_per_shard=args.tokens_per_shard,
    )

    # Pack valid split with different seed
    valid_manifest = pack_split(
        output_dir=args.output,
        dataset=args.dataset,
        version=args.version,
        mix=mix,
        n_samples=args.n_valid,
        seq_len=args.seq_len,
        seed=args.seed + 1_000_000,
        split="valid",
        tokens_per_shard=args.tokens_per_shard,
    )

    # Write combined config
    config = {
        "dataset": args.dataset,
        "version": args.version,
        "seq_len": args.seq_len,
        "vocab_size": SYNTHETIC_VOCAB_SIZE,
        "eos_token_id": SYNTHETIC_EOS,
        "seed": args.seed,
        "tasks": [parse_task_spec(s) for s in args.tasks],
        "train": train_manifest,
        "valid": valid_manifest,
    }

    config_path = args.output / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nDone. Config: {config_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
def pad_or_truncate(tokens: list[int], length: int, pad_token: int = EOS) -> list[int]:
    """
    Pad token sequence to exact length.

    We intentionally do *not* truncate: truncation can silently drop `ANSWER_START` or
    answer tokens, corrupting verification and making results uninterpretable.
    If a sample does not fit, adjust generator difficulty (e.g. n_nodes/depth) or
    increase `--seq-len`.
    """
    if len(tokens) > length:
        raise ValueError(
            f"Synthetic sample length {len(tokens)} exceeds target length {length}. "
            "Refusing to truncate; adjust generator parameters or increase --seq-len."
        )
    return tokens + [pad_token] * (length - len(tokens))
