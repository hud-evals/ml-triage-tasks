"""OpenAI embedding pipeline — used for openai_3small and openai_3large.

The small branch is production. The large branch was handed off by a
contractor and has not reproduced since — see `notes/slack_embeddings_thread.md`
and PR #088.

We batch at 128 inputs per request (below OpenAI's 2048 limit) to keep p95
latency under 5 seconds. Retries are linear-backoff with jitter.

Usage:
    python src/embed_openai.py --model text-embedding-3-small \
        --input embeddings/city_names.json --output embeddings/openai3small_cities.npy

See `cost_table.md` for the cost implications of a full re-embedding.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np


def load_names(path: Path) -> list[str]:
    return json.loads(path.read_text())


def encode_openai(names: list[str], model: str, batch_size: int = 128) -> np.ndarray:
    """Placeholder — calls the OpenAI embeddings endpoint in production.

    Not runnable from the audit sandbox (no network / no API key). The
    committed npy files under embeddings/ are the authoritative record.
    """
    raise NotImplementedError(
        "Re-embedding requires the OpenAI API, which is unavailable in the "
        "audit sandbox. Inspect embeddings/openai3small_*.npy and "
        "embeddings/openai3large_*.npy to understand the shipped state."
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    choices=("text-embedding-3-small", "text-embedding-3-large"))
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=128)
    args = ap.parse_args()
    names = load_names(args.input)
    vectors = encode_openai(names, model=args.model, batch_size=args.batch_size)
    np.save(args.output, vectors.astype(np.float32))
    print(f"wrote {args.output} with shape {vectors.shape}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
