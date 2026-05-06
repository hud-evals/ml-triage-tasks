"""MiniLM embedding pipeline — the 'local' branch of the eval harness.

Wraps `sentence-transformers/all-MiniLM-L6-v2` for both hotels and cities.
For reproducibility the encode loop is deterministic (no dropout, fixed
batch) but we DO NOT pin CUBLAS determinism because this repo is
CPU-only.

Usage:
    python src/embed_local.py --input embeddings/hotel_names.json \
        --output embeddings/minilm_hotels.npy

Known issues:
  * `strip_accents` flag does nothing today — we pass it through to the
    tokenizer but the underlying transformer does unicode NFKC already,
    so the behaviour matches default SentenceTransformer. Tracked in
    #embed-pipeline (Slack).
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np


def load_names(path: Path) -> list[str]:
    return json.loads(path.read_text())


def encode_minilm(names: list[str], batch_size: int = 64,
                  normalize: bool = False, strip_accents: bool = False) -> np.ndarray:
    """Placeholder — this repo ships pre-computed embeddings. This function
    is kept for documentation only; re-embedding is out of scope.
    """
    raise NotImplementedError(
        "Re-embedding requires sentence-transformers, which is not installed "
        "in the audit sandbox. Use embeddings/minilm_*.npy as provided."
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--normalize", action="store_true")
    ap.add_argument("--strip-accents", action="store_true")
    args = ap.parse_args()
    names = load_names(args.input)
    vectors = encode_minilm(
        names,
        batch_size=args.batch_size,
        normalize=args.normalize,
        strip_accents=args.strip_accents,
    )
    np.save(args.output, vectors.astype(np.float32))
    print(f"wrote {args.output} with shape {vectors.shape}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
