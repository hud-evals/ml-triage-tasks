from __future__ import annotations

import argparse
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

EVAL_BUNDLE_URL = "https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip"


def _download(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with urllib.request.urlopen(url) as r, tmp.open("wb") as f:
        shutil.copyfileobj(r, f)
    tmp.replace(out_path)


def _extract_eval_bundle(zip_path: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="nmoe_eval_bundle_") as tmpdir:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(tmpdir)
        extracted = Path(tmpdir) / "eval_bundle"
        if not extracted.is_dir():
            raise RuntimeError("zip does not contain expected top-level 'eval_bundle/' directory")
        final = out_dir / "eval_bundle"
        if final.exists():
            shutil.rmtree(final)
        shutil.move(str(extracted), str(final))
    return out_dir / "eval_bundle"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser("nmoe.eval.core.prep")
    ap.add_argument("--url", default=EVAL_BUNDLE_URL)
    ap.add_argument("--out-dir", default="/data/eval", help="Writes {out_dir}/eval_bundle/...")
    ap.add_argument("--force", action="store_true", help="Overwrite existing bundle")
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir)
    bundle_dir = out_dir / "eval_bundle"
    if bundle_dir.exists():
        if not args.force:
            print(f"[core/prep] exists: {bundle_dir} (use --force to overwrite)")
            return
        shutil.rmtree(bundle_dir)

    zip_path = out_dir / "eval_bundle.zip"
    print(f"[core/prep] downloading: {args.url}")
    _download(str(args.url), zip_path)
    print(f"[core/prep] downloaded: {zip_path} ({zip_path.stat().st_size / (1024 * 1024):.1f} MiB)")

    final = _extract_eval_bundle(zip_path, out_dir)
    print(f"[core/prep] ready: {final}")


if __name__ == "__main__":  # pragma: no cover
    main()
