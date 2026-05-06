"""Canonical top-K evaluation against gt.json.

Usage:
    python src/eval.py --config configs/<method>.yaml
    python src/eval.py --config configs/<method>.yaml --json > out.json

Outputs top_1 / top_2 / top_3 accuracy on the full eval subset
(hotels whose GT cities are all present in the city index).
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent


def _cosine_topk(h: np.ndarray, c: np.ndarray, k: int) -> np.ndarray:
    h = h / (np.linalg.norm(h, axis=1, keepdims=True) + 1e-12)
    c = c / (np.linalg.norm(c, axis=1, keepdims=True) + 1e-12)
    sims = h @ c.T
    part = np.argpartition(-sims, kth=k, axis=1)[:, :k]
    sub = np.take_along_axis(sims, part, axis=1)
    order = np.argsort(-sub, axis=1)
    return np.take_along_axis(part, order, axis=1)


def _fuzzy_topk(hotels: list[str], cities: list[str], scorer: str, k: int) -> np.ndarray:
    from rapidfuzz import fuzz, process
    fn = {"partial_ratio": fuzz.partial_ratio, "WRatio": fuzz.WRatio}[scorer]
    out = np.zeros((len(hotels), k), dtype=np.int32)
    for i, h in enumerate(hotels):
        for j, (_, _, idx) in enumerate(process.extract(h, cities, scorer=fn, limit=k)):
            out[i, j] = idx
    return out


def topk_accuracy(topk: np.ndarray, hotels: list[str], cities: np.ndarray,
                  gt: dict[str, list[str]], k: int) -> float:
    preds = cities[topk[:, :k]]
    hits = 0
    for i, h in enumerate(hotels):
        if set(preds[i]) & set(gt[h]):
            hits += 1
    return round(hits / len(hotels), 4)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    method = cfg["method"]
    hotels = json.loads((ROOT / "embeddings" / "hotel_names.json").read_text())
    cities = np.array(json.loads(
        (ROOT / "embeddings" / "city_names.json").read_text()))
    gt = json.loads((ROOT / cfg.get("gt_path", "ground_truth/gt.json")).read_text())
    gt = {h: gt[h] for h in hotels}

    if cfg.get("embedding_hotels_path"):
        hv = np.load(ROOT / cfg["embedding_hotels_path"])
        cv = np.load(ROOT / cfg["embedding_cities_path"])
        topk = _cosine_topk(hv, cv, k=3)
    else:
        topk = _fuzzy_topk(hotels, cities.tolist(), cfg["scorer"], k=3)

    result = {
        "method": method,
        "eval_script": "src/eval.py",
        "eval_n": len(hotels),
        "gt_path": cfg.get("gt_path", "ground_truth/gt.json"),
        "top_1": topk_accuracy(topk, hotels, cities, gt, 1),
        "top_2": topk_accuracy(topk, hotels, cities, gt, 2),
        "top_3": topk_accuracy(topk, hotels, cities, gt, 3),
    }
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"{method}: top1={result['top_1']}  top2={result['top_2']}  "
              f"top3={result['top_3']}  n={result['eval_n']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
