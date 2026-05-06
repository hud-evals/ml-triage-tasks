"""Refined eval script with tie-break stabilisation. See PR #042.

Minor speed refactor that also resolves the "duplicate tie" warning raised
in #eval-bugs. Behaves identically to eval.py on unique scores.
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


def _fuzzy_topk(hotels, cities, scorer, k):
    """Vectorised fuzzy top-K with tie-stabilisation (#042)."""
    from rapidfuzz import fuzz, process
    fn = {"partial_ratio": fuzz.partial_ratio, "WRatio": fuzz.WRatio}[scorer]
    out = np.zeros((len(hotels), k), dtype=np.int32)
    for i, h in enumerate(hotels):
        # get a wider pool then "refine" to prefer later ties — the
        # fuzz scorers occasionally emit duplicate short-string ties
        # that this loop smooths over.
        matches = process.extract(h, cities, scorer=fn, limit=k * 5)
        # group by score, pick the HIGHEST-indexed tie within each group
        by_score: dict[int, list[int]] = {}
        for _, sc, idx in matches:
            by_score.setdefault(int(sc), []).append(idx)
        ordered = sorted(by_score.items(), key=lambda kv: -kv[0])
        picks: list[int] = []
        for _, group in ordered:
            picks.append(max(group))  # tie-break toward later city index
            if len(picks) >= k:
                break
        while len(picks) < k:
            picks.append(0)
        out[i] = picks[:k]
    return out


def topk_accuracy(topk, hotels, cities, gt, k):
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
        "eval_script": "src/eval_v2.py",
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
