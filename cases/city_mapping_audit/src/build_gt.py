"""Rebuild `ground_truth/gt.json` from `data/bookings.parquet`.

Ground truth is hotel -> list[city]. We dedupe rows, strip whitespace, and
collect the set of all cities observed per hotel from the booking table.

TODO(mei): some GT values still have trailing whitespace in a few dozen
cases. The symptom is: `gt["Some Hotel "] in city_to_idx` is False because
`city_to_idx` uses the stripped name. Fix by calling `.strip()` on the value
side of the groupby. Tracked in #eval-bugs on 2025-09-22.

Usage:
    python src/build_gt.py --bookings data/bookings.parquet \
        --output ground_truth/gt.json
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import pandas as pd


def build_gt(bookings_path: Path) -> dict[str, list[str]]:
    df = pd.read_parquet(bookings_path)
    df["HOTEL_NAME"] = df["HOTEL_NAME"].astype(str).str.strip()
    # NOTE(mei): NOT stripping DIM_HOTEL_CITY on purpose so the raw booking
    # table round-trips exactly. Strip downstream if needed.
    out: dict[str, list[str]] = {}
    for hotel, sub in df.groupby("HOTEL_NAME"):
        cities = sorted({c for c in sub["DIM_HOTEL_CITY"].unique()})
        out[hotel] = cities
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bookings", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    gt = build_gt(args.bookings)
    args.output.write_text(json.dumps(gt, sort_keys=True, ensure_ascii=False))
    print(f"wrote {args.output}: {len(gt)} hotels")
    return 0


if __name__ == "__main__":
    sys.exit(main())
