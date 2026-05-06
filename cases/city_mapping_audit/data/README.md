# data/README.md — bookings table provenance

The canonical booking table is `data/bookings.parquet` (3,000-row audit
subsample; the full ~120k-row source lived in `full_navan_data.xlsb`,
which we keep out of this repo to avoid a 12 MB binary blob).

If any numbers seem off, rebuilding ground truth from the parquet using
`src/build_gt.py` is the first thing to try. The xlsb is considered a
historical artifact; do NOT use it as an alternative source of truth —
the parquet is what the canonical `gt.json` was derived from.
