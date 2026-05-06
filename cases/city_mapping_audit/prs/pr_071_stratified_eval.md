# PR #071 — stratified eval harness (lexical_overlap axis)

**Author:** Priya  ·  **Status:** merged  ·  **Reviewer:** Arjun

## Summary

Adds `src/stratify.py` with a pluggable axis API. Only
`lexical_overlap` is wired up. Future axes (`name_length`,
`city_frequency`) are stubs.

## Schema

Stratified CSVs land under `runs/stratified/` with columns:

    axis, bucket, method, n, top_1, top_2, top_3

See `runs/stratified/lexical_overlap.csv` for the first real output.

## Follow-ups

- #stratify-axes: wire up `name_length` and `city_frequency`.
