# PR #066 — DRAFT: merge train/test booking tables

**Author:** Mei  ·  **Status:** draft / not merged  ·  **Reviewer:** —

## Problem

`data/bookings.parquet` currently only contains one split. We used to
have train/test tables but consolidated them months ago. PR adds back
a `split` column for reproducibility.

## Non-blocker

Not required for the audit. Filed for completeness.
