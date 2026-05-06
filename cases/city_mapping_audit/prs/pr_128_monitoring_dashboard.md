# PR #128 — DRAFT: post-ship top-K monitoring dashboard

**Author:** Priya  ·  **Status:** draft  ·  **Reviewer:** —

## Summary

Adds a weekly-refresh dashboard for top-K on a fresh sample of
5k hotels from the prior week's bookings. Detects drift.

Metrics: top-1, top-3, per-bucket (lexical_overlap).

## Data source

`SELECT hotel_name, dim_hotel_city FROM bookings_raw WHERE booked_at >
NOW() - INTERVAL '7 days' LIMIT 5000`.

## Alert thresholds

- top-1 drop >= 3 pp vs the 3k-subset baseline: alert.
- top-3 drop >= 2 pp vs the 3k-subset baseline: alert.

## Risk

None — read-only, analytics table.
