# ADR-004: Chain-KB integration — scoping, not a decision yet

**Status:** draft / research  ·  **Author:** Arjun  ·  **Date:** 2025-12-01

## Context

Post-ship analysis confirms the no-overlap bucket (1601 / 3000 hotels
in the subset, ~53%) is the structural ceiling for name-only
retrieval. Best top-1 on that slice is 12%. To break past it we need
external signal.

The cheapest external signal we have line-of-sight to is a chain
knowledge-base: given that a hotel's name starts with "Marriott" (or
"Hilton" etc.), look up Marriott's published property list and filter
candidate cities to the ones with a Marriott property.

## Candidates

1. **Build internally** — scrape / license per-chain property lists.
   Low-medium lift, ongoing maintenance.
2. **Vendor: HotelKB Inc.** — licensed data, monthly refresh.
   Reasonable price, medium trust.
3. **Vendor: BrandAtlas** — similar to HotelKB, higher price, higher
   trust (they already have a relationship with several chain HQs).

## Next

- Priya scopes options 1 + 2; Arjun scopes option 3.
- Decision target: 2026-02-15 for ADR-004 final.
