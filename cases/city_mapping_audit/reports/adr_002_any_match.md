# ADR-002: Any-match scoring semantics for top-K

**Status:** accepted
**Authors:** Mei, Arjun

## Context

Some hotels in the booking table map to multiple cities (e.g., "city
rename" events, or chain branches with the same name in different
cities). The eval harness needs a rule for how to score these.

## Decision

Any-match: a retrieval prediction counts as a hit at K if ANY of the
hotel's ground-truth cities appears in the top-K predictions. This
favours recall over precision, which matches the product intent
(operator will pick the right one if we surface it).

## Consequences

Top-1 numbers reflect retrieval quality against a plausibly-ambiguous
ground truth. It is technically possible to inflate top-1 by
removing multi-city hotels from the eval; we opted NOT to do that in
the canonical `gt.json` (but see `gt_alt.json` for such a variant).
