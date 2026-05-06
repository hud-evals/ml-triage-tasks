# One-pager: where fuzzy actually wins

**Author:** Priya Joshi  ·  **Date:** 2025-10-02

Contra Mei's `notes/onepager_fuzzy_rejected.md`, fuzzy methods are not
dead. On the no-overlap bucket they collapse to <5% top-1, but on the
overlap bucket (1,399 hotels in the 3k subset) `partial_ratio` actually
edges out `openai_3small` at top-3 (0.998 vs 0.938).

## Where fuzzy wins

- **Short ASCII names with clean city substring.** Think "Paris Marriott"
  where "Paris" is right there.
- **Non-Latin scripts? Depends.** Counter to popular belief, fuzzy's
  behaviour on non-Latin names is fine when the name has a transliterated
  city in it.

## My recommendation

Ship openai_3small as the primary ranker, but route short-ASCII
overlap-positive names through fuzzy first and only fall back to
embeddings on a miss. That's a 1-line routing rule and picks up
the 1 pp top-1 we're leaving on the table.

## Disagreement with Mei

Mei's one-pager cites a 95% miss rate for fuzzy on non-English names. I
have NOT been able to reproduce that number from any artifact in this
repo. The actual partial_ratio miss rate on the canonical 3k subset is
~55% (top-1 ≈ 0.44). The 95% figure appears to come from an "internal
hard names subset" that I cannot find.
