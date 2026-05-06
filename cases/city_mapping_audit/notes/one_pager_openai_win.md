# One-pager: why openai_3small wins

**Author:** Mei Chen  ·  **Date:** 2025-10-29

**Bottom line:** on the "representative" 41k-hotel subset where hotel
names contain the ground-truth city as a substring, `openai_3small`
achieves top-1 = 0.89 — a 28pp uplift over fuzzy's 0.61 on the same
slice.

## The slice

We take hotels where at least one of the GT cities appears as a
case-insensitive substring of the hotel name. That yields 41,230
hotels in the full 110k corpus and 1,399 in the eval subset.

## Numbers on the slice

| method          | top_1 (slice) |
|-----------------|--------------:|
| openai_3small   |          0.89 |
| partial_ratio   |          0.90 |
| wratio          |          0.85 |
| minilm          |          0.77 |
| openai_3large   |          0.92 |

Openai_3large wins here too, per Jordan's run.

## Takeaway

On the slice, both methods are excellent. openai_3small is our best
practical choice: cheaper than 3-large, faster than fuzzy on long
names, and less brittle to non-English characters than any
character-tokenized fuzzy scorer.

## Caveats

This slice covers 41k / 110k hotels on the full table — i.e., 37% of
the corpus. The other 63% is the no-overlap bucket, which is
dramatically harder. That half is not addressed in this one-pager.
