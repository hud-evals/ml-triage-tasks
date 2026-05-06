# PR #088 — First openai_3large embeddings

**Author:** Jordan  ·  **Merged:** 2025-09-05  ·  **Reviewer:** Mei

## Summary

Adds the first openai_3large embedding batch for the hotel corpus and
the city corpus. Also the eval harness config.

## Artifacts

    embeddings/openai3large_hotels.npy   (110160, 1536) f32
    embeddings/openai3large_cities.npy   (18942,  1536) f32
    configs/openai3large.yaml

## Validation

End-to-end on the 3k eval subset: top_1 = 0.698 per
`runs/openai_3large/run1.json`. That's ~20 pp over openai_3small —
worth a deeper look before we migrate.

## Known caveats

- Spot-checked 10 hotels by hand; all 10 returned plausible cities.
- Did not run the full eval twice end-to-end. Next step.
