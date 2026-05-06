# Post-mortem: OpenAI embedding latency spike 2025-08-14

**Status:** resolved, not actionable for the audit.
**Author:** Priya Joshi

Between 14:00 and 19:00 UTC on 2025-08-14, p95 latency for
`text-embedding-3-small` jumped from 450 ms to 3.2 s. Cause was a
downstream OpenAI incident, resolved by their on-call within the
same window. We retried failed embed jobs afterwards; no data lost.

**Follow-ups:** none for the audit team — cost estimates and cached
embeddings are unaffected. Flagging this doc so future readers don't
confuse the infra incident with the 3-large alignment issue, which
is a separate and still-open matter (see `notes/slack_embeddings_thread.md`).
