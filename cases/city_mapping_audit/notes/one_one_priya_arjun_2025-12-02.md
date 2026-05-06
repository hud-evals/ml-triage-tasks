# 1:1 notes — Priya / Arjun, 2025-12-02

**Agenda:** post-ship monitoring, openai_3large, chain-KB scoping.

- Shipped openai_3small 2025-11-15 as planned. Top-K on a fresh
  weekly sample is within 0.3 pp of the 3k-subset eval. Healthy.
- openai_3large: Priya looked at the npy files again. Confirmed the
  row-order hypothesis — a quick probe showed the hotel vector at
  row 0 aligns with city vector at row ~1200, not row 0. So
  `openai3large_*.npy` are shuffled.
- Re-embed decision: Arjun has the $3 budget. Target Q1 to resolve.
- Chain-KB vendor scoping: three options on the table, early days.

**Follow-ups:**
- Priya: write up the row-order confirmation as a comment on
  `runs/openai_3large/run1.json` (or a standalone note).
- Arjun: ADR-003 draft for chain-KB direction by EOQ.
