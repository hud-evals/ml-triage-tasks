# Morning standup — 2026-03-09

**Attendees:** Arjun, Priya

**Priya:**
- still can't repro openai_3large. the npy files look the right shape
  and the norms are fine but the cosine similarities are garbage.
  I have a hunch it's a row-order thing.

**Arjun:**
- agree with your hunch. I'll add a note to final_recommendation.md
  that 3-large is not-yet-verified. don't ship it.

**Priya:**
- ok. also I'm going to write up the tie-break bug in a ticket with
  evidence. eval_v2 is not equivalent to eval on integer-scored
  scorers. repro is in `scratch/tiebreak_repro.py`.

**Arjun:**
- thanks. park that for now, focus on the leadership review.
