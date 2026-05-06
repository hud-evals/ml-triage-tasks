# #eval-bugs (pasted transcript, 2025-10-18 + follow-ups)

**priya.j**  11:04 AM
hey mei — I'm seeing that `eval.py` and `eval_v2.py` give slightly
different top-1 numbers on the fuzzy runs. about 1 pp difference. is
that expected?

**mei.c**  11:06 AM
eval_v2 was a cleanup PR (PR #042) — same semantics. I'm surprised
you're seeing anything. can you show me the diff?

**priya.j**  11:09 AM
attached. both on the 3k eval subset, partial_ratio scorer:

    eval.py:     top_1=0.4407  top_2=0.4913  top_3=0.4997
    eval_v2.py:  top_1=0.4523  top_2=0.4985  top_3=0.4997

top_3 is identical but top_1 disagrees by 1.16pp. top_2 by 0.72pp.

**mei.c**  11:12 AM
hmm that's more than I'd expect. can you confirm you're using the same
gt.json on both?

**priya.j**  11:14 AM
yep, same `ground_truth/gt.json`. same hotel list. only the script
differs.

**mei.c**  11:18 AM
weird. tie-break stability maybe — rapidfuzz emits duplicate scores on
integer scorers like partial_ratio. I'll look at it when I have a
chance. for now keep using `eval.py` for anything that lands in
`runs/`.

**jordan.r**  12:40 PM
eval_v2 looks fine to me, I've been using it for openai runs and the
numbers match eval.py within 0.001 on what I've spot-checked.

**mei.c**  12:41 PM
that's float cosine, very few ties. priya's case is integer partial_
ratio which hits tons of ties on short names.

**mei.c**  12:45 PM
ok briefly glanced at eval_v2.py. the refactor collects the top-K*5
hits then "smooths" ties by picking the max-index within each
score group. eval.py uses stable argpartition + argsort which picks
the first tie. so for integer scorers these diverge systematically.
priya, your number is real.

**priya.j**  12:47 PM
so should we revert eval_v2?

**mei.c**  12:50 PM
…ugh. I don't want to churn pipelines right before the leadership
review. and jordan's openai runs through eval_v2 are fine because
float cosine. let me think.

**jordan.r**  12:55 PM
honestly I'd just leave it. it's a 1 pp delta on fuzzy only, and fuzzy
isn't our recommended method anyway. if we ever do ship fuzzy it's as
a short-ASCII fallback where the tie-break direction doesn't really
matter for the product.

**mei.c**  12:56 PM
ok ok. tentatively keep both scripts. but flag the divergence in a
ticket so we don't forget.

**priya.j**  2:30 PM
…ok. but note for the record, something is up at least on integer-
scored scorers. filing a ticket.

**priya.j**  2:33 PM
ticket is #eval-v2-tie-break. unassigned.

**— follow-up, 2025-10-25 —**

**priya.j**  9:10 AM
adding a repro script under `scratch/tiebreak_repro.py` for anyone
who wants to poke at this later. also — question: which scripts does
each run JSON under `runs/` use? I want to make sure the numbers in
reports match the canonical eval.

**mei.c**  9:15 AM
good question. the `eval_script` field in each run JSON is
authoritative. my convention: eval.py for everything ship-worthy,
eval_v2.py for anything experimental.

**priya.j**  9:20 AM
ok scanning runs/. I see `runs/openai_3small/run2.json` uses
eval_v2.py. is that intentional?

**mei.c**  9:22 AM
…it should use eval.py. can you check when that was produced?

**priya.j**  9:23 AM
git log on the file says 2025-10-14 17:42 from your user. looks like
an accidental checkpoint?

**mei.c**  9:25 AM
oh no. that'd mean the numbers in run2 are inflated by the tie-break
thing. let me look.

**mei.c**  9:40 AM
confirmed — run2's top-1 is 0.4813 on openai_3small, and re-running
with eval.py gives 0.4687. that's within the ~1 pp delta you'd expect
from the tie-break behaviour. so run2 is junk.

**priya.j**  9:42 AM
delete it?

**mei.c**  9:45 AM
…we can't really delete committed history cleanly and I want the audit
trail. let me leave run2 in place and mark it in the ADR, but the
canonical number stays run1.

**priya.j**  9:46 AM
ok but this is the kind of thing the Q4 review deck should not miss.

**mei.c**  9:47 AM
yes. noted. I'll flag it in the draft.

**— follow-up, 2025-10-30 —**

**arjun.p**  11:00 AM
mei, the ADR draft doesn't mention the eval_v2 situation. is that
intentional?

**mei.c**  11:02 AM
…good catch. will add a footnote. the headline is still "ship openai_
3small based on run1.json numbers" but readers should know run2 exists
and is inflated.

**arjun.p**  11:03 AM
thx.

**mei.c**  11:30 AM
footnote added. also flagged the 3-large situation in the same ADR.

— end of channel —
