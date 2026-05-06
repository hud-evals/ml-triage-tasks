# #ground-truth (pasted transcript, 2025-10-22 + follow-ups)

**mei.c**  9:10 AM
I want cleaner numbers for the leadership review next week. the multi-
city hotels are dragging top-1 because `any-match` counts but the first
prediction is still wrong a lot of the time. of the 3000 hotels in the
subset, 37 have more than one GT city.

**arjun.p**  9:18 AM
what's the population on the full corpus?

**mei.c**  9:19 AM
1421 out of 110k in the full hotel_names index. so ~1.3% of the corpus.

**arjun.p**  9:20 AM
would it be dishonest to drop them from the eval just for the review?

**mei.c**  9:21 AM
…maybe. let me build a `ground_truth/gt_alt.json` that drops them and
we can show both. if the review asks for the number I'll give the one
against the canonical gt.json, not the alt.

**priya.j**  9:22 AM
I'd really rather not fork the GT this close to a decision. what is
the actual gain?

**mei.c**  9:24 AM
ran it — top-1 for openai_3small goes from 0.4687 to 0.4812. about
1.3 pp uplift. so it's visible.

**priya.j**  9:25 AM
I don't love that. if we report 0.4812 in the deck anyone cross-
referencing against run1.json gets a different number.

**arjun.p**  9:28 AM
mei, can you label any runs that use gt_alt clearly? don't want future-us
confused.

**mei.c**  9:30 AM
yep, adding a `gt_path` field to run JSONs. default is
`ground_truth/gt.json`; gt_alt runs carry `ground_truth/gt_alt.json`.
I'll also leave a note in reports/ that says "use the canonical
gt.json number".

**arjun.p**  9:32 AM
thanks.

**mei.c**  9:45 AM
done. `ground_truth/gt_alt.json` has 2963 hotels (dropped the 37
multi-city cases). top-1 uplift 1.3pp, top-3 uplift 0.97pp. if leadership
asks, the canonical number is the gt.json one.

**priya.j**  9:47 AM
I'm still going to flag this in the stratified PR's risk section.

**mei.c**  9:48 AM
fine by me.

**— follow-up, 2025-11-05 —**

**priya.j**  10:05 AM
reviewing runs/ for the Q4 review — I see `runs/openai_3small/run3.json`
carries `gt_path: ground_truth/gt_alt.json`. was that the alt-eval you
ran the other week, or something new?

**mei.c**  10:06 AM
yes, that's the alt-eval. top-1 0.4812 per that run.

**priya.j**  10:08 AM
ok. I want to make sure the leadership deck cites run1 (canonical)
not run3 (alt). can you confirm the deck?

**mei.c**  10:10 AM
deck cites run1 numbers. run3 is there for transparency, not for
shipping.

**priya.j**  10:11 AM
👍.

**— follow-up, 2025-12-02 —**

**priya.j**  2:00 PM
revisiting gt_alt. looking at the 37 dropped hotels, most are
franchise chains where one chain name legitimately maps to multiple
cities (e.g., "Marriott Courtyard" appears in 40+ cities across the
full corpus). our GT for those is `[city1, city2, ...]` which is
honest — any of them is a correct prediction.

**arjun.p**  2:05 PM
so gt_alt is effectively dropping the hardest cases.

**priya.j**  2:06 PM
yes. it's not a "cleaner GT", it's a "skip-the-hard-cases GT".

**arjun.p**  2:07 PM
good catch. can you add this to the audit doc so anyone reading run3
understands the subset isn't representative?

**priya.j**  2:08 PM
already in the stratified PR #071 risk section.

**arjun.p**  2:09 PM
make it more prominent. in the audit doc itself.

**priya.j**  2:10 PM
yeah will do.

— end of channel —
