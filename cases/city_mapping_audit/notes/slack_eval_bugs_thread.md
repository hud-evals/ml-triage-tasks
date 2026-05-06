# #eval-bugs (pasted transcript, 2025-10-14)

**priya.j**  2:14 PM
i'm re-running wratio on the eval subset and getting top-1 that's slightly
higher than last week's run1.json — is that expected?

**mei.c**  2:15 PM
probably tie-break ordering. we have a lot of integer-scored ties. what
script are you using?

**priya.j**  2:15 PM
eval_v2.py — i thought that was the current one post-#042?

**mei.c**  2:18 PM
hmm. eval_v2 was supposed to be a cleanup not a semantics change. i'll
take a look this week. until then please use eval.py for anything that
lands in runs/.

**jordan.r**  2:21 PM
eval_v2 looks fine to me, i've been using it for openai runs. the numbers
match eval.py within 0.001 on what i've spot-checked.

**mei.c**  2:22 PM
ok let's leave it then. circling back if anything actually breaks.
