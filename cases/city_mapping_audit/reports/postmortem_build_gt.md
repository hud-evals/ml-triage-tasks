# Post-mortem: gt_whitespace.json incident (abbreviated)

**Status:** not active — kept for context.

An intermediate iteration of `build_gt.py` produced a `gt.json` with
trailing whitespace on some city values (e.g., "Paris "). That copy was
saved separately as `ground_truth/gt_whitespace.json` before it was
overwritten by the fixed build. If any artifact references that file, it
will silently drop all hotels whose GT city has trailing whitespace (the
lookup into `city_to_idx` fails). Audit accordingly.
