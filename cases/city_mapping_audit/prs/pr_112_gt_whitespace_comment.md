# PR #112 — Add explanatory comment to gt_whitespace.json

**Author:** Priya  ·  **Status:** draft  ·  **Reviewer:** —

## Summary

`ground_truth/gt_whitespace.json` is a historical artifact from an
early iteration of `src/build_gt.py` that produced city values with
trailing whitespace on ~1% of entries. The file is still in the
repo but not referenced by any harness I can find. Adding a comment
to the top of the file to head off anyone who stumbles onto it and
assumes it's the canonical GT.

## Risk

None. No functional change.
