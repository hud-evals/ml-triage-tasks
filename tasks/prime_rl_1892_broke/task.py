"""Prime-RL benchmark_regression failure — investigation-style framing.

The bundle reads like a real "I'm investigating a regression" dump:
- `prime-rl/` — full repo at the failing commit (83c21ddac)
- `ci-dump/` — failing run + job JSONs, annotations, check-runs at head + parent,
              compare view, PR events, AND a 50-entry run history on main for
              the week before the failure (`runs-before-1892.json`)
- `patches/` — six PRs that landed in the 3-day window before + at the red flip:
              #1880, #1881, #1882, #1883, #1891, #1892. Agent has to determine
              which one actually caused benchmark_regression to flip red, not
              just blame the most recent.
- `baselines-snapshots/` — the 1gpu + 4gpu baseline JSONs at parent (a76eec30)
                           and head (83c21ddac). They're identical because
                           #1892 didn't update them.

Rubric shifts from "find the needle" to "explain the regression":
  1. regression_identification: correctly pin #1892 among the adjacent PRs
  2. mechanism: explain WHY peak memory scales with chunk_size
  3. fix_direction: revert / opt-in, not silent baseline bump
"""

from env import diagnose_ci_failure

PROMPT = (
    "benchmark_regression just went red on prime-rl main. which PR "
    "broke it, why (walk me through the mechanism — not just what "
    "changed but how that change ends up tripping the assert), and "
    "what to do about it? write it up as `REPORT.md` here (~750 "
    "words), drop file paths / line numbers / code in backticks so i "
    "can follow the trail. everything i could scrape is in this folder."
)

RUBRIC = {
    # Axis 1 — which PR actually caused the regression (6 candidates in patches/)
    "regression_identification": (
        "Score 1 if the report correctly identifies PR #1892 as the "
        "cause of the benchmark_regression failure AND supports that "
        "claim by at least implicitly ruling out the other adjacent PRs "
        "(#1880, #1881, #1882, #1883, #1891) — none of those touch "
        "`src/prime_rl/configs/trainer.py`, `src/prime_rl/trainer/`, or "
        "any of the fused LM head code. The proximate code change is "
        "the 2048 -> 8192 default flip in `auto_setup_fused_lm_head_"
        "chunk_size` (`prime-rl/src/prime_rl/configs/trainer.py`, ~line "
        "808). Score 0 if the report blames an adjacent PR (orchestrator "
        "changes, vllm bump, ruff formatting, inference configs), "
        "blames a decoy inside #1892 (the added warn-below-8192 "
        "validator, the CHANGELOG entry), uncritically accepts the "
        "PR's 'numerical stability' framing as evidence it's benign, "
        "or calls the failure a flake."
    ),
    # Axis 2 — mechanism: WHY does peak memory specifically shift when chunk_size changes
    "mechanism": (
        "Score 1 if the report explains at kernel level why raising "
        "`fused_lm_head_chunk_size` from 2048 to 8192 shifts peak "
        "memory specifically (and not proportionally MFU / throughput / "
        "step_time). Has to reference the `[chunk_size, vocab]` "
        "intermediate materialised inside the fused LM head — "
        "`_ChunkedLogProbEntropyFn` in "
        "`prime-rl/src/prime_rl/trainer/models/layers/lm_head.py` "
        "(or the `fused_lm_head.py` kernel) — and explain that "
        "quadrupling the chunk size quadruples that block's memory "
        "without a proportional compute/throughput change. Also "
        "acknowledges that the 1% MEMORY_TOLERANCE is much tighter "
        "than the 10% METRIC_TOLERANCE on MFU / throughput / "
        "step_time (`prime-rl/tests/integration/"
        "test_benchmark_regression.py`). Score 0 if the explanation "
        "is surface-level ('bigger chunk uses more memory') without "
        "citing the kernel or explaining why memory specifically "
        "breached when other metrics didn't, or if the report doesn't "
        "attempt a mechanism at all."
    ),
    # Axis 3 — fix direction: revert/opt-in, NOT silent baseline bump
    "fix_direction": (
        "Score 1 if the report recommends either (a) reverting the "
        "default flip back to 2048, OR (b) making 8192 opt-in — e.g. "
        "default the setting to `'disabled'` / 2048 and pin 8192 "
        "explicitly only in the CI configs that want to exercise it. "
        "Both directions preserve the regression-signal for memory-"
        "sensitive users. Score 0 if the report recommends silently "
        "regenerating and committing new baselines (`benchmarks/"
        "baselines/*.json`) to accept the new memory numbers — that "
        "hides the regression for the next person who touches the "
        "knob — OR if it recommends retrying the build, merging the "
        "PR as-is, or loosening the tolerance."
    ),
}

task = diagnose_ci_failure.task(
    prompt=PROMPT,
    case="prime_rl_1892",
    rubric=RUBRIC,
    axis_weights={
        "regression_identification": 2.0,
        "mechanism":                 2.0,
        "fix_direction":             1.5,
    },
    anti_fake={"min_verified": 3},
)
task.slug = "prime_rl_1892_broke"
task.metadata = {
    "category": "ci-diagnosis",
    "repo": "PrimeIntellect-ai/prime-rl",
    "run_id": 22423809771,
    "pr": 1892,
    "commit": "83c21ddac5af2e1d747475e88a7349c2bd319ec7",
}
