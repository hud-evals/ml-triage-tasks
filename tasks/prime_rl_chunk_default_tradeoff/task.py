"""Audit a flipped default in prime-rl's fused LM head chunk-size knob."""

from env import diagnose_research_study

PROMPT = (
    "need your read on PR #1892 — flipped RL default for "
    "`fused_lm_head_chunk_size` from 2048 → 8192, claim is "
    "numerical stability, no ablation shipped.\n\n"
    "i've got 30B and 235B runs on the calendar next week that'll "
    "hit this default. before monday i want to decide whether to "
    "pin `fused_lm_head_chunk_size` explicitly in those configs. "
    "in particular: for each of the two upcoming runs, tell me "
    "the closest config shape we already have measurements for in "
    "the repo, what the current peak memory is, and what the flip "
    "would predict for it. if the nearest reference is too far "
    "off to be useful, say so and explain why.\n\n"
    "the checkout has the repo plus a few auxiliary dumps next to "
    "it. figure out what's on disk before you cite anything — i "
    "don't want a writeup that references files nobody bothered to "
    "open.\n\n"
    "drop the writeup in `REPORT.md` right here in the working "
    "directory before you hand it back — i'm reading the file, not "
    "your chat output. 800-1200 words. argue me into or out of the "
    "flip with specific configs cited, give me the one number that "
    "would flip your call, and the strongest objection the infra "
    "team will raise. don't restate the PR body — i've read it, "
    "and anything you could write without opening the repo is a "
    "waste of my monday."
)

RUBRIC = {
    "kernel_memory_math": (
        "Score 4 if the report explains the fused LM head kernel "
        "with direct reference to the actual prime-rl source. Must "
        "identify the [chunk, vocab] intermediate tensor "
        "materialised inside `_ChunkedLogProbEntropyFn` (in "
        "`prime-rl/src/prime_rl/trainer/models/layers/lm_head.py` "
        "or the fused kernel file) and connect chunk ~4× to memory "
        "~4× on that slab. Extra credit: quantifies the slab in "
        "absolute GiB (chunk × vocab × dtype bytes — Qwen3 vocab "
        "~151k, fp16 2 bytes → 2048 chunk ≈ 620 MB, 8192 chunk ≈ "
        "2.4 GB per rank). Score 3 for correct mechanism but no "
        "source citation or vague on the intermediate. Score 2 "
        "for 'bigger chunk = more memory' with one code ref. Score "
        "1 for no kernel engagement, just quoting the PR body. "
        "Score 0 if kernel is not mentioned at all."
    ),
    "empirical_distribution": (
        "Score 4 requires all four of: (i) back-calculate the "
        "fused-head intermediate size from first principles "
        "(chunk × vocab × dtype_bytes; Qwen3 vocab ~151k, fp16 2B "
        "→ 2048 chunk ≈ 620 MB, 8192 chunk ≈ 2.4 GB — report must "
        "show the arithmetic, not hand-wave); (ii) confront that "
        "prediction with PR #1895's observed deltas (1gpu +9.12 "
        "GiB / +40%, 4gpu +9.12 GiB / +65%) — note that the "
        "observed jump is ~4-5× the pure slab size, so the "
        "intermediate alone doesn't explain it; derive or "
        "hypothesize what else is growing (gradient-of-logits, "
        "entropy buffer, the per-chunk log-softmax denominator); "
        "(iii) predict which of the 60 baseline configs in "
        "`prime-rl/benchmarks/baselines/` are on the edge of OOM "
        "after the flip — requires reading ≥6 baseline JSONs and "
        "comparing peak_memory to the device's HBM (A6000 48 GB, "
        "H100 80 GB, H200 141 GB, B200 192 GB); (iv) identifies "
        "at least one config where the flip would push memory "
        ">90% of the GPU's HBM. Score 3 if 3 of 4 are present. "
        "Score 2 if only the PR #1895 deltas are cited without "
        "back-calculation or headroom analysis. Score 1 for a "
        "single baseline anecdote. Score 0 if the baselines "
        "aren't used quantitatively."
    ),
    "stability_claim_audit": (
        "Score 4 if the report takes the 'numerical stability' "
        "claim in PR #1892's body seriously, then shows it's "
        "unbacked in the shipped bundle: no ablation, no loss-"
        "divergence plot, no unit test that demonstrates "
        "instability below 8192, no reference commit or issue "
        "showing divergence — the only artefact added is a "
        "`warnings.warn(below threshold)` guardrail in "
        "`configs/trainer.py`. Also discusses when chunked log-"
        "softmax / log-prob computation would actually lose "
        "precision (log-sum-exp accumulator in fp16 across ~151k-"
        "way softmax, gradient accumulation ordering across "
        "chunks) and whether a 4× chunk raise is a plausible "
        "mitigation for those specific failure modes. Score 3 for "
        "finding the claim unbacked but weak on the numerical-math "
        "side. Score 2 for 'the PR says stability, no evidence "
        "shipped' without engaging the math. Score 1 for just "
        "restating the PR body. Score 0 if the stability claim is "
        "accepted at face value or not audited."
    ),
    "structural_asymmetry": (
        "Score 4 if the report identifies all three structural "
        "facts that only a multi-file read surfaces: (1) SFT "
        "forces `fused_lm_head_chunk_size = 'disabled'` in "
        "`prime-rl/src/prime_rl/configs/sft.py` (around line "
        "328), bypassing the knob entirely; (2) RL uses `'auto'` "
        "which now resolves to 8192 via "
        "`auto_setup_fused_lm_head_chunk_size` in "
        "`configs/trainer.py` (around line 805); (3) the "
        "benchmark runner at "
        "`benchmarks/scripts/run_single_benchmark.py` never "
        "passes `--model.fused-lm-head-chunk-size`, so every RL "
        "benchmark takes the default (which is why the "
        "regression was inevitable across the RL fleet but "
        "invisible to SFT). Explains why this asymmetry is or "
        "isn't defensible — e.g. is it principled (SFT doesn't "
        "need chunking for its sequence shapes) or accidental? "
        "Score 3 for 2 of 3 facts. Score 2 for 1 of 3. Score 1 "
        "if only the trainer.py change is noted. Score 0 for no "
        "multi-file synthesis."
    ),
    "historical_evolution": (
        "Score 4 if the report reconstructs the knob's evolution "
        "from `prime-rl/CHANGELOG.md`: introduced in #1525 "
        "(default None), later made int | Literal['auto', "
        "'disabled'] in #1649 with `auto` resolving to 2048, "
        "then flipped by #1892 so `auto` resolves to 8192 — three "
        "changes to this parameter, each shipping without "
        "published ablation. Frames the #1892 change in that arc "
        "(pattern of under-justified default changes on the same "
        "knob). Score 3 for 2 of 3 historical references. Score "
        "2 for mentioning the parameter has moved before. Score "
        "1 for only #1892. Score 0 if CHANGELOG is not consulted."
    ),
    "recommendation": (
        "Score 4 if the report delivers a data-grounded policy "
        "recommendation that (a) takes a clear position, (b) "
        "defends that position with specific evidence pulled from "
        "axes 1-5, (c) addresses edge cases — e.g. 'revert auto "
        "to 2048, require explicit 8192 opt-in in configs that "
        "want it, demand an ablation PR before any future raise "
        "of this default', or 'keep 8192 only for configs where "
        "vocab × seq_len < threshold X, else force explicit', or "
        "similar. Score 3 for a defended position missing edge-"
        "case nuance. Score 2 for 'revert to 2048' without "
        "conditions or defence. Score 1 for 'accept 8192 as-is' "
        "with no pushback. Score 0 if no recommendation or the "
        "recommendation contradicts the evidence in the rest of "
        "the report."
    ),
}

AXIS_WEIGHTS = {
    "kernel_memory_math":     1.5,
    "empirical_distribution": 3.0,   # discriminating — genuine computation
    "stability_claim_audit":  1.5,
    "structural_asymmetry":   2.5,   # forces 3-file multi-file synthesis (sft.py + trainer.py + run_single_benchmark.py)
    "historical_evolution":   1.5,   # forces CHANGELOG + PR dump reading across #1525, #1649, #1892
    "recommendation":         1.5,
}

HARD_CAPS = [
    {
        "name": "no_empirical_or_kernel_evidence",
        "description": (
            "The recommendation cites neither specific numbers "
            "from the baseline JSONs (or from PR #1895's baseline "
            "diff) nor any reference into the fused-head kernel "
            "source. Indicates the agent skimmed the PR body and "
            "wrote a plausible-sounding opinion without actually "
            "reading across the repo. Cap so fluent guessing "
            "can't coast."
        ),
        "cap": 0.5,
    },
    {
        "name": "shallow_investigation",
        "description": (
            "The report pulls evidence from fewer than three of "
            "the distinct data roots available on disk. The roots "
            "are: (1) `prime-rl/src/` source (kernel, configs, "
            "trainer entrypoints); (2) `prime-rl/benchmarks/` "
            "(baseline JSONs, BENCHMARKS.md, the regression test, "
            "the benchmark runner script); (3) `prime-rl/.git` + "
            "`prime-rl/CHANGELOG.md` (history); (4) `prs/` (gh-api "
            "PR dumps for #1525/#1649/#1892/#1895 context and "
            "diffs); (5) `ci-logs/` (unzipped GHA archives with "
            "per-step memory trajectories); (6) `gpu_runs.jsonl` "
            "(the run index). A report drawing on <3 of these is "
            "shallow by construction — agent wrote a plausible "
            "opinion from one source and stopped. Cap so "
            "single-directory grounding can't score high."
        ),
        "cap": 0.5,
    },
]

BONUS = {
    "description": (
        "The report notices that "
        "`tests/integration/test_benchmark_regression.py` only "
        "asserts 2 of 60 baselines under "
        "`prime-rl/benchmarks/baselines/` (the 1xa6000 + 4xa6000 "
        "rl-full flash_attention_2 seq_len=65536 configs) — so "
        "even the known memory change from #1892 went silently "
        "undetected across the other 58 configs. Structural "
        "observation a senior reviewer would catch, connects the "
        "empirical analysis to the CI-coverage gap."
    ),
    "value": 0.15,
}

task = diagnose_research_study(
    prompt=PROMPT,
    case="prime_rl_chunk_default_tradeoff",
    rubric=RUBRIC,
    axis_weights=AXIS_WEIGHTS,
    axis_scale=4,
    hard_caps=HARD_CAPS,
    bonus=BONUS,
    report_filename="REPORT.md",
    anti_fake={"min_verified": 5, "max_fabricated_ratio": 0.20},
)
task.slug = "prime_rl_chunk_default_tradeoff"
task.columns = {
    "category": "ml-tradeoff-analysis",
    "repo": "PrimeIntellect-ai/prime-rl",
    "knob": "model.fused_lm_head_chunk_size",
    "primary_pr": 1892,
    "reading_target_files": 15,
}
