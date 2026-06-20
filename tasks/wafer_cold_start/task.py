"""Triage a kernel-launch cold-start spike against multi-arm experimental data.

A bench engineer chases a transient per-launch slowdown that appears after
`torch.utils.cpp_extension.load(...)`. A coworker forwarded an external
writeup naming `cpp_extension.load`-induced cpu migration as the cause and
prescribing `sched_setaffinity` pinning as the fix. The diagnostic harness
in the bundle ran a five-arm controlled experiment in fresh containers
(idle_pin, migrate, fork_only, load_open, load_pin), four reps each, on the
same Modal H100 substrate.

What the data says (curator ground truth — hidden from agent):
  * `idle_pin` and `migrate` arms: A2 iter 0 indistinguishable from steady
    state (27–42 us, same noise as steady). Pure migration does NOT
    produce a spike on this hardware.
  * `fork_only`, `load_open`, `load_pin` arms: A2 iter 0 is 176–278 us,
    ~6–9x steady. Pinning across the load does NOT reduce the spike.
    The spike is one iteration; full recovery by iter 2–3.
  * Therefore the blog's mechanism (migration → cold L1/L2 of CUDA
    dispatch structures) is REFUTED on this substrate. The blog's
    primary fix (pre-load sched_setaffinity) does NOT work here.
  * The trigger is fork+exec itself — `subprocess.run(["/bin/true"])`
    six times is sufficient to produce the same spike. The compile/.so
    machinery is not necessary, just the fork+wait.
  * Mechanism is per-process state invalidation through the
    fork/exec/waitpid path (TLB shootdown, COW page handling, possibly
    CUDA driver-side stream/event pool state). It is NOT per-core
    cache, because `migrate` alone doesn't reproduce it.
  * Correct fix: post-trigger warmup. Even ~5 launches after any
    fork+exec are enough — the spike recovers within 1–3 iters in
    every load_* and fork_only rep.

The task asks the agent for a postmortem.md with six fixed sections.
The grader scores per-section against a rubric that rewards specifically
reading the multi-arm data and disconfirming the blog claim from it.
"""

from env import diagnose_research_study

PROMPT = (
    "the kernel-bench rig is doing something annoying. same custom "
    "cuda kernel, same node, sometimes it benches 44% slower than "
    "reference, sometimes 12% faster. the difference between the "
    "two runs is whether the python process compiled any other "
    "extensions via `torch.utils.cpp_extension.load(...)` "
    "immediately before the timed window.\n\n"
    "i ran a five-arm controlled experiment in fresh containers — "
    "idle_pin, migrate, fork_only, load_open, load_pin — four reps "
    "each. the bench code is in `harness/bench.py`. the per-iter "
    "csvs and sched logs are under `bench_artifacts/`.\n\n"
    "a coworker forwarded an external writeup arguing the cause is "
    "process migration during the nvcc compile and the fix is "
    "`sched_setaffinity` pinning before any `cpp_extension.load`. "
    "that hypothesis is in `hypothesis/blog_hypothesis.md`. treat "
    "it as a candidate, not a conclusion — i want to know if it "
    "actually holds on this hardware.\n\n"
    "write `postmortem.md` in this directory. use exactly these "
    "section headings, lowercase, single line each, in this order:\n\n"
    "# root_cause\n"
    "# evidence_chain\n"
    "# hypothesis_test\n"
    "# minimum_fix\n"
    "# alternative_fix\n"
    "# generalization\n\n"
    "in each section, cite specific arms / reps / iteration ranges "
    "from `bench_artifacts/`. say which CSV row supports each "
    "claim. distinguish what each arm rules IN from what it rules "
    "OUT. include actual code in `minimum_fix`. don't edit `harness/` "
    "or `bench_artifacts/`. no internet."
)

RUBRIC = {
    "root_cause_correctness": (
        "A top answer names fork+exec (specifically the "
        "fork/wait/waitpid cycle initiated by any subprocess "
        "spawn, including but not limited to `cpp_extension.load`) "
        "as the TRIGGER, and distinguishes the trigger from the "
        "MECHANISM (per-process state invalidation — likely TLB "
        "shootdown from COW handling, CUDA driver stream/event "
        "pool state reset on the next launch, or similar — not "
        "per-core L1/L2 cache, since `migrate` doesn't reproduce "
        "the spike). The answer should explicitly note that the "
        "spike is independent of cpu affinity changes. Lower "
        "scores for naming migration as the trigger, conflating "
        "trigger with mechanism, or vague phrasing ('cache "
        "effects', 'system noise') without identifying which "
        "cache and ruling out which alternatives."
    ),
    "evidence_chain": (
        "A top answer cites specific arms, phases, and iteration "
        "ranges. For every claim, points at which "
        "`<arm>_rep<NN>/per_iter.csv` row or sched.jsonl event "
        "supports it. Uses the arm structure to factor causes: "
        "`idle_pin` and `migrate` rule OUT pure-migration as "
        "sufficient cause; `fork_only` rules IN fork+exec as "
        "sufficient cause; `load_open` vs `load_pin` rules OUT "
        "affinity-during-load as the variable that matters. "
        "Quantifies the spike (A2 iter 0 around 200 us vs steady "
        "around 25-30 us, ~6-9x) and the recovery window "
        "(1-3 iters, not 50). Lower scores for vague claims "
        "without specific iter or rep references, or for citing "
        "only one or two arms when the experiment had five."
    ),
    "hypothesis_test": (
        "A top answer engages the writeup in "
        "`hypothesis/blog_hypothesis.md` claim by claim and "
        "tests each against the data. Specifically: the claim "
        "that migration is the mechanism is REFUTED — by "
        "comparing `migrate` (no spike) against `fork_only` "
        "and `load_open` (spike). The claim that "
        "`sched_setaffinity` pre-load fixes the issue is "
        "REFUTED — by comparing `load_open` and `load_pin` "
        "(both spike, indistinguishable magnitudes). The "
        "claim that the spike lasts ~50 iters is REFUTED — "
        "by reading the per-iter csvs (recovery is 1-3 iters). "
        "The claim that the spike exists is CONFIRMED. The "
        "alternative fix the writeup mentions in passing — "
        "post-load warmup — is the one that actually works "
        "here. Lower scores for accepting the blog's "
        "conclusion without testing it, for treating the "
        "writeup as authoritative, or for missing that the "
        "blog's primary fix demonstrably does not work."
    ),
    "minimum_fix": (
        "A top answer gives a concrete code snippet of the "
        "smallest change to the harness that eliminates the "
        "spike. Correct answer is post-load (or post-fork) "
        "warmup — running a handful of launches after any "
        "`cpp_extension.load` or `subprocess.run` and before "
        "the timed window. The justification ties the fix back "
        "to the agent's mechanism story (the invalidated "
        "per-process state is warmed back up by a few launches). "
        "Lower scores for prescribing `sched_setaffinity` "
        "pinning (which the data shows does not work), for "
        "code-free prose, or for prescribing a fix without "
        "naming how many warmup iterations are needed (3-5 is "
        "sufficient per the data; 5000 from the blog is "
        "vastly overkill here)."
    ),
    "alternative_fix": (
        "A top answer offers a second, materially different fix "
        "with a different tradeoff — e.g. drop the first N "
        "iterations from the reported median; lift the "
        "extension cache out of process (preload all extensions "
        "at process start, never load mid-bench); use a "
        "long-lived bench worker process that does its loads "
        "once at startup; use cudaGraphs to bypass the per-"
        "launch overhead the spike lives in. The answer says "
        "when each is preferable. Lower scores for offering "
        "only minor variants of the minimum fix, or for "
        "listing fixes without their tradeoffs."
    ),
    "generalization": (
        "A top answer names 2-4 other common operations that "
        "would produce the same spike, each backed by the "
        "agent's mechanism story. Plausible items: any "
        "`subprocess.run(...)` (matched by the `fork_only` arm); "
        "`os.system`; `multiprocessing.Pool` startup; "
        "`tempfile.NamedTemporaryFile`-style heavy filesystem "
        "ops if they hit the same syscall path; loading any "
        "new shared object via dlopen/ctypes/CDLL. Lower "
        "scores for generic phrasing ('compile-time work'), "
        "for items that are not implied by the agent's "
        "mechanism, or for fewer than two items."
    ),
    "receipt_grounding": (
        "A top answer cites concrete file paths "
        "(`bench_artifacts/<arm>_rep<NN>/per_iter.csv`), "
        "specific iter ranges, specific timestamps from "
        "sched.jsonl, and lines from `harness/bench.py`. "
        "Cross-references the arm-action branch in "
        "`bench.py` against the sched events recorded in "
        "the JSONL. Lower scores for vague citations or "
        "leaning on a single artifact type."
    ),
}

AXIS_WEIGHTS = {
    "root_cause_correctness": 0.22,
    "evidence_chain":         0.20,
    "hypothesis_test":        0.18,
    "minimum_fix":            0.15,
    "alternative_fix":        0.10,
    "generalization":         0.08,
    "receipt_grounding":      0.07,
}

HARD_CAPS = [
    {
        "name": "names_migration_as_trigger",
        "cap": 0.25,
        "description": (
            "Trigger if root_cause names cpu migration as the "
            "cause of the spike. The `migrate` arm directly "
            "refutes this: explicit `sched_setaffinity` to a "
            "different cpu produces no measurable spike at "
            "A2 iter 0 (27-42 us, identical to `idle_pin`)."
        ),
    },
    {
        "name": "recommends_sched_setaffinity_fix",
        "cap": 0.30,
        "description": (
            "Trigger if minimum_fix prescribes "
            "`sched_setaffinity` pinning before "
            "`cpp_extension.load(...)` as the fix. The "
            "`load_pin` arm directly refutes this: pinning "
            "across the load produces the same 200+ us "
            "spike as `load_open`."
        ),
    },
    {
        "name": "blog_unconditionally_accepted",
        "cap": 0.40,
        "description": (
            "Trigger if hypothesis_test accepts the writeup's "
            "claim of migration-as-mechanism and pinning-as-fix "
            "without testing either against the multi-arm data. "
            "The blog's claim is refuted on this substrate; "
            "accepting it without empirical test is the central "
            "trap."
        ),
    },
    {
        "name": "spike_lasts_fifty_iters",
        "cap": 0.50,
        "description": (
            "Trigger if the report claims the spike lasts ~50 "
            "iterations (matching the blog's claim) when the "
            "per_iter.csv files show recovery in 1-3 iterations "
            "in every load_* and fork_only rep."
        ),
    },
    {
        "name": "fork_only_arm_ignored",
        "cap": 0.50,
        "description": (
            "Trigger if hypothesis_test or evidence_chain does "
            "not engage the `fork_only` arm. That arm is the "
            "decisive evidence that any fork+exec — not "
            "specifically compilation or .so loading — is "
            "sufficient. Ignoring it leaves the mechanism story "
            "incomplete."
        ),
    },
    {
        "name": "no_code_in_fix",
        "cap": 0.55,
        "description": (
            "Trigger if minimum_fix is prose only, with no "
            "code snippet showing the actual change. The prompt "
            "asked for an actual change."
        ),
    },
    {
        "name": "no_section_structure",
        "cap": 0.40,
        "description": (
            "Trigger if the report does not have the six "
            "required sections (root_cause, evidence_chain, "
            "hypothesis_test, minimum_fix, alternative_fix, "
            "generalization) with the specified headings, in "
            "the specified order."
        ),
    },
    {
        "name": "no_arm_comparisons",
        "cap": 0.45,
        "description": (
            "Trigger if evidence_chain does not contrast at "
            "least three of the five arms against each other "
            "to factor causes. A diagnosis from this bundle "
            "requires using the arm structure; a single-arm "
            "diagnosis cannot reach the correct answer."
        ),
    },
    {
        "name": "no_receipts",
        "cap": 0.50,
        "description": (
            "Trigger if the report cites essentially no "
            "concrete receipts — no per_iter.csv references, "
            "no specific iter ranges, no sched.jsonl events, "
            "no harness/bench.py line references."
        ),
    },
]

BONUS = {
    "value": 0.05,
    "description": (
        "Trigger if the report explicitly notes the asymmetry "
        "the multi-arm experiment surfaces: the writeup's "
        "SYMPTOM (cpp_extension.load triggers a spike) is "
        "confirmed, but its MECHANISM (migration → cold L1/L2) "
        "and its primary FIX (pre-load `sched_setaffinity`) are "
        "both refuted on this substrate. The calibrated move is "
        "to keep the writeup's symptom observation, replace its "
        "mechanism with a fork/exec-induced per-process-state "
        "story, and replace its primary fix with post-event "
        "warmup. Cites both the `migrate` arm (refutes "
        "mechanism) and `load_pin` arm (refutes fix) by name."
    ),
}

task = diagnose_research_study(
    prompt=PROMPT,
    case="wafer_cold_start",
    rubric=RUBRIC,
    axis_weights=AXIS_WEIGHTS,
    axis_scale=4,
    hard_caps=HARD_CAPS,
    bonus=BONUS,
    report_filename="postmortem.md",
    anti_fake={"min_verified": 3},
)
task.slug = "wafer_cold_start"
task.columns = {
    "category": "benchmark-methodology-triage",
    "source_blog": "wafer.ai/blog/where-did-my-microseconds-go",
    "substrate": "Modal H100:1, pytorch 2.5.1+cu124, 24-core affinity",
    "task_shape": "blog_claim_stress_test_with_alternative_mechanism",
    "decisive_arm": "fork_only",
    "refuting_arms_for_blog": ["migrate", "load_pin"],
    "spike_recovery_iters": "1-3 (not the blog's ~50)",
}
