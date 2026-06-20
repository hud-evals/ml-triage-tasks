"""Diagnose a Kimi Delta Attention decode-step bottleneck on H100.

Setup is from the wafer.ai "profile-guided-optimization" blog narrative —
an engineer profiling KDA decode latency. We don't have the blog's
specific kernel source, but we do have FLA's canonical KDA implementation
(`fla-org/flash-linear-attention/fla/ops/kda/{fused_recurrent,chunk}.py`)
which is real production code. Bench was run on a Modal H100 80GB at a
realistic Kimi-K2-style decode shape (`B=1, T=1, HV=16, K=V=128`).

What the bundle's data actually shows:

  - bench.json:   per_call_us = 51.3 us (cuda.Event over 500 reps)
  - profile_table.txt: fused_recurrent_kda_fwd_kernel Self CUDA = 2.4 us
                       per call (10-call average)
  - grid_estimate.txt: BK=BV=64, NK=NV=2, B=1, HV=16
                       grid_blocks = 64; sm_count = 132;
                       blocks_per_sm = 0.485
  - hardware.txt:  H100 80GB HBM3, 132 SMs, sm_90a

Two distinct observations sit in this data; a top answer engages both:

  (A) Grid undersize. fused_recurrent_kda's grid is
      `NK * NV * B * HV`. T does NOT enter — the recurrence is serial
      along T. At decode (B=T=1, HV=16, K=V=128), the grid is 64, so
      0.485 blocks per SM. That's the wave-count finding from the blog's
      mental model ("64 blocks on a 145-SM GPU, 0.04 waves/SM" — our
      observed 0.485 is higher because we have HV=16 vs the blog's
      apparent HV=1 or fewer, but the *class* of issue is the same).

  (B) CPU launch overhead dominates wall time. 51 us wall vs 2.4 us GPU
      → ~95% of the per-call time is CPU-side (Python/Triton wrapper,
      driver path, kernel launch). At decode the kernel itself is too
      short to amortize the per-launch overhead.

The fix space is therefore TWO-axis:

  Addressing (A) — grid undersize:
    - increase B by grouping multiple decode streams ("batched decode" —
      common in serving)
    - process multiple HV per block (reduce HV-axis grid factor → does
      NOT help; we want MORE blocks not fewer)
    - increase block-level parallelism via cooperative groups → marginal
    - switch to chunk_kda for prefill (not applicable to decode where
      T=1; chunk's chunk-axis parallelism collapses to 1)

  Addressing (B) — CPU overhead:
    - cuda graphs around the decode step (captures launch sequence once,
      replays without per-call Python/driver overhead — best leverage)
    - fuse multiple decode steps per Python call (architectural; harder)
    - compile-time autotune cache (already in place via Triton)
    - avoid the `aten::zeros_like` allocation that profile_table.txt
      shows (10us per call on output allocation, comparable to GPU work
      itself — small win but real)

Best practical answer for production decode:
    cuda graphs + batched-decode → both observations addressed; in vLLM
    and SGLang this is the standard play for linear-attention paths.

The blog's float4-vs-scalar narrative is NOT the bottleneck here. The
canonical FLA kernel is already vectorized (Triton's pointer arithmetic
handles vectorization automatically); proposing "vectorize to float4"
is a misread. Agents who regurgitate the blog narrative without engaging
the actual numbers in the bundle should not score well.
"""

from env import diagnose_research_study

PROMPT = (
    "i'm chasing decode latency for our kimi k2 stack. the kda fused-"
    "recurrent path is the bottleneck. ran the canonical fla "
    "`fused_recurrent_kda` on h100 at the realistic decode shape "
    "(B=1, T=1, HV=16, K=V=128) and got numbers that don't match my "
    "intuition. need a diagnosis before i pick what to build.\n\n"
    "the workspace has the kernel sources from fla "
    "(`fused_recurrent.py` and `chunk.py` from `fla/ops/kda/`), the "
    "bench script that produced the measurements (`bench.py`), and "
    "the artifacts that bench produced: `bench.json`, "
    "`profile_table.txt`, `grid_estimate.txt`, `trace.json`, "
    "`hardware.txt`. there's a brief at `fla_brief.md` if you need "
    "the KDA math.\n\n"
    "i want a diagnosis, not a list of every possible optimization. "
    "specifically:\n"
    "  - what is actually eating the per-call time?\n"
    "  - what is the gap to the H100 roofline for THIS shape?\n"
    "  - which class of fix would close it, with the strongest leverage?\n\n"
    "write `diagnosis.md` in this directory. exactly these section "
    "headings, lowercase, one per line, in order:\n\n"
    "# bench_summary\n"
    "# grid_analysis\n"
    "# cpu_vs_gpu_split\n"
    "# bottleneck_call\n"
    "# fix_proposal\n"
    "# alternatives_considered\n\n"
    "cite specific numbers from `bench.json`, lines/functions in "
    "`profile_table.txt`, kernel-source line ranges in "
    "`fused_recurrent.py`, and SM/spec values from `hardware.txt`. no "
    "internet, no running anything."
)

RUBRIC = {
    "grid_undersize_diagnosis": (
        "A top answer surfaces `grid_blocks = NK * NV * B * HV = 64` "
        "from `grid_estimate.txt` and contrasts it against the 132 SMs "
        "in `hardware.txt`, yielding `blocks_per_sm = 0.485`. Notes "
        "that T does not enter the grid — the recurrence is serial "
        "along the sequence axis, so decode (T=1) collapses one axis "
        "of available parallelism. Cites the grid line in "
        "`fused_recurrent.py` that computes the program-id mapping. "
        "Lower scores for missing the grid math, claiming the grid is "
        "adequate, or attributing the issue to per-block parallelism "
        "without computing the grid-to-SM ratio."
    ),
    "cpu_overhead_diagnosis": (
        "A top answer reads both `bench.json` (per_call_us = 51.3) and "
        "`profile_table.txt` (`fused_recurrent_kda_fwd_kernel` Self "
        "CUDA = 2.4 us per call averaged over 10 reps) and concludes "
        "that ~95% of the per-call wall time is CPU-side, not GPU "
        "work. Distinguishes wall-clock measurement (`cuda.Event` "
        "around 500 reps) from profiler measurement (which has its "
        "own overhead but is reliable for GPU-side time). Lower "
        "scores for assuming the kernel is GPU-bound, for treating "
        "the 51us as the kernel's GPU runtime, or for missing the "
        "gap entirely."
    ),
    "mechanism_quality": (
        "A top answer explains WHY each bottleneck arises from the "
        "kernel's structure. Grid: the recurrence forces serial "
        "execution along T, so the grid only multiplexes "
        "`B * HV * NK * NV`; with the default heuristic "
        "`BK = BV = 64` and `K = V = 128`, `NK = NV = 2`, and decode "
        "fixes `B = 1`, so the grid is `2 * 2 * 1 * 16 = 64`. CPU "
        "overhead: the Triton kernel itself is small (one wave of "
        "register-resident state per block), and each Python-level "
        "call goes through the FLA wrapper → Triton runtime → CUDA "
        "driver, none of which is amortized at one-call-per-token. "
        "Lower scores for hand-waving ('overhead is high'), or for "
        "blaming Triton vs CUDA without engaging the actual cost path."
    ),
    "fix_proposal": (
        "A top answer proposes ONE primary fix with the strongest "
        "leverage, and justifies why it addresses the bottleneck "
        "identified. For this profile the best single answer is "
        "**cuda graphs around the decode step** — it amortizes the "
        "per-call CPU launch overhead which dominates wall time. A "
        "second answer that's also valid: **batched decode** — group "
        "K concurrent decode streams to scale the grid from "
        "64 -> 64*K and saturate the SMs. Combining the two is the "
        "production play and gets full marks. Lower scores for "
        "proposing 'vectorize to float4' (Triton already vectorizes; "
        "this is a misread of the blog narrative), for proposing "
        "fixes that address neither bottleneck, or for proposing a "
        "list of unranked options without picking one."
    ),
    "alternatives_considered": (
        "A top answer mentions `chunk_kda` as a counterfactual — it "
        "adds chunk-axis parallelism which would help for prefill — "
        "but correctly argues it does NOT help decode (T=1 ⇒ "
        "chunk-axis grid collapses to 1; the recurrence still "
        "serializes). Notes other options that look attractive but "
        "don't fit: tuning `BK/BV` (already at 64 which is the heuristic "
        "default; smaller would help grid but hurt arithmetic intensity); "
        "increasing HV (model architecture, not a runtime fix); "
        "fp16/bf16 quantization (already bf16). Lower scores for "
        "promoting chunk_kda as the decode fix, or for not "
        "considering alternatives at all."
    ),
    "profile_reading": (
        "A top answer quotes specific values from at least three of "
        "`bench.json`, `profile_table.txt`, `grid_estimate.txt`, "
        "`hardware.txt`. References specific named entries in the "
        "profile table (e.g. `fused_recurrent_kda_fwd_kernel`, "
        "`aten::zeros_like`, `cudaLaunchKernel`). Lower scores for "
        "vague references or for citing only one artifact."
    ),
    "receipt_grounding": (
        "A top answer ties each claim to a specific source line: "
        "a kernel line in `fused_recurrent.py`, a row in "
        "`profile_table.txt`, a field in `bench.json`, an SM count "
        "in `hardware.txt`. Lower scores for vague citations."
    ),
}

AXIS_WEIGHTS = {
    "grid_undersize_diagnosis": 0.20,
    "cpu_overhead_diagnosis":   0.20,
    "mechanism_quality":        0.15,
    "fix_proposal":             0.18,
    "alternatives_considered":  0.10,
    "profile_reading":          0.10,
    "receipt_grounding":        0.07,
}

HARD_CAPS = [
    {
        "name": "regurgitates_float4_narrative",
        "cap": 0.30,
        "description": (
            "Trigger if the diagnosis claims the bottleneck is scalar "
            "memory access or unvectorized loads and proposes 'use "
            "float4' / 'pack as float4 vectors' as the fix. The "
            "canonical FLA Triton kernel auto-vectorizes via "
            "tl.load/tl.store on block-pointer tiles; there are no "
            "scalar register-resident arrays the agent has access to "
            "rewrite. This is a regurgitation of the wafer.ai blog's "
            "narrative without engaging the actual bundle data."
        ),
    },
    {
        "name": "assumes_gpu_bound",
        "cap": 0.40,
        "description": (
            "Trigger if the diagnosis treats the 51.3us per_call_us "
            "from `bench.json` as the kernel's GPU runtime. "
            "`profile_table.txt` directly shows the kernel's GPU "
            "time at 2.4us per call. Conflating these is a fatal "
            "misread."
        ),
    },
    {
        "name": "misses_grid_undersize",
        "cap": 0.45,
        "description": (
            "Trigger if the diagnosis does not compute or surface "
            "`blocks_per_sm` from the grid_estimate vs hardware.txt "
            "SM count. This is the central observation about grid "
            "parallelism on this kernel at this shape."
        ),
    },
    {
        "name": "misses_cpu_gpu_gap",
        "cap": 0.45,
        "description": (
            "Trigger if the diagnosis does not engage with the "
            "wall-clock-vs-GPU-time gap (51us vs 2.4us). This is the "
            "second central observation; missing it collapses the "
            "diagnosis."
        ),
    },
    {
        "name": "chunk_kda_as_decode_fix",
        "cap": 0.50,
        "description": (
            "Trigger if the diagnosis proposes `chunk_kda` as the fix "
            "for the decode bottleneck. The chunk algorithm "
            "parallelizes the prefix chunk axis, which is 1 at decode "
            "(T=1), so the grid does not grow. chunk_kda is the right "
            "answer for prefill, not decode."
        ),
    },
    {
        "name": "no_specific_fix",
        "cap": 0.55,
        "description": (
            "Trigger if the report lists multiple fix proposals "
            "without picking one as the strongest-leverage answer. "
            "The prompt explicitly asked for a single primary "
            "recommendation."
        ),
    },
    {
        "name": "no_section_structure",
        "cap": 0.40,
        "description": (
            "Trigger if the report does not have the six required "
            "section headings (bench_summary, grid_analysis, "
            "cpu_vs_gpu_split, bottleneck_call, fix_proposal, "
            "alternatives_considered) in the specified order."
        ),
    },
    {
        "name": "no_receipts",
        "cap": 0.50,
        "description": (
            "Trigger if the report cites essentially no concrete "
            "values from `bench.json`, `profile_table.txt`, "
            "`grid_estimate.txt`, `hardware.txt`, or kernel source."
        ),
    },
]

BONUS = {
    "value": 0.05,
    "description": (
        "Trigger if the report explicitly notes that the data tells "
        "a different story than a 'vectorize the inner loop' "
        "narrative might suggest — that the kernel is short and the "
        "main cost is per-launch infrastructure, so the highest-"
        "leverage fix is to reduce the number of launches (CUDA "
        "graphs, batched-decode, multi-step fusion), not to "
        "restructure inside the existing block."
    ),
}

task = diagnose_research_study(
    prompt=PROMPT,
    case="wafer_kda_diag",
    rubric=RUBRIC,
    axis_weights=AXIS_WEIGHTS,
    axis_scale=4,
    hard_caps=HARD_CAPS,
    bonus=BONUS,
    report_filename="diagnosis.md",
    anti_fake={"min_verified": 3},
)
task.slug = "wafer_kda_diag"
task.columns = {
    "category": "profile-driven-bottleneck-diagnosis",
    "source_repos": [
        "fla-org/flash-linear-attention (fla/ops/kda)",
        "wafer.ai/blog/profile-guided-optimization (task shape inspiration)",
    ],
    "hardware": "NVIDIA H100 80GB HBM3, 132 SMs (sm_90a)",
    "decode_shape": {"B": 1, "T": 1, "H": 16, "HV": 16, "K": 128, "V": 128},
    "measured_per_call_us": 51.3,
    "measured_gpu_us_per_call": 2.4,
    "measured_grid_blocks": 64,
    "task_shape": "real_workflow_profile_diagnosis_with_two_signals",
}
