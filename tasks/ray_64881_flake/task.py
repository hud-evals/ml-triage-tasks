"""Ray-project/ray premerge build 64881: diagnose the red CI.

Ground truth (not shown to agent): the `:ray: core: cpp worker tests` job
failed in build_dashboard_front_end -> `npm ci` with 'cb() never called!'.
PR only changes C++ semaphore code — unrelated. Recommend: retry.
"""

from env import diagnose_ci_failure

PROMPT = (
    "ray-project/ray premerge 64881 on my PR (commit a816079) is red "
    "and i want a sanity check before i bug the oncall. i scraped "
    "what i could into this folder — repo at the sha, all the "
    "buildkite per-job logs, my diff, some GHArchive hours around "
    "the build for outside signal. can you tell me what actually "
    "blew up, whether it's on me or just infra, and what i should do? "
    "write it up as `REPORT.md` in this folder — drop the specific "
    "log lines or code in backticks so i can click through."
)

RUBRIC = {
    "proximate_cause": (
        "Build 64881 has multiple failing jobs. The one with the most "
        "misleading display name is `:ray: core: cpp worker tests`, which "
        "did NOT fail in any C++ test — it died earlier, inside "
        "install_ray -> build_dashboard_front_end -> `npm ci` with the "
        "classic Node.js bug 'cb() never called!'. A correct report must "
        "either (a) name this npm/dashboard failure explicitly as the "
        "proximate cause of that job, OR (b) recognize the job's display "
        "label is misleading and identify the actual failing layer. "
        "Reports that stop at 'C++ test failed' or blame the Cython "
        "'performance hint' GIL warnings, Bazel 'directory does not "
        "exist' warnings, or the PR's C++ diff are WRONG."
    ),
    "pr_attribution": (
        "The PR (#62762) modifies C++ only — POSIX semaphore portability "
        "fixes in experimental_mutable_object_manager.cc. Those files "
        "compiled and linked fine; nothing in the PR touches the "
        "dashboard frontend or npm dependencies. The failure is "
        "UNRELATED to the PR and is a transient CI/infra flake. External "
        "confirmation: the PR was merged a few hours later (see the "
        "gharchive/ slices)."
    ),
    "recommended_action": (
        "Retry the build (or equivalent: 'rerun the failed job', 'this "
        "is infra flake, not a code issue'). Correct answer must NOT "
        "recommend modifying the PR's C++ code and must NOT recommend "
        "blocking the merge pending investigation."
    ),
}

task = diagnose_ci_failure.task(
    prompt=PROMPT,
    case="ray_64881",
    rubric=RUBRIC,
    axis_weights={
        "proximate_cause": 2.0,
        "pr_attribution": 1.5,
        "recommended_action": 1.0,
    },
    anti_fake={"min_verified": 3},
)
task.slug = "ray_64881_flake"
task.metadata = {
    "category": "ci-diagnosis",
    "repo": "ray-project/ray",
    "build": 64881,
    "pr": 62762,
    "commit": "a816079028009d72479e8d10354d7b15dc323f8a",
}
