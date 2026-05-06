"""HUD environment for diagnosing Ray CI failures from pre-packaged log bundles.

Design:
  * Agent reads the bundle via bash + edit tools, then submits a **written
    report** as free text, along with a list of verbatim `evidence_quotes`
    copied from files inside the bundle.
  * Grading combines two signals:
      1. Anti-fake gate: every `evidence_quote` must literally appear in at
         least one file under /opt/ray_bundle. If the agent hallucinated
         quotes, this fails and the final score is floored to 0.
      2. Agentic judge: an LLM reads the report (plus the rubric) and scores
         three rubric axes — proximate cause, PR-vs-flake attribution,
         recommended action — each 0 or 1. Judge output is JSON.
  * Final score = gate * weighted sum of judge axes.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from hud import Environment
from hud.tools.coding import BashTool
from hud.tools.types import ContentResult, EvaluationResult, SubScore

env = Environment(name="ml-triage-tasks")

# Cases are baked into the image under /opt/ci_cases/<slug>/. The env
# process runs as root so it can read baked cases, but every bash command
# the agent runs is demoted to uid 1000 and sees only /workspace.
# /app, /opt/ci_cases, and the installed grader module are locked 0700
# root:root so uid 1000 cannot traverse in.
CASES_ROOT = Path(os.environ.get("CI_CASES_ROOT", "/opt/ci_cases"))
WORK = Path(os.environ.get("CI_WORK", "/workspace"))

AGENT_UID = 1000
AGENT_GID = 1000


try:
    WORK.mkdir(parents=True, exist_ok=True)
    # If running as root (production), make sure /workspace is owned by the
    # agent uid so it can write there after demotion.
    if os.geteuid() == 0:
        os.chown(WORK, AGENT_UID, AGENT_GID)
except OSError:
    pass


def _chown_recursive(root: Path, uid: int, gid: int) -> None:
    """Chown `root` and everything beneath it. No-op if not root."""
    if os.geteuid() != 0:
        return
    try:
        os.chown(root, uid, gid)
    except OSError:
        pass
    for p in root.rglob("*"):
        try:
            os.chown(p, uid, gid, follow_symlinks=False)
        except OSError:
            continue


def _mount_case(case: str) -> Path:
    """Copy one case's contents into /workspace so /workspace *is* the case
    dir. Hard-copies (not symlinks) so `ls -la` can't leak the case slug
    via link targets. Chowns the copy to the agent uid.
    """
    import shutil
    src = CASES_ROOT / case
    if not src.is_dir():
        raise FileNotFoundError(f"case not found: {src}")
    for existing in WORK.iterdir():
        try:
            if existing.is_symlink() or existing.is_file():
                existing.unlink()
            else:
                shutil.rmtree(existing)
        except OSError:
            pass
    for child in src.iterdir():
        dst = WORK / child.name
        if child.is_dir():
            shutil.copytree(child, dst, symlinks=False)
        else:
            shutil.copy2(child, dst)
    _chown_recursive(WORK, AGENT_UID, AGENT_GID)
    return src


# Env-var prefixes that the bash subprocess must NOT see. Anything starting
# with these or matching the exact names below is scrubbed before exec.
_SCRUB_PREFIXES = ("HUD_", "CI_", "AWS_", "AZURE_", "GCP_", "OPENAI_", "ANTHROPIC_", "PRIME_", "WANDB_")
_SCRUB_EXACT = {"PYTHONPATH", "PYTHONHOME", "API_KEY", "HF_TOKEN"}


def _agent_env() -> dict[str, str]:
    """Minimal env passed into the demoted bash subprocess. Strips HUD_API_KEY,
    HUD_GATEWAY_URL, and other parent-process secrets."""
    out: dict[str, str] = {}
    for k, v in os.environ.items():
        if k in _SCRUB_EXACT:
            continue
        if any(k.startswith(pfx) for pfx in _SCRUB_PREFIXES):
            continue
        out[k] = v
    out["HOME"] = str(WORK)
    out["PWD"] = str(WORK)
    out["PATH"] = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    out["SHELL"] = "/bin/bash"
    return out


def _demote_preexec() -> None:
    """preexec_fn for subprocess: drop to agent uid + clear supplementary
    groups. No-op if we aren't root (e.g. local dev on macOS)."""
    if os.geteuid() != 0:
        return
    os.setgroups([])
    os.setgid(AGENT_GID)
    os.setuid(AGENT_UID)


@env.tool(name="bash")
async def bash(command: str) -> ContentResult:
    """Run a bash command. Each call starts a fresh shell rooted in the
    workspace folder — no session state between calls, so `cd` and env
    changes don't persist. Use absolute paths rooted at the workspace,
    or paths relative to it.

    Args:
        command: the shell command to run. stdout + stderr are both
            returned in the output; non-zero exit codes are reported but
            do not raise.
    """
    import asyncio
    proc = await asyncio.create_subprocess_shell(
        command,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=str(WORK),
        env=_agent_env(),
        preexec_fn=_demote_preexec,
    )
    try:
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=120.0)
    except asyncio.TimeoutError:
        proc.kill()
        return ContentResult(output="command timed out after 120s", error="timeout")
    text = (stdout or b"").decode("utf-8", errors="replace")
    if len(text) > 20_000:
        text = text[:20_000] + f"\n... [truncated, {len(text) - 20_000} more bytes]"
    rc = proc.returncode
    if rc != 0:
        text = f"{text}\n[exit {rc}]"
    return ContentResult(output=text or "(no output)")


# ============================================================================
# Submission
# ============================================================================

_SUBMISSION: dict[str, Any] = {}


def _load_report(filename: str = "REPORT.md") -> None:
    """Read the report file from the workspace (if present) into _SUBMISSION."""
    _SUBMISSION.clear()
    path = WORK / filename
    try:
        text = path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return
    _SUBMISSION.update({"report": text, "citations": _extract_citations(text)})


# ============================================================================
# Anti-fake: verify structural citations (paths, PRs, SHAs, run ids) exist
# ============================================================================
#
# Literal substring matching against bundle text is too brittle — ANSI
# escapes, line-prefixed timestamps, paraphrase, and Unicode normalisation
# all make correct-but-clean quotes fail the grep. And the load-bearing
# citations in an audit report (PR numbers, commit SHAs, file paths) are
# typically under any reasonable length floor anyway.
#
# Instead we extract *identifiers* the agent cites — things that are either
# real or not, with no paraphrase axis — and check each one exists in the
# bundle. The failure mode we're guarding against is fabrication: citing
# `#1899` when #1899 isn't in the window, citing `deadbeefcafe` when no such
# sha exists, citing `src/fake/path.py` when the file isn't in the repo.
# Hallucinated prose without a fabricated identifier is the judge's job
# (the judge will downrank a report that asserts things the rubric says
# are wrong).

_PATH_EXT = (
    r"py|json|jsonl|md|txt|yaml|yml|toml|patch|csv|tsv|"
    r"parquet|db|sqlite|sqlite3|log|"
    r"ipynb|rs|c|cpp|h|hpp|go|ts|tsx|js|sh"
)
_PATH_RE = re.compile(
    r"[a-zA-Z0-9_][a-zA-Z0-9_./-]*/[a-zA-Z0-9_./-]+\.(?:" + _PATH_EXT + r")"
)

_CITATION_PATTERNS = (
    # commit sha: 8-40 hex, word-bounded.
    ("sha",    re.compile(r"(?<![0-9a-f])([0-9a-f]{8,40})(?![0-9a-f])")),
    # PR / issue: "#1892", "PR #1892", "pull/1892", "issue/1892"
    ("pr",     re.compile(r"(?:#|pull/|issue/|PR\s*#?)(\d{3,6})\b", re.I)),
    # nmoe-style run id: `run_<epoch-seconds>_<pid>`  (e.g. run_1776990705_32).
    # Capture the full token so verification matches directory names exactly.
    ("run_id", re.compile(r"\b(run_\d{10,12}_\d{1,6})(?![0-9a-zA-Z_])")),
    # GHA-style bare run id: 10-12 digit run number (often prefixed `run-` or `run `).
    ("run_id", re.compile(r"\brun[- ](\d{10,12})(?![0-9a-zA-Z_])")),
    # file path: at least one `/`, ends in a known source extension.
    ("path",   _PATH_RE),
)


def _extract_citations(report: str) -> list[dict[str, str]]:
    """Pull identifier-style citations out of the report.

    Returns a list of {"kind", "value", "raw"} dicts. Identifiers are the
    ones a report naturally uses to point at bundle contents: PR numbers,
    commit SHAs, GHA run ids, repo-relative file paths.
    """
    seen: set[tuple[str, str]] = set()
    out: list[dict[str, str]] = []
    for kind, pat in _CITATION_PATTERNS:
        for m in pat.finditer(report):
            # Path regex has no capture group; fall back to group(0).
            val = (m.group(1) if m.groups() else m.group(0)).strip()
            if not val:
                continue
            if kind == "sha" and len(val) < 8:
                continue
            if kind == "sha" and val.isdigit():
                # Pure digits aren't a sha even if they happen to be ≥8 chars.
                continue
            key = (kind, val.lower() if kind in ("sha", "path") else val)
            if key in seen:
                continue
            seen.add(key)
            out.append({"kind": kind, "value": val, "raw": m.group(0)})
    return out


_BUNDLE_CACHE: dict[str, dict[str, Any]] = {}


def _bundle_index(case_root: Path) -> dict[str, Any]:
    """Build a lookup index once per case: set of short shas, PR numbers,
    run ids, and relative file paths present in the bundle. Used for O(1)
    citation verification instead of linear grep-per-citation.
    """
    key = str(case_root)
    if key in _BUNDLE_CACHE:
        return _BUNDLE_CACHE[key]
    shas: set[str] = set()
    prs: set[str] = set()
    run_ids: set[str] = set()
    paths: set[str] = set()            # real files under case_root (lowercase)
    mentioned_paths: set[str] = set()  # path strings that appear in bundle text
    if case_root.is_dir():
        text_blobs: list[str] = []
        for p in case_root.rglob("*"):
            if p.is_file():
                rel = str(p.relative_to(case_root))
                paths.add(rel.lower())
                name = p.name
                if name.endswith(".json") and p.parent.name == "prs":
                    stem = name.split("-")[0].removesuffix(".json")
                    if stem.isdigit():
                        prs.add(stem)
                # Match both GHA-style (run-1234567890) and nmoe-style
                # (run_1776990705_32) directory names. Store the full token
                # so verification compares against what a report would cite.
                if p.parent:
                    pname = p.parent.name
                    m = re.match(r"^run[-_]?(\d{8,}(?:_\d+)?)$", pname)
                    if m:
                        # Keep both the numeric token and the full dir name
                        # as valid verifications.
                        run_ids.add(m.group(1))
                        run_ids.add(pname)
                # text-ish files get harvested for mentioned identifiers.
                if p.suffix.lower() in {
                    ".jsonl", ".json", ".txt", ".md", ".py",
                    ".patch", ".yaml", ".yml", ".toml", ".log",
                }:
                    try:
                        text_blobs.append(p.read_text(encoding="utf-8", errors="replace"))
                    except OSError:
                        continue
        # Harvest SHAs, PR numbers, and path-like strings from all text
        # content. PRs and paths referenced in CHANGELOGs, PR bodies, or
        # source comments are real citations even if we didn't bundle the
        # corresponding <N>.json / <path> file.
        mentioned_prs: set[str] = set()
        pr_mention_re = re.compile(r"(?:#|pull/|issue/|PR\s*#?)(\d{3,6})\b", re.I)
        for txt in text_blobs:
            for m in re.finditer(r"[0-9a-f]{8,40}", txt):
                v = m.group(0).lower()
                if not v.isdigit():
                    shas.add(v)
            for m in _PATH_RE.finditer(txt):
                mentioned_paths.add(m.group(0).lower())
            for m in pr_mention_re.finditer(txt):
                mentioned_prs.add(m.group(1))
        # Also: if a cloned repo (.git) is in the bundle, harvest its full
        # commit-sha set so short prefixes resolve.
        for gitdir in case_root.rglob(".git"):
            try:
                import subprocess as _sp
                out = _sp.run(
                    ["git", "--git-dir", str(gitdir), "rev-list", "--all"],
                    capture_output=True, text=True, timeout=10,
                )
                for line in out.stdout.splitlines():
                    if re.fullmatch(r"[0-9a-f]{40}", line.strip()):
                        shas.add(line.strip().lower())
                break
            except Exception:
                pass
    index = {
        "shas": shas,
        "prs": prs | mentioned_prs,  # union — PRs referenced in text are real
        "run_ids": run_ids,
        "paths": paths,
        "mentioned_paths": mentioned_paths,
    }
    _BUNDLE_CACHE[key] = index
    return index


def _verify_citation(cit: dict[str, str], idx: dict[str, Any]) -> bool:
    """Check whether a single citation exists in the bundle index."""
    kind, val = cit["kind"], cit["value"]
    if kind == "sha":
        v = val.lower()
        # match by prefix: full sha matches its own prefix;
        # 8-12 char short shas match any indexed sha that starts with them.
        return any(s.startswith(v) or v.startswith(s) for s in idx["shas"])
    if kind == "pr":
        return val in idx["prs"]
    if kind == "run_id":
        return val in idx["run_ids"]
    if kind == "path":
        v = val.lower()
        candidate = idx["paths"] | idx["mentioned_paths"]
        # Accept exact, or suffix match (agent may cite a shorter repo-
        # relative path while the bundle path has a prefix dir), or the
        # reverse (agent cites a full path that's mentioned as a suffix
        # somewhere in bundle text).
        if v in candidate:
            return True
        tag = "/" + v
        return any(p == v or p.endswith(tag) or v.endswith("/" + p) for p in candidate)
    return False


def _anti_fake(
    max_fabricated_ratio: float,
    min_verified: int,
    case_root: Path,
) -> tuple[bool, list[dict[str, Any]]]:
    """Check that the agent's cited identifiers actually exist in the bundle.

    Passes when BOTH hold:
      - at least `min_verified` citations verify (floor on grounding), AND
      - fabricated fraction <= `max_fabricated_ratio` (ceiling on hallucination).

    An agent that cites three real PRs and zero fake things passes cleanly.
    An agent that cites `#99999`, `deadbeef`, `src/fake.py` fails even if
    the prose is eloquent. An agent that cites many real things plus one
    typo stays safe as long as it's below the ratio.
    """
    cits = _SUBMISSION.get("citations", []) or []
    idx = _bundle_index(case_root)
    results: list[dict[str, Any]] = []
    for c in cits:
        ok = _verify_citation(c, idx)
        results.append({**c, "verified": ok})
    n_total = len(results)
    n_verified = sum(1 for r in results if r["verified"])
    n_fake = n_total - n_verified
    ratio = (n_fake / n_total) if n_total else 0.0
    passed = n_verified >= min_verified and ratio <= max_fabricated_ratio
    return passed, results



# ============================================================================
# Scaled judge (0..N per axis, hard caps, bonus)
# ============================================================================


SCALED_JUDGE_SYSTEM = """You are a strict research-reconstruction grader.

You will be given:
  * A rubric with N named axes. Each axis has a ground-truth description and
    is scored on an integer scale from 0 to MAX (inclusive).
  * Optionally, a set of hard-cap conditions. Each cap has a name, a
    description of the failure it detects, and a cap value. If the cap
    triggers, the final reward is capped at that value.
  * Optionally, a bonus condition with a description and a value. If the
    bonus triggers, its value is added after caps are applied.
  * The candidate's written report.

For each axis:
  MAX    = materially reconstructs the hidden boundary on this dimension.
  MAX-1  = mostly right with minor gaps or one clear lapse.
  MAX/2  = partial; the core claim is present but under-supported, vague,
           or mixed with incorrect claims.
  1      = token gesture, fundamentally wrong framing.
  0      = absent, inverted, or fundamentally wrong.

For each cap/bonus: mark `triggered: true` only if the report clearly
satisfies the described condition. When in doubt, do not trigger.

Respond with a single JSON object of the form:
{
  "axes":  {"<axis_name>": {"score": int, "why": "<short>"}, ...},
  "caps":  {"<cap_name>":  {"triggered": bool, "why": "<short>"}, ...},
  "bonus": {"triggered": bool, "why": "<short>"}   // omit key if no bonus
}
No prose before or after the JSON object.
"""


def _run_scaled_judge(
    rubric: dict[str, str],
    axis_scale: int,
    hard_caps: list[dict[str, Any]] | None,
    bonus: dict[str, Any] | None,
) -> dict[str, Any]:
    """Ask the LLM judge for 0..axis_scale per-axis scores plus cap/bonus flags."""
    from openai import OpenAI  # required, installed via pyproject

    api_key = os.environ.get("HUD_API_KEY") or os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("HUD_GATEWAY_URL", "https://inference.hud.ai")
    model = os.environ.get("CI_JUDGE_MODEL", "claude-sonnet-4-5")
    if not api_key:
        raise RuntimeError(
            "Grader cannot run: neither HUD_API_KEY nor OPENAI_API_KEY "
            "is set in the container environment. Deploy the env with "
            "a judge key or inject one at remote-run time. Failing "
            "loudly so this does not silently score every run as 0."
        )

    report = _SUBMISSION.get("report", "")

    caps_block = (
        "\n".join(f"- {c['name']}: {c['description']} (cap={c['cap']})" for c in hard_caps)
        if hard_caps else "(none)"
    )
    bonus_block = (
        f"{bonus['description']} (value={bonus['value']})" if bonus else "(none)"
    )
    user_msg = (
        f"AXIS SCORING SCALE: 0..{axis_scale}\n\n"
        "RUBRIC (axis -> ground truth):\n"
        + "\n".join(f"- {k}: {v}" for k, v in rubric.items())
        + "\n\nHARD CAPS:\n" + caps_block
        + "\n\nBONUS:\n" + bonus_block
        + "\n\nCANDIDATE REPORT:\n" + report
    )

    client = OpenAI(api_key=api_key, base_url=base_url)
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SCALED_JUDGE_SYSTEM.replace("MAX", str(axis_scale))},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.0,
        )
        content = (resp.choices[0].message.content or "").strip()
    except Exception as e:
        return {"_error": f"judge call failed: {type(e).__name__}: {e}"}

    text = content
    if text.startswith("```"):
        text = text.strip("`")
        text = text.split("\n", 1)[1] if "\n" in text else text
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        return {"_error": f"judge returned non-JSON: {e}", "raw": content[:500]}


def _grade_scaled(
    rubric: dict[str, str],
    axis_weights: dict[str, float],
    axis_scale: int,
    hard_caps: list[dict[str, Any]] | None,
    bonus: dict[str, Any] | None,
    anti_fake: dict[str, Any],
    case_root: Path,
    report_filename: str,
) -> EvaluationResult:
    if not _SUBMISSION:
        print(f"[grade] no {report_filename} found at scenario end", file=sys.stderr)
        return EvaluationResult(
            reward=0.0,
            content=f"No {report_filename} written to the case folder before the agent stopped.",
            info={"reason": "report_missing"},
        )

    min_verified = int(anti_fake.get("min_verified", 3))
    max_ratio = float(anti_fake.get("max_fabricated_ratio", 0.33))
    passed, cit_results = _anti_fake(max_ratio, min_verified, case_root)
    n_verified = sum(1 for r in cit_results if r["verified"])
    n_total = len(cit_results)
    for r in cit_results:
        tag = "OK" if r["verified"] else "FAKE"
        print(f"[grade] cite {tag:4s} {r['kind']:6s} {r['value']}", file=sys.stderr)
    info: dict[str, Any] = {
        "citations": {
            "verified": n_verified,
            "total": n_total,
            "min_required": min_verified,
            "max_fabricated_ratio": max_ratio,
            "details": cit_results,
        }
    }
    if not passed:
        reason = (
            "insufficient_grounding"
            if n_verified < min_verified
            else "too_many_fabricated_citations"
        )
        print(f"[grade] anti-fake FAILED — {n_verified}/{n_total} verified "
              f"(need ≥{min_verified}, ≤{max_ratio:.0%} fake)", file=sys.stderr)
        return EvaluationResult(
            reward=0.0,
            content=(
                f"Anti-fake failed ({reason}): "
                f"{n_verified}/{n_total} citations verify against the bundle."
            ),
            info={**info, "reason": reason},
        )
    print(f"[grade] anti-fake OK — {n_verified}/{n_total} citations verified",
          file=sys.stderr)

    judge = _run_scaled_judge(rubric, axis_scale, hard_caps, bonus)
    if "_error" in judge:
        print(f"[grade] judge error: {judge['_error']}", file=sys.stderr)
        return EvaluationResult(
            reward=0.0,
            isError=True,
            content=f"Judge call failed: {judge['_error']}",
            info={**info, "judge_error": judge["_error"]},
        )

    axes_out = judge.get("axes") or {}
    caps_out = judge.get("caps") or {}
    bonus_out = judge.get("bonus") or {}

    total_w = sum(axis_weights.values()) or 1.0
    score = 0.0
    subscores: list[SubScore] = []
    axis_summaries: list[str] = []
    judge_axes: dict[str, Any] = {}
    for axis, w in axis_weights.items():
        entry = axes_out.get(axis) or {}
        raw = entry.get("score", 0)
        try:
            s = int(raw)
        except (TypeError, ValueError):
            s = 0
        s = max(0, min(axis_scale, s))
        why = str(entry.get("why", ""))
        wn = w / total_w
        fraction = s / axis_scale if axis_scale else 0.0
        print(f"[grade] {axis:32s} w={wn:.2f} s={s}/{axis_scale} {why}", file=sys.stderr)
        score += wn * fraction
        subscores.append(SubScore(name=axis, weight=wn, value=fraction))
        axis_summaries.append(f"{axis}={s}/{axis_scale}")
        judge_axes[axis] = {"score": s, "why": why}

    pre_cap = score
    applied_caps: list[dict[str, Any]] = []
    for cap in hard_caps or []:
        entry = caps_out.get(cap["name"]) or {}
        triggered = bool(entry.get("triggered", False))
        applied_caps.append({
            "name": cap["name"],
            "triggered": triggered,
            "cap": cap["cap"],
            "why": entry.get("why", ""),
        })
        if triggered:
            print(f"[grade] cap {cap['name']} TRIGGERED -> reward <= {cap['cap']}",
                  file=sys.stderr)
            score = min(score, cap["cap"])

    bonus_applied = False
    bonus_why = ""
    if bonus:
        triggered = bool(bonus_out.get("triggered", False))
        bonus_why = str(bonus_out.get("why", ""))
        if triggered:
            bonus_applied = True
            score += float(bonus["value"])
            print(f"[grade] bonus TRIGGERED (+{bonus['value']}): {bonus_why}",
                  file=sys.stderr)

    reward = max(0.0, min(1.0, score))
    content = (
        f"reward={reward:.3f} (pre-cap={pre_cap:.3f}) | "
        f"{' '.join(axis_summaries)} | "
        f"citations: {n_verified}/{n_total} verified"
    )
    info.update({
        "judge": judge_axes,
        "caps": applied_caps,
        "bonus": {"applied": bonus_applied, "why": bonus_why} if bonus else None,
        "pre_cap_reward": pre_cap,
    })
    return EvaluationResult(reward=reward, content=content, info=info, subscores=subscores)


# ============================================================================
# Scenario: research-study reconstruction (scaled rubric, report_filename)
# ============================================================================


@env.scenario(name="diagnose_research_study")
async def diagnose_research_study(
    prompt: str,
    rubric: dict[str, str],
    case: str,
    axis_weights: dict[str, float],
    axis_scale: int = 4,
    hard_caps: list[dict[str, Any]] | None = None,
    bonus: dict[str, Any] | None = None,
    report_filename: str = "REPORT.md",
    anti_fake: dict[str, Any] | None = None,
):
    """Same materialise-then-grade flow as diagnose_ci_failure, but with
    0..N per-axis scoring, optional hard caps, and an optional bonus. The
    agent writes `report_filename` into /work; grader applies anti-fake +
    scaled judge + caps + bonus.
    """
    case_root = _mount_case(case)
    _SUBMISSION.clear()
    _BUNDLE_CACHE.pop(str(case_root), None)
    try:
        (WORK / report_filename).unlink()
    except (OSError, FileNotFoundError):
        pass
    yield prompt
    _load_report(report_filename)
    yield _grade_scaled(
        rubric=rubric,
        axis_weights=axis_weights,
        axis_scale=axis_scale,
        hard_caps=hard_caps,
        bonus=bonus,
        anti_fake=anti_fake or {"min_verified": 3, "max_fabricated_ratio": 0.33},
        case_root=case_root,
        report_filename=report_filename,
    )


# ============================================================================
# Public toolkit
# ============================================================================

mount_case = _mount_case
load_report = _load_report
extract_citations = _extract_citations
bundle_index = _bundle_index
verify_citation = _verify_citation
anti_fake_gate = _anti_fake
run_scaled_judge = _run_scaled_judge


if __name__ == "__main__":
    env.run(transport="stdio")
