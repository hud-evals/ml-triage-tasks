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

env = Environment(name="ci-triage-tasks")

# Cases are baked into the image under /opt/ci_cases/<slug>/, invisible to the
# agent. At scenario-start time we clear /work and symlink its children to the
# selected case's children — so the agent's bash cwd is `/work` and `ls` shows
# repo/ logs/ prs/ etc. with no case-slug prefix in any path they see.
CASES_ROOT = Path(os.environ.get("CI_CASES_ROOT", "/opt/ci_cases"))
WORK = Path(os.environ.get("CI_WORK", "/work"))


try:
    WORK.mkdir(parents=True, exist_ok=True)
except OSError:
    pass


def _mount_case(case: str) -> Path:
    """Copy one case's contents into `/work` so `/work` *is* the case dir.

    We don't symlink — `ls -la` on a symlink tree would leak the case slug
    via the link target (e.g. `/opt/ci_cases/<slug>/repo`). Hard-copying
    keeps every path the agent sees unprefixed and slug-free. Returns the
    source path so the grader can anti-fake against it.
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
    return src


@env.tool(name="bash")
async def bash(command: str) -> ContentResult:
    """Run a bash command. Each call starts a fresh shell rooted at the
    folder I scraped into — no session state between calls, so `cd` and
    environment changes don't persist. Use absolute paths or paths
    relative to the current folder.

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
    )
    try:
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=120.0)
    except asyncio.TimeoutError:
        proc.kill()
        return ContentResult(output="command timed out after 120s", error="timeout")
    text = (stdout or b"").decode("utf-8", errors="replace")
    # Hard cap on output size to keep the agent's context tight.
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


_QUOTE_RE = re.compile(
    r"`([^`\n]{40,})`"          # backtick-fenced snippet, ≥40 chars, single line
    r"|\"([^\"\n]{40,})\""      # double-quoted line, ≥40 chars
    r"|'([^'\n]{40,})'"          # single-quoted line, ≥40 chars
)


def _extract_quotes(report: str) -> list[str]:
    """Pull plausibly-verbatim substrings out of the report prose.

    Matches backtick-fenced and quoted snippets ≥25 chars on a single line.
    These are what the anti-fake gate will grep against the bundle.
    """
    seen: set[str] = set()
    out: list[str] = []
    for m in _QUOTE_RE.finditer(report):
        q = next((g for g in m.groups() if g), "").strip()
        if q and q not in seen:
            seen.add(q)
            out.append(q)
    return out


REPORT_PATH = WORK / "REPORT.md"


def _load_report() -> None:
    """Read REPORT.md from the workspace (if present) into _SUBMISSION."""
    _SUBMISSION.clear()
    try:
        text = REPORT_PATH.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return
    _SUBMISSION.update({"report": text, "quotes": _extract_quotes(text)})


# ============================================================================
# Anti-fake: verify each evidence quote literally appears in the bundle
# ============================================================================


_BUNDLE_CACHE: dict[str, dict[str, str]] = {}


def _load_case_text(case_root: Path) -> dict[str, str]:
    """Lazy-load all text content under `case_root` into memory."""
    key = str(case_root)
    if key in _BUNDLE_CACHE:
        return _BUNDLE_CACHE[key]
    data: dict[str, str] = {}
    if case_root.is_dir():
        for p in case_root.rglob("*"):
            if not p.is_file():
                continue
            try:
                data[str(p)] = p.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
    _BUNDLE_CACHE[key] = data
    return data


def _verify_quote(quote: str, case_root: Path) -> str | None:
    """Return the first file under case_root whose content contains `quote`."""
    needle = quote.strip()
    if len(needle) < 40:
        return None
    for path, text in _load_case_text(case_root).items():
        if needle in text:
            return path
    return None


def _anti_fake(min_verified: int, case_root: Path) -> tuple[bool, list[dict[str, Any]]]:
    """At least `min_verified` quotes must literally appear inside case_root.

    Extra unverified quotes do NOT penalize — the gate is a floor on grounded
    evidence, not a quota on overall cleanliness. Fabricated quotes are
    visible in the per-quote log for debugging.
    """
    quotes = _SUBMISSION.get("quotes", []) or []
    results: list[dict[str, Any]] = []
    for q in quotes:
        path = _verify_quote(q, case_root)
        results.append({
            "quote": q[:80] + ("..." if len(q) > 80 else ""),
            "verified": path is not None,
            "source": path,
        })
    verified = sum(1 for r in results if r["verified"])
    passed = verified >= min_verified
    return passed, results


# ============================================================================
# Agentic judge
# ============================================================================


JUDGE_SYSTEM = """You are a strict CI-failure-diagnosis grader.

You will be given:
  * A rubric with N named axes. Each axis comes with a ground-truth description.
  * The candidate's written diagnosis report.
  * A list of evidence quotes the candidate supplied (already verified as
    literally present in the source bundle).

Score each axis strictly as 0 or 1 (no partials):
  * 1 = the report unambiguously asserts the correct answer for that axis
        and supports it with reasoning consistent with the evidence.
  * 0 = the report is wrong, missing, hedging between right and wrong, or
        asserts the right conclusion but for wrong reasons.

Respond with a single JSON object whose keys are EXACTLY the axis names you
were given, each mapping to {"score": 0|1, "why": "<short reason>"}. No prose
before or after the JSON object.
"""


def _run_judge(rubric: dict[str, str]) -> dict[str, Any]:
    """Ask an LLM judge to score the submission against `rubric`.

    Uses the HUD inference gateway via OpenAI-compatible client. Requires
    HUD_API_KEY in the container environment. Returns a dict like
    `{axis: {"score": int, "why": str}}`. On failure returns all-zero with
    the error message attached.
    """
    try:
        from openai import OpenAI
    except ImportError:
        return {"_error": "openai package not installed in env"}

    api_key = os.environ.get("HUD_API_KEY") or os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("HUD_GATEWAY_URL", "https://inference.hud.ai")
    model = os.environ.get("CI_JUDGE_MODEL", "claude-sonnet-4-5")
    if not api_key:
        return {"_error": "no HUD_API_KEY / OPENAI_API_KEY set"}

    report = _SUBMISSION.get("report", "")
    quotes = _SUBMISSION.get("quotes", [])
    quotes_block = "\n---\n".join(f"[{i}] {q}" for i, q in enumerate(quotes))

    user_msg = (
        "RUBRIC (axis -> ground truth):\n"
        + "\n".join(f"- {k}: {v}" for k, v in rubric.items())
        + "\n\nCANDIDATE REPORT:\n"
        + report
        + "\n\nEVIDENCE QUOTES (already verified to exist in the source bundle):\n"
        + (quotes_block or "(none)")
    )

    client = OpenAI(api_key=api_key, base_url=base_url)
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM},
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


# ============================================================================
# Grading
# ============================================================================


def _grade(
    rubric: dict[str, str],
    axis_weights: dict[str, float],
    anti_fake: dict[str, Any],
    case_root: Path,
) -> EvaluationResult:
    if not _SUBMISSION:
        print("[grade] no REPORT.md found at scenario end", file=sys.stderr)
        return EvaluationResult(
            reward=0.0,
            content="No REPORT.md written to the case folder before the agent stopped.",
            info={"reason": "report_missing"},
        )

    min_verified = int(anti_fake.get("min_verified", 2))
    passed, quote_results = _anti_fake(min_verified=min_verified, case_root=case_root)
    for r in quote_results:
        tag = "OK" if r["verified"] else "UNVERIFIED"
        print(f"[grade] quote {tag:11s} {r['quote']!r}"
              + (f" in {r['source']}" if r["source"] else ""),
              file=sys.stderr)
    n_verified = sum(1 for r in quote_results if r["verified"])
    n_total = len(quote_results)
    info: dict[str, Any] = {
        "quotes": {
            "verified": n_verified,
            "total": n_total,
            "min_required": min_verified,
            "details": quote_results,
        }
    }
    if not passed:
        print(f"[grade] anti-fake FAILED — only {n_verified}/{n_total} quotes verified "
              f"(need {min_verified})", file=sys.stderr)
        return EvaluationResult(
            reward=0.0,
            content=(
                f"Anti-fake gate failed: only {n_verified}/{n_total} backtick-quoted "
                f"snippets were literally present in the case bundle "
                f"(need {min_verified})."
            ),
            info={**info, "reason": "anti_fake_failed"},
        )
    print(f"[grade] anti-fake OK — {n_verified}/{n_total} quotes verified", file=sys.stderr)

    judge = _run_judge(rubric)
    if "_error" in judge:
        print(f"[grade] judge error: {judge['_error']}", file=sys.stderr)
        return EvaluationResult(
            reward=0.0,
            isError=True,
            content=f"Judge call failed: {judge['_error']}",
            info={**info, "judge_error": judge["_error"]},
        )

    total_w = sum(axis_weights.values()) or 1.0
    score = 0.0
    subscores: list[SubScore] = []
    axis_summaries: list[str] = []
    judge_axes: dict[str, Any] = {}
    for axis, w in axis_weights.items():
        entry = judge.get(axis) or {}
        raw = entry.get("score", 0)
        s = int(raw) if isinstance(raw, (int, float)) or str(raw).isdigit() else 0
        why = str(entry.get("why", ""))
        wn = w / total_w
        print(f"[grade] {axis:28s} w={wn:.2f} s={s} {why}", file=sys.stderr)
        score += wn * s
        subscores.append(SubScore(name=axis, weight=wn, value=float(s)))
        axis_summaries.append(f"{axis}={s}")
        judge_axes[axis] = {"score": s, "why": why}
    reward = max(0.0, min(1.0, score))
    content = (
        f"reward={reward:.3f} | "
        f"{' '.join(axis_summaries)} | "
        f"evidence: {n_verified}/{n_total} quotes grounded"
    )
    info["judge"] = judge_axes
    return EvaluationResult(reward=reward, content=content, info=info, subscores=subscores)


# ============================================================================
# Scenario
# ============================================================================


@env.scenario(name="diagnose_ci_failure")
async def diagnose_ci_failure(
    prompt: str,
    rubric: dict[str, str],
    case: str,
    axis_weights: dict[str, float] | None = None,
    anti_fake: dict[str, Any] | None = None,
):
    """Materialise the named case under /work, then run diagnosis + grading.

    The agent never sees the case slug or the /opt/ci_cases path: their bash
    cwd is /work, populated with the case's children (repo/, logs/, prs/
    etc.). The agent writes a free-form REPORT.md file into /work; the grader
    reads it and judges via anti-fake quote verification + LLM rubric. The
    task prompt is passed through verbatim — no env-side wrapper.
    """
    case_root = _mount_case(case)
    _SUBMISSION.clear()
    _BUNDLE_CACHE.pop(str(case_root), None)
    try:
        REPORT_PATH.unlink()
    except (OSError, FileNotFoundError):
        pass
    yield prompt
    _load_report()
    yield _grade(
        rubric=rubric,
        axis_weights=axis_weights or {
            "proximate_cause": 1.0,
            "pr_attribution": 1.0,
            "recommended_action": 1.0,
        },
        anti_fake=anti_fake or {"min_verified": 2},
        case_root=case_root,
    )


if __name__ == "__main__":
    env.run(transport="stdio")
