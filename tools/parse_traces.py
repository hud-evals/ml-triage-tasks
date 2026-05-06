"""Slice HUD job telemetry into a reviewable per-rollout summary.

Pulls `GET /telemetry/job/<id>/trace-ids` + `GET /telemetry/trace/<id>`
from the HUD platform (or loads them from a previously-fetched cache
dir), extracts per-rollout reward / status / subscores / tool-call
counts, and aggregates rewards across the group.

Scenario-agnostic: no assumption about the grader's `info` shape or
the agent's deliverable. The full `info` dict is rendered raw under
verbose mode; subscores aggregate by name (a HUD-level concept).

Usage
-----
    python tools/parse_traces.py --job <job_id>
    python tools/parse_traces.py --job <job_id> -v          # also show bash + raw info
    python tools/parse_traces.py --job <job_id> --only 9d4  # one trace by prefix
    python tools/parse_traces.py --job <job_id> --raw-dir /tmp/hud_traces
    python tools/parse_traces.py --from-dir /tmp/hud_traces  # offline, already fetched

Auth: reads HUD_API_KEY from env.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# --- fetch ------------------------------------------------------------------


def _hud_api_url() -> str:
    return os.environ.get("HUD_API_URL", "https://api.hud.ai").rstrip("/")


def _hud_api_key() -> str:
    k = os.environ.get("HUD_API_KEY")
    if not k:
        raise SystemExit("HUD_API_KEY not set")
    return k


def _get(path: str) -> Any:
    url = f"{_hud_api_url()}/{path.lstrip('/')}"
    req = urllib.request.Request(
        url, headers={"Authorization": f"Bearer {_hud_api_key()}", "Accept": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def fetch_job(job_id: str) -> dict[str, Any]:
    return _get(f"telemetry/job/{job_id}")


def fetch_trace_ids(job_id: str) -> list[str]:
    data = _get(f"telemetry/job/{job_id}/trace-ids")
    return list(data.get("trace_ids") or [])


def fetch_trace(trace_id: str) -> dict[str, Any]:
    return _get(f"telemetry/trace/{trace_id}")


def pull_job_to_dir(job_id: str, out_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    job = fetch_job(job_id)
    (out_dir / "job.json").write_text(json.dumps(job, indent=2), encoding="utf-8")
    traces: list[dict[str, Any]] = []
    for tid in fetch_trace_ids(job_id):
        t = fetch_trace(tid)
        (out_dir / f"trace-{tid}.json").write_text(json.dumps(t, indent=2), encoding="utf-8")
        traces.append(t)
    return job, traces


# --- load from disk ---------------------------------------------------------


def load_dir(path: Path) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    job = None
    job_path = path / "job.json"
    if job_path.is_file():
        job = json.loads(job_path.read_text())
    traces: list[dict[str, Any]] = []
    for f in sorted(path.glob("trace-*.json")):
        traces.append(json.loads(f.read_text()))
    return job, traces


# --- rollout slicing --------------------------------------------------------


@dataclass
class Rollout:
    trace_id: str
    status: str | None = None
    reward: float | None = None
    is_error: bool = False
    created_at: str | None = None
    content_line: str | None = None        # EvaluationResult.content
    subscores: dict[str, float] = field(default_factory=dict)
    info: dict[str, Any] = field(default_factory=dict)  # raw EvaluationResult.info
    bash_calls: list[dict[str, Any]] = field(default_factory=list)
    n_llm_turns: int = 0
    n_tool_calls: int = 0


def _attr_text_content(result: Any) -> str:
    """MCP tools-call result payload → string."""
    if isinstance(result, dict):
        contents = result.get("content") or result.get("contents") or []
        if isinstance(contents, list):
            parts = []
            for c in contents:
                if isinstance(c, dict) and (c.get("type") == "text" or "text" in c):
                    parts.append(c.get("text") or "")
            if parts:
                return "\n".join(parts)
        if isinstance(result.get("output"), str):
            return result["output"]
    if isinstance(result, str):
        return result
    return ""


def slice_rollout(trace: dict[str, Any]) -> Rollout:
    r = Rollout(trace_id=trace.get("id") or "unknown")
    r.status = trace.get("status")
    r.reward = trace.get("reward")
    r.created_at = trace.get("created_at")

    meta = trace.get("metadata") or {}
    er = meta.get("evaluation_result") or {}
    if isinstance(er, dict):
        if isinstance(er.get("reward"), (int, float)):
            r.reward = float(er["reward"])
        r.content_line = er.get("content")
        r.is_error = bool(er.get("isError"))
        for ss in er.get("subscores") or []:
            if isinstance(ss, dict) and "name" in ss and "value" in ss:
                try:
                    r.subscores[str(ss["name"])] = float(ss["value"])
                except (TypeError, ValueError):
                    pass
        if isinstance(er.get("info"), dict):
            r.info = er["info"]

    for span in trace.get("trajectory") or []:
        if not isinstance(span, dict):
            continue
        name = (span.get("name") or "").lower()
        attrs = span.get("attributes") or {}
        req = (attrs.get("request") or {}) if isinstance(attrs, dict) else {}
        params = (req.get("params") or {}) if isinstance(req, dict) else {}
        result = attrs.get("result") if isinstance(attrs, dict) else None

        if "inference" in name or "completion" in name or "chat" in name:
            r.n_llm_turns += 1
            continue

        if name == "tools/call.mcp":
            r.n_tool_calls += 1
            tool_name = params.get("name") if isinstance(params, dict) else None
            if tool_name == "bash":
                args = params.get("arguments") or {}
                cmd = args.get("command") if isinstance(args, dict) else None
                out = _attr_text_content(result)
                r.bash_calls.append({"command": cmd, "output": out})
            continue

        if name == "resources/read.mcp":
            text = _attr_text_content(result)
            if not text:
                continue
            try:
                payload = json.loads(text)
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue
            if isinstance(payload.get("info"), dict):
                r.info = payload["info"]
            if isinstance(payload.get("reward"), (int, float)):
                r.reward = float(payload["reward"])

    return r


# --- rendering --------------------------------------------------------------


def _fmt_reward(r: float | None) -> str:
    return f"{r:.3f}" if isinstance(r, (int, float)) else "—"


def _pretty_info(info: dict[str, Any], indent: str = "      ", max_chars: int = 1800) -> str:
    """Pretty-print a generic info dict. Truncates long values."""
    if not info:
        return ""
    try:
        text = json.dumps(info, indent=2, default=str)
    except Exception:
        text = repr(info)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n... [truncated]"
    return "\n".join(indent + line for line in text.splitlines())


def render_rollout(r: Rollout, verbose: bool = False) -> str:
    lines: list[str] = [f"=== trace {r.trace_id[:8]}  reward={_fmt_reward(r.reward)}"]
    lines.append(
        f"    status={r.status}  tool_calls={r.n_tool_calls}  llm_turns={r.n_llm_turns}"
        + ("  ERROR" if r.is_error else "")
    )
    if r.content_line:
        lines.append(f"    [{r.content_line}]")
    if r.subscores:
        lines.append("    subscores:")
        for name, value in sorted(r.subscores.items(), key=lambda kv: -kv[1]):
            lines.append(f"      {name:32s} {value:.3f}")
    if verbose and r.info:
        lines.append("    info:")
        lines.append(_pretty_info(r.info))
    if verbose and r.bash_calls:
        lines.append(f"    bash ({len(r.bash_calls)} total, last 15):")
        for bc in r.bash_calls[-15:]:
            cmd = (bc.get("command") or "").splitlines()[0][:140]
            lines.append(f"      $ {cmd}")
    return "\n".join(lines)


def render_aggregate(rollouts: list[Rollout], job: dict[str, Any] | None) -> str:
    lines: list[str] = ["", "=== aggregate ==="]
    if job:
        lines.append(f"job: {job.get('name')}  id={job.get('id')}")
        m = job.get("metrics") or {}
        if m:
            lines.append(
                f"platform metrics: avg_reward={m.get('average_reward')}  "
                f"completion_rate={m.get('completion_rate')}  "
                f"avg_steps={m.get('average_steps')}  "
                f"time_elapsed={m.get('time_elapsed')}"
            )

    n = len(rollouts)
    lines.append(f"rollouts: {n}")
    if not n:
        return "\n".join(lines)

    rewards = [r.reward for r in rollouts if isinstance(r.reward, (int, float))]
    n_errors = sum(1 for r in rollouts if r.is_error)
    n_no_reward = n - len(rewards)
    if rewards:
        mean = statistics.mean(rewards)
        median = statistics.median(rewards)
        std = statistics.pstdev(rewards) if len(rewards) > 1 else 0.0
        lines.append(
            f"reward: mean={mean:.3f}  median={median:.3f}  "
            f"min={min(rewards):.3f}  max={max(rewards):.3f}  "
            f"std={std:.3f}  n_graded={len(rewards)}/{n}"
        )
    if n_errors:
        lines.append(f"errors: {n_errors}/{n}")
    if n_no_reward and n_no_reward != n_errors:
        lines.append(f"missing reward: {n_no_reward}/{n}")

    # Per-subscore aggregates (HUD-generic; not scenario-specific).
    sub_vals: dict[str, list[float]] = defaultdict(list)
    for r in rollouts:
        for k, v in r.subscores.items():
            sub_vals[k].append(v)
    if sub_vals:
        lines.append("subscores (mean across rollouts):")
        for k, vs in sorted(sub_vals.items(), key=lambda kv: -sum(kv[1]) / len(kv[1])):
            lines.append(f"  {k:32s} {sum(vs)/len(vs):.3f}   (n={len(vs)})")

    # Info-key frequency: which top-level keys do scenarios put in info?
    # Useful for spotting schema drift across rollouts.
    key_counts: Counter = Counter()
    for r in rollouts:
        for k in r.info:
            key_counts[k] += 1
    if key_counts and any(c < n for c in key_counts.values()):
        lines.append("info keys (count of rollouts with each key):")
        for k, c in key_counts.most_common():
            lines.append(f"  {c:3d}/{n}  {k}")

    # Generic task-health warnings.
    warnings: list[str] = []
    if rewards and max(rewards) == min(rewards) and len(rewards) > 1:
        warnings.append(f"!! all {len(rewards)} graded rollouts scored {max(rewards):.3f} — task not discriminating")
    if rewards and max(rewards) == 0.0:
        warnings.append("!! every graded rollout scored 0 — agent failure or grader broken")
    if n_errors and n_errors / n > 0.3:
        warnings.append(f"!! {n_errors}/{n} rollouts errored")
    if n_no_reward and n_no_reward / n > 0.3:
        warnings.append(f"!! {n_no_reward}/{n} rollouts produced no reward")
    if warnings:
        lines.append("")
        lines.extend(warnings)

    return "\n".join(lines)


# --- cli --------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--job", help="job id to fetch from the platform")
    grp.add_argument("--from-dir", help="directory of already-fetched job.json + trace-*.json")
    p.add_argument("--raw-dir", help="cache dir to write fetched traces to (default: /tmp/hud_job_<job_id>)")
    p.add_argument("-v", "--verbose", action="store_true", help="show bash commands + raw info")
    p.add_argument("--only", help="filter to rollouts whose trace_id starts with this prefix")
    args = p.parse_args(argv)

    if args.job:
        raw_dir = Path(args.raw_dir or f"/tmp/hud_job_{args.job}")
        job, traces = pull_job_to_dir(args.job, raw_dir)
        print(f"[fetched] {len(traces)} traces to {raw_dir}", file=sys.stderr)
    else:
        job, traces = load_dir(Path(args.from_dir))

    rollouts = [slice_rollout(t) for t in traces]
    if args.only:
        rollouts = [r for r in rollouts if r.trace_id.startswith(args.only)]

    rollouts.sort(key=lambda r: (r.reward if r.reward is not None else -1.0), reverse=True)
    for r in rollouts:
        print(render_rollout(r, verbose=args.verbose))
        print()
    print(render_aggregate(rollouts, job))
    return 0


if __name__ == "__main__":
    sys.exit(main())
