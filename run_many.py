"""Run one task N times in parallel against a model, using hud.eval(group=...).

Usage:
    uv run --with hud-python --with openai python run_many.py \
        --task prime_rl_1892_broke --model grok-4.20 --n 5
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import os
import statistics
import sys
import time
from pathlib import Path

import hud
from hud.agents import OpenAIChatAgent
from hud.settings import settings
from openai import AsyncOpenAI

IMAGE = "ci-triage-tasks:local"
TASK_DIR = Path(__file__).parent / "tasks"
GATEWAY = os.environ.get("HUD_GATEWAY_URL", "https://inference.hud.ai")


def _load_task(name: str):
    pkg_dir = TASK_DIR / name
    for p in (str(pkg_dir.parent.parent), str(TASK_DIR)):
        if p not in sys.path:
            sys.path.insert(0, p)
    return importlib.import_module(f"{name}.task").task


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="prime_rl_1892_broke")
    parser.add_argument("--model", default="grok-4.20")
    parser.add_argument("--judge-model", default="claude-sonnet-4-5")
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--n", type=int, default=5)
    args = parser.parse_args()

    api_key = settings.api_key or os.environ.get("HUD_API_KEY")
    if not api_key:
        raise SystemExit("HUD_API_KEY not set")

    task = _load_task(args.task)
    task.env = hud.Environment("ci-triage-tasks").connect_image(
        IMAGE,
        env_vars={
            "HUD_API_KEY": api_key,
            "HUD_GATEWAY_URL": GATEWAY,
            "CI_JUDGE_MODEL": args.judge_model,
        },
    )
    client = AsyncOpenAI(base_url=GATEWAY, api_key=api_key)

    print(f"=== {args.task} | agent={args.model} | judge={args.judge_model} | n={args.n} (parallel) ===")
    start = time.time()
    async with hud.eval(task, group=args.n, max_concurrent=args.n) as ctx:
        agent = OpenAIChatAgent(openai_client=client, model=args.model)
        await agent.run(ctx, max_steps=args.max_steps)
    dt = time.time() - start

    rewards: list[float] = []
    no_report = 0
    for i, sub in enumerate(getattr(ctx, "results", []) or [], 1):
        r = getattr(sub, "reward", None)
        # The grader flags "agent never wrote REPORT.md" with
        # info.reason == "report_missing" on the full EvaluationResult
        # stored at sub.evaluation_result. Those runs score reward=0 but
        # aren't signal about diagnostic ability — exclude from the mean
        # so we don't bias the aggregate with tool-failure noise.
        er = getattr(sub, "evaluation_result", None)
        info = getattr(er, "info", None) or {}
        if info.get("reason") == "report_missing":
            print(f"[{i}/{args.n}] no_report (no REPORT.md written)")
            no_report += 1
            continue
        print(f"[{i}/{args.n}] reward={r!r}")
        if r is not None:
            rewards.append(float(r))

    print(f"\nwall time: {dt:.0f}s")
    print(f"wrote REPORT.md: {len(rewards)}/{args.n}"
          + (f" (no_report: {no_report})" if no_report else ""))
    if rewards:
        print(
            f"summary (over runs with a report): "
            f"mean={statistics.mean(rewards):.3f} "
            f"median={statistics.median(rewards):.3f} "
            f"min={min(rewards):.3f} max={max(rewards):.3f}"
        )
    else:
        print("no runs produced a REPORT.md")


if __name__ == "__main__":
    asyncio.run(main())
