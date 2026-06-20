"""Run one task N times in parallel against a model with Task.run(group=...).

Usage:
    uv run --with hud-python --with openai python run_many.py \
        --task prime_rl_chunk_default_tradeoff --model grok-4.20 --n 5
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

from hud import DockerRuntime
from hud.agents import create_agent
from hud.settings import settings

IMAGE = "ml-triage-tasks:local"
ENV_ROOT = Path(__file__).resolve().parent.parent
TASK_DIR = ENV_ROOT / "tasks"
GATEWAY = os.environ.get("HUD_GATEWAY_URL", "https://inference.beta.hud.ai")


def _load_task(name: str):
    for p in (str(ENV_ROOT), str(TASK_DIR)):
        if p not in sys.path:
            sys.path.insert(0, p)
    return importlib.import_module(f"{name}.task").task


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="prime_rl_chunk_default_tradeoff")
    parser.add_argument("--model", default="grok-4.20")
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--n", type=int, default=5)
    args = parser.parse_args()

    api_key = settings.api_key or os.environ.get("HUD_API_KEY")
    if not api_key:
        raise SystemExit("HUD_API_KEY not set")

    forwarded = {"HUD_API_KEY": api_key, "HUD_GATEWAY_URL": GATEWAY}
    for k in ("CI_JUDGE_MODEL", "JUDGE_MODEL"):
        if os.environ.get(k):
            forwarded[k] = os.environ[k]

    task = _load_task(args.task)
    run_args: list[str] = []
    for key, value in forwarded.items():
        run_args.extend(["-e", f"{key}={value}"])
    runtime = DockerRuntime(IMAGE, run_args=run_args)

    print(f"=== {args.task} | agent={args.model} | n={args.n} (parallel) ===")
    start = time.time()
    agent = create_agent(args.model, max_steps=args.max_steps)
    job = await task.run(agent, runtime=runtime, group=args.n, max_concurrent=args.n)
    dt = time.time() - start

    rewards: list[float] = []
    for i, run in enumerate(job.runs, 1):
        r = getattr(run, "reward", None)
        print(f"[{i}/{args.n}] reward={r!r}")
        if r is not None:
            rewards.append(float(r))

    print(f"\nwall time: {dt:.0f}s")
    print(f"got reward: {len(rewards)}/{args.n}")
    if rewards:
        print(
            f"summary: "
            f"mean={statistics.mean(rewards):.3f} "
            f"median={statistics.median(rewards):.3f} "
            f"min={min(rewards):.3f} max={max(rewards):.3f}"
        )
    else:
        print("no runs produced a reward")


if __name__ == "__main__":
    asyncio.run(main())
