"""Run an agent against the sandboxed env in Docker, all local.

Usage:
    uv run --with 'hud-python' --with openai python local_test.py \
        --task prime_rl_chunk_default_tradeoff --model grok-4.20

    uv run --with 'hud-python' python local_test.py --list

The container runs the MCP env; this script connects via connect_image()
and drives the agent through the HUD inference gateway.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import os
import sys
from pathlib import Path

import hud
from hud.agents import OpenAIChatAgent
from hud.settings import settings
from openai import AsyncOpenAI

IMAGE = "ml-triage-tasks:local"
ENV_ROOT = Path(__file__).resolve().parent.parent
TASK_DIR = ENV_ROOT / "tasks"
GATEWAY = os.environ.get("HUD_GATEWAY_URL", "https://inference.hud.ai")


def _available_tasks() -> list[str]:
    return sorted(
        d.name for d in TASK_DIR.iterdir()
        if d.is_dir() and (d / "task.py").exists()
    )


def _load_task(name: str):
    sys.path.insert(0, str(TASK_DIR))
    sys.path.insert(0, str(ENV_ROOT))
    mod = importlib.import_module(f"{name}.task")
    return mod.task


async def main() -> None:
    available = _available_tasks()
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="prime_rl_chunk_default_tradeoff", choices=available)
    parser.add_argument("--model", default="grok-4.20")
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    if args.list:
        for t in available:
            print(t)
        return

    api_key = settings.api_key or os.environ.get("HUD_API_KEY")
    if not api_key:
        raise SystemExit("HUD_API_KEY not set. export HUD_API_KEY=... first.")

    task = _load_task(args.task)

    forwarded = {"HUD_API_KEY": api_key, "HUD_GATEWAY_URL": GATEWAY}
    for k in ("CI_JUDGE_MODEL", "JUDGE_MODEL"):
        if os.environ.get(k):
            forwarded[k] = os.environ[k]

    docker_env = hud.Environment("ml-triage-tasks")
    docker_env.connect_image(IMAGE, env_vars=forwarded)
    task.env = docker_env

    print(f"=== {task.slug} | agent={args.model} ===")
    client = AsyncOpenAI(base_url=GATEWAY, api_key=api_key)
    async with hud.eval(task) as ctx:
        agent = OpenAIChatAgent(openai_client=client, model=args.model)
        await agent.run(ctx, max_steps=args.max_steps)
        print(f"Reward: {getattr(ctx, 'reward', None)}")


if __name__ == "__main__":
    asyncio.run(main())
