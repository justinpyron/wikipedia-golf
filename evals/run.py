"""CLI runner for Wikipedia Golf evaluation experiments.

Usage:
    python -m evals.run -d smoke -v baseline
"""

import argparse
import asyncio
import subprocess
from collections.abc import Awaitable, Callable
from dataclasses import asdict

import logfire
from dotenv import load_dotenv
from pydantic import BaseModel
from pydantic_ai.messages import ModelMessage

from agent import WikiGolfDeps, build_agent
from evals.datasets import DATASETS
from variants import VARIANTS

load_dotenv()
logfire.configure(service_name="wiki-golf-evals", environment="dev")


class WikiGolfEvalInput(BaseModel):
    origin: str
    destination: str


class WikiGolfEvalOutput(BaseModel):
    path: list[str]
    messages: list[ModelMessage]


def build_task(variant) -> Callable[[WikiGolfEvalInput], Awaitable[WikiGolfEvalOutput]]:
    """Build an eval-compatible async callable from a variant."""
    agent = build_agent(variant)

    async def task(inputs: WikiGolfEvalInput) -> WikiGolfEvalOutput:
        deps = WikiGolfDeps(origin=inputs.origin, destination=inputs.destination)
        result = await agent.run(variant.user_prompt, deps=deps)
        return WikiGolfEvalOutput(
            path=deps.path,
            messages=result.all_messages(),
        )

    return task


def get_git_sha() -> str:
    """Get the current git SHA."""
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .decode("ascii")
            .strip()
        )
    except Exception:
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Wikipedia Golf evaluations")
    parser.add_argument(
        "-d",
        "--dataset",
        required=True,
        choices=list(DATASETS.keys()),
    )
    parser.add_argument(
        "-v",
        "--variant",
        required=True,
        choices=[v.name for v in VARIANTS],
    )
    parser.add_argument(
        "-c",
        "--max-concurrency",
        type=int,
        default=25,
        help="Max number of concurrent eval cases (default: 25)",
    )
    args = parser.parse_args()

    try:
        variant = VARIANTS[args.variant]
    except KeyError:
        raise KeyError(f"Variant '{args.variant}' does not exist in VARIANTS.")

    try:
        dataset = DATASETS[args.dataset]
    except KeyError:
        raise KeyError(f"Dataset '{args.dataset}' does not exist in DATASETS.")
    task = build_task(variant)

    metadata = {
        "variant": asdict(variant),
        "git_sha": get_git_sha(),
    }

    report = asyncio.run(
        dataset.evaluate(task, metadata=metadata, max_concurrency=args.max_concurrency)
    )
    report.print()


if __name__ == "__main__":
    main()
