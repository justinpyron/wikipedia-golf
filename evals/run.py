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

from agent import WikiGolfDeps, WikiGolfInput, WikiGolfOutput, build_agent
from evals.datasets import DATASETS
from variants import VARIANTS

load_dotenv()
logfire.configure(service_name="wiki-golf-evals", environment="dev")


def build_task(variant) -> Callable[[WikiGolfInput], Awaitable[WikiGolfOutput]]:
    """Build an eval-compatible async callable from a variant."""
    agent = build_agent(variant)

    async def task(inputs: WikiGolfInput) -> WikiGolfOutput:
        deps = WikiGolfDeps(origin=inputs.origin, destination=inputs.destination)
        result = await agent.run(variant.user_prompt, deps=deps)
        return result.output

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
    variant_names = [v.name for v in VARIANTS]

    parser = argparse.ArgumentParser(description="Run Wikipedia Golf evaluations")
    parser.add_argument("-d", "--dataset", required=True, choices=list(DATASETS.keys()))
    parser.add_argument("-v", "--variant", required=True, choices=variant_names)
    args = parser.parse_args()

    variant = next(v for v in VARIANTS if v.name == args.variant)
    dataset = DATASETS[args.dataset]
    task = build_task(variant)

    metadata = {
        "variant": asdict(variant),
        "git_sha": get_git_sha(),
    }

    report = asyncio.run(dataset.evaluate(task, metadata=metadata))
    report.print()


if __name__ == "__main__":
    main()
