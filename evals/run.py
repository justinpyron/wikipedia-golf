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

from agent import WikiGolfDeps, build_agent
from evals.datasets import DATASETS
from evals.types import WikiGolfEvalInput, WikiGolfEvalOutput
from variants import VARIANTS

load_dotenv()
logfire.configure(service_name="wiki-golf-evals", environment="dev")


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


def get_git_info() -> tuple[str, str]:
    """Get the current git SHA and commit subject atomically."""
    try:
        output = subprocess.check_output(
            ["git", "log", "-1", "--pretty=format:%H%n%s"], text=True
        ).strip()
        sha, msg = output.split("\n", 1)
        return sha, msg
    except Exception:
        return "unknown", "unknown"


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
        variant = next(v for v in VARIANTS if v.name == args.variant)
    except StopIteration:
        raise KeyError(f"Variant '{args.variant}' does not exist in VARIANTS.")

    try:
        dataset = DATASETS[args.dataset]
    except KeyError:
        raise KeyError(f"Dataset '{args.dataset}' does not exist in DATASETS.")
    task = build_task(variant)

    sha, msg = get_git_info()
    metadata = {
        "variant": asdict(variant),
        "git_sha": sha,
        "git_commit_message": msg,
    }

    report = asyncio.run(
        dataset.evaluate(task, metadata=metadata, max_concurrency=args.max_concurrency)
    )
    report.print()


if __name__ == "__main__":
    main()
