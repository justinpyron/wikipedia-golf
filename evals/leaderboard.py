"""Run Wikipedia Golf evaluations across multiple variants for leaderboard workflows.

Usage:
    uv run python -m evals.leaderboard [-d easy] [-c 25]
"""

import argparse
import asyncio
import subprocess
from collections.abc import Awaitable, Callable
from dataclasses import asdict

import logfire
from dotenv import load_dotenv
from pydantic_ai import UsageLimits
from pydantic_evals.reporting import EvaluationReport

from agent import WikiGolfDeps, build_agent
from evals.datasets import DATASETS
from evals.types import WikiGolfEvalInput, WikiGolfEvalOutput
from variants import SYSTEM_PROMPT_V0_0, SYSTEM_PROMPT_V1_0, AgentVariant

load_dotenv()
logfire.configure(service_name="wiki-golf-evals", environment="dev")

LEADERBOARD_VARIANTS: list[AgentVariant] = [
    AgentVariant(
        name="leaderboard_dev-gpt-5.4-mini",
        model="openai:gpt-5.4-mini",
        system_prompt=SYSTEM_PROMPT_V1_0,
    ),
    AgentVariant(
        name="leaderboard_dev-gpt-5.4-nano",
        model="openai:gpt-5.4-nano",
        system_prompt=SYSTEM_PROMPT_V1_0,
    ),
]

# Maximum number of tool calls (page visits) allowed per game
MAX_TOOL_CALLS = 20

# Maximum number of LLM requests (model turns) allowed per game
MAX_LLM_REQUESTS = 30


def build_task(variant) -> Callable[[WikiGolfEvalInput], Awaitable[WikiGolfEvalOutput]]:
    """Build an eval-compatible async callable from a variant."""
    agent = build_agent(variant)

    async def task(inputs: WikiGolfEvalInput) -> WikiGolfEvalOutput:
        deps = WikiGolfDeps(origin=inputs.origin, destination=inputs.destination)
        result = await agent.run(
            variant.user_prompt,
            deps=deps,
            usage_limits=UsageLimits(
                request_limit=MAX_LLM_REQUESTS,
                tool_calls_limit=MAX_TOOL_CALLS,
            ),
        )
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


async def run_all(
    dataset_name: str,
    max_concurrency: int,
) -> dict[str, EvaluationReport[WikiGolfEvalInput, WikiGolfEvalOutput, None]]:
    dataset = DATASETS[dataset_name]
    sha, msg = get_git_info()
    results: dict[
        str, EvaluationReport[WikiGolfEvalInput, WikiGolfEvalOutput, None]
    ] = {}

    for variant in LEADERBOARD_VARIANTS:
        task = build_task(variant)
        metadata = {
            "variant": asdict(variant),
            "git_sha": sha,
            "git_commit_message": msg,
        }
        report = await dataset.evaluate(
            task,
            name=variant.name,
            metadata=metadata,
            max_concurrency=max_concurrency,
        )
        results[variant.name] = report
        print(f"Finished experiment {variant.name!r} ({len(report.cases)} cases).")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Wikipedia Golf evaluations across leaderboard variants"
    )
    parser.add_argument(
        "-d",
        "--dataset",
        default="easy",
        choices=list(DATASETS.keys()),
    )
    parser.add_argument(
        "-c",
        "--max-concurrency",
        type=int,
        default=25,
        help="Max number of concurrent eval cases (default: 25)",
    )
    args = parser.parse_args()

    asyncio.run(run_all(args.dataset, args.max_concurrency))


if __name__ == "__main__":
    main()
