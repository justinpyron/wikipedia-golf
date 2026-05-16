"""Run Wikipedia Golf evaluations across multiple variants for leaderboard workflows.

Usage:
    uv run python -m evals.leaderboard [-d easy] [-c 25] [--save]
"""

import argparse
import asyncio
import subprocess
import uuid
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import logfire
import pandas as pd
from dotenv import load_dotenv
from pydantic_ai import UsageLimits
from pydantic_evals.reporting import EvaluationReport
from pydantic_evals.reporting.analyses import ScalarResult

from agent import WikiGolfDeps, build_agent
from evals.datasets import DATASETS
from evals.types import WikiGolfEvalInput, WikiGolfEvalOutput
from evals.variants_leaderboard import VARIANTS_LEADERBOARD

LEADERBOARD_OUTPUT_DIR = Path(__file__).resolve().parent / "leaderboards"
MAX_TOOL_CALLS = 20
MAX_LLM_REQUESTS = 30


load_dotenv()
logfire.configure(service_name="wiki-golf-evals", environment="dev")


def make_leaderboard_run_id() -> str:
    """Unique sweep id ``leaderboard_YYYYMMDD_<six_hex>`` (local calendar date).

    Uses the first six hex nibbles from a UUID4.

    Example: ``leaderboard_20260309_a3f2e1``.
    """
    day = datetime.now().strftime("%Y%m%d")
    uid = uuid.uuid4().hex[:6]
    return f"leaderboard_{day}_{uid}"


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
    leaderboard_run_id: str,
) -> dict[str, EvaluationReport[WikiGolfEvalInput, WikiGolfEvalOutput, None]]:
    dataset = DATASETS[dataset_name]
    sha, msg = get_git_info()
    results: dict[
        str, EvaluationReport[WikiGolfEvalInput, WikiGolfEvalOutput, None]
    ] = {}

    for variant in VARIANTS_LEADERBOARD.values():
        task = build_task(variant)
        metadata = {
            "leaderboard_run_id": leaderboard_run_id,
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


def make_leaderboard_df(
    reports_by_variant: Mapping[
        str,
        EvaluationReport[WikiGolfEvalInput, WikiGolfEvalOutput, None],
    ],
) -> pd.DataFrame:
    """Build a leaderboard table from ``WikiGolfExperimentMetrics`` outputs.

    Each ``EvaluationReport`` should contain ``ScalarResult`` analyses (as produced by
    :class:`~evals.evaluators.WikiGolfExperimentMetrics`). One row per variant
    (mapping key), one column per analysis title.

    Ignores non-scalar analyses. Missing scalars for a variant become NA.
    """
    variant_names: list[str] = []
    rows: list[dict[str, int | float]] = []
    column_order: list[str] = []
    seen_titles: set[str] = set()

    for variant_name, report in reports_by_variant.items():
        variant_names.append(variant_name)
        row: dict[str, int | float] = {}
        for a in report.analyses:
            if not isinstance(a, ScalarResult):
                continue
            row[a.title] = a.value
            if a.title not in seen_titles:
                column_order.append(a.title)
                seen_titles.add(a.title)
        rows.append(row)

    if not variant_names:
        return pd.DataFrame()

    df = pd.DataFrame.from_records(rows, index=variant_names)
    ordered_cols = [c for c in column_order if c in df.columns]
    df = df.reindex(columns=ordered_cols)
    df.index.name = "variant"
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Wikipedia Golf evaluations across leaderboard variants"
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Write analyses table as JSON under evals/leaderboards/",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        default="easy",
        choices=sorted(list(DATASETS.keys())),
    )
    parser.add_argument(
        "-c",
        "--max-concurrency",
        type=int,
        default=25,
        help="Max number of concurrent eval cases (default: 25)",
    )
    args = parser.parse_args()

    leaderboard_run_id = make_leaderboard_run_id()
    print(f"Launching sweep: leaderboard_run_id: {leaderboard_run_id}\n")

    reports = asyncio.run(
        run_all(args.dataset, args.max_concurrency, leaderboard_run_id)
    )
    df = make_leaderboard_df(reports)
    print(df)
    print()

    if args.save:
        LEADERBOARD_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        json_path = LEADERBOARD_OUTPUT_DIR / f"{leaderboard_run_id}.json"
        json_path.write_text(
            df.to_json(orient="split", indent=4, date_format="iso"),
            encoding="utf-8",
        )
        print(f"\nWrote JSON table to: {json_path}")


if __name__ == "__main__":
    main()
