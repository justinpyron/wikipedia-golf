"""Generate a leaderboard by evaluating multiple agent variants.

Usage:
    uv run python -m evals.run_leaderboard [-d easy] [-c 25] [--save]
"""

import argparse
import asyncio
import uuid
from collections.abc import Mapping
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import logfire
import pandas as pd
from dotenv import load_dotenv
from pydantic_evals.reporting import EvaluationReport
from pydantic_evals.reporting.analyses import ScalarResult

from evals.datasets import DATASETS
from evals.utils import WikiGolfEvalInput, WikiGolfEvalOutput, build_task, get_git_info
from evals.variants_leaderboard import VARIANTS_LEADERBOARD

LEADERBOARD_OUTPUT_DIR = Path(__file__).resolve().parent / "leaderboards"
MAX_CONCURRENCY = 10


load_dotenv()
logfire.configure(service_name="wiki-golf-evals", environment="dev")


def make_leaderboard_run_id() -> str:
    """Unique sweep id ``run_YYYYMMDD_HHhmm_<six_hex>`` (local date and time).

    Uses the first six hex nibbles from a UUID4.

    Example: ``run_20260309_14h30_a3f2e1``.
    """
    now = datetime.now()
    day = now.strftime("%Y%m%d")
    time_part = now.strftime("%Hh%M")
    uid = uuid.uuid4().hex[:6]
    return f"run_{day}_{time_part}_{uid}"


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
        try:
            report = await dataset.evaluate(
                task,
                name=variant.name,
                metadata=metadata,
                max_concurrency=max_concurrency,
            )
        except Exception as e:
            print(f"FAILED experiment {variant.name!r}: {e}")
            continue
        results[variant.name] = report
        print(
            f"Finished experiment {variant.name!r} "
            f"({len(report.cases)} cases, {len(report.failures)} failures)."
        )

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
        default="leaderboard",
        choices=sorted(list(DATASETS.keys())),
    )
    parser.add_argument(
        "-c",
        "--max-concurrency",
        type=int,
        default=MAX_CONCURRENCY,
        help=f"Max number of concurrent eval cases (default: {MAX_CONCURRENCY})",
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
