"""CLI runner for Wikipedia Golf evaluation experiments.

Usage:
    python -m evals.run -d smoke -v v1_0
"""

import argparse
import asyncio
from dataclasses import asdict

import logfire
from dotenv import load_dotenv

from evals.datasets import DATASETS
from evals.utils import build_task, get_git_info
from evals.variants_experiments import VARIANTS_EXPERIMENTS

load_dotenv()
logfire.configure(service_name="wiki-golf-evals", environment="dev")


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
        choices=sorted(VARIANTS_EXPERIMENTS),
    )
    parser.add_argument(
        "-c",
        "--max-concurrency",
        type=int,
        default=25,
        help="Max number of concurrent eval cases (default: 25)",
    )
    args = parser.parse_args()

    variant = VARIANTS_EXPERIMENTS[args.variant]
    dataset = DATASETS[args.dataset]
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
