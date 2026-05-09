"""CLI runner for Wikipedia Golf evaluation experiments.

Usage:
    python -m evals.run -d smoke -v baseline
"""

import argparse
import asyncio

import logfire
from dotenv import load_dotenv

from agent import build_task
from evals.datasets import DATASETS
from variants import VARIANTS

load_dotenv()
logfire.configure(service_name="wiki-golf-evals", environment="dev")


def main() -> None:
    variant_names = [v.name for v in VARIANTS]

    parser = argparse.ArgumentParser(description="Run Wikipedia Golf evaluations")
    parser.add_argument("-d", "--dataset", required=True, choices=list(DATASETS.keys()))
    parser.add_argument("-v", "--variant", required=True, choices=variant_names)
    args = parser.parse_args()

    variant = next(v for v in VARIANTS if v.name == args.variant)
    dataset = DATASETS[args.dataset]
    task = build_task(variant)

    report = asyncio.run(dataset.evaluate(task, name=variant.name))
    report.print()


if __name__ == "__main__":
    main()
