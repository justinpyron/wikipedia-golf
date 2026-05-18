"""Full dataset: combination of easy + moderate + hard cases."""

from pydantic_evals import Case, Dataset

from evals.evaluators import (
    ModelRequestCount,
    NoModelRetries,
    PathLength,
    ReachedDestination,
    RunCostUsd,
    WikiGolfExperimentMetrics,
)
from evals.types import WikiGolfEvalInput, WikiGolfEvalOutput

dataset: Dataset[WikiGolfEvalInput, WikiGolfEvalOutput] = Dataset(
    name="full",
    cases=[],
    evaluators=[
        ReachedDestination(),
        PathLength(),
        ModelRequestCount(),
        RunCostUsd(),
        NoModelRetries(),
    ],
    report_evaluators=[WikiGolfExperimentMetrics()],
)
