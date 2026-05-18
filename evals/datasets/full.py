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
from evals.utils import WikiGolfEvalInput, WikiGolfEvalOutput

dataset: Dataset[WikiGolfEvalInput, WikiGolfEvalOutput] = Dataset(
    name="full",
    cases=[],
    evaluators=[
        ReachedDestination(),
        ModelRequestCount(),
        PathLength(),
        NoModelRetries(),
        RunCostUsd(),
    ],
    report_evaluators=[WikiGolfExperimentMetrics()],
)
