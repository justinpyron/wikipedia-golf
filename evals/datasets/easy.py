"""Easy dataset. Destination reachable in 1 move directly from origin."""

from pydantic_evals import Case, Dataset

from evals.evaluators import (
    NoModelRetries,
    ReachedDestination,
    RunCostUsd,
    StepCount,
    WikiGolfExperimentMetrics,
)
from evals.types import WikiGolfEvalInput, WikiGolfEvalOutput

dataset: Dataset[WikiGolfEvalInput, WikiGolfEvalOutput] = Dataset(
    name="easy",
    cases=[
        Case(
            name="case_1",
            inputs=WikiGolfEvalInput(
                origin="French_Revolution",
                destination="Coup_of_18_Brumaire",
            ),
        ),
        Case(
            name="case_2",
            inputs=WikiGolfEvalInput(
                origin="1941",
                destination="Archie_Clark_(basketball)",
            ),
        ),
        Case(
            name="case_3",
            inputs=WikiGolfEvalInput(
                origin="Pinyin",
                destination="American_Library_Association",
            ),
        ),
        Case(
            name="case_4",
            inputs=WikiGolfEvalInput(
                origin="2026_World_Snooker_Championship",
                destination="Gary_Wilson_(snooker_player)",
            ),
        ),
        Case(
            name="case_5",
            inputs=WikiGolfEvalInput(
                origin="Fireworks",
                destination="Dutch_Safety_Board",
            ),
        ),
    ],
    evaluators=[
        ReachedDestination(),
        StepCount(),
        RunCostUsd(),
        NoModelRetries(),
    ],
    report_evaluators=[WikiGolfExperimentMetrics()],
)
