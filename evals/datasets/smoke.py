"""Smoke dataset — small set of cases for fast iteration."""

from pydantic_evals import Case, Dataset

from evals.evaluators import AllValidLinksUsed, PathLength, ReachedDestination
from evals.types import WikiGolfEvalInput, WikiGolfEvalOutput

dataset: Dataset[WikiGolfEvalInput, WikiGolfEvalOutput] = Dataset(
    name="smoke",
    cases=[
        Case(
            name="case_1",
            inputs=WikiGolfEvalInput(
                origin="Python_(programming_language)",
                destination="Guido_van_Rossum",
            ),
        ),
        Case(
            name="case_2",
            inputs=WikiGolfEvalInput(
                origin="Basketball",
                destination="United_States",
            ),
        ),
        Case(
            name="case_3",
            inputs=WikiGolfEvalInput(
                origin="Eiffel_Tower",
                destination="France",
            ),
        ),
    ],
    evaluators=[ReachedDestination(), PathLength(), AllValidLinksUsed()],
)
