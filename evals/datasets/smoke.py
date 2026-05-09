"""Smoke dataset — small set of cases for fast iteration."""

from pydantic_evals import Case, Dataset

from evals.evaluators import AllValidLinksUsed, PathLength, ReachedDestination
from evals.run import WikiGolfEvalInput, WikiGolfEvalOutput

dataset: Dataset[WikiGolfEvalInput, WikiGolfEvalOutput] = Dataset(
    name="smoke",
    cases=[
        Case(
            name="python_to_guido",
            inputs=WikiGolfEvalInput(
                origin="Python_(programming_language)",
                destination="Guido_van_Rossum",
            ),
        ),
        Case(
            name="basketball_to_united_states",
            inputs=WikiGolfEvalInput(
                origin="Basketball",
                destination="United_States",
            ),
        ),
        Case(
            name="eiffel_tower_to_france",
            inputs=WikiGolfEvalInput(
                origin="Eiffel_Tower",
                destination="France",
            ),
        ),
    ],
    evaluators=[ReachedDestination(), PathLength(), AllValidLinksUsed()],
)
