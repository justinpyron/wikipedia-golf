"""Smoke dataset — small set of cases for fast iteration."""

from pydantic_evals import Case, Dataset

from agent import WikiGolfInput, WikiGolfOutput
from evals.evaluators import AllValidLinksUsed, PathLength, ReachedDestination

dataset: Dataset[WikiGolfInput, WikiGolfOutput] = Dataset(
    name="smoke",
    cases=[
        Case(
            name="python_to_guido",
            inputs=WikiGolfInput(
                origin="Python_(programming_language)",
                destination="Guido_van_Rossum",
            ),
        ),
        Case(
            name="basketball_to_united_states",
            inputs=WikiGolfInput(
                origin="Basketball",
                destination="United_States",
            ),
        ),
        Case(
            name="eiffel_tower_to_france",
            inputs=WikiGolfInput(
                origin="Eiffel_Tower",
                destination="France",
            ),
        ),
    ],
    evaluators=[ReachedDestination(), PathLength(), AllValidLinksUsed()],
)
