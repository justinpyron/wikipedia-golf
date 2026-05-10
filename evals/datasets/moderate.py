"""Moderate dataset. Destination reachable in 2 moves from origin."""

from pydantic_evals import Case, Dataset

from evals.evaluators import AllValidLinksUsed, PathLength, ReachedDestination
from evals.types import WikiGolfEvalInput, WikiGolfEvalOutput

dataset: Dataset[WikiGolfEvalInput, WikiGolfEvalOutput] = Dataset(
    name="moderate",
    cases=[
        Case(
            name="case_1",
            inputs=WikiGolfEvalInput(
                origin="French_Revolution",
                destination="Purge",
            ),
        ),
        Case(
            name="case_2",
            inputs=WikiGolfEvalInput(
                origin="1941",
                destination="Quito",
            ),
        ),
        Case(
            name="case_3",
            inputs=WikiGolfEvalInput(
                origin="Pinyin",
                destination="Chinese_Maritime_Customs_Service",
            ),
        ),
        Case(
            name="case_4",
            inputs=WikiGolfEvalInput(
                origin="2026_World_Snooker_Championship",
                destination="Sky_Television_(1984–1990)",
            ),
        ),
        Case(
            name="case_5",
            inputs=WikiGolfEvalInput(
                origin="Fireworks",
                destination="Woodpecker",
            ),
        ),
    ],
    evaluators=[ReachedDestination(), PathLength(), AllValidLinksUsed()],
)
