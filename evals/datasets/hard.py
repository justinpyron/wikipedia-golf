"""Hard dataset. Destination reachable in 3 or more moves from origin."""

from pydantic_evals import Case, Dataset

from evals.evaluators import AllValidLinksUsed, PathLength, ReachedDestination
from evals.types import WikiGolfEvalInput, WikiGolfEvalOutput

dataset: Dataset[WikiGolfEvalInput, WikiGolfEvalOutput] = Dataset(
    name="hard",
    cases=[
        Case(
            name="case_1",
            inputs=WikiGolfEvalInput(
                origin="French_Revolution",
                destination="Besix",
            ),
        ),
        Case(
            name="case_2",
            inputs=WikiGolfEvalInput(
                origin="1941",
                destination="Stockholm_Concert_Hall",
            ),
        ),
        Case(
            name="case_3",
            inputs=WikiGolfEvalInput(
                origin="Pinyin",
                destination="Zero-width_non-joiner",
            ),
        ),
        Case(
            name="case_4",
            inputs=WikiGolfEvalInput(
                origin="2026_World_Snooker_Championship",
                destination="Anti-lock_braking_system",
            ),
        ),
        Case(
            name="case_5",
            inputs=WikiGolfEvalInput(
                origin="Fireworks",
                destination="Genuflection",
            ),
        ),
    ],
    evaluators=[ReachedDestination(), PathLength(), AllValidLinksUsed()],
)
