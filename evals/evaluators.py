"""Custom evaluators for Wikipedia Golf experiments."""

from dataclasses import dataclass

import numpy as np
from pydantic_ai.messages import RetryPromptPart
from pydantic_evals.evaluators import (
    Evaluator,
    EvaluatorContext,
    ReportEvaluator,
    ReportEvaluatorContext,
)
from pydantic_evals.reporting.analyses import ScalarResult

from agent import estimate_cost
from evals.types import WikiGolfEvalInput, WikiGolfEvalOutput


class ReachedDestination(Evaluator):
    """Check if the agent's path ends at the destination."""

    def evaluate(
        self, ctx: EvaluatorContext[WikiGolfEvalInput, WikiGolfEvalOutput]
    ) -> bool:
        if not ctx.output.path:
            return False
        return ctx.output.path[-1] == ctx.inputs.destination


class StepCount(Evaluator):
    """Return the number of steps (hops) between origin and destination."""

    def evaluate(
        self, ctx: EvaluatorContext[WikiGolfEvalInput, WikiGolfEvalOutput]
    ) -> int:
        return len(ctx.output.path) - 1


class AllValidLinksUsed(Evaluator):
    """Check if the agent only attempted to use valid links."""

    def evaluate(
        self, ctx: EvaluatorContext[WikiGolfEvalInput, WikiGolfEvalOutput]
    ) -> bool:
        # Check messages for any RetryPromptPart, which indicates a ModelRetry was triggered
        for message in ctx.output.messages:
            if hasattr(message, "parts"):
                for part in message.parts:
                    if isinstance(part, RetryPromptPart):
                        return False
        return True


# Names match `BaseEvaluator.get_serialization_name()` on ReportCase.
ASSERTION_REACHED_DESTINATION = ReachedDestination.__name__
ASSERTION_ALL_VALID_LINKS = AllValidLinksUsed.__name__
SCORE_STEP_COUNT = StepCount.__name__


@dataclass
class WikiGolfExperimentMetrics(
    ReportEvaluator[WikiGolfEvalInput, WikiGolfEvalOutput, None]
):
    """Aggregate duration, cost, success rate, and step counts across all dataset cases."""

    def evaluate(
        self, ctx: ReportEvaluatorContext[WikiGolfEvalInput, WikiGolfEvalOutput, None]
    ) -> list[ScalarResult]:
        report = ctx.report
        cases = report.cases
        n_failures = len(report.failures)
        n_attempts = len(cases) + n_failures

        durations = [c.task_duration for c in cases]
        costs = [float(estimate_cost(c.output.messages)) for c in cases]
        steps: list[float] = []
        for c in cases:
            sc = c.scores.get(SCORE_STEP_COUNT)
            if sc is not None:
                steps.append(float(sc.value))

        reached_hits = 0
        for c in cases:
            a = c.assertions.get(ASSERTION_REACHED_DESTINATION)
            if a is not None and a.value:
                reached_hits += 1
        reached_pct = (100.0 * reached_hits / n_attempts) if n_attempts else 0.0

        valid_hits = 0
        for c in cases:
            a = c.assertions.get(ASSERTION_ALL_VALID_LINKS)
            if a is not None and a.value:
                valid_hits += 1
        valid_pct = (100.0 * valid_hits / n_attempts) if n_attempts else 0.0

        return [
            ScalarResult(
                title="Median task duration",
                value=float(np.median(durations)) if durations else 0.0,
                unit="s",
                description="Median task duration over successful cases (seconds).",
            ),
            ScalarResult(
                title="Median cost",
                value=float(np.median(costs)) if costs else 0.0,
                unit="USD",
                description="Median estimated cost per successful case from model response usage.",
            ),
            ScalarResult(
                title="Reached destination rate",
                value=round(reached_pct, 2),
                unit="%",
                description="Share of dataset cases that reached the destination; task failures count as not reached.",
            ),
            ScalarResult(
                title="Median step count",
                value=float(np.median(steps)) if steps else 0.0,
                description="Median hop count (StepCount) over successful cases.",
            ),
            ScalarResult(
                title="All valid links rate",
                value=round(valid_pct, 2),
                unit="%",
                description="Share of cases with no invalid link attempts; task failures count as invalid.",
            ),
        ]
