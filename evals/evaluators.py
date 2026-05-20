"""Custom evaluators for Wikipedia Golf experiments."""

from dataclasses import dataclass

import numpy as np
from pydantic_ai.messages import ModelResponse, RetryPromptPart
from pydantic_evals.evaluators import (
    Evaluator,
    EvaluatorContext,
    ReportEvaluator,
    ReportEvaluatorContext,
)
from pydantic_evals.reporting.analyses import ScalarResult

from agent import estimate_run_cost_usd
from evals.utils import WikiGolfEvalInput, WikiGolfEvalOutput


class ReachedDestination(Evaluator):
    """Check if the agent's path ends at the destination."""

    def evaluate(
        self, ctx: EvaluatorContext[WikiGolfEvalInput, WikiGolfEvalOutput]
    ) -> bool:
        if not ctx.output.path:
            return False
        return ctx.output.path[-1] == ctx.inputs.destination


class ModelRequestCount(Evaluator):
    """Return how many LLM requests occurred in the run."""

    def evaluate(
        self, ctx: EvaluatorContext[WikiGolfEvalInput, WikiGolfEvalOutput]
    ) -> int:
        return sum(1 for m in ctx.output.messages if isinstance(m, ModelResponse))


class PathLength(Evaluator):
    """Return the number of edges in the path from origin to destination.

    Counts ``len(path) - 1``: each edge is one legal ``get_links`` navigation
    from one article to the next.
    """

    def evaluate(
        self, ctx: EvaluatorContext[WikiGolfEvalInput, WikiGolfEvalOutput]
    ) -> int:
        return len(ctx.output.path) - 1


class NoModelRetries(Evaluator):
    """Return True iff no RetryPromptPart appears (no ModelRetry in the transcript)."""

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


class RunCostUsd(Evaluator):
    """Return estimated total USD cost for the run."""

    def evaluate(
        self, ctx: EvaluatorContext[WikiGolfEvalInput, WikiGolfEvalOutput]
    ) -> float:
        return estimate_run_cost_usd(ctx.output.messages)


# Names of evaluator results on case objects (importance / presentation order)
ASSERTION_REACHED_DESTINATION = ReachedDestination.__name__
SCORE_MODEL_REQUEST_COUNT = ModelRequestCount.__name__
SCORE_PATH_LENGTH = PathLength.__name__
ASSERTION_NO_MODEL_RETRIES = NoModelRetries.__name__
SCORE_RUN_COST_USD = RunCostUsd.__name__


@dataclass
class WikiGolfExperimentMetrics(
    ReportEvaluator[WikiGolfEvalInput, WikiGolfEvalOutput, None]
):
    """Aggregate duration, cost, path length, model request counts, and success rates."""

    def evaluate(
        self, ctx: ReportEvaluatorContext[WikiGolfEvalInput, WikiGolfEvalOutput, None]
    ) -> list[ScalarResult]:
        report = ctx.report
        cases = report.cases
        n_failures = len(report.failures)
        n_attempts = len(cases) + n_failures

        durations = [c.task_duration for c in cases]
        costs = [estimate_run_cost_usd(c.output.messages) for c in cases]
        model_request_counts: list[int] = []
        for c in cases:
            sc = c.scores.get(SCORE_MODEL_REQUEST_COUNT)
            if sc is not None:
                model_request_counts.append(int(sc.value))

        path_lengths: list[int] = []
        for c in cases:
            sc = c.scores.get(SCORE_PATH_LENGTH)
            if sc is not None:
                path_lengths.append(int(sc.value))

        reached_hits = 0
        for c in cases:
            a = c.assertions.get(ASSERTION_REACHED_DESTINATION)
            if a is not None and a.value:
                reached_hits += 1
        reached_pct = (100.0 * reached_hits / n_attempts) if n_attempts else 0.0

        no_retry_hits = 0
        for c in cases:
            a = c.assertions.get(ASSERTION_NO_MODEL_RETRIES)
            if a is not None and a.value:
                no_retry_hits += 1
        no_retry_pct = (100.0 * no_retry_hits / n_attempts) if n_attempts else 0.0

        return [
            ScalarResult(
                title="Reached destination rate",
                value=round(reached_pct, 2),
                unit="%",
                description="Share of dataset cases that reached the destination; task failures count as not reached.",
            ),
            ScalarResult(
                title="Model request count (mean)",
                value=float(np.mean(model_request_counts))
                if model_request_counts
                else 0.0,
                description="Mean number of LLM requests per successful run.",
            ),
            ScalarResult(
                title="Model request count (median)",
                value=float(np.median(model_request_counts))
                if model_request_counts
                else 0.0,
                description="Median number of LLM requests per successful run.",
            ),
            ScalarResult(
                title="Path length (mean)",
                value=float(np.mean(path_lengths)) if path_lengths else 0.0,
                description="Mean hop count (edges on path from origin to destination); not model turns.",
            ),
            ScalarResult(
                title="Path length (median)",
                value=float(np.median(path_lengths)) if path_lengths else 0.0,
                description="Median hop count (edges on path from origin to destination); not model turns.",
            ),
            ScalarResult(
                title="No model retries rate",
                value=round(no_retry_pct, 2),
                unit="%",
                description="Share of cases with no RetryPromptPart in messages (tool/output ModelRetry); failures count against this.",
            ),
            ScalarResult(
                title="Cost (median)",
                value=float(np.median(costs)) if costs else 0.0,
                unit="USD",
                description="Median estimated cost per successful case from model response usage.",
            ),
            ScalarResult(
                title="Cost (total)",
                value=float(sum(costs)) if costs else 0.0,
                unit="USD",
                description="Sum of estimated costs over successful cases from model response usage.",
            ),
            ScalarResult(
                title="Task duration (median)",
                value=float(np.median(durations)) if durations else 0.0,
                unit="s",
                description="Median task duration over successful cases (seconds).",
            ),
            ScalarResult(
                title="Task duration (total)",
                value=float(sum(durations)) if durations else 0.0,
                unit="s",
                description="Sum of task durations over successful cases (seconds).",
            ),
        ]
