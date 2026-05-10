"""Custom evaluators for Wikipedia Golf experiments."""

from pydantic_ai.messages import RetryPromptPart
from pydantic_evals.evaluators import Evaluator, EvaluatorContext

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
