"""Custom evaluators for Wikipedia Golf experiments."""

from dataclasses import dataclass

from pydantic_ai import ModelRetry
from pydantic_evals.evaluators import Evaluator, EvaluatorContext


class ReachedDestination(Evaluator):
    """Check if the agent's path ends at the destination."""

    def evaluate(self, ctx: EvaluatorContext) -> bool:
        if not ctx.output.path:
            return False
        return ctx.output.path[-1] == ctx.inputs.destination


class PathLength(Evaluator):
    """Return the number of hops in the agent's path."""

    def evaluate(self, ctx: EvaluatorContext) -> int:
        return len(ctx.output.path)


class AllValidLinksUsed(Evaluator):
    """Check if the agent only attempted to use valid links."""

    def evaluate(self, ctx: EvaluatorContext) -> bool:
        return not any(isinstance(m, ModelRetry) for m in ctx.trace.messages)
