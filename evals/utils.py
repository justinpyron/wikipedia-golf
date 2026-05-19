"""Shared types and helpers for Wikipedia Golf evaluations."""

import subprocess
from collections.abc import Awaitable, Callable

from pydantic import BaseModel
from pydantic_ai import UsageLimits
from pydantic_ai.messages import ModelMessage

from agent import AgentVariant, WikiGolfDeps, build_agent

MAX_TOOL_CALLS = 20  # page visits allowed per game
MAX_LLM_REQUESTS = 30  # model turns allowed per game


class WikiGolfEvalInput(BaseModel):
    origin: str
    destination: str


class WikiGolfEvalOutput(BaseModel):
    path: list[str]
    messages: list[ModelMessage]


def build_task(
    variant: AgentVariant,
) -> Callable[[WikiGolfEvalInput], Awaitable[WikiGolfEvalOutput]]:
    """Build an eval-compatible async callable from a variant."""
    agent = build_agent(variant)

    async def task(inputs: WikiGolfEvalInput) -> WikiGolfEvalOutput:
        deps = WikiGolfDeps(origin=inputs.origin, destination=inputs.destination)
        result = await agent.run(
            variant.user_prompt,
            deps=deps,
            usage_limits=UsageLimits(
                request_limit=MAX_LLM_REQUESTS,
                tool_calls_limit=MAX_TOOL_CALLS,
            ),
        )
        return WikiGolfEvalOutput(
            path=deps.path,
            messages=result.all_messages(),
        )

    return task


def get_git_info() -> tuple[str, str]:
    """Get the current git SHA and commit subject atomically."""
    try:
        output = subprocess.check_output(
            ["git", "log", "-1", "--pretty=format:%H%n%s"], text=True
        ).strip()
        sha, msg = output.split("\n", 1)
        return sha, msg
    except Exception:
        return "unknown", "unknown"
