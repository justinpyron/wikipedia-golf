"""Agent configurations for ``python -m evals.leaderboard``."""

from agent import AgentVariant
from prompts import SYSTEM_PROMPT_V1_0

_LEADERBOARD: list[AgentVariant] = [
    AgentVariant(
        name="gpt-5.4-mini",
        model="openai-responses:gpt-5.4-mini-2026-03-17",
        system_prompt=SYSTEM_PROMPT_V1_0,
    ),
    AgentVariant(
        name="gpt-5.4-nano",
        model="openai-responses:gpt-5.4-nano-2026-03-17",
        system_prompt=SYSTEM_PROMPT_V1_0,
    ),
]

VARIANTS_LEADERBOARD = {v.name: v for v in _LEADERBOARD}
