"""Agent configurations for ``python -m evals.leaderboard``."""

from agent import AgentVariant
from prompts import SYSTEM_PROMPT_V1_0

_LEADERBOARD: list[AgentVariant] = [
    AgentVariant(
        name="leaderboard_dev-gpt-5.4-mini",
        model="openai:gpt-5.4-mini",
        system_prompt=SYSTEM_PROMPT_V1_0,
    ),
    AgentVariant(
        name="leaderboard_dev-gpt-5.4-nano",
        model="openai:gpt-5.4-nano",
        system_prompt=SYSTEM_PROMPT_V1_0,
    ),
]

VARIANTS_LEADERBOARD = {v.name: v for v in _LEADERBOARD}
