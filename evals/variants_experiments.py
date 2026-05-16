"""Agent configurations for ``python -m evals.run``."""

from agent import AgentVariant
from prompts import SYSTEM_PROMPT_V0_0, SYSTEM_PROMPT_V1_0

_EXPERIMENTS: list[AgentVariant] = [
    AgentVariant(
        name="v0_0",
        model="openai:gpt-5.4-mini",
        system_prompt=SYSTEM_PROMPT_V0_0,
    ),
    AgentVariant(
        name="v1_0",
        model="openai:gpt-5.4-mini",
        system_prompt=SYSTEM_PROMPT_V1_0,
    ),
    AgentVariant(
        name="v1_0_anthropic",
        model="anthropic:claude-haiku-4-5-20251001",
        system_prompt=SYSTEM_PROMPT_V1_0,
    ),
]

VARIANTS_EXPERIMENTS = {v.name: v for v in _EXPERIMENTS}
