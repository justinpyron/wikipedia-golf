"""Agent configurations for ``python -m evals.run``."""

from agent import AgentVariant
from prompts import SYSTEM_PROMPT_V2_0

_EXPERIMENTS: list[AgentVariant] = [
    AgentVariant(
        name="openai-responses:gpt-5.4-mini-2026-03-17",
        model="openai-responses:gpt-5.4-mini-2026-03-17",
        system_prompt=SYSTEM_PROMPT_V2_0,
    ),
    AgentVariant(
        name="openai-responses:gpt-5.4-nano-2026-03-17",
        model="openai-responses:gpt-5.4-nano-2026-03-17",
        system_prompt=SYSTEM_PROMPT_V2_0,
    ),
    AgentVariant(
        name="google:gemini-3.1-flash-lite",
        model="google:gemini-3.1-flash-lite",
        system_prompt=SYSTEM_PROMPT_V2_0,
    ),
    AgentVariant(
        name="xai:grok-4.3",
        model="xai:grok-4.3",
        system_prompt=SYSTEM_PROMPT_V2_0,
    ),
    AgentVariant(
        name="together:moonshotai/Kimi-K2.6",
        model="together:moonshotai/Kimi-K2.6",
        system_prompt=SYSTEM_PROMPT_V2_0,
    ),
    AgentVariant(
        name="together:zai-org/GLM-5.1",
        model="together:zai-org/GLM-5.1",
        system_prompt=SYSTEM_PROMPT_V2_0,
    ),
]

VARIANTS_EXPERIMENTS = {v.name: v for v in _EXPERIMENTS}
