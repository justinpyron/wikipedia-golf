"""Agent configurations for ``python -m evals.leaderboard``."""

from agent import AgentVariant
from prompts import SYSTEM_PROMPT_V2_0

_LEADERBOARD: list[AgentVariant] = [
    # === OpenAI ===
    AgentVariant(
        name="GPT-5.4-nano (medium thinking)",
        model="openai-responses:gpt-5.4-nano-2026-03-17",
        system_prompt=SYSTEM_PROMPT_V2_0,
        thinking="medium",
    ),
    AgentVariant(
        name="GPT-5.4-mini (medium thinking)",
        model="openai-responses:gpt-5.4-mini-2026-03-17",
        system_prompt=SYSTEM_PROMPT_V2_0,
        thinking="medium",
    ),
    AgentVariant(
        name="GPT-5.4 (medium thinking)",
        model="openai-responses:gpt-5.4-2026-03-05",
        system_prompt=SYSTEM_PROMPT_V2_0,
        thinking="medium",
    ),
    # === Gemini ===
    AgentVariant(
        name="Gemini 3.1 Flash Lite (medium thinking)",
        model="google:gemini-3.1-flash-lite",
        system_prompt=SYSTEM_PROMPT_V2_0,
        thinking="medium",
    ),
    AgentVariant(
        name="Gemini 3.5 Flash (medium thinking)",
        model="google:gemini-3.5-flash",
        system_prompt=SYSTEM_PROMPT_V2_0,
        thinking="medium",
    ),
    AgentVariant(
        name="Gemini 3.1 Pro (medium thinking)",
        model="google:gemini-3.1-pro-preview",
        system_prompt=SYSTEM_PROMPT_V2_0,
        thinking="medium",
    ),
    # === Grok ===
    AgentVariant(
        name="Grok 4.3 (default thinking)",
        model="xai:grok-4.3",
        system_prompt=SYSTEM_PROMPT_V2_0,
    ),
    # === Together ===
    AgentVariant(
        name="GLM 5.1 (default thinking)",
        model="together:zai-org/GLM-5.1",
        system_prompt=SYSTEM_PROMPT_V2_0,
    ),
    AgentVariant(
        name="Kimi K2.6 (default thinking)",
        model="together:moonshotai/Kimi-K2.6",
        system_prompt=SYSTEM_PROMPT_V2_0,
    ),
]

VARIANTS_LEADERBOARD = {v.name: v for v in _LEADERBOARD}
