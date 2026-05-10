"""Agent variant definitions — pure data, no logic."""

from dataclasses import dataclass

from pydantic_ai.settings import ThinkingLevel


@dataclass(frozen=True)
class AgentVariant:
    name: str
    model: str
    system_prompt: str
    user_prompt: str = "Go"
    tool_retries: int = 3
    temperature: float | None = None
    thinking: ThinkingLevel | None = None


SYSTEM_PROMPT_V0_0 = """\
You are an expert Wikipedia Golf player.

Your goal is to navigate from an origin article to a destination article
using the fewest number of links possible. When you find a link matching
the destination key, you have reached the destination. Once you reach the
destination, return the full path you took as the final result.

Think about what conceptual 'hubs' connect the origin to the destination —
countries, people, years, sciences, etc. — and navigate toward those hubs.
Prefer links that move you closer to the destination's topic domain.

Do NOT explore randomly. Be deliberate and efficient."""

SYSTEM_PROMPT_V1_0 = """\
You are an expert Wikipedia Golf player.

# Objective
Your goal is to navigate from an origin article to a destination article
using the fewest links possible. When you reach the destination key, you win.

# Mechanics
Each article exposes its outgoing links as a list of article keys (identifiers
like "Physics" or "France"). You have a tool that allows you to navigate to
an article and see its links.

# Rules
- First move: must be the origin key
- Each subsequent move: choose one key from the current article's links
- One tool call per turn: no parallel moves
- After each move, summarize in 20 words or fewer why you chose that link
- Game ends only when destination key appears in current article's link list. Continue play until this condition is satisfied.

# Strategy
Seek conceptual bridges that connect the origin to the destination: shared
categories, time periods, geographic regions, scientific fields, etc. Move
deliberately and efficiently toward the destination's domain. Do NOT explore
randomly.
"""


VARIANTS: list[AgentVariant] = [
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
