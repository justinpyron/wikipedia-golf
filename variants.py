"""Agent variant definitions — pure data, no logic."""

from dataclasses import dataclass


@dataclass(frozen=True)
class AgentVariant:
    name: str
    model: str
    system_prompt: str
    user_prompt: str = "Go"
    tool_retries: int = 3
    temperature: float | None = None


BASELINE_SYSTEM_PROMPT = """\
You are an expert Wikipedia Golf player.

Your goal is to navigate from an origin article to a destination article
using the fewest number of links possible. When you find a link matching
the destination key, you have reached the destination. Once you reach the
destination, return the full path you took as the final result.

Think about what conceptual 'hubs' connect the origin to the destination —
countries, people, years, sciences, etc. — and navigate toward those hubs.
Prefer links that move you closer to the destination's topic domain.

Do NOT explore randomly. Be deliberate and efficient."""

VARIANTS: list[AgentVariant] = [
    AgentVariant(
        name="baseline",
        model="openai:gpt-5.4-mini",
        system_prompt=BASELINE_SYSTEM_PROMPT,
    ),
]
