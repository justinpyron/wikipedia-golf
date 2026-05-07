import asyncio
import sys
from dataclasses import dataclass, field
from typing import Annotated

from pydantic import BaseModel, model_validator
from pydantic_ai import Agent, ModelRetry, RunContext, UsageLimits
from pydantic_ai.models.openai import OpenAIChatModel

from wiki import ArticleLinks, fetch_article_links

SYSTEM_PROMPT = """You are an expert Wikipedia Golf player.
Your goal is to navigate from an origin article to a destination article
using the fewest number of links possible.
Think about what conceptual 'hubs' connect the origin to the destination —
countries, people, years, sciences, etc. — and navigate toward those hubs.
Prefer links that move you closer to the destination's topic domain.
Do NOT explore randomly. Be deliberate and efficient."""

REQUEST_LIMIT = 20


@dataclass
class WikiGolfDeps:
    origin: str
    destination: str
    path: list[str] = field(default_factory=list)


class WikiGolfResult(BaseModel):
    path: list[str]
    steps: int

    @model_validator(mode="after")
    def validate_result(self) -> "WikiGolfResult":
        if not self.path:
            raise ValueError("Path cannot be empty")
        if self.steps != len(self.path) - 1:
            raise ValueError(
                f"Steps ({self.steps}) must be len(path) - 1 ({len(self.path) - 1})"
            )
        return self


agent = Agent(
    "openai:gpt-4o",
    deps_type=WikiGolfDeps,
    output_type=WikiGolfResult,
    system_prompt=SYSTEM_PROMPT,
)


@agent.tool
async def get_links(ctx: RunContext[WikiGolfDeps], key: str) -> str:
    """Fetch all navigable links from a Wikipedia article.

    Call this to see which pages you can navigate to from the given article.
    The 'key' should be the identifier for the article.
    """
    # Path tracking: append the key we are currently exploring
    if not ctx.deps.path or ctx.deps.path[-1] != key:
        ctx.deps.path.append(key)

    result = fetch_article_links(key)
    if result is None:
        raise ModelRetry(f"Could not fetch links for '{key}'. Try a different key.")

    # Check if destination is in the links to help the agent notice victory
    link_keys = {link.key for link in result.links}
    if ctx.deps.destination in link_keys:
        return (
            f"VICTORY CONDITION MET: The destination '{ctx.deps.destination}' is available in the links below!\n\n"
            + result.to_markdown_table(omit=["title"])
        )

    return result.to_markdown_table(omit=["title"])


@agent.system_prompt
def game_state_prompt(ctx: RunContext[WikiGolfDeps]) -> str:
    deps = ctx.deps
    path_str = (
        " -> ".join(deps.path)
        if deps.path
        else "(none — start by fetching links from the origin)"
    )
    return f"""You are playing Wikipedia Golf.

Origin: {deps.origin}
Destination: {deps.destination}
Path so far: {path_str}

Your goal: navigate from origin to destination by following links, minimizing total hops.
When you find a link matching the destination key, you have reached the destination.
Once you reach the destination, return the full path you took as the final result."""


async def play_wikipedia_golf(
    origin: str, destination: str, model: str
) -> WikiGolfResult:
    deps = WikiGolfDeps(origin=origin, destination=destination)
    # The origin is the first step in the path
    result = await agent.run(
        f"Play Wikipedia Golf. Start from the origin article: {origin}",
        deps=deps,
        model=model,
        usage_limits=UsageLimits(request_limit=REQUEST_LIMIT),
    )
    return result.output


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python agent.py <origin_key> <destination_key> [model]")
        sys.exit(1)

    origin_key = sys.argv[1]
    destination_key = sys.argv[2]
    model_name = sys.argv[3] if len(sys.argv) > 3 else "openai:gpt-4o"

    async def main():
        print(f"Starting Wikipedia Golf: {origin_key} -> {destination_key}")
        try:
            result = await play_wikipedia_golf(
                origin_key, destination_key, model=model_name
            )
            print("\nSuccess!")
            print(f"Path: {' -> '.join(result.path)}")
            print(f"Total steps: {result.steps}")
        except Exception as e:
            print(f"\nAn error occurred: {e}")

    asyncio.run(main())
