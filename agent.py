import asyncio
import sys
from dataclasses import dataclass, field

from dotenv import load_dotenv
from pydantic import BaseModel
from pydantic_ai import Agent, ModelRetry, RunContext, UsageLimits
from pydantic_ai.models.openai import OpenAIChatModel

from wiki import fetch_article_links

load_dotenv()

SYSTEM_PROMPT = """You are an expert Wikipedia Golf player.
Your goal is to navigate from an origin article to a destination article
using the fewest number of links possible.
Think about what conceptual 'hubs' connect the origin to the destination —
countries, people, years, sciences, etc. — and navigate toward those hubs.
Prefer links that move you closer to the destination's topic domain.
Do NOT explore randomly. Be deliberate and efficient."""
# TODO: Update system prompt with instructions about wikipedia article keys.
# TODO: Update system prompt with guidance/strategy on how to use Wikipedia.
# E.g.: there must be an exact match; being close is not sufficient.


REQUEST_LIMIT = 10
DEFAULT_MODEL = "openai:gpt-5.4-mini"


@dataclass
class WikiGolfDeps:
    origin: str
    destination: str
    path: list[str] = field(default_factory=list)


class WikiGolfOutput(BaseModel):
    path: list[str]


agent = Agent(
    DEFAULT_MODEL,
    deps_type=WikiGolfDeps,
    output_type=WikiGolfOutput,
    system_prompt=SYSTEM_PROMPT,
)


@agent.tool
async def get_links(ctx: RunContext[WikiGolfDeps], key: str) -> str:
    """Fetch all navigable links from a Wikipedia article.

    Args:
        key: The Wikipedia article key (identifier) to fetch links from.
    """
    result = fetch_article_links(key)
    if result is None:
        raise ModelRetry(f"Could not fetch links for '{key}'. Try a different key.")
    ctx.deps.path.append(key)

    # Check if destination is in the links to help the agent notice victory
    out = result.to_markdown_table(omit=["title"])
    dst = ctx.deps.destination
    if dst in {link.key for link in result.links}:
        out += f"\n\nVICTORY CONDITION MET: The destination '{dst}' is available in the links above!"
    else:
        out += f"\n\nThe destination '{dst}' is not in the links above. Keep searching!"

    return out


@agent.system_prompt
def game_state_prompt(ctx: RunContext[WikiGolfDeps]) -> str:
    return f"""You are playing Wikipedia Golf.

Origin: {ctx.deps.origin}
Destination: {ctx.deps.destination}

Your goal: navigate from origin to destination by following links, minimizing total hops.
When you find a link matching the destination key, you have reached the destination.
Once you reach the destination, return the full path you took as the final result."""


async def play_wikipedia_golf(
    origin: str, destination: str, model: str
) -> WikiGolfOutput:
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
    model_name = sys.argv[3] if len(sys.argv) > 3 else DEFAULT_MODEL

    async def main():
        print(f"Starting Wikipedia Golf: {origin_key} -> {destination_key}")
        try:
            result = await play_wikipedia_golf(
                origin_key, destination_key, model=model_name
            )
            print("\nSuccess!")
            print(f"Path: {' -> '.join(result.path)}")
        except Exception as e:
            print(f"\nAn error occurred: {e}")

    asyncio.run(main())
