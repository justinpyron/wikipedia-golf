"""Factory for building Wikipedia Golf agents from variant configurations."""

from dataclasses import dataclass, field

from pydantic import BaseModel
from pydantic_ai import Agent, ModelRetry, RunContext

from variants import AgentVariant
from wiki import fetch_article_links


@dataclass
class WikiGolfDeps:
    origin: str
    destination: str
    path: list[str] = field(default_factory=list)


class WikiGolfOutput(BaseModel):
    path: list[str]


class WikiGolfInput(BaseModel):
    origin: str
    destination: str


def build_agent(variant: AgentVariant) -> Agent[WikiGolfDeps, WikiGolfOutput]:
    """Construct a fully-configured Wikipedia Golf agent from a variant."""
    agent = Agent(
        variant.model,
        deps_type=WikiGolfDeps,
        output_type=WikiGolfOutput,
        system_prompt=variant.system_prompt,
        instrument=True,
    )

    @agent.system_prompt
    def game_specs_prompt(ctx: RunContext[WikiGolfDeps]) -> str:
        return (
            f"You are playing Wikipedia Golf with the following constraints:\n"
            f"Origin: {ctx.deps.origin}\n"
            f"Destination: {ctx.deps.destination}\n"
        )

    @agent.tool(retries=variant.tool_retries)
    async def get_links(ctx: RunContext[WikiGolfDeps], key: str) -> str:
        """Fetch all navigable links from a Wikipedia article.

        Args:
            key: The Wikipedia article key (identifier) to fetch links from.
        """
        result = fetch_article_links(key)
        if result is None:
            raise ModelRetry(
                f"Could not fetch links for '{key}'. "
                "Try a different key. "
                "The key must be the origin key or exist in the output of a previous tool call."
            )
        ctx.deps.path.append(key)

        out = result.to_markdown_table(omit=["text", "title"])
        dst = ctx.deps.destination
        if dst in {link.key for link in result.links}:
            out += f"\n\nVICTORY CONDITION MET: The destination '{dst}' is available in the links above!"
        else:
            out += f"\n\nThe destination '{dst}' is not in the links above. Keep searching!"

        return out

    return agent
