"""Factory for building Wikipedia Golf agents from variant configurations."""

from dataclasses import dataclass, field

from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.settings import ModelSettings

from variants import AgentVariant
from wiki import WikiAPIError, WikiArticleNotFoundError, fetch_article_links


@dataclass
class WikiGolfDeps:
    origin: str
    destination: str
    path: list[str] = field(default_factory=list)
    candidate_keys: set[str] = field(default_factory=set)


def build_agent(variant: AgentVariant) -> Agent[WikiGolfDeps, str]:
    """Construct a fully-configured Wikipedia Golf agent from a variant."""
    model_settings: ModelSettings | None = None
    if variant.temperature is not None or variant.thinking is not None:
        model_settings = ModelSettings(
            temperature=variant.temperature,
            thinking=variant.thinking,
        )

    agent = Agent(
        variant.model,
        deps_type=WikiGolfDeps,
        system_prompt=variant.system_prompt,
        model_settings=model_settings,
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
        """Navigate to a Wikipedia article and return its outgoing links.

        Validates that the move is legal (origin on first turn, reachable link
        thereafter). If key equals destination, declares victory immediately
        without fetching. Otherwise fetches and returns keys of articles linked to
        from the requested article.

        Args:
            key: Key of article to navigate to. Will be validated against game rules.
        """
        # PHASE 1: Validate the move
        is_first_move = len(ctx.deps.path) == 0

        if is_first_move:
            if key != ctx.deps.origin:
                raise ModelRetry(
                    f"INVALID FIRST MOVE: Must start at origin '{ctx.deps.origin}'. Got: '{key}'"
                )
        else:
            if key not in ctx.deps.candidate_keys:
                raise ModelRetry(
                    f"ILLEGAL MOVE: '{key}' is not available from the current page. "
                    f"Valid keys: {sorted(ctx.deps.candidate_keys)}"
                )

        # PHASE 2: Victory (destination reached - no fetch needed)
        if key == ctx.deps.destination:
            ctx.deps.path.append(key)
            return f"VICTORY: Reached destination '{key}'.\nPATH: {' -> '.join(ctx.deps.path)}"

        # PHASE 3: Fetch and advance
        try:
            result = fetch_article_links(key)
        except WikiArticleNotFoundError:
            raise ModelRetry(
                f"PAGE NOT FOUND: '{key}' does not exist. "
                "Choose a different key from the available links."
            )
        except WikiAPIError as e:
            raise ModelRetry(
                f"NETWORK ERROR fetching '{key}': {e}. "
                "Try again or choose another link from the available links."
            )

        ctx.deps.path.append(key)
        ctx.deps.candidate_keys = {link.key for link in result.links}

        message = result.to_markdown_table(omit=["text", "title"])
        dst = ctx.deps.destination
        if dst in ctx.deps.candidate_keys:
            message += f"\n\n🎯 DESTINATION '{dst}' IS AVAILABLE! Call get_links('{dst}') to win."

        return message

    return agent
