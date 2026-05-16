"""Factory for building Wikipedia Golf agents from variant configurations."""

from dataclasses import dataclass, field
from decimal import Decimal

from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.messages import ModelRequest, ModelResponse
from pydantic_ai.settings import ModelSettings, ThinkingLevel

from wiki import WikiAPIError, WikiArticleNotFoundError, fetch_article_links


@dataclass(frozen=True)
class AgentVariant:
    name: str
    model: str
    system_prompt: str
    user_prompt: str = "Go"
    tool_retries: int = 3
    temperature: float | None = None
    thinking: ThinkingLevel | None = None


@dataclass
class WikiGolfDeps:
    origin: str
    destination: str
    path: list[str] = field(default_factory=list)
    candidate_keys: set[str] = field(default_factory=set)


@dataclass
class AgentResult:
    """Result of a Wikipedia Golf agent run with usage statistics."""

    path: list[str]
    duration_seconds: float
    total_tokens: int
    estimated_cost_usd: float


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
            f"# Parameters\n"
            f"You are playing Wikipedia Golf with the following parameters:\n"
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
                    f"Valid keys are listed in the output of the previous tool call."
                )
            if len(ctx.deps.path) > 0 and key == ctx.deps.path[-1]:
                raise ModelRetry(
                    f"REDUNDANT MOVE: Cannot navigate from '{key}' back to itself. "
                    "Choose a different page from the available links."
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

        message = result.to_list()
        dst = ctx.deps.destination
        if dst in ctx.deps.candidate_keys:
            message += f"\n\n🎯 DESTINATION '{dst}' IS AVAILABLE! Call get_links('{dst}') to win."

        return message

    @agent.output_validator
    def require_recorded_victory(ctx: RunContext[WikiGolfDeps], output: str) -> str:
        if ctx.partial_output:
            return output
        if ctx.deps.path and ctx.deps.path[-1] == ctx.deps.destination:
            return output
        raise ModelRetry(
            "You stopped prematurely. You have not visited the destination yet. "
            "Victory is only possible by calling get_links with the *exact* destination "
            "key *when that key appears among the current page’s links*."
        )

    return agent


def estimate_cost(messages: list[ModelResponse | ModelRequest]) -> Decimal:
    """Sum the estimated USD cost across every model response in the run."""
    return sum(
        (m.cost().total_price for m in messages if isinstance(m, ModelResponse)),
        Decimal(0),
    )
