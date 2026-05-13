"""Wikipedia Golf Dash Frontend

A clean, minimal web app for playing Wikipedia Golf with an AI agent.
Aesthetic: Augusta-inspired (whisper white, Augusta green, championship gold).
"""

import asyncio
import os
import time
from decimal import Decimal

import dash
import logfire
from dash import Dash, Input, Output, State, callback, dcc, html
from dash.exceptions import PreventUpdate
from dotenv import load_dotenv
from pydantic_ai import UsageLimits
from pydantic_ai.agent import AgentRunResult

from agent import AgentResult, WikiGolfDeps, build_agent, estimate_cost
from variants import SYSTEM_PROMPT_V1_0, VARIANTS, AgentVariant
from wiki import find_articles

load_dotenv()
logfire.configure(
    service_name="wiki-golf",
    environment=os.getenv("LOGFIRE_ENV", "dev"),
)

# Maximum number of tool calls (page visits) allowed per game
MAX_TOOL_CALLS = 20

# Maximum number of LLM requests (model turns) allowed per game
MAX_LLM_REQUESTS = 30

# Number of search results to display
SEARCH_RESULTS_LIMIT = 5

# Default LLM model - used as initial value and fallback
DEFAULT_MODEL = "openai:gpt-5.4-nano"

# Default temperature setting
DEFAULT_TEMPERATURE = 0.7

# Debounce delay for search-as-you-type (milliseconds)
SEARCH_DEBOUNCE_MS = 1000

# ============================================================================
# STYLES & THEME
# ============================================================================


def create_selected_display(
    data: dict | None, is_destination: bool = False
) -> html.Div | None:
    """Create the selected article display card with reset button."""
    if data is None:
        return None

    thumbnail_url = data.get("thumbnail")
    thumbnail = (
        html.Img(src=thumbnail_url, className="wg-selected-thumbnail")
        if thumbnail_url
        else html.Div("📄", className="wg-selected-thumbnail-placeholder")
    )

    card_class = (
        "wg-selected-card wg-selected-card-dest"
        if is_destination
        else "wg-selected-card wg-selected-card-origin"
    )

    return html.Div(
        [
            thumbnail,
            html.Div(
                [
                    html.Div(
                        data.get("title", ""),
                        className="wg-selected-title",
                    ),
                    html.Div(
                        data.get("description") or "No description available",
                        className="wg-selected-desc",
                    ),
                ],
                className="wg-selected-content",
            ),
            html.Button(
                "×",
                id=f"{'dest' if is_destination else 'origin'}-reset-btn",
                className="wg-reset-btn",
                n_clicks=0,
            ),
        ],
        id=f"{'dest' if is_destination else 'origin'}-selected-display",
        className=f"{card_class} slide-in",
    )


# ============================================================================
# LAYOUT
# ============================================================================


app = Dash(
    __name__,
    title="Wikipedia Golf",
    suppress_callback_exceptions=True,
)

app.layout = html.Div(
    [
        # Header (About / Settings: anchored flyouts, no layout reflow)
        html.Div(
            [
                html.H1("W I K I P E D I A   G O L F", className="wg-title"),
                html.Div(
                    [
                        html.Div(
                            [
                                html.Button(
                                    "About",
                                    id="about-toggle",
                                    type="button",
                                    n_clicks=0,
                                    className="wg-header-link",
                                ),
                                html.Span(
                                    "·",
                                    className="wg-header-util-sep",
                                    **{"aria-hidden": "true"},
                                ),
                                html.Button(
                                    "Settings",
                                    id="settings-toggle",
                                    type="button",
                                    n_clicks=0,
                                    className="wg-header-link",
                                ),
                            ],
                            className="wg-header-util-row",
                        ),
                        html.Div(
                            [
                                html.Div(
                                    html.Div(
                                        [
                                            html.P(
                                                "Wikipedia Golf is the game of navigating from one Wikipedia article to another "
                                                "using the fewest links possible.",
                                                className="wg-about-text",
                                            ),
                                            html.A(
                                                html.B("Learn more →"),
                                                href="https://en.wikipedia.org/wiki/Wikipedia:Wiki_Game",
                                                target="_blank",
                                                className="wg-about-link",
                                            ),
                                            html.Div(style={"height": "40px"}),
                                            html.P(
                                                "In this app, an AI agent plays the game, based on start/end articles you set.",
                                                className="wg-about-text",
                                            ),
                                            html.A(
                                                html.B("View source code →"),
                                                href="https://github.com/justinpyron/wikipedia-golf",
                                                target="_blank",
                                                className="wg-about-link",
                                            ),
                                        ],
                                        className="wg-panel-content",
                                    ),
                                    id="about-panel",
                                    className="wg-flyout-panel",
                                    style={"display": "none"},
                                ),
                                html.Div(
                                    html.Div(
                                        [
                                            html.Div(
                                                "Model", className="wg-settings-label"
                                            ),
                                            dcc.RadioItems(
                                                id="llm-radio",
                                                options=[
                                                    {
                                                        "label": [
                                                            html.Img(
                                                                src="/assets/logo_openai.svg",
                                                                height=20,
                                                                style={
                                                                    "marginRight": "10px"
                                                                },
                                                            ),
                                                            html.Span(
                                                                "GPT-5.4 Nano",
                                                                style={
                                                                    "fontSize": "14px",
                                                                    "lineHeight": "1",
                                                                },
                                                            ),
                                                        ],
                                                        "value": "openai:gpt-5.4-nano",
                                                    },
                                                    {
                                                        "label": [
                                                            html.Img(
                                                                src="/assets/logo_openai.svg",
                                                                height=20,
                                                                style={
                                                                    "marginRight": "10px"
                                                                },
                                                            ),
                                                            html.Span(
                                                                "GPT-5.4 Mini",
                                                                style={
                                                                    "fontSize": "14px",
                                                                    "lineHeight": "1",
                                                                },
                                                            ),
                                                        ],
                                                        "value": "openai:gpt-5.4-mini",
                                                    },
                                                    {
                                                        "label": [
                                                            html.Img(
                                                                src="/assets/logo_openai.svg",
                                                                height=20,
                                                                style={
                                                                    "marginRight": "10px"
                                                                },
                                                            ),
                                                            html.Span(
                                                                "GPT-5.4",
                                                                style={
                                                                    "fontSize": "14px",
                                                                    "lineHeight": "1",
                                                                },
                                                            ),
                                                        ],
                                                        "value": "openai:gpt-5.4",
                                                    },
                                                    {
                                                        "label": [
                                                            html.Img(
                                                                src="/assets/logo_claude.svg",
                                                                height=20,
                                                                style={
                                                                    "marginRight": "10px"
                                                                },
                                                            ),
                                                            html.Span(
                                                                "Claude Haiku 4.5",
                                                                style={
                                                                    "fontSize": "14px",
                                                                    "lineHeight": "1",
                                                                },
                                                            ),
                                                        ],
                                                        "value": "anthropic:claude-haiku-4-5",
                                                    },
                                                    {
                                                        "label": [
                                                            html.Img(
                                                                src="/assets/logo_claude.svg",
                                                                height=20,
                                                                style={
                                                                    "marginRight": "10px"
                                                                },
                                                            ),
                                                            html.Span(
                                                                "Claude Sonnet 4.6",
                                                                style={
                                                                    "fontSize": "14px",
                                                                    "lineHeight": "1",
                                                                },
                                                            ),
                                                        ],
                                                        "value": "anthropic:claude-sonnet-4-6",
                                                    },
                                                ],
                                                value=DEFAULT_MODEL,
                                                labelStyle={
                                                    "display": "flex",
                                                    "alignItems": "center",
                                                    "marginBottom": "2px",
                                                    "cursor": "pointer",
                                                    "padding": "6px 0",
                                                },
                                                inputStyle={
                                                    "marginRight": "10px",
                                                    "marginTop": "0",
                                                    "marginBottom": "0",
                                                },
                                            ),
                                            html.Div(
                                                "Temperature",
                                                className="wg-settings-label",
                                            ),
                                            dcc.Slider(
                                                id="temperature-slider",
                                                min=0.0,
                                                max=1.0,
                                                step=0.1,
                                                value=DEFAULT_TEMPERATURE,
                                                marks={
                                                    i / 10: str(i / 10)
                                                    for i in range(11)
                                                },
                                                allow_direct_input=False,
                                            ),
                                        ],
                                        className="wg-panel-content",
                                    ),
                                    id="settings-panel",
                                    className="wg-flyout-panel",
                                    style={"display": "none"},
                                ),
                            ],
                            className="wg-flyout-stack",
                        ),
                    ],
                    id="header-flyout-anchor",
                    className="wg-header-flyout-anchor",
                ),
            ],
            className="wg-header",
        ),
        # Search Section - Origin and Destination side by side
        html.Div(
            [
                # Origin Section
                html.Div(
                    [
                        html.Div("Origin", className="wg-section-label"),
                        # Search container (shown when no selection)
                        html.Div(
                            [
                                html.Div(
                                    [
                                        dcc.Input(
                                            id="origin-input",
                                            type="text",
                                            placeholder="Search...",
                                            className="wg-input",
                                            autoComplete="off",
                                        ),
                                        html.Div(id="origin-search-results"),
                                        html.Div(
                                            id="origin-error",
                                            className="wg-error mt-sm",
                                            style={"display": "none"},
                                        ),
                                    ],
                                    className="wg-search-wrapper",
                                ),
                            ],
                            id="origin-search-container",
                            className="wg-search-container",
                        ),
                        # Selected display (shown when article selected)
                        html.Div(id="origin-selected-wrapper"),
                    ],
                    className="wg-origin-dest-item",
                ),
                # Destination Section
                html.Div(
                    [
                        html.Div("Destination", className="wg-section-label"),
                        # Search container (shown when no selection)
                        html.Div(
                            [
                                html.Div(
                                    [
                                        dcc.Input(
                                            id="dest-input",
                                            type="text",
                                            placeholder="Search...",
                                            className="wg-input",
                                            autoComplete="off",
                                        ),
                                        html.Div(id="dest-search-results"),
                                        html.Div(
                                            id="dest-error",
                                            className="wg-error mt-sm",
                                            style={"display": "none"},
                                        ),
                                    ],
                                    className="wg-search-wrapper",
                                ),
                            ],
                            id="dest-search-container",
                            className="wg-search-container",
                        ),
                        # Selected display (shown when article selected)
                        html.Div(id="dest-selected-wrapper"),
                    ],
                    className="wg-origin-dest-item",
                ),
            ],
            className="wg-origin-dest-container",
        ),
        # Data Stores
        dcc.Store(id="origin-search-results-data", data=[]),
        dcc.Store(id="dest-search-results-data", data=[]),
        dcc.Store(id="origin-data", data=None),
        dcc.Store(id="dest-data", data=None),
        dcc.Store(id="header-flyout-store", data=None),
        # Search debounce: intervals fire once after SEARCH_DEBOUNCE_MS of inactivity
        dcc.Interval(
            id="origin-debounce-interval",
            interval=SEARCH_DEBOUNCE_MS,
            n_intervals=0,
            disabled=True,
            max_intervals=1,
        ),
        dcc.Store(id="origin-pending-query", data=None),
        dcc.Interval(
            id="dest-debounce-interval",
            interval=SEARCH_DEBOUNCE_MS,
            n_intervals=0,
            disabled=True,
            max_intervals=1,
        ),
        dcc.Store(id="dest-pending-query", data=None),
        # Settings Stores - defaults come from RadioItems/slider value props
        dcc.Store(id="selected-llm", data=None),
        dcc.Store(id="selected-temperature", data=None),
        # Tee Off Button
        html.Button(
            "Tee Off",
            id="tee-off-button",
            className="wg-button wg-button-disabled",
            disabled=True,
        ),
        # Loading state: label + pulsating dot (duration shown in results)
        html.Div(
            [
                html.Div(
                    "Finding path",
                    className="wg-loading-label",
                ),
                html.Div(className="wg-pulse-dot"),
            ],
            id="loading-spinner",
            className="wg-loading-container",
            style={"display": "none"},
        ),
        # Updated by clientside when Tee Off fires so stale results hide immediately.
        dcc.Store(id="tee-off-clear-sentinel", data=None),
        # Result Section
        html.Div(
            [
                html.Div(id="result-links-hero", className="wg-result-links-hero"),
                html.Div(id="result-path", className="wg-path-container"),
                html.Div(id="result-usage-stats", className="wg-result-usage"),
            ],
            id="result-container",
            className="wg-result-container",
            style={"display": "none"},
        ),
        # Agent error
        html.Div(
            id="agent-error",
            className="wg-error mt-lg",
            style={"display": "none"},
        ),
    ],
    className="wg-container",
)


# ============================================================================
# CALLBACKS - Search
# ============================================================================


@callback(
    Output("origin-search-results-data", "data"),
    Output("origin-error", "children"),
    Output("origin-error", "style"),
    Input("origin-debounce-interval", "n_intervals"),
    State("origin-pending-query", "data"),
    prevent_initial_call=True,
)
def search_origin(n_intervals: int, input_value: str | None) -> tuple:
    """Search Wikipedia after debounce period elapses."""
    if not n_intervals or not input_value or len(input_value) < 2:
        return [], None, {"display": "none"}

    try:
        results = find_articles(input_value, limit=SEARCH_RESULTS_LIMIT)
        if not results:
            return (
                [],
                f'No articles found matching "{input_value}"',
                {"display": "block"},
            )

        # Store search results as list of dicts
        search_results_data = [
            {
                "key": r.key,
                "title": r.title,
                "description": r.description,
                "thumbnail": r.thumbnail.get("url") if r.thumbnail else None,
            }
            for r in results
        ]

        return search_results_data, None, {"display": "none"}

    except Exception:
        return (
            [],
            "Unable to search. Please try again.",
            {"display": "block"},
        )


@callback(
    Output("dest-search-results-data", "data"),
    Output("dest-error", "children"),
    Output("dest-error", "style"),
    Input("dest-debounce-interval", "n_intervals"),
    State("dest-pending-query", "data"),
    prevent_initial_call=True,
)
def search_dest(n_intervals: int, input_value: str | None) -> tuple:
    """Search Wikipedia after debounce period elapses."""
    if not n_intervals or not input_value or len(input_value) < 2:
        return [], None, {"display": "none"}

    try:
        results = find_articles(input_value, limit=SEARCH_RESULTS_LIMIT)
        if not results:
            return (
                [],
                f'No articles found matching "{input_value}"',
                {"display": "block"},
            )

        search_results_data = [
            {
                "key": r.key,
                "title": r.title,
                "description": r.description,
                "thumbnail": r.thumbnail.get("url") if r.thumbnail else None,
            }
            for r in results
        ]

        return search_results_data, None, {"display": "none"}

    except Exception:
        return (
            [],
            "Unable to search. Please try again.",
            {"display": "block"},
        )


# Clientside debounce: each keystroke stores the query and restarts the timer
app.clientside_callback(
    """
    function(value) {
        return [value, 0, false];
    }
    """,
    Output("origin-pending-query", "data"),
    Output("origin-debounce-interval", "n_intervals"),
    Output("origin-debounce-interval", "disabled"),
    Input("origin-input", "value"),
    prevent_initial_call=True,
)

app.clientside_callback(
    """
    function(value) {
        return [value, 0, false];
    }
    """,
    Output("dest-pending-query", "data"),
    Output("dest-debounce-interval", "n_intervals"),
    Output("dest-debounce-interval", "disabled"),
    Input("dest-input", "value"),
    prevent_initial_call=True,
)


# ============================================================================
# CALLBACKS - Render Search Results (single writer per output)
# ============================================================================


@callback(
    Output("origin-search-results", "children"),
    Input("origin-search-results-data", "data"),
    prevent_initial_call=True,
)
def render_origin_search_results(search_results_data: list[dict]) -> html.Div | None:
    """Render origin search results from store data."""
    if not search_results_data:
        return None

    result_items = []
    for i, result in enumerate(search_results_data):
        result_items.append(
            html.Div(
                [
                    html.Div(
                        result["title"],
                        className="wg-search-result-title",
                    ),
                    html.Div(
                        result.get("description") or "",
                        className="wg-search-result-desc",
                    ),
                ],
                id={"type": "origin-search-result", "index": i},
                className="wg-search-result-item",
                n_clicks=0,
            )
        )

    return html.Div(result_items, className="wg-search-results-container")


@callback(
    Output("dest-search-results", "children"),
    Input("dest-search-results-data", "data"),
    prevent_initial_call=True,
)
def render_dest_search_results(search_results_data: list[dict]) -> html.Div | None:
    """Render destination search results from store data."""
    if not search_results_data:
        return None

    result_items = []
    for i, result in enumerate(search_results_data):
        result_items.append(
            html.Div(
                [
                    html.Div(
                        result["title"],
                        className="wg-search-result-title",
                    ),
                    html.Div(
                        result.get("description") or "",
                        className="wg-search-result-desc",
                    ),
                ],
                id={"type": "dest-search-result", "index": i},
                className="wg-search-result-item",
                n_clicks=0,
            )
        )

    return html.Div(result_items, className="wg-search-results-container")


# ============================================================================
# CALLBACKS - Selection (user clicks a search result)
# ============================================================================


@callback(
    Output("origin-data", "data"),
    Output("origin-search-results-data", "data", allow_duplicate=True),
    Output("origin-input", "value"),
    Input({"type": "origin-search-result", "index": dash.ALL}, "n_clicks"),
    State("origin-search-results-data", "data"),
    prevent_initial_call=True,
)
def select_origin(n_clicks: list[int | None], search_results_data: list[dict]) -> tuple:
    """Handle origin article selection from search results."""
    # Check for actual click
    if not n_clicks or all(c is None or c == 0 for c in n_clicks):
        raise PreventUpdate

    ctx = dash.callback_context
    triggered_id = ctx.triggered_id
    if not triggered_id:
        raise PreventUpdate

    clicked_index = triggered_id.get("index")
    if clicked_index is None or clicked_index >= len(search_results_data):
        raise PreventUpdate

    if n_clicks[clicked_index] is None or n_clicks[clicked_index] == 0:
        raise PreventUpdate

    selected_data = search_results_data[clicked_index]

    # Return selected data, clear search results, clear input
    return selected_data, [], ""


@callback(
    Output("dest-data", "data"),
    Output("dest-search-results-data", "data", allow_duplicate=True),
    Output("dest-input", "value"),
    Input({"type": "dest-search-result", "index": dash.ALL}, "n_clicks"),
    State("dest-search-results-data", "data"),
    prevent_initial_call=True,
)
def select_dest(n_clicks: list[int | None], search_results_data: list[dict]) -> tuple:
    """Handle destination article selection from search results."""
    if not n_clicks or all(c is None or c == 0 for c in n_clicks):
        raise PreventUpdate

    ctx = dash.callback_context
    triggered_id = ctx.triggered_id
    if not triggered_id:
        raise PreventUpdate

    clicked_index = triggered_id.get("index")
    if clicked_index is None or clicked_index >= len(search_results_data):
        raise PreventUpdate

    if n_clicks[clicked_index] is None or n_clicks[clicked_index] == 0:
        raise PreventUpdate

    selected_data = search_results_data[clicked_index]

    return selected_data, [], ""


# ============================================================================
# CALLBACKS - Progressive Disclosure (show/hide search vs selected)
# ============================================================================


@callback(
    Output("origin-search-container", "className"),
    Output("origin-selected-wrapper", "children"),
    Input("origin-data", "data"),
)
def update_origin_display(origin_data: dict | None) -> tuple:
    """Show/hide origin search container and update selected display."""
    if origin_data is None:
        # No selection - show search, hide selected
        search_class = "wg-search-container"
        selected_display = create_selected_display(None, False)
    else:
        # Has selection - hide search, show selected
        search_class = "wg-search-container hidden"
        selected_display = create_selected_display(origin_data, False)

    return search_class, selected_display


@callback(
    Output("dest-search-container", "className"),
    Output("dest-selected-wrapper", "children"),
    Input("dest-data", "data"),
)
def update_dest_display(dest_data: dict | None) -> tuple:
    """Show/hide destination search container and update selected display."""
    if dest_data is None:
        # No selection - show search, hide selected
        search_class = "wg-search-container"
        selected_display = create_selected_display(None, True)
    else:
        # Has selection - hide search, show selected
        search_class = "wg-search-container hidden"
        selected_display = create_selected_display(dest_data, True)

    return search_class, selected_display


# ============================================================================
# CALLBACKS - Reset (clear selection and return to search)
# ============================================================================


@callback(
    Output("origin-data", "data", allow_duplicate=True),
    Output("origin-input", "value", allow_duplicate=True),
    Input("origin-reset-btn", "n_clicks"),
    prevent_initial_call=True,
)
def reset_origin(n_clicks: int | None) -> tuple:
    """Reset origin selection and return to search state."""
    if not n_clicks:
        raise PreventUpdate

    return None, ""


@callback(
    Output("dest-data", "data", allow_duplicate=True),
    Output("dest-input", "value", allow_duplicate=True),
    Input("dest-reset-btn", "n_clicks"),
    prevent_initial_call=True,
)
def reset_dest(n_clicks: int | None) -> tuple:
    """Reset destination selection and return to search state."""
    if not n_clicks:
        raise PreventUpdate

    return None, ""


# ============================================================================
# CALLBACKS - Tee Off Button
# ============================================================================


@callback(
    Output("tee-off-button", "disabled"),
    Output("tee-off-button", "className"),
    Input("origin-data", "data"),
    Input("dest-data", "data"),
)
def toggle_button(origin_data: dict | None, dest_data: dict | None) -> tuple:
    """Enable/disable Tee Off button based on selections."""
    can_tee_off = origin_data is not None and dest_data is not None

    if can_tee_off:
        return False, "wg-button"
    return True, "wg-button wg-button-disabled"


def build_links_traveled_hero(agent_result: AgentResult) -> html.Div:
    """Hero line: N link(s) traveled."""
    links_count = len(agent_result.path) - 1
    return html.Div(
        [
            html.Span(str(links_count), className="wg-scorecard-hero-number"),
            html.Span(
                " link" if links_count == 1 else " links",
                className="wg-scorecard-hero-label",
            ),
            html.Span(" traveled", className="wg-scorecard-hero-label"),
        ],
        className="wg-scorecard-hero",
    )


def build_usage_stats_block(agent_result: AgentResult) -> html.Div:
    """Duration, cost, and tokens row with divider."""
    duration_formatted = f"{agent_result.duration_seconds:.1f}s"
    tokens_formatted = f"{agent_result.total_tokens:,}"
    cost_formatted = f"${agent_result.estimated_cost_usd:.4f}"

    return html.Div(
        [
            html.Div(className="wg-scorecard-divider"),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div("Duration", className="wg-scorecard-stat-label"),
                            html.Div(
                                duration_formatted,
                                className="wg-scorecard-stat-value",
                            ),
                        ],
                        className="wg-scorecard-stat",
                    ),
                    html.Div(
                        [
                            html.Div("Cost", className="wg-scorecard-stat-label"),
                            html.Div(
                                cost_formatted,
                                className="wg-scorecard-stat-value",
                            ),
                        ],
                        className="wg-scorecard-stat",
                    ),
                    html.Div(
                        [
                            html.Div("Tokens", className="wg-scorecard-stat-label"),
                            html.Div(
                                tokens_formatted,
                                className="wg-scorecard-stat-value",
                            ),
                        ],
                        className="wg-scorecard-stat",
                    ),
                ],
                className="wg-scorecard-stats-row",
            ),
        ],
    )


def build_path_elements(path: list[str]) -> list:
    """Build the path display elements with arrows."""
    elements = []
    for i, step in enumerate(path):
        is_dest = i == len(path) - 1
        elements.append(
            html.Span(
                step.replace("_", " "),
                className="wg-path-step-dest" if is_dest else "wg-path-step",
            )
        )
        if i < len(path) - 1:
            elements.append(html.Span("→", className="wg-path-arrow"))
    return elements


def calculate_cost(result: AgentRunResult) -> Decimal:
    """Calculate cost from agent run result, returning Decimal."""
    try:
        return estimate_cost(result)
    except Exception:
        return Decimal("0")


# ============================================================================
# CALLBACKS - Run Agent
# ============================================================================


@callback(
    Output("result-container", "style"),
    Output("result-links-hero", "children"),
    Output("result-path", "children"),
    Output("result-usage-stats", "children"),
    Output("agent-error", "children"),
    Output("agent-error", "style"),
    Input("tee-off-button", "n_clicks"),
    State("origin-data", "data"),
    State("dest-data", "data"),
    State("selected-llm", "data"),
    State("selected-temperature", "data"),
    running=[
        # Show loading spinner while agent runs, hide when complete
        (Output("loading-spinner", "style"), {"display": "block"}, {"display": "none"}),
        (Output("tee-off-button", "disabled"), True, False),
        (
            Output("tee-off-button", "className"),
            "wg-button wg-button-disabled",
            "wg-button",
        ),
    ],
    prevent_initial_call=True,
)
def run_agent(
    n_clicks: int | None,
    origin_data: dict | None,
    dest_data: dict | None,
    selected_llm: str | None,
    selected_temperature: float | None,
) -> tuple:
    """Run the Wikipedia Golf agent and display results."""
    if n_clicks is None or not origin_data or not dest_data:
        raise PreventUpdate

    try:
        origin_key = origin_data.get("key")
        dest_key = dest_data.get("key")

        # Use selected settings or defaults
        model = selected_llm or DEFAULT_MODEL
        temperature = (
            selected_temperature
            if selected_temperature is not None
            else DEFAULT_TEMPERATURE
        )
        variant = AgentVariant(
            name="user_configured",
            model=model,
            system_prompt=SYSTEM_PROMPT_V1_0,
            temperature=temperature,
        )
        agent = build_agent(variant)

        deps = WikiGolfDeps(origin=origin_key, destination=dest_key)
        start_time = time.time()

        async def run():
            return await agent.run(
                variant.user_prompt,
                deps=deps,
                usage_limits=UsageLimits(
                    request_limit=MAX_LLM_REQUESTS,
                    tool_calls_limit=MAX_TOOL_CALLS,
                ),
            )

        result = asyncio.run(run())
        duration_seconds = time.time() - start_time

        usage = result.usage()
        total_tokens = usage.total_tokens if usage else 0
        estimated_cost = calculate_cost(result)

        agent_result = AgentResult(
            path=deps.path,
            duration_seconds=duration_seconds,
            total_tokens=total_tokens,
            estimated_cost_usd=float(estimated_cost),
        )

        if not agent_result.path:
            return (
                {"display": "none"},
                None,
                None,
                None,
                "The agent could not find a path. Please try again.",
                {"display": "block"},
            )

        path_elements = build_path_elements(agent_result.path)

        return (
            {"display": "block"},
            build_links_traveled_hero(agent_result),
            path_elements,
            build_usage_stats_block(agent_result),
            None,
            {"display": "none"},
        )

    except Exception:
        return (
            {"display": "none"},
            None,
            None,
            None,
            "The agent encountered an error. Please try again.",
            {"display": "block"},
        )


# ============================================================================
# CALLBACKS - Header flyout (About / Settings)
# ============================================================================


@callback(
    Output("header-flyout-store", "data"),
    Input("about-toggle", "n_clicks"),
    Input("settings-toggle", "n_clicks"),
    State("header-flyout-store", "data"),
    prevent_initial_call=True,
)
def toggle_header_flyout(
    about_clicks: int | None,
    settings_clicks: int | None,
    current: str | None,
) -> str | None:
    """Accordion: one panel at a time; click active toggle closes."""
    ctx = dash.callback_context
    if not ctx.triggered_id:
        raise PreventUpdate
    tid = ctx.triggered_id
    if tid == "about-toggle":
        return None if current == "about" else "about"
    if tid == "settings-toggle":
        return None if current == "settings" else "settings"
    raise PreventUpdate


@callback(
    Output("about-toggle", "className"),
    Output("settings-toggle", "className"),
    Output("header-flyout-anchor", "className"),
    Output("about-panel", "style"),
    Output("settings-panel", "style"),
    Input("header-flyout-store", "data"),
)
def render_header_flyout(data: str | None) -> tuple:
    """Sync link state and panel visibility from store (incl. outside click)."""
    link = "wg-header-link"
    about_cls = f"{link} {link}-active" if data == "about" else link
    settings_cls = f"{link} {link}-active" if data == "settings" else link
    anchor_cls = "wg-header-flyout-anchor" + (" is-open" if data else "")
    about_style = {"display": "block"} if data == "about" else {"display": "none"}
    settings_style = {"display": "block"} if data == "settings" else {"display": "none"}
    return about_cls, settings_cls, anchor_cls, about_style, settings_style


# ============================================================================
# CALLBACKS - Settings Persistence
# ============================================================================


@callback(
    Output("selected-llm", "data"),
    Input("llm-radio", "value"),
    prevent_initial_call=True,
)
def store_llm_selection(value: str | None) -> str | None:
    """Store selected LLM model."""
    return value


@callback(
    Output("selected-temperature", "data"),
    Input("temperature-slider", "value"),
    prevent_initial_call=True,
)
def store_temperature(value: float | None) -> float | None:
    """Store temperature setting."""
    return value


# ============================================================================
# CLIENTSIDE CALLBACK - Loading Counter
# ============================================================================

# Clear prior run output as soon as Tee Off is clicked (before server returns).
app.clientside_callback(
    """
    function(n_clicks) {
        if (!n_clicks) {
            return window.dash_clientside.no_update;
        }
        const sp = window.dash_clientside.set_props;
        sp("result-container", {style: {display: "none"}});
        sp("result-links-hero", {children: null});
        sp("result-path", {children: null});
        sp("result-usage-stats", {children: null});
        sp("agent-error", {style: {display: "none"}, children: null});
        return Date.now();
    }
    """,
    Output("tee-off-clear-sentinel", "data"),
    Input("tee-off-button", "n_clicks"),
    prevent_initial_call=True,
)


if __name__ == "__main__":
    app.run(debug=True, port=8050)
