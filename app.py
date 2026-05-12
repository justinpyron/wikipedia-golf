"""Wikipedia Golf Dash Frontend

A clean, minimal web app for playing Wikipedia Golf with an AI agent.
Aesthetic: Augusta-inspired (whisper white, Augusta green, championship gold).
"""

import asyncio
import os

import dash
import logfire
from dash import Dash, Input, Output, State, callback, dcc, html
from dash.exceptions import PreventUpdate
from dotenv import load_dotenv
from pydantic_ai import UsageLimits

from agent import WikiGolfDeps, build_agent
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
        # Header
        html.Div(
            [
                html.H1("W I K I P E D I A   G O L F", className="wg-title"),
                # About Section - Collapsible pill below title
                html.Div(
                    [
                        html.Details(
                            [
                                html.Summary(
                                    [
                                        html.Span("ℹ", className="wg-pill-icon"),
                                        html.Span("About", className="wg-pill-text"),
                                    ],
                                    className="wg-pill",
                                ),
                                html.Div(
                                    [
                                        # Section 1: What is Wikipedia Golf
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
                                        # Section 2: About this app
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
                                    className="wg-about-content",
                                ),
                            ],
                            id="about-details",
                            className="wg-pill-details",
                        ),
                    ],
                    className="wg-about-wrapper",
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
                                        html.Button(
                                            "Search",
                                            id="origin-search-btn",
                                            className="wg-button mt-md",
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
                                        html.Button(
                                            "Search",
                                            id="dest-search-btn",
                                            className="wg-button mt-md",
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
        # Settings Stores
        dcc.Store(id="selected-llm", data="openai:gpt-5.4-mini"),
        dcc.Store(id="selected-temperature", data=0.7),
        # Tee Off Button
        html.Button(
            "Tee Off",
            id="tee-off-button",
            className="wg-button wg-button-disabled",
            disabled=True,
        ),
        # Settings Section - Collapsible pill below Tee Off
        html.Div(
            [
                html.Details(
                    [
                        html.Summary(
                            [
                                html.Span("⚙", className="wg-pill-icon"),
                                html.Span("Settings", className="wg-pill-text"),
                            ],
                            className="wg-pill",
                        ),
                        html.Div(
                            [
                                html.Div("Model", className="wg-settings-label"),
                                dcc.RadioItems(
                                    id="llm-radio",
                                    options=[
                                        {
                                            "label": [
                                                html.Img(
                                                    src="/assets/logo_openai.svg",
                                                    height=20,
                                                    style={"marginRight": "10px"},
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
                                                    src="/assets/logo_claude.svg",
                                                    height=20,
                                                    style={"marginRight": "10px"},
                                                ),
                                                html.Span(
                                                    "Claude Haiku 4.5",
                                                    style={
                                                        "fontSize": "14px",
                                                        "lineHeight": "1",
                                                    },
                                                ),
                                            ],
                                            "value": "anthropic:claude-haiku-4-5-20251001",
                                        },
                                    ],
                                    value="openai:gpt-5.4-mini",
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
                                html.Div("Temperature", className="wg-settings-label"),
                                dcc.Slider(
                                    id="temperature-slider",
                                    min=0.0,
                                    max=1.0,
                                    step=0.1,
                                    value=0.7,
                                    marks={i / 10: str(i / 10) for i in range(11)},
                                    allow_direct_input=False,
                                ),
                            ],
                            className="wg-settings-content",
                        ),
                    ],
                    id="settings-details",
                    className="wg-pill-details",
                ),
            ],
            className="wg-settings-wrapper",
        ),
        # Loading spinner
        html.Div(
            [
                html.Div(
                    "⛳ The agent is finding the best path...",
                    className="wg-spinner-text",
                ),
            ],
            id="loading-spinner",
            className="wg-spinner",
            style={"display": "none"},
        ),
        # Result Section
        html.Div(
            [
                html.Div("Path Found", className="wg-result-header"),
                html.Div(id="result-path", className="wg-path-container"),
                html.Div(id="result-stats", className="wg-path-stats"),
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
    Input("origin-search-btn", "n_clicks"),
    State("origin-input", "value"),
    prevent_initial_call=True,
)
def search_origin(n_clicks: int | None, input_value: str | None) -> tuple:
    """Search Wikipedia when user clicks origin search button."""
    if not n_clicks or not input_value or len(input_value) < 2:
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
    Input("dest-search-btn", "n_clicks"),
    State("dest-input", "value"),
    prevent_initial_call=True,
)
def search_dest(n_clicks: int | None, input_value: str | None) -> tuple:
    """Search Wikipedia when user clicks destination search button."""
    if not n_clicks or not input_value or len(input_value) < 2:
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


# ============================================================================
# CALLBACKS - Run Agent
# ============================================================================


@callback(
    Output("loading-spinner", "style"),
    Output("result-container", "style"),
    Output("result-path", "children"),
    Output("result-stats", "children"),
    Output("agent-error", "children"),
    Output("agent-error", "style"),
    Input("tee-off-button", "n_clicks"),
    State("origin-data", "data"),
    State("dest-data", "data"),
    State("selected-llm", "data"),
    State("selected-temperature", "data"),
    running=[
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
        model = selected_llm or "openai:gpt-5.4-mini"
        temperature = selected_temperature if selected_temperature is not None else 0.7
        variant = AgentVariant(
            name="user_configured",
            model=model,
            system_prompt=SYSTEM_PROMPT_V1_0,
            temperature=temperature,
        )
        agent = build_agent(variant)

        deps = WikiGolfDeps(origin=origin_key, destination=dest_key)

        async def run():
            return await agent.run(
                variant.user_prompt,
                deps=deps,
                usage_limits=UsageLimits(
                    request_limit=MAX_LLM_REQUESTS,
                    tool_calls_limit=MAX_TOOL_CALLS,
                ),
            )

        asyncio.run(run())
        path = deps.path

        if not path:
            return (
                {"display": "none"},
                {"display": "none"},
                None,
                None,
                "The agent could not find a path. Please try again.",
                {"display": "block"},
            )

        path_elements = []
        for i, step in enumerate(path):
            is_dest = i == len(path) - 1
            path_elements.append(
                html.Span(
                    step.replace("_", " "),
                    className="wg-path-step-dest" if is_dest else "wg-path-step",
                )
            )
            if i < len(path) - 1:
                path_elements.append(html.Span("→", className="wg-path-arrow"))

        stats = f"{len(path) - 1} links traveled"

        return (
            {"display": "none"},
            {"display": "block"},
            path_elements,
            stats,
            None,
            {"display": "none"},
        )

    except Exception:
        return (
            {"display": "none"},
            {"display": "none"},
            None,
            None,
            "The agent encountered an error. Please try again.",
            {"display": "block"},
        )


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


if __name__ == "__main__":
    app.run(debug=True, port=8050)
