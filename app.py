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
from variants import VARIANTS
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

# Augusta-inspired color palette
COLORS = {
    "whisper": "#F8F7F4",
    "white": "#FFFFFF",
    "augusta_green": "#1E4D2B",
    "billiard_green": "#2D5A3D",
    "championship_gold": "#F4C430",
    "charcoal": "#1A1A1A",
    "slate": "#5A5A5A",
    "mist": "#E5E3DF",
    "light_mist": "#F0EFED",
}

app = Dash(
    __name__,
    title="Wikipedia Golf",
    suppress_callback_exceptions=True,
)

# Styles
CONTAINER_STYLE = {
    "maxWidth": "900px",
    "margin": "0 auto",
    "padding": "48px 24px",
    "backgroundColor": COLORS["whisper"],
    "minHeight": "100vh",
    "fontFamily": "'Inter', -apple-system, BlinkMacSystemFont, sans-serif",
}

HEADER_STYLE = {
    "textAlign": "center",
    "marginBottom": "48px",
}

TITLE_STYLE = {
    "fontSize": "32px",
    "fontWeight": "500",
    "color": COLORS["augusta_green"],
    "letterSpacing": "0.05em",
    "margin": "0 0 8px 0",
    "fontFamily": "'Crimson Text', Georgia, serif",
}

SUBTITLE_STYLE = {
    "fontSize": "14px",
    "color": COLORS["slate"],
    "margin": "0",
    "fontWeight": "400",
}

SECTION_LABEL_STYLE = {
    "fontSize": "12px",
    "fontWeight": "600",
    "color": COLORS["slate"],
    "textTransform": "uppercase",
    "letterSpacing": "0.1em",
    "marginBottom": "8px",
}

INPUT_STYLE = {
    "width": "100%",
    "height": "48px",
    "padding": "0 16px",
    "fontSize": "16px",
    "lineHeight": "1.5",
    "border": f"1px solid {COLORS['mist']}",
    "borderRadius": "4px",
    "backgroundColor": COLORS["white"],
    "color": COLORS["charcoal"],
    "outline": "none",
    "boxSizing": "border-box",
    "transition": "border-color 0.2s ease",
}

SEARCH_WRAPPER_STYLE = {
    "position": "relative",
    "zIndex": "100",
}

SEARCH_RESULTS_CONTAINER_STYLE = {
    "position": "absolute",
    "top": "100%",
    "left": "0",
    "right": "0",
    "marginTop": "4px",
    "border": f"1px solid {COLORS['mist']}",
    "borderRadius": "4px",
    "overflow": "hidden",
    "boxShadow": "0 4px 12px rgba(0,0,0,0.15)",
    "zIndex": "1000",
    "backgroundColor": COLORS["white"],
}

SEARCH_RESULT_ITEM_STYLE = {
    "padding": "12px 16px",
    "cursor": "pointer",
    "backgroundColor": COLORS["white"],
    "borderBottom": f"1px solid {COLORS['light_mist']}",
    "transition": "background-color 0.15s ease",
}


BUTTON_STYLE = {
    "width": "100%",
    "padding": "16px 32px",
    "fontSize": "14px",
    "fontWeight": "600",
    "textTransform": "uppercase",
    "letterSpacing": "0.15em",
    "color": COLORS["white"],
    "backgroundColor": COLORS["augusta_green"],
    "border": "none",
    "borderRadius": "2px",
    "cursor": "pointer",
    "transition": "background-color 0.2s ease",
}

BUTTON_DISABLED_STYLE = {
    **BUTTON_STYLE,
    "backgroundColor": COLORS["mist"],
    "color": COLORS["slate"],
    "cursor": "not-allowed",
}

SPINNER_STYLE = {
    "textAlign": "center",
    "padding": "48px 0",
    "color": COLORS["augusta_green"],
    "fontSize": "14px",
}

# Selected article display styles (inline card)
SELECTED_CARD_STYLE = {
    "display": "flex",
    "alignItems": "flex-start",
    "gap": "16px",
    "padding": "16px",
    "backgroundColor": COLORS["white"],
    "border": f"1px solid {COLORS['mist']}",
    "borderRadius": "4px",
    "position": "relative",
    "transition": "all 0.3s ease",
}

SELECTED_CARD_ORIGIN_STYLE = {
    **SELECTED_CARD_STYLE,
    "borderLeft": f"4px solid {COLORS['augusta_green']}",
}

SELECTED_CARD_DEST_STYLE = {
    **SELECTED_CARD_STYLE,
    "borderLeft": f"4px solid {COLORS['championship_gold']}",
}

SELECTED_THUMBNAIL_STYLE = {
    "width": "48px",
    "height": "48px",
    "objectFit": "cover",
    "borderRadius": "4px",
    "flexShrink": "0",
    "border": f"1px solid {COLORS['light_mist']}",
}

SELECTED_THUMBNAIL_PLACEHOLDER_STYLE = {
    "width": "48px",
    "height": "48px",
    "backgroundColor": COLORS["light_mist"],
    "borderRadius": "4px",
    "flexShrink": "0",
    "display": "flex",
    "alignItems": "center",
    "justifyContent": "center",
    "fontSize": "20px",
    "color": COLORS["slate"],
}

SELECTED_CONTENT_STYLE = {
    "flex": "1",
    "minWidth": "0",
}

SELECTED_TITLE_STYLE = {
    "fontSize": "18px",
    "fontWeight": "600",
    "color": COLORS["charcoal"],
    "margin": "0 0 4px 0",
    "lineHeight": "1.3",
}

SELECTED_DESC_STYLE = {
    "fontSize": "13px",
    "color": COLORS["slate"],
    "margin": "0",
    "lineHeight": "1.4",
    "display": "-webkit-box",
    "WebkitLineClamp": "2",
    "WebkitBoxOrient": "vertical",
    "overflow": "hidden",
    "textOverflow": "ellipsis",
}

RESET_BUTTON_STYLE = {
    "position": "absolute",
    "top": "12px",
    "right": "12px",
    "width": "28px",
    "height": "28px",
    "border": "none",
    "borderRadius": "4px",
    "backgroundColor": COLORS["light_mist"],
    "color": COLORS["slate"],
    "fontSize": "16px",
    "cursor": "pointer",
    "display": "flex",
    "alignItems": "center",
    "justifyContent": "center",
    "transition": "all 0.2s ease",
    "padding": "0",
    "lineHeight": "1",
}

# Animation style for search container transitions
SEARCH_CONTAINER_STYLE = {
    "transition": "all 0.3s ease",
}

RESULT_CONTAINER_STYLE = {
    "marginTop": "48px",
    "padding": "32px",
    "backgroundColor": COLORS["white"],
    "border": f"1px solid {COLORS['mist']}",
    "borderRadius": "4px",
    "textAlign": "center",
}

RESULT_HEADER_STYLE = {
    "fontSize": "12px",
    "fontWeight": "600",
    "color": COLORS["slate"],
    "textTransform": "uppercase",
    "letterSpacing": "0.1em",
    "marginBottom": "24px",
}

PATH_CONTAINER_STYLE = {
    "display": "flex",
    "alignItems": "center",
    "justifyContent": "center",
    "gap": "12px",
    "flexWrap": "wrap",
}

PATH_STEP_STYLE = {
    "fontSize": "15px",
    "fontWeight": "500",
    "color": COLORS["charcoal"],
    "fontFamily": "'JetBrains Mono', monospace",
}

PATH_STEP_DEST_STYLE = {
    **PATH_STEP_STYLE,
    "color": COLORS["championship_gold"],
    "fontWeight": "700",
}

PATH_ARROW_STYLE = {
    "fontSize": "14px",
    "color": COLORS["slate"],
}

PATH_STATS_STYLE = {
    "marginTop": "24px",
    "fontSize": "13px",
    "color": COLORS["slate"],
}

ERROR_STYLE = {
    "marginTop": "8px",
    "padding": "10px 12px",
    "backgroundColor": "#FEF2F2",
    "border": "1px solid #FECACA",
    "borderRadius": "4px",
    "color": "#DC2626",
    "fontSize": "13px",
}


def create_selected_display(
    data: dict | None, is_destination: bool = False
) -> html.Div | None:
    """Create the selected article display card with reset button."""
    if data is None:
        return None

    thumbnail_url = data.get("thumbnail")
    thumbnail = (
        html.Img(src=thumbnail_url, style=SELECTED_THUMBNAIL_STYLE)
        if thumbnail_url
        else html.Div("📄", style=SELECTED_THUMBNAIL_PLACEHOLDER_STYLE)
    )

    card_style = (
        SELECTED_CARD_DEST_STYLE if is_destination else SELECTED_CARD_ORIGIN_STYLE
    )

    return html.Div(
        [
            thumbnail,
            html.Div(
                [
                    html.Div(
                        data.get("title", ""),
                        style=SELECTED_TITLE_STYLE,
                    ),
                    html.Div(
                        data.get("description") or "No description available",
                        style=SELECTED_DESC_STYLE,
                    ),
                ],
                style=SELECTED_CONTENT_STYLE,
            ),
            html.Button(
                "×",
                id=f"{'dest' if is_destination else 'origin'}-reset-btn",
                style=RESET_BUTTON_STYLE,
                className="reset-btn",
                n_clicks=0,
            ),
        ],
        id=f"{'dest' if is_destination else 'origin'}-selected-display",
        style=card_style,
        className="selected-card slide-in",
    )


# CSS animations via inline style tag since Dash doesn't support index_string well with debug mode
# We'll add a clientside callback or use dcc.Store to trigger CSS classes

# ============================================================================
# LAYOUT
# ============================================================================

app.layout = html.Div(
    [
        # Header
        html.Div(
            [
                html.H1("W I K I P E D I A   G O L F", style=TITLE_STYLE),
                html.P("with an AI agent", style=SUBTITLE_STYLE),
            ],
            style=HEADER_STYLE,
        ),
        # Search Section - Origin and Destination side by side
        html.Div(
            [
                # Origin Section
                html.Div(
                    [
                        html.Div("From", style=SECTION_LABEL_STYLE),
                        # Search container (shown when no selection)
                        html.Div(
                            [
                                html.Div(
                                    [
                                        dcc.Input(
                                            id="origin-input",
                                            type="text",
                                            placeholder="Search for origin article...",
                                            style=INPUT_STYLE,
                                            autoComplete="off",
                                        ),
                                        html.Div(id="origin-search-results"),
                                        html.Div(
                                            id="origin-error",
                                            style={
                                                **ERROR_STYLE,
                                                "display": "none",
                                                "marginTop": "4px",
                                            },
                                        ),
                                        html.Button(
                                            "Search",
                                            id="origin-search-btn",
                                            style={
                                                **BUTTON_STYLE,
                                                "marginTop": "12px",
                                                "width": "100%",
                                                "position": "relative",
                                                "zIndex": "1",
                                            },
                                        ),
                                    ],
                                    style=SEARCH_WRAPPER_STYLE,
                                ),
                            ],
                            id="origin-search-container",
                            className="search-container",
                        ),
                        # Selected display (shown when article selected)
                        html.Div(id="origin-selected-wrapper"),
                    ],
                    style={"flex": "1", "minWidth": "300px"},
                ),
                # Destination Section
                html.Div(
                    [
                        html.Div("To", style=SECTION_LABEL_STYLE),
                        # Search container (shown when no selection)
                        html.Div(
                            [
                                html.Div(
                                    [
                                        dcc.Input(
                                            id="dest-input",
                                            type="text",
                                            placeholder="Search for destination article...",
                                            style=INPUT_STYLE,
                                            autoComplete="off",
                                        ),
                                        html.Div(id="dest-search-results"),
                                        html.Div(
                                            id="dest-error",
                                            style={
                                                **ERROR_STYLE,
                                                "display": "none",
                                                "marginTop": "4px",
                                            },
                                        ),
                                        html.Button(
                                            "Search",
                                            id="dest-search-btn",
                                            style={
                                                **BUTTON_STYLE,
                                                "marginTop": "12px",
                                                "width": "100%",
                                                "position": "relative",
                                                "zIndex": "1",
                                            },
                                        ),
                                    ],
                                    style=SEARCH_WRAPPER_STYLE,
                                ),
                            ],
                            id="dest-search-container",
                            className="search-container",
                        ),
                        # Selected display (shown when article selected)
                        html.Div(id="dest-selected-wrapper"),
                    ],
                    style={"flex": "1", "minWidth": "300px"},
                ),
            ],
            style={
                "display": "flex",
                "gap": "48px",
                "marginBottom": "32px",
                "flexWrap": "wrap",
            },
        ),
        # Data Stores
        dcc.Store(id="origin-search-results-data", data=[]),
        dcc.Store(id="dest-search-results-data", data=[]),
        dcc.Store(id="origin-data", data=None),
        dcc.Store(id="dest-data", data=None),
        # Tee Off Button
        html.Button(
            "Tee Off",
            id="tee-off-button",
            style=BUTTON_DISABLED_STYLE,
            disabled=True,
        ),
        # Loading spinner
        html.Div(
            [
                html.Div(
                    "⛳ The agent is finding the best path...",
                    style={"color": COLORS["augusta_green"], "fontSize": "16px"},
                ),
            ],
            id="loading-spinner",
            style={**SPINNER_STYLE, "display": "none"},
        ),
        # Result Section
        html.Div(
            [
                html.Div("Path Found", style=RESULT_HEADER_STYLE),
                html.Div(id="result-path", style=PATH_CONTAINER_STYLE),
                html.Div(id="result-stats", style=PATH_STATS_STYLE),
            ],
            id="result-container",
            style={**RESULT_CONTAINER_STYLE, "display": "none"},
        ),
        # Agent error
        html.Div(
            id="agent-error",
            style={**ERROR_STYLE, "marginTop": "24px", "display": "none"},
        ),
    ],
    style=CONTAINER_STYLE,
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
        return [], None, {**ERROR_STYLE, "display": "none"}

    try:
        results = find_articles(input_value, limit=SEARCH_RESULTS_LIMIT)
        if not results:
            return (
                [],
                f'No articles found matching "{input_value}"',
                {**ERROR_STYLE, "display": "block"},
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

        return search_results_data, None, {**ERROR_STYLE, "display": "none"}

    except Exception:
        return (
            [],
            "Unable to search. Please try again.",
            {**ERROR_STYLE, "display": "block"},
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
        return [], None, {**ERROR_STYLE, "display": "none"}

    try:
        results = find_articles(input_value, limit=SEARCH_RESULTS_LIMIT)
        if not results:
            return (
                [],
                f'No articles found matching "{input_value}"',
                {**ERROR_STYLE, "display": "block"},
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

        return search_results_data, None, {**ERROR_STYLE, "display": "none"}

    except Exception:
        return (
            [],
            "Unable to search. Please try again.",
            {**ERROR_STYLE, "display": "block"},
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
                        style={
                            "fontSize": "14px",
                            "fontWeight": "500",
                            "color": COLORS["charcoal"],
                        },
                    ),
                    html.Div(
                        result.get("description") or "",
                        style={
                            "fontSize": "12px",
                            "color": COLORS["slate"],
                            "marginTop": "2px",
                        },
                    ),
                ],
                id={"type": "origin-search-result", "index": i},
                style=SEARCH_RESULT_ITEM_STYLE,
                n_clicks=0,
            )
        )

    return html.Div(result_items, style=SEARCH_RESULTS_CONTAINER_STYLE)


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
                        style={
                            "fontSize": "14px",
                            "fontWeight": "500",
                            "color": COLORS["charcoal"],
                        },
                    ),
                    html.Div(
                        result.get("description") or "",
                        style={
                            "fontSize": "12px",
                            "color": COLORS["slate"],
                            "marginTop": "2px",
                        },
                    ),
                ],
                id={"type": "dest-search-result", "index": i},
                style=SEARCH_RESULT_ITEM_STYLE,
                n_clicks=0,
            )
        )

    return html.Div(result_items, style=SEARCH_RESULTS_CONTAINER_STYLE)


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
    Output("origin-search-container", "style"),
    Output("origin-selected-wrapper", "children"),
    Input("origin-data", "data"),
)
def update_origin_display(origin_data: dict | None) -> tuple:
    """Show/hide origin search container and update selected display."""
    if origin_data is None:
        # No selection - show search, hide selected
        search_style = SEARCH_CONTAINER_STYLE
        selected_display = create_selected_display(None, False)
    else:
        # Has selection - hide search, show selected
        search_style = {**SEARCH_CONTAINER_STYLE, "display": "none"}
        selected_display = create_selected_display(origin_data, False)

    return search_style, selected_display


@callback(
    Output("dest-search-container", "style"),
    Output("dest-selected-wrapper", "children"),
    Input("dest-data", "data"),
)
def update_dest_display(dest_data: dict | None) -> tuple:
    """Show/hide destination search container and update selected display."""
    if dest_data is None:
        # No selection - show search, hide selected
        search_style = SEARCH_CONTAINER_STYLE
        selected_display = create_selected_display(None, True)
    else:
        # Has selection - hide search, show selected
        search_style = {**SEARCH_CONTAINER_STYLE, "display": "none"}
        selected_display = create_selected_display(dest_data, True)

    return search_style, selected_display


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
    Output("tee-off-button", "style"),
    Input("origin-data", "data"),
    Input("dest-data", "data"),
)
def toggle_button(origin_data: dict | None, dest_data: dict | None) -> tuple:
    """Enable/disable Tee Off button based on selections."""
    can_tee_off = origin_data is not None and dest_data is not None

    if can_tee_off:
        return False, BUTTON_STYLE
    return True, BUTTON_DISABLED_STYLE


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
    running=[
        (Output("tee-off-button", "disabled"), True, False),
        (Output("tee-off-button", "style"), BUTTON_DISABLED_STYLE, BUTTON_STYLE),
    ],
    prevent_initial_call=True,
)
def run_agent(
    n_clicks: int | None,
    origin_data: dict | None,
    dest_data: dict | None,
) -> tuple:
    """Run the Wikipedia Golf agent and display results."""
    if n_clicks is None or not origin_data or not dest_data:
        raise PreventUpdate

    loading_style = {**SPINNER_STYLE, "display": "block"}
    result_style = {**RESULT_CONTAINER_STYLE, "display": "none"}

    try:
        origin_key = origin_data.get("key")
        dest_key = dest_data.get("key")

        variant = VARIANTS[1]
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
                {**SPINNER_STYLE, "display": "none"},
                result_style,
                None,
                None,
                "The agent could not find a path. Please try again.",
                {**ERROR_STYLE, "display": "block"},
            )

        path_elements = []
        for i, step in enumerate(path):
            is_dest = i == len(path) - 1
            path_elements.append(
                html.Span(
                    step.replace("_", " "),
                    style=PATH_STEP_DEST_STYLE if is_dest else PATH_STEP_STYLE,
                )
            )
            if i < len(path) - 1:
                path_elements.append(html.Span("→", style=PATH_ARROW_STYLE))

        stats = f"{len(path) - 1} links traveled"

        return (
            {**SPINNER_STYLE, "display": "none"},
            {**RESULT_CONTAINER_STYLE, "display": "block"},
            path_elements,
            stats,
            None,
            {**ERROR_STYLE, "display": "none"},
        )

    except Exception:
        return (
            {**SPINNER_STYLE, "display": "none"},
            result_style,
            None,
            None,
            "The agent encountered an error. Please try again.",
            {**ERROR_STYLE, "display": "block"},
        )


if __name__ == "__main__":
    app.run(debug=True, port=8050)
