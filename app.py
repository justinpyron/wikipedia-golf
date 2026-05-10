"""Wikipedia Golf Dash Frontend

A clean, minimal web app for playing Wikipedia Golf with an AI agent.
Aesthetic: Augusta-inspired (whisper white, Augusta green, championship gold).
"""

import asyncio
from typing import Any

import dash
from dash import Dash, Input, Output, State, callback, dcc, html
from dash.exceptions import PreventUpdate
from dotenv import load_dotenv
from pydantic_ai import UsageLimits

from agent import WikiGolfDeps, build_agent
from variants import VARIANTS
from wiki import ArticleSearchResult, find_articles

load_dotenv()

# Maximum number of tool calls (page visits) allowed per game
MAX_TOOL_CALLS = 20

# Maximum number of LLM requests (model turns) allowed per game
MAX_LLM_REQUESTS = 30

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
    "maxWidth": "680px",
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

SECTION_STYLE = {
    "marginBottom": "32px",
}

SECTION_LABEL_STYLE = {
    "fontSize": "12px",
    "fontWeight": "600",
    "color": COLORS["slate"],
    "textTransform": "uppercase",
    "letterSpacing": "0.1em",
    "marginBottom": "12px",
}

INPUT_STYLE = {
    "width": "100%",
    "padding": "14px 16px",
    "fontSize": "16px",
    "border": f"1px solid {COLORS['mist']}",
    "borderRadius": "4px",
    "backgroundColor": COLORS["white"],
    "color": COLORS["charcoal"],
    "outline": "none",
    "boxSizing": "border-box",
    "transition": "border-color 0.2s ease",
}

INPUT_FOCUS_STYLE = {
    "borderColor": COLORS["augusta_green"],
}

RESULTS_CONTAINER_STYLE = {
    "display": "flex",
    "gap": "12px",
    "marginTop": "16px",
    "flexWrap": "wrap",
}

RESULT_CARD_STYLE = {
    "flex": "1",
    "minWidth": "120px",
    "maxWidth": "calc(20% - 10px)",
    "padding": "12px",
    "backgroundColor": COLORS["white"],
    "border": f"1px solid {COLORS['mist']}",
    "borderRadius": "4px",
    "cursor": "pointer",
    "transition": "all 0.2s ease",
}

RESULT_CARD_HOVER_STYLE = {
    "borderColor": COLORS["augusta_green"],
    "boxShadow": "0 2px 8px rgba(0,0,0,0.04)",
}

RESULT_TITLE_STYLE = {
    "fontSize": "13px",
    "fontWeight": "500",
    "color": COLORS["charcoal"],
    "margin": "0 0 4px 0",
    "lineHeight": "1.3",
    "overflow": "hidden",
    "textOverflow": "ellipsis",
    "whiteSpace": "nowrap",
}

RESULT_DESC_STYLE = {
    "fontSize": "11px",
    "color": COLORS["slate"],
    "margin": "0",
    "lineHeight": "1.3",
    "overflow": "hidden",
    "textOverflow": "ellipsis",
    "display": "-webkit-box",
    "WebkitLineClamp": "2",
    "WebkitBoxOrient": "vertical",
}

PREVIEW_CONTAINER_STYLE = {
    "display": "flex",
    "gap": "24px",
    "marginTop": "32px",
    "marginBottom": "32px",
    "justifyContent": "center",
}

PREVIEW_CARD_STYLE = {
    "flex": "1",
    "maxWidth": "300px",
    "padding": "20px",
    "backgroundColor": COLORS["white"],
    "border": f"1px solid {COLORS['mist']}",
    "borderRadius": "4px",
    "textAlign": "center",
}

PREVIEW_CARD_DEST_STYLE = {
    **PREVIEW_CARD_STYLE,
    "borderColor": COLORS["championship_gold"],
}

PREVIEW_THUMBNAIL_STYLE = {
    "width": "80px",
    "height": "80px",
    "objectFit": "cover",
    "borderRadius": "4px",
    "marginBottom": "12px",
    "border": f"1px solid {COLORS['light_mist']}",
}

PREVIEW_THUMBNAIL_PLACEHOLDER_STYLE = {
    "width": "80px",
    "height": "80px",
    "backgroundColor": COLORS["light_mist"],
    "borderRadius": "4px",
    "marginBottom": "12px",
    "marginLeft": "auto",
    "marginRight": "auto",
    "display": "flex",
    "alignItems": "center",
    "justifyContent": "center",
    "fontSize": "24px",
    "color": COLORS["slate"],
}

PREVIEW_TITLE_STYLE = {
    "fontSize": "16px",
    "fontWeight": "600",
    "color": COLORS["charcoal"],
    "margin": "0 0 8px 0",
    "lineHeight": "1.3",
}

PREVIEW_DESC_STYLE = {
    "fontSize": "13px",
    "color": COLORS["slate"],
    "margin": "0",
    "lineHeight": "1.4",
}

PREVIEW_KEY_STYLE = {
    "fontSize": "11px",
    "fontFamily": "'JetBrains Mono', monospace",
    "color": COLORS["slate"],
    "backgroundColor": COLORS["light_mist"],
    "padding": "4px 8px",
    "borderRadius": "3px",
    "marginTop": "12px",
    "display": "inline-block",
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
    "marginTop": "24px",
}

BUTTON_DISABLED_STYLE = {
    **BUTTON_STYLE,
    "backgroundColor": COLORS["mist"],
    "color": COLORS["slate"],
    "cursor": "not-allowed",
}

BUTTON_HOVER_STYLE = {
    "backgroundColor": COLORS["billiard_green"],
}

SPINNER_STYLE = {
    "textAlign": "center",
    "padding": "48px 0",
    "color": COLORS["augusta_green"],
    "fontSize": "14px",
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
    "marginTop": "12px",
    "padding": "12px 16px",
    "backgroundColor": "#FEF2F2",
    "border": "1px solid #FECACA",
    "borderRadius": "4px",
    "color": "#DC2626",
    "fontSize": "13px",
}

DIVIDER_STYLE = {
    "height": "1px",
    "backgroundColor": COLORS["mist"],
    "margin": "48px 0",
    "border": "none",
}


def create_article_card(
    result: ArticleSearchResult, index: int, prefix: str
) -> html.Div:
    """Create a clickable article suggestion card."""
    # Use dict-style ID for pattern-matching callbacks
    card_id = {"type": f"{prefix}-card", "index": index}
    # Store data in data-* attributes (valid HTML/Dash props)
    thumbnail_url = result.thumbnail.get("url") if result.thumbnail else ""
    return html.Div(
        [
            html.Div(
                result.title,
                style=RESULT_TITLE_STYLE,
                title=result.title,
            ),
            html.Div(
                result.description or "No description available",
                style=RESULT_DESC_STYLE,
            )
            if result.description
            else None,
        ],
        id=card_id,
        style=RESULT_CARD_STYLE,
        n_clicks=0,
        **{
            "data-key": result.key,
            "data-title": result.title,
            "data-description": result.description or "",
            "data-thumbnail": thumbnail_url,
        },
    )


def create_preview_card(data: dict | None, is_destination: bool = False) -> html.Div:
    """Create a preview card for a selected article."""
    if data is None:
        return html.Div(
            [
                html.Div("?", style=PREVIEW_THUMBNAIL_PLACEHOLDER_STYLE),
                html.Div(
                    "Destination" if is_destination else "Origin",
                    style=PREVIEW_TITLE_STYLE,
                ),
                html.Div("Not selected", style=PREVIEW_DESC_STYLE),
            ],
            style=PREVIEW_CARD_STYLE,
        )

    thumbnail_url = data.get("thumbnail")
    thumbnail = (
        html.Img(src=thumbnail_url, style=PREVIEW_THUMBNAIL_STYLE)
        if thumbnail_url
        else html.Div("📄", style=PREVIEW_THUMBNAIL_PLACEHOLDER_STYLE)
    )

    return html.Div(
        [
            thumbnail,
            html.Div(data.get("title", ""), style=PREVIEW_TITLE_STYLE),
            html.Div(
                data.get("description") or "No description available",
                style=PREVIEW_DESC_STYLE,
            ),
            html.Div(data.get("key", ""), style=PREVIEW_KEY_STYLE),
        ],
        style=PREVIEW_CARD_DEST_STYLE if is_destination else PREVIEW_CARD_STYLE,
    )


app.layout = html.Div(
    [
        # Header
        html.Div(
            [
                html.H1("W I K I P E D I A   G O L F", style=TITLE_STYLE),
                html.P("Navigate from any article to any other", style=SUBTITLE_STYLE),
            ],
            style=HEADER_STYLE,
        ),
        # Origin Section
        html.Div(
            [
                html.Div("From", style=SECTION_LABEL_STYLE),
                dcc.Input(
                    id="origin-input",
                    type="text",
                    placeholder="Search for starting article...",
                    style=INPUT_STYLE,
                    debounce=True,
                ),
                html.Div(id="origin-results-container", style=RESULTS_CONTAINER_STYLE),
                html.Div(id="origin-error", style=ERROR_STYLE),
            ],
            style=SECTION_STYLE,
        ),
        # Destination Section
        html.Div(
            [
                html.Div("To", style=SECTION_LABEL_STYLE),
                dcc.Input(
                    id="dest-input",
                    type="text",
                    placeholder="Search for destination article...",
                    style=INPUT_STYLE,
                    debounce=True,
                ),
                html.Div(id="dest-results-container", style=RESULTS_CONTAINER_STYLE),
                html.Div(id="dest-error", style=ERROR_STYLE),
            ],
            style=SECTION_STYLE,
        ),
        # Preview Section
        html.Div(
            [
                create_preview_card(None, False),
                html.Div(
                    "→",
                    style={
                        "fontSize": "24px",
                        "color": COLORS["slate"],
                        "alignSelf": "center",
                    },
                ),
                create_preview_card(None, True),
            ],
            id="preview-container",
            style=PREVIEW_CONTAINER_STYLE,
        ),
        # Hidden stores for selected articles
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


@callback(
    Output("origin-results-container", "children"),
    Output("origin-error", "children"),
    Output("origin-error", "style"),
    Output("origin-data", "data", allow_duplicate=True),
    Input("origin-input", "value"),
    prevent_initial_call=True,
)
def update_origin_results(query: str | None) -> tuple:
    """Search for articles when origin input changes."""
    if not query or len(query) < 2:
        raise PreventUpdate

    try:
        results = find_articles(query, limit=5)
        if not results:
            return (
                None,
                f'No articles found matching "{query}"',
                {**ERROR_STYLE, "display": "block"},
                None,  # Clear selection
            )

        cards = [create_article_card(r, i, "origin") for i, r in enumerate(results)]
        return (
            cards,
            None,
            {**ERROR_STYLE, "display": "none"},
            None,
        )  # Clear selection on new search

    except Exception as e:
        return (
            None,
            "Unable to search. Please try again.",
            {**ERROR_STYLE, "display": "block"},
            None,  # Clear selection
        )


@callback(
    Output("dest-results-container", "children"),
    Output("dest-error", "children"),
    Output("dest-error", "style"),
    Output("dest-data", "data", allow_duplicate=True),
    Input("dest-input", "value"),
    prevent_initial_call=True,
)
def update_dest_results(query: str | None) -> tuple:
    """Search for articles when destination input changes."""
    if not query or len(query) < 2:
        raise PreventUpdate

    try:
        results = find_articles(query, limit=5)
        if not results:
            return (
                None,
                f'No articles found matching "{query}"',
                {**ERROR_STYLE, "display": "block"},
                None,  # Clear selection
            )

        cards = [create_article_card(r, i, "dest") for i, r in enumerate(results)]
        return (
            cards,
            None,
            {**ERROR_STYLE, "display": "none"},
            None,
        )  # Clear selection on new search

    except Exception as e:
        return (
            None,
            "Unable to search. Please try again.",
            {**ERROR_STYLE, "display": "block"},
            None,  # Clear selection
        )


@callback(
    Output("origin-data", "data"),
    Output("origin-results-container", "style", allow_duplicate=True),
    Output("origin-input", "value", allow_duplicate=True),
    Input({"type": "origin-card", "index": dash.ALL}, "n_clicks"),
    State({"type": "origin-card", "index": dash.ALL}, "data-key"),
    State({"type": "origin-card", "index": dash.ALL}, "data-title"),
    State({"type": "origin-card", "index": dash.ALL}, "data-description"),
    State({"type": "origin-card", "index": dash.ALL}, "data-thumbnail"),
    prevent_initial_call=True,
)
def select_origin(
    n_clicks: list[int | None],
    keys: list[str | None],
    titles: list[str | None],
    descriptions: list[str | None],
    thumbnails: list[str | None],
) -> tuple:
    """Handle origin article selection."""
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate

    # Find which card was clicked (n_clicks > 0)
    for i, clicks in enumerate(n_clicks):
        if clicks and clicks > 0:
            data = {
                "key": keys[i],
                "title": titles[i],
                "description": descriptions[i],
                "thumbnail": thumbnails[i],
            }
            return (
                data,
                {**RESULTS_CONTAINER_STYLE, "display": "none"},
                titles[i] or "",
            )

    raise PreventUpdate


@callback(
    Output("dest-data", "data"),
    Output("dest-results-container", "style", allow_duplicate=True),
    Output("dest-input", "value", allow_duplicate=True),
    Input({"type": "dest-card", "index": dash.ALL}, "n_clicks"),
    State({"type": "dest-card", "index": dash.ALL}, "data-key"),
    State({"type": "dest-card", "index": dash.ALL}, "data-title"),
    State({"type": "dest-card", "index": dash.ALL}, "data-description"),
    State({"type": "dest-card", "index": dash.ALL}, "data-thumbnail"),
    prevent_initial_call=True,
)
def select_dest(
    n_clicks: list[int | None],
    keys: list[str | None],
    titles: list[str | None],
    descriptions: list[str | None],
    thumbnails: list[str | None],
) -> tuple:
    """Handle destination article selection."""
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate

    # Find which card was clicked (n_clicks > 0)
    for i, clicks in enumerate(n_clicks):
        if clicks and clicks > 0:
            data = {
                "key": keys[i],
                "title": titles[i],
                "description": descriptions[i],
                "thumbnail": thumbnails[i],
            }
            return (
                data,
                {**RESULTS_CONTAINER_STYLE, "display": "none"},
                titles[i] or "",
            )

    raise PreventUpdate


@callback(
    Output("preview-container", "children"),
    Input("origin-data", "data"),
    Input("dest-data", "data"),
)
def update_preview(origin_data: dict | None, dest_data: dict | None) -> list:
    """Update the preview cards when selections change."""
    return [
        create_preview_card(origin_data, False),
        html.Div(
            "→",
            style={"fontSize": "24px", "color": COLORS["slate"], "alignSelf": "center"},
        ),
        create_preview_card(dest_data, True),
    ]


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

    # Show spinner
    loading_style = {**SPINNER_STYLE, "display": "block"}
    result_style = {**RESULT_CONTAINER_STYLE, "display": "none"}

    try:
        origin_key = origin_data.get("key")
        dest_key = dest_data.get("key")

        # Use default variant
        variant = VARIANTS[0]
        agent = build_agent(variant)

        # Run the agent with usage limits
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

        result = asyncio.run(run())
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

        # Build path display
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

    except Exception as e:
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
