"""Build a markdown leaderboard scorecard from a saved eval JSON table.

Usage:
    uv run python -m evals.build_leaderboard_md run_20260520_10h46_0a4cea

Expects ``evals/leaderboards/<run_id>.json`` (as written by ``run_leaderboard --save``).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

LEADERBOARD_OUTPUT_DIR = Path(__file__).resolve().parent / "leaderboards"
AGENT_COLUMN = "Agent"

# Source column -> display name for the markdown table (insertion order = column order).
TABLE_COLUMNS: dict[str, str] = {
    AGENT_COLUMN: "Agent",
    "Reached destination rate": "Completion rate",
    "Cost (median)": "Run cost (median)",
}

# Source column -> sort ascending for the markdown table (insertion order = sort priority).
TABLE_SORT_ASCENDING: dict[str, bool] = {
    "Reached destination rate": False,
    "Cost (median)": True,
}

DEFAULT_PLOT_X = "Cost (median)"
DEFAULT_PLOT_Y = "Reached destination rate"
DEFAULT_PLOT_X_LABEL = "Run cost (median)"
DEFAULT_PLOT_Y_LABEL = "Completion rate"
DEFAULT_REVERSE_X_AXIS = True

VIBRANT_PALETTE = [
    "#FF4C4C",
    "#4287f5",
    "#22c55e",
    "#edbe2e",
    "#a637ea",
    "#FF8C00",
    "#E040FB",
    "#1DE9B6",
    "#536dfe",
    "#fb3586",
    "#31caff",
    "#f2711c",
    "#05c46b",
    "#686de0",
    "#fa8231",
]


def load_leaderboard_df(json_path: Path) -> pd.DataFrame:
    """Load a ``orient='split'`` leaderboard JSON and add an ``Agent`` column."""
    df = pd.read_json(json_path, orient="split")
    if df.index.name != "variant":
        df.index.name = "variant"
    return df.reset_index(names=AGENT_COLUMN)


def _require_columns(df: pd.DataFrame, columns: list[str], context: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        available = ", ".join(df.columns)
        raise ValueError(
            f"{context}: missing column(s) {missing!r}. Available: {available}"
        )


def make_table_markdown(df: pd.DataFrame, *, round_digits: int = 3) -> str:
    """Return a sorted markdown table for the scorecard."""
    source_columns = list(TABLE_COLUMNS.keys())
    sort_columns = list(TABLE_SORT_ASCENDING.keys())
    sort_ascending = list(TABLE_SORT_ASCENDING.values())

    unknown_sort = set(sort_columns) - set(source_columns)
    if unknown_sort:
        raise ValueError(
            f"TABLE_SORT_ASCENDING keys must appear in TABLE_COLUMNS: {unknown_sort!r}"
        )

    _require_columns(df, source_columns, "table")

    table_df = (
        df.sort_values(by=sort_columns, ascending=sort_ascending)[source_columns]
        .reset_index(drop=True)
        .round(round_digits)
        .rename(columns=TABLE_COLUMNS)
    )
    return table_df.to_markdown(index=False)


def save_leaderboard_plot(
    df: pd.DataFrame,
    output_path: Path,
    *,
    x: str = DEFAULT_PLOT_X,
    y: str = DEFAULT_PLOT_Y,
    reverse_x_axis: bool = DEFAULT_REVERSE_X_AXIS,
    reverse_y_axis: bool = False,
    x_label: str | None = DEFAULT_PLOT_X_LABEL,
    y_label: str | None = DEFAULT_PLOT_Y_LABEL,
) -> None:
    """Save a scatter plot of two leaderboard metrics."""
    _require_columns(df, [x, y], "plot")

    sns.set_theme(
        style="white",
        context="notebook",
        font_scale=1.2,
        rc={
            "axes.edgecolor": "#22223b",
            "axes.linewidth": 1.2,
            "axes.labelweight": "bold",
            "axes.labelsize": 14,
            "axes.titlesize": 16,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
        },
    )

    agent_count = df[AGENT_COLUMN].nunique() if AGENT_COLUMN in df.columns else len(df)
    repeated_palette = (VIBRANT_PALETTE * ((agent_count // len(VIBRANT_PALETTE)) + 1))[
        :agent_count
    ]

    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.grid(visible=False, axis="both")

    sns.scatterplot(
        data=df,
        x=x,
        y=y,
        s=270,
        marker="o",
        linewidth=2.0,
        edgecolor="#1a1a1a",
        hue=AGENT_COLUMN if AGENT_COLUMN in df.columns else None,
        palette=repeated_palette if AGENT_COLUMN in df.columns else VIBRANT_PALETTE,
        legend=False,
        alpha=1.0,
        ax=ax,
    )

    ax.grid(visible=True, linestyle="--", linewidth=0.5, alpha=1, axis="both")

    for _, row in df.iterrows():
        text = ax.text(
            row[x],
            row[y],
            str(row[AGENT_COLUMN]),
            fontsize=10,
            fontweight="medium",
            ha="left",
            va="bottom",
            color="#22223b",
            alpha=0.97,
            backgroundcolor="white",
            zorder=5,
        )
        text.set_bbox(
            dict(
                facecolor="white",
                edgecolor="none",
                boxstyle="round,pad=0.16",
                alpha=0.75,
            )
        )

    ax.set_xlabel(x_label if x_label is not None else x, fontsize=14, labelpad=10)
    ax.set_ylabel(y_label if y_label is not None else y, fontsize=14, labelpad=10)

    for spine in ["left", "bottom"]:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_color("#22223b")
        ax.spines[spine].set_linewidth(1.1)
    for spine in ["right", "top"]:
        ax.spines[spine].set_visible(False)

    if reverse_x_axis:
        ax.invert_xaxis()
    if reverse_y_axis:
        ax.invert_yaxis()

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def build_scorecard_markdown(
    *,
    title: str,
    table_markdown: str,
    plot_filename: str,
) -> str:
    """Combine table and plot reference into one markdown document."""
    return "\n".join(
        [
            f"# {title}",
            "",
            "## Leaderboard",
            "",
            table_markdown,
            "",
            "## Cost vs. completion",
            "",
            f"![{plot_filename}]({plot_filename})",
            "",
        ]
    )


def default_output_paths(json_path: Path) -> tuple[Path, Path]:
    """Derive ``.md`` and ``.png`` paths from the JSON stem in the same directory."""
    stem = json_path.stem
    out_dir = json_path.parent
    return out_dir / f"{stem}.md", out_dir / f"{stem}.png"


def build_leaderboard_scorecard(json_path: Path) -> tuple[Path, Path]:
    """Load JSON, write plot PNG and combined markdown scorecard."""
    out_md, out_plot = default_output_paths(json_path)

    df = load_leaderboard_df(json_path)
    if df.empty:
        raise ValueError(f"Leaderboard table is empty: {json_path}")

    scorecard_title = f"Leaderboard — {json_path.stem}"
    table_md = make_table_markdown(df)
    save_leaderboard_plot(df, out_plot)

    plot_filename = out_plot.name
    md_body = build_scorecard_markdown(
        title=scorecard_title,
        table_markdown=table_md,
        plot_filename=plot_filename,
    )
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(md_body, encoding="utf-8")

    return out_md, out_plot


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a markdown leaderboard scorecard from a saved eval JSON"
    )
    parser.add_argument(
        "run_id",
        help="Leaderboard run id (e.g. run_20260520_10h46_0a4cea)",
    )
    args = parser.parse_args()

    json_path = LEADERBOARD_OUTPUT_DIR / f"{args.run_id}.json"
    if not json_path.is_file():
        raise SystemExit(f"Leaderboard JSON not found: {json_path}")

    md_path, plot_path = build_leaderboard_scorecard(json_path)
    print(f"Wrote markdown scorecard: {md_path}")
    print(f"Wrote plot: {plot_path}")


if __name__ == "__main__":
    main()
