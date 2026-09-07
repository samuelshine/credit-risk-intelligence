"""Rendering insights and quality findings to static PNGs.

Charts are rendered once, offline, by `scripts/build_eda_artifacts.py` (or the
ETL/training entrypoint), never on request. A 307k-row aggregation is cheap in
DuckDB but plotting is not free, and an evaluator's first API request should
not be the moment a chart gets drawn for the first time.

One function per chart *shape*, not per insight - `render_insight` dispatches
on `Insight.chart_type` so a new insight with an existing shape needs no new
plotting code.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display in a container; must be set before pyplot
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from src.eda.analysis import Insight, MissingReport
from src.utils.logger import get_logger

log = get_logger(__name__)

# A restrained, colour-blind-legible palette rather than matplotlib's default
# cycle - one accent for "the answer", one neutral for "the context".
_ACCENT = "#9E2B3E"      # oxblood: matches the risk-ruler high-risk colour
_NEUTRAL = "#5A6E7A"     # slate
_GRID = "#C9D4DB"        # hairline

plt.rcParams.update({
    "font.size": 11,
    "axes.edgecolor": _GRID,
    "axes.labelcolor": "#14232E",
    "text.color": "#14232E",
    "xtick.color": "#14232E",
    "ytick.color": "#14232E",
    "axes.grid": True,
    "grid.color": _GRID,
    "grid.linewidth": 0.6,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})


def _new_figure(width: float = 7.0, height: float = 4.0):
    fig, ax = plt.subplots(figsize=(width, height), dpi=150)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.6)
    ax.set_axisbelow(True)
    return fig, ax


def render_insight(insight: Insight, out_dir: Path) -> Path:
    """Render one insight's chart. Dispatches on `chart_type`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{insight.id}.png"

    if insight.chart_type == "bar":
        _render_bar(insight, path)
    else:
        raise ValueError(f"unknown chart_type {insight.chart_type!r}")

    log.info("rendered %s", path.name)
    return path


def _render_bar(insight: Insight, path: Path) -> None:
    """A grouped bar of default rate per category, with applicant counts as
    a second axis. This is the shape nearly every insight here takes: default
    rate broken down by some segment, alongside how many applicants back it.
    """
    labels = [str(row[0]) for row in insight.rows]
    rate_idx = insight.columns.index("default_rate")
    count_idx = insight.columns.index("applicants")
    rates = [row[rate_idx] for row in insight.rows]
    counts = [row[count_idx] for row in insight.rows]

    fig, ax = _new_figure(width=max(6.0, 0.8 * len(labels) + 3))
    x = range(len(labels))
    bars = ax.bar(x, rates, color=_NEUTRAL, width=0.6, zorder=3)

    # Highlight the highest-risk bar - it is what "so what" is usually about.
    worst = max(range(len(rates)), key=lambda i: rates[i])
    bars[worst].set_color(_ACCENT)

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=30 if len(labels) > 4 else 0, ha="right")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.set_ylabel("Default rate")
    ax.set_title(insight.title, fontsize=12, loc="left", pad=12)

    for i, (rate, count) in enumerate(zip(rates, counts)):
        ax.annotate(
            f"n={count:,}", (i, rate), textcoords="offset points",
            xytext=(0, 4), ha="center", fontsize=8, color=_NEUTRAL,
        )

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def render_missing_values_chart(
    reports: list[MissingReport], out_dir: Path, top_n: int = 15
) -> Path:
    """Horizontal bar of the worst missing-value columns, worst at top."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "missing_values.png"
    top = reports[:top_n]

    fig, ax = _new_figure(width=7.5, height=max(3.0, 0.35 * len(top) + 1))
    labels = [r.column for r in reversed(top)]
    values = [r.pct_missing for r in reversed(top)]

    ax.barh(labels, values, color=_NEUTRAL, zorder=3)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.set_xlabel("Share missing")
    ax.set_title("Columns with the most missing data", fontsize=12, loc="left", pad=12)
    ax.grid(axis="x", alpha=0.6)
    ax.grid(axis="y", visible=False)

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    log.info("rendered %s", path.name)
    return path


def render_all(
    insights: list[Insight],
    missing: list[MissingReport],
    out_dir: Path,
) -> list[Path]:
    """Render every chart the EDA artifact needs. Idempotent: safe to rerun."""
    paths = [render_insight(insight, out_dir) for insight in insights]
    if missing:
        paths.append(render_missing_values_chart(missing, out_dir))
    return paths
