"""Charts must render as valid images without a display. Not pixel-testing the
plots (fragile, low value) - just that the pipeline produces a real PNG for
every insight shape and does not crash headless.
"""

from __future__ import annotations

from pathlib import Path

from src.eda.analysis import Insight, MissingReport


def test_bar_insight_renders_a_valid_png(tmp_path: Path) -> None:
    from src.eda.charts import render_insight

    insight = Insight(
        id="test_insight", title="Test insight",
        business_question="Does X relate to Y?", chart_type="bar",
        columns=["segment", "applicants", "default_rate"],
        rows=[("A", 100, 0.05), ("B", 200, 0.15), ("C", 50, 0.02)],
        headline="B defaults most, at 15.0%.", so_what="Because reasons.",
    )
    path = render_insight(insight, tmp_path)
    assert path.exists()
    assert path.read_bytes().startswith(b"\x89PNG")
    assert path.stat().st_size > 1000


def test_unknown_chart_type_raises(tmp_path: Path) -> None:
    import pytest
    from src.eda.charts import render_insight

    insight = Insight(
        id="bad", title="t", business_question="q", chart_type="pie",
        columns=[], rows=[], headline="h", so_what="s",
    )
    with pytest.raises(ValueError, match="unknown chart_type"):
        render_insight(insight, tmp_path)


def test_missing_values_chart_renders(tmp_path: Path) -> None:
    from src.eda.charts import render_missing_values_chart

    reports = [
        MissingReport("application_train", f"COL_{i}", "other", 0.9 - i * 0.05, 90, 100)
        for i in range(5)
    ]
    path = render_missing_values_chart(reports, tmp_path)
    assert path.exists()
    assert path.read_bytes().startswith(b"\x89PNG")


def test_render_all_produces_one_file_per_insight_plus_missing(tmp_path: Path) -> None:
    from src.eda.charts import render_all

    insights = [
        Insight(id=f"i{n}", title="t", business_question="q", chart_type="bar",
                columns=["s", "applicants", "default_rate"],
                rows=[("A", 10, 0.1)], headline="h", so_what="s")
        for n in range(3)
    ]
    missing = [MissingReport("t", "c", "other", 0.5, 5, 10)]
    paths = render_all(insights, missing, tmp_path)
    assert len(paths) == 4
    assert all(p.exists() for p in paths)
