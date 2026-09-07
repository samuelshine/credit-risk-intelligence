"""Tests for EDA analysis functions, against a small deterministic dataset.

Deterministic rather than random: an insight's headline is asserted against an
exact expected number, which only works if the underlying data is fixed. The
adversarial/shape testing that random data is good for already lives in the
ETL and feature-engineering rehearsals (see docs/PROGRESS.md).
"""

from __future__ import annotations

import os

import pytest


@pytest.fixture(scope="module")
def db(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("eda")
    os.environ["DUCKDB_PATH"] = str(tmp / "t.duckdb")
    os.environ["DATA_DIR"] = str(tmp)
    os.environ["DUCKDB_MEMORY_LIMIT"] = "1GB"

    from src.utils.config import get_settings
    get_settings.cache_clear()

    from src.data.database import reset_readonly_connection, writable_connection
    reset_readonly_connection()

    with writable_connection() as conn:
        # 20 applicants: exactly 4 defaults (20% - deliberately high so small
        # counts give exact, easy-to-check fractions).
        conn.execute("""
            CREATE TABLE application_train AS
            SELECT
                100000 + i                                    AS SK_ID_CURR,
                (i % 5 = 0)::BIGINT                            AS TARGET,
                CASE WHEN i < 10 THEN 'M' WHEN i < 19 THEN 'F' ELSE 'XNA' END
                                                                AS CODE_GENDER,
                CASE WHEN i < 10 THEN -8000 ELSE 365243 END    AS DAYS_EMPLOYED,
                -(20 + i) * 365                                AS DAYS_BIRTH,
                50000.0 + i * 1000                             AS AMT_INCOME_TOTAL,
                100000.0 + i * 5000                            AS AMT_CREDIT,
                CASE WHEN i < 12 THEN 'Cash loans' ELSE 'Revolving loans' END
                                                                AS NAME_CONTRACT_TYPE,
                CASE WHEN i % 2 = 0 THEN 0.1 ELSE 0.9 END      AS EXT_SOURCE_2,
                CASE WHEN i < 15 THEN 1.5 ELSE NULL END        AS APARTMENTS_AVG
            FROM range(20) t(i)
        """)
        # Set income to NULL for one applicant, deliberately, to test the
        # missing-value report against an exact known count.
        conn.execute(
            "UPDATE application_train SET AMT_INCOME_TOTAL = NULL "
            "WHERE SK_ID_CURR = 100000"
        )
        conn.execute("""
            CREATE TABLE bureau AS
            SELECT
                100000 + i                                     AS SK_ID_CURR,
                5000000 + i                                    AS SK_ID_BUREAU,
                'Active'                                       AS CREDIT_ACTIVE,
                CASE WHEN i < 5 THEN 1000.0 ELSE 0.0 END       AS AMT_CREDIT_SUM_OVERDUE
            FROM range(15) t(i)
        """)
        conn.execute("""
            CREATE TABLE previous_application AS
            SELECT
                100000 + i                                     AS SK_ID_CURR,
                1000000 + i                                    AS SK_ID_PREV,
                CASE WHEN i < 5 THEN 'Refused' ELSE 'Approved' END
                                                                AS NAME_CONTRACT_STATUS
            FROM range(15) t(i)
        """)
        conn.execute("""
            CREATE TABLE installments_payments AS
            SELECT
                100000 + i                                     AS SK_ID_CURR,
                1000000 + i                                    AS SK_ID_PREV,
                -100.0                                          AS DAYS_INSTALMENT,
                CASE WHEN i < 5 THEN -80.0 ELSE -110.0 END     AS DAYS_ENTRY_PAYMENT
            FROM range(15) t(i)
        """)
    yield tmp
    reset_readonly_connection()


def test_dataset_summary_reports_every_table(db) -> None:
    from src.data.database import get_readonly_connection
    from src.eda.analysis import dataset_summary

    summary = dataset_summary(get_readonly_connection())
    names = {s.name for s in summary}
    assert names == {
        "application_train", "bureau", "previous_application",
        "installments_payments",
    }
    app = next(s for s in summary if s.name == "application_train")
    assert app.row_count == 20
    assert app.column_count == 10


def test_feature_categorisation_groups_columns_sensibly(db) -> None:
    from src.data.database import get_readonly_connection
    from src.eda.analysis import feature_categories

    grouped = feature_categories(get_readonly_connection())
    assert "SK_ID_CURR" in grouped["identifier"]
    assert "TARGET" in grouped["target"]
    assert "CODE_GENDER" in grouped["demographic"]
    assert "AMT_INCOME_TOTAL" in grouped["financial"]
    assert "EXT_SOURCE_2" in grouped["external_score"]
    assert "APARTMENTS_AVG" in grouped["housing_detail"]


def test_missing_value_report_matches_the_known_gap(db) -> None:
    from src.data.database import get_readonly_connection
    from src.eda.analysis import missing_value_report

    report = missing_value_report(
        get_readonly_connection(), tables=("application_train",)
    )
    income = next(r for r in report if r.column == "AMT_INCOME_TOTAL")
    assert income.n_missing == 1
    assert income.pct_missing == pytest.approx(1 / 20)

    apartments = next(r for r in report if r.column == "APARTMENTS_AVG")
    assert apartments.n_missing == 5  # rows 15..19
    assert report[0].pct_missing >= report[-1].pct_missing  # worst-first


def test_quality_findings_report_exact_known_values(db) -> None:
    from src.data.database import get_readonly_connection
    from src.eda.analysis import data_quality_findings

    findings = {f.id: f for f in data_quality_findings(get_readonly_connection())}

    assert findings["days_employed_sentinel"].value["count"] == 10
    assert findings["gender_xna"].value["count"] == 1
    assert findings["class_imbalance"].value["default_rate"] == pytest.approx(0.2)
    assert findings["class_imbalance"].severity == "action_needed"


def test_ext_source_insight_reflects_the_actual_split(db) -> None:
    from src.data.database import get_readonly_connection
    from src.eda.analysis import insight_default_by_ext_source

    insight = insight_default_by_ext_source(get_readonly_connection())
    assert insight.columns == ["decile", "applicants", "default_rate"]
    assert sum(r[1] for r in insight.rows) == 20
    assert "EXT_SOURCE_2" in insight.headline


def test_ext_source_lift_direction_is_not_inverted(isolated_db) -> None:
    """Regression test: an earlier version computed top-decile-rate /
    bottom-decile-rate (< 1, described as 'a 0.2x difference') instead of
    bottom-over-top, which understates a 6x risk gap as a 0.2x one. Uses a
    dedicated fixture where the true ratio is a known, checkable value.

    `isolated_db` (see conftest.py) gives a private database and restores all
    global config/connection state afterward, so this cannot leak into the
    other tests in this module that rely on the shared `db` fixture.
    """
    conn = isolated_db
    # 100 applicants, EXT_SOURCE_2 = i/100 (spreads evenly across deciles).
    # Bottom decile (lowest score, i<10) defaults 40%; top decile (i>=90)
    # defaults 10% - a known, exact 4x gap, bottom over top.
    conn.execute("""
        CREATE TABLE application_train AS
        SELECT i AS SK_ID_CURR, i / 100.0 AS EXT_SOURCE_2,
               (CASE WHEN i < 10 THEN i % 5 < 2
                     WHEN i >= 90 THEN i % 10 = 0
                     ELSE i % 4 = 0 END)::BIGINT AS TARGET
        FROM range(100) t(i)
    """)

    from src.eda.analysis import insight_default_by_ext_source
    insight = insight_default_by_ext_source(conn)

    bottom_rate = next(r[2] for r in insight.rows if r[0] == 1)
    top_rate = next(r[2] for r in insight.rows if r[0] == 10)
    assert bottom_rate > top_rate  # sanity: the fixture is set up as intended

    # The headline must state the risk gap the right way round: bottom over
    # top (>1x, "more risk"), never top over bottom (<1x).
    import re
    match = re.search(r"([\d.]+)x the risk", insight.headline)
    assert match, insight.headline
    stated_lift = float(match.group(1))
    assert stated_lift == pytest.approx(bottom_rate / top_rate, rel=0.01)
    assert stated_lift > 1.0


def test_loan_burden_insight_finds_the_true_peak_not_the_endpoints(
    isolated_db,
) -> None:
    """Regression test: an earlier version assumed quintile 1 and quintile 5
    were the safest/riskiest, which silently hid a non-monotonic
    relationship where the middle quintile was actually riskiest."""
    conn = isolated_db
    # AMT_CREDIT increases monotonically with i, so ntile(5) on the resulting
    # ratio puts i in [0,19] in quintile 1, [40,59] in quintile 3, [80,99] in
    # quintile 5. TARGET=1 only for i in [45,54] - squarely inside quintile 3
    # and nowhere else - so quintile 3 defaults most, 1 and 5 default zero.
    conn.execute("""
        CREATE TABLE application_train AS
        SELECT i AS SK_ID_CURR,
               100000.0 AS AMT_INCOME_TOTAL,
               (1 + i) * 20000.0 AS AMT_CREDIT,
               (i BETWEEN 45 AND 54)::BIGINT AS TARGET
        FROM range(100) t(i)
    """)

    from src.eda.analysis import insight_loan_burden_vs_default
    insight = insight_loan_burden_vs_default(conn)

    riskiest_quintile = max(insight.rows, key=lambda r: r[3])[0]
    assert riskiest_quintile == 3  # the middle, not quintile 1 or 5
    assert "quintile 3" in insight.headline.lower()
    assert "non-linear" in insight.title.lower()


def test_bureau_overdue_insight_finds_the_exact_split(db) -> None:
    from src.data.database import get_readonly_connection
    from src.eda.analysis import insight_bureau_history_vs_default

    insight = insight_bureau_history_vs_default(get_readonly_connection())
    by_status = {row[0]: row for row in insight.rows}
    assert by_status["has overdue bureau credit"][1] == 5
    assert by_status["no overdue bureau credit"][1] == 10  # clients 105..114


def test_previous_refusal_insight_finds_the_exact_split(db) -> None:
    from src.data.database import get_readonly_connection
    from src.eda.analysis import insight_previous_refusal_vs_default

    insight = insight_previous_refusal_vs_default(get_readonly_connection())
    by_history = {row[0]: row for row in insight.rows}
    assert by_history["previously refused"][1] == 5
    assert by_history["never refused"][1] == 10


def test_installment_lateness_insight_finds_the_exact_split(db) -> None:
    from src.data.database import get_readonly_connection
    from src.eda.analysis import insight_installment_lateness_vs_default

    insight = insight_installment_lateness_vs_default(get_readonly_connection())
    by_behaviour = {row[0]: row for row in insight.rows}
    # DAYS_ENTRY_PAYMENT(-110) - DAYS_INSTALMENT(-100) = -10 (early) for i>=5;
    # -80 - -100 = +20 (late) for i<5.
    assert by_behaviour["often pay late"][1] == 5
    assert by_behaviour["pay on time or early"][1] == 10


def test_run_all_insights_skips_missing_tables_gracefully(db) -> None:
    """credit_card_balance and pos_cash_balance don't exist in this fixture -
    the run must skip whatever needs them rather than raising."""
    from src.data.database import get_readonly_connection
    from src.eda.analysis import run_all_insights

    insights = run_all_insights(get_readonly_connection())
    ids = {i.id for i in insights}
    assert "ext_source_decile" in ids
    assert len(insights) >= 6


def test_build_eda_artifacts_is_json_serialisable(db) -> None:
    from src.data.database import get_readonly_connection
    from src.eda.analysis import build_eda_artifacts
    from src.utils.helpers import to_jsonable
    import json

    artifacts = build_eda_artifacts(get_readonly_connection())
    # Round-trips through JSON without error - this is what write_json does.
    json.dumps(to_jsonable(artifacts))
    assert artifacts["insights"]
    assert artifacts["table_summary"]
    assert artifacts["data_quality_findings"]
