"""Exploratory analysis, as a library of pure functions over DuckDB.

Every function here takes a connection and returns data - never a string of
prose with numbers baked in. The headline sentence for each insight is
composed from the returned numbers by `build_eda_artifacts`, so a number in
the UI or in `docs/EDA_FINDINGS.md` can never drift from what the query
actually returned.

This module is imported by three things that must never disagree: the API's
`/api/eda/*` routes, `notebooks/eda.py`, and `docs/EDA_FINDINGS.md`'s
generation script. One set of queries, three presentations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import duckdb

from src.data.database import get_readonly_connection, list_tables, table_row_counts
from src.utils.logger import get_logger, log_duration

log = get_logger(__name__)

# --------------------------------------------------------------------------- #
# Dataset summary & feature categorisation
# --------------------------------------------------------------------------- #

#: Business grouping of application_train's columns. Built from the dataset's
#: own naming conventions and the Kaggle glossary, not guessed per-column.
#: A column not matched by any pattern below falls into "other".
FEATURE_CATEGORY_PATTERNS: dict[str, tuple[str, ...]] = {
    "identifier": ("SK_ID_CURR", "SK_ID_BUREAU", "SK_ID_PREV"),
    "target": ("TARGET",),
    "demographic": (
        "CODE_GENDER", "DAYS_BIRTH", "CNT_CHILDREN", "CNT_FAM_MEMBERS",
        "NAME_FAMILY_STATUS", "NAME_EDUCATION_TYPE", "NAME_HOUSING_TYPE",
        "OCCUPATION_TYPE", "ORGANIZATION_TYPE",
    ),
    "financial": (
        "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE",
        "NAME_INCOME_TYPE", "NAME_CONTRACT_TYPE", "FLAG_OWN_CAR",
        "FLAG_OWN_REALTY", "OWN_CAR_AGE",
    ),
    "employment": ("DAYS_EMPLOYED", "DAYS_REGISTRATION", "DAYS_ID_PUBLISH"),
    "external_score": ("EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"),
    "credit_history": (
        "CREDIT_ACTIVE", "CREDIT_TYPE", "AMT_CREDIT_SUM", "AMT_CREDIT_SUM_DEBT",
        "AMT_CREDIT_SUM_OVERDUE", "CREDIT_DAY_OVERDUE", "DAYS_CREDIT",
        "NAME_CONTRACT_STATUS", "CODE_REJECT_REASON",
    ),
    "repayment_behaviour": (
        "DAYS_INSTALMENT", "DAYS_ENTRY_PAYMENT", "AMT_INSTALMENT",
        "AMT_PAYMENT", "SK_DPD", "SK_DPD_DEF", "STATUS", "MONTHS_BALANCE",
    ),
    "document_flag": ("FLAG_DOCUMENT",),
    "housing_detail": (
        "APARTMENTS", "BASEMENTAREA", "YEARS_BEGINEXPLUATATION", "YEARS_BUILD",
        "COMMONAREA", "ELEVATORS", "ENTRANCES", "FLOORSMAX", "FLOORSMIN",
        "LANDAREA", "LIVINGAPARTMENTS", "LIVINGAREA", "NONLIVINGAPARTMENTS",
        "NONLIVINGAREA",
    ),
    "region": ("REGION_", "REG_REGION", "REG_CITY", "LIVE_REGION", "LIVE_CITY"),
    "contact_flag": (
        "FLAG_MOBIL", "FLAG_EMP_PHONE", "FLAG_WORK_PHONE", "FLAG_CONT_MOBILE",
        "FLAG_PHONE", "FLAG_EMAIL",
    ),
}


def categorise_column(column: str) -> str:
    """Which business category a column belongs to, by name pattern."""
    upper = column.upper()
    for category, patterns in FEATURE_CATEGORY_PATTERNS.items():
        if any(upper == p or upper.startswith(p) for p in patterns):
            return category
    return "other"


@dataclass
class TableSummary:
    name: str
    description: str
    row_count: int
    column_count: int
    approx_size_mb: float


def dataset_summary(conn: duckdb.DuckDBPyConnection | None = None) -> list[TableSummary]:
    """One row per table: what it is, how big it is."""
    from src.talk_to_data.catalog import TABLE_DESCRIPTIONS

    conn = conn or get_readonly_connection()
    tables = [t for t in list_tables(conn) if not t.startswith("_")]
    summaries = []
    for name in tables:
        n_cols = conn.execute(
            "SELECT count(*) FROM information_schema.columns "
            "WHERE table_schema='main' AND table_name=?", [name],
        ).fetchone()[0]
        n_rows = conn.execute(f'SELECT count(*) FROM "{name}"').fetchone()[0]
        # DuckDB has no direct per-table byte size for CSV-origin tables;
        # rows x columns x 8 bytes is a rough but consistent proxy, good
        # enough to compare tables' relative weight to a reader.
        approx_mb = (n_rows * n_cols * 8) / 1e6
        summaries.append(TableSummary(
            name=name, description=TABLE_DESCRIPTIONS.get(name, ""),
            row_count=n_rows, column_count=n_cols, approx_size_mb=approx_mb,
        ))
    return sorted(summaries, key=lambda s: -s.row_count)


def feature_categories(
    conn: duckdb.DuckDBPyConnection | None = None, table: str = "application_train"
) -> dict[str, list[str]]:
    """Group one table's columns into business categories."""
    conn = conn or get_readonly_connection()
    columns = [
        r[0] for r in conn.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema='main' AND table_name=? ORDER BY ordinal_position",
            [table],
        ).fetchall()
    ]
    grouped: dict[str, list[str]] = {}
    for column in columns:
        grouped.setdefault(categorise_column(column), []).append(column)
    return grouped


@dataclass
class MissingReport:
    table: str
    column: str
    category: str
    pct_missing: float
    n_missing: int
    n_total: int


def missing_value_report(
    conn: duckdb.DuckDBPyConnection | None = None,
    tables: tuple[str, ...] = ("application_train",),
) -> list[MissingReport]:
    """Missingness per column, worst first.

    Built with one query per table using DuckDB's `COLUMNS(*)` expression
    rather than one query per column - 122 columns is 122x fewer round trips.
    """
    conn = conn or get_readonly_connection()
    reports: list[MissingReport] = []

    for table in tables:
        n_total = conn.execute(f'SELECT count(*) FROM "{table}"').fetchone()[0]
        if n_total == 0:
            continue
        columns = [
            r[0] for r in conn.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_schema='main' AND table_name=?", [table],
            ).fetchall()
        ]
        select_list = ", ".join(
            f'count(*) - count("{c}") AS "{c}"' for c in columns
        )
        row = conn.execute(
            f'SELECT {select_list} FROM "{table}"'
        ).fetchone()

        for column, n_missing in zip(columns, row):
            if n_missing == 0:
                continue
            reports.append(MissingReport(
                table=table, column=column, category=categorise_column(column),
                pct_missing=n_missing / n_total, n_missing=n_missing, n_total=n_total,
            ))

    return sorted(reports, key=lambda r: -r.pct_missing)


@dataclass
class QualityFinding:
    id: str
    description: str
    value: Any
    severity: str  # "info" | "notable" | "action_needed"


def data_quality_findings(
    conn: duckdb.DuckDBPyConnection | None = None,
) -> list[QualityFinding]:
    """Known and suspected data-quality issues, checked against the live data.

    Each finding states what was checked and what was found - the values here
    are queried fresh every time, never hard-coded, so a finding is only ever
    as stale as the database it ran against.
    """
    conn = conn or get_readonly_connection()
    findings: list[QualityFinding] = []
    cols = _columns(conn, "application_train")
    n_total = conn.execute("SELECT count(*) FROM application_train").fetchone()[0]
    if n_total == 0:
        return findings

    if "DAYS_EMPLOYED" in cols:
        n = conn.execute(
            "SELECT count(*) FROM application_train WHERE DAYS_EMPLOYED = 365243"
        ).fetchone()[0]
        findings.append(QualityFinding(
            "days_employed_sentinel",
            "DAYS_EMPLOYED uses 365243 (1000 years) as a sentinel for "
            "'not employed' rather than a true value.",
            {"count": n, "pct": round(n / n_total, 4)}, "action_needed",
        ))

    if "CODE_GENDER" in cols:
        n = conn.execute(
            "SELECT count(*) FROM application_train WHERE CODE_GENDER = 'XNA'"
        ).fetchone()[0]
        findings.append(QualityFinding(
            "gender_xna", "CODE_GENDER contains an 'XNA' (unknown) category.",
            {"count": n, "pct": round(n / max(n_total, 1), 6)},
            "notable" if n else "info",
        ))

    if "AMT_INCOME_TOTAL" in cols:
        row = conn.execute(
            "SELECT max(AMT_INCOME_TOTAL), median(AMT_INCOME_TOTAL), "
            "       avg(AMT_INCOME_TOTAL) "
            "FROM application_train"
        ).fetchone()
        max_income, median_income, mean_income = row
        findings.append(QualityFinding(
            "income_outlier",
            "AMT_INCOME_TOTAL's maximum is far above its median, indicating "
            "at least one extreme outlier.",
            {"max": max_income, "median": median_income, "mean": mean_income,
             "max_to_median_ratio": round((max_income or 0) / max(median_income or 1, 1), 1)},
            "notable",
        ))

    days_columns = [c for c in cols if c.startswith("DAYS_") and c != "DAYS_EMPLOYED"]
    if days_columns:
        positive_checks = {
            c: conn.execute(
                f'SELECT count(*) FROM application_train WHERE "{c}" > 0'
            ).fetchone()[0]
            for c in days_columns
        }
        unexpected = {c: n for c, n in positive_checks.items() if n > 0}
        findings.append(QualityFinding(
            "days_columns_sign",
            "DAYS_* columns are documented as negative offsets from the "
            "application date; checked all of them for unexpected positive values.",
            {"columns_checked": len(days_columns), "columns_with_positive_values": unexpected},
            "notable" if unexpected else "info",
        ))

    housing_cols = [c for c in cols if categorise_column(c) == "housing_detail"]
    if housing_cols:
        report = missing_value_report(conn, ("application_train",))
        housing_missing = [r.pct_missing for r in report if r.column in housing_cols]
        if housing_missing:
            findings.append(QualityFinding(
                "building_stats_missing",
                "The building/apartment statistics block is missing for a "
                "large share of applicants (most live in housing types the "
                "block does not describe, e.g. rented or with parents).",
                {"columns": len(housing_cols),
                 "min_pct_missing": round(min(housing_missing), 3),
                 "max_pct_missing": round(max(housing_missing), 3)},
                "notable",
            ))

    if "TARGET" in cols:
        rate = conn.execute("SELECT avg(TARGET) FROM application_train").fetchone()[0]
        findings.append(QualityFinding(
            "class_imbalance",
            "TARGET is heavily imbalanced: most applicants repay.",
            {"default_rate": round(rate, 4),
             "imbalance_ratio": round((1 - rate) / rate, 1) if rate else None},
            "action_needed",
        ))

    return findings


def _columns(conn: duckdb.DuckDBPyConnection, table: str) -> set[str]:
    return {
        r[0] for r in conn.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema='main' AND table_name=?", [table],
        ).fetchall()
    }


# --------------------------------------------------------------------------- #
# Business insights
# --------------------------------------------------------------------------- #
@dataclass
class Insight:
    """One business insight: the question, the query, the data, and a headline
    sentence composed from the data itself - never pre-written prose.
    """

    id: str
    title: str
    business_question: str
    chart_type: str
    columns: list[str]
    rows: list[tuple[Any, ...]]
    headline: str
    so_what: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "title": self.title,
            "business_question": self.business_question,
            "chart_type": self.chart_type, "columns": self.columns,
            "rows": [list(r) for r in self.rows],
            "headline": self.headline, "so_what": self.so_what,
        }


def _fetch(conn: duckdb.DuckDBPyConnection, sql: str) -> tuple[list[str], list[tuple]]:
    relation = conn.execute(sql)
    columns = [d[0] for d in relation.description]
    return columns, relation.fetchall()


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x * 100:.1f}%"


def insight_default_by_ext_source(conn: duckdb.DuckDBPyConnection) -> Insight:
    sql = """
    WITH scored AS (
        SELECT TARGET, ntile(10) OVER (ORDER BY EXT_SOURCE_2) AS decile
        FROM application_train WHERE EXT_SOURCE_2 IS NOT NULL
    )
    SELECT decile, count(*) AS applicants, avg(TARGET) AS default_rate
    FROM scored GROUP BY decile ORDER BY decile
    """
    cols, rows = _fetch(conn, sql)
    # decile 1 = lowest EXT_SOURCE_2 score; decile 10 = highest score. The
    # score is protective, so decile 1 has the *higher* default rate - the
    # lift is expressed as "riskiest over safest", not "last row over first".
    bottom_score_decile, top_score_decile = rows[0], rows[-1]
    bottom_rate, top_rate = bottom_score_decile[2], top_score_decile[2]
    lift = (bottom_rate / top_rate) if top_rate else None
    return Insight(
        id="ext_source_decile", title="External credit score predicts default sharply",
        business_question="How does the default rate change across EXT_SOURCE_2 deciles?",
        chart_type="bar", columns=cols, rows=rows,
        headline=(
            f"The bottom EXT_SOURCE_2 decile defaults at {_pct(bottom_rate)}, versus "
            f"{_pct(top_rate)} in the top decile"
            + (f" - {lift:.1f}x the risk." if lift else ".")
        ),
        so_what=(
            "EXT_SOURCE_2 alone separates risk more than most raw application "
            "fields; it should anchor both the model and any manual review rules."
        ),
    )


def insight_age_vs_default(conn: duckdb.DuckDBPyConnection) -> Insight:
    sql = """
    SELECT floor(-DAYS_BIRTH / 365.25 / 10) * 10 AS age_band,
           count(*) AS applicants, avg(TARGET) AS default_rate
    FROM application_train
    WHERE DAYS_BIRTH IS NOT NULL
    GROUP BY age_band ORDER BY age_band
    """
    cols, rows = _fetch(conn, sql)
    youngest, oldest = rows[0], rows[-1]
    return Insight(
        id="age_vs_default", title="Younger applicants default more often",
        business_question="How does default rate vary with applicant age?",
        chart_type="bar", columns=cols, rows=rows,
        headline=(
            f"Applicants in their {int(youngest[0])}s default at {_pct(youngest[2])}, "
            f"versus {_pct(oldest[2])} for those in their {int(oldest[0])}s."
        ),
        so_what="Age bands are a legitimate, low-cost pre-screen signal, though "
                "policy must ensure they never substitute for individual assessment.",
    )


def insight_employment_vs_default(conn: duckdb.DuckDBPyConnection) -> Insight:
    sql = """
    SELECT CASE
             WHEN DAYS_EMPLOYED = 365243 THEN 'not employed'
             WHEN -DAYS_EMPLOYED / 365.25 < 2 THEN 'under 2 years'
             WHEN -DAYS_EMPLOYED / 365.25 < 5 THEN '2-5 years'
             ELSE 'over 5 years'
           END AS employment_band,
           count(*) AS applicants, avg(TARGET) AS default_rate
    FROM application_train GROUP BY employment_band ORDER BY default_rate DESC
    """
    cols, rows = _fetch(conn, sql)
    worst = rows[0]
    return Insight(
        id="employment_vs_default", title="Short employment tenure is the riskiest segment",
        business_question="Does time in the current job relate to default?",
        chart_type="bar", columns=cols, rows=rows,
        headline=f"'{worst[0]}' applicants default most often, at {_pct(worst[2])}.",
        so_what="Employment tenure captures income stability that raw income alone "
                "misses, and is cheap to verify at application time.",
    )


def insight_loan_burden_vs_default(conn: duckdb.DuckDBPyConnection) -> Insight:
    sql = """
    WITH scored AS (
        SELECT TARGET, AMT_CREDIT / nullif(AMT_INCOME_TOTAL, 0) AS credit_income_ratio,
               ntile(5) OVER (ORDER BY AMT_CREDIT / nullif(AMT_INCOME_TOTAL, 0)) AS quintile
        FROM application_train WHERE AMT_INCOME_TOTAL > 0
    )
    SELECT quintile, count(*) AS applicants,
           avg(credit_income_ratio) AS avg_credit_income_ratio,
           avg(TARGET) AS default_rate
    FROM scored GROUP BY quintile ORDER BY quintile
    """
    cols, rows = _fetch(conn, sql)
    # Found by actual value, not by assuming quintile 1 or 5 is the extreme -
    # on the real data this relationship is not monotonic (see so_what).
    riskiest = max(rows, key=lambda r: r[3])
    safest = min(rows, key=lambda r: r[3])
    is_monotonic = riskiest[0] == rows[-1][0] and safest[0] == rows[0][0]

    title = (
        "Heavier loan burden relative to income raises default risk"
        if is_monotonic else
        "Loan burden relative to income has a non-linear relationship with default"
    )
    so_what = (
        "Credit-to-income ratio is an engineered feature the model already "
        "uses, and doubles as a plain-English underwriting rule."
        if is_monotonic else
        "The riskiest quintile is not the heaviest-burden one, so a simple "
        "'flag high ratios' rule would miss it - this is better left to the "
        "model, which can combine the ratio with other signals, than encoded "
        "as a standalone threshold rule."
    )
    return Insight(
        id="loan_burden_vs_default", title=title,
        business_question="Does a higher credit-to-income ratio predict default?",
        chart_type="bar", columns=cols, rows=rows,
        headline=(
            f"Quintile {int(riskiest[0])} (avg ratio {riskiest[2]:.1f}x income) "
            f"defaults most often, at {_pct(riskiest[3])}, versus {_pct(safest[3])} "
            f"in quintile {int(safest[0])} (avg ratio {safest[2]:.1f}x)."
        ),
        so_what=so_what,
    )


def insight_contract_type_vs_default(conn: duckdb.DuckDBPyConnection) -> Insight:
    sql = """
    SELECT NAME_CONTRACT_TYPE, count(*) AS applicants, avg(TARGET) AS default_rate
    FROM application_train GROUP BY NAME_CONTRACT_TYPE ORDER BY default_rate DESC
    """
    cols, rows = _fetch(conn, sql)
    worst = rows[0]
    return Insight(
        id="contract_type_vs_default", title="Loan type is associated with different risk levels",
        business_question="Do cash loans and revolving loans default at different rates?",
        chart_type="bar", columns=cols, rows=rows,
        headline=f"{worst[0]} applicants default at {_pct(worst[2])}, the highest of the contract types.",
        so_what="Product-level risk differences justify differentiated pricing "
                "or underwriting thresholds by loan type.",
    )


def insight_bureau_history_vs_default(conn: duckdb.DuckDBPyConnection) -> Insight:
    sql = """
    WITH per_client AS (
        SELECT SK_ID_CURR,
               count(*) FILTER (WHERE CREDIT_ACTIVE = 'Active') AS active_credits,
               sum(AMT_CREDIT_SUM_OVERDUE) AS total_overdue
        FROM bureau GROUP BY SK_ID_CURR
    )
    SELECT CASE WHEN p.total_overdue > 0 THEN 'has overdue bureau credit'
                ELSE 'no overdue bureau credit' END AS overdue_status,
           count(*) AS applicants, avg(a.TARGET) AS default_rate
    FROM application_train a JOIN per_client p ON p.SK_ID_CURR = a.SK_ID_CURR
    GROUP BY overdue_status ORDER BY default_rate DESC
    """
    cols, rows = _fetch(conn, sql)
    worst = rows[0]
    best = rows[-1] if len(rows) > 1 else rows[0]
    return Insight(
        id="bureau_overdue_vs_default", title="Overdue debt at other institutions is a strong warning sign",
        business_question="Do clients with overdue bureau credit default more on this loan?",
        chart_type="bar", columns=cols, rows=rows,
        headline=(
            f"Clients with overdue bureau credit default at {_pct(worst[2])}, "
            f"versus {_pct(best[2])} for those with none."
        ),
        so_what="Bureau data adds signal application data alone cannot see - "
                "obligations already in trouble elsewhere.",
    )


def insight_previous_refusal_vs_default(conn: duckdb.DuckDBPyConnection) -> Insight:
    sql = """
    WITH per_client AS (
        SELECT SK_ID_CURR,
               count(*) FILTER (WHERE NAME_CONTRACT_STATUS = 'Refused') AS refusals
        FROM previous_application GROUP BY SK_ID_CURR
    )
    SELECT CASE WHEN p.refusals > 0 THEN 'previously refused' ELSE 'never refused' END AS history,
           count(*) AS applicants, avg(a.TARGET) AS default_rate
    FROM application_train a JOIN per_client p ON p.SK_ID_CURR = a.SK_ID_CURR
    GROUP BY history ORDER BY default_rate DESC
    """
    cols, rows = _fetch(conn, sql)
    worst = rows[0]
    return Insight(
        id="previous_refusal_vs_default", title="A past refusal at this lender predicts future default",
        business_question="Do clients who were refused before default more when approved later?",
        chart_type="bar", columns=cols, rows=rows,
        headline=f"Clients {worst[0]} default at {_pct(worst[2])}.",
        so_what="Internal application history is free to use and should weigh "
                "into repeat-applicant underwriting.",
    )


def insight_installment_lateness_vs_default(conn: duckdb.DuckDBPyConnection) -> Insight:
    sql = """
    WITH per_client AS (
        SELECT SK_ID_CURR,
               avg(DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT) AS avg_days_late
        FROM installments_payments WHERE DAYS_ENTRY_PAYMENT IS NOT NULL
        GROUP BY SK_ID_CURR
    )
    SELECT CASE WHEN p.avg_days_late > 5 THEN 'often pay late'
                ELSE 'pay on time or early' END AS payment_behaviour,
           count(*) AS applicants, avg(a.TARGET) AS default_rate
    FROM application_train a JOIN per_client p ON p.SK_ID_CURR = a.SK_ID_CURR
    GROUP BY payment_behaviour ORDER BY default_rate DESC
    """
    cols, rows = _fetch(conn, sql)
    worst = rows[0]
    return Insight(
        id="installment_lateness_vs_default", title="Late repayment behaviour on past installments carries forward",
        business_question="Do clients who paid past installments late default more on new loans?",
        chart_type="bar", columns=cols, rows=rows,
        headline=f"Clients who {worst[0]} default at {_pct(worst[2])}.",
        so_what="Observed repayment behaviour, not just stated intent, is one "
                "of the most directly actionable signals in the dataset.",
    )


#: Every insight, run in this order to build the artifact file.
INSIGHT_FUNCTIONS: tuple[Callable[[duckdb.DuckDBPyConnection], Insight], ...] = (
    insight_default_by_ext_source,
    insight_age_vs_default,
    insight_employment_vs_default,
    insight_loan_burden_vs_default,
    insight_contract_type_vs_default,
    insight_bureau_history_vs_default,
    insight_previous_refusal_vs_default,
    insight_installment_lateness_vs_default,
)


def run_all_insights(
    conn: duckdb.DuckDBPyConnection | None = None,
) -> list[Insight]:
    """Run every insight, skipping (and logging) any whose table is missing.

    Skipping rather than failing matters for `lite` mode and for partial data
    during development - the artifact file should hold whatever can currently
    be computed, not nothing.
    """
    conn = conn or get_readonly_connection()
    results = []
    for fn in INSIGHT_FUNCTIONS:
        try:
            with log_duration(log, f"insight: {fn.__name__}"):
                results.append(fn(conn))
        except Exception as exc:
            # Broader than duckdb.Error on purpose: an insight can also fail
            # with a plain Python error when a query legitimately returns
            # fewer rows than the function assumes - e.g. the two-category
            # insights (bureau overdue, previous refusal, ...) INNER JOIN a
            # child-table aggregate and index into `rows[0]`/`rows[-1]`,
            # which raises IndexError rather than a duckdb error when that
            # child table has no rows for this database (a real state: a
            # freshly created lite/sample/test database can legitimately
            # have an empty bureau table). One insight failing this way
            # should not take down the whole EDA build.
            log.warning("skipping %s: %s", fn.__name__, exc)
    return results


def build_eda_artifacts(
    conn: duckdb.DuckDBPyConnection | None = None,
) -> dict[str, Any]:
    """Everything the API's EDA section and the notebook need, in one payload."""
    conn = conn or get_readonly_connection()
    with log_duration(log, "build EDA artifacts"):
        summary = dataset_summary(conn)
        categories = feature_categories(conn)
        missing = missing_value_report(conn, tables=tuple(
            t.name for t in summary if t.name in
            {"application_train", "bureau", "previous_application"}
        ))
        quality = data_quality_findings(conn)
        insights = run_all_insights(conn)

    return {
        "table_summary": [s.__dict__ for s in summary],
        "feature_categories": categories,
        "missing_values": [m.__dict__ for m in missing[:40]],
        "data_quality_findings": [
            {"id": f.id, "description": f.description, "value": f.value,
             "severity": f.severity}
            for f in quality
        ],
        "insights": [i.as_dict() for i in insights],
    }
