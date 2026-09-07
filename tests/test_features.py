"""Tests for SQL-based feature engineering.

The determinism test exists because of a real bug: `build_feature_sql` had no
`ORDER BY`, so DuckDB's row order was not guaranteed identical across two
executions of the exact same query. Since `train_test_split(random_state=42)`
splits by position, an unordered query silently made "the same random_state"
produce a *different* split every run - caught when a holdout evaluation
reconstructed from a fresh query call scored 0.855 ROC-AUC against the 0.785
the original training run measured on its own in-process split.
"""

from __future__ import annotations

import re

import pytest


def _create_empty_child_tables(conn) -> None:
    """Create every non-application child table with its real column set
    (borrowed from tests/conftest.py's realistic schema), empty. The feature
    SQL joins all of them regardless of what a test cares about, so it needs
    every join target to exist with the right columns even when a test only
    populates a couple of rows in one or two of them."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from conftest import _TABLES  # local import: only used by tests

    for table, (_, columns) in _TABLES.items():
        if table == "application_train":
            continue
        ddl = ", ".join(f'"{name}" {dtype}' for name, dtype in columns)
        conn.execute(f'CREATE TABLE "{table}" ({ddl})')


def test_feature_sql_has_a_deterministic_order_by() -> None:
    """Regression test for the reproducibility bug above. A future edit that
    drops the ORDER BY would silently reintroduce split leakage that no
    metric-shape test would catch - this checks the SQL text directly."""
    from src.data.features import build_feature_sql

    sql = build_feature_sql("application_train")
    assert re.search(r"ORDER BY\s+app\.SK_ID_CURR\s*$", sql.strip())


def test_feature_query_returns_identical_row_order_across_calls(isolated_db) -> None:
    """The bug's actual symptom: two independent executions must agree on
    row order, not just on which rows are present."""
    conn = isolated_db
    conn.execute("""
        CREATE TABLE application_train AS
        SELECT i AS SK_ID_CURR, (i % 10 = 0)::BIGINT AS TARGET,
               50000.0 + i AS AMT_INCOME_TOTAL, 100000.0 + i AS AMT_CREDIT,
               -10000 - i AS DAYS_BIRTH, -1000 - i AS DAYS_EMPLOYED,
               NULL::DOUBLE AS EXT_SOURCE_1, NULL::DOUBLE AS EXT_SOURCE_2,
               NULL::DOUBLE AS EXT_SOURCE_3, NULL::DOUBLE AS AMT_ANNUITY,
               NULL::DOUBLE AS AMT_GOODS_PRICE, 1::BIGINT AS CNT_FAM_MEMBERS,
               0::BIGINT AS CNT_CHILDREN, -100 AS DAYS_REGISTRATION,
               -100 AS DAYS_ID_PUBLISH
        FROM range(500) t(i)
    """)
    _create_empty_child_tables(conn)

    from src.data.features import build_feature_sql
    sql = build_feature_sql("application_train")

    first = conn.execute(sql)
    idx = [d[0] for d in first.description].index("SK_ID_CURR")
    ids_first = [r[idx] for r in first.fetchall()]
    ids_second = [r[idx] for r in conn.execute(sql).fetchall()]
    assert ids_first == ids_second
    assert ids_first == sorted(ids_first)  # ascending by SK_ID_CURR, as declared


def test_load_features_separates_ids_target_and_excludes_them_from_the_frame(
    isolated_db,
) -> None:
    conn = isolated_db
    conn.execute("""
        CREATE TABLE application_train AS
        SELECT i AS SK_ID_CURR, (i % 10 = 0)::BIGINT AS TARGET,
               50000.0 + i AS AMT_INCOME_TOTAL, 100000.0 + i AS AMT_CREDIT,
               -10000 - i AS DAYS_BIRTH, -1000 - i AS DAYS_EMPLOYED,
               NULL::DOUBLE AS EXT_SOURCE_1, NULL::DOUBLE AS EXT_SOURCE_2,
               NULL::DOUBLE AS EXT_SOURCE_3, NULL::DOUBLE AS AMT_ANNUITY,
               NULL::DOUBLE AS AMT_GOODS_PRICE, 1::BIGINT AS CNT_FAM_MEMBERS,
               0::BIGINT AS CNT_CHILDREN, -100 AS DAYS_REGISTRATION,
               -100 AS DAYS_ID_PUBLISH,
               CASE WHEN i % 3 = 0 THEN 'A' ELSE 'B' END AS NAME_EDUCATION_TYPE
        FROM range(200) t(i)
    """)
    _create_empty_child_tables(conn)

    from src.data.features import load_features
    matrix = load_features("application_train", conn=conn)

    assert len(matrix) == 200
    assert "SK_ID_CURR" not in matrix.frame.columns
    assert "TARGET" not in matrix.frame.columns
    assert matrix.target is not None
    assert set(matrix.ids) == set(range(200))
    assert "NAME_EDUCATION_TYPE" in matrix.categorical_columns
    assert str(matrix.frame["NAME_EDUCATION_TYPE"].dtype) == "category"
