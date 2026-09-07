"""Adversarial tests for the generated-SQL validator.

These are the tests that matter most in this repo. The validator is what stands
between a language model's output and the database, so it is tested against
what an attacker or a confused model would actually emit, not just the happy
path.
"""

from __future__ import annotations

import pytest

from src.talk_to_data.catalog import Catalog, ColumnInfo, TableInfo
from src.talk_to_data.sql_validator import validate_sql


def _column(name: str, data_type: str = "DOUBLE") -> ColumnInfo:
    return ColumnInfo(name=name, data_type=data_type)


@pytest.fixture(scope="module")
def catalog() -> Catalog:
    """A miniature catalog with the real column names and shapes."""
    return Catalog(
        tables={
            "application_train": TableInfo(
                name="application_train",
                row_count=307_511,
                columns={
                    c.name: c
                    for c in [
                        _column("SK_ID_CURR", "BIGINT"),
                        _column("TARGET", "BIGINT"),
                        _column("CODE_GENDER", "VARCHAR"),
                        _column("AMT_INCOME_TOTAL"),
                        _column("AMT_CREDIT"),
                        _column("DAYS_BIRTH", "BIGINT"),
                        _column("EXT_SOURCE_2"),
                        _column("NAME_CONTRACT_TYPE", "VARCHAR"),
                    ]
                },
            ),
            "bureau": TableInfo(
                name="bureau",
                row_count=1_716_428,
                columns={
                    c.name: c
                    for c in [
                        _column("SK_ID_CURR", "BIGINT"),
                        _column("SK_ID_BUREAU", "BIGINT"),
                        _column("CREDIT_ACTIVE", "VARCHAR"),
                        _column("AMT_CREDIT_SUM"),
                        _column("DAYS_CREDIT", "BIGINT"),
                    ]
                },
            ),
            "bureau_balance": TableInfo(
                name="bureau_balance",
                row_count=27_299_925,
                columns={
                    c.name: c
                    for c in [
                        _column("SK_ID_BUREAU", "BIGINT"),
                        _column("MONTHS_BALANCE", "BIGINT"),
                        _column("STATUS", "VARCHAR"),
                    ]
                },
            ),
        }
    )


# --------------------------------------------------------------------------- #
# Queries that must be allowed
# --------------------------------------------------------------------------- #
VALID_QUERIES = [
    pytest.param(
        "SELECT count(*) FROM application_train",
        id="simple-aggregate",
    ),
    pytest.param(
        "SELECT CODE_GENDER, avg(TARGET) AS default_rate FROM application_train "
        "GROUP BY CODE_GENDER ORDER BY default_rate DESC",
        id="group-by-rate",
    ),
    pytest.param(
        "SELECT count(*) FROM application_train WHERE AMT_INCOME_TOTAL > 200000",
        id="filtered-count",
    ),
    pytest.param(
        "SELECT a.CODE_GENDER, count(DISTINCT b.SK_ID_BUREAU) AS credits "
        "FROM application_train a JOIN bureau b ON a.SK_ID_CURR = b.SK_ID_CURR "
        "GROUP BY a.CODE_GENDER",
        id="multi-table-join",
    ),
    pytest.param(
        "WITH active AS ("
        "  SELECT SK_ID_CURR, sum(AMT_CREDIT_SUM) AS total FROM bureau "
        "  WHERE CREDIT_ACTIVE = 'Active' GROUP BY SK_ID_CURR"
        ") "
        "SELECT a.TARGET, avg(x.total) AS avg_exposure "
        "FROM application_train a JOIN active x ON x.SK_ID_CURR = a.SK_ID_CURR "
        "GROUP BY a.TARGET",
        id="cte-with-alias",
    ),
    pytest.param(
        "SELECT * FROM ("
        "  SELECT SK_ID_CURR, AMT_CREDIT FROM application_train"
        ") t ORDER BY t.AMT_CREDIT DESC",
        id="derived-table",
    ),
    pytest.param(
        "SELECT ntile, avg(TARGET) FROM ("
        "  SELECT TARGET, ntile(10) OVER (ORDER BY EXT_SOURCE_2) AS ntile "
        "  FROM application_train WHERE EXT_SOURCE_2 IS NOT NULL"
        ") GROUP BY ntile ORDER BY ntile",
        id="window-function-decile",
    ),
    pytest.param(
        "SELECT b.STATUS, count(*) FROM bureau_balance b "
        "JOIN bureau u ON u.SK_ID_BUREAU = b.SK_ID_BUREAU GROUP BY b.STATUS",
        id="three-level-join-key",
    ),
]


@pytest.mark.parametrize("sql", VALID_QUERIES)
def test_valid_queries_pass(catalog: Catalog, sql: str) -> None:
    result = validate_sql(sql, catalog, max_rows=500)
    assert result.ok, result.errors
    assert result.tables_used
    assert result.limit_applied is not None


# --------------------------------------------------------------------------- #
# Queries that must be refused
# --------------------------------------------------------------------------- #
INJECTION_ATTEMPTS = [
    pytest.param("DROP TABLE application_train", "DROP", id="drop-table"),
    pytest.param("DELETE FROM application_train WHERE TARGET = 1", "DELETE", id="delete"),
    pytest.param("UPDATE application_train SET TARGET = 0", "UPDATE", id="update"),
    pytest.param("INSERT INTO bureau VALUES (1, 2, 'x', 3, 4)", "INSERT", id="insert"),
    pytest.param(
        "SELECT count(*) FROM application_train; DROP TABLE bureau",
        "single statement",
        id="stacked-statement",
    ),
    pytest.param("ATTACH '/tmp/evil.duckdb' AS evil", "ATTACH", id="attach"),
    pytest.param("PRAGMA database_list", "PRAGMA", id="pragma"),
    pytest.param(
        "COPY application_train TO '/tmp/leak.csv'", "COPY", id="copy-exfiltration"
    ),
    pytest.param(
        "CREATE TABLE evil AS SELECT * FROM application_train", "CREATE", id="create"
    ),
]


@pytest.mark.parametrize("sql,expected_token", INJECTION_ATTEMPTS)
def test_write_and_admin_statements_are_refused(
    catalog: Catalog, sql: str, expected_token: str
) -> None:
    result = validate_sql(sql, catalog)
    assert not result.ok
    assert expected_token.lower() in result.error_message.lower()


FILESYSTEM_ESCAPES = [
    pytest.param("SELECT * FROM read_csv('/etc/passwd')", id="read-csv"),
    pytest.param("SELECT * FROM read_csv_auto('/etc/passwd')", id="read-csv-auto"),
    pytest.param("SELECT * FROM read_parquet('/data/secret.parquet')", id="read-parquet"),
    pytest.param("SELECT * FROM read_json_auto('/etc/hosts')", id="read-json"),
    pytest.param("SELECT * FROM glob('/etc/*')", id="glob"),
    pytest.param("SELECT * FROM parquet_scan('/x.parquet')", id="parquet-scan"),
    pytest.param("SELECT * FROM sqlite_scan('/x.db', 'users')", id="sqlite-scan"),
]


@pytest.mark.parametrize("sql", FILESYSTEM_ESCAPES)
def test_filesystem_functions_are_refused(catalog: Catalog, sql: str) -> None:
    """These parse as ordinary SELECTs, so only a function check catches them.

    DuckDB's read-only mode does *not* block them, which is why this check
    exists rather than relying on the connection alone.
    """
    result = validate_sql(sql, catalog)
    assert not result.ok
    assert "not allowed" in result.error_message.lower() or "does not exist" in result.error_message.lower()


def test_unknown_table_is_refused_with_a_suggestion(catalog: Catalog) -> None:
    result = validate_sql("SELECT * FROM applications", catalog)
    assert not result.ok
    assert "applications" in result.error_message


def test_hallucinated_column_is_refused_with_a_suggestion(catalog: Catalog) -> None:
    """The commonest model failure: a plausible column that does not exist."""
    result = validate_sql(
        "SELECT avg(AMT_INCOME) FROM application_train", catalog
    )
    assert not result.ok
    assert "AMT_INCOME" in result.error_message
    assert "AMT_INCOME_TOTAL" in result.error_message


def test_hallucinated_qualified_column_is_refused(catalog: Catalog) -> None:
    result = validate_sql(
        "SELECT a.CUSTOMER_AGE FROM application_train a", catalog
    )
    assert not result.ok
    assert "CUSTOMER_AGE" in result.error_message


def test_column_from_wrong_table_is_refused(catalog: Catalog) -> None:
    """STATUS exists, but on bureau_balance, not application_train."""
    result = validate_sql("SELECT a.STATUS FROM application_train a", catalog)
    assert not result.ok
    assert "STATUS" in result.error_message


def test_table_outside_the_allowlist_is_refused(catalog: Catalog) -> None:
    result = validate_sql(
        "SELECT * FROM bureau", catalog, allowed_tables={"application_train"}
    )
    assert not result.ok
    assert "not available" in result.error_message.lower()


# --------------------------------------------------------------------------- #
# Normalisation
# --------------------------------------------------------------------------- #
def test_limit_is_injected_when_absent(catalog: Catalog) -> None:
    result = validate_sql("SELECT * FROM application_train", catalog, max_rows=250)
    assert result.ok
    assert result.limit_applied == 250
    assert "LIMIT 250" in result.sql.upper()


def test_oversized_limit_is_clamped(catalog: Catalog) -> None:
    result = validate_sql(
        "SELECT * FROM application_train LIMIT 100000", catalog, max_rows=500
    )
    assert result.ok
    assert result.limit_applied == 500


def test_smaller_limit_is_respected(catalog: Catalog) -> None:
    result = validate_sql(
        "SELECT * FROM application_train LIMIT 10", catalog, max_rows=500
    )
    assert result.ok
    assert result.limit_applied == 10


def test_markdown_fence_is_tolerated(catalog: Catalog) -> None:
    result = validate_sql(
        "```sql\nSELECT count(*) FROM application_train\n```", catalog
    )
    assert result.ok


def test_unparseable_sql_is_reported_clearly(catalog: Catalog) -> None:
    result = validate_sql("SELECT FROM WHERE GROUP", catalog)
    assert not result.ok


def test_empty_query_is_refused(catalog: Catalog) -> None:
    assert not validate_sql("   ", catalog).ok


def test_comment_disguised_injection_is_refused(catalog: Catalog) -> None:
    """A regex-based guard would likely miss this. AST parsing does not."""
    sql = "SELECT count(*) /* harmless */ FROM application_train; -- \nDROP TABLE bureau"
    result = validate_sql(sql, catalog)
    assert not result.ok
