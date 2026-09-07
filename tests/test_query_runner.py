"""Tests for query execution: limits, timeouts and result shaping."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def seeded_db(tmp_path_factory) -> Path:
    """A tiny database with the shape the runner will meet in production."""
    tmp = tmp_path_factory.mktemp("qr")
    os.environ["DUCKDB_PATH"] = str(tmp / "t.duckdb")
    os.environ["DATA_DIR"] = str(tmp)
    os.environ["DUCKDB_MEMORY_LIMIT"] = "1GB"

    from src.utils.config import get_settings
    get_settings.cache_clear()

    from src.data.database import reset_readonly_connection, writable_connection
    reset_readonly_connection()
    with writable_connection() as conn:
        conn.execute(
            "CREATE TABLE application_train AS "
            "SELECT i AS SK_ID_CURR, (i % 12 = 0)::BIGINT AS TARGET, "
            "       (i % 2)::VARCHAR AS CODE_GENDER "
            "FROM range(1000) t(i)"
        )
    yield tmp
    reset_readonly_connection()


def test_simple_query_returns_rows(seeded_db) -> None:
    from src.talk_to_data.query_runner import run_query

    result = run_query("SELECT count(*) AS n FROM application_train")
    assert result.ok
    assert result.columns == ["n"]
    assert result.rows[0][0] == 1000
    assert not result.truncated


def test_row_limit_marks_truncation(seeded_db) -> None:
    from src.talk_to_data.query_runner import run_query

    result = run_query("SELECT * FROM application_train", max_rows=10)
    assert result.ok
    assert result.row_count == 10
    assert result.truncated


def test_exact_limit_is_not_flagged_truncated(seeded_db) -> None:
    """Off-by-one guard: 10 rows under a limit of 10 is complete, not cut."""
    from src.talk_to_data.query_runner import run_query

    result = run_query("SELECT * FROM application_train LIMIT 10", max_rows=10)
    assert result.row_count == 10
    assert not result.truncated


def test_empty_result_is_not_an_error(seeded_db) -> None:
    from src.talk_to_data.query_runner import run_query

    result = run_query("SELECT * FROM application_train WHERE SK_ID_CURR < 0")
    assert result.ok
    assert result.is_empty
    assert "no rows matched" in result.to_markdown()


def test_engine_error_is_returned_not_raised(seeded_db) -> None:
    from src.talk_to_data.query_runner import run_query

    result = run_query("SELECT nonexistent_column FROM application_train")
    assert not result.ok
    assert "nonexistent_column" in result.error


def test_timeout_interrupts_and_leaves_connection_usable(seeded_db) -> None:
    """A slow question must not poison the connection for the next one."""
    from src.talk_to_data.query_runner import run_query

    slow = "SELECT count(*) FROM range(30000000000) a, range(100) b"
    result = run_query(slow, timeout_seconds=1)
    assert not result.ok
    assert "longer than 1 second" in result.error

    after = run_query("SELECT count(*) AS n FROM application_train")
    assert after.ok and after.rows[0][0] == 1000


def test_markdown_rendering_is_compact(seeded_db) -> None:
    from src.talk_to_data.query_runner import run_query

    result = run_query(
        "SELECT CODE_GENDER, avg(TARGET) AS default_rate "
        "FROM application_train GROUP BY CODE_GENDER ORDER BY CODE_GENDER"
    )
    table = result.to_markdown()
    assert "CODE_GENDER | default_rate" in table
    assert "---" in table
    # Every 12th row defaults and 12 is even, so all defaults land in the
    # even-id bucket: 84/500 = 0.168, and the odd bucket is exactly zero.
    assert "0.168" in table
    # Trailing zeros are stripped rather than padded to full float precision.
    assert "0.0000" not in table


def test_records_round_trip(seeded_db) -> None:
    from src.talk_to_data.query_runner import run_query

    result = run_query("SELECT SK_ID_CURR, TARGET FROM application_train LIMIT 3")
    records = result.as_records()
    assert len(records) == 3
    assert set(records[0]) == {"SK_ID_CURR", "TARGET"}
