"""Executing validated SQL and shaping the result for a reader.

By the time a query reaches here it has been parsed, checked against the
catalog and given a row limit. What is left is the operational side: run it
without letting one question monopolise the process, and turn the result into
something both a person and the summarising model can read.

DuckDB has no statement timeout, so one is imposed with a watchdog timer that
calls `interrupt()` on the cursor. Verified against DuckDB 1.5.5: the running
query raises `InterruptException` and the cursor remains usable afterwards, so
a slow question does not poison the connection for the next one.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

import duckdb

from src.data.database import query_connection
from src.utils.config import get_settings
from src.utils.logger import get_logger

log = get_logger(__name__)


class QueryTimeout(RuntimeError):
    """The query exceeded its time budget and was interrupted."""


@dataclass
class QueryResult:
    """Rows from one query, plus everything needed to present them honestly."""

    columns: list[str] = field(default_factory=list)
    rows: list[tuple[Any, ...]] = field(default_factory=list)
    row_count: int = 0
    truncated: bool = False
    duration_ms: int = 0
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def is_empty(self) -> bool:
        return self.row_count == 0

    def as_records(self) -> list[dict[str, Any]]:
        return [dict(zip(self.columns, row)) for row in self.rows]

    def to_markdown(self, max_rows: int = 50) -> str:
        """Render rows as a compact table for the summarising prompt.

        Markdown rather than JSON: it costs roughly 30% fewer tokens for the
        same rows because the column names are written once instead of on
        every record.
        """
        if not self.columns:
            return "(no columns)"
        if not self.rows:
            return "(no rows matched)"

        shown = self.rows[:max_rows]
        lines = [" | ".join(self.columns), " | ".join("---" for _ in self.columns)]
        lines.extend(
            " | ".join(_format_cell(value) for value in row) for row in shown
        )
        if len(self.rows) > len(shown):
            lines.append(f"... {len(self.rows) - len(shown)} more row(s) not shown")
        return "\n".join(lines)


def _format_cell(value: Any) -> str:
    """Format one cell for display.

    Rounds floats to four significant decimals: a rate returned as
    0.08072881234 adds tokens and reads as false precision, and the
    summarising prompt is told to round anyway.
    """
    if value is None:
        return "null"
    if isinstance(value, float):
        if value != value:  # NaN
            return "null"
        return f"{value:,.4f}".rstrip("0").rstrip(".") if abs(value) < 1e6 else f"{value:,.0f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def run_query(
    sql: str,
    *,
    timeout_seconds: int | None = None,
    max_rows: int | None = None,
) -> QueryResult:
    """Execute a validated query under a time and row budget.

    Errors are returned on the result rather than raised: the caller feeds the
    message back to the model for a single repair attempt, and an exception
    would make that the unusual path rather than the expected one.
    """
    settings = get_settings()
    timeout = timeout_seconds or settings.sql_timeout_seconds
    limit = max_rows or settings.max_sql_rows

    cursor = query_connection()
    watchdog = threading.Timer(timeout, cursor.interrupt)
    watchdog.daemon = True

    started = time.perf_counter()
    try:
        watchdog.start()
        relation = cursor.execute(sql)
        columns = [d[0] for d in (relation.description or [])]
        # One row over the limit tells us whether more existed, without
        # materialising a result set we would only throw away.
        rows = relation.fetchmany(limit + 1)
    except duckdb.InterruptException:
        return QueryResult(
            error=(
                f"The query took longer than {timeout} seconds and was stopped. "
                f"Try narrowing it, for example by filtering to fewer rows."
            ),
            duration_ms=int((time.perf_counter() - started) * 1000),
        )
    except duckdb.Error as exc:
        return QueryResult(
            error=_readable_error(exc),
            duration_ms=int((time.perf_counter() - started) * 1000),
        )
    finally:
        watchdog.cancel()

    duration_ms = int((time.perf_counter() - started) * 1000)
    truncated = len(rows) > limit
    if truncated:
        rows = rows[:limit]

    log.info("query returned %d row(s) in %dms%s",
             len(rows), duration_ms, " (truncated)" if truncated else "")

    return QueryResult(
        columns=columns,
        rows=rows,
        row_count=len(rows),
        truncated=truncated,
        duration_ms=duration_ms,
    )


def _readable_error(exc: duckdb.Error) -> str:
    """Turn a DuckDB error into something a person and a model can both act on.

    The validator catches most of these first; anything reaching here is a
    genuine engine-level problem, so the message is kept but trimmed of the
    query dump DuckDB appends.
    """
    message = str(exc).split("\n")[0].strip()
    prefix = "Catalog Error: "
    if message.startswith(prefix):
        message = message[len(prefix):]
    return message or "The query could not be executed."
