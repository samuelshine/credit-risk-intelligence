"""DuckDB connection management.

DuckDB is the spine of this platform. It does three jobs that would otherwise
need three tools:

* **ETL**   - reads 2.7 GB of CSV straight into typed tables.
* **Features** - the per-applicant aggregations over 55M rows of credit history
  are expressed as SQL, which keeps memory flat and makes every model input
  auditable by reading a query rather than tracing pandas.
* **Chatbot** - the same tables the model was built on are what the
  natural-language questions run against, so the numbers can never disagree.

Two connection flavours, and the distinction is a security boundary rather than
a convenience: ETL and training open read-write; anything touching
LLM-generated SQL opens **read-only**. Even if a malicious query slipped past
`sql_validator`, DuckDB itself would reject the write.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import duckdb

from src.utils.config import Settings, get_settings
from src.utils.logger import get_logger

log = get_logger(__name__)

# A read-only DuckDB connection is safe to share across threads, and FastAPI
# serves requests on a thread pool, so we cache one process-wide rather than
# reopening the database file per request.
_readonly_conn: duckdb.DuckDBPyConnection | None = None
_readonly_lock = threading.Lock()


def _apply_pragmas(conn: duckdb.DuckDBPyConnection, settings: Settings) -> None:
    """Bound DuckDB's resource use.

    Without a memory limit DuckDB will happily use everything available and get
    OOM-killed on a 2 GB Render instance mid-ingest. With one it spills to disk
    and merely runs slower, which is the failure mode we want.
    """
    conn.execute(f"SET memory_limit='{settings.duckdb_memory_limit}'")
    conn.execute(f"SET threads={settings.duckdb_threads}")
    conn.execute("SET preserve_insertion_order=false")


def database_exists(settings: Settings | None = None) -> bool:
    """True when the DuckDB file has been built."""
    settings = settings or get_settings()
    return Path(settings.duckdb_path).exists()


@contextmanager
def writable_connection(
    settings: Settings | None = None,
) -> Iterator[duckdb.DuckDBPyConnection]:
    """Open a read-write connection for ETL, feature building and training.

    Always used as a context manager so the file is closed and its WAL
    checkpointed - an open write handle would block the read-only connections
    the API needs.
    """
    settings = settings or get_settings()
    path = Path(settings.duckdb_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    conn = duckdb.connect(str(path), read_only=False)
    try:
        _apply_pragmas(conn, settings)
        yield conn
    finally:
        conn.close()


def get_readonly_connection(
    settings: Settings | None = None,
) -> duckdb.DuckDBPyConnection:
    """Process-wide read-only connection, opened on first use.

    Raises if the database has not been built yet; callers in the API check
    `database_exists()` first and surface a setup message instead of a 500.
    """
    global _readonly_conn
    settings = settings or get_settings()

    if _readonly_conn is not None:
        return _readonly_conn

    with _readonly_lock:
        if _readonly_conn is not None:  # another thread won the race
            return _readonly_conn

        path = Path(settings.duckdb_path)
        if not path.exists():
            raise FileNotFoundError(
                f"DuckDB database not found at {path}. "
                f"Build it with: python -m src.data.loader"
            )

        conn = duckdb.connect(str(path), read_only=True)
        _apply_pragmas(conn, settings)
        log.info("opened read-only DuckDB connection at %s", path)
        _readonly_conn = conn
        return _readonly_conn


def query_connection() -> duckdb.DuckDBPyConnection:
    """A per-caller cursor over the shared read-only connection.

    DuckDB cursors are independent execution contexts, so concurrent API
    requests cannot interleave on one another's results. This is what
    `query_runner` uses for every LLM-generated query.
    """
    return get_readonly_connection().cursor()


def reset_readonly_connection() -> None:
    """Drop the cached read-only connection.

    Needed after the ETL rebuilds the database inside a live process (the
    Render first-boot path), where the cached handle would otherwise point at
    a file that no longer exists.
    """
    global _readonly_conn
    with _readonly_lock:
        if _readonly_conn is not None:
            _readonly_conn.close()
            _readonly_conn = None
            log.info("closed cached read-only DuckDB connection")


def list_tables(conn: duckdb.DuckDBPyConnection | None = None) -> list[str]:
    """User tables in the main schema, alphabetically."""
    conn = conn or get_readonly_connection()
    rows = conn.execute(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_schema = 'main' ORDER BY table_name"
    ).fetchall()
    return [r[0] for r in rows]


def table_row_counts(
    conn: duckdb.DuckDBPyConnection | None = None,
) -> dict[str, int]:
    """Row count per table. Cheap on DuckDB - it reads table metadata."""
    conn = conn or get_readonly_connection()
    return {
        name: conn.execute(f'SELECT count(*) FROM "{name}"').fetchone()[0]
        for name in list_tables(conn)
    }
