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
LLM-generated SQL opens **read-only** *and* with external access disabled.

That second flag matters more than it looks. A plain read-only DuckDB
connection still happily runs `read_csv('/etc/passwd')`, `glob('/etc/*')` and
even `COPY ... TO '/tmp/out.csv'` - read-only protects the *database*, not the
filesystem. Measured on DuckDB 1.5.5:

    read-only alone                  read-only + enable_external_access=false
    ------------------------------   ----------------------------------------
    read_csv arbitrary file  ALLOW   read_csv arbitrary file        PermissionException
    glob filesystem          ALLOW   glob filesystem                PermissionException
    COPY TO file             ALLOW   COPY TO file                   PermissionException
    ATTACH another database  ALLOW   ATTACH another database        PermissionException

`enable_external_access` is a locked setting: once the database is open
read-only it cannot be turned back on from inside a query, so a generated
query cannot escalate its own privileges. This is the outermost of three
layers - the others are AST validation in `talk_to_data.sql_validator` and the
read-only handle itself.
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


def _ready_marker_path(settings: Settings) -> Path:
    """Sentinel written only after a full, successful ETL run.

    Deliberately not the same signal as "the .duckdb file exists": DuckDB
    creates that file the moment a connection opens, long before every table
    is loaded. Measured against a real concurrent run - the Render first-boot
    scenario this exists for - a request arriving while `etl` is mid-build
    hit a `CatalogException` ("bureau_balance does not exist") and surfaced
    as a raw 500, because `database_exists()` had already returned true. The
    marker is the fix: written once, after `_verify_row_counts` and
    `dump_schema` both succeed, so its presence is a real completeness
    guarantee rather than a guess from file existence.
    """
    return Path(settings.duckdb_path).with_suffix(".ready")


def mark_database_ready(settings: Settings | None = None) -> None:
    """Called once, at the end of a successful `build_database()`."""
    settings = settings or get_settings()
    _ready_marker_path(settings).touch()


def clear_database_ready(settings: Settings | None = None) -> None:
    """Called before a (re)build starts.

    From this point until `mark_database_ready()` runs, `database_exists()`
    must report false for every caller - a crash partway through a forced
    rebuild must never leave a stale marker pointing at an incomplete
    database. `missing_ok=True` since a fresh build has no marker to clear.
    """
    settings = settings or get_settings()
    _ready_marker_path(settings).unlink(missing_ok=True)


def database_exists(settings: Settings | None = None) -> bool:
    """True once a full ETL run has completed successfully.

    Named for what every caller actually wants to know - "is the database
    safe to query" - even though what it checks is the completion marker,
    not (only) the file. Kept as one function rather than two so a caller can
    never accidentally pick the weaker, file-existence-only check.
    """
    settings = settings or get_settings()
    return _ready_marker_path(settings).exists()


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

        conn = duckdb.connect(
            str(path),
            read_only=True,
            # Locked at connect time; see the module docstring. Without this a
            # generated query can read any file the process can read.
            config={"enable_external_access": "false"},
        )
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
