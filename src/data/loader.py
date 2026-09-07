"""ETL: Kaggle CSVs -> typed DuckDB tables.

Run as a module (this is what the compose `etl` service and the Render
first-boot path both invoke):

    python -m src.data.loader              # honours DATA_MODE from .env
    python -m src.data.loader --mode lite  # stratified sample
    python -m src.data.loader --force      # rebuild from scratch

Design notes
------------
**Types are inferred from a full scan, not guessed.** `read_csv_auto` with
`sample_size=-1` reads every row before deciding a column's type. Hand-writing
220 column declarations from memory would be faster to write and quietly wrong;
the schema this produces is dumped to `sql/schema.sql` afterwards so the
committed DDL always reflects what was actually built.

**`lite` mode is a stratified sample, not a head().** Taking the first N rows
would bias every table by application ID ordering. Instead we sample
`SK_ID_CURR` stratified on TARGET so the 8.1% default rate is preserved, then
filter each child table to those clients so referential integrity survives.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import duckdb

from src.data.acquire import SOURCE_FILES, SourceFile, ensure_dataset
from src.data.database import (
    clear_database_ready,
    database_exists,
    mark_database_ready,
    list_tables,
    table_row_counts,
    writable_connection,
)
from src.utils.config import PROJECT_ROOT, Settings, get_settings
from src.utils.logger import configure_logging, get_logger, log_duration

log = get_logger(__name__)

#: Tables the platform loads. `sample_submission` is competition scaffolding
#: with no analytical value, so it is skipped rather than cluttering the
#: chatbot's schema card with a table nobody would ask about.
SKIP_FILES = {"sample_submission.csv"}

#: Applications retained in `lite` mode. 50k keeps every EDA and chatbot number
#: within roughly +/-0.2pp of the full-data value while fitting a 512 MB box.
LITE_SAMPLE_SIZE = 50_000

#: Percentage of `application_test` kept in `lite` mode. It carries no TARGET,
#: so it cannot be stratified and is simply thinned to a comparable size.
_LITE_TEST_PCT = 16

#: The foreign keys that join the seven tables together. Indexed after load
#: because every feature aggregation and most chatbot joins hit them.
INDEXED_KEYS: dict[str, tuple[str, ...]] = {
    "application_train": ("SK_ID_CURR",),
    "application_test": ("SK_ID_CURR",),
    "bureau": ("SK_ID_CURR", "SK_ID_BUREAU"),
    "bureau_balance": ("SK_ID_BUREAU",),
    "previous_application": ("SK_ID_CURR", "SK_ID_PREV"),
    "installments_payments": ("SK_ID_CURR", "SK_ID_PREV"),
    "credit_card_balance": ("SK_ID_CURR", "SK_ID_PREV"),
    "pos_cash_balance": ("SK_ID_CURR", "SK_ID_PREV"),
}

#: How each table is restricted to the sampled clients in `lite` mode.
#:
#: The filter is applied *during* the CREATE TABLE, not by deleting afterwards -
#: on `bureau_balance` that is the difference between materialising 27M rows and
#: reading past them.
#:
#: `bureau_balance` has no SK_ID_CURR of its own, so it joins through `bureau`,
#: which is why load order matters and `bureau` is listed first in SOURCE_FILES.
#: `application_test` is unlabelled, so its clients are absent from the
#: TARGET-stratified sample and it gets its own independent subsample instead.
_LITE_FILTERS: dict[str, str] = {
    "application_train": "SK_ID_CURR IN (SELECT SK_ID_CURR FROM _lite_clients)",
    "application_test": f"hash(SK_ID_CURR) % 100 < {_LITE_TEST_PCT}",
    "bureau": "SK_ID_CURR IN (SELECT SK_ID_CURR FROM _lite_clients)",
    "bureau_balance": "SK_ID_BUREAU IN (SELECT SK_ID_BUREAU FROM bureau)",
    "previous_application": "SK_ID_CURR IN (SELECT SK_ID_CURR FROM _lite_clients)",
    "installments_payments": "SK_ID_CURR IN (SELECT SK_ID_CURR FROM _lite_clients)",
    "credit_card_balance": "SK_ID_CURR IN (SELECT SK_ID_CURR FROM _lite_clients)",
    "pos_cash_balance": "SK_ID_CURR IN (SELECT SK_ID_CURR FROM _lite_clients)",
}


def _read_csv_expr(path: Path, *, encoding: str = "utf-8") -> str:
    """A `read_csv` call that reads types from the whole file.

    `sample_size=-1` disables sampling. On this dataset the cost is one extra
    pass and the payoff is that `DAYS_CREDIT_ENDDATE` (sparse, occasionally
    huge) and the `AMT_*` columns get correct types instead of a type derived
    from whichever 20k rows the sniffer happened to look at.
    """
    return (
        f"read_csv('{path.as_posix()}', "
        f"sample_size=-1, "
        f"header=true, "
        f"encoding='{encoding}', "
        f"null_padding=true)"
    )


def _is_encoding_error(exc: Exception) -> bool:
    """Distinguish a text-decoding failure from a genuine schema problem."""
    message = str(exc).lower()
    return any(
        token in message
        for token in ("utf", "encoding", "encoded", "invalid unicode")
    )


def _transcode_to_utf8(csv_path: Path) -> Path:
    """Rewrite a non-UTF-8 CSV as UTF-8 alongside the original.

    Tries the encodings Kaggle exports actually use, in order of likelihood.
    The result is cached on disk, so a rebuild does not repeat the work.
    """
    utf8_path = csv_path.with_suffix(".utf8.csv")
    if utf8_path.exists():
        return utf8_path

    raw = csv_path.read_bytes()
    for encoding in ("cp1252", "latin-1"):
        try:
            text = raw.decode(encoding)
        except UnicodeDecodeError:
            continue
        utf8_path.write_text(text, encoding="utf-8")
        log.info("transcoded %s from %s to UTF-8", csv_path.name, encoding)
        return utf8_path

    raise RuntimeError(
        f"Could not decode {csv_path.name} as UTF-8, cp1252 or latin-1."
    )


def _load_table(
    conn: duckdb.DuckDBPyConnection,
    source: SourceFile,
    raw_dir: Path,
    *,
    lite: bool,
) -> int:
    """Create one table from its CSV and return the row count loaded."""
    csv_path = raw_dir / source.filename
    if not csv_path.exists():
        raise FileNotFoundError(f"expected {csv_path}")

    size_mb = csv_path.stat().st_size / 1e6
    label = f"load {source.table:<22} ({size_mb:>7.1f} MB)"

    where = ""
    if lite and source.table in _LITE_FILTERS:
        where = f" WHERE {_LITE_FILTERS[source.table]}"

    with log_duration(log, label):
        conn.execute(f'DROP TABLE IF EXISTS "{source.table}"')
        try:
            conn.execute(
                f'CREATE TABLE "{source.table}" AS SELECT * FROM '
                f"{_read_csv_expr(csv_path)}{where}"
            )
        except duckdb.Error as exc:
            if not _is_encoding_error(exc):
                raise
            # HomeCredit_columns_description.csv is cp1252, not UTF-8: it
            # carries curly quotes. DuckDB speaks only utf-8/utf-16/latin-1,
            # and cp1252's 0x93/0x94 are undefined in strict latin-1, so the
            # file has to be transcoded in Python before DuckDB can read it.
            log.warning("%s is not valid UTF-8, transcoding", source.filename)
            utf8_path = _transcode_to_utf8(csv_path)
            conn.execute(
                f'CREATE TABLE "{source.table}" AS SELECT * FROM '
                f"{_read_csv_expr(utf8_path)}{where}"
            )

    return conn.execute(f'SELECT count(*) FROM "{source.table}"').fetchone()[0]


def _build_lite_client_sample(
    conn: duckdb.DuckDBPyConnection, raw_dir: Path
) -> int:
    """Pick a TARGET-stratified sample of clients before loading child tables.

    Stratified so the sampled default rate matches the population's. Seeded so
    a `lite` build is reproducible and two evaluators see the same numbers.
    """
    train_csv = raw_dir / "application_train.csv"
    conn.execute("DROP TABLE IF EXISTS _lite_clients")
    conn.execute(
        f"""
        CREATE TABLE _lite_clients AS
        WITH src AS (SELECT SK_ID_CURR, TARGET FROM {_read_csv_expr(train_csv)}),
             ranked AS (
                 SELECT SK_ID_CURR, TARGET,
                        row_number() OVER (
                            PARTITION BY TARGET ORDER BY hash(SK_ID_CURR * 2654435761)
                        ) AS rn,
                        count(*) OVER (PARTITION BY TARGET) AS stratum_n,
                        count(*) OVER () AS total_n
                 FROM src
             )
        SELECT SK_ID_CURR, TARGET FROM ranked
        WHERE rn <= greatest(1, cast(round({LITE_SAMPLE_SIZE} * stratum_n / total_n) AS BIGINT))
        """
    )
    n = conn.execute("SELECT count(*) FROM _lite_clients").fetchone()[0]
    rate = conn.execute("SELECT avg(TARGET) FROM _lite_clients").fetchone()[0]
    log.info("lite sample: %s clients, default rate %.4f", f"{n:,}", rate)
    return n


def _create_indexes(conn: duckdb.DuckDBPyConnection) -> None:
    """Index the join keys. Skipped silently for tables that were not loaded."""
    present = set(list_tables(conn))
    with log_duration(log, "create indexes"):
        for table, keys in INDEXED_KEYS.items():
            if table not in present:
                continue
            for key in keys:
                conn.execute(
                    f'CREATE INDEX IF NOT EXISTS "idx_{table}_{key.lower()}" '
                    f'ON "{table}" ("{key}")'
                )


def _verify_row_counts(
    conn: duckdb.DuckDBPyConnection, *, lite: bool
) -> list[str]:
    """Compare loaded counts against Kaggle's published figures.

    A truncated or partially-extracted download is otherwise completely silent
    and would corrupt every downstream number, so this runs on every build.
    Skipped in `lite` mode, where smaller counts are the whole point.
    """
    problems: list[str] = []
    if lite:
        log.info("row-count verification skipped (lite mode samples the data)")
        return problems

    counts = table_row_counts(conn)
    for source in SOURCE_FILES:
        if source.expected_rows is None or source.filename in SKIP_FILES:
            continue
        actual = counts.get(source.table)
        if actual is None:
            continue
        if actual != source.expected_rows:
            problems.append(
                f"{source.table}: loaded {actual:,}, "
                f"expected {source.expected_rows:,}"
            )
        else:
            log.info("  verified %-22s %12s rows", source.table, f"{actual:,}")
    return problems


def dump_schema(conn: duckdb.DuckDBPyConnection, path: Path) -> Path:
    """Write the built schema to `sql/schema.sql` as generated documentation.

    Generated rather than hand-maintained so the committed DDL cannot drift
    away from the database the platform actually queries.
    """
    lines = [
        "-- Home Credit Default Risk - DuckDB schema",
        "-- GENERATED by `python -m src.data.loader`. Do not edit by hand.",
        "--",
        "-- Types are inferred from a full scan of each CSV (sample_size=-1),",
        "-- so this file records what was actually built, not an intention.",
        "",
    ]
    for table in list_tables(conn):
        if table.startswith("_"):
            continue
        cols = conn.execute(
            "SELECT column_name, data_type, is_nullable "
            "FROM information_schema.columns "
            "WHERE table_schema='main' AND table_name=? "
            "ORDER BY ordinal_position",
            [table],
        ).fetchall()
        n = conn.execute(f'SELECT count(*) FROM "{table}"').fetchone()[0]
        lines.append(f"-- {n:,} rows, {len(cols)} columns")
        lines.append(f'CREATE TABLE "{table}" (')
        body = [
            f'    "{c}" {t}{"" if nullable == "YES" else " NOT NULL"}'
            for c, t, nullable in cols
        ]
        lines.append(",\n".join(body))
        lines.append(");")
        lines.append("")

    for table, keys in INDEXED_KEYS.items():
        for key in keys:
            lines.append(
                f'CREATE INDEX "idx_{table}_{key.lower()}" ON "{table}" ("{key}");'
            )
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    log.info("wrote schema to %s", path)
    return path


def build_database(
    settings: Settings | None = None,
    *,
    mode: str | None = None,
    force: bool = False,
) -> dict[str, int]:
    """Build the DuckDB database from the raw CSVs. Returns row counts."""
    settings = settings or get_settings()
    lite = (mode or settings.data_mode) == "lite"

    if database_exists(settings) and not force:
        log.info("database already exists at %s (use --force to rebuild)",
                 settings.duckdb_path)
        with writable_connection(settings) as conn:
            return table_row_counts(conn)

    raw_dir = ensure_dataset()
    started = time.perf_counter()

    if force and Path(settings.duckdb_path).exists():
        Path(settings.duckdb_path).unlink()
        log.info("removed existing database for a clean rebuild")

    clear_database_ready(settings)

    log.info("building DuckDB (%s mode) at %s",
             "lite" if lite else "full", settings.duckdb_path)

    with writable_connection(settings) as conn:
        if lite:
            _build_lite_client_sample(conn, raw_dir)

        for source in SOURCE_FILES:
            if source.filename in SKIP_FILES:
                continue
            rows = _load_table(conn, source, raw_dir, lite=lite)
            log.info("  -> %-22s %12s rows", source.table, f"{rows:,}")

        if lite:
            conn.execute("DROP TABLE IF EXISTS _lite_clients")

        _create_indexes(conn)

        problems = _verify_row_counts(conn, lite=lite)
        if problems:
            raise RuntimeError(
                "Row counts do not match the published dataset - the download "
                "is likely incomplete:\n  " + "\n  ".join(problems)
            )

        dump_schema(conn, PROJECT_ROOT / "sql" / "schema.sql")
        counts = table_row_counts(conn)

    # Written only after the writable connection above has closed (so the
    # file is fully flushed) and every check has passed - see
    # database_exists()'s docstring for why this, not file existence, is
    # what every reader actually checks before querying.
    mark_database_ready(settings)

    size_gb = Path(settings.duckdb_path).stat().st_size / 1e9
    log.info(
        "database built in %.1fs: %d tables, %s rows, %.2f GB",
        time.perf_counter() - started,
        len(counts),
        f"{sum(counts.values()):,}",
        size_gb,
    )
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the DuckDB database.")
    parser.add_argument("--mode", choices=("full", "lite"),
                        help="override DATA_MODE from the environment")
    parser.add_argument("--force", action="store_true",
                        help="rebuild even if the database already exists")
    args = parser.parse_args()

    configure_logging()
    build_database(mode=args.mode, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
