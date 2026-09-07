"""The database catalog: what tables and columns actually exist.

This is the ground truth the whole talk-to-data system is anchored to. It is
read from DuckDB's own `information_schema` at runtime, never hard-coded, which
means two things fall out for free:

* the **schema card** shown to the model describes the database that exists,
  not the one someone documented six months ago; and
* the **validator** can reject a hallucinated column by checking it against the
  same source of truth, rather than against a copy that could drift.

Business meanings come from `HomeCredit_columns_description.csv`, which Kaggle
ships with the competition. Using the official glossary rather than descriptions
we invent keeps the model's understanding of a column tied to what the data
publisher actually said it means.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache

import duckdb

from src.data.acquire import SOURCE_FILES
from src.data.database import get_readonly_connection
from src.utils.logger import get_logger

log = get_logger(__name__)

#: Table-level purpose, taken from the dataset's own file manifest.
TABLE_DESCRIPTIONS: dict[str, str] = {f.table: f.description for f in SOURCE_FILES}

#: Tables the chatbot is allowed to see at all. `column_descriptions` is
#: metadata about the schema rather than data to analyse, and exposing it
#: invites questions that are really about the glossary, not the portfolio.
CHATBOT_TABLES: tuple[str, ...] = (
    "application_train",
    "bureau",
    "bureau_balance",
    "previous_application",
    "installments_payments",
    "credit_card_balance",
    "pos_cash_balance",
)


@dataclass(frozen=True)
class ColumnInfo:
    """One column, with its type and (where known) its business meaning."""

    name: str
    data_type: str
    description: str = ""

    @property
    def is_identifier(self) -> bool:
        return self.name.startswith("SK_ID")

    @property
    def is_numeric(self) -> bool:
        return any(
            token in self.data_type.upper()
            for token in ("INT", "DOUBLE", "DECIMAL", "FLOAT", "REAL", "NUMERIC")
        )


@dataclass(frozen=True)
class TableInfo:
    """One table: its columns, size and purpose."""

    name: str
    columns: dict[str, ColumnInfo]
    row_count: int
    description: str = ""

    @property
    def column_names(self) -> set[str]:
        return set(self.columns)


@dataclass(frozen=True)
class Catalog:
    """Everything the SQL layer needs to know about the database."""

    tables: dict[str, TableInfo] = field(default_factory=dict)

    # -- lookups (case-insensitive: DuckDB identifiers are, and so are LLMs) --
    def _resolve_table(self, table: str) -> str | None:
        if table in self.tables:
            return table
        lowered = table.lower()
        for name in self.tables:
            if name.lower() == lowered:
                return name
        return None

    def has_table(self, table: str) -> bool:
        return self._resolve_table(table) is not None

    def has_column(self, table: str, column: str) -> bool:
        resolved = self._resolve_table(table)
        if resolved is None:
            return False
        cols = self.tables[resolved].columns
        return column in cols or column.lower() in {c.lower() for c in cols}

    def columns_of(self, table: str) -> set[str]:
        resolved = self._resolve_table(table)
        return self.tables[resolved].column_names if resolved else set()

    def tables_containing(self, column: str) -> list[str]:
        """Every table carrying this column - used to suggest a fix."""
        lowered = column.lower()
        return [
            name
            for name, info in self.tables.items()
            if lowered in {c.lower() for c in info.columns}
        ]

    def suggest_column(self, column: str, table: str | None = None) -> str | None:
        """Closest real column name, for a useful error rather than a bare no.

        Matches on normalised text (case and underscores removed), which catches
        the mistakes a language model actually makes - `amt_income` for
        `AMT_INCOME_TOTAL`, `gender` for `CODE_GENDER`.
        """
        candidates = (
            self.columns_of(table)
            if table and self.has_table(table)
            else {c for info in self.tables.values() for c in info.columns}
        )
        if not candidates:
            return None

        def norm(value: str) -> str:
            return re.sub(r"[^a-z0-9]", "", value.lower())

        target = norm(column)
        if not target:
            return None

        scored: list[tuple[int, str]] = []
        for candidate in candidates:
            normalised = norm(candidate)
            if normalised == target:
                return candidate
            if target in normalised or normalised in target:
                scored.append((abs(len(normalised) - len(target)), candidate))
        return min(scored)[1] if scored else None

    @property
    def chatbot_tables(self) -> list[str]:
        """Queryable tables, in the order a person would think about them."""
        return [t for t in CHATBOT_TABLES if t in self.tables]

    def total_rows(self) -> int:
        return sum(t.row_count for t in self.tables.values())


def _load_column_descriptions(
    conn: duckdb.DuckDBPyConnection,
) -> dict[tuple[str, str], str]:
    """Read Kaggle's column glossary into `(table, column) -> description`.

    The shipped CSV has an unnamed index column and names tables as
    `application_{train|test}.csv`, so both the column layout and the table
    names are normalised here rather than assumed.
    """
    try:
        cols = [
            r[0]
            for r in conn.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_schema='main' AND table_name='column_descriptions'"
            ).fetchall()
        ]
    except duckdb.Error:
        return {}
    if not cols:
        return {}

    def pick(*patterns: str) -> str | None:
        for pattern in patterns:
            for col in cols:
                if re.fullmatch(pattern, col, flags=re.IGNORECASE):
                    return col
        return None

    table_col, row_col, desc_col = pick("table"), pick("row"), pick("description")
    if not all((table_col, row_col, desc_col)):
        log.warning("column glossary has unexpected columns %s, skipping", cols)
        return {}

    rows = conn.execute(
        f'SELECT "{table_col}", "{row_col}", "{desc_col}" FROM column_descriptions'
    ).fetchall()

    out: dict[tuple[str, str], str] = {}
    for raw_table, raw_column, description in rows:
        if not raw_column or not description:
            continue
        for table in _normalise_glossary_table(str(raw_table)):
            out[(table, str(raw_column).strip())] = " ".join(str(description).split())
    return out


def _normalise_glossary_table(raw: str) -> list[str]:
    """Map a glossary table label onto our DuckDB table names.

    `application_{train|test}.csv` covers two tables; `POS_CASH_balance.csv`
    is stored as `pos_cash_balance`.
    """
    name = raw.strip().removesuffix(".csv").strip()
    if "{" in name:  # application_{train|test}
        prefix = name.split("{")[0].rstrip("_")
        variants = re.search(r"\{([^}]*)\}", name)
        if variants:
            return [f"{prefix}_{v.strip()}" for v in variants.group(1).split("|")]
    return [name.lower()]


def build_catalog(conn: duckdb.DuckDBPyConnection | None = None) -> Catalog:
    """Read the live catalog out of DuckDB."""
    conn = conn or get_readonly_connection()
    descriptions = _load_column_descriptions(conn)

    rows = conn.execute(
        "SELECT table_name, column_name, data_type "
        "FROM information_schema.columns "
        "WHERE table_schema = 'main' "
        "ORDER BY table_name, ordinal_position"
    ).fetchall()

    grouped: dict[str, dict[str, ColumnInfo]] = {}
    for table, column, data_type in rows:
        if table.startswith("_"):  # transient ETL scratch tables
            continue
        grouped.setdefault(table, {})[column] = ColumnInfo(
            name=column,
            data_type=data_type,
            description=descriptions.get((table, column), ""),
        )

    tables = {
        name: TableInfo(
            name=name,
            columns=columns,
            row_count=conn.execute(f'SELECT count(*) FROM "{name}"').fetchone()[0],
            description=TABLE_DESCRIPTIONS.get(name, ""),
        )
        for name, columns in grouped.items()
    }

    described = sum(1 for c in descriptions if c[0] in tables)
    log.info(
        "catalog: %d tables, %d columns (%d with glossary descriptions)",
        len(tables),
        sum(len(t.columns) for t in tables.values()),
        described,
    )
    return Catalog(tables=tables)


@lru_cache(maxsize=1)
def get_catalog() -> Catalog:
    """Process-wide catalog. The schema is static once the ETL has run."""
    return build_catalog()
