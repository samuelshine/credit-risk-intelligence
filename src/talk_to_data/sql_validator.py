"""Validation of model-generated SQL, by parsing rather than pattern-matching.

Every query the language model writes passes through here before it reaches the
database. The checks are performed on a parsed abstract syntax tree, because a
regex over SQL text is defeated by comments, string literals, whitespace and
casing, and gives no way to tell a column reference from a column *name inside
a string*.

Layered defence
---------------
This module is layer two of three. Layer one is the prompt, which is not a
security control at all - a language model can always be talked into emitting
something unintended. Layer three is the DuckDB connection itself, opened
read-only with `enable_external_access=false` (see `src.data.database`).

Layer three alone would be enough to keep the system *safe*, but not enough to
keep it *honest*: a query against a hallucinated column would reach the engine
and come back as an opaque "column not found" error. Catching it here means we
can name the mistake, suggest the real column, and hand the model a specific
correction to retry with.

What is checked
---------------
1. It parses as SQL at all, in the DuckDB dialect.
2. Exactly one statement - no `SELECT 1; DROP TABLE bureau`.
3. The statement is a read: `SELECT`, or a `WITH` wrapping one.
4. No write, DDL or administrative node anywhere in the tree.
5. No filesystem or network function. This is the check that matters most:
   `SELECT * FROM read_csv('/etc/passwd')` is a perfectly ordinary `SELECT`
   node, and only a function-name check distinguishes it.
6. Every table exists, and is one the chatbot is allowed to read.
7. Every column exists on the table it is attributed to, resolved through
   CTEs, subqueries and aliases.
8. A row limit is present, injected or clamped as needed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import sqlglot
from sqlglot import expressions as exp
from sqlglot.errors import ParseError
from sqlglot.optimizer.scope import build_scope

from src.talk_to_data.catalog import Catalog
from src.utils.logger import get_logger

log = get_logger(__name__)

DIALECT = "duckdb"

#: Node types that mutate data, change schema, or administer the server.
#: Presence of any of these anywhere in the tree fails validation outright.
FORBIDDEN_NODES: tuple[type[exp.Expression], ...] = (
    exp.Insert, exp.Update, exp.Delete, exp.Drop, exp.Create, exp.Alter,
    exp.TruncateTable, exp.Merge, exp.Copy, exp.Attach, exp.Detach,
    exp.Pragma, exp.Set, exp.Command, exp.Grant, exp.Use, exp.Transaction,
    exp.Commit, exp.Rollback, exp.AlterSet,
)

#: Functions that reach outside the database. DuckDB's read-only mode does not
#: restrict these - measured, not assumed - so they are refused by name.
#: Matching is on a normalised name, so `READ_CSV` and `read_csv_auto` both hit.
FORBIDDEN_FUNCTION_PATTERNS: tuple[str, ...] = (
    r"read_.*",          # read_csv, read_parquet, read_json, read_text, read_blob
    r".*_scan",          # parquet_scan, postgres_scan, sqlite_scan, iceberg_scan
    r"glob",
    r"sniff_csv",
    r"parquet_.*",
    r"delta_.*",
    r"iceberg_.*",
    r"getenv",
    r"install",
    r"load",
    r"which_secret",
    r"create_secret",
    r".*_secret.*",
)

_FORBIDDEN_FUNCTION_RE = re.compile(
    "|".join(f"(?:{p})" for p in FORBIDDEN_FUNCTION_PATTERNS), re.IGNORECASE
)


@dataclass
class ValidationResult:
    """Outcome of validating one query.

    `sql` is the rewritten query to execute - the original with a row limit
    applied. `errors` are phrased for two audiences at once: a person reading
    the UI, and the model reading them back on a retry, which is why they name
    the offending identifier and suggest a replacement.
    """

    ok: bool
    sql: str = ""
    errors: list[str] = field(default_factory=list)
    tables_used: list[str] = field(default_factory=list)
    limit_applied: int | None = None

    @property
    def error_message(self) -> str:
        return "; ".join(self.errors)


def _strip_markdown_fence(sql: str) -> str:
    """Remove a ```sql fence if the model wrapped its answer in one.

    Prompted against, but cheap to tolerate: rejecting a correct query over
    formatting would be a needless retry.
    """
    text = sql.strip()
    if not text.startswith("```"):
        return text
    text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
    return re.sub(r"\s*```$", "", text).strip()


def _check_forbidden_nodes(tree: exp.Expression) -> list[str]:
    for node in tree.walk():
        if isinstance(node, FORBIDDEN_NODES):
            name = type(node).__name__.upper()
            return [
                f"Only read-only SELECT queries are allowed, but this query "
                f"contains a {name} operation."
            ]
    return []


def _check_forbidden_functions(tree: exp.Expression) -> list[str]:
    errors: list[str] = []
    for node in tree.find_all(exp.Func, exp.Anonymous):
        name = (
            node.name
            if isinstance(node, exp.Anonymous)
            else getattr(node, "sql_name", lambda: type(node).__name__)()
        )
        if name and _FORBIDDEN_FUNCTION_RE.fullmatch(str(name)):
            errors.append(
                f"The function {name}() reads outside the database and is not "
                f"allowed. Query only the tables listed in the schema."
            )
    return errors


def _check_tables(
    tree: exp.Expression, catalog: Catalog, allowed: set[str]
) -> tuple[list[str], list[str]]:
    """Validate every real table reference. Returns (errors, tables_used)."""
    errors: list[str] = []
    used: list[str] = []

    # Names introduced by WITH are not physical tables and must be exempted.
    cte_names = {
        cte.alias_or_name.lower() for cte in tree.find_all(exp.CTE)
    }

    for table in tree.find_all(exp.Table):
        name = table.name
        if not name or name.lower() in cte_names:
            continue
        if not catalog.has_table(name):
            close = _closest(name, list(catalog.tables))
            hint = f" Did you mean {close}?" if close else ""
            errors.append(f"Table '{name}' does not exist.{hint}")
            continue
        if name.lower() not in allowed:
            errors.append(
                f"Table '{name}' is not available to the assistant. "
                f"Available tables: {', '.join(sorted(allowed))}."
            )
            continue
        if name not in used:
            used.append(name)

    if not used and not errors:
        errors.append("The query does not read from any known table.")
    return errors, used


def _check_columns(tree: exp.Expression, catalog: Catalog) -> list[str]:
    """Validate column references, resolved through CTEs and subqueries.

    sqlglot's scope analysis tells us, for each part of the query, which
    sources are visible and whether each is a physical table or a derived
    result. Columns attributed to a derived source are skipped: their names
    come from a projection we have already validated further down the tree.
    """
    errors: list[str] = []
    try:
        root = build_scope(tree)
    except Exception as exc:  # sqlglot raises various types on odd input
        log.debug("scope analysis unavailable (%s); skipping column check", exc)
        return errors
    if root is None:
        return errors

    for scope in root.traverse():
        # alias -> physical table name, for the sources in this scope only
        physical: dict[str, str] = {}
        has_derived = False
        for alias, source in scope.sources.items():
            if isinstance(source, exp.Table):
                physical[alias.lower()] = source.name
            else:
                has_derived = True

        for column in scope.columns:
            qualifier, name = column.table, column.name
            if not name or name == "*":
                continue

            if qualifier:
                table = physical.get(qualifier.lower())
                if table is None:
                    continue  # qualifies a CTE or subquery; validated in its own scope
                if not catalog.has_column(table, name):
                    errors.append(_unknown_column(catalog, name, table))
            else:
                if has_derived:
                    continue  # cannot attribute it to one physical table
                if any(catalog.has_column(t, name) for t in physical.values()):
                    continue
                if physical:
                    errors.append(_unknown_column(catalog, name, None))

    # Deduplicate while preserving order - one bad column can appear many times.
    return list(dict.fromkeys(errors))


def _unknown_column(catalog: Catalog, column: str, table: str | None) -> str:
    suggestion = catalog.suggest_column(column, table)
    where = f" on table '{table}'" if table else ""
    if suggestion:
        owners = catalog.tables_containing(suggestion)
        located = f" (in {owners[0]})" if owners and not table else ""
        return (
            f"Column '{column}' does not exist{where}. "
            f"Did you mean '{suggestion}'{located}?"
        )
    return f"Column '{column}' does not exist{where}."


def _closest(name: str, candidates: list[str]) -> str | None:
    target = re.sub(r"[^a-z0-9]", "", name.lower())
    for candidate in candidates:
        if re.sub(r"[^a-z0-9]", "", candidate.lower()) == target:
            return candidate
    for candidate in candidates:
        normalised = re.sub(r"[^a-z0-9]", "", candidate.lower())
        if target and (target in normalised or normalised in target):
            return candidate
    return None


def _apply_limit(tree: exp.Expression, max_rows: int) -> tuple[exp.Expression, int]:
    """Guarantee a bounded result set.

    A missing LIMIT on a 27M-row table would stream the whole table into the
    API process. An oversized one is clamped rather than rejected, since the
    model's intent was still a valid question.
    """
    target = tree.this if isinstance(tree, exp.Subquery) else tree

    existing = target.args.get("limit") if isinstance(target, exp.Query) else None
    if existing is not None:
        try:
            requested = int(existing.expression.name)
        except (AttributeError, ValueError):
            requested = max_rows
        if requested <= max_rows:
            return tree, requested

    return target.limit(max_rows), max_rows


def validate_sql(
    sql: str,
    catalog: Catalog,
    *,
    max_rows: int = 500,
    allowed_tables: set[str] | None = None,
) -> ValidationResult:
    """Validate and normalise one generated query.

    Returns a result rather than raising: the caller feeds `errors` back to the
    model for a single repair attempt, and a raised exception would make that
    control flow read as an error path rather than the expected one.
    """
    cleaned = _strip_markdown_fence(sql)
    if not cleaned:
        return ValidationResult(ok=False, errors=["The query is empty."])

    allowed = allowed_tables or {t.lower() for t in catalog.chatbot_tables}

    try:
        statements = [s for s in sqlglot.parse(cleaned, dialect=DIALECT) if s]
    except ParseError as exc:
        first = str(exc).splitlines()[0]
        return ValidationResult(ok=False, errors=[f"The query is not valid SQL: {first}"])

    if not statements:
        return ValidationResult(ok=False, errors=["The query is empty."])
    if len(statements) > 1:
        return ValidationResult(
            ok=False,
            errors=[
                f"Expected a single statement but found {len(statements)}. "
                f"Write one SELECT query."
            ],
        )

    tree = statements[0]

    errors = _check_forbidden_nodes(tree)
    if errors:
        return ValidationResult(ok=False, errors=errors)

    if not isinstance(tree, (exp.Select, exp.Union, exp.Subquery)):
        return ValidationResult(
            ok=False,
            errors=[
                f"Only SELECT queries are allowed, but this is a "
                f"{type(tree).__name__.upper()} statement."
            ],
        )

    errors += _check_forbidden_functions(tree)
    table_errors, tables_used = _check_tables(tree, catalog, allowed)
    errors += table_errors

    # Column checks only make sense once the tables resolve.
    if not table_errors:
        errors += _check_columns(tree, catalog)

    if errors:
        return ValidationResult(ok=False, errors=errors, tables_used=tables_used)

    limited, applied = _apply_limit(tree, max_rows)
    return ValidationResult(
        ok=True,
        sql=limited.sql(dialect=DIALECT, pretty=True),
        tables_used=tables_used,
        limit_applied=applied,
    )
