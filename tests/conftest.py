"""Shared fixtures.

`realistic_catalog` mirrors the real Home Credit schema closely enough to
measure prompt size against: the same seven tables, the same row counts, and
the real column names. The wide blocks in `application_train` (the 42 building
statistics, the 20 document flags) are generated from their real naming
patterns rather than typed out, because it is their *shape* that matters for
token measurement.

Numbers reported from this fixture are re-measured against the real database
once the ETL has run; see docs/PROMPTS.md.
"""

from __future__ import annotations

import os
from typing import Iterator

import duckdb
import pytest

from src.talk_to_data.catalog import Catalog, ColumnInfo, TableInfo

_ENV_KEYS = ("DUCKDB_PATH", "DATA_DIR", "DUCKDB_MEMORY_LIMIT")


@pytest.fixture
def isolated_db(tmp_path_factory) -> Iterator[duckdb.DuckDBPyConnection]:
    """A private DuckDB database, isolated from every other test's state.

    Several test modules point the process-wide config and the cached
    read-only connection at a shared temp database via environment variables.
    A test that needs its *own* one-off database - a specific fixture shape a
    regression test depends on - must not leave that global state pointed at
    its private database when it finishes, or every test that runs after it
    in the same module silently queries the wrong data. This fixture does the
    save/mutate/restore so individual tests don't have to get it right by hand.
    """
    from src.utils.config import get_settings
    from src.data.database import reset_readonly_connection, writable_connection

    saved_env = {key: os.environ.get(key) for key in _ENV_KEYS}
    tmp = tmp_path_factory.mktemp("isolated")
    os.environ["DUCKDB_PATH"] = str(tmp / "t.duckdb")
    os.environ["DATA_DIR"] = str(tmp)
    os.environ["DUCKDB_MEMORY_LIMIT"] = "1GB"
    get_settings.cache_clear()
    reset_readonly_connection()

    with writable_connection() as conn:
        yield conn

    reset_readonly_connection()
    for key, value in saved_env.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    get_settings.cache_clear()

_APPLICATION_CORE = [
    ("SK_ID_CURR", "BIGINT"), ("TARGET", "BIGINT"),
    ("NAME_CONTRACT_TYPE", "VARCHAR"), ("CODE_GENDER", "VARCHAR"),
    ("FLAG_OWN_CAR", "VARCHAR"), ("FLAG_OWN_REALTY", "VARCHAR"),
    ("CNT_CHILDREN", "BIGINT"), ("AMT_INCOME_TOTAL", "DOUBLE"),
    ("AMT_CREDIT", "DOUBLE"), ("AMT_ANNUITY", "DOUBLE"),
    ("AMT_GOODS_PRICE", "DOUBLE"), ("NAME_TYPE_SUITE", "VARCHAR"),
    ("NAME_INCOME_TYPE", "VARCHAR"), ("NAME_EDUCATION_TYPE", "VARCHAR"),
    ("NAME_FAMILY_STATUS", "VARCHAR"), ("NAME_HOUSING_TYPE", "VARCHAR"),
    ("REGION_POPULATION_RELATIVE", "DOUBLE"), ("DAYS_BIRTH", "BIGINT"),
    ("DAYS_EMPLOYED", "BIGINT"), ("DAYS_REGISTRATION", "DOUBLE"),
    ("DAYS_ID_PUBLISH", "BIGINT"), ("OWN_CAR_AGE", "DOUBLE"),
    ("OCCUPATION_TYPE", "VARCHAR"), ("CNT_FAM_MEMBERS", "DOUBLE"),
    ("REGION_RATING_CLIENT", "BIGINT"),
    ("REGION_RATING_CLIENT_W_CITY", "BIGINT"),
    ("WEEKDAY_APPR_PROCESS_START", "VARCHAR"),
    ("HOUR_APPR_PROCESS_START", "BIGINT"),
    ("ORGANIZATION_TYPE", "VARCHAR"), ("EXT_SOURCE_1", "DOUBLE"),
    ("EXT_SOURCE_2", "DOUBLE"), ("EXT_SOURCE_3", "DOUBLE"),
    ("OBS_30_CNT_SOCIAL_CIRCLE", "DOUBLE"),
    ("DEF_30_CNT_SOCIAL_CIRCLE", "DOUBLE"),
    ("DAYS_LAST_PHONE_CHANGE", "DOUBLE"),
]

_BUILDING_STATS = [
    "APARTMENTS", "BASEMENTAREA", "YEARS_BEGINEXPLUATATION", "YEARS_BUILD",
    "COMMONAREA", "ELEVATORS", "ENTRANCES", "FLOORSMAX", "FLOORSMIN",
    "LANDAREA", "LIVINGAPARTMENTS", "LIVINGAREA", "NONLIVINGAPARTMENTS",
    "NONLIVINGAREA",
]


def _application_columns() -> list[tuple[str, str]]:
    cols = list(_APPLICATION_CORE)
    for stat in _BUILDING_STATS:                       # 14 x 3 = 42 columns
        for suffix in ("AVG", "MODE", "MEDI"):
            cols.append((f"{stat}_{suffix}", "DOUBLE"))
    for n in range(2, 22):                             # 20 document flags
        cols.append((f"FLAG_DOCUMENT_{n}", "BIGINT"))
    for period in ("HOUR", "DAY", "WEEK", "MON", "QRT", "YEAR"):
        cols.append((f"AMT_REQ_CREDIT_BUREAU_{period}", "DOUBLE"))
    for flag in ("REG_REGION_NOT_LIVE_REGION", "REG_REGION_NOT_WORK_REGION",
                 "LIVE_REGION_NOT_WORK_REGION", "REG_CITY_NOT_LIVE_CITY",
                 "REG_CITY_NOT_WORK_CITY", "LIVE_CITY_NOT_WORK_CITY",
                 "FLAG_MOBIL", "FLAG_EMP_PHONE", "FLAG_WORK_PHONE",
                 "FLAG_CONT_MOBILE", "FLAG_PHONE", "FLAG_EMAIL"):
        cols.append((flag, "BIGINT"))
    return cols


_TABLES: dict[str, tuple[int, list[tuple[str, str]]]] = {
    "application_train": (307_511, _application_columns()),
    "bureau": (1_716_428, [
        ("SK_ID_CURR", "BIGINT"), ("SK_ID_BUREAU", "BIGINT"),
        ("CREDIT_ACTIVE", "VARCHAR"), ("CREDIT_CURRENCY", "VARCHAR"),
        ("DAYS_CREDIT", "BIGINT"), ("CREDIT_DAY_OVERDUE", "BIGINT"),
        ("DAYS_CREDIT_ENDDATE", "DOUBLE"), ("DAYS_ENDDATE_FACT", "DOUBLE"),
        ("AMT_CREDIT_MAX_OVERDUE", "DOUBLE"), ("CNT_CREDIT_PROLONG", "BIGINT"),
        ("AMT_CREDIT_SUM", "DOUBLE"), ("AMT_CREDIT_SUM_DEBT", "DOUBLE"),
        ("AMT_CREDIT_SUM_LIMIT", "DOUBLE"), ("AMT_CREDIT_SUM_OVERDUE", "DOUBLE"),
        ("CREDIT_TYPE", "VARCHAR"), ("DAYS_CREDIT_UPDATE", "BIGINT"),
        ("AMT_ANNUITY", "DOUBLE"),
    ]),
    "bureau_balance": (27_299_925, [
        ("SK_ID_BUREAU", "BIGINT"), ("MONTHS_BALANCE", "BIGINT"),
        ("STATUS", "VARCHAR"),
    ]),
    "previous_application": (1_670_214, [
        ("SK_ID_PREV", "BIGINT"), ("SK_ID_CURR", "BIGINT"),
        ("NAME_CONTRACT_TYPE", "VARCHAR"), ("AMT_ANNUITY", "DOUBLE"),
        ("AMT_APPLICATION", "DOUBLE"), ("AMT_CREDIT", "DOUBLE"),
        ("AMT_DOWN_PAYMENT", "DOUBLE"), ("AMT_GOODS_PRICE", "DOUBLE"),
        ("NAME_CONTRACT_STATUS", "VARCHAR"), ("DAYS_DECISION", "BIGINT"),
        ("CODE_REJECT_REASON", "VARCHAR"), ("NAME_CLIENT_TYPE", "VARCHAR"),
        ("NAME_GOODS_CATEGORY", "VARCHAR"), ("NAME_PORTFOLIO", "VARCHAR"),
        ("NAME_YIELD_GROUP", "VARCHAR"), ("CNT_PAYMENT", "DOUBLE"),
        ("DAYS_FIRST_DUE", "DOUBLE"), ("DAYS_LAST_DUE", "DOUBLE"),
        ("NFLAG_INSURED_ON_APPROVAL", "DOUBLE"),
    ]),
    "installments_payments": (13_605_401, [
        ("SK_ID_PREV", "BIGINT"), ("SK_ID_CURR", "BIGINT"),
        ("NUM_INSTALMENT_VERSION", "DOUBLE"), ("NUM_INSTALMENT_NUMBER", "BIGINT"),
        ("DAYS_INSTALMENT", "DOUBLE"), ("DAYS_ENTRY_PAYMENT", "DOUBLE"),
        ("AMT_INSTALMENT", "DOUBLE"), ("AMT_PAYMENT", "DOUBLE"),
    ]),
    "credit_card_balance": (3_840_312, [
        ("SK_ID_PREV", "BIGINT"), ("SK_ID_CURR", "BIGINT"),
        ("MONTHS_BALANCE", "BIGINT"), ("AMT_BALANCE", "DOUBLE"),
        ("AMT_CREDIT_LIMIT_ACTUAL", "BIGINT"),
        ("AMT_DRAWINGS_CURRENT", "DOUBLE"), ("AMT_PAYMENT_CURRENT", "DOUBLE"),
        ("AMT_TOTAL_RECEIVABLE", "DOUBLE"), ("CNT_DRAWINGS_CURRENT", "BIGINT"),
        ("NAME_CONTRACT_STATUS", "VARCHAR"), ("SK_DPD", "BIGINT"),
        ("SK_DPD_DEF", "BIGINT"),
    ]),
    "pos_cash_balance": (10_001_358, [
        ("SK_ID_PREV", "BIGINT"), ("SK_ID_CURR", "BIGINT"),
        ("MONTHS_BALANCE", "BIGINT"), ("CNT_INSTALMENT", "DOUBLE"),
        ("CNT_INSTALMENT_FUTURE", "DOUBLE"),
        ("NAME_CONTRACT_STATUS", "VARCHAR"), ("SK_DPD", "BIGINT"),
        ("SK_DPD_DEF", "BIGINT"),
    ]),
}


def create_empty_child_tables(conn) -> None:
    """Create every non-application table from `_TABLES` above, empty.

    The feature SQL (`src.data.features.build_feature_sql`) joins all six
    child tables regardless of what a test cares about, so any test that
    exercises it - directly, or indirectly via `RiskModel`/`Explainer`/
    `src.ml.rules` - needs every join target to exist with the real column
    set, even when only `application_train` itself is populated.
    """
    for table, (_, columns) in _TABLES.items():
        if table == "application_train":
            continue
        ddl = ", ".join(f'"{name}" {dtype}' for name, dtype in columns)
        conn.execute(f'CREATE TABLE "{table}" ({ddl})')


@pytest.fixture(scope="session")
def realistic_catalog() -> Catalog:
    from src.talk_to_data.catalog import TABLE_DESCRIPTIONS

    return Catalog(
        tables={
            name: TableInfo(
                name=name,
                row_count=rows,
                description=TABLE_DESCRIPTIONS.get(name, ""),
                columns={
                    c: ColumnInfo(name=c, data_type=t) for c, t in columns
                },
            )
            for name, (rows, columns) in _TABLES.items()
        }
    )
