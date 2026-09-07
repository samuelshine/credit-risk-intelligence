"""Small shared utilities.

Deliberately narrow: formatting for human-facing output, and JSON handling that
survives the numpy/pandas scalar types our pipelines produce. Anything with real
domain logic belongs in the module that owns that domain, not here.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# JSON
# --------------------------------------------------------------------------- #
def to_jsonable(obj: Any) -> Any:
    """Recursively convert numpy/pandas scalars into plain Python.

    `json.dumps` rejects `np.float32`, `np.int64`, `NaT` and friends, all of
    which fall out of DuckDB -> pandas -> sklearn naturally. Non-finite floats
    become None because `NaN` is not valid JSON and browsers reject it.
    """
    if obj is None or isinstance(obj, (str, bool)):
        return obj
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        value = float(obj)
        return value if math.isfinite(value) else None
    if isinstance(obj, (np.ndarray,)):
        return [to_jsonable(x) for x in obj.tolist()]
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    if obj is pd.NaT:
        return None
    if isinstance(obj, pd.Series):
        return to_jsonable(obj.to_dict())
    if isinstance(obj, pd.DataFrame):
        return [to_jsonable(rec) for rec in obj.to_dict(orient="records")]
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(x) for x in obj]
    return str(obj)


def write_json(path: Path, payload: Any, *, indent: int = 2) -> Path:
    """Write JSON atomically, creating parent directories as needed.

    Atomic because the API reads these artifacts while training may be
    rewriting them; a half-written file would surface as a 500.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(to_jsonable(payload), indent=indent), encoding="utf-8")
    tmp.replace(path)
    return path


def read_json(path: Path, default: Any = None) -> Any:
    """Read JSON, returning `default` when the file is absent."""
    path = Path(path)
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# Human-facing formatting
# --------------------------------------------------------------------------- #
def format_currency(value: float | None) -> str:
    """Home Credit amounts are unitless in the source data.

    We render them as plain grouped numbers rather than inventing a currency
    symbol the dataset does not actually specify.
    """
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "n/a"
    return f"{value:,.0f}"


def format_pct(fraction: float | None, decimals: int = 1) -> str:
    """0.0807 -> '8.1%'."""
    if fraction is None or (isinstance(fraction, float) and not math.isfinite(fraction)):
        return "n/a"
    return f"{fraction * 100:.{decimals}f}%"


def days_to_years(days: float | None) -> float | None:
    """Home Credit stores DAYS_* as negative offsets from the application date.

    `DAYS_BIRTH = -12005` is a 32.9-year-old applicant. Returns a positive
    magnitude in years, which is what every human-facing surface wants.
    """
    if days is None or (isinstance(days, float) and not math.isfinite(days)):
        return None
    return abs(float(days)) / 365.25


def humanise_column(name: str) -> str:
    """Turn a raw Home Credit column name into readable words.

    'AMT_INCOME_TOTAL' -> 'Amt income total'. Used as the fallback label when a
    column has no curated business description; the curated map in
    `src/eda/analysis.py` takes precedence wherever one exists.
    """
    return name.replace("_", " ").strip().capitalize()
