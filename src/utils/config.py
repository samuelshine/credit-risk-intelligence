"""Central configuration.

Every tunable in the platform is read from the environment exactly once, here,
and reached elsewhere through `get_settings()`. Nothing else in `src/` calls
`os.environ`, so `.env.example` stays an honest, complete inventory of what the
application can be configured with.

Paths default to the *container* layout (`/app/...`). When running outside
Docker they are rewritten relative to the repository root, so the same code and
the same `.env` work in both places without a separate local config file.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Repository root: src/utils/config.py -> src/utils -> src -> <root>
PROJECT_ROOT = Path(__file__).resolve().parents[2]

#: Marks the in-container layout. Any configured path starting with this prefix
#: is re-anchored to PROJECT_ROOT when the app is not running inside the image.
_CONTAINER_PREFIX = Path("/app")


class Settings(BaseSettings):
    """Typed view over the environment. See `.env.example` for prose docs."""

    model_config = SettingsConfigDict(
        env_file=(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # -- LLM ------------------------------------------------------------------
    google_api_key: str = ""
    gemini_model_sql: str = "gemini-3.7-flash"
    gemini_model_summary: str = "gemini-3.5-flash-lite"

    # -- Dataset --------------------------------------------------------------
    kaggle_api_token: str = ""
    data_mode: str = "full"

    # -- Paths ----------------------------------------------------------------
    data_dir: Path = Path("/app/data")
    duckdb_path: Path = Path("/app/data/credit_risk.duckdb")
    models_dir: Path = Path("/app/models")

    # -- DuckDB ---------------------------------------------------------------
    duckdb_memory_limit: str = "8GB"
    duckdb_threads: int = 4

    # -- Talk-to-data guardrails ---------------------------------------------
    max_sql_rows: int = Field(default=500, ge=1, le=10_000)
    sql_timeout_seconds: int = Field(default=30, ge=1, le=300)
    max_summary_rows: int = Field(default=50, ge=1, le=500)

    # -- API ------------------------------------------------------------------
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "INFO"

    @field_validator("data_mode")
    @classmethod
    def _check_data_mode(cls, v: str) -> str:
        v = v.strip().lower()
        if v not in {"full", "lite"}:
            raise ValueError(f"DATA_MODE must be 'full' or 'lite', got {v!r}")
        return v

    @field_validator("log_level")
    @classmethod
    def _check_log_level(cls, v: str) -> str:
        v = v.strip().upper()
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v not in allowed:
            raise ValueError(f"LOG_LEVEL must be one of {sorted(allowed)}, got {v!r}")
        return v

    @field_validator("data_dir", "duckdb_path", "models_dir")
    @classmethod
    def _localise_path(cls, v: Path) -> Path:
        """Re-anchor container paths to the repo root when not in the container.

        Inside the image `/app` exists and the path is used as-is. Outside it,
        `/app/data/credit_risk.duckdb` becomes `<repo>/data/credit_risk.duckdb`.
        """
        if _CONTAINER_PREFIX.exists() or not v.is_absolute():
            return v
        try:
            return PROJECT_ROOT / v.relative_to(_CONTAINER_PREFIX)
        except ValueError:
            # An absolute path the operator chose deliberately; leave it alone.
            return v

    # -- Derived --------------------------------------------------------------
    @property
    def raw_data_dir(self) -> Path:
        """Where the unzipped Kaggle CSVs live."""
        return self.data_dir / "raw"

    @property
    def charts_dir(self) -> Path:
        """Rendered EDA chart images, served read-only by the API."""
        return self.models_dir / "charts"

    @property
    def llm_enabled(self) -> bool:
        """False when no key is configured.

        The API still starts and every non-LLM section keeps working; only the
        chatbot and the generated narrative degrade, with a clear message.
        """
        return bool(self.google_api_key.strip())

    def describe(self) -> dict[str, object]:
        """Config summary safe to log or expose - never includes secrets."""
        return {
            "data_mode": self.data_mode,
            "duckdb_path": str(self.duckdb_path),
            "models_dir": str(self.models_dir),
            "duckdb_memory_limit": self.duckdb_memory_limit,
            "duckdb_threads": self.duckdb_threads,
            "max_sql_rows": self.max_sql_rows,
            "llm_enabled": self.llm_enabled,
            "gemini_model_sql": self.gemini_model_sql if self.llm_enabled else None,
            "gemini_model_summary": self.gemini_model_summary if self.llm_enabled else None,
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings singleton."""
    return Settings()
