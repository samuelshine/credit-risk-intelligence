"""FastAPI app factory.

    uvicorn src.api.main:app --host 0.0.0.0 --port 8000

One process serves both the JSON API (under `/api/*`) and the static UI
(`ui/`) - no separate frontend server, no CORS to configure, and identical
behaviour locally and in the Docker image.

Startup does not require the database or the trained model to exist: the app
comes up regardless, and each route reports its own "not ready yet" state.
This matters most on Render's first boot, where the dataset may still be
downloading in the background (see docs/DEPLOYMENT.md) - `/health` has to
answer immediately so the platform's health check doesn't kill the container
mid-ingest.
"""

from __future__ import annotations

import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from src.api.schemas import HealthResponse
from src.data.database import database_exists
from src.llm.gemini import get_llm_client
from src.ml.predict import get_risk_model
from src.utils.config import Settings, get_settings
from src.utils.logger import configure_logging, get_logger

log = get_logger(__name__)

#: ui/ is a sibling of src/, both under the repo root.
UI_DIR = Path(__file__).resolve().parents[2] / "ui"


def _ingest_in_background(settings: Settings) -> None:
    """Build the database in a daemon thread, without blocking startup.

    Only used where there is no companion `etl` container to do it - Render
    runs a single web service, so the API has to ingest for itself (see
    `Settings.auto_ingest_on_startup`).

    A thread rather than an asyncio task because `build_database()` is
    blocking, CPU- and IO-bound DuckDB work; on the event loop it would stall
    every request for the duration, which is the exact opposite of the point.
    Failures are logged and swallowed: a platform health check must keep
    getting a 200 from `/health` even if ingestion fails, and every
    DB-dependent route already degrades to a clear 503 on its own.
    """
    def run() -> None:
        from src.data.loader import build_database

        try:
            log.info("no database found; starting background ingestion")
            build_database(settings)
            log.info("background ingestion finished; database is ready")
        except Exception:
            log.exception(
                "background ingestion failed - the API stays up, but the "
                "chatbot and applicant lookup will report 503 until a "
                "database exists"
            )

    threading.Thread(target=run, name="etl", daemon=True).start()


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    settings = get_settings()
    log.info("starting API: %s", settings.describe())
    if settings.auto_ingest_on_startup and not database_exists(settings):
        _ingest_in_background(settings)
    yield
    log.info("shutting down API")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Credit Risk Intelligence Platform",
        version="1.0.0",
        lifespan=lifespan,
    )

    from src.api.routes import ask, eda, explain, rules, score

    app.include_router(eda.router, prefix="/api/eda", tags=["eda"])
    app.include_router(score.router, prefix="/api", tags=["score"])
    app.include_router(explain.router, prefix="/api", tags=["explain"])
    app.include_router(rules.router, prefix="/api", tags=["rules"])
    app.include_router(ask.router, prefix="/api", tags=["ask"])

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        settings = get_settings()
        client = get_llm_client()
        model = get_risk_model()
        return HealthResponse(
            status="ok",
            database_ready=database_exists(settings),
            model_ready=model.is_trained,
            llm_ready=client.available,
            data_mode=settings.data_mode,
            resolved_sql_model=(
                client.resolve_model("sql") if client.available else None
            ),
            resolved_summary_model=(
                client.resolve_model("summary") if client.available else None
            ),
        )

    # Static UI last: FastAPI matches routes in registration order, and a
    # catch-all StaticFiles mount at "/" would otherwise shadow /api/* and
    # /health if registered first.
    if UI_DIR.exists():
        app.mount("/", StaticFiles(directory=str(UI_DIR), html=True), name="ui")
    else:
        log.warning("ui/ directory not found at %s; UI will not be served", UI_DIR)

    return app


app = create_app()
