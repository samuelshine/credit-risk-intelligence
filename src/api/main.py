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

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from src.api.schemas import HealthResponse
from src.data.database import database_exists
from src.llm.gemini import get_llm_client
from src.ml.predict import get_risk_model
from src.utils.config import get_settings
from src.utils.logger import configure_logging, get_logger

log = get_logger(__name__)

#: ui/ is a sibling of src/, both under the repo root.
UI_DIR = Path(__file__).resolve().parents[2] / "ui"


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    settings = get_settings()
    log.info("starting API: %s", settings.describe())
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
