"""EDA routes: serve the precomputed artifacts, never recompute on request.

`models/eda_artifacts.json` and `models/charts/*.png` are built once by
`scripts/build_eda_artifacts.py` (run after the ETL, or as part of the
Docker entrypoint). A 307k-row aggregation is cheap in DuckDB but is still
not something a browser tab should wait on - these routes just read a file.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from src.utils.config import get_settings
from src.utils.helpers import read_json

router = APIRouter()


def _artifacts() -> dict:
    settings = get_settings()
    path = settings.models_dir / "eda_artifacts.json"
    data = read_json(path)
    if data is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "EDA has not been computed yet. Run: "
                "python scripts/build_eda_artifacts.py"
            ),
        )
    return data


@router.get("/summary")
def get_summary() -> dict:
    """Dataset summary and feature categorisation."""
    data = _artifacts()
    return {
        "table_summary": data["table_summary"],
        "feature_categories": data["feature_categories"],
    }


@router.get("/quality")
def get_quality() -> dict:
    """Missing-value report and data-quality findings."""
    data = _artifacts()
    return {
        "missing_values": data["missing_values"],
        "data_quality_findings": data["data_quality_findings"],
    }


@router.get("/insights")
def get_insights() -> dict:
    """The business insights, each with its headline, so-what, and chart path."""
    data = _artifacts()
    return {
        "insights": [
            {**insight, "chart_url": f"/api/eda/charts/{insight['id']}.png"}
            for insight in data["insights"]
        ]
    }


@router.get("/charts/{filename}")
def get_chart(filename: str) -> FileResponse:
    """Serve one rendered chart PNG.

    `filename` is validated against the actual directory listing rather than
    trusted as a path - this is a public, unauthenticated route.
    """
    settings = get_settings()
    charts_dir = settings.charts_dir.resolve()
    candidate = (charts_dir / filename).resolve()

    if candidate.parent != charts_dir or not candidate.name.endswith(".png"):
        raise HTTPException(status_code=400, detail="Invalid chart filename.")
    if not candidate.exists():
        raise HTTPException(status_code=404, detail=f"Chart {filename!r} not found.")

    return FileResponse(candidate, media_type="image/png")
