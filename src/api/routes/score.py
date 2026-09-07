"""Scoring routes: pick an applicant, or enter one, get a risk score.

Both paths share one response shape and one underlying `RiskModel` call, so
the UI cannot get a different notion of "the score" depending on which form
it used.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from src.api.schemas import ApplicantSummary, ScoreRequest, ScoreResponse
from src.data.database import database_exists, get_readonly_connection
from src.ml.predict import ModelNotTrained, get_risk_model
from src.utils.config import get_settings

router = APIRouter()


def _require_database() -> None:
    if not database_exists(get_settings()):
        raise HTTPException(
            status_code=503,
            detail="The database has not been built yet. Run: python -m src.data.loader",
        )


@router.get("/applicants/sample", response_model=list[ApplicantSummary])
def sample_applicants(n: int = Query(default=20, ge=1, le=100)) -> list[ApplicantSummary]:
    """A handful of real applicants for the UI's "pick one" picker.

    Sampled across the risk spectrum (via a hashed, seed-stable ordering)
    rather than the first N rows, so the picker isn't accidentally all
    low-risk or all one segment.
    """
    _require_database()
    conn = get_readonly_connection()
    rows = conn.execute(
        """
        SELECT SK_ID_CURR, AMT_INCOME_TOTAL, AMT_CREDIT, CODE_GENDER,
               NAME_EDUCATION_TYPE, TARGET
        FROM application_train
        ORDER BY hash(SK_ID_CURR * 2654435761)
        LIMIT ?
        """,
        [n],
    ).fetchall()
    columns = ["sk_id_curr", "amt_income_total", "amt_credit", "code_gender",
               "name_education_type", "target"]
    return [ApplicantSummary(**dict(zip(columns, row))) for row in rows]


@router.post("/score", response_model=ScoreResponse)
def score(request: ScoreRequest) -> ScoreResponse:
    model = get_risk_model()
    try:
        if request.sk_id_curr is not None:
            _require_database()
            conn = get_readonly_connection()
            prediction = model.predict_one(request.sk_id_curr, conn=conn)
            actual = conn.execute(
                "SELECT TARGET FROM application_train WHERE SK_ID_CURR = ?",
                [request.sk_id_curr],
            ).fetchone()
            actual_target = actual[0] if actual else None
        else:
            prediction = model.predict_from_raw(request.raw_fields)
            actual_target = None
    except ModelNotTrained as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return ScoreResponse(**prediction.as_dict(), actual_target=actual_target)
