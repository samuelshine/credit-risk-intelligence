"""Explanation route: score an applicant, then explain the score.

Deliberately re-scores rather than accepting a probability from the client -
an explanation has to match a score this process actually computed, not one
handed to it, or the two could silently disagree (a stale score from an
earlier model version, a tampered value, a rounding difference).
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from src.api.schemas import ExplainResponse, FactorResponse, ScoreRequest
from src.data.database import database_exists, get_readonly_connection
from src.ml.explain import explain_and_narrate, get_explainer
from src.ml.predict import ModelNotTrained, get_risk_model, load_applicant_frame
from src.utils.config import get_settings

router = APIRouter()


@router.post("/explain", response_model=ExplainResponse)
def explain(request: ScoreRequest) -> ExplainResponse:
    model = get_risk_model()
    try:
        if request.sk_id_curr is not None:
            if not database_exists(get_settings()):
                raise HTTPException(
                    status_code=503,
                    detail="The database has not been built yet. Run: python -m src.data.loader",
                )
            conn = get_readonly_connection()
            frame = load_applicant_frame(conn, request.sk_id_curr)
            if frame.empty:
                raise HTTPException(
                    status_code=404,
                    detail=f"SK_ID_CURR {request.sk_id_curr} not found.",
                )
            sk_id_curr = request.sk_id_curr
        else:
            import pandas as pd

            frame = pd.DataFrame([request.raw_fields])
            sk_id_curr = None

        probability = float(model.predict_proba(frame)[0])
        band, _ = model.band_for(probability)

        explanation = explain_and_narrate(
            frame, probability=probability, band=band, base_rate=model.base_rate,
            sk_id_curr=sk_id_curr, explainer=get_explainer(),
        )
    except ModelNotTrained as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return ExplainResponse(
        sk_id_curr=sk_id_curr,
        probability=probability,
        risk_band=band,
        base_value=explanation.base_value,
        top_factors=[FactorResponse(**c.as_dict()) for c in explanation.contributions],
        narrative=explanation.narrative,
        narrative_error=explanation.narrative_error,
    )
