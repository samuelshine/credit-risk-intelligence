"""Rules route: serve the precomputed surrogate-tree policy rules.

Like the EDA routes, this reads a file (`models/rules.json`) rather than
refitting the surrogate tree per request - the tree is a summary of the
trained model, not something that changes between requests.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from src.api.schemas import RulesResponse
from src.utils.config import get_settings
from src.utils.helpers import read_json

router = APIRouter()


@router.get("/rules", response_model=RulesResponse)
def get_rules() -> RulesResponse:
    settings = get_settings()
    data = read_json(settings.models_dir / "rules.json")
    if data is None:
        raise HTTPException(
            status_code=503,
            detail="Rules have not been derived yet. Run: python -m src.ml.rules",
        )
    return RulesResponse(**data)
