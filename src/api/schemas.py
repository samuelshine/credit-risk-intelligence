"""Pydantic request/response models for the API.

Kept separate from the route modules so the shape of the HTTP contract is
readable in one place, and so `src/ml`/`src/talk_to_data` never import
FastAPI - those modules work standalone (as the test suite proves) and the
API is a thin layer on top, not a place business logic lives.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator


class HealthResponse(BaseModel):
    status: str
    database_ready: bool
    model_ready: bool
    llm_ready: bool
    data_mode: str
    resolved_sql_model: str | None = None
    resolved_summary_model: str | None = None


class ApplicantSummary(BaseModel):
    """One row for the "pick an applicant" list in the UI."""

    sk_id_curr: int
    amt_income_total: float | None = None
    amt_credit: float | None = None
    code_gender: str | None = None
    name_education_type: str | None = None
    target: int | None = None  # only present for application_train rows


class ScoreRequest(BaseModel):
    """Either an existing applicant id, or a hand-entered set of fields -
    never both. `raw_fields` accepts any subset of application-table columns;
    anything omitted is treated as missing, exactly as it would be for a
    genuinely new applicant with no history in the child tables yet.
    """

    sk_id_curr: int | None = None
    raw_fields: dict[str, Any] | None = None

    @model_validator(mode="after")
    def _exactly_one_source(self) -> "ScoreRequest":
        if bool(self.sk_id_curr) == bool(self.raw_fields):
            raise ValueError(
                "Provide exactly one of sk_id_curr or raw_fields, not both "
                "and not neither."
            )
        return self


class ScoreResponse(BaseModel):
    sk_id_curr: int | None
    probability: float
    risk_band: str
    band_threshold: float
    base_rate: float
    lift_over_base_rate: float
    actual_target: int | None = Field(
        default=None,
        description="TARGET if this is a labelled training applicant, else null.",
    )


class FactorResponse(BaseModel):
    feature: str
    label: str
    value: Any
    shap_value: float
    direction: str


class ExplainResponse(BaseModel):
    sk_id_curr: int | None
    probability: float
    risk_band: str
    base_value: float
    top_factors: list[FactorResponse]
    narrative: str | None
    narrative_error: str | None = None


class RuleResponse(BaseModel):
    conditions: list[str]
    sentence: str
    population_share: float
    n_applicants: int
    observed_default_rate: float
    lift_over_base_rate: float
    mean_predicted_score: float


class RulesResponse(BaseModel):
    fidelity: dict[str, float]
    base_rate: float
    n_rules: int
    rules: list[RuleResponse]


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)


class AskResponse(BaseModel):
    question: str
    answer: str
    sql: str | None
    columns: list[str]
    rows: list[list[Any]]
    row_count: int
    truncated: bool
    refused: bool
    error: str | None
    repaired: bool
    tables_used: list[str]
    total_tokens: int
    duration_ms: int
