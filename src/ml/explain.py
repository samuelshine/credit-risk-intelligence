"""Explainability: SHAP for the "why", Gemini for the plain-English sentence.

Two explanations, computed differently on purpose:

**Global** (which features matter across the whole portfolio) is computed
once, offline, over a sample of the training data, and cached to
`models/eda_artifacts.json`-adjacent `models/shap_global.json`. It answers
"what does this model generally weigh" for the UI's model-transparency view.

**Local** (why did *this* applicant get *this* score) is computed on demand,
per request - it has to be, since it depends on the specific applicant.

SHAP explains the model's raw log-odds output, not the calibrated probability
the user sees (see `RiskModel.booster`'s docstring for why - calibration is a
monotone remap fit after the fact and has no per-feature decomposition).
Contributions are therefore always described as directional ("pushed the risk
up/down"), never as an exact slice of the displayed percentage - the honest
version of the story rather than a precise-looking but false one.

The narrative is grounded the same way the chatbot's answers are: the model
sees only the computed contributions, is told to name nothing else, and never
sees the database.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.llm.gemini import LLMUnavailable, get_llm_client
from src.ml.predict import ModelNotTrained, RiskModel, get_risk_model
from src.talk_to_data.prompt_templates import (
    EXPLANATION_SYSTEM_PROMPT,
    EXPLANATION_USER_PROMPT,
)
from src.utils.config import get_settings
from src.utils.helpers import humanise_column, write_json
from src.utils.logger import get_logger, log_duration

log = get_logger(__name__)

#: How many features are surfaced per local explanation. Enough to tell a
#: coherent story, few enough to stay readable - SHAP itself often assigns a
#: small nonzero weight to dozens of features, most too small to matter.
TOP_N_FACTORS = 6

#: Rows sampled for the global explanation. Exact SHAP values for 307k rows
#: would take minutes; 5,000 rows is enough for a stable importance ranking
#: and finishes in seconds.
GLOBAL_SAMPLE_SIZE = 5_000


@dataclass
class FeatureContribution:
    """One feature's signed push on one applicant's score."""

    feature: str
    label: str
    value: Any
    shap_value: float
    direction: str  # "increases_risk" | "decreases_risk"

    def as_dict(self) -> dict[str, Any]:
        return {
            "feature": self.feature,
            "label": self.label,
            "value": _display_value(self.value),
            "shap_value": round(self.shap_value, 4),
            "direction": self.direction,
        }


@dataclass
class LocalExplanation:
    sk_id_curr: int | None
    base_value: float
    contributions: list[FeatureContribution]
    narrative: str | None = None
    narrative_error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "sk_id_curr": self.sk_id_curr,
            "base_value": round(self.base_value, 4),
            "top_factors": [c.as_dict() for c in self.contributions],
            "narrative": self.narrative,
            "narrative_error": self.narrative_error,
        }


def _display_value(value: Any) -> Any:
    """Round a feature's raw value for display, leaving non-numerics alone."""
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return None
        return round(float(value), 2)
    if pd.isna(value):
        return None
    return str(value)


class Explainer:
    """SHAP over the loaded `RiskModel`. One `TreeExplainer`, built once."""

    def __init__(self, model: RiskModel | None = None) -> None:
        self._model = model or get_risk_model()
        self._explainer = None
        self._lock = threading.Lock()

    def _ensure_explainer(self):
        if self._explainer is not None:
            return self._explainer
        with self._lock:
            if self._explainer is None:
                import shap  # lazy: keeps import cost off API startup

                with log_duration(log, "build SHAP TreeExplainer"):
                    self._explainer = shap.TreeExplainer(self._model.booster)
        return self._explainer

    def explain_frame(self, frame: pd.DataFrame) -> tuple[np.ndarray, float]:
        """Raw SHAP values and base value for an already-aligned frame."""
        explainer = self._ensure_explainer()
        aligned = self._model.align_frame(frame)
        shap_values = explainer.shap_values(aligned)
        base_value = explainer.expected_value
        if isinstance(base_value, (list, np.ndarray)):
            base_value = base_value[0]
        return np.asarray(shap_values), float(base_value)

    def explain_one(
        self, frame: pd.DataFrame, *, sk_id_curr: int | None = None,
        top_n: int = TOP_N_FACTORS,
    ) -> LocalExplanation:
        """Explain a single applicant's row (`frame` must have exactly one row)."""
        if len(frame) != 1:
            raise ValueError("explain_one expects a single-row frame")

        shap_values, base_value = self.explain_frame(frame)
        aligned = self._model.align_frame(frame)
        row_values = shap_values[0]

        order = np.argsort(-np.abs(row_values))[:top_n]
        contributions = [
            FeatureContribution(
                feature=aligned.columns[i],
                label=humanise_column(aligned.columns[i]),
                value=aligned.iloc[0, i],
                shap_value=float(row_values[i]),
                direction="increases_risk" if row_values[i] > 0 else "decreases_risk",
            )
            for i in order
        ]
        return LocalExplanation(
            sk_id_curr=sk_id_curr, base_value=base_value, contributions=contributions,
        )

    def global_importance(self, sample_size: int = GLOBAL_SAMPLE_SIZE) -> dict[str, Any]:
        """Mean |SHAP| across a sample - the "what matters overall" ranking."""
        from src.data.database import get_readonly_connection
        from src.data.features import load_features

        with log_duration(log, f"global SHAP over {sample_size:,} rows"):
            matrix = load_features(
                "application_train", conn=get_readonly_connection(), limit=sample_size
            )
            shap_values, base_value = self.explain_frame(matrix.frame)
            mean_abs = np.abs(shap_values).mean(axis=0)
            aligned = self._model.align_frame(matrix.frame)

        ranking = sorted(
            zip(aligned.columns, mean_abs), key=lambda pair: -pair[1]
        )
        return {
            "base_value": round(base_value, 4),
            "sample_size": len(matrix),
            "ranking": [
                {"feature": name, "label": humanise_column(name),
                 "mean_abs_shap": round(float(value), 5)}
                for name, value in ranking[:30]
            ],
        }


def _format_factors_for_prompt(contributions: list[FeatureContribution]) -> str:
    lines = []
    for c in contributions:
        direction = "pushes risk UP" if c.shap_value > 0 else "pushes risk DOWN"
        lines.append(f"- {c.label} = {_display_value(c.value)}: {direction} "
                     f"(SHAP {c.shap_value:+.3f})")
    return "\n".join(lines)


def narrate(explanation: LocalExplanation, *, probability: float, band: str) -> str:
    """Turn a local explanation into 2-4 plain-English sentences via Gemini.

    Grounded the same way the chatbot is: the model sees only the computed
    contributions and the two summary numbers, never the database or the
    model's internals, and is instructed to name nothing else.
    """
    client = get_llm_client()
    if not client.available:
        raise LLMUnavailable(
            "GOOGLE_API_KEY is not set, so the plain-English explanation is "
            "unavailable. The numeric factors below are still exact."
        )
    user = EXPLANATION_USER_PROMPT.format(
        probability=probability, band=band, base_rate=explanation.base_value,
        factors=_format_factors_for_prompt(explanation.contributions),
    )
    response = client.generate(
        role="summary", system=EXPLANATION_SYSTEM_PROMPT, user=user,
        max_output_tokens=300,
    )
    return response.text


def explain_and_narrate(
    frame: pd.DataFrame, *, probability: float, band: str,
    sk_id_curr: int | None = None, explainer: Explainer | None = None,
) -> LocalExplanation:
    """The full local-explanation path the API's `/api/explain` route calls:
    SHAP contributions, then a narrative from them. Degrades gracefully if
    no LLM key is configured - the numeric factors are still returned."""
    explainer = explainer or get_explainer()
    explanation = explainer.explain_one(frame, sk_id_curr=sk_id_curr)
    try:
        explanation.narrative = narrate(explanation, probability=probability, band=band)
    except LLMUnavailable as exc:
        explanation.narrative_error = str(exc)
    return explanation


def build_and_save_global_importance() -> dict[str, Any]:
    """Compute the global ranking once (e.g. right after training) and save
    it - the API's model-transparency view reads the file, never recomputes."""
    settings = get_settings()
    explainer = get_explainer()
    result = explainer.global_importance()
    write_json(settings.models_dir / "shap_global.json", result)
    log.info("wrote global SHAP importance for %d features", len(result["ranking"]))
    return result


_explainer: Explainer | None = None
_explainer_lock = threading.Lock()


def get_explainer() -> Explainer:
    global _explainer
    if _explainer is None:
        with _explainer_lock:
            if _explainer is None:
                _explainer = Explainer()
    return _explainer
