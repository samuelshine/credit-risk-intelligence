"""Tests for SHAP-based explanations, grounding, and the narrative prompt."""

from __future__ import annotations

import pandas as pd
import pytest

from tests.conftest_ml import build_tiny_trained_model, point_settings_at


@pytest.fixture
def explainer_and_data(tmp_path):
    point_settings_at(tmp_path)
    X, y = build_tiny_trained_model(tmp_path)
    from src.ml.explain import Explainer
    from src.ml.predict import RiskModel
    return Explainer(RiskModel()), X, y


def test_explain_one_rejects_multi_row_frames(explainer_and_data) -> None:
    explainer, X, _ = explainer_and_data
    with pytest.raises(ValueError, match="single-row"):
        explainer.explain_one(X.iloc[:2])


def test_explain_one_returns_top_n_signed_contributions(explainer_and_data) -> None:
    explainer, X, _ = explainer_and_data
    explanation = explainer.explain_one(X.iloc[[0]], sk_id_curr=42)

    assert explanation.sk_id_curr == 42
    assert 1 <= len(explanation.contributions) <= 6
    assert all(c.direction in {"increases_risk", "decreases_risk"} for c in explanation.contributions)
    # Sorted by |SHAP|, largest first.
    magnitudes = [abs(c.shap_value) for c in explanation.contributions]
    assert magnitudes == sorted(magnitudes, reverse=True)


def test_high_ext_source_mean_shows_up_as_risk_decreasing(explainer_and_data) -> None:
    """Grounding sanity check against the fixture's known, real signal."""
    explainer, _, _ = explainer_and_data
    protective_applicant = pd.DataFrame({
        "EXT_SOURCE_MEAN": [0.95], "AMT_INCOME_TOTAL": [150_000],
        "AMT_CREDIT": [300_000], "CODE_GENDER": pd.Categorical(["F"]),
    })
    explanation = explainer.explain_one(protective_applicant, top_n=4)
    ext_source = next(
        c for c in explanation.contributions if c.feature == "EXT_SOURCE_MEAN"
    )
    assert ext_source.direction == "decreases_risk"
    assert ext_source.shap_value < 0


def test_global_importance_ranks_the_real_driver_first(explainer_and_data) -> None:
    """EXT_SOURCE_MEAN is the fixture's only real signal; income/credit are
    pure noise. The global ranking must reflect that, not just run without
    error - a ranking that put noise first would be a silent regression."""
    from src.data.database import get_readonly_connection  # noqa: F401 (not used; import-load path only)

    explainer, X, y = explainer_and_data
    shap_values, base_value = explainer.explain_frame(X)
    import numpy as np
    mean_abs = dict(zip(X.columns, np.abs(shap_values).mean(axis=0)))
    top_feature = max(mean_abs, key=mean_abs.get)
    assert top_feature == "EXT_SOURCE_MEAN"


def test_explanation_serialises_only_computed_values(explainer_and_data) -> None:
    """Every value in as_dict() must trace back to something SHAP actually
    computed - the grounding guarantee the narrative prompt depends on."""
    explainer, X, _ = explainer_and_data
    explanation = explainer.explain_one(X.iloc[[0]])
    payload = explanation.as_dict()

    feature_names = {c["feature"] for c in payload["top_factors"]}
    assert feature_names <= set(X.columns)
    assert payload["narrative"] is None  # narrate() not called by explain_one alone


def test_narrate_prompt_uses_the_real_base_rate_not_shap_base_value(
    explainer_and_data, monkeypatch
) -> None:
    """Regression test: an earlier version passed explanation.base_value
    (SHAP's log-odds baseline, typically a negative number like -2.96) as
    the prompt's "Portfolio average" - which rendered as "-296.1%". The
    prompt must show the real portfolio default rate instead."""
    from src.ml.explain import narrate

    explainer, X, _ = explainer_and_data
    explanation = explainer.explain_one(X.iloc[[0]])
    assert explanation.base_value < 0  # sanity: it really is a log-odds value

    captured = {}

    class _StubClient:
        available = True

        def generate(self, *, role, system, user, **kwargs):
            captured["user"] = user
            from src.llm.gemini import LLMResponse, Usage
            return LLMResponse(text="ok", usage=Usage(model="stub"))

    monkeypatch.setattr("src.ml.explain.get_llm_client", lambda: _StubClient())
    narrate(explanation, probability=0.1, band="Low", base_rate=0.0807)

    assert "8.1%" in captured["user"] or "0.0807" in captured["user"]
    assert "-296" not in captured["user"]
    assert str(round(explanation.base_value, 1)) not in captured["user"]


def test_narrate_reports_llm_unavailable_without_a_key(
    explainer_and_data, monkeypatch
) -> None:
    from src.llm.gemini import GeminiClient, LLMUnavailable
    from src.ml.explain import narrate
    from src.utils.config import Settings

    explainer, X, _ = explainer_and_data
    explanation = explainer.explain_one(X.iloc[[0]])

    # Patched where it's looked up (src.ml.explain's imported name), not
    # where it's defined - explain.py already bound its own reference to the
    # real get_llm_client at import time, so patching src.llm.gemini's copy
    # would silently do nothing and let this test make a real API call.
    monkeypatch.setattr(
        "src.ml.explain.get_llm_client",
        lambda: GeminiClient(Settings(google_api_key="")),
    )
    with pytest.raises(LLMUnavailable):
        narrate(explanation, probability=0.1, band="Low", base_rate=0.08)


def test_explain_and_narrate_degrades_gracefully_without_a_key(
    explainer_and_data, monkeypatch
) -> None:
    """The numeric factors must still come back even when the narrative
    cannot be generated - the API route this feeds must never 500 just
    because no LLM key is configured."""
    from src.llm.gemini import GeminiClient
    from src.ml.explain import explain_and_narrate
    from src.utils.config import Settings

    explainer, X, _ = explainer_and_data
    monkeypatch.setattr(
        "src.ml.explain.get_llm_client",
        lambda: GeminiClient(Settings(google_api_key="")),
    )
    explanation = explain_and_narrate(
        X.iloc[[0]], probability=0.1, band="Low", base_rate=0.08, explainer=explainer,
    )

    assert explanation.narrative is None
    assert explanation.narrative_error is not None
    assert len(explanation.contributions) > 0
