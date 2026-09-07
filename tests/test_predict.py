"""Tests for RiskModel: loading trained artifacts and scoring applicants."""

from __future__ import annotations

import pytest

from tests.conftest_ml import build_tiny_trained_model, point_settings_at


@pytest.fixture
def trained_model(tmp_path):
    point_settings_at(tmp_path)
    X, y = build_tiny_trained_model(tmp_path)
    from src.ml.predict import RiskModel
    return RiskModel(), X, y


def test_is_trained_reflects_artifact_presence(tmp_path) -> None:
    point_settings_at(tmp_path)
    from src.ml.predict import RiskModel
    assert not RiskModel().is_trained

    build_tiny_trained_model(tmp_path)
    assert RiskModel().is_trained


def test_missing_model_raises_a_clear_error(tmp_path) -> None:
    point_settings_at(tmp_path)
    from src.ml.predict import ModelNotTrained, RiskModel

    with pytest.raises(ModelNotTrained, match="python -m src.ml.train"):
        RiskModel()._ensure_loaded()


def test_predict_proba_returns_calibrated_probabilities_in_range(trained_model) -> None:
    model, X, y = trained_model
    probs = model.predict_proba(X)
    assert len(probs) == len(X)
    assert ((probs >= 0) & (probs <= 1)).all()


def test_higher_ext_source_mean_predicts_lower_risk(trained_model) -> None:
    """The fixture's known signal: EXT_SOURCE_MEAN is protective. If the
    model learned nothing, this would be flat or reversed."""
    import pandas as pd

    model, _, _ = trained_model
    low_score = pd.DataFrame({
        "EXT_SOURCE_MEAN": [0.1] * 20, "AMT_INCOME_TOTAL": [100_000] * 20,
        "AMT_CREDIT": [300_000] * 20, "CODE_GENDER": pd.Categorical(["F"] * 20),
    })
    high_score = pd.DataFrame({
        "EXT_SOURCE_MEAN": [0.9] * 20, "AMT_INCOME_TOTAL": [100_000] * 20,
        "AMT_CREDIT": [300_000] * 20, "CODE_GENDER": pd.Categorical(["F"] * 20),
    })
    assert model.predict_proba(low_score).mean() > model.predict_proba(high_score).mean()


def test_band_for_uses_saved_thresholds(trained_model) -> None:
    model, _, _ = trained_model
    band, threshold = model.band_for(0.0)
    assert band == "Low"
    assert threshold == 0.0


def test_predict_from_raw_handles_a_hand_entered_applicant(trained_model) -> None:
    model, _, _ = trained_model
    prediction = model.predict_from_raw({
        "EXT_SOURCE_MEAN": 0.5, "AMT_INCOME_TOTAL": 120_000,
        "AMT_CREDIT": 400_000, "CODE_GENDER": "M",
    })
    assert prediction.sk_id_curr is None
    assert 0 <= prediction.probability <= 1
    assert prediction.band in {"Low", "Medium", "High"}
    assert prediction.lift_over_base_rate == round(
        prediction.probability / prediction.base_rate, 2
    )


def test_predict_from_raw_fills_missing_columns_with_nan(trained_model) -> None:
    """A hand-entered applicant won't have bureau/previous-application
    derived features - the model must still score them via LightGBM's
    native NaN handling, not crash on a missing column."""
    model, _, _ = trained_model
    prediction = model.predict_from_raw({"EXT_SOURCE_MEAN": 0.5})
    assert 0 <= prediction.probability <= 1


def test_align_frame_drops_unknown_and_orders_to_feature_list(trained_model) -> None:
    import pandas as pd

    model, _, _ = trained_model
    frame = pd.DataFrame({
        "EXT_SOURCE_MEAN": [0.5], "SOME_UNSEEN_COLUMN": ["x"],
        "AMT_INCOME_TOTAL": [100_000], "AMT_CREDIT": [300_000],
        "CODE_GENDER": pd.Categorical(["F"]),
    })
    aligned = model.align_frame(frame)
    assert list(aligned.columns) == model.feature_list
    assert "SOME_UNSEEN_COLUMN" not in aligned.columns
