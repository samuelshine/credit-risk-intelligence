"""Tests for surrogate rule extraction.

The core claim to verify: every number attached to a rule (population share,
observed default rate, lift) is measured directly against the real labels for
applicants in that leaf - not read off the surrogate tree's own prediction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.ml.rules import derive_rules, encode_for_surrogate


@pytest.fixture
def clean_split_data():
    """200 rows with an exact, known split: X < 5 -> always defaults,
    X >= 5 -> never defaults. A leaf's default rate must come out exactly
    1.0 or 0.0, not something the tree merely predicts."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"SCORE_FEATURE": rng.uniform(0, 10, 200)})
    y = (X["SCORE_FEATURE"] < 5).astype(int).to_numpy()
    # A model score must move the same direction as TARGET (higher = riskier).
    # SCORE_FEATURE < 5 is the risky group here, so the stand-in score is
    # (10 - SCORE_FEATURE)/10 - inverted, not SCORE_FEATURE/10 directly.
    scores = (10.0 - X["SCORE_FEATURE"].to_numpy()) / 10.0
    return X, y, scores


def test_rules_partition_the_full_population(clean_split_data) -> None:
    X, y, scores = clean_split_data
    rules, _ = derive_rules(X, scores, y, max_depth=2, min_samples_leaf=10)
    assert sum(r.n_applicants for r in rules) == len(y)
    assert pytest.approx(sum(r.population_share for r in rules), rel=1e-6) == 1.0


def test_rule_default_rate_is_measured_not_predicted(clean_split_data) -> None:
    """The defining property of this module: a rule covering only
    X < 5 rows must report observed_default_rate == 1.0 exactly, because
    every one of those rows truly has TARGET=1 - read from y, not inferred."""
    X, y, scores = clean_split_data
    rules, _ = derive_rules(X, scores, y, max_depth=1, min_samples_leaf=10)

    assert len(rules) == 2
    high_risk = max(rules, key=lambda r: r.observed_default_rate)
    low_risk = min(rules, key=lambda r: r.observed_default_rate)
    assert high_risk.observed_default_rate == pytest.approx(1.0)
    assert low_risk.observed_default_rate == pytest.approx(0.0)


def test_lift_is_rate_over_base_rate(clean_split_data) -> None:
    X, y, scores = clean_split_data
    rules, _ = derive_rules(X, scores, y, max_depth=1, min_samples_leaf=10)
    base_rate = y.mean()
    for rule in rules:
        assert rule.lift_over_base_rate == pytest.approx(
            rule.observed_default_rate / base_rate, rel=1e-6
        )


def test_fidelity_reports_a_real_r2_and_auc(clean_split_data) -> None:
    """On this fixture the surrogate can fit the pattern almost perfectly,
    so fidelity should be high - a broken fidelity computation (e.g. wrong
    array alignment) would instead show near-zero or NaN."""
    X, y, scores = clean_split_data
    _, fidelity = derive_rules(X, scores, y, max_depth=2, min_samples_leaf=10)
    assert fidelity["r2_vs_calibrated_score"] > 0.9
    assert fidelity["roc_auc_vs_actual_target"] > 0.95


def test_rule_sentence_is_readable(clean_split_data) -> None:
    X, y, scores = clean_split_data
    rules, _ = derive_rules(X, scores, y, max_depth=1, min_samples_leaf=10)
    sentence = rules[0].as_sentence()
    assert sentence.startswith("IF ")
    assert "THEN" in sentence
    assert "observed default rate" in sentence


def test_rules_are_sorted_riskiest_first(clean_split_data) -> None:
    X, y, scores = clean_split_data
    rules, _ = derive_rules(X, scores, y, max_depth=2, min_samples_leaf=10)
    rates = [r.observed_default_rate for r in rules]
    assert rates == sorted(rates, reverse=True)


def test_encode_for_surrogate_converts_categories_to_numeric_codes() -> None:
    frame = pd.DataFrame({
        "NUMERIC": [1.0, 2.0, 3.0],
        "CATEGORY": pd.Categorical(["A", "B", "A"]),
    })
    encoded = encode_for_surrogate(frame)
    assert encoded["CATEGORY"].dtype != "category"
    assert set(encoded["CATEGORY"].unique()) <= {0.0, 1.0}
    assert list(encoded["NUMERIC"]) == [1.0, 2.0, 3.0]


def test_encode_for_surrogate_maps_unseen_category_to_nan() -> None:
    frame = pd.DataFrame({"CATEGORY": pd.Categorical(["A", "B"], categories=["A", "B", "C"])})
    # Simulate a category present in .categories but absent from the data -
    # cat.codes is -1 for actual nulls, which encode_for_surrogate must not
    # confuse with a real code.
    frame.loc[0, "CATEGORY"] = None
    encoded = encode_for_surrogate(frame)
    assert np.isnan(encoded["CATEGORY"].iloc[0])
