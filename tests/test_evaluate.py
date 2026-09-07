"""Tests for evaluation metrics, against known constructed cases so the
numbers are checked exactly, not just "did not crash"."""

from __future__ import annotations

import numpy as np
import pytest

from src.ml.evaluate import compute_ks_statistic, compute_lift_table, evaluate


def test_perfect_separation_gives_ks_of_one() -> None:
    y = np.array([0] * 50 + [1] * 50)
    prob = np.array([0.1] * 50 + [0.9] * 50)
    assert compute_ks_statistic(y, prob) == pytest.approx(1.0)


def test_identical_score_distributions_give_ks_of_zero() -> None:
    # A large sample, since KS on two random draws from the same distribution
    # is only approximately 0 - it shrinks with n, and n=200 is genuinely
    # noisy enough to fail a tight tolerance for reasons unrelated to
    # correctness.
    rng = np.random.default_rng(0)
    shared = rng.uniform(0, 1, 20_000)
    y = np.array([0] * 10_000 + [1] * 10_000)
    assert compute_ks_statistic(y, shared) == pytest.approx(0.0, abs=0.03)


def test_lift_table_riskiest_decile_has_higher_default_rate() -> None:
    rng = np.random.default_rng(0)
    n = 1000
    y = (rng.uniform(0, 1, n) < 0.1).astype(int)
    # Score correlated with y, plus noise.
    prob = y * 0.6 + rng.uniform(0, 0.3, n)
    table = compute_lift_table(y, prob, n_bins=10)

    assert len(table) == 10
    assert table[0]["default_rate"] >= table[-1]["default_rate"]
    # Cumulative capture is monotone non-decreasing and reaches 1.0.
    cumulative = [row["cumulative_defaulters_caught_pct"] for row in table]
    assert cumulative == sorted(cumulative)
    assert cumulative[-1] == pytest.approx(1.0)


def test_lift_table_population_shares_sum_to_one() -> None:
    rng = np.random.default_rng(1)
    y = (rng.uniform(0, 1, 997) < 0.08).astype(int)  # not evenly divisible by 10
    prob = rng.uniform(0, 1, 997)
    table = compute_lift_table(y, prob, n_bins=10)
    assert sum(row["population_share"] for row in table) == pytest.approx(1.0, abs=0.01)


def test_evaluate_confusion_matrix_matches_a_known_case() -> None:
    y = np.array([0, 0, 0, 1, 1, 1])
    prob = np.array([0.1, 0.2, 0.6, 0.3, 0.7, 0.9])
    report = evaluate(y, prob, threshold=0.5)

    # At threshold 0.5: predicted positive = indices 2, 4, 5 (probs .6/.7/.9)
    # true labels there = 0, 1, 1 -> TP=2, FP=1; predicted negative = 0,1,3
    # true labels there = 0, 0, 1 -> TN=2, FN=1.
    assert report.confusion == {
        "true_negative": 2, "false_positive": 1,
        "false_negative": 1, "true_positive": 2,
    }
    assert report.precision_at_threshold == pytest.approx(2 / 3)
    assert report.recall_at_threshold == pytest.approx(2 / 3)


def test_evaluate_base_rate_matches_the_label_mean() -> None:
    y = np.array([0, 0, 0, 0, 1])
    prob = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    report = evaluate(y, prob, threshold=0.5)
    assert report.base_rate == pytest.approx(0.2)


def test_evaluate_as_dict_is_json_serialisable() -> None:
    import json

    y = np.array([0, 1] * 50)
    prob = np.random.default_rng(0).uniform(0, 1, 100)
    report = evaluate(y, prob, threshold=0.3)
    json.dumps(report.as_dict())  # raises if any numpy scalar leaked through


def test_evaluate_and_save_writes_the_expected_file(tmp_path) -> None:
    from src.ml.evaluate import evaluate_and_save

    y = np.array([0, 1] * 50)
    prob = np.random.default_rng(0).uniform(0, 1, 100)
    path = tmp_path / "metrics.json"
    evaluate_and_save(y, prob, threshold=0.5, path=path)

    assert path.exists()
    import json
    payload = json.loads(path.read_text())
    assert "lift_table" in payload
    assert "calibration_curve" in payload
    assert len(payload["lift_table"]) == 10
