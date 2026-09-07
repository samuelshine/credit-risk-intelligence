"""Shared helper for ML module tests: a tiny, fast, fully real trained model.

Not a mock - an actual LightGBM booster and isotonic calibrator, fit on a
small synthetic dataset with genuine signal, saved through the exact same
code paths `src/ml/train.py` uses (`booster_.save_model`, `joblib.dump`,
`write_json`). This is what lets `predict.py`, `explain.py` and `rules.py`
be tested against real artifacts in milliseconds instead of the ~4 minute
real training run.
"""

from __future__ import annotations

import os
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator

from src.utils.helpers import write_json


def build_tiny_trained_model(models_dir: Path, *, n_rows: int = 800, seed: int = 0):
    """Fit and save a small real model with genuine, checkable signal.

    Signal: EXT_SOURCE_MEAN is protective (higher -> lower risk), matching
    the real dataset's dominant pattern - tests can assert on its SHAP sign.
    """
    rng = np.random.default_rng(seed)
    ext_source_mean = rng.uniform(0, 1, n_rows)
    amt_income_total = rng.uniform(30_000, 300_000, n_rows)
    amt_credit = rng.uniform(50_000, 1_000_000, n_rows)
    code_gender = rng.choice(["M", "F"], n_rows)

    risk = 3.0 * (1 - ext_source_mean) + rng.normal(0, 0.3, n_rows)
    y = (risk > np.quantile(risk, 0.92)).astype(int)  # ~8% base rate

    X = pd.DataFrame({
        "EXT_SOURCE_MEAN": ext_source_mean,
        "AMT_INCOME_TOTAL": amt_income_total,
        "AMT_CREDIT": amt_credit,
        "CODE_GENDER": pd.Categorical(code_gender),
    })

    split = int(n_rows * 0.7)
    X_fit, X_holdout = X.iloc[:split], X.iloc[split:]
    y_fit, y_holdout = y[:split], y[split:]

    booster_model = lgb.LGBMClassifier(
        n_estimators=50, max_depth=3, min_child_samples=10,
        random_state=seed, verbosity=-1,
    )
    booster_model.fit(X_fit, y_fit, categorical_feature=["CODE_GENDER"])

    calibrator = CalibratedClassifierCV(
        FrozenEstimator(booster_model), method="isotonic"
    )
    calibrator.fit(X_holdout, y_holdout)

    models_dir.mkdir(parents=True, exist_ok=True)
    booster_model.booster_.save_model(str(models_dir / "model.txt"))
    joblib.dump(calibrator, models_dir / "calibrator.pkl")
    write_json(models_dir / "feature_list.json", {
        "features": list(X.columns),
        "categorical_features": ["CODE_GENDER"],
    })

    calibrated_holdout = calibrator.predict_proba(X_holdout)[:, 1]
    base_rate = float(y_holdout.mean())
    order = np.argsort(-calibrated_holdout)
    high_cut = max(1, len(y_holdout) // 10)
    write_json(models_dir / "bands.json", {
        "base_rate": round(base_rate, 4),
        "bands": {
            "High": {"score_threshold": round(float(calibrated_holdout[order][high_cut - 1]), 4),
                      "population_share": 0.1, "default_rate": 0.3, "lift_over_base_rate": 3.0},
            "Medium": {"score_threshold": round(float(np.median(calibrated_holdout)), 4),
                        "population_share": 0.4, "default_rate": 0.1, "lift_over_base_rate": 1.2},
            "Low": {"score_threshold": 0.0,
                     "population_share": 0.5, "default_rate": 0.02, "lift_over_base_rate": 0.3},
        },
    })
    return X, y


def point_settings_at(tmp_path: Path) -> None:
    """Point process-wide config at a temp models dir, clearing the cache."""
    os.environ["MODELS_DIR"] = str(tmp_path)
    from src.utils.config import get_settings
    get_settings.cache_clear()
