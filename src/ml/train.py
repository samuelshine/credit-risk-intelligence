"""Train the credit-default model.

    python -m src.ml.train

Pipeline: load features from DuckDB -> compare imbalance strategies on
stratified CV -> refit the chosen strategy on the full training split ->
calibrate -> derive risk bands and an operating threshold -> save artifacts.

Design decisions, and why they're made here rather than assumed:

**LightGBM, not deep learning.** ~307k rows and ~200 mixed numeric/categorical
tabular features is squarely gradient-boosted-tree territory; on tabular data
at this scale GBMs consistently match or beat neural approaches while training
in seconds and needing no GPU, which matters for a Docker image an evaluator
runs locally.

**The imbalance comparison is run, not asserted.** Four strategies -
none, `scale_pos_weight`, SMOTE, random undersampling - are scored on the
*same* CV splits so the comparison is fair. The result is written to
`models/imbalance_comparison.json` so the choice in docs/MODEL_CARD.md is
backed by a number, not a rule of thumb.

**Selection metric is PR-AUC, not accuracy.** At an 11.4:1 imbalance, a model
that always predicts "no default" scores 92% accuracy and is useless. PR-AUC
rewards separating the minority class, which is the actual business problem.

**Calibration is separate from imbalance handling.** Reweighting or resampling
improves ranking (PR-AUC) but distorts predicted probabilities - a
`scale_pos_weight`-trained model's "0.5" does not mean a 50% empirical default
rate. Isotonic calibration is fit afterward, on held-out data, specifically to
restore that meaning, since the risk score shown to a user has to be honest
about what it claims.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

from src.data.database import get_readonly_connection
from src.data.features import FeatureMatrix, load_features
from src.utils.config import Settings, get_settings
from src.utils.helpers import write_json
from src.utils.logger import configure_logging, get_logger, log_duration

log = get_logger(__name__)

RANDOM_STATE = 42
N_FOLDS = 5

#: LightGBM parameters. Conservative depth/leaves and a learning rate small
#: enough that early stopping, not a fixed round count, decides when to stop -
#: the alternative (a large fixed n_estimators) overfits some folds and
#: underfits others.
LGBM_PARAMS: dict[str, Any] = dict(
    objective="binary",
    n_estimators=2000,
    learning_rate=0.03,
    num_leaves=31,
    max_depth=-1,
    min_child_samples=50,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=RANDOM_STATE,
    verbosity=-1,
    n_jobs=-1,
)

EARLY_STOPPING_ROUNDS = 100


@dataclass
class FoldScore:
    fold: int
    roc_auc: float
    pr_auc: float
    brier: float


@dataclass
class StrategyResult:
    name: str
    fold_scores: list[FoldScore]

    @property
    def mean_pr_auc(self) -> float:
        return float(np.mean([f.pr_auc for f in self.fold_scores]))

    @property
    def std_pr_auc(self) -> float:
        return float(np.std([f.pr_auc for f in self.fold_scores]))

    @property
    def mean_roc_auc(self) -> float:
        return float(np.mean([f.roc_auc for f in self.fold_scores]))

    @property
    def mean_brier(self) -> float:
        return float(np.mean([f.brier for f in self.fold_scores]))

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "mean_pr_auc": round(self.mean_pr_auc, 5),
            "std_pr_auc": round(self.std_pr_auc, 5),
            "mean_roc_auc": round(self.mean_roc_auc, 5),
            "mean_brier": round(self.mean_brier, 5),
            "fold_scores": [asdict(f) for f in self.fold_scores],
        }


def _fit_lgbm(
    X_train: pd.DataFrame, y_train: pd.Series,
    X_val: pd.DataFrame, y_val: pd.Series,
    *, scale_pos_weight: float | None = None,
) -> lgb.LGBMClassifier:
    """Fit one LightGBM model with early stopping on the validation fold."""
    params = dict(LGBM_PARAMS)
    if scale_pos_weight is not None:
        params["scale_pos_weight"] = scale_pos_weight

    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_X=X_val, eval_y=y_val,
        eval_metric="average_precision",
        categorical_feature=[
            c for c in X_train.columns if str(X_train[c].dtype) == "category"
        ],
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
    )
    return model


def _score(y_true: np.ndarray, y_prob: np.ndarray, fold: int) -> FoldScore:
    return FoldScore(
        fold=fold,
        roc_auc=float(roc_auc_score(y_true, y_prob)),
        pr_auc=float(average_precision_score(y_true, y_prob)),
        brier=float(brier_score_loss(y_true, y_prob)),
    )


def compare_imbalance_strategies(
    X: pd.DataFrame, y: pd.Series,
) -> dict[str, StrategyResult]:
    """Run all four strategies on the same CV splits and score each.

    Same splits for every strategy is what makes the comparison meaningful -
    otherwise a strategy could look better purely from an easier fold split.
    """
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    pos_weight = float((y == 0).sum() / max((y == 1).sum(), 1))
    results: dict[str, list[FoldScore]] = {
        "none": [], "scale_pos_weight": [], "smote": [], "undersample": [],
    }

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        with log_duration(log, f"fold {fold}/{N_FOLDS}: none"):
            model = _fit_lgbm(X_train, y_train, X_val, y_val)
            results["none"].append(
                _score(y_val, model.predict_proba(X_val)[:, 1], fold)
            )

        with log_duration(log, f"fold {fold}/{N_FOLDS}: scale_pos_weight"):
            model = _fit_lgbm(
                X_train, y_train, X_val, y_val, scale_pos_weight=pos_weight
            )
            results["scale_pos_weight"].append(
                _score(y_val, model.predict_proba(X_val)[:, 1], fold)
            )

        # SMOTE and undersampling need numeric-only input (no pandas category
        # dtype, which imblearn's nearest-neighbour and random samplers can't
        # index), so categoricals are integer-coded first, fit only on the
        # training fold to avoid leaking validation-fold category frequencies.
        X_train_enc, X_val_enc = _encode_categoricals_for_resampling(X_train, X_val)

        with log_duration(log, f"fold {fold}/{N_FOLDS}: smote"):
            from imblearn.over_sampling import SMOTE

            imputed = SimpleImputer(strategy="median").fit(X_train_enc)
            X_train_imp = pd.DataFrame(
                imputed.transform(X_train_enc), columns=X_train_enc.columns
            )
            X_res, y_res = SMOTE(random_state=RANDOM_STATE).fit_resample(
                X_train_imp, y_train.reset_index(drop=True)
            )
            model = _fit_lgbm(X_res, y_res, X_val_enc, y_val)
            results["smote"].append(
                _score(y_val, model.predict_proba(X_val_enc)[:, 1], fold)
            )

        with log_duration(log, f"fold {fold}/{N_FOLDS}: undersample"):
            from imblearn.under_sampling import RandomUnderSampler

            X_res, y_res = RandomUnderSampler(random_state=RANDOM_STATE).fit_resample(
                X_train_enc, y_train
            )
            model = _fit_lgbm(X_res, y_res, X_val_enc, y_val)
            results["undersample"].append(
                _score(y_val, model.predict_proba(X_val_enc)[:, 1], fold)
            )

    return {name: StrategyResult(name=name, fold_scores=scores)
            for name, scores in results.items()}


def _encode_categoricals_for_resampling(
    X_train: pd.DataFrame, X_val: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Integer-code category columns, fit on train only, applied to both.

    A category unseen in the training fold maps to -1 rather than raising -
    resampling folds are small enough that a rare category can plausibly be
    absent from one fold entirely.
    """
    X_train, X_val = X_train.copy(), X_val.copy()
    for col in X_train.columns:
        if str(X_train[col].dtype) == "category":
            categories = X_train[col].cat.categories
            mapping = {cat: i for i, cat in enumerate(categories)}
            # .astype(object) first: mapping a Categorical series returns
            # another Categorical, whose fillna() rejects -1 as "not a known
            # category" even though we're deliberately replacing the dtype.
            X_train[col] = X_train[col].astype(object).map(mapping).astype("float64")
            X_val[col] = (
                X_val[col].astype(object).map(mapping).fillna(-1).astype("float64")
            )
    return X_train, X_val


def fit_baseline(X: pd.DataFrame, y: pd.Series) -> tuple[Pipeline, float]:
    """Logistic regression baseline: median-impute, scale, one-hot.

    Not the shipped model - it exists so docs/MODEL_CARD.md can report how
    much the engineered features and LightGBM actually buy over a transparent
    linear reference point, which is the honest way to justify model choice.
    """
    numeric_cols = [c for c in X.columns if str(X[c].dtype) != "category"]
    categorical_cols = [c for c in X.columns if str(X[c].dtype) == "category"]

    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), numeric_cols),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", max_categories=20)),
        ]), categorical_cols),
    ])
    pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores = []
    with log_duration(log, "baseline: 5-fold CV"):
        for train_idx, val_idx in skf.split(X, y):
            pipeline.fit(X.iloc[train_idx], y.iloc[train_idx])
            prob = pipeline.predict_proba(X.iloc[val_idx])[:, 1]
            scores.append(average_precision_score(y.iloc[val_idx], prob))

    mean_pr_auc = float(np.mean(scores))
    log.info("baseline (logistic regression) mean PR-AUC: %.4f", mean_pr_auc)
    pipeline.fit(X, y)  # refit on all data for the final saved baseline
    return pipeline, mean_pr_auc


def derive_risk_bands(
    y_true: np.ndarray, y_prob: np.ndarray, base_rate: float,
) -> dict[str, Any]:
    """Cut Low/Medium/High from the score distribution's actual behaviour.

    Bands are defined by where they land in the *observed* default rate, not
    by arbitrary score cutoffs like 0.3/0.7 - a 0.3 threshold means nothing
    on its own when the base rate is 8%. High = top decile by score; Low =
    bottom half; Medium = everything between. Cutoffs and their observed
    rates are what's saved - never hard-coded elsewhere.
    """
    order = np.argsort(-y_prob)
    sorted_prob = y_prob[order]
    sorted_true = y_true[order]
    n = len(y_true)

    high_cut_idx = max(1, n // 10)
    low_cut_idx = n // 2

    high_score_threshold = float(sorted_prob[high_cut_idx - 1])
    low_score_threshold = float(sorted_prob[low_cut_idx - 1])

    def band_stats(mask: np.ndarray) -> dict[str, float]:
        share = float(mask.sum() / n)
        rate = float(sorted_true[mask].mean()) if mask.sum() else 0.0
        return {"population_share": round(share, 4), "default_rate": round(rate, 4),
                "lift_over_base_rate": round(rate / base_rate, 2) if base_rate else 0.0}

    high_mask = np.arange(n) < high_cut_idx
    medium_mask = (np.arange(n) >= high_cut_idx) & (np.arange(n) < low_cut_idx)
    low_mask = np.arange(n) >= low_cut_idx

    return {
        "base_rate": round(float(base_rate), 4),
        "bands": {
            "High": {"score_threshold": round(high_score_threshold, 4),
                     **band_stats(high_mask)},
            "Medium": {"score_threshold": round(low_score_threshold, 4),
                       **band_stats(medium_mask)},
            "Low": {"score_threshold": 0.0, **band_stats(low_mask)},
        },
    }


def band_for_score(score: float, bands: dict[str, Any]) -> str:
    """Classify a single score using saved band thresholds."""
    if score >= bands["bands"]["High"]["score_threshold"]:
        return "High"
    if score >= bands["bands"]["Medium"]["score_threshold"]:
        return "Medium"
    return "Low"


def find_cost_optimal_threshold(
    y_true: np.ndarray, y_prob: np.ndarray, *, fn_cost: float, fp_cost: float,
) -> dict[str, Any]:
    """The operating threshold that minimises expected cost.

    A missed default (false negative) costs the bank far more than a
    wrongly-flagged good applicant (false positive) - a written-off loan
    versus a manual review or a slightly more cautious offer. `fn_cost` and
    `fp_cost` are stated assumptions, not measured facts about this dataset,
    and are documented as such in docs/MODEL_CARD.md.
    """
    thresholds = np.unique(np.round(y_prob, 4))
    best = {"threshold": 0.5, "cost": float("inf")}
    for t in thresholds:
        pred = (y_prob >= t).astype(int)
        fn = int(((pred == 0) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        cost = fn * fn_cost + fp * fp_cost
        if cost < best["cost"]:
            best = {"threshold": float(t), "cost": float(cost), "fn": fn, "fp": fp}
    return best


def train(
    settings: Settings | None = None, *, sample_rows: int | None = None,
) -> dict[str, Any]:
    """Full training run. Returns the summary that gets saved alongside the
    artifacts, so a caller (or a test) can inspect what happened without
    re-reading files from disk."""
    settings = settings or get_settings()
    started = time.perf_counter()

    conn = get_readonly_connection()
    matrix: FeatureMatrix = load_features("application_train", conn=conn, limit=sample_rows)
    X, y = matrix.frame, matrix.target
    if y is None:
        raise RuntimeError("application_train has no TARGET column")

    base_rate = float(y.mean())
    log.info("training on %s rows, %d features, base rate %.4f",
             f"{len(X):,}", X.shape[1], base_rate)

    baseline_pipeline, baseline_pr_auc = fit_baseline(X, y)

    with log_duration(log, "imbalance strategy comparison (4 strategies x 5 folds)"):
        comparison = compare_imbalance_strategies(X, y)
    for result in comparison.values():
        log.info("  %-18s PR-AUC %.4f (+/- %.4f)  ROC-AUC %.4f  Brier %.4f",
                 result.name, result.mean_pr_auc, result.std_pr_auc,
                 result.mean_roc_auc, result.mean_brier)

    chosen_name = max(comparison, key=lambda k: comparison[k].mean_pr_auc)
    log.info("chosen strategy: %s (highest mean PR-AUC)", chosen_name)

    # Final split: hold out 20% for calibration + band/threshold derivation,
    # kept separate from anything the model or its imbalance handling saw.
    from sklearn.model_selection import train_test_split
    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    X_fit, X_early, y_fit, y_early = train_test_split(
        X_train, y_train, test_size=0.1, stratify=y_train, random_state=RANDOM_STATE
    )

    pos_weight = float((y_fit == 0).sum() / max((y_fit == 1).sum(), 1))
    with log_duration(log, f"final fit ({chosen_name}) on full training split"):
        if chosen_name == "scale_pos_weight":
            final_model = _fit_lgbm(X_fit, y_fit, X_early, y_early,
                                     scale_pos_weight=pos_weight)
        elif chosen_name == "smote":
            X_fit_enc, X_early_enc = _encode_categoricals_for_resampling(X_fit, X_early)
            imputed = SimpleImputer(strategy="median").fit(X_fit_enc)
            X_fit_imp = pd.DataFrame(imputed.transform(X_fit_enc), columns=X_fit_enc.columns)
            from imblearn.over_sampling import SMOTE
            X_res, y_res = SMOTE(random_state=RANDOM_STATE).fit_resample(
                X_fit_imp, y_fit.reset_index(drop=True)
            )
            final_model = _fit_lgbm(X_res, y_res, X_early_enc, y_early)
            X_holdout = _encode_categoricals_for_resampling(X_fit, X_holdout)[1]
        elif chosen_name == "undersample":
            X_fit_enc, X_early_enc = _encode_categoricals_for_resampling(X_fit, X_early)
            from imblearn.under_sampling import RandomUnderSampler
            X_res, y_res = RandomUnderSampler(random_state=RANDOM_STATE).fit_resample(
                X_fit_enc, y_fit
            )
            final_model = _fit_lgbm(X_res, y_res, X_early_enc, y_early)
            X_holdout = _encode_categoricals_for_resampling(X_fit, X_holdout)[1]
        else:
            final_model = _fit_lgbm(X_fit, y_fit, X_early, y_early)

    # Calibrate on the untouched holdout, isotonic (monotone, no shape
    # assumption) - see the module docstring for why this is separate from
    # whichever imbalance strategy won above.
    with log_duration(log, "isotonic calibration"):
        # FrozenEstimator marks final_model as already fitted, so
        # CalibratedClassifierCV only fits the isotonic map on X_holdout/
        # y_holdout - the base model is never refit or exposed to this data
        # any other way. Replaces the removed cv="prefit" API (sklearn 1.6+).
        calibrator = CalibratedClassifierCV(
            FrozenEstimator(final_model), method="isotonic"
        )
        calibrator.fit(X_holdout, y_holdout)

    calibrated_prob = calibrator.predict_proba(X_holdout)[:, 1]
    raw_prob = final_model.predict_proba(X_holdout)[:, 1]

    bands = derive_risk_bands(y_holdout.to_numpy(), calibrated_prob, base_rate)
    threshold = find_cost_optimal_threshold(
        y_holdout.to_numpy(), calibrated_prob, fn_cost=10.0, fp_cost=1.0,
    )

    settings.models_dir.mkdir(parents=True, exist_ok=True)
    final_model.booster_.save_model(str(settings.models_dir / "model.txt"))
    import joblib
    joblib.dump(calibrator, settings.models_dir / "calibrator.pkl")
    joblib.dump(baseline_pipeline, settings.models_dir / "baseline_model.pkl")
    write_json(settings.models_dir / "feature_list.json", {
        "features": matrix.feature_names,
        "categorical_features": matrix.categorical_columns,
    })
    write_json(settings.models_dir / "bands.json", bands)
    write_json(settings.models_dir / "threshold.json", threshold)
    write_json(settings.models_dir / "imbalance_comparison.json", {
        "chosen_strategy": chosen_name,
        "base_rate": base_rate,
        "strategies": {name: r.as_dict() for name, r in comparison.items()},
        "baseline_logistic_regression_pr_auc": round(baseline_pr_auc, 5),
    })

    from src.ml.evaluate import evaluate_and_save
    evaluate_and_save(
        y_holdout.to_numpy(), calibrated_prob,
        threshold=threshold["threshold"],
        path=settings.models_dir / "metrics.json",
    )

    summary = {
        "rows_trained": len(X),
        "n_features": X.shape[1],
        "base_rate": base_rate,
        "baseline_pr_auc": round(baseline_pr_auc, 5),
        "chosen_imbalance_strategy": chosen_name,
        "holdout_roc_auc": round(float(roc_auc_score(y_holdout, calibrated_prob)), 5),
        "holdout_pr_auc": round(float(average_precision_score(y_holdout, calibrated_prob)), 5),
        "holdout_brier_raw": round(float(brier_score_loss(y_holdout, raw_prob)), 5),
        "holdout_brier_calibrated": round(float(brier_score_loss(y_holdout, calibrated_prob)), 5),
        "operating_threshold": threshold["threshold"],
        "training_seconds": round(time.perf_counter() - started, 1),
    }
    write_json(settings.models_dir / "train_summary.json", summary)
    log.info("training complete: %s", json.dumps(summary, indent=2))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the credit-default model.")
    parser.add_argument("--sample-rows", type=int, default=None,
                        help="train on a row subset (fast dev iteration)")
    args = parser.parse_args()
    configure_logging()
    train(sample_rows=args.sample_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
