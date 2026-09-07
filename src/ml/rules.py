"""Business rules distilled from the model.

LightGBM is not something a credit officer or a regulator can read. This
module fits a small, genuinely readable decision tree to *approximate* the
LightGBM model's risk score, then turns each of its leaves into a sentence a
policy document could contain:

    IF EXT_SOURCE_2 <= 0.31 AND CREDIT_INCOME_RATIO > 4.2
    THEN 8.4% of applicants, observed default rate 21.3% (2.6x base rate)

The surrogate approximates the *score*, not the label - a `DecisionTreeRegressor`
trained to predict the calibrated probability, not a classifier trained on
TARGET. Predicting the label directly would give a tree that mimics guessing
"most likely class," losing exactly the risk gradation the score exists to
capture; predicting the score keeps the tree honest about ranking risk, which
is what a policy rule needs to sort applicants by.

Every number in a rule - population share, default rate, lift - is measured
directly against `TARGET` for the applicants who actually fall into that leaf,
not read off the tree's own prediction. That is what "surrogate fidelity"
below reports: how much the simplification costs, stated plainly rather than
assumed away.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.tree import DecisionTreeRegressor, _tree

from src.utils.config import get_settings
from src.utils.helpers import humanise_column, write_json
from src.utils.logger import get_logger, log_duration

log = get_logger(__name__)

#: Shallow enough to read as a policy document: depth 4 is at most 16 leaves,
#: each requiring at most 4 conditions to reach.
MAX_DEPTH = 4
MIN_SAMPLES_LEAF = 500


@dataclass
class Rule:
    conditions: list[str]
    population_share: float
    n_applicants: int
    observed_default_rate: float
    lift_over_base_rate: float
    mean_predicted_score: float

    def as_sentence(self) -> str:
        where = " AND ".join(self.conditions) if self.conditions else "always"
        return (
            f"IF {where} "
            f"THEN {self.population_share:.1%} of applicants "
            f"({self.n_applicants:,}), observed default rate "
            f"{self.observed_default_rate:.1%} "
            f"({self.lift_over_base_rate:.1f}x base rate)"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "conditions": self.conditions,
            "sentence": self.as_sentence(),
            "population_share": round(self.population_share, 4),
            "n_applicants": self.n_applicants,
            "observed_default_rate": round(self.observed_default_rate, 4),
            "lift_over_base_rate": round(self.lift_over_base_rate, 2),
            "mean_predicted_score": round(self.mean_predicted_score, 4),
        }


def _leaf_conditions(
    tree: _tree.Tree, feature_names: list[str],
) -> dict[int, list[str]]:
    """Every leaf's root-to-leaf path, as human-readable conditions.

    Walks the tree once, accumulating the conjunction of splits that leads to
    each leaf - this is the actual decision path, not a summary of it.
    """
    conditions: dict[int, list[str]] = {}

    def recurse(node: int, path: list[str]) -> None:
        if tree.feature[node] == _tree.TREE_UNDEFINED:
            conditions[node] = path
            return
        name = feature_names[tree.feature[node]]
        label = humanise_column(name)
        threshold = tree.threshold[node]
        recurse(tree.children_left[node], path + [f"{label} <= {threshold:.3g}"])
        recurse(tree.children_right[node], path + [f"{label} > {threshold:.3g}"])

    recurse(0, [])
    return conditions


def derive_rules(
    X: pd.DataFrame, calibrated_scores: np.ndarray, y_true: np.ndarray,
    *, max_depth: int = MAX_DEPTH, min_samples_leaf: int = MIN_SAMPLES_LEAF,
) -> tuple[list[Rule], dict[str, float]]:
    """Fit the surrogate, extract one Rule per leaf, and report fidelity.

    `X` should be numeric-only (categoricals pre-encoded) - a shallow
    regression tree over one-hot or ordinal-coded categoricals is still
    readable at this depth, and keeps this module independent of exactly how
    LightGBM's native categorical handling works internally.
    """
    surrogate = DecisionTreeRegressor(
        max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=42,
    )
    with log_duration(log, "fit rule-extraction surrogate tree"):
        surrogate.fit(X, calibrated_scores)

    surrogate_pred = surrogate.predict(X)
    fidelity = {
        "r2_vs_calibrated_score": round(float(r2_score(calibrated_scores, surrogate_pred)), 4),
        "roc_auc_vs_actual_target": round(float(roc_auc_score(y_true, surrogate_pred)), 4),
        "full_model_roc_auc_vs_actual_target": round(
            float(roc_auc_score(y_true, calibrated_scores)), 4
        ),
    }

    leaf_ids = surrogate.apply(X)
    conditions_by_leaf = _leaf_conditions(surrogate.tree_, list(X.columns))
    base_rate = float(y_true.mean())
    n_total = len(y_true)

    rules = []
    for leaf_id in np.unique(leaf_ids):
        mask = leaf_ids == leaf_id
        n = int(mask.sum())
        observed_rate = float(y_true[mask].mean())
        rules.append(Rule(
            conditions=conditions_by_leaf.get(leaf_id, []),
            population_share=n / n_total,
            n_applicants=n,
            observed_default_rate=observed_rate,
            lift_over_base_rate=(observed_rate / base_rate) if base_rate else 0.0,
            mean_predicted_score=float(calibrated_scores[mask].mean()),
        ))

    rules.sort(key=lambda r: -r.observed_default_rate)
    return rules, fidelity


def encode_for_surrogate(X: pd.DataFrame) -> pd.DataFrame:
    """Integer-code categorical columns for the surrogate tree.

    A regression tree only needs an ordering to split on, not a meaningful
    one - unlike a linear model, the split threshold just partitions
    categories into two groups, so an arbitrary consistent code is enough.
    """
    encoded = X.copy()
    for column in encoded.columns:
        if str(encoded[column].dtype) == "category":
            encoded[column] = encoded[column].cat.codes.replace(-1, np.nan)
    return encoded


def build_and_save_rules() -> dict[str, Any]:
    """The full path `python -m src.ml.rules` and the post-training pipeline
    use: load the trained model, score the training data, fit the surrogate,
    save `models/rules.json`."""
    from src.data.database import get_readonly_connection
    from src.data.features import load_features
    from src.ml.predict import get_risk_model

    settings = get_settings()
    model = get_risk_model()
    conn = get_readonly_connection()
    matrix = load_features("application_train", conn=conn)

    with log_duration(log, "score training data for rule extraction"):
        scores = model.predict_proba(matrix.frame)

    X_encoded = encode_for_surrogate(model.align_frame(matrix.frame))
    rules, fidelity = derive_rules(X_encoded, scores, matrix.target.to_numpy())

    payload = {
        "fidelity": fidelity,
        "base_rate": float(matrix.target.mean()),
        "n_rules": len(rules),
        "rules": [r.as_dict() for r in rules],
    }
    write_json(settings.models_dir / "rules.json", payload)
    log.info("derived %d rules, surrogate R^2=%.3f, surrogate ROC-AUC=%.4f "
             "(full model: %.4f)", len(rules), fidelity["r2_vs_calibrated_score"],
             fidelity["roc_auc_vs_actual_target"],
             fidelity["full_model_roc_auc_vs_actual_target"])
    for rule in rules:
        log.info("  %s", rule.as_sentence())
    return payload


def main() -> int:
    from src.utils.logger import configure_logging
    configure_logging()
    build_and_save_rules()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
