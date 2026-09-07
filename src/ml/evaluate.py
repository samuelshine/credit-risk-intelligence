"""Model evaluation: the metrics a credit-risk model actually needs.

`src/ml/train.py` already records CV scores during the imbalance comparison;
this module is the deeper, single-model evaluation run once against the final
calibrated model on the held-out split, producing everything
`docs/MODEL_CARD.md` reports and everything the UI's evaluation view shows.

Why these specific metrics, not just accuracy:

- **ROC-AUC** is the conventional headline number, included for comparability
  with other work on this dataset, but it treats all thresholds as equally
  interesting, which a deployed model does not.
- **PR-AUC** is what the imbalance comparison actually optimised for (see
  `train.py`) - it is sensitive to how well the model ranks the rare positive
  class, which is the real business objective.
- **KS statistic** (max separation between the cumulative TPR and FPR curves)
  is the number credit-risk teams have used for decades to judge a scorecard,
  because it has a direct reading: "how cleanly does the score separate
  goods from bads."
- **Brier score**, before and after calibration, is what proves calibration
  did something - PR-AUC and ROC-AUC are rank-only and would not move even if
  every predicted probability were wrong by a constant factor.
- **Lift/gains by decile** is the table a credit officer actually reads: "if
  we review the riskiest 10% of applicants, how many of the eventual
  defaulters have we caught?"
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import ks_2samp
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

from src.utils.config import get_settings
from src.utils.helpers import write_json
from src.utils.logger import get_logger, log_duration

log = get_logger(__name__)


@dataclass
class EvaluationReport:
    roc_auc: float
    pr_auc: float
    ks_statistic: float
    brier_score: float
    base_rate: float
    threshold: float
    confusion: dict[str, int]
    precision_at_threshold: float
    recall_at_threshold: float
    calibration_curve: dict[str, list[float]]
    lift_table: list[dict[str, Any]]
    roc_curve_points: dict[str, list[float]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "roc_auc": round(self.roc_auc, 5),
            "pr_auc": round(self.pr_auc, 5),
            "ks_statistic": round(self.ks_statistic, 5),
            "brier_score": round(self.brier_score, 5),
            "base_rate": round(self.base_rate, 5),
            "threshold": round(self.threshold, 5),
            "confusion_matrix": self.confusion,
            "precision_at_threshold": round(self.precision_at_threshold, 5),
            "recall_at_threshold": round(self.recall_at_threshold, 5),
            "calibration_curve": self.calibration_curve,
            "lift_table": self.lift_table,
            "roc_curve": self.roc_curve_points,
        }


def compute_ks_statistic(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Kolmogorov-Smirnov statistic between the score distributions of the
    two classes - the maximum gap between their CDFs. 0 = no separation,
    1 = perfect separation."""
    scores_pos = y_prob[y_true == 1]
    scores_neg = y_prob[y_true == 0]
    return float(ks_2samp(scores_pos, scores_neg).statistic)


def compute_lift_table(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10,
) -> list[dict[str, Any]]:
    """Gains/lift by score decile, riskiest first - the table a credit
    officer reads to answer "how many defaulters would reviewing the top N%
    catch."""
    order = np.argsort(-y_prob)
    sorted_true = y_true[order]
    n = len(y_true)
    total_positives = int(y_true.sum())
    base_rate = total_positives / n if n else 0.0

    rows = []
    cumulative_positives = 0
    bin_edges = np.linspace(0, n, n_bins + 1).astype(int)
    for i in range(n_bins):
        start, end = bin_edges[i], bin_edges[i + 1]
        bucket = sorted_true[start:end]
        bucket_positives = int(bucket.sum())
        cumulative_positives += bucket_positives
        rows.append({
            "decile": i + 1,
            "population_share": round((end - start) / n, 4),
            "default_rate": round(bucket_positives / max(end - start, 1), 4),
            "lift": round(
                (bucket_positives / max(end - start, 1)) / base_rate, 2
            ) if base_rate else 0.0,
            "cumulative_defaulters_caught_pct": round(
                cumulative_positives / max(total_positives, 1), 4
            ),
        })
    return rows


def evaluate(
    y_true: np.ndarray, y_prob: np.ndarray, *, threshold: float,
) -> EvaluationReport:
    """Compute every metric in one pass over a probability/label pair."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0

    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="quantile")
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    # Sub-sample the ROC curve to a plottable size - the raw curve has one
    # point per unique score, which on 300k+ rows is far more than a chart
    # or a JSON payload needs.
    idx = np.linspace(0, len(fpr) - 1, min(200, len(fpr))).astype(int)

    return EvaluationReport(
        roc_auc=float(roc_auc_score(y_true, y_prob)),
        pr_auc=float(average_precision_score(y_true, y_prob)),
        ks_statistic=compute_ks_statistic(y_true, y_prob),
        brier_score=float(brier_score_loss(y_true, y_prob)),
        base_rate=float(y_true.mean()),
        threshold=float(threshold),
        confusion={"true_negative": int(tn), "false_positive": int(fp),
                   "false_negative": int(fn), "true_positive": int(tp)},
        precision_at_threshold=float(precision),
        recall_at_threshold=float(recall),
        calibration_curve={
            "mean_predicted": [round(float(x), 4) for x in mean_pred],
            "fraction_positive": [round(float(x), 4) for x in frac_pos],
        },
        lift_table=compute_lift_table(y_true, y_prob),
        roc_curve_points={
            "fpr": [round(float(x), 4) for x in fpr[idx]],
            "tpr": [round(float(x), 4) for x in tpr[idx]],
        },
    )


def evaluate_and_save(
    y_true: np.ndarray, y_prob: np.ndarray, *, threshold: float,
    path: Path | None = None,
) -> EvaluationReport:
    """Convenience wrapper: evaluate and write `models/metrics.json`."""
    settings = get_settings()
    path = path or (settings.models_dir / "metrics.json")
    with log_duration(log, "compute evaluation metrics"):
        report = evaluate(y_true, y_prob, threshold=threshold)
    write_json(path, report.as_dict())
    log.info(
        "ROC-AUC %.4f  PR-AUC %.4f  KS %.4f  Brier %.4f  "
        "precision@t %.3f  recall@t %.3f",
        report.roc_auc, report.pr_auc, report.ks_statistic, report.brier_score,
        report.precision_at_threshold, report.recall_at_threshold,
    )
    return report
