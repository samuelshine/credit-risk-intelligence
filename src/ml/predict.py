"""Inference: a raw applicant in, a risk score and band out.

Loads the artifacts `src/ml/train.py` saves - the LightGBM booster, the
isotonic calibrator, the feature list, and the band/threshold cutoffs - once
per process, then scores applicants against them. This is the module the API's
`/api/score` route calls, and the only place "probability -> band" logic lives,
so the UI and any future batch-scoring script can never disagree about what a
score means.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from src.data.database import get_readonly_connection
from src.data.features import EXCLUDED_COLUMNS, build_feature_sql
from src.utils.config import Settings, get_settings
from src.utils.helpers import read_json
from src.utils.logger import get_logger

log = get_logger(__name__)


class ModelNotTrained(RuntimeError):
    """Raised when a prediction is requested but no trained artifacts exist."""


@dataclass
class Prediction:
    sk_id_curr: int | None
    probability: float
    band: str
    band_threshold: float
    base_rate: float
    lift_over_base_rate: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "sk_id_curr": self.sk_id_curr,
            "probability": round(self.probability, 5),
            "risk_band": self.band,
            "band_threshold": self.band_threshold,
            "base_rate": self.base_rate,
            "lift_over_base_rate": self.lift_over_base_rate,
        }


class RiskModel:
    """Loads trained artifacts once and scores applicants against them.

    A missing artifact raises `ModelNotTrained` with instructions rather than
    a bare `FileNotFoundError` - the API route catches this and returns a
    clear "not trained yet" response instead of a 500.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._booster: lgb.Booster | None = None
        self._calibrator = None
        self._feature_list: list[str] | None = None
        self._categorical_features: list[str] = []
        self._bands: dict[str, Any] | None = None
        self._lock = threading.Lock()

    @property
    def is_trained(self) -> bool:
        return (self._settings.models_dir / "model.txt").exists()

    def _ensure_loaded(self) -> None:
        if self._booster is not None:
            return
        with self._lock:
            if self._booster is not None:
                return
            models_dir = self._settings.models_dir
            model_path = models_dir / "model.txt"
            if not model_path.exists():
                raise ModelNotTrained(
                    f"No trained model found at {model_path}. Run: "
                    f"python -m src.ml.train"
                )
            self._booster = lgb.Booster(model_file=str(model_path))
            self._calibrator = joblib.load(models_dir / "calibrator.pkl")
            feature_info = read_json(models_dir / "feature_list.json")
            self._feature_list = feature_info["features"]
            self._categorical_features = feature_info["categorical_features"]
            self._bands = read_json(models_dir / "bands.json")
            log.info("loaded model artifacts from %s", models_dir)

    @property
    def booster(self) -> lgb.Booster:
        """The raw LightGBM model, for SHAP (`src/ml/explain.py`).

        SHAP explains the *raw* model's log-odds output, not the calibrated
        probability the API returns - calibration is a monotone remapping
        fit afterward, and has no per-feature decomposition of its own. The
        UI's explanation is honest about this: it describes contributions as
        "pushed the risk up/down", never as a slice of the exact displayed
        percentage.
        """
        self._ensure_loaded()
        return self._booster

    @property
    def feature_list(self) -> list[str]:
        self._ensure_loaded()
        return list(self._feature_list)

    @property
    def categorical_features(self) -> list[str]:
        self._ensure_loaded()
        return list(self._categorical_features)

    @property
    def base_rate(self) -> float:
        self._ensure_loaded()
        return float(self._bands["base_rate"])

    def align_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Public entry point to the same column alignment `predict_proba`
        uses internally - `explain.py` needs the exact frame SHAP will see,
        not just a probability."""
        return self._prepare_frame(frame)

    def band_for(self, probability: float) -> tuple[str, float]:
        """Which band a probability falls in, and that band's threshold."""
        self._ensure_loaded()
        bands = self._bands["bands"]
        if probability >= bands["High"]["score_threshold"]:
            return "High", bands["High"]["score_threshold"]
        if probability >= bands["Medium"]["score_threshold"]:
            return "Medium", bands["Medium"]["score_threshold"]
        return "Low", bands["Low"]["score_threshold"]

    def _prepare_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Align an arbitrary feature frame to exactly the columns and dtypes
        the model was trained on - same order, same categorical typing, any
        column the model never saw dropped, any it expects but is missing
        filled with NaN (LightGBM handles missing values natively).

        Built as one dict-of-Series then one DataFrame constructor call,
        rather than assigning into an empty frame column by column: the
        latter reallocates the whole block manager on every single-column
        insert (~200 times here), which pandas itself flags as
        PerformanceWarning: DataFrame is highly fragmented - real and
        noticeable at the ~60k-row holdout size this is called on.
        """
        self._ensure_loaded()
        missing = np.full(len(frame), np.nan)
        columns = {
            column: frame[column] if column in frame.columns else missing
            for column in self._feature_list
        }
        aligned = pd.DataFrame(columns, index=frame.index)
        for column in self._categorical_features:
            if column in aligned.columns:
                aligned[column] = aligned[column].astype("category")
        return aligned

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        """Calibrated default probability for each row of a feature frame."""
        self._ensure_loaded()
        aligned = self._prepare_frame(frame)
        return self._calibrator.predict_proba(aligned)[:, 1]

    def predict_one(
        self, sk_id_curr: int, *, conn=None,
    ) -> Prediction:
        """Score one applicant already present in `application_train` or
        `application_test`, by id - the path the UI's "pick an applicant"
        demo uses."""
        self._ensure_loaded()
        conn = conn or get_readonly_connection()
        frame = _load_single_applicant(conn, sk_id_curr)
        if frame.empty:
            raise ValueError(f"SK_ID_CURR {sk_id_curr} not found")
        return self._build_prediction(sk_id_curr, frame)

    def predict_from_raw(self, applicant: dict[str, Any]) -> Prediction:
        """Score a hand-entered applicant (the UI's manual-entry form).

        `applicant` supplies raw application-table fields; any field the
        feature pipeline derives from child tables (bureau, previous
        applications, ...) is simply absent for a hand-entered applicant and
        is treated as missing, exactly as it would be for a genuinely new
        client with no credit history yet.
        """
        self._ensure_loaded()
        frame = pd.DataFrame([applicant])
        return self._build_prediction(None, frame)

    def _build_prediction(
        self, sk_id_curr: int | None, frame: pd.DataFrame,
    ) -> Prediction:
        probability = float(self.predict_proba(frame)[0])
        band, band_threshold = self.band_for(probability)
        base_rate = self._bands["base_rate"]
        return Prediction(
            sk_id_curr=sk_id_curr,
            probability=probability,
            band=band,
            band_threshold=band_threshold,
            base_rate=base_rate,
            lift_over_base_rate=round(probability / base_rate, 2) if base_rate else 0.0,
        )


def _load_single_applicant(conn, sk_id_curr: int) -> pd.DataFrame:
    """Build the full feature row for one applicant via the same SQL used at
    training time, filtered to one id - guarantees prediction-time features
    are computed identically to training-time features.

    Filters by wrapping the feature query as a subquery rather than appending
    a WHERE clause to its text: `build_feature_sql` ends with an ORDER BY
    (needed for reproducible train/test splits - see its docstring), and a
    bare WHERE appended after that is a syntax error.
    """
    def filtered(source_table: str) -> str:
        return (
            f"SELECT * FROM ({build_feature_sql(source_table)}) t "
            f"WHERE t.SK_ID_CURR = {int(sk_id_curr)}"
        )

    frame = conn.execute(filtered("application_train")).df()
    # SK_ID_CURR may exist in application_test too; fall back there if absent
    # from the labelled table (a genuinely unlabelled/held-out applicant).
    if frame.empty:
        frame = conn.execute(filtered("application_test")).df()
    return frame.drop(columns=[c for c in EXCLUDED_COLUMNS if c in frame.columns])


_model: RiskModel | None = None
_model_lock = threading.Lock()


def get_risk_model() -> RiskModel:
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                _model = RiskModel()
    return _model
