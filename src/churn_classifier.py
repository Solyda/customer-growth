# src/churn_classifier.py

"""Churn prediction wrapper.

This project trains a LightGBM model in the notebooks and persists it via
``Booster.save_model(...)`` (LightGBM native text/binary format) rather than pickle.

Therefore inference MUST load the model via ``lightgbm.Booster(model_file=...)``.
This module also supports a fallback to ``joblib.load`` in case you later persist
a sklearn-wrapped LightGBM estimator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import joblib
import lightgbm as lgb
import pandas as pd

from src.build_features_churn import FEATURE_COLS, build_features_churn
from src.config import CHURN_MODEL_PATH


@dataclass(frozen=True)
class ChurnPrediction:
    churn_probability: float
    churn_label: str


class ChurnClassifier:
    """Churn prediction using a LightGBM Booster."""

    def __init__(self, threshold: float = 0.5):
        self.booster = self._load_booster(CHURN_MODEL_PATH)
        self.threshold = float(threshold)

        # Prefer the feature names stored in the Booster (training-time truth).
        self.feature_names = list(self.booster.feature_name()) or list(FEATURE_COLS)

    @staticmethod
    def _load_booster(path) -> lgb.Booster:
        path_str = str(path)

        # 1) Try native LightGBM model file first (matches notebook: Booster.save_model).
        try:
            return lgb.Booster(model_file=path_str)
        except Exception:
            pass

        # 2) Fallback: joblib/pickle (sklearn wrapper or Booster pickled).
        obj = joblib.load(path_str)
        if isinstance(obj, lgb.Booster):
            return obj
        if hasattr(obj, "booster_"):
            return obj.booster_
        if hasattr(obj, "_Booster"):
            return obj._Booster  # noqa: SLF001
        raise TypeError("Churn model does not contain a LightGBM Booster.")

    def predict(
        self,
        transactions: pd.DataFrame,
        customer_id: str,
        snapshot_date: Optional[pd.Timestamp] = None,
    ) -> dict:
        X = build_features_churn(
            transactions=transactions,
            customer_id=customer_id,
            snapshot_date=snapshot_date,
        )

        missing = [c for c in self.feature_names if c not in X.columns]
        extra = [c for c in X.columns if c not in self.feature_names]
        if missing:
            raise ValueError(
                f"Missing churn features: {missing}. Available columns: {list(X.columns)}"
            )
        if extra:
            # Keep deterministic, but do not silently use extra columns.
            X = X[self.feature_names]

        X = X[self.feature_names]

        prob = float(self.booster.predict(X)[0])
        label = "high_risk" if prob >= self.threshold else "low_risk"

        return {
            "churn_probability": prob,
            "churn_label": label,
        }
