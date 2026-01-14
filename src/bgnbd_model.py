# src/bgnbd_model.py

"""BG/NBD inference wrapper.

Notebook persists the model via ``BetaGeoFitter.save_model(...)`` which is a pickle.
To avoid version/behavior differences of lifetimes' ``load_model`` API, we load
the artifact via ``joblib.load``.
"""

from __future__ import annotations

import joblib
from lifetimes import BetaGeoFitter

from src.build_features_bgnbd_gamma import build_bgnbd_summary
from src.config import BGNBD_MODEL_PATH


class BGNBDModel:
    def __init__(self):
        model = joblib.load(str(BGNBD_MODEL_PATH))
        if not isinstance(model, BetaGeoFitter):
            raise TypeError(f"Expected BetaGeoFitter, got {type(model)}")
        self.model: BetaGeoFitter = model

    def predict(
        self,
        transactions,
        customer_id: str,
        horizon_days: int = 60,
    ) -> dict:
        summary = build_bgnbd_summary(transactions=transactions, customer_id=customer_id)

        freq = float(summary["frequency"].iloc[0])
        recency = float(summary["recency"].iloc[0])
        T = float(summary["T"].iloc[0])

        p_alive = self.model.conditional_probability_alive(freq, recency, T)
        expected_tx = self.model.conditional_expected_number_of_purchases_up_to_time(
            horizon_days, freq, recency, T
        )

        return {
            "p_alive": float(p_alive),
            "expected_transactions": float(expected_tx),
        }
