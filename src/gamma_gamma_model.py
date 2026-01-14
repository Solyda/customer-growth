# src/gamma_gamma_model.py

"""Gamma-Gamma inference wrapper.

Notebook persists the model via ``GammaGammaFitter.save_model(...)`` (pickle).
We load via ``joblib.load`` to keep behavior stable.
"""

from __future__ import annotations

import joblib
import pandas as pd
from lifetimes import GammaGammaFitter

from src.config import GAMMA_GAMMA_MODEL_PATH


class GammaGammaModel:
    def __init__(self):
        model = joblib.load(str(GAMMA_GAMMA_MODEL_PATH))
        if not isinstance(model, GammaGammaFitter):
            raise TypeError(f"Expected GammaGammaFitter, got {type(model)}")
        self.model: GammaGammaFitter = model

    def predict_expected_monetary(self, summary_df: pd.DataFrame) -> float:
        """Return expected average profit / transaction for a single customer."""
        freq = float(summary_df["frequency"].iloc[0])
        monetary = float(summary_df["monetary_value"].iloc[0])

        exp_monetary = self.model.conditional_expected_average_profit(freq, monetary)
        return float(exp_monetary)
