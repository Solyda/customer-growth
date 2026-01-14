# src/clv_bgnbd.py
from __future__ import annotations

import pandas as pd

from src.bgnbd_model import BGNBDModel
from src.gamma_gamma_model import GammaGammaModel
from src.build_features_bgnbd_gamma import build_bgnbd_summary


class CLVBGNBD:
    """
    Matches notebooks/05_1_clv_bgnbd_gamma_gamma.ipynb:
      ggf.customer_lifetime_value(
          bgf, frequency, recency, T, monetary_value,
          time=<horizon_months>, discount_rate=0.01, freq="D"
      )
    """

    def __init__(self, discount_rate: float = 0.01):
        self.bgnbd = BGNBDModel()
        self.gg = GammaGammaModel()
        self.discount_rate = float(discount_rate)

    def estimate(
        self,
        transactions: pd.DataFrame,
        customer_id: str,
        horizon_months: int = 3,  # notebook default: CLV_3m
    ) -> dict:
        summary = build_bgnbd_summary(transactions=transactions, customer_id=customer_id)

        freq = float(summary["frequency"].iloc[0])
        monetary_value = float(summary["monetary_value"].iloc[0])

        if freq <= 0 or monetary_value <= 0:
            clv = 0.0
        else:
            clv_series = self.gg.model.customer_lifetime_value(
                self.bgnbd.model,
                summary["frequency"],
                summary["recency"],
                summary["T"],
                summary["monetary_value"],
                time=int(horizon_months),
                discount_rate=self.discount_rate,
                freq="D",
            )
            clv = float(clv_series.iloc[0])

        return {"method": "bgnbd", "clv": clv, "horizon_months": int(horizon_months)}
