# src/clv_survival.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from src.survival_model import SurvivalModel
from src.gamma_gamma_model import GammaGammaModel
from src.build_features_bgnbd_gamma import build_bgnbd_summary


@dataclass(frozen=True)
class SurvivalCLVConfig:
    annual_discount_rate: float = 0.10
    dt: int = 7
    eps: float = 1e-10


class CLVSurvival:
    """
    Matches notebooks/05_2_clv_survival_gamma_gamma.ipynb time-dependent CLV:
      CLV = ∫ S_cond(t) * lambda_per_day * E[order_value] * discount(t) dt
    """

    def __init__(self, cfg: SurvivalCLVConfig = SurvivalCLVConfig()):
        self.survival = SurvivalModel()
        self.gg = GammaGammaModel()
        self.cfg = cfg

    @staticmethod
    def _study_end(transactions: pd.DataFrame, observation_end_date: Optional[pd.Timestamp]) -> pd.Timestamp:
        if observation_end_date is not None:
            return pd.to_datetime(observation_end_date)
        return pd.to_datetime(pd.to_datetime(transactions["transaction_date"]).max())

    @staticmethod
    def _customer_tx(transactions: pd.DataFrame, customer_id: str) -> pd.DataFrame:
        tx = transactions.loc[transactions["customer_id"] == customer_id].copy()
        if tx.empty:
            raise ValueError(f"customer_id not found in transactions: {customer_id}")
        tx["transaction_date"] = pd.to_datetime(tx["transaction_date"])
        return tx

    def estimate(
        self,
        transactions: pd.DataFrame,
        customer_id: str,
        horizon_months: int = 12,  # notebook integrates 365d ~ 12m
        observation_end_date: Optional[pd.Timestamp] = None,
    ) -> dict:
        cfg = self.cfg
        study_end = self._study_end(transactions, observation_end_date)

        tx = self._customer_tx(transactions, customer_id)
        first_tx = tx["transaction_date"].min()
        last_tx = tx["transaction_date"].max()

        tenure_days = float((last_tx - first_tx).days)
        tenure_days = max(tenure_days, 1.0)

        total_txn = float(len(tx))
        lambda_per_day = total_txn / tenure_days

        # expected avg order value (Gamma-Gamma) exactly like notebook
        summary = build_bgnbd_summary(transactions=transactions, customer_id=customer_id)
        freq = float(summary["frequency"].iloc[0])
        monetary_value = float(summary["monetary_value"].iloc[0])

        if freq <= 0 or monetary_value <= 0 or lambda_per_day <= 0:
            return {"method": "survival", "clv": 0.0, "horizon_months": int(horizon_months)}

        exp_avg_order_value = float(
            self.gg.model.conditional_expected_average_profit(
                summary["frequency"],
                summary["monetary_value"],
            ).iloc[0]
        )

        # Conditional survival from now: S(a+t)/S(a)
        age_now = float((study_end - first_tx).days)
        age_now = max(age_now, 0.0)

        horizon_days = int(round(horizon_months * 365.0 / 12.0))  # 12 -> 365
        future_times = np.arange(0, horizon_days + cfg.dt, cfg.dt, dtype=float)

        X = self.survival._make_X(transactions, customer_id, observation_end_date=study_end)

        times_needed = np.unique(np.concatenate(([age_now], age_now + future_times))).astype(float)
        S = self.survival.model.predict_survival_function(X, times=times_needed).iloc[:, 0].astype(float)

        t_src = S.index.values.astype(float)
        s_src = S.values.astype(float)

        S_a = float(np.interp(age_now, t_src, s_src))
        S_a = float(np.clip(S_a, cfg.eps, 1.0))

        S_a_t = np.interp(age_now + future_times, t_src, s_src)
        S_cond = S_a_t / S_a

        discount = 1.0 / ((1.0 + cfg.annual_discount_rate) ** (future_times / 365.0))

        clv_curve = S_cond * lambda_per_day * exp_avg_order_value * discount
        clv = float(np.trapz(clv_curve, x=future_times))

        return {"method": "survival", "clv": clv, "horizon_months": int(horizon_months)}
