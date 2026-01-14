# src/survival_model.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from src.build_features_survival import build_features_survival
from src.config import SURVIVAL_MODEL_PATH


@dataclass(frozen=True)
class SurvivalPredictConfig:
    """
    Notebook 04 behavior:
    - survival_curve: unconditional S(t) at horizons (t is days since origin)
    - expected_remaining_lifetime: integral of conditional survival from "now"
      S(a+t)/S(a) over t in [0..T]
    """
    horizons_days_default: tuple[int, ...] = (7, 30, 60, 90, 180, 365)
    T_days: int = 365
    dt: int = 7
    eps: float = 1e-10


class SurvivalModel:
    """CoxPH inference wrapper (robust schema + provides _make_X)."""

    def __init__(self, model_path: str = SURVIVAL_MODEL_PATH):
        self.model_path = model_path
        self.model = self._load_model(model_path)

    @staticmethod
    def _load_model(model_path: str):
        # Prefer lifelines native load; fallback joblib (common in notebooks).
        try:
            from lifelines import CoxPHFitter

            m = CoxPHFitter()
            m.load(model_path)
            return m
        except Exception:
            import joblib  # noqa: WPS433

            return joblib.load(model_path)

    def _expected_covariates(self) -> list[str]:
        if hasattr(self.model, "params_"):
            return [str(c) for c in self.model.params_.index]
        raise RuntimeError("Cannot infer covariate names from Cox model (missing params_).")

    @staticmethod
    def _study_end(transactions: pd.DataFrame, observation_end_date: Optional[pd.Timestamp]) -> pd.Timestamp:
        if observation_end_date is not None:
            return pd.to_datetime(observation_end_date)
        return pd.to_datetime(pd.to_datetime(transactions["transaction_date"]).max())

    @staticmethod
    def _first_tx(transactions: pd.DataFrame, customer_id: str) -> pd.Timestamp:
        tx = transactions.loc[transactions["customer_id"] == customer_id, "transaction_date"]
        if tx.empty:
            raise ValueError(f"customer_id not found in transactions: {customer_id}")
        return pd.to_datetime(tx.min())

    def _make_X(
        self,
        transactions: pd.DataFrame,
        customer_id: str,
        observation_end_date: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        """
        Backward-compatible helper (some code calls SurvivalModel._make_X()).

        Returns a single-row DataFrame with covariates aligned to model.params_ order.
        """
        study_end = self._study_end(transactions, observation_end_date)

        X = build_features_survival(
            transactions=transactions,
            customer_id=customer_id,
            observation_end_date=study_end,
        )

        if X.empty:
            raise ValueError(f"customer_id not found after feature build: {customer_id}")

        expected = self._expected_covariates()
        missing = [c for c in expected if c not in X.columns]
        if missing:
            raise ValueError(f"Missing survival covariates: {missing}. Available: {list(X.columns)}")

        return X[expected].copy()

    def predict(
        self,
        transactions: pd.DataFrame,
        customer_id: str,
        horizons_days: Optional[Iterable[int]] = None,
        observation_end_date: Optional[pd.Timestamp] = None,
        cfg: SurvivalPredictConfig = SurvivalPredictConfig(),
    ) -> dict:
        """
        Returns:
        - survival_curve: unconditional S(t) from origin at horizons_days (days)
        - expected_remaining_lifetime: conditional ERL from now over [0..T]
        """
        study_end = self._study_end(transactions, observation_end_date)
        first_tx = self._first_tx(transactions, customer_id)
        age_now = float((study_end - first_tx).days)

        X = self._make_X(transactions, customer_id, observation_end_date=study_end)

        # 1) Unconditional survival curve S(t) from origin
        hs = list(cfg.horizons_days_default if horizons_days is None else horizons_days)
        hs = sorted({int(h) for h in hs if int(h) >= 0})

        survival_curve: list[dict] = []
        if hs:
            S_h = self.model.predict_survival_function(X, times=hs).iloc[:, 0].astype(float)
            survival_curve = [{"day": int(t), "prob": float(S_h.loc[t])} for t in hs]

        # 2) Expected remaining lifetime (conditional from now): integrate S(a+t)/S(a)
        future_times = np.arange(0, cfg.T_days + cfg.dt, cfg.dt, dtype=float)

        t_max_needed = int(max(0.0, age_now) + cfg.T_days)
        t_grid = np.arange(0, t_max_needed + cfg.dt, cfg.dt, dtype=float)

        S_grid = self.model.predict_survival_function(X, times=t_grid).iloc[:, 0].astype(float)
        t_src = S_grid.index.values.astype(float)
        s_src = S_grid.values.astype(float)

        S_a = float(np.interp(age_now, t_src, s_src))
        S_a = float(np.clip(S_a, cfg.eps, 1.0))

        S_a_t = np.interp(age_now + future_times, t_src, s_src)
        S_cond = S_a_t / S_a

        expected_remaining_lifetime = float(np.trapz(S_cond, x=future_times))

        return {
            "customer_id": customer_id,
            "survival_curve": survival_curve,
            "expected_remaining_lifetime": expected_remaining_lifetime,
        }
