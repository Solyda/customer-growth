# src/build_features_survival.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Optional

import numpy as np
import pandas as pd

from src.config import TX_PATH


@dataclass(frozen=True)
class SurvivalFeatureConfig:
    window_30d: int = 30
    window_90d: int = 90
    ratio_eps: float = 1e-6


def build_features_survival(
    transactions: Optional[pd.DataFrame] = None,
    customer_id: Optional[str] = None,
    observation_end_date: Optional[pd.Timestamp] = None,
    cfg: SurvivalFeatureConfig = SurvivalFeatureConfig(),
) -> pd.DataFrame:
    """
    Build Cox covariates exactly as notebooks/04_survival_analysis.ipynb.

    Required input columns:
    - customer_id
    - transaction_date
    - amount

    Output columns:
    - tenure_days
    - recency_at_end
    - txn_per_month
    - amount_per_month
    - freq_30d
    - freq_90d
    - freq_ratio_30d_90d

    Index: customer_id
    """
    df = pd.read_parquet(TX_PATH) if transactions is None else transactions.copy()

    need = {"customer_id", "transaction_date", "amount"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"transactions missing columns: {sorted(missing)}")

    df["transaction_date"] = pd.to_datetime(df["transaction_date"])

    study_end = pd.to_datetime(observation_end_date) if observation_end_date is not None else df["transaction_date"].max()

    g = df.groupby("customer_id", sort=False)
    first_tx = g["transaction_date"].min()
    last_tx = g["transaction_date"].max()

    tenure_days = (last_tx - first_tx).dt.days.clip(lower=0).astype(float)
    recency_at_end = (study_end - last_tx).dt.days.clip(lower=0).astype(float)

    lifetime = g.agg(
        total_txn=("transaction_date", "count"),
        total_amount=("amount", "sum"),
    ).astype(float)

    tenure_months = (tenure_days / 30.0).clip(lower=1.0)
    txn_per_month = (lifetime["total_txn"] / tenure_months).astype(float)
    amount_per_month = (lifetime["total_amount"] / tenure_months).astype(float)

    end_30 = study_end - timedelta(days=cfg.window_30d)
    end_90 = study_end - timedelta(days=cfg.window_90d)

    df_recent = df.assign(
        in_30d=(df["transaction_date"] >= end_30).astype(int),
        in_90d=(df["transaction_date"] >= end_90).astype(int),
    )
    recent = (
        df_recent.groupby("customer_id", sort=False)
        .agg(freq_30d=("in_30d", "sum"), freq_90d=("in_90d", "sum"))
        .reindex(lifetime.index, fill_value=0)
        .astype(float)
    )

    features = pd.DataFrame(
        {
            "tenure_days": tenure_days.reindex(lifetime.index).fillna(0.0),
            "recency_at_end": recency_at_end.reindex(lifetime.index).fillna(0.0),
            "txn_per_month": txn_per_month.reindex(lifetime.index).fillna(0.0),
            "amount_per_month": amount_per_month.reindex(lifetime.index).fillna(0.0),
            "freq_30d": recent["freq_30d"],
            "freq_90d": recent["freq_90d"],
        },
        index=lifetime.index,
    )

    features["freq_ratio_30d_90d"] = features["freq_30d"] / (features["freq_90d"] + cfg.ratio_eps)

    if customer_id is not None:
        if customer_id not in features.index:
            return features.iloc[0:0].copy()
        return features.loc[[customer_id]].copy()

    return features
