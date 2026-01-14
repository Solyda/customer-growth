# src/build_features_bgnbd_gamma.py
from __future__ import annotations

import pandas as pd
from lifetimes.utils import summary_data_from_transaction_data

from src.config import TX_PATH


def build_bgnbd_summary(
    transactions: pd.DataFrame | None = None,
    customer_id: str | None = None,
    observation_end_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """
    Build Lifetimes summary table (match notebooks).
    Output columns:
      - frequency
      - recency
      - T
      - monetary_value
    Units: days (freq="D")
    """
    df = pd.read_parquet(TX_PATH) if transactions is None else transactions.copy()
    df["transaction_date"] = pd.to_datetime(df["transaction_date"])

    if observation_end_date is None:
        observation_end_date = df["transaction_date"].max()

    summary = summary_data_from_transaction_data(
        df,
        customer_id_col="customer_id",
        datetime_col="transaction_date",
        monetary_value_col="amount",
        observation_period_end=observation_end_date,
        freq="D",
    )

    summary = summary[["frequency", "recency", "T", "monetary_value"]]

    if customer_id is not None:
        if customer_id not in summary.index:
            raise ValueError(f"Customer {customer_id} not found in transactions.")
        summary = summary.loc[[customer_id]]

    return summary
