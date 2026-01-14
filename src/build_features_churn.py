# src/build_features_churn.py

from typing import Optional
from datetime import timedelta
import pandas as pd
from src.config import TX_PATH


FEATURE_COLS = [
    "recency",
    "freq_30d", "freq_60d", "freq_90d",
    "monetary_30d", "monetary_60d", "monetary_90d",
    "freq_ratio_30_90", "freq_ratio_60_90",
]


def build_features_churn(
    transactions: Optional[pd.DataFrame] = None,
    customer_id: Optional[str] = None,
    snapshot_date: Optional[pd.Timestamp] = None,
):
    """
    Build churn features EXACTLY as in training notebook.
    Churn label horizon: 90 days (fixed).
    """

    if transactions is None:
        transactions = pd.read_parquet(TX_PATH)

    df = transactions.copy()
    df["transaction_date"] = pd.to_datetime(df["transaction_date"])

    if snapshot_date is None:
        snapshot_date = df["transaction_date"].max()

    hist = df[df["transaction_date"] <= snapshot_date].copy()

    agg = hist.groupby("customer_id").agg(
        # Recency
        recency=(
            "transaction_date",
            lambda x: (snapshot_date - x.max()).days
        ),

        # Frequency windows
        freq_30d=(
            "transaction_date",
            lambda x: (x >= snapshot_date - timedelta(days=30)).sum()
        ),
        freq_60d=(
            "transaction_date",
            lambda x: (x >= snapshot_date - timedelta(days=60)).sum()
        ),
        freq_90d=(
            "transaction_date",
            lambda x: (x >= snapshot_date - timedelta(days=90)).sum()
        ),

        # Monetary windows
        monetary_30d=(
            "amount",
            lambda x: x[
                hist.loc[x.index, "transaction_date"] >= snapshot_date - timedelta(days=30)
            ].sum()
        ),
        monetary_60d=(
            "amount",
            lambda x: x[
                hist.loc[x.index, "transaction_date"] >= snapshot_date - timedelta(days=60)
            ].sum()
        ),
        monetary_90d=(
            "amount",
            lambda x: x[
                hist.loc[x.index, "transaction_date"] >= snapshot_date - timedelta(days=90)
            ].sum()
        ),
    )

    # Ratio / decay features
    agg["freq_ratio_30_90"] = agg["freq_30d"] / (agg["freq_90d"] + 1e-6)
    agg["freq_ratio_60_90"] = agg["freq_60d"] / (agg["freq_90d"] + 1e-6)

    if customer_id is not None:
        if customer_id not in agg.index:
            raise ValueError(f"Customer {customer_id} not found.")
        agg = agg.loc[[customer_id]]

    return agg[FEATURE_COLS].reset_index(drop=True)
