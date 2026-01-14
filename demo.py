# src/demo.py

import argparse
import json
import sys

import pandas as pd

from src.config import TX_PATH, DEFAULT_CLV_HORIZON_MONTHS
from src.churn_classifier import ChurnClassifier
from src.bgnbd_model import BGNBDModel
from src.survival_model import SurvivalModel
from src.clv_bgnbd import CLVBGNBD
from src.clv_survival import CLVSurvival


# churn filter threshold 
MIN_CHURN_PROB_THRESHOLD = 0.7785214653862376


def load_transactions() -> pd.DataFrame:
    df = pd.read_parquet(TX_PATH)
    df["transaction_date"] = pd.to_datetime(df["transaction_date"])
    return df


def print_json(obj) -> None:
    print(json.dumps(obj, indent=2, ensure_ascii=False))


def score_customer(df: pd.DataFrame, customer_id: str) -> dict:
    churn = ChurnClassifier().predict(df, customer_id)
    bgnbd = BGNBDModel().predict(df, customer_id, horizon_days=60)
    survival = SurvivalModel().predict(df, customer_id)
    clv_bgnbd = CLVBGNBD().estimate(df, customer_id)
    clv_survival = CLVSurvival().estimate(df, customer_id)

    return {
        "customer_id": customer_id,
        "churn_probability": churn["churn_probability"],
        "p_alive": bgnbd["p_alive"],
        "expected_remaining_lifetime": survival["expected_remaining_lifetime"],
        "clv_bgnbd": clv_bgnbd["clv"],
        "clv_survival": clv_survival["clv"],
    }


def predict_churn(df: pd.DataFrame, customer_id: str, horizon_days: int) -> dict:
    if int(horizon_days) != 90:
        raise ValueError(
            f"Churn model is trained on fixed horizon=90 days; received horizon_days={horizon_days}."
        )
    return ChurnClassifier().predict(df, customer_id)


def predict_survival(df: pd.DataFrame, customer_id: str) -> dict:
    return SurvivalModel().predict(df, customer_id)


def estimate_clv(df: pd.DataFrame, customer_id: str, method: str, horizon_months: int) -> dict:
    if method == "bgnbd":
        return CLVBGNBD().estimate(df, customer_id, horizon_months)
    if method == "survival":
        return CLVSurvival().estimate(df, customer_id, horizon_months)
    raise ValueError("method must be 'bgnbd' or 'survival'")


def rank_customers(
    df: pd.DataFrame,
    top_k: int,
    strategy: str,
    method: str,
    horizon_months: int,
) -> dict:
    """
    Ranking logic:

    - strategy=high_churn:
        priority_score = churn_probability

    - strategy=high_clv_high_churn:
        FILTER: churn_probability >= MIN_CHURN_PROB_THRESHOLD 
        priority_score = CLV (depends on method)
    """
    churn_model = ChurnClassifier()
    clv_bgnbd = CLVBGNBD()
    clv_survival = CLVSurvival()

    results: list[dict] = []

    customer_ids = df["customer_id"].dropna().astype(str).unique()

    for cid in customer_ids:
        churn_prob = float(churn_model.predict(df, cid)["churn_probability"])

        if strategy == "high_churn":
            results.append(
                {
                    "customer_id": cid,
                    "churn_probability": churn_prob,
                    "clv": None,
                    "priority_score": churn_prob,
                }
            )
            continue

        if strategy == "high_clv_high_churn":
            if churn_prob < MIN_CHURN_PROB_THRESHOLD:
                continue

            if method == "bgnbd":
                clv = float(clv_bgnbd.estimate(df, cid, horizon_months)["clv"])
            elif method == "survival":
                clv = float(clv_survival.estimate(df, cid, horizon_months)["clv"])
            else:
                raise ValueError("method must be 'bgnbd' or 'survival'")

            results.append(
                {
                    "customer_id": cid,
                    "churn_probability": churn_prob,
                    "clv": clv,
                    "priority_score": clv,
                }
            )
            continue

        raise ValueError("Unknown strategy")

    # tie-break: same score -> prefer higher churn_probability
    results = sorted(
        results, key=lambda x: (x["priority_score"], x["churn_probability"]), reverse=True
    )[: int(top_k)]

    return {
        "strategy": strategy,
        "method": method,
        "horizon_months": int(horizon_months),
        "min_churn_prob_threshold": float(MIN_CHURN_PROB_THRESHOLD),
        "customers": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser("Customer Growth Demo")
    sub = parser.add_subparsers(dest="command", required=True)

    p1 = sub.add_parser("score_customer")
    p1.add_argument("customer_id")

    p2 = sub.add_parser("predict_churn")
    p2.add_argument("customer_id")
    p2.add_argument("--horizon_days", type=int, default=90)

    p3 = sub.add_parser("predict_survival")
    p3.add_argument("customer_id")

    p4 = sub.add_parser("estimate_clv")
    p4.add_argument("customer_id")
    p4.add_argument("--method", choices=["bgnbd", "survival"], required=True)
    p4.add_argument("--horizon_months", type=int, default=DEFAULT_CLV_HORIZON_MONTHS)

    p5 = sub.add_parser("rank_customers_for_retention")
    p5.add_argument("--top_k", type=int, default=100)
    p5.add_argument("--strategy", choices=["high_churn", "high_clv_high_churn"], required=True)
    p5.add_argument("--method", choices=["bgnbd", "survival"], default="bgnbd")
    p5.add_argument("--horizon_months", type=int, default=DEFAULT_CLV_HORIZON_MONTHS)

    args = parser.parse_args()
    df = load_transactions()

    try:
        if args.command == "score_customer":
            out = score_customer(df, args.customer_id)
        elif args.command == "predict_churn":
            out = predict_churn(df, args.customer_id, args.horizon_days)
        elif args.command == "predict_survival":
            out = predict_survival(df, args.customer_id)
        elif args.command == "estimate_clv":
            out = estimate_clv(df, args.customer_id, args.method, args.horizon_months)
        elif args.command == "rank_customers_for_retention":
            out = rank_customers(
                df=df,
                top_k=args.top_k,
                strategy=args.strategy,
                method=args.method,
                horizon_months=args.horizon_months,
            )
        else:
            raise RuntimeError("Unknown command")

        print_json(out)
    except Exception as e:
        print_json({"error": str(e)})
        sys.exit(1)


if __name__ == "__main__":
    main()
