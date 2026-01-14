# src/utils.py
import joblib

def load_model(path):
    return joblib.load(path)

def top_k(df, score_col, k_ratio):
    n = int(len(df) * k_ratio)
    return df.sort_values(score_col, ascending=False).head(n)
