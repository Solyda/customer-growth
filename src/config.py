# src/config.py
from pathlib import Path

# =========================
# Project base
# =========================
BASE_DIR = Path(__file__).resolve().parents[1]

# =========================
# Data paths
# =========================
DATA_DIR = BASE_DIR / "data"
DATA_INTERIM = DATA_DIR / "interim"
DATA_PROCESSED = DATA_DIR / "processed"

TX_PATH = DATA_INTERIM / "transactions_clean.parquet"
CHURN_LABEL_PATH = DATA_PROCESSED / "churn_labels.parquet"

# =========================
# Artifacts
# =========================
ARTIFACTS_DIR = BASE_DIR / "artifacts"
ART_MODELS = ARTIFACTS_DIR / "models"
ART_SCORES = ARTIFACTS_DIR / "scores"
ART_REPORTS = ARTIFACTS_DIR / "reports"

# =========================
# Model artifact paths
# =========================
BGNBD_MODEL_PATH = ART_MODELS / "bgnbd_model.pkl"
GAMMA_GAMMA_MODEL_PATH = ART_MODELS / "gamma_gamma_model.pkl"
CHURN_MODEL_PATH = ART_MODELS / "churn_lgbm.pkl"
SURVIVAL_MODEL_PATH = ART_MODELS / "survival_cox.pkl"

# =========================
# Runtime defaults
# =========================
DEFAULT_CLV_HORIZON_MONTHS = 12
DEFAULT_CHURN_HORIZON_DAYS = 60
DEFAULT_RETENTION_TOP_K = 100
CHURN_LABEL_HORIZON_DAYS = 90
DEFAULT_PREDICTION_HORIZON_DAYS = 60

# =========================
# Ensure dirs exist
# =========================
for p in [DATA_INTERIM, DATA_PROCESSED, ART_MODELS, ART_SCORES, ART_REPORTS]:
    p.mkdir(parents=True, exist_ok=True)

