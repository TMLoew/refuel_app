# backend/app/services/ml/train_demand_model.py
from pathlib import Path
import joblib
import pandas as pd
from sklearn.linear_model import Ridge

from backend.app.services.features import load_product_mix, load_trends, attach_trends

MODEL_DIR = Path("models")

FEATURES = [
    "visitors",
    "cardio_share",
    "temp_max_c",
    "precip_mm",
    "weekday",
    "month",
    "is_weekend",
    "trend_score",
]


def train_all():
    # Ensure a place exists to store the per-product Ridge models.
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Load engineered features and the (currently empty) trend frame.
    mix = load_product_mix("data/product_mix_daily.csv")
    trends = load_trends()  # aktuell leer, später PyTrends
    mix = attach_trends(mix, trends)

    # Suggested quantity stands in for sales until real targets arrive.
    target_col = "suggested_qty"

    for product in mix["product"].unique():
        dfp = mix[mix["product"] == product].dropna(subset=FEATURES + [target_col])

        if len(dfp) < 20:
            print(f"skip {product}: not enough rows ({len(dfp)})")
            continue

        X = dfp[FEATURES]
        y = dfp[target_col]

        model = Ridge(alpha=1.0, random_state=0)
        model.fit(X, y)

        # Save each product-specific model under a deterministic filename.
        model_path = MODEL_DIR / f"ridge_{product.replace(' ', '_')}.joblib"
        joblib.dump(model, model_path)
        print(f"saved model for {product} with {len(dfp)} rows")

if __name__ == "__main__":
    from sklearn.exceptions import ConvergenceWarning
    import warnings
    warnings.simplefilter("ignore", ConvergenceWarning)

    train_all()
