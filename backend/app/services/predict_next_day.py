# backend/app/services/predict_next_day.py
from pathlib import Path
import pandas as pd
import joblib

from backend.app.services.features import load_product_mix, load_trends, attach_trends

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

MODEL_DIR = Path("models")

def latest_day_rows() -> pd.DataFrame:
    df = load_product_mix("data/product_mix_daily.csv")
    dmax = df["date"].max()
    today = df[df["date"] == dmax].copy()

    # Demo: nächster Tag = letzter Tag + 1
    nd = pd.to_datetime(dmax) + pd.Timedelta(days=1)
    today["date"] = nd
    today["weekday"] = nd.weekday()
    today["month"] = nd.month
    today["is_weekend"] = (today["weekday"] >= 5).astype(int)

    trends = load_trends()  # aktuell leer, später PyTrends
    today = attach_trends(today, trends)
    return today

def predict_next_day():
    rows = latest_day_rows()
    out_rows = []

    for _, r in rows.iterrows():
        prod = r["product"]
        model_path = MODEL_DIR / f"ridge_{prod.replace(' ', '_')}.joblib"
        if not model_path.exists():
            print(f"no model for {prod}, skipping")
            continue

        model = joblib.load(model_path)
        x = r[FEATURES].values.reshape(1, -1)
        yhat = float(model.predict(x)[0])

        out_rows.append({
            "date": r["date"].date().isoformat(),
            "product": prod,
            "predicted_qty": int(round(max(0, yhat))),
        })

    df_out = pd.DataFrame(out_rows)
    Path("data").mkdir(parents=True, exist_ok=True)
    out_path = "data/forecast_next_day.csv"
    df_out.to_csv(out_path, index=False)
    print(f"saved: {out_path}")
    return df_out

if __name__ == "__main__":
    predict_next_day()

