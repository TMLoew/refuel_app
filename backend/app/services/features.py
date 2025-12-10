import pandas as pd

BASE_FEATURES = ["visitors","cardio_share","temp_max_c","precip_mm","weekday","month","is_weekend"]  # core columns for training

def load_product_mix(path="data/product_mix_daily.csv") -> pd.DataFrame:
    """Load the mix file and add basic date-derived features."""
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df["weekday"] = df["date"].dt.weekday
    df["month"] = df["date"].dt.month
    df["is_weekend"] = (df["weekday"] >= 5).astype(int)
    return df
