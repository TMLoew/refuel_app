# backend/app/services/features.py
from pathlib import Path
import pandas as pd


def load_product_mix(path="data/product_mix_daily.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    # Convert dates once so weekday/month helpers keep working downstream.
    df["date"] = pd.to_datetime(df["date"])
    df["weekday"] = df["date"].dt.weekday
    df["month"] = df["date"].dt.month
    # Binary weekend flag captures the traffic jump on Saturday/Sunday.
    df["is_weekend"] = (df["weekday"] >= 5).astype(int)
    # Trend score remains a constant placeholder until PyTrends is wired.
    df["trend_score"] = 0.5
    return df


def load_trends(path="data/trends_3m_CH.csv") -> pd.DataFrame:
    # Placeholder return until actual trend data is available.
    return pd.DataFrame()


def attach_trends(mix: pd.DataFrame, trends: pd.DataFrame) -> pd.DataFrame:
    # No-op for now because the trend source is still empty.
    return mix
