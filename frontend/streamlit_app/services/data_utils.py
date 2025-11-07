"""Shared data utilities for Streamlit pages."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .weather_pipeline import (
    Ingredient,
    build_synthetic_weather_frame,
    fetch_hourly_weather_frame,
    month_to_season,
    seasonality,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PREFERRED_DATASETS = [
    PROJECT_ROOT / "data" / "gym_badges_0630_2200_long.csv",
    PROJECT_ROOT / "data" / "gym_badges.csv",
]
DATA_FILE = next((path for path in PREFERRED_DATASETS if path.exists()), PREFERRED_DATASETS[-1])
PROCUREMENT_PLAN_FILE = PROJECT_ROOT / "data" / "procurement_plan.csv"
POS_LOG_FILE = PROJECT_ROOT / "data" / "pos_runtime_log.csv"

WEATHER_SCENARIOS: Dict[str, Dict[str, float]] = {
    "Temperate & sunny": {"temp_offset": 2.0, "precip_multiplier": 0.7, "humidity_offset": -3},
    "Cold snap": {"temp_offset": -6.0, "precip_multiplier": 1.0, "humidity_offset": 4},
    "Humid heatwave": {"temp_offset": 5.5, "precip_multiplier": 0.4, "humidity_offset": 8},
    "Storm front": {"temp_offset": -1.0, "precip_multiplier": 1.8, "humidity_offset": 10},
}

SNACK_PROMOS: Dict[str, Dict[str, float]] = {
    "Baseline offer": {"demand_boost": 0.0, "price_delta": 0.0},
    "Protein sampler": {"demand_boost": 0.12, "price_delta": 0.05},
    "Buy-one-get-one": {"demand_boost": 0.28, "price_delta": -0.18},
    "Corporate snack box": {"demand_boost": 0.18, "price_delta": 0.1},
}

CHECKIN_FEATURES = [
    "hour",
    "is_weekend",
    "temperature_c",
    "precipitation_mm",
    "humidity_pct",
    "event_intensity",
    "snack_price",
    "sin_hour",
    "cos_hour",
    "sin_doy",
    "cos_doy",
]

SNACK_FEATURES = CHECKIN_FEATURES + ["checkins"]


@st.cache_data(show_spinner=False)
def load_enriched_data(
    csv_path: Path = DATA_FILE,
    use_weather_api: bool = False,
    cache_buster: float = 0.0,
) -> pd.DataFrame:
    """Load gym dataset and enrich it with weather + snack context."""
    _ = cache_buster  # ensures cache invalidation when requested
    if not csv_path.exists():
        return pd.DataFrame()

    df = pd.read_csv(csv_path)
    if df.empty:
        return df

    timestamp_candidates = ["ts_local_naive", "ts_local", "timestamp"]
    timestamp_col = next((col for col in timestamp_candidates if col in df.columns), None)
    if timestamp_col is None:
        raise ValueError(
            "Expected a timestamp column (one of: ts_local_naive, ts_local, timestamp) in gym_badges.csv."
        )

    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df = df.rename(columns={timestamp_col: "timestamp"})
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["hour"] = df["timestamp"].dt.hour
    df["weekday"] = df["timestamp"].dt.weekday
    df["is_weekend"] = df["weekday"].isin([5, 6]).astype(int)
    df["day_of_year"] = df["timestamp"].dt.dayofyear

    rng = np.random.default_rng(42)
    timestamps = tuple(df["timestamp"])
    weather_source = "synthetic"
    weather_frame = pd.DataFrame()
    weather_meta_extra: Dict[str, float] = {}

    if use_weather_api:
        try:
            weather_frame, weather_meta_extra = fetch_hourly_weather_frame(timestamps)
            if not weather_frame.empty:
                weather_source = "open-meteo"
        except Exception:
            weather_frame = pd.DataFrame()

    if weather_frame.empty:
        weather_frame = build_synthetic_weather_frame(timestamps)

    df = df.merge(weather_frame, on="timestamp", how="left")
    weather_cols = ["temperature_c", "precipitation_mm", "humidity_pct"]
    df[weather_cols] = df[weather_cols].ffill().bfill()
    df.attrs["weather_source"] = weather_source

    df["event_intensity"] = np.where(
        (df["weekday"].isin([1, 3]) & df["hour"].between(17, 20)),
        1.6,
        np.where(df["weekday"].isin([5, 6]) & df["hour"].between(8, 12), 1.2, 0.5),
    )

    banana = Ingredient("Banana", 2.5)
    df["season"] = df["timestamp"].dt.month.apply(month_to_season)
    season_values = df["season"].apply(
        lambda s: seasonality.get(banana.name, {}).get(s, (1.0, 1.0))
    )
    df["availability_index"] = season_values.apply(lambda tup: tup[0])
    df["price_index"] = season_values.apply(lambda tup: tup[1])
    base_prices = np.array(
        [banana.seasonal_price(season, price_idx) for season, price_idx in zip(df["season"], df["price_index"])]
    )
    df["snack_price"] = np.clip(base_prices + rng.normal(0, 0.05, len(df)), 2.0, 4.5)
    df["snack_units"] = np.clip(
        0.42 * df["checkins"]
        + 0.13 * df["temperature_c"]
        - 0.35 * df["precipitation_mm"]
        + 4.6 * df["event_intensity"]
        + 1.5 * df["availability_index"]
        - 1.4 * (df["snack_price"] - df["snack_price"].median())
        + rng.normal(0, 1.8, len(df)),
        0,
        None,
    )
    df["snack_revenue"] = df["snack_units"] * df["snack_price"]
    df["total_sessions"] = df["treadmill_sessions"] + df["strength_sessions"] + df["other_sessions"]
    df["utilization_ratio"] = (
        df["total_sessions"] / df["checkins"].clip(lower=1)
    ).clip(upper=2.5)
    df["weather_label"] = pd.cut(
        df["temperature_c"],
        bins=[-10, 5, 15, 25, 40],
        labels=["Freezing", "Cool", "Mild", "Warm"],
    )
    df = add_time_signals(df)
    df.attrs["weather_meta"] = {
        "source": weather_source,
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="minutes"),
        "coverage_start": weather_meta_extra.get("coverage_start", df["timestamp"].min().isoformat()),
        "coverage_end": weather_meta_extra.get("coverage_end", df["timestamp"].max().isoformat()),
        "latency_ms": weather_meta_extra.get("latency_ms"),
        "chunks": weather_meta_extra.get("chunks"),
    }
    return df


def add_time_signals(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Attach cyclical encodings required by the simple regressors."""
    enriched = dataframe.copy()
    enriched["sin_hour"] = np.sin(2 * np.pi * enriched["hour"] / 24)
    enriched["cos_hour"] = np.cos(2 * np.pi * enriched["hour"] / 24)
    enriched["sin_doy"] = np.sin(2 * np.pi * enriched["day_of_year"] / 365)
    enriched["cos_doy"] = np.cos(2 * np.pi * enriched["day_of_year"] / 365)
    return enriched


@st.cache_resource(show_spinner=False)
def train_models(df: pd.DataFrame) -> Tuple[Pipeline, Pipeline]:
    """Train lightweight regressors for attendance and snack demand."""
    if df.empty:
        return None, None

    feature_df = add_time_signals(df)
    required_cols = set(CHECKIN_FEATURES)
    missing = sorted(required_cols - set(feature_df.columns))
    if missing:
        raise ValueError(f"Missing engineered features: {missing}")
    checkin_model = Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", LinearRegression()),
        ]
    )
    checkin_model.fit(feature_df[CHECKIN_FEATURES], feature_df["checkins"])

    snack_model = Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", LinearRegression()),
        ]
    )
    snack_model.fit(feature_df[SNACK_FEATURES], feature_df["snack_units"])
    return checkin_model, snack_model


def build_scenario_forecast(
    history: pd.DataFrame,
    models: Tuple[Pipeline, Pipeline],
    scenario: Dict[str, float],
    anchor_timestamp: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """Create a future dataframe incorporating scenario adjustments."""
    checkin_model, snack_model = models
    if checkin_model is None or snack_model is None:
        return pd.DataFrame()

    horizon = scenario["horizon_hours"]
    anchor_ts = history["timestamp"].max()
    if anchor_timestamp is not None:
        anchor_ts = pd.to_datetime(anchor_timestamp)
    future_index = pd.date_range(anchor_ts + pd.Timedelta(hours=1), periods=horizon, freq="h")
    future = pd.DataFrame({"timestamp": future_index})
    future["hour"] = future["timestamp"].dt.hour
    future["weekday"] = future["timestamp"].dt.weekday
    future["is_weekend"] = future["weekday"].isin([5, 6]).astype(int)
    future["day_of_year"] = future["timestamp"].dt.dayofyear

    hourly_profile = (
        history.groupby(["weekday", "hour"])[
            ["temperature_c", "precipitation_mm", "humidity_pct", "event_intensity", "snack_price"]
        ]
        .median()
        .reset_index()
    )
    future = future.merge(hourly_profile, on=["weekday", "hour"], how="left")

    for col in ["temperature_c", "precipitation_mm", "humidity_pct", "event_intensity", "snack_price"]:
        future[col] = future[col].fillna(history[col].median())

    weather_adjust = WEATHER_SCENARIOS[scenario["weather_pattern"]]
    future["temperature_c"] += weather_adjust["temp_offset"] + scenario["temp_manual"]
    future["precipitation_mm"] = np.clip(
        future["precipitation_mm"] * weather_adjust["precip_multiplier"] + scenario["precip_manual"],
        0,
        None,
    )
    future["humidity_pct"] = np.clip(
        future["humidity_pct"] + weather_adjust["humidity_offset"], 30, 100
    )
    future["event_intensity"] = scenario["event_intensity"]

    promo = SNACK_PROMOS[scenario["snack_promo"]]
    future["snack_price"] = np.clip(
        future["snack_price"] * (1 + scenario["snack_price_change"] / 100) * (1 + promo["price_delta"]),
        1.5,
        None,
    )

    future_features = add_time_signals(future)
    future["pred_checkins"] = np.clip(
        checkin_model.predict(future_features[CHECKIN_FEATURES])
        * (1 + scenario["marketing_boost_pct"] / 100),
        0,
        None,
    )

    future_features["checkins"] = future["pred_checkins"]
    future["pred_snack_units"] = np.clip(
        snack_model.predict(future_features[SNACK_FEATURES]) * (1 + promo["demand_boost"]),
        0,
        None,
    )
    future["pred_snack_revenue"] = future["pred_snack_units"] * future["snack_price"]
    return future


def save_procurement_plan(plan: pd.DataFrame, metadata: Optional[Dict[str, str]] = None) -> None:
    """Persist the latest procurement plan to disk so other pages can load it."""
    if plan is None or plan.empty:
        return
    procurement_dir = PROCUREMENT_PLAN_FILE.parent
    procurement_dir.mkdir(parents=True, exist_ok=True)
    export = plan.copy()
    metadata = metadata or {}
    for key, value in metadata.items():
        export[key] = value
    if "date" in export.columns:
        export["date"] = pd.to_datetime(export["date"]).dt.strftime("%Y-%m-%d")
    export.to_csv(PROCUREMENT_PLAN_FILE, index=False)


def load_procurement_plan() -> pd.DataFrame:
    """Load the persisted procurement plan if one exists."""
    if not PROCUREMENT_PLAN_FILE.exists():
        return pd.DataFrame()
    df = pd.read_csv(PROCUREMENT_PLAN_FILE)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df


def append_pos_log(entry: Dict[str, float]) -> None:
    """Append a single POS entry to the runtime log."""
    POS_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([entry])
    header = not POS_LOG_FILE.exists()
    df.to_csv(POS_LOG_FILE, mode="a", header=header, index=False)


def load_pos_log() -> pd.DataFrame:
    """Load the POS runtime log."""
    if not POS_LOG_FILE.exists():
        return pd.DataFrame(columns=["timestamp", "sales_units", "stock_remaining", "checkins_recorded", "notes"])
    df = pd.read_csv(POS_LOG_FILE)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df
