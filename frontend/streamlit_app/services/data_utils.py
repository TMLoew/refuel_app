"""Shared data utilities for Streamlit pages."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

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
DATA_FILE = PROJECT_ROOT / "data" / "gym_badges.csv"

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
def load_enriched_data(csv_path: Path = DATA_FILE, use_weather_api: bool = False) -> pd.DataFrame:
    """Load gym dataset and enrich it with weather + snack context."""
    if not csv_path.exists():
        return pd.DataFrame()

    df = pd.read_csv(csv_path, parse_dates=["ts_local_naive"])
    if df.empty:
        return df

    df = df.rename(columns={"ts_local_naive": "timestamp"})
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["hour"] = df["timestamp"].dt.hour
    df["weekday"] = df["timestamp"].dt.weekday
    df["is_weekend"] = df["weekday"].isin([5, 6]).astype(int)
    df["day_of_year"] = df["timestamp"].dt.dayofyear

    rng = np.random.default_rng(42)
    timestamps = tuple(df["timestamp"])
    weather_source = "synthetic"
    weather_frame = pd.DataFrame()

    if use_weather_api:
        try:
            weather_frame = fetch_hourly_weather_frame(timestamps)
            if not weather_frame.empty:
                weather_source = "open-meteo"
        except Exception:
            weather_frame = pd.DataFrame()

    if weather_frame.empty:
        weather_frame = build_synthetic_weather_frame(timestamps)

    df = df.merge(weather_frame, on="timestamp", how="left")
    weather_cols = ["temperature_c", "precipitation_mm", "humidity_pct"]
    df[weather_cols] = df[weather_cols].fillna(method="ffill").fillna(method="bfill")
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
) -> pd.DataFrame:
    """Create a future dataframe incorporating scenario adjustments."""
    checkin_model, snack_model = models
    if checkin_model is None or snack_model is None:
        return pd.DataFrame()

    horizon = scenario["horizon_hours"]
    future_index = pd.date_range(
        history["timestamp"].max() + pd.Timedelta(hours=1), periods=horizon, freq="H"
    )
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
