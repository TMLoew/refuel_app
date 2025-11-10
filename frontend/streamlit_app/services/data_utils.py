"""Shared data utilities for Streamlit pages."""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
PRODUCT_MIX_FILE = PROJECT_ROOT / "data" / "product_mix_daily.csv"
PRODUCT_MIX_SNAPSHOT_FILE = PROJECT_ROOT / "data" / "product_mix_enriched.csv"
RESTOCK_POLICY_FILE = PROJECT_ROOT / "data" / "restock_policy.json"
DEFAULT_RESTOCK_POLICY: Dict[str, Any] = {
    "auto_enabled": False,
    "threshold_units": 40,
    "lot_size": 50,
    "cooldown_hours": 6,
    "last_auto_restock": None,
}

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


@st.cache_data(show_spinner=False)
def load_product_mix_data(csv_path: Path = PRODUCT_MIX_FILE) -> pd.DataFrame:
    """Load product mix recommendations (daily product shares & suggested qty)."""
    if not csv_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    if df.empty:
        return df
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df


def get_product_catalog(product_mix: pd.DataFrame) -> List[str]:
    """Return a sorted list of unique product names from the mix file."""
    if product_mix.empty or "product" not in product_mix.columns:
        return []
    return sorted(product_mix["product"].dropna().unique().tolist())


def build_daily_product_mix_view(telemetry: pd.DataFrame, product_mix: pd.DataFrame) -> pd.DataFrame:
    """Merge daily telemetry aggregates with mix recommendations."""
    if telemetry.empty or product_mix.empty:
        return pd.DataFrame()
    if "timestamp" not in telemetry.columns:
        raise ValueError("Telemetry frame missing 'timestamp' column required for daily aggregation.")

    telemetry = telemetry.copy()
    telemetry["timestamp"] = pd.to_datetime(telemetry["timestamp"])
    telemetry["date"] = telemetry["timestamp"].dt.normalize()
    daily_actuals = (
        telemetry.groupby("date")
        .agg(
            actual_checkins=("checkins", "sum"),
            actual_snack_units=("snack_units", "sum"),
            actual_snack_revenue=("snack_revenue", "sum"),
            avg_temp_c=("temperature_c", "mean"),
            avg_precip_mm=("precipitation_mm", "mean"),
        )
        .reset_index()
    )
    merged = product_mix.merge(daily_actuals, on="date", how="left")
    if "weight" in merged.columns and "actual_snack_units" in merged.columns:
        merged["implied_units"] = merged["actual_snack_units"] * merged["weight"]
        merged["unit_gap"] = merged["suggested_qty"] - merged["implied_units"]
    return merged


def compute_daily_actuals(telemetry: pd.DataFrame) -> pd.DataFrame:
    """Aggregate telemetry to daily totals for check-ins and snack demand."""
    if telemetry.empty or "timestamp" not in telemetry.columns:
        return pd.DataFrame()
    daily = (
        telemetry.assign(date=pd.to_datetime(telemetry["timestamp"]).dt.normalize())
        .groupby("date")
        .agg(
            actual_checkins=("checkins", "sum"),
            actual_snack_units=("snack_units", "sum"),
            actual_snack_revenue=("snack_revenue", "sum"),
            avg_temp_c=("temperature_c", "mean"),
            avg_precip_mm=("precipitation_mm", "mean"),
        )
        .reset_index()
    )
    return daily


def build_daily_forecast(forecast: pd.DataFrame) -> pd.DataFrame:
    """Aggregate hourly scenario forecast to daily totals."""
    if forecast.empty or "timestamp" not in forecast.columns:
        return pd.DataFrame()
    daily = (
        forecast.assign(date=pd.to_datetime(forecast["timestamp"]).dt.normalize())
        .groupby("date")
        .agg(
            pred_checkins=("pred_checkins", "sum"),
            pred_snack_units=("pred_snack_units", "sum"),
            pred_snack_revenue=("pred_snack_revenue", "sum"),
            avg_temp_c=("temperature_c", "mean"),
            avg_precip_mm=("precipitation_mm", "mean"),
        )
        .reset_index()
    )
    return daily


def allocate_product_level_forecast(
    daily_forecast: pd.DataFrame, product_mix: pd.DataFrame
) -> pd.DataFrame:
    """Distribute daily snack forecasts across products using mix weights."""
    if daily_forecast.empty or product_mix.empty:
        return pd.DataFrame()
    mix = product_mix.copy()
    mix["date"] = pd.to_datetime(mix["date"]).dt.normalize()
    merged = mix.merge(daily_forecast[["date", "pred_snack_units"]], on="date", how="inner")
    if merged.empty or "weight" not in merged.columns:
        return pd.DataFrame()
    merged["forecast_units"] = merged["pred_snack_units"] * merged["weight"]
    return merged


def save_product_mix_snapshot(snapshot: pd.DataFrame, metadata: Optional[Dict[str, str]] = None) -> None:
    """Persist the merged product mix view for downstream pages."""
    if snapshot.empty:
        return
    PRODUCT_MIX_SNAPSHOT_FILE.parent.mkdir(parents=True, exist_ok=True)
    export = snapshot.copy()
    metadata = metadata or {}
    for key, value in metadata.items():
        export[key] = value
    if "date" in export.columns:
        export["date"] = pd.to_datetime(export["date"]).dt.strftime("%Y-%m-%d")
    export.to_csv(PRODUCT_MIX_SNAPSHOT_FILE, index=False)


def load_product_mix_snapshot() -> pd.DataFrame:
    """Load the persisted merged product mix snapshot."""
    if not PRODUCT_MIX_SNAPSHOT_FILE.exists():
        return pd.DataFrame()
    df = pd.read_csv(PRODUCT_MIX_SNAPSHOT_FILE)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df


def append_pos_log(entry: Dict[str, Any]) -> None:
    """Append a single POS entry to the runtime log."""
    POS_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    record = entry.copy()
    breakdown = record.get("product_breakdown")
    if isinstance(breakdown, dict):
        record["product_breakdown"] = json.dumps(breakdown)
    elif breakdown in (None, "", []):
        record["product_breakdown"] = ""
    df = pd.DataFrame([record])
    header = not POS_LOG_FILE.exists()
    df.to_csv(POS_LOG_FILE, mode="a", header=header, index=False)


def load_pos_log() -> pd.DataFrame:
    """Load the POS runtime log."""
    if not POS_LOG_FILE.exists():
        return pd.DataFrame(columns=["timestamp", "sales_units", "stock_remaining", "checkins_recorded", "notes"])
    df = pd.read_csv(POS_LOG_FILE)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    if "product_breakdown" not in df.columns:
        df["product_breakdown"] = [{} for _ in range(len(df))]
    else:
        df["product_breakdown"] = df["product_breakdown"].apply(_safe_load_breakdown)
    return df


def load_restock_policy() -> Dict[str, Any]:
    """Load persisted auto-restock preferences."""
    policy = DEFAULT_RESTOCK_POLICY.copy()
    if RESTOCK_POLICY_FILE.exists():
        try:
            stored = json.loads(RESTOCK_POLICY_FILE.read_text())
            if isinstance(stored, dict):
                policy.update(stored)
        except Exception:
            pass
    return policy


def save_restock_policy(policy: Dict[str, Any]) -> None:
    """Persist auto-restock preferences."""
    RESTOCK_POLICY_FILE.parent.mkdir(parents=True, exist_ok=True)
    export = DEFAULT_RESTOCK_POLICY.copy()
    export.update(policy)
    RESTOCK_POLICY_FILE.write_text(json.dumps(export, indent=2))


def should_auto_restock(current_stock: float, policy: Dict[str, Any]) -> bool:
    """Determine whether an automatic restock should be triggered."""
    if current_stock is None:
        return False
    if not policy.get("auto_enabled", False):
        return False
    threshold = policy.get("threshold_units", DEFAULT_RESTOCK_POLICY["threshold_units"])
    if current_stock >= threshold:
        return False
    cooldown_hours = policy.get("cooldown_hours", DEFAULT_RESTOCK_POLICY["cooldown_hours"])
    last_event = policy.get("last_auto_restock")
    if last_event:
        try:
            last_dt = datetime.fromisoformat(str(last_event))
            if last_dt.tzinfo is None:
                last_dt = last_dt.replace(tzinfo=timezone.utc)
        except ValueError:
            last_dt = None
        if last_dt:
            elapsed = datetime.now(timezone.utc) - last_dt
            if elapsed < timedelta(hours=cooldown_hours):
                return False
    return True


def mark_auto_restock(policy: Dict[str, Any]) -> Dict[str, Any]:
    """Update restock policy metadata after an automatic restock action."""
    updated = policy.copy()
    updated["last_auto_restock"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    save_restock_policy(updated)
    return updated


def _safe_load_breakdown(value: Any) -> Dict[str, float]:
    if value is None:
        return {}
    if isinstance(value, float) and math.isnan(value):
        return {}
    if value == "":
        return {}
    if isinstance(value, dict):
        return value
    try:
        parsed = json.loads(value)
        if isinstance(parsed, dict):
            return {str(k): float(v) for k, v in parsed.items()}
    except Exception:
        pass
    return {}
