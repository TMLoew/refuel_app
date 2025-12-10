"""Shared data utilities for Streamlit pages."""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import sys

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.pipeline import Pipeline

try:
    from backend.app.services.ml.demand_model import (
        CHECKIN_FEATURES,
        SNACK_FEATURES,
        add_time_signals,
        load_models as load_persisted_models,
        save_models as persist_models,
        train_models as core_train_models,
    )
except ModuleNotFoundError:
    _root = Path(__file__).resolve().parents[3]
    if str(_root) not in sys.path:
        sys.path.append(str(_root))
    from backend.app.services.ml.demand_model import (  # type: ignore
        CHECKIN_FEATURES,
        SNACK_FEATURES,
        add_time_signals,
        load_models as load_persisted_models,
        save_models as persist_models,
        train_models as core_train_models,
    )

from .weather_pipeline import (
    Ingredient,
    build_synthetic_weather_frame,
    fetch_future_weather_forecast,
    fetch_hourly_weather_frame,
    month_to_season,
    seasonality,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
# Choose the first gym dataset that exists locally so widgets stay populated.
PREFERRED_DATASETS = [
    PROJECT_ROOT / "data" / "gym_badges_0630_2200_long.csv",
    PROJECT_ROOT / "data" / "gym_badges.csv",
]
DATA_FILE = next((path for path in PREFERRED_DATASETS if path.exists()), PREFERRED_DATASETS[-1])
PROCUREMENT_PLAN_FILE = PROJECT_ROOT / "data" / "procurement_plan.csv"
POS_LOG_FILE = PROJECT_ROOT / "data" / "pos_runtime_log.csv"
PRODUCT_MIX_FILE = PROJECT_ROOT / "data" / "product_mix_daily.csv"
PRODUCT_MIX_SNAPSHOT_FILE = PROJECT_ROOT / "data" / "product_mix_enriched.csv"
PRODUCT_PRICE_FILE = PROJECT_ROOT / "data" / "product_prices.csv"
WEATHER_PROFILE_FILE = PROJECT_ROOT / "data" / "weather_profile.json"
RESTOCK_POLICY_FILE = PROJECT_ROOT / "data" / "restock_policy.json"
WEATHER_CACHE_FILE = PROJECT_ROOT / "data" / "weather_cache.csv"
DEFAULT_PRODUCT_PRICE = 3.5
DEFAULT_RESTOCK_POLICY: Dict[str, Any] = {
    "auto_enabled": False,
    "threshold_units": 40,
    "lot_size": 50,
    "cooldown_hours": 6,
    "last_auto_restock": None,
}
DEFAULT_WEATHER_PROFILE = {
    "lat": 47.4239,
    "lon": 9.3748,
    "api_timeout": 10,
    "cache_hours": 6,
}

WEATHER_SCENARIOS: Dict[str, Dict[str, float]] = {
    "Temperate & sunny": {"temp_offset": 2.0, "precip_multiplier": 0.7, "humidity_offset": -3},
    "Cold snap": {"temp_offset": -6.0, "precip_multiplier": 1.0, "humidity_offset": 4},
    "Humid heatwave": {"temp_offset": 5.5, "precip_multiplier": 0.4, "humidity_offset": 8},
    "Storm front": {"temp_offset": -1.0, "precip_multiplier": 1.8, "humidity_offset": 10},
}


def _safe_precip_multiplier(numerator: float, denominator: float, floor: float = 0.05) -> float:
    # Guard against divide-by-zero while enforcing a reasonable floor.
    denominator = max(denominator, 0.05)
    multiplier = numerator / denominator if denominator else 1.0
    return max(floor, multiplier)


def _normalize_to_utc_naive(ts: Any) -> pd.Timestamp:
    # Convert anything the UI hands us into naive UTC timestamps.
    stamp = pd.to_datetime(ts)
    if stamp.tzinfo is not None:
        stamp = stamp.tz_convert("UTC").tz_localize(None)
    return stamp


def _load_cached_weather_window(start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if not WEATHER_CACHE_FILE.exists():
        return pd.DataFrame(), {}
    try:
        cache_df = pd.read_csv(WEATHER_CACHE_FILE, parse_dates=["timestamp"])
    except Exception:
        return pd.DataFrame(), {}
    if cache_df.empty or "timestamp" not in cache_df.columns:
        return pd.DataFrame(), {}
    timestamps = pd.to_datetime(cache_df["timestamp"])
    if getattr(timestamps.dt, "tz", None) is not None:
        timestamps = timestamps.dt.tz_convert("UTC").dt.tz_localize(None)
    cache_df["timestamp"] = timestamps
    # Slice cache to just the requested window; ignore older entries.
    mask = (cache_df["timestamp"] >= start_ts) & (cache_df["timestamp"] <= end_ts)
    subset = cache_df.loc[mask].copy()
    if subset.empty:
        return pd.DataFrame(), {}
    coverage_start = subset["timestamp"].min()
    coverage_end = subset["timestamp"].max()
    end_aware = coverage_end.tz_localize("UTC") if coverage_end.tzinfo is None else coverage_end.tz_convert("UTC")
    age_minutes = max(0.0, (pd.Timestamp.utcnow() - end_aware).total_seconds() / 60)
    meta = {
        "latency_ms": None,
        "chunks": 0,
        "coverage_start": coverage_start.isoformat(),
        "coverage_end": coverage_end.isoformat(),
        "cache_age_minutes": age_minutes,
    }
    return subset, meta


def _save_weather_cache(frame: pd.DataFrame) -> None:
    if frame.empty or "timestamp" not in frame.columns:
        return
    WEATHER_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    new_frame = frame.copy()
    new_frame["timestamp"] = pd.to_datetime(new_frame["timestamp"])
    if getattr(new_frame["timestamp"].dt, "tz", None) is not None:
        new_frame["timestamp"] = new_frame["timestamp"].dt.tz_convert("UTC").dt.tz_localize(None)
    if WEATHER_CACHE_FILE.exists():
        try:
            existing = pd.read_csv(WEATHER_CACHE_FILE, parse_dates=["timestamp"])
            existing["timestamp"] = pd.to_datetime(existing["timestamp"])
        except Exception:
            existing = pd.DataFrame(columns=new_frame.columns)
        combined = (
            pd.concat([existing, new_frame], ignore_index=True, sort=False)
            .drop_duplicates(subset="timestamp")
            .sort_values("timestamp")
        )
        # Drop stale weather beyond ~60 days to keep the cache light.
        cutoff = combined["timestamp"].max() - pd.Timedelta(days=60)
        combined = combined[combined["timestamp"] >= cutoff]
    else:
        combined = new_frame
    combined.to_csv(WEATHER_CACHE_FILE, index=False)


@st.cache_data(show_spinner=False)
def derive_weather_archetypes(history: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Build data-driven archetypes (offsets/multipliers) from historical telemetry.
    Falls back to the static scenarios if history is missing.
    """
    if history.empty:
        return {}
    required = {"temperature_c", "precipitation_mm", "humidity_pct"}
    if not required.issubset(history.columns):
        return {}

    # Work off medians/quantiles so outliers don't skew the archetypes.
    temp = history["temperature_c"].dropna()
    precip = history["precipitation_mm"].dropna()
    humidity = history["humidity_pct"].dropna()
    if temp.empty or precip.empty or humidity.empty:
        return {}

    temp_med = float(temp.median())
    temp_low = float(temp.quantile(0.25))
    temp_high = float(temp.quantile(0.75))

    precip_med = float(precip.median()) + 0.05
    precip_low = float(precip.quantile(0.25)) + 0.02
    precip_high = float(precip.quantile(0.75)) + 0.1

    humidity_med = float(humidity.median())
    humidity_low = float(humidity.quantile(0.25))
    humidity_high = float(humidity.quantile(0.75))

    archetypes = {
        "Historical cool & wet": {
            "temp_offset": temp_low - temp_med,
            "precip_multiplier": _safe_precip_multiplier(precip_high, precip_med, floor=0.2),
            "humidity_offset": humidity_high - humidity_med,
        },
        "Historical hot & dry": {
            "temp_offset": temp_high - temp_med,
            "precip_multiplier": _safe_precip_multiplier(precip_low, precip_med, floor=0.05),
            "humidity_offset": humidity_low - humidity_med,
        },
        "Historical storm surge": {
            "temp_offset": 0.5 * (temp_low - temp_med),
            "precip_multiplier": _safe_precip_multiplier(precip_high + 0.15, precip_med, floor=0.4),
            "humidity_offset": humidity_high - humidity_med + 2,
        },
        "Historical crisp morning": {
            "temp_offset": temp_low - temp_med - 1.2,
            "precip_multiplier": _safe_precip_multiplier(precip_low, precip_med, floor=0.05),
            "humidity_offset": humidity_low - humidity_med,
        },
    }
    return archetypes


def combined_weather_scenarios(history: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Merge static scenarios with any data-driven archetypes derived from history."""
    scenarios = dict(WEATHER_SCENARIOS)
    scenarios.update(derive_weather_archetypes(history))
    return scenarios


def sample_weather_archetype(history: pd.DataFrame, random_state: Optional[int] = None) -> Tuple[str, Dict[str, float]]:
    """Pick a random scenario (static or historical) for automated runs."""
    scenarios = combined_weather_scenarios(history)
    if not scenarios:
        return "Temperate & sunny", WEATHER_SCENARIOS["Temperate & sunny"]
    keys = list(scenarios.keys())
    rng = np.random.default_rng(random_state)
    choice = rng.choice(keys)
    return choice, scenarios[choice]


def build_enriched_history(
    csv_path: Path = DATA_FILE,
    use_weather_api: bool = True,
) -> pd.DataFrame:
    """Pure data loader so backend scripts can reuse the enrichment logic."""
    if not csv_path.exists():
        return pd.DataFrame()

    df = pd.read_csv(csv_path)
    if df.empty:
        return df

    # Support older CSV exports by checking multiple timestamp column names.
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
    weather_meta_extra: Dict[str, Any] = {}
    cache_start = _normalize_to_utc_naive(df["timestamp"].min()).floor("h")
    cache_end = _normalize_to_utc_naive(df["timestamp"].max()).ceil("h")

    if use_weather_api:
        try:
            weather_frame, weather_meta_extra = fetch_hourly_weather_frame(timestamps)
            if not weather_frame.empty:
                weather_source = "open-meteo"
                _save_weather_cache(weather_frame)
        except Exception:
            weather_frame = pd.DataFrame()

        if weather_frame.empty:
            cached_frame, cache_meta = _load_cached_weather_window(cache_start, cache_end)
            if not cached_frame.empty:
                weather_frame = cached_frame
                weather_meta_extra = cache_meta
                weather_source = "cached"

    if weather_frame.empty:
        # Final fallback keeps the pipeline running during outages.
        weather_frame = build_synthetic_weather_frame(timestamps)

    df = df.merge(weather_frame, on="timestamp", how="left")
    weather_cols = ["temperature_c", "precipitation_mm", "humidity_pct"]
    # Fill gaps so downstream modeling never sees NaNs.
    df[weather_cols] = df[weather_cols].ffill().bfill()
    df.attrs["weather_source"] = weather_source

    # Rudimentary event intensity heuristic to mimic peak/quiet periods.
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
    # Inject a little price noise per hour to avoid perfectly flat signals.
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
    # Attach metadata so the UI can describe the provenance of weather data.
    df.attrs["weather_meta"] = {
        "source": weather_source,
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="minutes"),
        "coverage_start": weather_meta_extra.get("coverage_start", df["timestamp"].min().isoformat()),
        "coverage_end": weather_meta_extra.get("coverage_end", df["timestamp"].max().isoformat()),
        "latency_ms": weather_meta_extra.get("latency_ms"),
        "chunks": weather_meta_extra.get("chunks"),
        "cache_age_minutes": weather_meta_extra.get("cache_age_minutes"),
    }
    return df


@st.cache_data(show_spinner=False)
def load_enriched_data(
    csv_path: Path = DATA_FILE,
    use_weather_api: bool = True,
    cache_buster: float = 0.0,
) -> pd.DataFrame:
    """Cached wrapper around build_enriched_history for Streamlit sessions."""
    _ = cache_buster  # ensures cache invalidation when requested
    return build_enriched_history(csv_path=csv_path, use_weather_api=use_weather_api)


@st.cache_resource(show_spinner=False)
def train_models(df: pd.DataFrame) -> Tuple[Optional[object], Optional[object]]:
    """Load cached models or train + persist a new pair."""
    if df.empty:
        return None, None
    persisted = load_persisted_models()
    if all(persisted):
        return persisted
    feature_df = add_time_signals(df)
    required_cols = set(CHECKIN_FEATURES)
    missing = sorted(required_cols - set(feature_df.columns))
    if missing:
        raise ValueError(f"Missing engineered features: {missing}")
    models = core_train_models(feature_df)
    persist_models(models)
    return models


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

    # Scenario defines how many future hours to generate.
    horizon = scenario["horizon_hours"]
    use_live_future = bool(scenario.get("use_live_weather", False))
    anchor_ts = history["timestamp"].max()
    if anchor_timestamp is not None:
        anchor_ts = pd.to_datetime(anchor_timestamp)
    future_index = pd.date_range(anchor_ts + pd.Timedelta(hours=1), periods=horizon, freq="h")
    future = pd.DataFrame({"timestamp": future_index})
    future["hour"] = future["timestamp"].dt.hour
    future["weekday"] = future["timestamp"].dt.weekday
    future["is_weekend"] = future["weekday"].isin([5, 6]).astype(int)
    future["day_of_year"] = future["timestamp"].dt.dayofyear

    # Median hourly profile offers a baseline before weather perturbations.
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

    if use_live_future:
        live_weather = pd.DataFrame()
        try:
            live_weather = fetch_future_weather_forecast(
                future_index[0],
                horizon,
            )
        except Exception:
            live_weather = pd.DataFrame()
        if not live_weather.empty:
            future = future.merge(live_weather, on="timestamp", how="left", suffixes=("", "_live"))
            for col in ["temperature_c", "precipitation_mm", "humidity_pct"]:
                live_col = f"{col}_live"
                if live_col in future.columns:
                    future[col] = future[live_col].combine_first(future[col])
                    future.drop(columns=[live_col], inplace=True, errors="ignore")

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

    future["snack_price"] = np.clip(
        future["snack_price"] * (1 + scenario["snack_price_change"] / 100),
        1.5,
        None,
    )

    future_features = add_time_signals(future)
    # Apply marketing boost after predicting attendance and keep values >=0.
    future["pred_checkins"] = np.clip(
        checkin_model.predict(future_features[CHECKIN_FEATURES])
        * (1 + scenario["marketing_boost_pct"] / 100),
        0,
        None,
    )

    future_features["checkins"] = future["pred_checkins"]
    # Feed predicted checkins into the snack model for consistency.
    future["pred_snack_units"] = np.clip(
        snack_model.predict(future_features[SNACK_FEATURES]),
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


def _default_price_frame() -> pd.DataFrame:
    # Derive the product universe from the mix file whenever possible.
    mix_df = load_product_mix_data()
    products = sorted(mix_df["product"].dropna().unique().tolist()) if not mix_df.empty else []
    if not products:
        products = ["Protein Shake", "Electrolyte Drink", "Iced Matcha", "Recovery Smoothie", "Isotonic Lemon"]
    return pd.DataFrame({"product": products, "unit_price": [DEFAULT_PRODUCT_PRICE for _ in products]})


def load_product_prices() -> pd.DataFrame:
    """Load per-product prices (fallback to defaults if file missing)."""
    if PRODUCT_PRICE_FILE.exists():
        df = pd.read_csv(PRODUCT_PRICE_FILE)
    else:
        df = pd.DataFrame()
    # Rebuild from defaults whenever the CSV is missing or malformed.
    if df.empty or "product" not in df.columns or "unit_price" not in df.columns:
        df = _default_price_frame()
        if not df.empty:
            PRODUCT_PRICE_FILE.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(PRODUCT_PRICE_FILE, index=False)
    df["unit_price"] = pd.to_numeric(df["unit_price"], errors="coerce")
    df = df.dropna(subset=["product"]).copy()
    return df


def save_product_prices(prices: pd.DataFrame) -> None:
    """Persist product pricing overrides."""
    if prices.empty or "product" not in prices.columns:
        return
    export = prices.copy()
    export = export[["product", "unit_price"]].copy()
    # Coerce inputs to floats and fall back to default pricing when blank.
    export["unit_price"] = pd.to_numeric(export["unit_price"], errors="coerce").fillna(DEFAULT_PRODUCT_PRICE)
    PRODUCT_PRICE_FILE.parent.mkdir(parents=True, exist_ok=True)
    export.to_csv(PRODUCT_PRICE_FILE, index=False)


def get_product_price_map() -> Dict[str, float]:
    df = load_product_prices()
    return {row["product"]: float(row["unit_price"]) for _, row in df.iterrows()}


def add_or_update_product_price(product: str, unit_price: float = DEFAULT_PRODUCT_PRICE) -> None:
    """Add a new product to the price sheet, or update if it already exists."""
    product = product.strip()
    if not product:
        return
    prices = load_product_prices()
    if prices.empty:
        updated = pd.DataFrame({"product": [product], "unit_price": [unit_price]})
    else:
        prices = prices.copy()
        if product in prices["product"].values:
            prices.loc[prices["product"] == product, "unit_price"] = unit_price
            updated = prices
        else:
            # Append new products so the POS can reference them immediately.
            new_row = pd.DataFrame({"product": [product], "unit_price": [unit_price]})
            updated = pd.concat([prices, new_row], ignore_index=True)
    save_product_prices(updated)


def remove_product_price(product: str) -> None:
    """Remove a product from the price sheet."""
    prices = load_product_prices()
    if prices.empty or product not in prices["product"].values:
        return
    updated = prices[prices["product"] != product].reset_index(drop=True)
    save_product_prices(updated)


def load_weather_profile() -> Dict[str, Any]:
    if WEATHER_PROFILE_FILE.exists():
        try:
            data = json.loads(WEATHER_PROFILE_FILE.read_text())
            return {**DEFAULT_WEATHER_PROFILE, **data}
        except Exception:
            return DEFAULT_WEATHER_PROFILE.copy()
    # First run: create the profile file with baked-in defaults.
    WEATHER_PROFILE_FILE.parent.mkdir(parents=True, exist_ok=True)
    WEATHER_PROFILE_FILE.write_text(json.dumps(DEFAULT_WEATHER_PROFILE, indent=2))
    return DEFAULT_WEATHER_PROFILE.copy()


def save_weather_profile(profile: Dict[str, Any]) -> None:
    WEATHER_PROFILE_FILE.parent.mkdir(parents=True, exist_ok=True)
    payload = {**DEFAULT_WEATHER_PROFILE, **profile}
    WEATHER_PROFILE_FILE.write_text(json.dumps(payload, indent=2))


def get_weather_coordinates() -> Tuple[float, float]:
    profile = load_weather_profile()
    return float(profile.get("lat", DEFAULT_WEATHER_PROFILE["lat"])), float(
        profile.get("lon", DEFAULT_WEATHER_PROFILE["lon"])
    )


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
    if "weight" not in mix.columns:
        return pd.DataFrame()

    grouped = {date: group for date, group in mix.groupby("date")}
    fallback_date = mix["date"].max()
    if fallback_date is pd.NaT:
        return pd.DataFrame()

    records = []
    for _, day_row in daily_forecast.iterrows():
        date = pd.to_datetime(day_row["date"]).normalize()
        forecast_units = day_row.get("pred_snack_units")
        if pd.isna(forecast_units):
            continue
        day_mix = grouped.get(date, grouped.get(fallback_date))
        if day_mix is None or day_mix.empty:
            continue
        for _, product_row in day_mix.iterrows():
            weight = product_row.get("weight", 0.0)
            if pd.isna(weight):
                continue
            entry = {
                "date": date,
                "product": product_row.get("product"),
                "weight": weight,
                "suggested_qty": product_row.get("suggested_qty"),
                "forecast_units": forecast_units * weight,
            }
            records.append(entry)
    return pd.DataFrame(records)


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
        # Sanitize the nested POS breakdown before writing to CSV.
        normalized = {str(k): int(v) for k, v in breakdown.items() if v and int(v) > 0}
        record["product_breakdown"] = json.dumps(normalized)
        if normalized:
            record["sales_units"] = int(sum(normalized.values()))
    elif breakdown in (None, "", []):
        record["product_breakdown"] = ""
    record["sales_units"] = int(record.get("sales_units", 0) or 0)
    df = pd.DataFrame([record])
    header = not POS_LOG_FILE.exists()
    df.to_csv(POS_LOG_FILE, mode="a", header=header, index=False)


def load_pos_log() -> pd.DataFrame:
    """Load the POS runtime log."""
    if not POS_LOG_FILE.exists():
        return pd.DataFrame(columns=["timestamp", "sales_units", "stock_remaining", "checkins_recorded", "notes"])
    df = pd.read_csv(POS_LOG_FILE)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df["timestamp"] = df["timestamp"].where(df["timestamp"].notna(), pd.NaT)
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
    # Respect the cooldown window before allowing another auto-restock.
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
    # Accept dicts/JSON strings and normalize everything else to {}.
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
