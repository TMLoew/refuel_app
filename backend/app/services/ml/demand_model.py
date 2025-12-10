"""Shared ML helpers: feature engineering, training, and persistence."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline

# Absolute paths keep artifacts in a predictable location.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_DIR = PROJECT_ROOT / "model"
CHECKIN_MODEL_FILE = MODEL_DIR / "checkins_hgb.joblib"
SNACK_MODEL_FILE = MODEL_DIR / "snacks_hgb.joblib"

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


def add_time_signals(dataframe: pd.DataFrame) -> pd.DataFrame:
    # Sine/cosine pairs encode cyclical time features without discontinuities.
    enriched = dataframe.copy()
    enriched["sin_hour"] = np.sin(2 * np.pi * enriched["hour"] / 24)
    enriched["cos_hour"] = np.cos(2 * np.pi * enriched["hour"] / 24)
    enriched["sin_doy"] = np.sin(2 * np.pi * enriched["day_of_year"] / 365)
    enriched["cos_doy"] = np.cos(2 * np.pi * enriched["day_of_year"] / 365)
    return enriched


def _train_checkin_model(feature_df: pd.DataFrame) -> Pipeline:
    # Separate helper makes the attendance pipeline easier to tweak.
    model = Pipeline(
        [
            (
                "model",
                HistGradientBoostingRegressor(
                    max_depth=6,
                    learning_rate=0.12,
                    max_iter=250,
                    l2_regularization=0.1,
                    random_state=42,
                ),
            )
        ]
    )
    model.fit(feature_df[CHECKIN_FEATURES], feature_df["checkins"])
    return model


def _train_snack_model(feature_df: pd.DataFrame) -> Pipeline:
    # Snack pipeline mirrors the check-in version but includes checkins as input.
    model = Pipeline(
        [
            (
                "model",
                HistGradientBoostingRegressor(
                    max_depth=6,
                    learning_rate=0.12,
                    max_iter=300,
                    l2_regularization=0.1,
                    random_state=7,
                ),
            )
        ]
    )
    model.fit(feature_df[SNACK_FEATURES], feature_df["snack_units"])
    return model


def train_models(dataframe: pd.DataFrame) -> Tuple[Pipeline, Pipeline]:
    # Guard against missing columns before fitting either pipeline.
    if dataframe.empty:
        raise ValueError("Cannot train models on an empty dataframe.")
    required_cols = set(SNACK_FEATURES + ["checkins", "snack_units"])
    missing = sorted(required_cols - set(dataframe.columns))
    if missing:
        raise ValueError(f"Missing columns for training: {missing}")
    checkin_model = _train_checkin_model(dataframe)
    snack_model = _train_snack_model(dataframe)
    return checkin_model, snack_model


def ensure_model_dir(model_dir: Path = MODEL_DIR) -> Path:
    # Centralized directory creation for all model artifacts.
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir


def save_models(models: Tuple[Pipeline, Pipeline], model_dir: Path = MODEL_DIR) -> Tuple[Path, Path]:
    # Persist both pipelines so later requests can reuse them.
    ensure_model_dir(model_dir)
    checkin_model, snack_model = models
    dump(checkin_model, model_dir / CHECKIN_MODEL_FILE.name)
    dump(snack_model, model_dir / SNACK_MODEL_FILE.name)
    return model_dir / CHECKIN_MODEL_FILE.name, model_dir / SNACK_MODEL_FILE.name


def load_models(model_dir: Path = MODEL_DIR) -> Tuple[Optional[Pipeline], Optional[Pipeline]]:
    # Return (None, None) whenever either artifact is absent or broken.
    checkin_path = model_dir / CHECKIN_MODEL_FILE.name
    snack_path = model_dir / SNACK_MODEL_FILE.name
    if not checkin_path.exists() or not snack_path.exists():
        return None, None
    try:
        return load(checkin_path), load(snack_path)
    except Exception:
        return None, None


def models_available(model_dir: Path = MODEL_DIR) -> bool:
    # Convenience flag the API can check before attempting to serve predictions.
    checkin_path = model_dir / CHECKIN_MODEL_FILE.name
    snack_path = model_dir / SNACK_MODEL_FILE.name
    return checkin_path.exists() and snack_path.exists()
