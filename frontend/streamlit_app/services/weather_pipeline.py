"""
Utilities for fetching real weather data and combining it with gym and snack signals.

This module mirrors the structure of the reference snippet that already existed in
the project description: it defines an ``Ingredient`` abstraction, a simple
``DataPipeline`` merger for sales/gym data, and helper functions for calling the
Open-Meteo API with safe fallbacks.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import requests

DEFAULT_LAT = 47.4239
DEFAULT_LON = 9.3748
DEFAULT_TZ = "Europe/Zurich"

seasonality: Dict[str, Dict[str, Tuple[float, float]]] = {
    "Banana": {
        "Winter": (0.6, 1.3),
        "Spring": (0.9, 1.1),
        "Summer": (1.0, 1.0),
        "Autumn": (0.8, 1.2),
    }
}


class Ingredient:
    def __init__(self, name: str, base_price_per_unit: float):
        self.name = name
        self.base_price_per_unit = base_price_per_unit

    @property
    def base_price_per_unit(self) -> float:
        return self._base_price_per_unit

    @base_price_per_unit.setter
    def base_price_per_unit(self, value: float) -> None:
        if value < 0:
            raise ValueError("Price must be >= 0")
        self._base_price_per_unit = float(value)

    def seasonal_price(self, season: str, price_index: float) -> float:
        return round(self.base_price_per_unit * price_index, 2)


def month_to_season(month: int) -> str:
    if month in (12, 1, 2):
        return "Winter"
    if month in (3, 4, 5):
        return "Spring"
    if month in (6, 7, 8):
        return "Summer"
    return "Autumn"


@lru_cache(maxsize=32)
def get_daily_temp(date_str: str, lat: float = DEFAULT_LAT, lon: float = DEFAULT_LON) -> float:
    """
    Fetch the maximum daily temperature for a local date.
    """
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        f"&start_date={date_str}&end_date={date_str}"
        "&daily=temperature_2m_max&timezone=Europe/Zurich"
    )
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        return float(resp.json()["daily"]["temperature_2m_max"][0])
    except Exception:
        return 18.0


@lru_cache(maxsize=8)
def _request_hourly_weather(
    start_date: str,
    end_date: str,
    lat: float = DEFAULT_LAT,
    lon: float = DEFAULT_LON,
    timezone: str = DEFAULT_TZ,
) -> Dict:
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        f"&start_date={start_date}&end_date={end_date}"
        "&hourly=temperature_2m,relative_humidity_2m,precipitation"
        f"&timezone={timezone}"
    )
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    return resp.json()


def fetch_hourly_weather_frame(
    timestamps: Sequence[pd.Timestamp],
    lat: float = DEFAULT_LAT,
    lon: float = DEFAULT_LON,
    timezone: str = DEFAULT_TZ,
) -> pd.DataFrame:
    """
    Retrieve hourly temperature, humidity, and precipitation covering the provided timestamps.
    """
    if not timestamps:
        return pd.DataFrame()

    start_date = min(timestamps).strftime("%Y-%m-%d")
    end_date = max(timestamps).strftime("%Y-%m-%d")

    payload = _request_hourly_weather(start_date, end_date, lat=lat, lon=lon, timezone=timezone)
    hourly = payload.get("hourly")
    if not hourly:
        raise ValueError("No hourly weather data returned by Open-Meteo")

    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(hourly["time"]),
            "temperature_c": hourly.get("temperature_2m"),
            "humidity_pct": hourly.get("relative_humidity_2m"),
            "precipitation_mm": hourly.get("precipitation"),
        }
    )
    return frame


class DataPipeline:
    """
    Merge sales rows with gym rows while injecting temperature and seasonality metadata.
    Mirrors the illustrative snippet shared in the project brief.
    """

    def __init__(self, ingredient: Ingredient, sales_rows: List[dict], gym_rows: List[dict]):
        self.ingredient = ingredient
        self.sales_rows = sales_rows
        self.gym_rows = gym_rows
        self._merged: List[dict] = []

    @staticmethod
    def _parse_month(date_str: str) -> int:
        return datetime.strptime(date_str, "%Y-%m-%d").month

    @property
    def merged(self) -> List[dict]:
        return self._merged

    def _season_for_date(self, date_str: str) -> str:
        return month_to_season(self._parse_month(date_str))

    def _seasonal_indices(self, season: str) -> Tuple[float, float]:
        return seasonality.get(self.ingredient.name, {}).get(season, (1.0, 1.0))

    def run(self) -> None:
        gym_by_date = {row["date"]: row["visitors"] for row in self.gym_rows}
        merged: List[dict] = []
        for row in self.sales_rows:
            d = row["date"]
            temp = get_daily_temp(d)
            visitors = gym_by_date.get(d, 0)
            season = self._season_for_date(d)
            availability, price_index = self._seasonal_indices(season)
            merged.append(
                {
                    "date": d,
                    "product": row["product"],
                    "qty": row["qty"],
                    "visitors": visitors,
                    "temp": temp,
                    "season": season,
                    "availability": availability,
                    "price_index": price_index,
                    "seasonal_price": self.ingredient.seasonal_price(season, price_index),
                    "dow": datetime.strptime(d, "%Y-%m-%d").weekday(),
                }
            )
        self._merged = merged


def summary(merged_rows: Iterable[dict]) -> str:
    rows = list(merged_rows)
    if not rows:
        return "No rows to summarize."

    n = len(rows)
    total_qty = sum(r["qty"] for r in rows)
    avg_temp = round(sum(r["temp"] for r in rows) / max(n, 1), 1)
    avg_vis = round(sum(r["visitors"] for r in rows) / max(n, 1), 1)
    seasons = {r["season"] for r in rows}
    by_season = {s: sum(r["qty"] for r in rows if r["season"] == s) for s in seasons}
    lines = [
        f"Records: {n}",
        f"Total sold: {total_qty}",
        f"Avg temp: {avg_temp}°C",
        f"Avg visitors: {avg_vis}",
        "Qty by season: " + ", ".join(f"{s}={by_season[s]}" for s in sorted(by_season)),
    ]
    return "\n".join(lines)


def build_synthetic_weather_frame(timestamps: Sequence[pd.Timestamp]) -> pd.DataFrame:
    """
    Deterministic synthetic fallback when the API cannot be reached.
    """
    if not timestamps:
        return pd.DataFrame()

    ts_series = pd.Series(pd.to_datetime(list(timestamps))).sort_values().reset_index(drop=True)
    day_of_year = ts_series.dt.dayofyear
    hour = ts_series.dt.hour
    rng = np.random.default_rng(42)
    seasonal_temp = 16 + 6 * np.sin(2 * np.pi * day_of_year / 365) + 4 * np.sin(2 * np.pi * hour / 24)
    temperature_c = seasonal_temp + rng.normal(0, 1.8, len(ts_series))
    precipitation_mm = np.clip(
        1.5 + 1.2 * np.cos(2 * np.pi * day_of_year / 365) + rng.normal(0, 0.6, len(ts_series)),
        0,
        None,
    )
    humidity_pct = np.clip(
        58 + 12 * np.sin(2 * np.pi * hour / 24) + rng.normal(0, 4, len(ts_series)),
        35,
        95,
    )
    return pd.DataFrame(
        {
            "timestamp": ts_series,
            "temperature_c": temperature_c,
            "precipitation_mm": precipitation_mm,
            "humidity_pct": humidity_pct,
        }
    )
