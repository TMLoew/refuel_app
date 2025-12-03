# backend/app/services/weather_service.py
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
import pandas as pd
import requests

LAT, LON = 47.4245, 9.3767  # St. Gallen
BASE = "https://api.open-meteo.com/v1/forecast"
DEFAULT_GYM_PATHS = [
    Path("data/gym_checkins_stgallen_2025_patterned.csv"),
    Path("data/gym_badges_0630_2200_long.csv"),
    Path("data/gym_badges.csv"),
]


def resolve_gym_csv_path(csv_path: str | None) -> Path:
    """Return the first available gym CSV, preferring the patterned 2025 file."""
    if csv_path:
        return Path(csv_path)
    for candidate in DEFAULT_GYM_PATHS:
        if candidate.exists():
            return candidate
    return DEFAULT_GYM_PATHS[1]

def fetch_weather_hourly(start_utc: datetime, end_utc: datetime) -> pd.DataFrame:
    """Fetch hourly weather (UTC) for St. Gallen from Open-Meteo and return a DataFrame."""
    params = {
        "latitude": LAT,
        "longitude": LON,
        "hourly": "temperature_2m,precipitation,rain,snowfall,wind_speed_10m,weathercode",
        "timezone": "UTC",
        "start_date": start_utc.strftime("%Y-%m-%d"),
        "end_date": end_utc.strftime("%Y-%m-%d"),
    }
    r = requests.get(BASE, params=params, timeout=30)
    r.raise_for_status()
    hourly = r.json()["hourly"]
    df = pd.DataFrame(hourly).rename(columns={"time": "ts_utc"})
    df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True)
    # keep only requested hours
    df = df[(df["ts_utc"] >= start_utc) & (df["ts_utc"] <= end_utc)]
    # convenience: local time
    df["ts_local"] = df["ts_utc"].dt.tz_convert("Europe/Zurich")
    df["date_local"] = df["ts_local"].dt.date
    df["hour_local"] = df["ts_local"].dt.hour
    return df

def save_latest(hours_back: int = 168) -> str:
    """Grab last N hours (default 7 days) and save CSV under data/."""
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    start = now - timedelta(hours=hours_back)
    df = fetch_weather_hourly(start, now)
    os.makedirs("data", exist_ok=True)
    out = "data/weather_stgallen_hourly.csv"
    df.to_csv(out, index=False)
    return out

def fetch_full_range(start_utc: datetime, end_utc: datetime) -> pd.DataFrame:
    """Chunked fetch across long ranges (Open-Meteo limit ~7 days per call)."""
    frames = []
    cursor = start_utc
    while cursor <= end_utc:
        chunk_end = min(cursor + timedelta(days=6, hours=23), end_utc)
        frames.append(fetch_weather_hourly(cursor, chunk_end))
        cursor = chunk_end + timedelta(hours=1)
    out = pd.concat(frames, ignore_index=True).drop_duplicates(subset="ts_utc")
    return out.sort_values("ts_utc").reset_index(drop=True)


def sync_weather_to_gym_csv(
    gym_csv_path: str | None = None,
    out_path: str = "data/weather_stgallen_hourly.csv",
    pad_days: int = 2,
) -> str:
    """Align weather file to cover entire gym dataset range."""
    p = resolve_gym_csv_path(gym_csv_path)
    if not p.exists():
        raise FileNotFoundError(f"Gym CSV missing: {p}")
    first_cols = pd.read_csv(p, nrows=1).columns
    ts_col = "ts_local" if "ts_local" in first_cols else "ts_local_naive"
    df = pd.read_csv(p, parse_dates=[ts_col])
    series = pd.to_datetime(df[ts_col])
    if series.dt.tz is None:
        series = series.dt.tz_localize("Europe/Zurich")
    else:
        series = series.dt.tz_convert("Europe/Zurich")
    start = (series.min() - pd.Timedelta(days=pad_days)).tz_convert("UTC")
    end = (series.max() + pd.Timedelta(days=pad_days)).tz_convert("UTC")
    weather = fetch_full_range(start, end)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    weather.to_csv(out_path, index=False)
    return out_path


if __name__ == "__main__":
    path = save_latest()
    print(f"saved: {path}")
