# backend/app/services/weather_service.py
import os
from datetime import datetime, timedelta, timezone
import pandas as pd
import requests

LAT, LON = 47.4245, 9.3767  # St. Gallen
BASE = "https://api.open-meteo.com/v1/forecast"

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

if __name__ == "__main__":
    path = save_latest()
    print(f"saved: {path}")
