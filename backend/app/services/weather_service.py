# backend/app/services/weather_services.py
import os
from datetime import datetime, timedelta, timezone
import pandas as pd
import requests

LAT, LON = 47.4245, 9.3767  # St. Gallen
BASE = "https://api.open-meteo.com/v1/forecast"

def fetch_weather_hourly(start_utc: datetime, end_utc: datetime) -> pd.DataFrame:
    """
    Fetch hourly weather for St. Gallen from Open-Meteo between start_utc and end_utc (UTC).
    Returns a DataFrame with ts_utc + weather columns + local time columns.
    """
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
    hourly = r.json().get("hourly", {})
    if "time" not in hourly:
        raise RuntimeError("Open-Meteo response did not contain hourly data")
    df = pd.DataFrame(hourly).rename(columns={"time": "ts_utc"})
    df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True)
    # keep only requested hours
    df = df[(df["ts_utc"] >= start_utc) & (df["ts_utc"] <= end_utc)].copy()
    # add local convenience columns
    df["ts_local"] = df["ts_utc"].dt.tz_convert("Europe/Zurich")
    df["date_local"] = df["ts_local"].dt.date
    df["hour_local"] = df["ts_local"].dt.hour
    return df

def save_latest(hours_back: int = 168) -> str:
    """
    Grab last N hours (default 7 days) and save CSV under data/weather_stgallen_hourly.csv.
    Returns the output path.
    """
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    start = now - timedelta(hours=hours_back)
    df = fetch_weather_hourly(start, now)
    os.makedirs("data", exist_ok=True)
    out = "data/weather_stgallen_hourly.csv"
    df.to_csv(out, index=False)
    print(f"saved: {out} ({len(df)} rows)")
    return out

if __name__ == "__main__":
    try:
        save_latest()
    except Exception as e:
        # make sure errors are visible in the terminal
        print("ERROR while fetching/saving weather:", repr(e))
