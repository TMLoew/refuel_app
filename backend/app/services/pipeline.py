# backend/app/services/pipeline.py
"""
Build a daily product mix for Refuel from gym data + weather.
Inputs:
  - data/gym_badges_0630_2200_long.csv (your 15-min gym data)
Outputs:
  - data/product_mix_daily.csv (daily plan per product)
"""

from pathlib import Path
from typing import Dict, List, Optional
from datetime import date as _date
import requests
import pandas as pd

# St. Gallen
LAT, LON = 47.4245, 9.3767

# Products we care about
PRODUCTS = ["Protein Shake", "Electrolyte Drink", "Iced Matcha", "Recovery Smoothie", "Isotonic Lemon"]

def month_to_season(month: int) -> str:
    if month in (12, 1, 2): return "Winter"
    if month in (3, 4, 5):  return "Spring"
    if month in (6, 7, 8):  return "Summer"
    return "Autumn"

def recommend_mix(temp_max_c: float, precip_mm: float, cardio_share: Optional[float]=None) -> Dict[str, float]:
    """
    Very simple rules:
      - Hot & dry  -> Electrolyte, Iced Matcha, Isotonic up; Recovery down
      - Rainy      -> Recovery & Protein up; Iced Matcha down
      - Cardio bias (treadmill) -> Electrolyte/Isotonic up; Strength -> Protein up
    Returns weights that sum ~1.0
    """
    hot = temp_max_c >= 24
    rainy = precip_mm >= 2.0

    base = {
        "Protein Shake":      0.22,
        "Electrolyte Drink":  0.22,
        "Iced Matcha":        0.18,
        "Recovery Smoothie":  0.20,
        "Isotonic Lemon":     0.18,
    }

    if hot and not rainy:
        base["Electrolyte Drink"] += 0.06
        base["Iced Matcha"]       += 0.05
        base["Isotonic Lemon"]    += 0.03
        base["Recovery Smoothie"] -= 0.07

    if rainy:
        base["Recovery Smoothie"] += 0.06
        base["Protein Shake"]     += 0.04
        base["Iced Matcha"]       -= 0.05

    if cardio_share is not None:
        cardio_share = max(0.0, min(1.0, float(cardio_share)))
        base["Electrolyte Drink"] += 0.08 * cardio_share
        base["Isotonic Lemon"]    += 0.05 * cardio_share
        base["Protein Shake"]     += 0.10 * (1 - cardio_share)

    total = sum(base.values())
    return {k: round(v/total, 4) for k, v in base.items()}

def fetch_daily_weather_range(start_date: str, end_date: str, lat: float = LAT, lon: float = LON) -> pd.DataFrame:
    """
    Get daily max temperature + precipitation for [start_date, end_date] (Europe/Zurich).
    - Uses Archive API for past dates, Forecast API for current/future.
    - Falls back to neutral weather if API fails or times out.
    """
    today = _date.today().strftime("%Y-%m-%d")
    base = "https://archive-api.open-meteo.com/v1/archive" if end_date <= today else "https://api.open-meteo.com/v1/forecast"
    url = (
        f"{base}?latitude={lat}&longitude={lon}"
        f"&start_date={start_date}&end_date={end_date}"
        "&daily=temperature_2m_max,precipitation_sum&timezone=Europe/Zurich"
    )
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        j = r.json().get("daily", {})
        times = pd.to_datetime(j.get("time", []))
        if len(times) == 0:
            raise RuntimeError("No 'daily' data returned")
        dates = pd.Series(times.strftime("%Y-%m-%d"))
        df = pd.DataFrame({
            "date": dates,
            "temp_max_c": j["temperature_2m_max"],
            "precip_mm": j["precipitation_sum"],
        })
    except Exception as e:
        print(f"⚠️ WARNING: weather fetch failed ({e}); using fallback neutral values.")
        dates = pd.date_range(start=start_date, end=end_date, freq="D")
        df = pd.DataFrame({
            "date": dates.strftime("%Y-%m-%d"),
            "temp_max_c": [20.0] * len(dates),
            "precip_mm": [0.0] * len(dates),
        })
    return df

def load_gym_daily(csv_path: str) -> pd.DataFrame:
    """
    From your 15-min gym CSV, build per-day:
      - visitors = sum(checkins)
      - cardio_share = treadmill_sessions / checkins
    Falls back sensibly if columns are missing.
    """
    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(f"Gym CSV not found: {csv_path}")

    df = pd.read_csv(p)
    ts_col = "ts_local" if "ts_local" in df.columns else ("ts_local_naive" if "ts_local_naive" in df.columns else None)
    if not ts_col:
        raise ValueError("Expected 'ts_local' or 'ts_local_naive' column in gym CSV")
    df["date"] = pd.to_datetime(df[ts_col]).dt.date.astype(str)

    has_checkins = "checkins" in df.columns
    has_treadmill = "treadmill_sessions" in df.columns

    if has_checkins:
        visitors = df.groupby("date", as_index=False)["checkins"].sum().rename(columns={"checkins":"visitors"})
    else:
        visitors = df.groupby("date", as_index=False).size().rename(columns={"size":"visitors"})

    if has_checkins and has_treadmill:
        cardio = df.groupby("date", as_index=False)[["treadmill_sessions","checkins"]].sum()
        cardio["cardio_share"] = (cardio["treadmill_sessions"] / cardio["checkins"].clip(lower=1)).clip(0,1)
        cardio = cardio[["date","cardio_share"]]
    else:
        cardio = pd.DataFrame({"date": visitors["date"], "cardio_share": 0.5})

    return visitors.merge(cardio, on="date", how="left")

def build_product_mix(
    gym_csv_path: str = "data/gym_badges_0630_2200_long.csv",
    out_csv_path: str = "data/product_mix_daily.csv",
    base_conversion: float = 0.35,  # % of visitors who buy a drink
) -> str:
    gym = load_gym_daily(gym_csv_path)
    if gym.empty:
        raise RuntimeError("Gym daily aggregation is empty.")

    start_date, end_date = min(gym["date"]), max(gym["date"])
    wx = fetch_daily_weather_range(start_date, end_date)

    df = gym.merge(wx, on="date", how="left")
    df["season"] = pd.to_datetime(df["date"]).dt.month.map(month_to_season)

    rows: List[dict] = []
    for _, r in df.iterrows():
        mix = recommend_mix(float(r["temp_max_c"]), float(r["precip_mm"]), cardio_share=float(r["cardio_share"]))
        total_drinks = int(round(r["visitors"] * base_conversion))
        for product, weight in mix.items():
            rows.append({
                "date": r["date"],
                "visitors": int(r["visitors"]),
                "cardio_share": round(float(r["cardio_share"]), 3),
                "temp_max_c": round(float(r["temp_max_c"]), 1),
                "precip_mm": round(float(r["precip_mm"]), 1),
                "season": r["season"],
                "product": product,
                "weight": weight,
                "suggested_qty": int(round(total_drinks * weight)),
                "hot_day": int(r["temp_max_c"] >= 24),
                "rainy_day": int(r["precip_mm"] >= 2.0),
            })

    out = pd.DataFrame(rows)
    Path("data").mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv_path, index=False)
    print(f"saved: {out_csv_path} with {len(out)} rows across {out['date'].nunique()} days")
    return out_csv_path

if __name__ == "__main__":
    try:
        build_product_mix()
    except Exception as e:
        print("ERROR:", repr(e))
