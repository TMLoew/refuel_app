import requests
from datetime import datetime

class Ingredient:
    def _init_(self, name: str, base_price_per_unit: float):
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
    if month in (12, 1, 2): return "Winter"
    if month in (3, 4, 5):  return "Spring"
    if month in (6, 7, 8):  return "Summer"
    return "Autumn"


sales_data = [
    {"date": "2025-10-24", "product": "Banana Shake", "qty": 28},
    {"date": "2025-10-25", "product": "Banana Shake", "qty": 32},
    {"date": "2025-10-26", "product": "Banana Shake", "qty": 21},
]
gym_data = [
    {"date": "2025-10-24", "visitors": 230},
    {"date": "2025-10-25", "visitors": 255},
    {"date": "2025-10-26", "visitors": 190},
]
products_set = {row["product"] for row in sales_data}


seasonality = {
    "Banana": {
        "Winter": (0.6, 1.3),
        "Summer": (1.0, 1.0),
        "Spring": (0.9, 1.1),
        "Autumn": (0.8, 1.2),
    }
}


def get_daily_temp(date_str: str, lat=47.4239, lon=9.3748) -> float:
    try:
        url = (
            "https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            f"&start_date={date_str}&end_date={date_str}"
            "&daily=temperature_2m_max&timezone=Europe/Zurich"
        )
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return float(r.json()["daily"]["temperature_2m_max"][0])
    except Exception:
        return 18.0  # fallback si API KO


class DataPipeline:
    def _init_(self, ingredient: Ingredient, sales_rows: list, gym_rows: list):
        self.ingredient = ingredient
        self.sales_rows = sales_rows
        self.gym_rows = gym_rows
        self._merged = []

    @staticmethod
    def _parse_month(date_str: str) -> int:
        return datetime.strptime(date_str, "%Y-%m-%d").month

    @property
    def merged(self) -> list:
        return self._merged

    def _season_for_date(self, date_str: str) -> str:
        return month_to_season(self._parse_month(date_str))

    def _seasonal_indices(self, season: str) -> tuple:
        # safe lookup: évite KeyError si l'ingrédient ou la saison manquent
        return seasonality.get(self.ingredient.name, {}).get(season, (1.0, 1.0))

    def run(self) -> None:
        gym_by_date = {row["date"]: row["visitors"] for row in self.gym_rows}
        merged = []
        for row in self.sales_rows:
            d = row["date"]
            temp = get_daily_temp(d)
            visitors = gym_by_date.get(d, 0)
            season = self._season_for_date(d)
            avail, p_index = self._seasonal_indices(season)
            merged.append({
                "date": d,
                "product": row["product"],
                "qty": row["qty"],
                "visitors": visitors,
                "temp": temp,
                "season": season,
                "availability": avail,
                "price_index": p_index,
                "seasonal_price": self.ingredient.seasonal_price(season, p_index),
                "dow": datetime.strptime(d, "%Y-%m-%d").weekday()
            })
        self._merged = merged


def summary(merged_rows: list) -> str:
    n = len(merged_rows)
    total_qty = sum(r["qty"] for r in merged_rows)
    avg_temp = round(sum(r["temp"] for r in merged_rows) / max(n,1), 1)
    avg_vis = round(sum(r["visitors"] for r in merged_rows) / max(n,1), 1)
    seasons = {r["season"] for r in merged_rows}
    by_season = {s: sum(r["qty"] for r in merged_rows if r["season"] == s) for s in seasons}
    lines = [
        f"Records: {n}",
        f"Total sold: {total_qty}",
        f"Avg temp: {avg_temp}°C",
        f"Avg visitors: {avg_vis}",
        "Qty by season: " + ", ".join(f"{s}={by_season[s]}" for s in sorted(by_season))
    ]
    return "\n".join(lines)

if _name_ == "_main_":
    print("Products:", products_set)
    banana = Ingredient("Banana", 2.5)
    pipe = DataPipeline(banana, sales_data, gym_data)
    pipe.run()
    for rec in pipe.merged:
        print(rec)
    print("\n--- SUMMARY ---")
    print(summary(pipe.merged))
