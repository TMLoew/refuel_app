# Refuel Operations Cockpit

Refuel is our data cockpit that blends weather signals, gym attendance, and snack demand to guide merchandising, staffing, and marketing for campus fuel bars. The repo contains both the Streamlit frontend and the (in-progress) backend services used to synthesize telemetry.

## Live Demo

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://refuelapp.streamlit.app/)

## Getting Started

```bash
pip install -r requirements.txt
streamlit run frontend/streamlit_app/Home.py
```

Drop your gym telemetry CSV into `data/gym_badges.csv`, then toggle “Use live weather API” inside the app to pull Open-Meteo context automatically.

To refresh backend datasets:

```bash
# Fetch weather aligned with the gym dataset
python backend/app/services/weather_service.py
# Rebuild product mix suggestions
python backend/app/services/pipeline.py
```

## Repository Layout

- `frontend/streamlit_app/` – multipage Streamlit app (dashboard, forecast explorer, what-if simulator, data editor, settings)
- `frontend/streamlit_app/services/` – shared loaders, weather API wrapper, synthetic data utilities
- `backend/` – placeholder for future API services
- `data/` – sample telemetry (`gym_badges.csv`)
- `logo.webp` – brand asset used across the UI

## Data Inputs & Persistence

- **Gym telemetry** – Drop hourly badge exports into `data/gym_badges_0630_2200_long.csv` (preferred) or `data/gym_badges.csv`. The loader auto-detects the first file it finds and enriches it with synthetic or live weather before the dashboard renders anything.
- **Product mix planning** – Keep daily merchandising guidance in `data/product_mix_daily.csv`. At runtime the app uses `build_daily_product_mix_view()` to merge that daily plan with aggregated telemetry so you can compare suggested units vs. implied demand without mutating the raw sources.
- **Snapshots** – Use the “Save snapshot” button in the dashboard’s Product Mix section to persist the merged view (including gaps and implied units) to `data/product_mix_enriched.csv`. Any Streamlit page or external notebook can reload it through `load_product_mix_snapshot()` for reproducible reviews.
- **POS console + restocks** – `data/pos_runtime_log.csv` now captures each counter entry with optional per-product breakdowns and current shelf stock. Auto-restock preferences live in `data/restock_policy.json` and can be managed from the POS Console.
- **Procurement autopilot** – Running the autopilot simulation inside `Home.py` writes the latest recommendation to `data/procurement_plan.csv`, while Streamlit POS events append to `data/pos_runtime_log.csv`.

## Forecasting & Automation

- The dashboard adds a “Daily snack outlook” widget that aggregates historical telemetry plus the ML scenario forecast to daily totals, making it easier to compare actual vs. predicted check-ins/snacks per day.
- Daily snack forecasts are allocated across the merchandising mix via `allocate_product_level_forecast`, so you can sanity-check product-level expectations against suggested quantities for the next 72 hours.
- The POS Console supports item-level logging and can auto-trigger restock entries when inventory dips below a configurable floor (with cooldown protection). Manual and automatic restocks both flow through the same log, so downstream planners see a unified stock history.

## Dev Tips

- Use the top navigation bar inside Streamlit to hop between modules.
- The Settings page exposes weather API latency, credentials placeholders, and webhook status.
- Forecast Explorer features coefficient breakdowns, residual diagnostics, and correlation heatmaps to understand drivers.

## Links

- **Streamlit Cloud**: https://refuelapp.streamlit.app/
- **Issues/feedback**: please use GitHub Issues in this repo.
