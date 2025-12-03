# Refuel Operations Cockpit

Refuel is our data cockpit that blends weather signals, gym attendance, and snack demand to guide merchandising, staffing, and marketing for campus fuel bars. The repo contains the Streamlit multipage frontend plus backend services that hydrate telemetry, train ML models, and persist recommendations.

## Live Demo

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://refuelapp.streamlit.app/)

## Getting Started

```bash
pip install -r requirements.txt
streamlit run frontend/streamlit_app/Home.py
```

Drop your gym telemetry CSV into `data/gym_checkins_stgallen_2025_patterned.csv` (or keep using `data/gym_badges*.csv`), then toggle “Use live weather API” inside the app to pull Open-Meteo context automatically.

To refresh backend datasets:

```bash
# Fetch weather aligned with the gym dataset
python backend/app/services/weather_service.py
# Rebuild product mix suggestions
python backend/app/services/pipeline.py
```

## Repository Layout & Feature Map

- `frontend/streamlit_app/Home.py` – landing hub with autopilot summaries, navigation shortcuts, and telemetry hero cards
- `frontend/streamlit_app/pages/1_Dashboard.py` – live dashboard (traffic, weather, mix tables, snapshot export)
- `frontend/streamlit_app/pages/2_Forecasts.py` – forecast explorer (scenario levers, model diagnostics, procurement export)
- `frontend/streamlit_app/pages/3_WhatIf_Sim.py` – what-if simulator for allocation & pricing experiments
- `frontend/streamlit_app/pages/4_Data_Editor.py` – CSV uploader/editor for telemetry
- `frontend/streamlit_app/pages/5_Settings_APIs.py` – API/health console (weather profile, webhook states, cache latency)
- `frontend/streamlit_app/pages/6_Statistics.py` – diagnostics on correlations, residuals, and seasonality
- `frontend/streamlit_app/pages/7_POS_Console.py` – POS entry (per-product sales logging, auto restock)
- `frontend/streamlit_app/pages/8_Price_Manager.py` – SKU price overrides
- `frontend/streamlit_app/services/data_utils.py` – core data layer (live weather → cache fallback, enrichment, ML helpers)
- `frontend/streamlit_app/services/weather_pipeline.py` – Open‑Meteo client + synthetic weather generator
- `backend/app/services/ml/` – shared demand model training + CLI trainer
- `backend/app/services/pipeline.py` – batch builder for daily product mix CSV
- `data/` – sample telemetry & persisted artifacts (`gym_badges*.csv`, product mix, weather cache, POS log, etc.)
- `logo.jpeg` – brand asset used across the UI

## Data Inputs & Persistence

- **Gym telemetry** – Drop hourly badge exports into `data/gym_checkins_stgallen_2025_patterned.csv` (preferred) or the legacy `data/gym_badges*.csv`. The loader auto-detects the first file it finds and enriches it with synthetic or live weather before the dashboard renders anything.
- **Product mix planning** – Keep daily merchandising guidance in `data/product_mix_daily.csv`. At runtime the app uses `build_daily_product_mix_view()` to merge that daily plan with aggregated telemetry so you can compare suggested units vs. implied demand without mutating the raw sources.
- **Snapshots** – Use the “Save snapshot” button in the dashboard’s Product Mix section to persist the merged view (including gaps and implied units) to `data/product_mix_enriched.csv`. Any Streamlit page or external notebook can reload it through `load_product_mix_snapshot()` for reproducible reviews.
- **POS console + restocks** – `data/pos_runtime_log.csv` captures every entry with per-product breakdowns and shelf stock. Auto restock preferences live in `data/restock_policy.json` and can be managed from the POS Console.
- **Procurement autopilot** – Running the autopilot simulation inside `Home.py` or publishing a scenario from the Forecast Explorer writes to `data/procurement_plan.csv`. The file carries `plan_*` metadata columns (weather pattern, promo, horizon, etc.) that downstream tabs surface automatically. Streamlit POS events append to `data/pos_runtime_log.csv`.
- **Price overrides** – Use the Price Manager page to edit `data/product_prices.csv` so each SKU has its own unit price. Those values feed into the dashboard mix snapshot, scenario allocations, and procurement exports.
- **Config validator** – Run `python scripts/validate_configs.py` to sanity-check that mix, price, and restock files exist and contain the expected schema before deploying changes.

## Live Weather & Caching

- Live weather is on by default across the app. Hourly Open‑Meteo pulls hydrate the telemetry loader, POS console, dashboards, and forecasts.
- Successful fetches are persisted to `data/weather_cache.csv`. When the API is unreachable, the loader transparently falls back to that cached window before resorting to synthetic weather, so the UI keeps rendering with real-world history.
- The Settings page surfaces latency and cache age so you can tell whether views are using live, cached, or fallback data.

## Forecast Explorer & Scenarios

- Dial in a scenario on `pages/2_Forecasts.py` (weather overrides, marketing boost, promo, horizon) to generate forward hourly predictions. Hover the ℹ️ badges beside each section to see the underlying math/formulas.
- Enable “Use live weather API” to pull Open-Meteo’s forward forecast so the demand outlook reflects real upcoming conditions rather than just synthetic profiles.
- Download the hourly CSV for external analysis or publish the aggregated plan directly into `data/procurement_plan.csv`. The “Procurement actions” expander also shows the last published plan timestamp so you know if the shared plan reflects your scenario.
- The “Daily rollup & product mix impact” table allocates forecasted snack demand across the merchandising mix using each product’s weight, giving procurement teams a SKU-level view for the next few days.

## Forecasting & Automation

- Lightweight HistGradientBoosting models for check-ins + snack demand live under `backend/app/services/ml/`. The Streamlit app loads persisted weights if present or retrains them automatically when enough telemetry exists, caching artifacts in `model/`.
- The Home hero (“Daily snack outlook”) aggregates historical telemetry plus the scenario forecast to daily totals, making it easier to compare actual vs. predicted check-ins/snacks per day.
- Product-level allocation happens via `allocate_product_level_forecast`, so every forecasted unit is mapped onto the current mix weights for SKU-specific planning.
- The autopilot simulation (Home) uses those models, live/cached weather, and price overrides to propose procurement actions and write them to `data/procurement_plan.csv` along with scenario metadata.
- The POS Console enforces per-product logging, auto-adjusts stock levels, and can trigger restocks when thresholds are breached (with cooldown). Manual and automatic restocks both flow through the same log, so downstream planners see a unified history.

## Settings & Operations

- Weather profile (lat/lon, cache horizon, timeout) can be edited on the Settings page and persists to `data/weather_profile.json`.
- Health cards show live API latency, gym sensor freshness (minutes since last telemetry), and POS heartbeat using real timestamps instead of placeholders.
- Token + webhook sections provide placeholders for ops notes while actual secrets remain managed by your deployment platform.

## Dev Tips

- Use the top navigation bar inside Streamlit to hop between modules.
- The Settings page exposes weather API latency, credentials placeholders, and webhook status.
- Forecast Explorer features coefficient breakdowns, residual diagnostics, and correlation heatmaps to understand drivers.
- POS Console defaults to the product catalog detected in `data/product_mix_daily.csv` (with a sensible fallback list) so staff can log SKU-level sales quickly.
- If you need to prime the ML models outside of Streamlit, run `python backend/app/services/ml/train_demand_model.py --csv data/gym_checkins_stgallen_2025_patterned.csv`.

## Links

- **Streamlit Cloud**: https://refuelapp.streamlit.app/
- **Issues/feedback**: please use GitHub Issues in this repo.

## Brand Palette

We follow the HSG color scheme across the UI and charts:

| Swatch | Hex | Preview |
|--------|-----|---------|
| HSG Grün | `#00802F` | <img src="data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='18' height='18'><rect width='18' height='18' fill='%2300802F'/></svg>" /> |
| Rot | `#EB6969` | <img src="data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='18' height='18'><rect width='18' height='18' fill='%23EB6969'/></svg>" /> |
| Blau | `#73A5AF` | <img src="data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='18' height='18'><rect width='18' height='18' fill='%2373A5AF'/></svg>" /> |
| Beige | `#E1D7C3` | <img src="data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='18' height='18'><rect width='18' height='18' fill='%23E1D7C3'/></svg>" /> |
| Gelb | `#FFF04B` | <img src="data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='18' height='18'><rect width='18' height='18' fill='%23FFF04B'/></svg>" /> |

The custom Plotly template and navigation pills use these values (plus darker accents) so the product, dashboard charts, and marketing assets all share a consistent look.
