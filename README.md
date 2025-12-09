# Refuel Operations Cockpit

Refuel is a Streamlit app that helps a gym shop plan snacks and drinks using simple forecasts, live weather, and a few editable CSV files.

## Quick start

```bash
pip install -r requirements.txt
streamlit run frontend/streamlit_app/App.py
```

Files the app expects:

- Telemetry: put your main CSV at `data/gym_checkins_stgallen_2025_patterned.csv` (or another `data/gym_badges*.csv` file). You can also upload a new file inside the app.
- Models: saved under `model/`. They retrain from the Data Workbench page; no math background needed.
- Weather cache: stored at `data/weather_cache.csv` if you turn on live weather.

## What you can do

- Dashboard: see current traffic, weather, and reorder advice.
- Forecasts: move sliders for weather, marketing, and prices to see simple “what if” outcomes.
- What‑If Lab: quick experiments for allocation and pricing.
- Data Workbench: upload and edit CSVs before making them active.
- POS Console: log sales/restocks and track stock.
- Price Manager: update price lists that feed the forecasts.
- Settings & Statistics: tweak weather profile and view basic charts.
- Autopilot: run a basic procurement simulation and export a plan to `data/procurement_plan.csv`.

## Data and models

- Everything is file-based and readable: product mix, prices, POS log, procurement plan, and restock policy all live in `data/`.
- Forecasts use lightweight tree models saved to `model/checkins_hgb.joblib` and `model/snacks_hgb.joblib`; retrain from the Data Workbench after changing data.
- Weather is either pulled live or loaded from the cached CSV so the app works offline.

## Handy scripts (optional)

- Align telemetry with weather: `python backend/app/services/weather_service.py`
- Rebuild product mix CSV: `python backend/app/services/pipeline.py`
- Validate configs: `python scripts/validate_configs.py`
- Train models from CLI: `python backend/app/services/ml/train_demand_model.py --csv data/gym_checkins_stgallen_2025_patterned.csv`

## Brand palette

Colors used in UI and Plotly templates:

- HSG Grün: `#00802F`
- Rot: `#EB6969`
- Blau: `#73A5AF`
- Beige: `#E1D7C3`
- Gelb: `#FFF04B`
