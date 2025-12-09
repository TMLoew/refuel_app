# Refuel Operations Cockpit

Refuel blends weather, gym attendance, and snack demand into one multipage Streamlit cockpit for merchandising, staffing, pricing, and procurement.

## Quick start

```bash
pip install -r requirements.txt
streamlit run frontend/streamlit_app/Home.py
```

Drop your telemetry CSV into `data/gym_checkins_stgallen_2025_patterned.csv` (or `data/gym_badges*.csv`). In the app you can upload a replacement file and make it live, then retrain the ML models from the Data Workbench.

## Feature map

- Dashboard: live traffic, weather, mix tables, snapshot export.
- Forecasts: scenario levers (weather, marketing, price), live weather option, procurement export, SKU rollup.
- What‑If Lab: quick simulations for allocation and pricing tweaks.
- Data Workbench: upload/preview/activate CSVs, inline data editor, retrain models on current data.
- Settings: weather profile and health cards.
- Statistics: correlation, scatter matrix, regression, daily rhythm tabs.
- POS Console: SKU-level sales logging, stock tracking, auto restock guardrails.
- Price Manager: adjust SKU prices feeding forecasts and mix.
- Autopilot: procurement simulation with scenario metadata saved to `data/procurement_plan.csv`.

## Data & models

- Telemetry loader auto-detects preferred datasets, enriches with live or cached Open‑Meteo weather, and writes cache to `data/weather_cache.csv`.
- ML: HistGradientBoosting models for check-ins and snacks persist to `model/checkins_hgb.joblib` and `model/snacks_hgb.joblib`; retrain from Data Workbench when data changes.
- Procurement, POS, and pricing artifacts live under `data/` (product mix, prices, restock policy, POS log, procurement plan, snapshots).

## Useful commands

- Fetch weather aligned to telemetry: `python backend/app/services/weather_service.py`
- Rebuild product mix CSV: `python backend/app/services/pipeline.py`
- Validate configs: `python scripts/validate_configs.py`
- Train models via CLI: `python backend/app/services/ml/train_demand_model.py --csv data/gym_checkins_stgallen_2025_patterned.csv`

## Brand palette

Colors used in UI and Plotly templates (hex only for reliable rendering):

- HSG Grün: `#00802F`
- Rot: `#EB6969`
- Blau: `#73A5AF`
- Beige: `#E1D7C3`
- Gelb: `#FFF04B`
