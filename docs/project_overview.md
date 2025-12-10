# Refuel Ops – Student Guide

This guide explains the project in plain language so any bachelor-level contributor can run, edit, and extend it. The app is a Streamlit dashboard for a gym shop: it combines weather, check-ins, and snack sales to suggest prices, stock, and simple plans.

---

## 1) How to run it

Requirements: Python 3.11+, pip, and the data files in the `data/` folder.

```bash
pip install -r requirements.txt
streamlit run frontend/streamlit_app/App.py
```

The app reads and writes local CSV/JSON files only, so it works offline. Live weather is optional and cached to `data/weather_cache.csv`.

---

## 2) How the code is organized

- `frontend/streamlit_app/pages/*.py` – each Streamlit page (dashboard, forecasts, POS, etc.).
- `frontend/streamlit_app/components/layout.py` – shared navigation and theme helpers.
- `frontend/streamlit_app/services/data_utils.py` – all data loading/saving, forecasting, and autopilot helpers.
- `data/` – every CSV/JSON the app needs (telemetry, prices, product mix, POS log, restock policy, procurement plan).
- `model/` – saved models for check-ins and snacks.
- `backend/` – optional scripts to rebuild datasets.
- `docs/` – documentation (this file plus others).

---

## 3) Data files you should know

| File | What it stores |
|------|----------------|
| `data/gym_badges*.csv` | Check-ins, snack sales, and weather columns; the main input. |
| `data/weather_cache.csv` | Cached weather so the app can work without the API. |
| `data/product_mix_daily.csv` | List of products and planned daily units. |
| `data/product_prices.csv` | Price list used by forecasts and the Price Manager. |
| `data/pos_log.csv` | Logged sales/restocks with optional per-product breakdown. |
| `data/restock_policy.json` | Auto-restock settings (thresholds and cooldown). |
| `data/autopilot_infinite.csv` and `data/procurement_plan.csv` | Outputs from the simple procurement simulation. |
| `rainy_day` column | Flag set to 1 when daily precipitation is at least 2mm (used to bias the product mix toward warm drinks). |

Everything is human-readable. If a file is missing, most pages will show an error message you can fix by adding the file.

---

## 4) Forecasting in simple terms

- Models live in `model/checkins_hgb.joblib` and `model/snacks_hgb.joblib`.
- They are lightweight tree models from scikit-learn; no advanced math needed to retrain.
- Retrain inside the app on the Data Workbench page after you swap in new telemetry.
- Scenario sliders (weather, marketing, price) adjust inputs before predictions are shown.

---

## 5) Quick tour of the pages

- **Dashboard (`1_Dashboard.py`)** – key metrics, recent weather, forecast chart, and a reorder suggestion.
- **Forecasts (`2_Forecasts.py`)** – compare multiple scenarios side by side and export procurement plans.
- **What-If Lab (`3_WhatIf_Sim.py`)** – simple sliders to test pricing or campaign ideas.
- **Data Workbench (`4_Data_Editor.py`)** – upload, preview, and edit CSV slices safely.
- **Settings & APIs (`5_Settings_APIs.py`)** – set weather location and view placeholders for tokens.
- **Statistics (`6_Statistics.py`)** – basic charts to spot patterns.
- **POS Console (`7_POS_Console.py`)** – log sales/restocks, see stock posture, adjust auto-restock rules.
- **Price Manager (`8_Price_Manager.py`)** – edit and save price lists in one grid.

---

## 6) Common workflows

**Upload new telemetry**
1. Open Data Workbench.
2. Drag in your CSV (match the columns in the table).
3. Review the preview, then activate it. Retrain models afterward.

**Log sales and restocks**
1. Open POS Console.
2. Enter per-product counts and any restock.
3. Submit; the app updates `data/pos_log.csv` and recalculates stock.

**Adjust prices**
1. Open Price Manager.
2. Edit values in the grid or add/remove products.
3. Click “Save prices” to rewrite `data/product_prices.csv`.

**Run a procurement simulation**
1. Go to Forecasts.
2. Pick a scenario with the sliders.
3. Start the autopilot; it will step through future hours and write plans to `data/procurement_plan.csv`.

---

## 7) Extending the app without heavy theory

- **Add a dataset:** place a CSV in `data/`, write a small loader in `data_utils.py`, and use it in a page.
- **Create a new page:** copy an existing file under `frontend/streamlit_app/pages/`, import `render_top_nav` and `render_footer`, and add an entry to `DEFAULT_NAV` in `layout.py`.
- **Change the model:** swap the algorithm in `train_models` inside `data_utils.py` if you want, but keep it simple (tree or linear models are enough).

---

## 8) Coding notes

- Use `st.session_state` for values that should persist between reruns (for example, the autopilot running flag).
- Write through helpers in `data_utils.py` so files stay consistent.
- If a required file is missing, show `st.error` and stop early.
- Keep functions small and side-effect free to make later testing straightforward.

---

## 9) Roadmap ideas (plain language)

- Switch from CSVs to a tiny database if multiple people edit at once.
- Separate the backend so Streamlit only calls APIs.
- Add email/Slack notifications for reorder alerts.
- Let users pick between a light and dark theme.
- Track model versions and training notes for transparency.

---

## 10) Glossary

| Term | Meaning here |
|------|--------------|
| **SKU** | A single product, like “Protein Shake.” |
| **Telemetry** | The raw table of check-ins, snack sales, and weather. |
| **Scenario** | A set of slider choices (weather, marketing, price) for a forecast. |
| **POS log** | The file that stores every sale or restock entry. |
| **Safety stock** | A buffer level where we warn before running out. |
| **Autopilot** | A simple loop that steps through the forecast and proposes when to reorder. |

Update this guide whenever flows or datasets change so new contributors can follow along.
