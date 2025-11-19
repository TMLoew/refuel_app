# Refuel Ops – Deep Technical Guide

This document serves as the canonical reference for the Refuel Ops project. It explains the motivation behind each feature, how the codebase is organized, and how data flows from CSVs into the analytics that Streamlit renders. The intent is to make the code understandable even for contributors who have never touched Streamlit or the repo before.

---

## 1. Product Vision and User Flow

Refuel Ops targets gym operators who manage a small retail shop (snacks, drinks) inside their facility. These operators need answers to three questions:

1. **How busy are we right now, and what caused it?** → dashboards with weather, marketing, and check-in telemetry.
2. **What should we plan for next?** → machine-learning forecasts with what-if controls.
3. **Do we need to restock or change pricing?** → live POS logging, reorder guidance, and price management.

### Typical daily flow

1. Staff opens Streamlit and lands on `1_Dashboard`.
2. They scan summary metrics, adjust scenario sliders (weather pattern, marketing boost) to stress-test the forecast.
3. They review the **Model-driven reorder guidance** block to see exactly when the next reorder should happen.
4. During the day, they log check-ins/sales on `7_POS_Console`.
5. Once per week they open `8_Price_Manager` to adjust SKU prices and `4_Data_Editor` to inspect raw telemetry.

Everything is file-backed. There is no remote API or database. Streamlit reads and writes CSV/JSON files in the repository itself, which means the app can run offline and code reviewers can inspect the data versioned with Git.

---

## 2. Architecture Overview

```
┌────────────────────────────────────────────────────────────────────────┐
│                         Streamlit Application                          │
│                                                                         │
│  ┌─────────────┬──────────────┬──────────────┬───────────────────────┐  │
│  │1_Dashboard  │2_Forecasts   │7_POS_Console │8_Price_Manager, ...   │  │
│  └─────┬───────┴────┬─────────┴─────┬────────┴───────┬──────────────┘  │
│        │             │              │                │                 │
│        │             │              │                │                 │
│  ┌─────▼─────┐ ┌────▼─────┐ ┌──────▼─────┐ ┌────────▼────────┐        │
│  │components/│ │services/ │ │ data/*.csv │ │ Session state   │        │
│  │layout.py  │ │data_utils│ │ weather JSON│ │ Forecast models │        │
│  └───────────┘ └──────────┘ └────────────┘ └────────────────┘        │
└────────────────────────────────────────────────────────────────────────┘
```

### Layers explained

| Layer | Description |
|-------|-------------|
| **Streamlit pages** (`frontend/streamlit_app/pages/*.py`) | Each file is an entry point exposed in the UI. All pages import the same layout components and service utilities. |
| **Components** (`components/layout.py`) | Houses `render_top_nav`, `sidebar_info_block`, and `render_footer`. Also injects theme CSS so all pages share the same look and feel. |
| **Service layer** (`services/data_utils.py`) | Handles every side effect: reading/writing CSVs, caching, forecasting, auto-restock logic, autopilot simulation. Think of it as the backend. |
| **Data** (`data/*.csv`) | CSVs tracked in Git. Examples: `gym_badges.csv` (telemetry), `product_prices.csv`, `product_mix_daily.csv`, `pos_log.csv`, `weather_cache.csv`, `autopilot_infinite.csv`, etc. |
| **Session state** (`st.session_state`) | Stores ephemeral UI state (e.g., autopilot timers, info-tip dismissals, recently viewed slider values). |

Because there is no separate API, everything is synchronous. A page calls a helper, receives a DataFrame/Dict, renders it, and writes back to disk if needed.

---

## 3. Repository Tour

| Path | Details |
|------|---------|
| `frontend/streamlit_app/pages/1_Dashboard.py` | Landing dashboard: KPIs, weather radar, what-if forecast chart, and model-driven reorder guidance. |
| `frontend/streamlit_app/pages/2_Forecasts.py` | Scenario lab with autopilot simulation (includes procurement autopilot panel). |
| `frontend/streamlit_app/pages/3_WhatIf_Sim.py` | Experimental elasticity simulator (less prominent in nav). |
| `frontend/streamlit_app/pages/4_Data_Editor.py` | Drag-and-drop CSV uploader at the top, editable grid, summary stats. |
| `frontend/streamlit_app/pages/5_Settings_APIs.py` | Weather profile management, webhook status, secrets placeholder, automation unlock. |
| `frontend/streamlit_app/pages/7_POS_Console.py` | POS entry form, live stock posture, auto restock policy editor, real vs. forecast comparison. |
| `frontend/streamlit_app/pages/8_Price_Manager.py` | Single editable grid for price list with add/remove controls. |
| `frontend/streamlit_app/components/layout.py` | Custom navigation bar, theme injection, sidebar/footer helpers, tooltip CSS. |
| `frontend/unused_ideas/` | Archived experiments such as the Inventory Sandbox game. |
| `frontend/streamlit_app/services/data_utils.py` | 800+ lines of shared logic (load/save functions, forecasting, autopilot). Every page interacts with this module. |
| `backend/` | Offline scripts or notebooks that generate enriched data (not required at runtime). |
| `docs/` | Documentation such as this file. |

---

## 4. Domain Terminology & Data Schemas

### 4.1 Terminology

| Term | Meaning in this project |
|------|------------------------|
| **SKU** (Stock Keeping Unit) | A specific product that can be sold individually (e.g., “Protein Shake”, “Iced Matcha”). Each SKU has its own row in `product_mix_daily.csv` and `product_prices.csv`. |
| **Telemetry row** | A single timestamp entry in `gym_badges.csv`, containing observed check-ins, snack sales, and weather features. |
| **POS entry** | A row in `pos_log.csv` representing a manual or auto-logged event (sales batch, restock, or automation). |
| **Scenario** | Dictionary describing the what-if configuration for forecasting (weather pattern, marketing boost, etc.). |
| **Autopilot** | Iterative simulation that steps through forecast horizons, applying restock rules to produce a procurement plan. |
| **Safety stock** | The minimum stock level (units) that should never be crossed. If projected stock dips below safety stock, a reorder is recommended. |

### 4.2 File schemas

| File | Columns | Description |
|------|---------|-------------|
| `data/gym_badges.csv` | `timestamp`, `checkins`, `snack_units`, `snack_revenue`, `temperature_c`, `precipitation_mm`, `humidity_pct`, ... | Master telemetry table. Every row corresponds to an observation interval (typically hourly). |
| `data/weather_cache.csv` | `timestamp`, `temperature_c`, `precipitation_mm`, `humidity_pct`, metadata columns | Cache of weather API responses. Joined into telemetry when `use_weather_api=True`. |
| `data/pos_log.csv` | `timestamp`, `sales_units`, `checkins_recorded`, `stock_remaining`, `notes`, `product_breakdown` (JSON), `source` | Append-only log of POS entries. `product_breakdown` stores per-SKU counts (e.g., `{"Protein Shake": 4}`) as JSON text. |
| `data/product_mix_daily.csv` | `date`, `product`, `units`, optional metadata | Planning file for SKUs. Used to infer the set of available SKUs. |
| `data/product_prices.csv` | `product`, `unit_price` | Master price list consumed by both forecasting and Price Manager. |
| `data/restock_policy.json` | `auto_enabled`, `threshold_units`, `lot_size`, `cooldown_hours`, `last_auto_restock` | Stored by `save_restock_policy` and read by POS Console + dashboard. |
| `data/autopilot_infinite.csv` | Generated columns: `date`, `scenario`, `price`, `demand_est`, `stock_after`, `reordered`, `reorder_qty`, `profit`, `plan_*` metadata | Output of autopilot simulation for auditability. |

Understanding these schemas is crucial; almost every helper in `data_utils.py` translates directly into the columns described above.

---

## 5. Service Layer (data_utils.py) – API Reference

The `data_utils.py` module is the glue. Below is a categorized list of its most relevant functions.

### 5.1 Telemetry + Weather

- `load_enriched_data(use_weather_api: bool, cache_buster: float = 0.0) -> pd.DataFrame`
  - Reads `gym_badges.csv`, merges weather cache if desired, ensures `timestamp` is tz-aware, and sets attributes (`attrs["weather_source"]`, `attrs["weather_meta"]`).
  - Memoization: pages call this inside `st.spinner` so they can display loading messages.
- `load_weather_profile() -> dict` / `save_weather_profile(profile: dict)`
  - Persist lat/lon plus API tuning parameters under `data/weather_profile.json`.

### 5.2 Forecasting

- `train_models(df: pd.DataFrame) -> Tuple[BaseEstimator, BaseEstimator]`
  - Typically returns `(LinearRegression, GradientBoostingRegressor)` for check-ins and snack units.
  - Adds feature engineering (lags, weather effects) internally.
- `build_scenario_forecast(df, models, scenario, anchor_timestamp=None)`
  - Accepts numerous scenario keys; gracefully ignores unknown keys for forward compatibility.
  - Produces a DataFrame sorted by timestamp with predicted metrics plus derived fields (e.g., `pred_snack_revenue = pred_snack_units * snack_price`).

### 5.3 POS Helpers

- `append_pos_log(payload: dict) -> None`
  - Payload keys: `timestamp` (ISO string or datetime), `sales_units`, `checkins_recorded`, `stock_remaining`, `notes`, `product_breakdown` (dict), optional `source`.
  - Appends to `pos_log.csv`, creating the file if missing.
- `load_pos_log() -> pd.DataFrame`
  - Returns DataFrame with parsed timestamps and `product_breakdown` converted back to dicts (via `_safe_load_breakdown`).

### 5.4 Pricing

- `load_product_mix_data()`, `load_product_prices()`, `save_product_prices(df)`
  - Provide typed DataFrames for SKU operations.
- `add_or_update_product_price(name, unit_price)` / `remove_product_price(name)`
  - Utility wrappers used by the Price Manager page.

### 5.5 Restock automation

- `load_restock_policy()`, `save_restock_policy(policy)`
  - Manage the JSON policy file.
- `should_auto_restock(current_stock: float, policy: dict) -> bool`
  - Enforces threshold and cooldown rules to prevent repeated restocks.
- `mark_auto_restock(policy)` updates `last_auto_restock` and persists the policy.

### 5.6 Autopilot

- `load_autopilot_history_file()`, `save_autopilot_history_file()`, `reset_autopilot_history_file()`
  - Manage the CSV log that stores autopilot runs.
- `run_auto_simulation(forecast_hours, starting_stock, ...) -> pd.DataFrame`
  - Converts hourly forecast to daily plan, applies pricing strategy, restocks when `stock_after <= safety_stock`.
- `advance_autopilot_block(...) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[str], Optional[str]]`
  - The main driver called repeatedly when autopilot is “running”. Returns updated history plus optional warnings/errors for the UI.

Understanding these APIs means you can follow how every page interacts with file-backed state.

---

## 4. Data Flows

### 4.1 Telemetry + Weather + Forecasting

1. **Source**: `data/gym_badges.csv` is the canonical telemetry table.
2. **Loader**: `load_enriched_data(use_weather_api=True)` reads the CSV and merges live weather (from `weather_cache.csv`). If the live API fails, a synthetic fallback is used, and metadata is attached: `df.attrs["weather_source"]`.
3. **Model training**: `train_models(df)` returns `(checkins_model, snacks_model)` and is memoized for the session.
4. **Scenario**: Each page builds a scenario dict:

   ```python
   scenario = {
       "horizon_hours": 24,
       "weather_pattern": selected_pattern,
       "temp_manual": temp_offset,
       "precip_manual": precip_offset,
       "event_intensity": event_slider,
       "marketing_boost_pct": marketing_slider,
       "snack_price_change": price_slider,
       "use_live_weather": use_weather_api_toggle,
   }
   ```

5. **Forecast**: `build_scenario_forecast(df, models, scenario)` returns a DataFrame with columns:
   - `timestamp` (hourly)
   - `pred_checkins`
   - `pred_snack_units`
   - `snack_price`
   - `pred_snack_revenue`
   - Weather columns (`temperature_c`, `precipitation_mm`)
6. **Visualization**: Dashboard, Forecasts, and POS Console read from this DataFrame to create charts and metrics.

```
data/gym_badges.csv ─▶ load_enriched_data ─▶ train_models ─▶ build_scenario_forecast
        ▲                                                      │
        │                                                      ▼
 weather_cache.csv                               Plotly charts, reorder planner
```

### 4.2 POS Logging & Reorder Guidance

1. **Input**: In `pages/7_POS_Console.py`, users fill out a form (timestamp, per-product logs, restocks, notes).
2. **Persistence**: `append_pos_log()` writes rows to `data/pos_log.csv`. It stores `product_breakdown` as JSON.
3. **Reading**: `load_pos_log()` converts timestamps, decodes JSON columns, and returns a typed DataFrame.
4. **Consumption**:
   - POS page shows recent entries, current stock, alerts.
   - Dashboard’s reorder planner reads the most recent stock, merges forecasted consumption, and advises a reorder time.

### 4.3 Pricing & Product Mix

1. `load_product_mix_data()` loads `data/product_mix_daily.csv`.
2. `load_product_prices()` reads `data/product_prices.csv`.
3. Price Manager merges them, fills missing prices with `DEFAULT_PRODUCT_PRICE`, and exposes them in `st.data_editor`.
4. `save_product_prices()` rewrites the CSV when the user clicks “Save prices”.

### 4.4 Auto-restock policies

1. Config stored in `data/restock_policy.json`.
2. `load_restock_policy()` returns a dict containing `auto_enabled`, `threshold_units`, `lot_size`, `cooldown_hours`, `last_auto_restock`.
3. POS Console uses `should_auto_restock(current_stock, policy)` to decide if a background restock should trigger.
4. `mark_auto_restock()` updates `last_auto_restock` and persists the policy.

---

## 5. Page-by-Page Behavior

### 5.1 Dashboard (`pages/1_Dashboard.py`)

1. **Top nav** via `render_top_nav`.
2. **Sidebar** toggles:
   - `use_weather_api` (bool)
   - `history_days` slider
   - scenario sliders grouped in expanders (“Weather & demand levers”, “Campaign & pricing knobs”).
3. **Telemetry load** (`load_enriched_data`) and model training (`train_models`).
4. **Weather shotcast** using saved lat/lon (`load_weather_profile`).
5. **Forecast chart** via `render_forecast_section` shows actual vs. predicted check-ins/snack units.
6. **Model-driven reorder guidance**:
   - `_latest_stock_reading()` fetches last `stock_remaining` from POS log.
   - `_estimate_daily_demand()` calculates average daily demand from history or forecast.
   - Users set `current_stock`, `safety_days`, `reorder_lot`.
   - The code calculates when projected stock crosses the safety buffer and renders metrics + an area chart.

### 5.2 Forecasts (`pages/2_Forecasts.py`)

*Highlights:*

- Same base forecast as the dashboard but with a table that lets users compare multiple scenarios side by side.
- **Autopilot simulation**: pressing the “start” button runs `advance_autopilot_block()` repeatedly (with Streamlit session state controlling the loop). Results are saved via `save_procurement_plan()`.
- Historical view of autopilot is stored in `data/autopilot_infinite.csv`.

### 5.3 What-if Simulator (`pages/3_WhatIf_Sim.py`)

- This page experiments with price elasticity, marketing campaigns, and weather “stories.” It shares `build_scenario_forecast` but presents results through interactive widgets (sliders, charts). It is kept near the end of navigation due to lower priority.

### 5.4 Data Editor (`pages/4_Data_Editor.py`)

- **Drag-and-drop** (`st.file_uploader`) sits at the top; it loads the file into `new_df` and shows `st.dataframe`.
- **Filter** controls (date slider, row count) slice the main dataset via `mask = data["timestamp"].between(...)`.
- **Editable grid** (`st.data_editor`) allows inline edits with `num_rows="dynamic"`.
- **Download** button exports the currently edited slice to CSV; no automatic persistence to avoid accidental overwrites.

### 5.5 Settings & APIs (`pages/5_Settings_APIs.py`)

- Displays current weather API profile (lat/lon/timeouts) with `load_weather_profile`/`save_weather_profile`.
- Shows placeholders for tokens and webhook status (mock data for now).
- Offers a “Download config JSON” button containing environment info.
- **Automation panel** (password protected): when `OPS_PASSWORD` is set in `.streamlit/secrets.toml`, users can run the procurement autopilot block right from this page. The previously visible procurement snapshot table was intentionally removed.

### 5.6 POS Console (`pages/7_POS_Console.py`)

1. **Form** collects per-product sales, restock delta, baseline stock, check-ins, and notes.
2. On submit:
   - Calculate `stock_remaining = max(0, baseline + restock_delta - logged_units)`.
   - Append to log via `append_pos_log`.
   - Reload `log_df` and `restock_policy` to display updated state.
3. **Auto-restock** button triggers manual restock entries.
4. **Live stock posture** offers sliders and number inputs to tweak restock policy; writing uses `save_restock_policy`.
5. **Instant forecast comparison**:
   - Merge `log_df` with `forecast` using `pd.merge_asof`.
   - Drop rows with missing timestamps before merging (fix implemented to avoid `ValueError: Merge keys contain null values`).
   - Show bar chart comparing actual vs. forecasted snack units.

### 5.7 Price Manager (`pages/8_Price_Manager.py`)

- Unifies all known SKUs in a single `st.data_editor`.
- Buttons:
  - “Save prices” calls `save_product_prices`.
  - “Reset to defaults” sets every SKU back to `DEFAULT_PRODUCT_PRICE`.
  - “Add / update product” and “Remove product” allow explicit SKU management.
- The old “Current file snapshot” table was removed; the editable grid covers that use case.

---

## 6. Styling & Navigation

`frontend/streamlit_app/components/layout.py` contains the theming logic:

1. `render_top_nav(active_page)` hides Streamlit’s default sidebar nav, injects CSS once per session, and renders custom buttons for each entry in `DEFAULT_NAV`.
2. `_inject_theme_css()` defines CSS variables:
   - `--refuel-primary`: accent color (#1f6f8b).
   - `--refuel-surface`: global background (set to white to match chart backgrounds).
   - `--refuel-pill-bg` / `--refuel-pill-border`: nav button colors.
3. Buttons are equally tall (64px), with consistent spacing, to prevent wrapping.
4. The navigation order is defined by `DEFAULT_NAV`; after UX feedback, “Statistics” precedes “What-if Lab”, and “Scenario Lab” (What-if page) was placed near the end.

Sidebar (`sidebar_info_block`) shows the logo, tagline, and data-refresh hint. `render_footer` prints credits and a link to the university.

---

## 7. Control-Flow Walkthroughs

### 7.1 Logging a new sale in POS Console

```
User clicks "Log entry"
        │
        ▼
POS form collects inputs
        │
        ├─ compute stock_remaining = max(0, baseline + restock_delta - sum(product_sales))
        │
        └─ payload = {
               "timestamp": timestamp.isoformat(),
               "sales_units": logged_units,
               "checkins_recorded": logged_checkins,
               "stock_remaining": stock_remaining,
               "notes": notes,
               "product_breakdown": {sku: qty, ...}
           }
                │
                ▼
append_pos_log(payload)  ──▶ writes to data/pos_log.csv
                │
                ▼
load_pos_log() ──▶ recent_entries table + latest_stock metric
```

*If `should_auto_restock(stock_remaining, policy)` returns True, the console immediately appends another entry with `sales_units=0`, updates the policy via `mark_auto_restock`, and surfaces a warning banner.*

### 7.2 Forecast + reorder plan sequence

1. Dashboard reads scenario inputs from sidebar.
2. Calls `build_scenario_forecast` → returns hourly predictions for the next `horizon_hours`.
3. `_latest_stock_reading()` fetches the most recent `stock_remaining` from `pos_log.csv`.
4. `_estimate_daily_demand()` averages historical snack units; if insufficient data, it extrapolates from the forecast by deriving the average hourly prediction and scaling to 24 hours.
5. `render_model_reorder_plan` computes cumulative forecast consumption, subtracts it from `current_stock`, and identifies:
   - `reorder_ts`: first timestamp when projected stock <= safety buffer.
   - `depletion_ts`: first timestamp when projected stock <= 0.
6. Metrics and a Plotly area chart visualize these thresholds.

### 7.3 Autopilot tick (Forecasts page)

```
Button toggles st.session_state["autopilot_running"] = True
        │
        ▼ (Streamlit reruns repeatedly while running)
autopilot_should_step() checks cooldown (1 sec default)
        │
        ├─ If time to step:
        │     advance_autopilot_block(base_history, models, autop_history, ...)
        │         ├─ build_scenario_forecast(...) with future anchor timestamp
        │         ├─ run_auto_simulation(...) to produce plan chunk
        │         └─ append plan chunk to autop_history
        │
        └─ Updated history saved via save_autopilot_history_file()
```

This pattern shows how Streamlit session state is used to emulate background loops.

---

## 8. Extensibility Guide

### 7.1 Adding a new dataset

1. Drop a CSV in `data/`.
2. Implement `load_new_dataset()` in `services/data_utils.py`.
   ```python
   def load_class_schedule() -> pd.DataFrame:
       path = DATA_DIR / "class_schedule.csv"
       if not path.exists():
           return pd.DataFrame()
       df = pd.read_csv(path)
       df["timestamp"] = pd.to_datetime(df["timestamp"])
       return df
   ```
3. Import the loader inside any page:
   ```python
   from frontend.streamlit_app.services.data_utils import load_class_schedule
   ```
4. Use the DataFrame as needed (Plotly charts, merges, etc.).

### 7.2 Creating a new page

1. Copy an existing page, rename (e.g., `pages/9_NewFeature.py`).
2. At minimum include:
   ```python
   from frontend.streamlit_app.components.layout import render_top_nav, render_footer

   render_top_nav("9_NewFeature.py")
   st.title("My Experiment")
   render_footer()
   ```
3. Add a `NavItem` entry to `DEFAULT_NAV` so it appears in the nav bar.

### 7.3 Altering forecasting logic

*Key hooks:*

- `train_models` – change algorithms, add features, or integrate external libraries (Prophet, ARIMA, etc.).
- `build_scenario_forecast` – modify how scenario sliders influence the feature matrix. All scenario keys are optional; extra keys are safely ignored.
- `render_model_reorder_plan` – adjust reorder recommendations logic as business rules evolve.

### 7.4 Persisting new settings

1. Create a JSON file under `data/` (e.g., `DATA_DIR / "notifications.json"`).
2. Add `load_notifications()` / `save_notifications()` helpers in `data_utils.py`.
3. Use them in a page, following the pattern used for restock policy.

---

## 9. Implementation Practices

1. **Session state** – Use `st.session_state` for user-facing toggles that must persist across reruns (e.g., autopilot running flag, inventory simulator state).
2. **Early exits** – When required files are missing, prefer `st.error` + `st.stop()` so the page fails gracefully.
3. **CSV writes** – Always go through the helper functions; do not call `to_csv` directly from pages because validation lives in the helpers.
4. **Logging** – Streamlit displays errors inline; consider adding `st.warning` when assumptions are violated (e.g., no telemetry yet).
5. **Testing** – While there is no automated test suite, pure functions in `data_utils.py` can be unit tested in the future; keep them deterministic and side-effect free when possible.

---

## 10. Future Roadmap Ideas

1. **Database backend** – Swap CSVs for SQLite or TinyDB to support concurrent editing and reduce merge conflicts.
2. **API separation** – Move forecasting/autopilot logic to `backend/` and expose endpoints; Streamlit would become a pure client.
3. **Notification engine** – Convert reorder guidance into scheduled alerts (email/Slack).
4. **Multi-theme support** – Add a user control for light/dark themes while keeping white-friendly charts as default.
5. **Model governance** – Track model versions, training metrics, and feature importance for reproducibility.

---

## 11. Glossary

| Term | Explanation |
|------|-------------|
| **Telemetry** | Combined dataset of check-ins, snack sales, revenue, and weather. |
| **Scenario** | A what-if configuration (weather pattern, temp shift, marketing boost, price change). |
| **Autopilot** | Simulation that steps through future days, logging restocks and profit under specific policies. |
| **POS log** | `data/pos_log.csv`; append-only ledger of manual and auto restock events. |
| **Safety stock** | Buffer inventory threshold used by the reorder planner to warn before stockouts. |

---

## Appendix A – Function Index by File

### A.1 `frontend/streamlit_app/pages/1_Dashboard.py`

- `render_summary_cards(df)` – Builds four `st.metric` cards from the last 24 rows.
- `render_history_charts(df, window_days)` – Generates two Plotly figures (usage/check-ins and weather trend).
- `render_weather_shotcast()` – Emits the Windy iframe HTML using saved lat/lon.
- `render_forecast_section(history, forecast)` – Aligns 48 hours of history with forecast rows and draws combined traces.
- `_latest_stock_reading()` – Loads POS log and returns the most recent numeric `stock_remaining`.
- `_estimate_daily_demand(history, forecast)` – Calculates a rolling average (14 days) and falls back to forecast-derived hourly averages.
- `render_model_reorder_plan(history, forecast)` – Houses reorder logic, safety-buffer calculations, and chart rendering.

### A.2 `frontend/streamlit_app/pages/7_POS_Console.py`

- Uses `_slugify` helper to build deterministic widget keys for SKUs.
- POS form is wrapped in `st.form("pos-entry")` to avoid partial writes.
- After submission, `append_pos_log` is called, followed by reloads of `log_df` and `restock_policy`.
- Auto restock path: evaluate `should_auto_restock`, log synthetic entry, call `mark_auto_restock`, show warning.
- Forecast comparison: drop null timestamps → `pd.merge_asof` with `tolerance=pd.Timedelta("2h")` → bar chart of actual vs. forecast.

### A.3 `frontend/streamlit_app/pages/8_Price_Manager.py`

- Computes `known_products` as the union of mix + price sheets, defaulting to price-only list if mix is empty.
- `st.data_editor` uses `st.column_config.TextColumn` and `NumberColumn` to enforce required fields and min/max rules.
- “Save prices” writes via `save_product_prices`; “Reset” builds a DataFrame filled with `DEFAULT_PRODUCT_PRICE`.
- Add/remove buttons call `add_or_update_product_price` / `remove_product_price` and trigger `st.rerun()` to refresh the grid.

### A.4 `frontend/streamlit_app/components/layout.py`

- `NavItem` dataclass defines `[label, emoji, path]`.
- `render_top_nav` builds columns, injects CSS, and renders one button per nav item.
- `_inject_theme_css` sets CSS variables (`--refuel-surface`, `--refuel-pill-bg`, etc.) and styles the pill buttons (height, spacing, hover state).
- `_ensure_tooltip_css` + `hover_tip` provide a reusable tooltip label component.

### A.5 `frontend/streamlit_app/services/data_utils.py`

- **CSV loaders**: `load_enriched_data`, `load_product_mix_data`, `load_product_prices`, `load_pos_log`, `load_restock_policy`, `load_autopilot_history_file`, etc.
- **Forecasting**: `train_models`, `_prepare_features`, `build_scenario_forecast`.
- **POS + policy**: `append_pos_log`, `_safe_load_breakdown`, `should_auto_restock`, `mark_auto_restock`.
- **Autopilot**: `run_auto_simulation`, `advance_autopilot_block`, `autopilot_should_step`.
- **Price management**: `save_product_prices`, `add_or_update_product_price`, `remove_product_price`.

---

Keep this guide up to date. Whenever you change a major flow or introduce new datasets/features, update the relevant sections so future contributors do not need to reverse-engineer the design.
