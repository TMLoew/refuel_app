from pathlib import Path
import sys
import json
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple
import time

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from streamlit.runtime.secrets import StreamlitSecretNotFoundError

from frontend.streamlit_app.components.layout import (
    render_top_nav,
    sidebar_info_block,
    render_footer,
    get_logo_path,
)
MODEL_DIR = ROOT_DIR / "model"
CHECKIN_MODEL_FILE = MODEL_DIR / "checkins_hgb.joblib"
SNACK_MODEL_FILE = MODEL_DIR / "snacks_hgb.joblib"
try:
    from frontend.streamlit_app.services.data_utils import (
        load_enriched_data,
        load_pos_log,
        load_weather_profile,
        save_weather_profile,
        build_scenario_forecast,
        train_models,
        save_procurement_plan,
        WEATHER_SCENARIOS,
    )
except ImportError:
    from frontend.streamlit_app.services.data_utils import load_enriched_data, load_pos_log  # type: ignore

    def load_weather_profile() -> dict:  # type: ignore[misc]
        return {"lat": 47.4239, "lon": 9.3748, "api_timeout": 10, "cache_hours": 6}
    def save_weather_profile(profile: dict) -> None:  # type: ignore[misc]
        pass
    def build_scenario_forecast(*_args, **_kwargs):  # type: ignore[misc]
        return pd.DataFrame()
    def train_models(*_args, **_kwargs):  # type: ignore[misc]
        return (None, None)
    def save_procurement_plan(*_args, **_kwargs):  # type: ignore[misc]
        return
    WEATHER_SCENARIOS = {}

AUTOPILOT_STATE_FILE = ROOT_DIR / "data" / "autopilot_infinite.csv"


def load_autopilot_history_file() -> pd.DataFrame:
    if not AUTOPILOT_STATE_FILE.exists():
        return pd.DataFrame()
    df = pd.read_csv(AUTOPILOT_STATE_FILE)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df


def save_autopilot_history_file(history_df: pd.DataFrame) -> None:
    AUTOPILOT_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    export = history_df.copy()
    if not export.empty and "date" in export.columns:
        export["date"] = pd.to_datetime(export["date"]).dt.strftime("%Y-%m-%d")
    export.to_csv(AUTOPILOT_STATE_FILE, index=False)


def reset_autopilot_history_file() -> None:
    if AUTOPILOT_STATE_FILE.exists():
        AUTOPILOT_STATE_FILE.unlink()


def autopilot_anchor_timestamp(base_history: pd.DataFrame, autop_history: pd.DataFrame) -> pd.Timestamp:
    if autop_history is not None and not autop_history.empty and "date" in autop_history.columns:
        last_day = pd.to_datetime(autop_history["date"].max())
        return last_day + pd.Timedelta(hours=23)
    return base_history["timestamp"].max()


def autopilot_should_step(step_interval_seconds: float = 1.0) -> bool:
    if not st.session_state.get("autopilot_running"):
        return False
    force = st.session_state.pop("autopilot_force_step", False)
    last_tick = st.session_state.get("autopilot_last_tick")
    now = datetime.now(timezone.utc)
    if last_tick is not None and last_tick.tzinfo is None:
        last_tick = last_tick.replace(tzinfo=timezone.utc)
    if force or last_tick is None or (now - last_tick).total_seconds() >= step_interval_seconds:
        st.session_state["autopilot_last_tick"] = now
        return True
    return False


def run_auto_simulation(
    forecast_hours: pd.DataFrame,
    starting_stock: float,
    safety_stock: float,
    reorder_qty: float,
    unit_cost: float,
    fee: float,
    price_strategy_pct: float,
    scenario_label: str,
) -> pd.DataFrame:
    if forecast_hours.empty:
        return pd.DataFrame()

    forecast = forecast_hours.copy()
    forecast["timestamp"] = pd.to_datetime(forecast["timestamp"])
    daily = (
        forecast.assign(date=forecast["timestamp"].dt.date)
        .groupby("date")
        .agg(
            temperature_c=("temperature_c", "mean"),
            snack_price=("snack_price", "mean"),
            demand=("pred_snack_units", "sum"),
            checkins=("pred_checkins", "sum"),
        )
        .reset_index()
    )
    if daily.empty:
        return pd.DataFrame()

    base_price = float(daily["snack_price"].mean())
    temp_mean = float(daily["temperature_c"].mean())
    price_min = base_price * 0.8 if base_price else 1.5
    price_max = base_price * 1.25 if base_price else 5.0

    rows = []
    stock = starting_stock
    for _, row in daily.iterrows():
        temp_bias = 1 + 0.008 * (row["temperature_c"] - temp_mean)
        target_price = float(
            np.clip(row["snack_price"] * temp_bias * (1 + price_strategy_pct / 100), price_min, price_max)
        )
        demand_adj = max(0.0, row["demand"])
        stock_before = stock
        sold = min(stock_before, demand_adj)
        stock_after = stock_before - sold
        profit = (target_price - unit_cost - fee) * sold
        reordered = ""
        reorder_qty_used = 0.0
        if stock_after <= safety_stock:
            stock_after += reorder_qty
            reordered = "Yes"
            reorder_qty_used = reorder_qty
        rows.append(
            {
                "date": pd.to_datetime(row["date"]),
                "scenario": scenario_label,
                "checkins_est": round(row["checkins"], 1),
                "temperature_c": round(row["temperature_c"], 1),
                "price": round(target_price, 2),
                "demand_est": round(demand_adj, 1),
                "sold": round(sold, 1),
                "profit": round(profit, 2),
                "stock_before": round(stock_before, 1),
                "stock_after": round(stock_after, 1),
                "reordered": reordered,
                "reorder_qty": reorder_qty_used,
            }
        )
        stock = stock_after
    return pd.DataFrame(rows)


def advance_autopilot_block(
    base_history: pd.DataFrame,
    models: Tuple,
    autop_history: pd.DataFrame,
    scenario_payload: Dict[str, float],
    current_stock: float,
    safety_stock: float,
    reorder_qty: float,
    auto_unit_cost: float,
    auto_fee: float,
    price_strategy_pct: float,
    sales_boost_pct: float,
    step_days: int,
    lead_time_days: int,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[str], Optional[str]]:
    scenario = dict(scenario_payload)
    scenario["horizon_hours"] = step_days * 24
    anchor_ts = autopilot_anchor_timestamp(base_history, autop_history)
    forecast_hours = build_scenario_forecast(
        base_history,
        models,
        scenario,
        anchor_timestamp=anchor_ts,
    )
    if forecast_hours.empty:
        return autop_history, None, None, "Forecast pipeline returned no data. Adjust the scenario and try again."
    if sales_boost_pct:
        forecast_hours["pred_snack_units"] *= 1 + sales_boost_pct / 100

    auto_df = run_auto_simulation(
        forecast_hours=forecast_hours,
        starting_stock=current_stock,
        safety_stock=safety_stock,
        reorder_qty=reorder_qty,
        unit_cost=auto_unit_cost,
        fee=auto_fee,
        price_strategy_pct=price_strategy_pct,
        scenario_label=scenario["weather_pattern"],
    )
    if auto_df.empty:
        return autop_history, None, None, "Simulation failed; need more historical telemetry."

    block_id = (
        1 if autop_history.empty or "sim_block" not in autop_history.columns else int(autop_history["sim_block"].max()) + 1
    )
    plan_id = datetime.now(timezone.utc).isoformat(timespec="seconds")
    auto_df["sim_block"] = block_id
    auto_df["sales_boost_pct"] = sales_boost_pct
    auto_df["plan_generated_at"] = plan_id
    scenario_metadata = {
        "plan_weather_pattern": scenario["weather_pattern"],
        "plan_marketing_boost_pct": f"{scenario.get('marketing_boost_pct', 0)}",
        "plan_price_change_pct": f"{scenario.get('snack_price_change', 0)}",
        "plan_price_strategy_pct": f"{price_strategy_pct}",
        "plan_unit_cost": f"{auto_unit_cost:.2f}",
        "plan_fee": f"{auto_fee:.2f}",
        "plan_horizon_days": f"{step_days}",
        "plan_lead_time_days": f"{lead_time_days}",
        "plan_safety_stock": f"{safety_stock:.1f}",
        "plan_reorder_qty": f"{reorder_qty:.1f}",
        "plan_temp_manual": f"{scenario.get('temp_manual', 0)}",
        "plan_precip_manual": f"{scenario.get('precip_manual', 0)}",
        "plan_event_intensity": f"{scenario.get('event_intensity', 1.0)}",
        "plan_sales_boost_pct": f"{sales_boost_pct}",
        "plan_block_id": f"{block_id}",
    }
    for key, val in scenario_metadata.items():
        auto_df[key] = val

    save_procurement_plan(auto_df, metadata=scenario_metadata)
    st.session_state["auto_results"] = auto_df
    autop_history = pd.concat([autop_history, auto_df], ignore_index=True)
    save_autopilot_history_file(autop_history)
    summary = (
        f"Advanced {step_days} day(s). Ending stock {auto_df['stock_after'].iloc[-1]:.0f} units · "
        f"profit CHF{auto_df['profit'].sum():.0f}."
    )
    return autop_history, auto_df, summary, None


def render_autopilot_panel(data: pd.DataFrame, use_live_weather: bool) -> None:
st.subheader("Weather-aware autopilot")
    if data.empty:
        st.warning("Need telemetry to simulate autopilot scenarios. Upload a CSV first.")
        return
    models = train_models(data)
    if not models or models[0] is None or models[1] is None:
        st.warning("Need more data to train the forecast models. Upload more rows, then try again.")
        return

    daily_summary = (
        data.set_index("timestamp")
        .resample("D")
        .agg({"checkins": "sum", "snack_units": "sum", "temperature_c": "mean", "snack_price": "mean"})
        .reset_index()
    )
    daily_summary["date"] = daily_summary["timestamp"].dt.date
    avg_units = float(data["snack_units"].mean())
    unit_cost_default = round(float(data["snack_price"].mean()) * 0.6, 2)
    operating_fee_default = 0.2
    derived_conversion = float(data["snack_units"].sum()) / max(float(data["checkins"].sum()), 1.0)
    derived_conversion = float(np.clip(derived_conversion, 0.05, 0.9))
    auto_step_days = 1
    lead_time_auto = 7
    service_factor = 1.65
    demand_std = float(daily_summary["snack_units"].std() or avg_units * 0.1)
    mean_checkins = float(daily_summary["checkins"].mean())
    lead_time_demand = derived_conversion * mean_checkins * lead_time_auto
    safety_auto = max(0.0, lead_time_demand + service_factor * demand_std * np.sqrt(lead_time_auto))
    reorder_qty_auto = safety_auto + lead_time_demand
    starting_auto = reorder_qty_auto * 2

    auto_unit_cost = st.number_input(
        "Sim unit cost (CHF)", min_value=0.1, value=unit_cost_default, step=0.1, key="auto-unit-cost-settings"
    )
    auto_fee = st.slider(
        "Sim per-transaction fee (CHF)", 0.0, 2.0, operating_fee_default, step=0.1, key="auto-fee-settings"
    )

    scenario_cols = st.columns(2)
    weather_pattern = scenario_cols[0].selectbox(
        "Weather pattern", list(WEATHER_SCENARIOS.keys()) or ["Temperate"], key="auto-weather-settings"
    )
    marketing_boost = scenario_cols[1].slider("Marketing boost (%)", 0, 80, 10, key="auto-marketing-settings")

    manual_cols = st.columns(3)
    temp_manual = manual_cols[0].slider("Manual temp shift (°C)", -8, 8, 0, key="auto-temp-settings")
    precip_manual = manual_cols[1].slider(
        "Manual precipitation shift (mm)", -3.0, 3.0, 0.0, step=0.1, key="auto-precip-settings"
    )
    event_intensity = manual_cols[2].slider("Event intensity", 0.2, 2.5, 1.0, step=0.1, key="auto-event-settings")

    price_change = st.slider("Baseline price change (%)", -20, 25, 0, key="auto-price-change-settings")
    price_strategy = st.slider("Dynamic price aggressiveness (%)", -10, 15, 0, key="auto-price-strategy-settings")
    sales_boost_pct = st.slider(
        "Sales boost (%)",
        min_value=0,
        max_value=200,
        value=0,
        step=5,
        key="auto-sales-boost-settings",
        help="Apply an extra uplift to snack demand during the infinite simulation.",
    )

    if "autopilot_history" not in st.session_state:
        st.session_state["autopilot_history"] = load_autopilot_history_file()
    if "autopilot_running" not in st.session_state:
        st.session_state["autopilot_running"] = False

    autop_history = st.session_state["autopilot_history"]
    current_stock = starting_auto if autop_history.empty else float(autop_history["stock_after"].iloc[-1])
    cola, colb, colc = st.columns(3)
    cola.metric("Derived conversion", f"{derived_conversion:.2f}")
    colb.metric("Recommended safety stock", f"{safety_auto:.0f} units")
    colc.metric("Recommended reorder qty", f"{reorder_qty_auto:.0f} units")

    scenario_payload = {
        "weather_pattern": weather_pattern,
        "temp_manual": temp_manual,
        "precip_manual": precip_manual,
        "event_intensity": event_intensity,
        "marketing_boost_pct": marketing_boost,
        "snack_price_change": price_change,
        "use_live_weather": use_live_weather,
    }

    action_cols = st.columns([0.4, 0.3, 0.3])
    play_clicked = action_cols[0].button("▶️ Play / Advance", key="auto-play-settings", use_container_width=True)
    pause_clicked = action_cols[1].button(
        "⏸ Pause",
        key="auto-pause-settings",
        use_container_width=True,
        disabled=not st.session_state["autopilot_running"],
    )
    reset_clicked = action_cols[2].button("♻️ Reset", key="auto-reset-settings", use_container_width=True)

    if play_clicked:
        st.session_state["autopilot_running"] = True
        st.session_state["autopilot_force_step"] = True
        st.session_state["autopilot_last_tick"] = None

    if pause_clicked:
        st.session_state["autopilot_running"] = False
        st.info("Autopilot paused. Press Play to continue generating future days.")

    if reset_clicked:
        reset_autopilot_history_file()
        st.session_state["autopilot_history"] = pd.DataFrame()
        st.session_state["autopilot_running"] = False
        st.session_state.pop("auto_results", None)
        st.success("Autopilot state reset. You're back at the starting conditions.")
        st.rerun()

    should_step = autopilot_should_step()
    if should_step:
        autop_history, _, summary, error = advance_autopilot_block(
            base_history=data,
            models=models,
            autop_history=autop_history,
            scenario_payload=scenario_payload,
            current_stock=current_stock,
            safety_stock=safety_auto,
            reorder_qty=reorder_qty_auto,
            auto_unit_cost=auto_unit_cost,
            auto_fee=auto_fee,
            price_strategy_pct=price_strategy,
            sales_boost_pct=sales_boost_pct,
            step_days=auto_step_days,
            lead_time_days=lead_time_auto,
        )
        st.session_state["autopilot_history"] = autop_history
        if error:
            st.session_state["autopilot_running"] = False
            st.warning(error)
        else:
            st.session_state["autopilot_last_summary"] = summary
            if st.session_state.get("autopilot_running"):
                time.sleep(1.0)
                st.rerun()

    autop_history = st.session_state["autopilot_history"]
    autop_status = "Running" if st.session_state["autopilot_running"] else "Paused"
    st.caption(
        f"Status: **{autop_status}** · data is saved to `{AUTOPILOT_STATE_FILE.name}` so you can reopen it later."
    )

    if autop_history.empty:
        auto_df_display = st.session_state.get("auto_results")
        if isinstance(auto_df_display, pd.DataFrame) and not auto_df_display.empty:
            auto_df_display = auto_df_display.copy()
            auto_df_display["date"] = pd.to_datetime(auto_df_display["date"])
            metrics_cols = st.columns(3)
            metrics_cols[0].metric("Simulation days", f"{len(auto_df_display):.0f}")
            if "profit" in auto_df_display.columns:
                metrics_cols[1].metric("Plan profit", f"CHF{auto_df_display['profit'].sum():.0f}")
            else:
                metrics_cols[1].metric("Plan profit", "n/a")
            metrics_cols[2].metric("Ending stock", f"{auto_df_display['stock_after'].iloc[-1]:.0f} units")
            auto_fig = px.line(auto_df_display, x="date", y="stock_after", title="Latest plan trajectory")
            auto_fig.add_hline(y=safety_auto, line_dash="dot", line_color="orange", annotation_text="Safety stock")
            reorder_points = auto_df_display[auto_df_display["reordered"] == "Yes"]
            if not reorder_points.empty:
                auto_fig.add_scatter(
                    x=reorder_points["date"],
                    y=reorder_points["stock_after"],
                    mode="markers",
                    marker=dict(color="green", size=10),
                    name="Reorders",
                )
            st.plotly_chart(auto_fig, use_container_width=True)
            st.dataframe(
                auto_df_display[
                    ["date", "scenario", "price", "demand_est", "sold", "stock_after", "reordered", "reorder_qty", "profit"]
                ],
                use_container_width=True,
                height=320,
            )
        else:
            st.info("No autopilot history yet. Press Play to generate the first block of days.")
    else:
        history_view = autop_history.copy()
        history_view["date"] = pd.to_datetime(history_view["date"])
        auto_fig = px.line(history_view, x="date", y="stock_after", title="Autopilot stock trajectory (infinite run)")
        auto_fig.add_hline(y=safety_auto, line_dash="dot", line_color="orange", annotation_text="Safety stock")
        reorder_points = history_view[history_view["reordered"] == "Yes"]
        if not reorder_points.empty:
            auto_fig.add_scatter(
                x=reorder_points["date"],
                y=reorder_points["stock_after"],
                mode="markers",
                marker=dict(color="green", size=10),
                name="Reorders",
            )
        st.plotly_chart(auto_fig, use_container_width=True)
        st.dataframe(
            history_view[
                ["date", "scenario", "price", "demand_est", "sold", "stock_after", "reordered", "reorder_qty", "profit"]
            ],
            use_container_width=True,
            height=320,
        )
        download_blob = history_view.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download autopilot history (CSV)",
            download_blob,
            file_name="autopilot_infinite_history.csv",
            mime="text/csv",
        )
        last_summary = st.session_state.get("autopilot_last_summary")
        if last_summary:
            st.caption(f"Last tick · {last_summary}")
PAGE_ICON = get_logo_path() or "⚙️"
st.set_page_config(page_title="Settings & APIs", page_icon=PAGE_ICON, layout="wide")

render_top_nav("5_Settings_APIs.py")
st.title("Settings & API Console")
st.caption("Manage weather settings, quick health checks, and simple automation controls.")
active_env = "Default"

with st.sidebar:
    sidebar_info_block()

st.subheader("Weather API configuration")
profile = load_weather_profile()
with st.form("weather-form"):
    col1, col2 = st.columns(2)
    with col1:
        lat = st.number_input("Latitude", value=float(profile["lat"]), step=0.1, format="%.4f")
        api_timeout = st.number_input("Timeout (seconds)", min_value=1, max_value=60, value=10)
    with col2:
        lon = st.number_input("Longitude", value=float(profile["lon"]), step=0.1, format="%.4f")
        cache_hours = st.slider("Cache horizon (hours)", 1, 24, int(profile.get("cache_hours", 6)))
    submitted = st.form_submit_button("Save weather profile", use_container_width=True)
    if submitted:
        save_weather_profile({"lat": float(lat), "lon": float(lon), "api_timeout": api_timeout, "cache_hours": cache_hours})
        st.success(
            f"Saved weather coordinates ({lat:.4f}, {lon:.4f}) with timeout={api_timeout}s and cache={cache_hours}h."
        )

st.subheader("API health")
data_sample = load_enriched_data(use_weather_api=True)
weather_meta = data_sample.attrs.get("weather_meta", {})
latency = weather_meta.get("latency_ms")
if latency:
    weather_value = "✅ OK"
    weather_delta = f"latency {latency:.0f} ms"
elif weather_meta:
    weather_value = "ℹ️ Cached"
    weather_delta = "using cached weather"
else:
    weather_value = "ℹ️ Pending"
    weather_delta = "no API calls yet"

# Model lifecycle
st.subheader("Forecast models")
st.caption("Retrain the saved models on the current dataset (overwrites the joblib files).")
retrain_cols = st.columns([0.4, 0.6])
with retrain_cols[0]:
    retrain_clicked = st.button("🔄 Retrain models on current data", type="primary")
with retrain_cols[1]:
    st.caption(f"Model files: `{CHECKIN_MODEL_FILE.name}`, `{SNACK_MODEL_FILE.name}`")

if retrain_clicked:
    data_for_training = load_enriched_data(use_weather_api=True, cache_buster=pd.Timestamp.utcnow().timestamp())
    if data_for_training.empty:
        st.error("No data available to train. Upload telemetry first.")
    else:
        # Clear persisted models and cached resource to force fresh training
        for model_path in (CHECKIN_MODEL_FILE, SNACK_MODEL_FILE):
            try:
                if model_path.exists():
                    model_path.unlink()
            except Exception as exc:  # pragma: no cover - streamlit interaction
                st.warning(f"Could not remove {model_path.name}: {exc}")
        try:
            train_models.clear()  # type: ignore[attr-defined]
        except Exception:
            pass
        with st.spinner("Training attendance and snack models..."):
            try:
                models = train_models(data_for_training)
            except Exception as exc:  # pragma: no cover - streamlit interaction
                st.error(f"Training failed: {exc}")
                models = (None, None)
        if models and all(models):
            st.success("Models retrained and saved. Forecast pages will use the new fit.")
        else:
            st.warning("Models were not produced. Check logs and dataset completeness.")

now_utc = pd.Timestamp.now(timezone.utc)
if data_sample.empty or "timestamp" not in data_sample.columns:
    gym_value = "ℹ️ No data"
    gym_delta = "upload telemetry"
else:
    latest_gym_ts = pd.to_datetime(data_sample["timestamp"].max())
    if latest_gym_ts.tzinfo is None:
        latest_gym_ts = latest_gym_ts.tz_localize("UTC")
    gym_delay_min = max(0, (now_utc - latest_gym_ts).total_seconds() / 60)
    if gym_delay_min <= 5:
        gym_value = "✅ Fresh"
    elif gym_delay_min <= 30:
        gym_value = "⚠️ Delay"
    else:
        gym_value = "❌ Stale"
    gym_delta = f"{gym_delay_min:.0f} min old" if gym_delay_min < 90 else f"{gym_delay_min/60:.1f} h old"

pos_log = load_pos_log()
if pos_log.empty:
    pos_value = "ℹ️ No data"
    pos_delta = "no POS events"
else:
    latest_pos_ts = pd.to_datetime(pos_log["timestamp"].max())
    if latest_pos_ts.tzinfo is None:
        latest_pos_ts = latest_pos_ts.tz_localize("UTC")
    pos_delay_min = max(0, (now_utc - latest_pos_ts).total_seconds() / 60)
    if pos_delay_min <= 10:
        pos_value = "✅ Live"
    elif pos_delay_min <= 60:
        pos_value = "⚠️ Delay"
    else:
        pos_value = "❌ Stale"
    pos_delta = f"{pos_delay_min:.0f} min old" if pos_delay_min < 120 else f"{pos_delay_min/60:.1f} h old"

health_cols = st.columns(3)
health_cols[0].metric("Weather API", weather_value, delta=weather_delta)
health_cols[1].metric("Gym sensors", gym_value, delta=gym_delta)
health_cols[2].metric("POS snacks", pos_value, delta=pos_delta)

st.subheader("Secrets & tokens")
with st.expander("Current tokens (redacted)", expanded=False):
    token_store = {
        "open_meteo": "sk-***meteo",
        "gym_webhook": "whsec-***123",
        "pos_service": "pat-***refuel",
        "last_rotation": datetime.now(timezone.utc).isoformat(),
    }
    st.json(token_store)
    st.info("Manage actual secrets via your deployment platform; this panel is a placeholder for ops runbooks.")

st.subheader("Webhooks")
webhooks = [
    {"name": "Gym turnstiles", "status": "active", "last_event": "2 min ago"},
    {"name": "Snack POS", "status": "active", "last_event": "5 min ago"},
    {"name": "Marketing automation", "status": "paused", "last_event": "3 days ago"},
]
st.table(webhooks)

st.subheader("Export settings")
export_blob = json.dumps({"env": active_env, "lat": float(lat), "lon": float(lon)}, indent=2)
st.download_button("Download config JSON", export_blob, file_name="refuel_config.json", mime="application/json")

st.subheader("Advanced automation (restricted)")
try:
    ops_secret = st.secrets.get("OPS_PASSWORD") or st.secrets.get("ops_password")
except StreamlitSecretNotFoundError:
    ops_secret = None

if not ops_secret:
    st.info("Set OPS_PASSWORD in `.streamlit/secrets.toml` to unlock the procurement autopilot controls.")
else:
    entered_pw = st.text_input("Ops password", type="password")
    if entered_pw == ops_secret:
        render_autopilot_panel(data_sample, use_live_weather=True)
    elif entered_pw:
        st.error("Incorrect password.")

render_footer()
