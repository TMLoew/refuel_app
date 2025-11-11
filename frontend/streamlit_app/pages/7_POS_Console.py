from pathlib import Path
import sys
from datetime import datetime, time, timedelta
from typing import Dict

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import pandas as pd
import plotly.express as px
import streamlit as st

def _slugify(label: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "-" for ch in label)

DEFAULT_WEATHER_SCENARIOS = {
    "Temperate & sunny": {"temp_offset": 2.0, "precip_multiplier": 0.7, "humidity_offset": -3},
    "Cold snap": {"temp_offset": -6.0, "precip_multiplier": 1.0, "humidity_offset": 4},
    "Humid heatwave": {"temp_offset": 5.5, "precip_multiplier": 0.4, "humidity_offset": 8},
    "Storm front": {"temp_offset": -1.0, "precip_multiplier": 1.8, "humidity_offset": 10},
}

DEFAULT_RESTOCK_POLICY = {
    "auto_enabled": False,
    "threshold_units": 40,
    "lot_size": 50,
    "cooldown_hours": 6,
    "last_auto_restock": None,
}

try:
    from frontend.streamlit_app.components.layout import (
        render_top_nav,
        sidebar_info_block,
        render_footer,
        get_logo_path,
    )
    from frontend.streamlit_app.services import data_utils as _du
except ModuleNotFoundError:  # pragma: no cover
    from components.layout import render_top_nav, sidebar_info_block, render_footer, get_logo_path
    from services import data_utils as _du


def _fallback_policy() -> Dict:
    return DEFAULT_RESTOCK_POLICY.copy()


WEATHER_SCENARIOS = getattr(_du, "WEATHER_SCENARIOS", DEFAULT_WEATHER_SCENARIOS)
append_pos_log = getattr(_du, "append_pos_log")
build_scenario_forecast = getattr(_du, "build_scenario_forecast", lambda *_args, **_kwargs: pd.DataFrame())
load_enriched_data = getattr(_du, "load_enriched_data", lambda *_args, **_kwargs: pd.DataFrame())
load_product_mix_data = getattr(_du, "load_product_mix_data", lambda *_args, **_kwargs: pd.DataFrame())
load_pos_log = getattr(_du, "load_pos_log", lambda: pd.DataFrame())
load_restock_policy = getattr(_du, "load_restock_policy", _fallback_policy)
mark_auto_restock = getattr(_du, "mark_auto_restock", lambda policy: policy)
save_restock_policy = getattr(_du, "save_restock_policy", lambda *_args, **_kwargs: None)
should_auto_restock = getattr(_du, "should_auto_restock", lambda *_args, **_kwargs: False)
get_product_catalog = getattr(_du, "get_product_catalog", lambda *_args, **_kwargs: [])
train_models = getattr(_du, "train_models", lambda *_args, **_kwargs: (None, None))

PAGE_ICON = get_logo_path() or "🧾"
st.set_page_config(page_title="POS Console", page_icon=PAGE_ICON, layout="wide")

render_top_nav("7_POS_Console.py")
st.title("POS Console")
st.caption("Log live sales + check-ins at the counter and receive instant low stock alerts & forecasts.")

with st.sidebar:
    sidebar_info_block()
    use_weather_api = st.toggle("Use live weather API", value=False, key="pos-weather")
    scenario_name = st.selectbox("Weather scenario", list(WEATHER_SCENARIOS.keys()), key="pos-weather-pattern")
    horizon_hours = st.slider("Forecast horizon (hours)", 3, 24, 12, step=3)

with st.spinner("Loading telemetry & models..."):
    base_data = load_enriched_data(use_weather_api=use_weather_api)
if base_data.empty:
    st.error("Need telemetry data in `data/gym_badges.csv` (or `*_long.csv`).")
    st.stop()

models = train_models(base_data)
log_df = load_pos_log()
product_mix_df = load_product_mix_data()
product_catalog = get_product_catalog(product_mix_df)
restock_policy = load_restock_policy()
latest_stock = None
if not log_df.empty and "stock_remaining" in log_df.columns:
    latest_stock = int(
        round(log_df.sort_values("timestamp", ascending=False)["stock_remaining"].iloc[0])
    )
    st.session_state["pos_stock_tip_dismissed"] = True

col_form, col_alert = st.columns([0.6, 0.4])

with col_form:
    st.subheader("Log POS activity")
    if latest_stock is None and not st.session_state.get("pos_stock_tip_dismissed"):
        st.info(
            "Heads-up: please enter the current shelf stock for your first POS entry so we can auto-track it afterwards."
        )
    with st.form("pos-entry"):
        now = datetime.now()
        entry_date = st.date_input("Date", value=now.date())
        entry_time = st.time_input("Time", value=time(hour=now.hour, minute=now.minute))
        logged_sales = st.number_input("Snacks sold (units)", min_value=0, value=0, step=1)
        logged_checkins = st.number_input("Gym check-ins recorded", min_value=0, value=0, step=1)
        restock_delta = st.number_input(
            "Units restocked before this entry",
            min_value=0,
            value=0,
            step=1,
            help="If you added stock since the previous entry, capture it here.",
        )
        per_product_sales: Dict[str, int] = {}
        if product_catalog:
            with st.expander("Optional: detail units by product", expanded=False):
                st.caption("Overrides the total snack units input above when filled in.")
                for product in product_catalog:
                    widget_key = f"pos-sales-{_slugify(product)}"
                    per_product_sales[product] = st.number_input(
                        f"{product} units",
                        min_value=0,
                        value=0,
                        step=1,
                        key=widget_key,
                    )
        if latest_stock is None:
            baseline_stock = st.number_input(
                "Current stock on shelf (units)",
                min_value=0,
                value=50,
                step=1,
                help="Needed for the very first log entry so we can track stock going forward.",
            )
        else:
            baseline_stock = latest_stock
            st.caption(
                f"Baseline stock auto-filled from the last reading ({baseline_stock:.0f} units). "
                "We will subtract the sales you log below."
            )
        notes = st.text_input("Notes / special context", "")
        submitted = st.form_submit_button("Log entry", width="stretch")

    if submitted:
        timestamp = datetime.combine(entry_date, entry_time)
        effective_stock_before = baseline_stock + restock_delta
        product_breakdown = {product: qty for product, qty in per_product_sales.items() if qty > 0}
        logged_units = sum(product_breakdown.values()) if product_breakdown else logged_sales
        stock_remaining = int(max(0, effective_stock_before - logged_units))
        append_pos_log(
            {
                "timestamp": timestamp.isoformat(),
                "sales_units": int(logged_units),
                "checkins_recorded": int(logged_checkins),
                "stock_remaining": stock_remaining,
                "notes": notes,
                "product_breakdown": product_breakdown,
            }
        )
        st.success(
            f"Entry captured. Auto-updated shelf stock to {stock_remaining:.0f} units after sales & restocks."
        )
        log_df = load_pos_log()
        restock_policy = load_restock_policy()
        if not log_df.empty and "stock_remaining" in log_df.columns:
            latest_stock = int(
                round(log_df.sort_values("timestamp", ascending=False)["stock_remaining"].iloc[0])
            )
            st.session_state["pos_stock_tip_dismissed"] = True
        if should_auto_restock(stock_remaining, restock_policy):
            lot_size = int(restock_policy.get("lot_size", 50))
            auto_stock = stock_remaining + lot_size
            append_pos_log(
                {
                    "timestamp": datetime.now().isoformat(),
                    "sales_units": 0,
                    "checkins_recorded": 0,
                    "stock_remaining": auto_stock,
                    "notes": f"Auto restock +{lot_size} units (threshold {restock_policy.get('threshold_units')}u)",
                    "product_breakdown": {},
                }
            )
            restock_policy = mark_auto_restock(restock_policy)
            st.warning(
                f"Auto restock triggered (+{lot_size} units). Shelf stock reset to {auto_stock:.0f} units.",
                icon="⚠️",
            )
            log_df = load_pos_log()
            latest_stock = auto_stock
            stock_remaining = auto_stock

    st.subheader("Recent entries")
    recent_entries = log_df.sort_values("timestamp", ascending=False).head(10).copy()
    for col in ("sales_units", "checkins_recorded", "stock_remaining"):
        if col in recent_entries.columns:
            numeric_series = pd.to_numeric(recent_entries[col], errors="coerce")
            recent_entries[col] = numeric_series.round().astype("Int64")
    if "product_breakdown" in recent_entries.columns:
        recent_entries["product_breakdown"] = recent_entries["product_breakdown"].apply(
            lambda entry: ", ".join(f"{name}: {int(qty)}" for name, qty in entry.items())
            if isinstance(entry, dict) and entry
            else ""
        )
    st.dataframe(recent_entries, width="stretch", height=260)

with col_alert:
    st.subheader("Live stock posture")
    shared_reorder_default = st.session_state.get("inventory_reorder_days", 3)
    reorder_days = st.slider(
        "Restock coverage (days)",
        min_value=1,
        max_value=21,
        value=shared_reorder_default,
        key="inventory_reorder_days",
        help="Shared with the dashboard + home planners so all tools use the same buffer.",
    )
    restock_lot = st.number_input(
        "Restock lot size (units)",
        min_value=5,
        value=50,
        step=5,
        key="pos-restock-amount",
    )
    next_restock_date = (datetime.now() + timedelta(days=reorder_days)).date()
    st.metric("Next planned restock", next_restock_date.strftime("%Y-%m-%d"), f"every {reorder_days} d")
    with st.expander("Auto restock policy", expanded=False):
        st.caption("Automatically logs a restock entry when stock dips below the configured floor.")
        auto_enabled = st.checkbox(
            "Enable auto restock",
            value=restock_policy.get("auto_enabled", False),
            key="pos-auto-restock-enabled",
        )
        threshold_units = st.number_input(
            "Trigger threshold (units)",
            min_value=5,
            value=int(restock_policy.get("threshold_units", 40)),
            step=5,
            key="pos-auto-threshold",
        )
        auto_lot = st.number_input(
            "Auto restock lot (units)",
            min_value=5,
            value=int(restock_policy.get("lot_size", 50)),
            step=5,
            key="pos-auto-lot",
        )
        cooldown_hours = st.number_input(
            "Cooldown between auto restocks (hours)",
            min_value=1,
            value=int(restock_policy.get("cooldown_hours", 6)),
            step=1,
            key="pos-auto-cooldown",
        )
        last_auto = restock_policy.get("last_auto_restock", "never")
        st.caption(f"Last auto restock: {last_auto or 'never'}")
        if st.button("Save auto restock policy", key="pos-save-auto-policy"):
            updated_policy = {
                "auto_enabled": auto_enabled,
                "threshold_units": threshold_units,
                "lot_size": auto_lot,
                "cooldown_hours": cooldown_hours,
                "last_auto_restock": restock_policy.get("last_auto_restock"),
            }
            save_restock_policy(updated_policy)
            restock_policy = updated_policy
            st.success("Auto restock policy saved.")
    if st.button("Call restock now", key="pos-restock-btn"):
        if latest_stock is None:
            st.warning("Log an initial stock reading before triggering a restock.")
        else:
            new_stock = latest_stock + restock_lot
            append_pos_log(
                {
                    "timestamp": datetime.now().isoformat(),
                    "sales_units": 0,
                    "checkins_recorded": 0,
                    "stock_remaining": int(new_stock),
                    "notes": f"Manual restock +{restock_lot} units (coverage {reorder_days}d)",
                    "product_breakdown": {},
                }
            )
            st.success(f"Restock captured. Shelf stock now {int(new_stock)} units.")
            log_df = load_pos_log()
            latest_stock = int(
                round(log_df.sort_values("timestamp", ascending=False)["stock_remaining"].iloc[0])
            )
            st.session_state["pos_stock_tip_dismissed"] = True

    daily_usage = (
        base_data.set_index("timestamp")["snack_units"]
        .resample("D")
        .sum()
        .reset_index(name="daily_snacks")
    )
    avg_daily = float(daily_usage["daily_snacks"].mean()) if not daily_usage.empty else 50.0
    safety_floor = max(10.0, avg_daily * 1.5)
    if latest_stock is None:
        st.info("No POS entries yet. Log your first sale above to unlock alerts.")
    else:
        st.metric("Latest stock reading", f"{latest_stock:.0f} units", help="From the most recent POS entry.")
        if latest_stock < safety_floor:
            st.error(
                f"LOW STOCK! Current level {latest_stock:.0f} < recommended floor {safety_floor:.0f}. Trigger a reorder now."
            )
        else:
            st.success(f"Stock is healthy. Recommended floor: {safety_floor:.0f} units.")

st.subheader("Instant forecast")
if models[0] is None or models[1] is None:
    st.warning("Need more telemetry to train models, so forecast is unavailable.")
else:
    scenario = {
        "horizon_hours": horizon_hours,
        "weather_pattern": scenario_name,
        "temp_manual": 0.0,
        "precip_manual": 0.0,
        "event_intensity": 1.0,
        "marketing_boost_pct": 0,
        "snack_price_change": 0,
        "use_live_weather": use_weather_api,
    }
    forecast = build_scenario_forecast(base_data, models, scenario)
    if forecast.empty:
        st.warning("Forecast pipeline returned no data.")
    else:
        forecast["hour"] = forecast["timestamp"].dt.strftime("%H:%M")
        st.metric("Expected snacks (next 2h)", f"{forecast.head(2)['pred_snack_units'].sum():.0f} units")
        st.metric("Expected check-ins (next 2h)", f"{forecast.head(2)['pred_checkins'].sum():.0f}")
        fc_fig = px.bar(
            forecast,
            x="hour",
            y="pred_snack_units",
            title="Snack demand outlook (hourly)",
            labels={"pred_snack_units": "Units", "hour": "Hour"},
        )
        st.plotly_chart(fc_fig, width="stretch")

        compare_cols = st.columns(2)
        with compare_cols[0]:
            st.caption("Forecast table")
            st.dataframe(
                forecast[
                    [
                        "timestamp",
                        "temperature_c",
                        "pred_checkins",
                        "pred_snack_units",
                        "snack_price",
                    ]
                ].rename(
                    columns={
                        "timestamp": "Time",
                        "temperature_c": "Temp °C",
                        "pred_checkins": "Check-ins",
                        "pred_snack_units": "Snacks",
                        "snack_price": "Price (CHF)",
                    }
                ),
                width="stretch",
                height=260,
            )
        with compare_cols[1]:
            if not log_df.empty:
                merged = pd.merge_asof(
                    log_df.sort_values("timestamp"),
                    forecast[["timestamp", "pred_snack_units", "pred_checkins"]].sort_values("timestamp"),
                    on="timestamp",
                    direction="nearest",
                    tolerance=pd.Timedelta("2h"),
                )
                merged = merged.dropna(subset=["pred_snack_units"])
                if not merged.empty:
                    err_fig = px.bar(
                        merged.tail(20),
                        x="timestamp",
                        y=["sales_units", "pred_snack_units"],
                        barmode="group",
                        title="Logged vs. forecast snacks",
                    )
                    st.plotly_chart(err_fig, width="stretch")
                else:
                    st.info("Log entries are too far from the forecast horizon to compare.")
            else:
                st.info("Need at least one POS entry to compare actual vs. forecast.")

render_footer()
