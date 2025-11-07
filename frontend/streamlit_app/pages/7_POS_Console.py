from pathlib import Path
import sys
from datetime import datetime, time

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import pandas as pd
import plotly.express as px
import streamlit as st

try:
    from frontend.streamlit_app.components.layout import (
        render_top_nav,
        sidebar_info_block,
        render_footer,
        get_logo_path,
    )
    from frontend.streamlit_app.services.data_utils import (
        WEATHER_SCENARIOS,
        append_pos_log,
        build_scenario_forecast,
        load_enriched_data,
        load_pos_log,
        train_models,
    )
except ModuleNotFoundError:  # pragma: no cover
    from components.layout import render_top_nav, sidebar_info_block, render_footer, get_logo_path
    from services.data_utils import (
        WEATHER_SCENARIOS,
        append_pos_log,
        build_scenario_forecast,
        load_enriched_data,
        load_pos_log,
        train_models,
    )

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

col_form, col_alert = st.columns([0.6, 0.4])

with col_form:
    st.subheader("Log POS activity")
    with st.form("pos-entry"):
        now = datetime.now()
        entry_date = st.date_input("Date", value=now.date())
        entry_time = st.time_input("Time", value=time(hour=now.hour, minute=now.minute))
        logged_sales = st.number_input("Snacks sold (units)", min_value=0.0, value=0.0, step=1.0)
        logged_checkins = st.number_input("Gym check-ins recorded", min_value=0.0, value=0.0, step=1.0)
        stock_remaining = st.number_input("Current stock on shelf (units)", min_value=0.0, value=50.0, step=1.0)
        notes = st.text_input("Notes / special context", "")
        submitted = st.form_submit_button("Log entry", use_container_width=True)

    if submitted:
        timestamp = datetime.combine(entry_date, entry_time)
        append_pos_log(
            {
                "timestamp": timestamp.isoformat(),
                "sales_units": logged_sales,
                "checkins_recorded": logged_checkins,
                "stock_remaining": stock_remaining,
                "notes": notes,
            }
        )
        st.success("Entry captured. Low stock & forecast panels updated.")
        log_df = load_pos_log()

    st.subheader("Recent entries")
    st.dataframe(
        log_df.sort_values("timestamp", ascending=False).head(10),
        use_container_width=True,
        height=260,
    )

with col_alert:
    st.subheader("Live stock posture")
    daily_usage = (
        base_data.set_index("timestamp")["snack_units"]
        .resample("D")
        .sum()
        .reset_index(name="daily_snacks")
    )
    avg_daily = float(daily_usage["daily_snacks"].mean()) if not daily_usage.empty else 50.0
    safety_floor = max(10.0, avg_daily * 1.5)
    latest_stock = log_df.sort_values("timestamp", ascending=False)["stock_remaining"].iloc[0] if not log_df.empty else None
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
        "snack_promo": "Baseline offer",
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
        st.plotly_chart(fc_fig, use_container_width=True)

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
                        "snack_price": "Price (€)",
                    }
                ),
                use_container_width=True,
                height=260,
            )
        with compare_cols[1]:
            if not log_df.empty:
                merged = pd.merge_asof(
                    log_df.sort_values("timestamp"),
                    forecast[["timestamp", "pred_snack_units", "pred_checkins"]].sort_values("timestamp"),
                    on="timestamp",
                    direction="nearest",
                    tolerance=pd.Timedelta("2H"),
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
                    st.plotly_chart(err_fig, use_container_width=True)
                else:
                    st.info("Log entries are too far from the forecast horizon to compare.")
            else:
                st.info("Need at least one POS entry to compare actual vs. forecast.")

render_footer()
