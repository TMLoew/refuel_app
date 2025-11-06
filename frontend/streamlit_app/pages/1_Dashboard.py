from pathlib import Path
import sys
from datetime import datetime, timezone

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from frontend.streamlit_app.components.layout import (
    render_top_nav,
    sidebar_info_block,
    render_footer,
    get_logo_path,
)
from frontend.streamlit_app.services.data_utils import (
    SNACK_PROMOS,
    WEATHER_SCENARIOS,
    build_scenario_forecast,
    load_enriched_data,
    train_models,
)

PAGE_ICON = get_logo_path() or "💪"
st.set_page_config(
    page_title="Refuel Ops Dashboard",
    layout="wide",
    page_icon=PAGE_ICON,
)



def render_summary_cards(df: pd.DataFrame) -> None:
    recent = df.tail(24)
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Avg hourly check-ins (24h)", f"{recent['checkins'].mean():.1f}")
    col_b.metric("Snack units sold (24h)", f"{recent['snack_units'].sum():.0f}")
    col_c.metric("Snack revenue (24h)", f"€{recent['snack_revenue'].sum():.0f}")
    peak_hour = df.loc[df["checkins"].idxmax()]
    col_d.metric(
        "Peak load",
        f"{int(peak_hour['checkins'])} check-ins",
        f"{peak_hour['timestamp'].strftime('%a %H:%M')}",
    )


def render_history_charts(df: pd.DataFrame, window_days: int) -> None:
    history_window = df[df["timestamp"] >= df["timestamp"].max() - pd.Timedelta(days=window_days)]
    usage_fig = go.Figure()
    usage_fig.add_trace(
        go.Scatter(
            x=history_window["timestamp"],
            y=history_window["checkins"],
            mode="lines",
            name="Check-ins",
            line=dict(color="#2E86AB"),
        )
    )
    usage_fig.add_trace(
        go.Scatter(
            x=history_window["timestamp"],
            y=history_window["snack_units"],
            mode="lines",
            name="Snack units",
            yaxis="y2",
            line=dict(color="#F18F01"),
        )
    )
    usage_fig.update_layout(
        title="Gym traffic vs. snack demand",
        xaxis_title="Timestamp",
        yaxis_title="Check-ins",
        yaxis2=dict(title="Snack units", overlaying="y", side="right"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=380,
    )

    weather_fig = px.line(
        history_window,
        x="timestamp",
        y=["temperature_c", "precipitation_mm"],
        title="Weather trend (synthetic blend)",
        labels={"value": "Value", "variable": "Metric"},
    )
    weather_fig.update_layout(height=380, legend=dict(orientation="h", yanchor="bottom", y=1.02))

    col1, col2 = st.columns(2)
    col1.plotly_chart(usage_fig, use_container_width=True)
    col2.plotly_chart(weather_fig, use_container_width=True)


def render_forecast_section(history: pd.DataFrame, forecast: pd.DataFrame) -> None:
    if forecast.empty:
        st.warning("Not enough data to train the forecasting widgets yet.")
        return

    combined = history[
        history["timestamp"] >= history["timestamp"].max() - pd.Timedelta(hours=48)
    ][["timestamp", "checkins", "snack_units"]].copy()
    combined.rename(columns={"checkins": "actual_checkins", "snack_units": "actual_snacks"}, inplace=True)

    future_plot = forecast[["timestamp", "pred_checkins", "pred_snack_units"]].copy()
    combined = combined.merge(future_plot, on="timestamp", how="outer").sort_values("timestamp")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=combined["timestamp"],
            y=combined["actual_checkins"],
            mode="lines",
            name="Actual check-ins",
            line=dict(color="#2E86AB"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=combined["timestamp"],
            y=combined["pred_checkins"],
            mode="lines",
            name="Forecast check-ins",
            line=dict(color="#2E86AB", dash="dash"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=combined["timestamp"],
            y=combined["actual_snacks"],
            mode="lines",
            name="Actual snack units",
            line=dict(color="#F18F01"),
            yaxis="y2",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=combined["timestamp"],
            y=combined["pred_snack_units"],
            mode="lines",
            name="Forecast snack units",
            line=dict(color="#F18F01", dash="dash"),
            yaxis="y2",
        )
    )
    fig.update_layout(
        title="Scenario forecast",
        xaxis_title="Timestamp",
        yaxis=dict(title="Check-ins"),
        yaxis2=dict(title="Snack units", overlaying="y", side="right"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=420,
        shapes=[
            dict(
                type="rect",
                xref="x",
                yref="paper",
                x0=history["timestamp"].max(),
                x1=forecast["timestamp"].max(),
                y0=0,
                y1=1,
                fillcolor="rgba(200,200,200,0.15)",
                line=dict(width=0),
            )
        ],
    )
    st.plotly_chart(fig, use_container_width=True)

    kpi_cols = st.columns(3)
    kpi_cols[0].metric(
        "Forecast check-ins",
        f"{forecast['pred_checkins'].sum():.0f}",
        f"{forecast['pred_checkins'].max():.0f} peak/hr",
    )
    kpi_cols[1].metric(
        "Forecast snack units",
        f"{forecast['pred_snack_units'].sum():.0f}",
        f"{forecast['pred_snack_units'].max():.0f} peak/hr",
    )
    kpi_cols[2].metric(
        "Forecast snack revenue",
        f"€{forecast['pred_snack_revenue'].sum():.0f}",
        f"avg price €{forecast['snack_price'].mean():.2f}",
    )

    st.dataframe(
        forecast[
            [
                "timestamp",
                "temperature_c",
                "precipitation_mm",
                "pred_checkins",
                "pred_snack_units",
                "snack_price",
                "pred_snack_revenue",
            ]
        ]
        .rename(
            columns={
                "temperature_c": "Temp (°C)",
                "precipitation_mm": "Precip (mm)",
                "pred_checkins": "Check-ins",
                "pred_snack_units": "Snack units",
                "snack_price": "Snack price (€)",
                "pred_snack_revenue": "Snack revenue (€)",
            }
        )
        .set_index("timestamp"),
        use_container_width=True,
    )


def render_inventory_game(df: pd.DataFrame) -> None:
    st.subheader("Inventory sandbox")
    daily_usage = (
        df.resample("D", on="timestamp")["snack_units"]
        .sum()
        .reset_index()
        .rename(columns={"snack_units": "daily_snacks"})
    )
    if daily_usage.empty:
        st.info("Not enough data to simulate inventory yet.")
        return

    avg_daily = float(daily_usage["daily_snacks"].mean()) if not daily_usage.empty else 10.0
    default_start = max(10.0, round(avg_daily * 4, 1))
    with st.expander("Game controls", expanded=True):
        start_stock = st.number_input(
            "Starting stock (units)", min_value=0.0, value=default_start, step=10.0
        )
        low_threshold = st.number_input(
            "Low-stock alert threshold", min_value=0.0, value=max(5.0, round(avg_daily, 1))
        )
        reorder_amount = st.number_input(
            "Reorder amount", min_value=0.0, value=max(10.0, round(avg_daily * 2, 1)), step=5.0
        )
        col_a, col_b = st.columns(2)
        if col_a.button("Reset inventory game"):
            st.session_state["stock_level"] = start_stock
            st.session_state["stock_day_idx"] = 0
            st.session_state["stock_history"] = []
            st.rerun()
        if col_b.button("Reorder now"):
            st.session_state["stock_level"] = st.session_state.get("stock_level", start_stock) + reorder_amount
            st.rerun()

    if "stock_level" not in st.session_state:
        st.session_state["stock_level"] = start_stock
    if "stock_day_idx" not in st.session_state:
        st.session_state["stock_day_idx"] = 0
    if "stock_history" not in st.session_state:
        st.session_state["stock_history"] = []

    idx = st.session_state["stock_day_idx"] % len(daily_usage)
    current_day = daily_usage.iloc[idx]

    col1, col2, col3 = st.columns(3)
    col1.metric("Simulated date", str(current_day["timestamp"].date()))
    col2.metric("Stock level", f"{st.session_state['stock_level']:.0f} units")
    col3.metric("Projected demand", f"{current_day['daily_snacks']:.0f} units")

    if st.session_state["stock_level"] <= low_threshold:
        st.warning("Low stock! Consider reordering before the next day.")

    if st.button("Next day ➡️"):
        st.session_state["stock_level"] = max(
            0.0, st.session_state["stock_level"] - current_day["daily_snacks"]
        )
        st.session_state["stock_history"].append(
            {
                "date": current_day["timestamp"].date(),
                "stock_end": st.session_state["stock_level"],
                "consumption": current_day["daily_snacks"],
            }
        )
        st.session_state["stock_day_idx"] = (idx + 1) % len(daily_usage)
        st.rerun()

    if st.session_state["stock_history"]:
        hist_df = pd.DataFrame(st.session_state["stock_history"])
        stock_fig = px.line(hist_df, x="date", y="stock_end", title="Stock level over simulated days")
        stock_fig.update_traces(mode="lines+markers")
        st.plotly_chart(stock_fig, use_container_width=True)


def render_dashboard() -> None:
    render_top_nav("1_Dashboard.py")
    st.title("Refuel Performance Cockpit")
    st.caption(
        "Blending weather mood, gym traffic, and snack behavior to guide staffing, procurement, and marketing."
    )

    with st.sidebar:
        sidebar_info_block()
        st.subheader("Scenario controls")
        use_weather_api = st.toggle("Use live weather API", value=False)
        refresh_weather = st.button("🔄 Refresh weather data", use_container_width=True)

    cache_buster = datetime.now(timezone.utc).timestamp() if refresh_weather else 0.0
    with st.spinner("Loading telemetry and contextual data..."):
        data = load_enriched_data(use_weather_api=use_weather_api, cache_buster=cache_buster)
    if data.empty:
        st.error("No gym data found yet. Drop a CSV into `data/gym_badges.csv` to get started.")
        return

    total_days = max(1, int((data["timestamp"].max() - data["timestamp"].min()).days) or 1)
    history_days = st.sidebar.slider(
        "History window (days)",
        min_value=3,
        max_value=max(3, total_days),
        value=min(7, max(3, total_days)),
    )

    weather_source = data.attrs.get("weather_source", "synthetic")
    weather_meta = data.attrs.get("weather_meta", {})
    st.caption(
        f"Weather source · {'Open-Meteo API' if weather_source == 'open-meteo' else 'synthetic fallback'}"
    )
    if weather_meta:
        latency_text = f"{weather_meta.get('latency_ms', 0):.0f} ms" if weather_meta.get("latency_ms") else "n/a"
        st.caption(
            f"Weather last synced {weather_meta.get('updated_at', 'n/a')} UTC · latency {latency_text} · coverage {weather_meta.get('coverage_start', '?')} → {weather_meta.get('coverage_end', '?')}"
        )
    if use_weather_api and weather_source != "open-meteo":
        st.warning("Live weather API unreachable. Using synthetic fallback instead.")

    models = train_models(data)

    with st.sidebar:
        with st.expander("Weather & demand levers", expanded=True):
            horizon_hours = st.slider("Forecast horizon (hours)", min_value=6, max_value=72, value=24, step=6)
            weather_pattern = st.selectbox("Weather pattern", list(WEATHER_SCENARIOS.keys()))
            temp_manual = st.slider("Manual temperature shift (°C)", min_value=-8, max_value=8, value=0)
            precip_manual = st.slider(
                "Manual precipitation shift (mm)", min_value=-2.0, max_value=2.0, value=0.0, step=0.1
            )
        with st.expander("Campaign & pricing knobs", expanded=True):
            event_intensity = st.slider(
                "Gym event intensity",
                min_value=0.2,
                max_value=2.5,
                value=1.0,
                step=0.1,
                help="Represents corporate challenges or class launches (baseline ~0.5).",
            )
            marketing_boost_pct = st.slider(
                "Marketing reach boost (%)",
                min_value=0,
                max_value=80,
                value=10,
                step=5,
            )
            snack_price_change = st.slider(
                "Snack price change (%)",
                min_value=-30,
                max_value=40,
                value=0,
                step=5,
            )
            snack_promo = st.selectbox("Snack activation", list(SNACK_PROMOS.keys()))

    scenario = {
        "horizon_hours": horizon_hours,
        "weather_pattern": weather_pattern,
        "temp_manual": temp_manual,
        "precip_manual": precip_manual,
        "event_intensity": event_intensity,
        "marketing_boost_pct": marketing_boost_pct,
        "snack_price_change": snack_price_change,
        "snack_promo": snack_promo,
    }

    render_summary_cards(data)
    render_history_charts(data, history_days)

    forecast_df = build_scenario_forecast(data, models, scenario)
    st.subheader("What-if forecast")
    render_forecast_section(data, forecast_df)
    render_inventory_game(data)
    render_footer()


def _safe_render() -> None:
    try:
        render_dashboard()
    except Exception as exc:
        st.error("The dashboard crashed while rendering. See details below and please share this trace.")
        st.exception(exc)
        raise


_safe_render()
