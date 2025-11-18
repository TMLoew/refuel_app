from pathlib import Path
import sys
from datetime import datetime, timezone
from typing import Optional

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from frontend.streamlit_app.components.layout import (
    render_top_nav,
    sidebar_info_block,
    render_footer,
    get_logo_path,
)
try:
    from frontend.streamlit_app.components.layout import hover_tip
except ImportError:
    try:
        from components.layout import hover_tip  # type: ignore
    except ImportError:
        def hover_tip(label: str, tooltip: str) -> None:
            st.caption(f"{label}: {tooltip}")

from frontend.streamlit_app.services.data_utils import (
    WEATHER_SCENARIOS,
    build_scenario_forecast,
    load_enriched_data,
    load_pos_log,
    load_weather_profile,
    save_weather_profile,
    train_models,
)
from frontend.streamlit_app.services import weather_pipeline

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
    col_c.metric("Snack revenue (24h)", f"CHF{recent['snack_revenue'].sum():.0f}")
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


def render_weather_shotcast() -> None:
    profile = load_weather_profile()
    lat, lon = profile["lat"], profile["lon"]
    iframe = f"""
    <iframe
        src="https://embed.windy.com/embed2.html?lat={lat:.3f}&lon={lon:.3f}&zoom=8&level=surface&overlay=rainAccu&menu=&message=true&marker=true&calendar=now&pressure=&type=map&location=coordinates&detail=true&detailLat={lat:.3f}&detailLon={lon:.3f}&metricWind=default&metricTemp=default&radarRange=-1"
        style="width:100%; height:340px; border:0; border-radius:12px;"
        allowfullscreen
    ></iframe>
    """
    components.html(iframe, height=340)


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


def _latest_stock_reading() -> Optional[float]:
    """Fetch the most recent stock reading from the POS log if available."""
    try:
        log_df = load_pos_log()
    except Exception:
        return None
    if log_df.empty or "stock_remaining" not in log_df.columns:
        return None
    sorted_log = (
        log_df.sort_values("timestamp", ascending=False)["stock_remaining"]
        .pipe(pd.to_numeric, errors="coerce")
        .dropna()
    )
    if sorted_log.empty:
        return None
    return float(sorted_log.iloc[0])


def _estimate_daily_demand(history: pd.DataFrame, forecast: pd.DataFrame) -> float:
    """Use historical + forecast data to approximate daily snack demand."""
    if not history.empty:
        daily_usage = (
            history.resample("D", on="timestamp")["snack_units"]
            .sum()
            .tail(14)
        )
        daily_mean = float(daily_usage.mean()) if not daily_usage.empty else float("nan")
        if pd.notna(daily_mean) and daily_mean > 0:
            return daily_mean

    if forecast.empty:
        return 20.0

    forecast_view = forecast.sort_values("timestamp")[["timestamp", "pred_snack_units"]].copy()
    freq_seconds = forecast_view["timestamp"].diff().dt.total_seconds().median()
    freq_hours = freq_seconds / 3600.0 if pd.notna(freq_seconds) and freq_seconds else 1.0
    hourly_mean = float(forecast_view["pred_snack_units"].mean())
    estimated_daily = hourly_mean * (24.0 / max(freq_hours, 0.1))
    return max(5.0, estimated_daily)


def render_model_reorder_plan(history: pd.DataFrame, forecast: pd.DataFrame) -> None:
    st.subheader("Model-driven reorder guidance")
    if forecast.empty:
        st.info("Run the scenario forecast above to unlock reorder recommendations.")
        return

    avg_daily_demand = _estimate_daily_demand(history, forecast)
    default_stock = max(20.0, round(avg_daily_demand * 3))
    latest_stock = _latest_stock_reading()
    if latest_stock is not None:
        default_stock = float(latest_stock)

    forecast_view = (
        forecast.sort_values("timestamp")[["timestamp", "pred_snack_units"]]
        .copy()
        .assign(pred_snack_units=lambda df_: df_["pred_snack_units"].clip(lower=0).fillna(0))
    )

    with st.expander("Reorder assumptions", expanded=True):
        col_a, col_b, col_c = st.columns(3)
        current_stock = col_a.number_input(
            "Current stock (units)",
            min_value=0.0,
            value=float(default_stock),
            step=5.0,
            help="Auto-filled from the latest POS entry when available.",
        )
        safety_days = col_b.slider(
            "Safety stock (days of demand)",
            min_value=0.0,
            max_value=5.0,
            value=1.0,
            step=0.5,
            help="We will trigger a reorder before the projected stock drops below this buffer.",
        )
        reorder_lot = col_c.number_input(
            "Preferred reorder lot (units)",
            min_value=0.0,
            value=float(max(10.0, round(avg_daily_demand))),
            step=5.0,
        )

    safety_units = safety_days * avg_daily_demand
    forecast_view["cumulative_units"] = forecast_view["pred_snack_units"].cumsum()
    forecast_view["projected_stock"] = current_stock - forecast_view["cumulative_units"]

    reorder_ts: Optional[pd.Timestamp] = None
    if current_stock <= safety_units:
        reorder_ts = pd.Timestamp.now()
    else:
        below_buffer = forecast_view["projected_stock"] <= safety_units
        if below_buffer.any():
            reorder_ts = forecast_view.loc[below_buffer, "timestamp"].iloc[0]

    depletion_ts: Optional[pd.Timestamp] = None
    below_zero = forecast_view["projected_stock"] <= 0
    if below_zero.any():
        depletion_ts = forecast_view.loc[below_zero, "timestamp"].iloc[0]

    def _format_eta(target: Optional[pd.Timestamp]) -> str:
        if target is None:
            return "n/a"
        eta_hours = (target - pd.Timestamp.now()).total_seconds() / 3600.0
        if eta_hours < 0:
            return "due"
        return f"in {eta_hours:.1f} h"

    col1, col2, col3 = st.columns(3)
    if reorder_ts is not None:
        col1.metric("Next reorder target", reorder_ts.strftime("%a %H:%M"), _format_eta(reorder_ts))
    else:
        col1.metric("Next reorder target", "Beyond forecast", "extend horizon")
    if depletion_ts is not None:
        col2.metric("Projected stockout", depletion_ts.strftime("%a %H:%M"), _format_eta(depletion_ts))
    else:
        col2.metric("Projected stockout", "Outside horizon", "buffer sufficient")
    projected_floor = float(forecast_view["projected_stock"].min()) if not forecast_view.empty else float("nan")
    col3.metric(
        "Min projected stock",
        f"{projected_floor:.0f} units",
        f"Safety floor {safety_units:.0f}u",
    )

    if reorder_ts is not None:
        st.success(
            f"Plan to reorder ~{reorder_lot:.0f} units by **{reorder_ts.strftime('%Y-%m-%d %H:%M')}** "
            f"before the forecast dips below the {safety_units:.0f} unit buffer."
        )
    else:
        st.info(
            "The current forecast horizon never touches the safety buffer. Increase the horizon or lower "
            "the buffer if you need a nearer recommendation."
        )

    stock_chart = px.area(
        forecast_view,
        x="timestamp",
        y="projected_stock",
        title="Projected stock vs. safety buffer",
        labels={"projected_stock": "Projected stock (units)"},
    )
    stock_chart.add_hline(
        y=safety_units,
        line_dash="dash",
        line_color="#E74C3C",
        annotation_text=f"Safety floor ({safety_units:.0f})",
        annotation_position="bottom left",
    )
    stock_chart.update_traces(line_shape="hv")
    st.plotly_chart(stock_chart, use_container_width=True)

    st.dataframe(
        forecast_view[["timestamp", "pred_snack_units", "projected_stock"]].rename(
            columns={
                "timestamp": "Time",
                "pred_snack_units": "Predicted sales",
                "projected_stock": "Projected stock",
            }
        ),
        use_container_width=True,
        height=260,
    )

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
        f"CHF{forecast['pred_snack_revenue'].sum():.0f}",
        f"avg price CHF{forecast['snack_price'].mean():.2f}",
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
                "snack_price": "Snack price (CHF)",
                "pred_snack_revenue": "Snack revenue (CHF)",
            }
        )
        .set_index("timestamp"),
        use_container_width=True,
    )
def render_dashboard() -> None:
    render_top_nav("1_Dashboard.py", show_logo=False)
    st.title("Refuel Performance Cockpit")
    st.caption(
        "Blending weather mood, gym traffic, and snack behavior to guide staffing, procurement, and marketing."
    )

    with st.sidebar:
        sidebar_info_block()
        st.subheader("Scenario controls")
        use_weather_api = st.toggle("Use live weather API", value=True)
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

    scenario = {
        "horizon_hours": horizon_hours,
        "weather_pattern": weather_pattern,
        "temp_manual": temp_manual,
        "precip_manual": precip_manual,
        "event_intensity": event_intensity,
        "marketing_boost_pct": marketing_boost_pct,
        "snack_price_change": snack_price_change,
        "use_live_weather": use_weather_api,
    }

    render_summary_cards(data)
    render_history_charts(data, history_days)
    st.subheader("Weather shotcast")
    caption_col, button_col = st.columns([0.75, 0.25])
    caption_col.caption("Windy radar & cloud layers centered on the configured coordinates.")
    if button_col.button("Center on St. Gallen", key="shotcast-center-page"):
        save_weather_profile(
            {
                "lat": weather_pipeline.DEFAULT_LAT,
                "lon": weather_pipeline.DEFAULT_LON,
            }
        )
        st.success("Shotcast centered on St. Gallen.")
        st.experimental_rerun()
    render_weather_shotcast()

    forecast_df = build_scenario_forecast(data, models, scenario)
    st.subheader("What-if forecast")
    render_forecast_section(data, forecast_df)
    render_model_reorder_plan(data, forecast_df)
    render_footer()


def _safe_render() -> None:
    try:
        render_dashboard()
    except Exception as exc:
        st.error("The dashboard crashed while rendering. See details below and please share this trace.")
        st.exception(exc)
        raise


_safe_render()
