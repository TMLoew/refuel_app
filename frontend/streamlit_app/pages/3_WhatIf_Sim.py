from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import pandas as pd
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

PAGE_ICON = get_logo_path() or "🧪"
st.set_page_config(page_title="What-if Simulator", page_icon=PAGE_ICON, layout="wide")

render_top_nav("3_WhatIf_Sim.py")
st.title("Scenario Lab")
st.caption("Stack two competing scenarios, stress test the demand outlook, and quantify the upside/downside.")

with st.sidebar:
    sidebar_info_block()
    use_weather_api = st.toggle("Use live weather API", value=False, key="whatif-weather")

data = load_enriched_data(use_weather_api=use_weather_api)
if data.empty:
    st.error("No telemetry data found. Use the Data Editor to upload a CSV.")
    st.stop()

models = train_models(data)

default_scenario = {
    "horizon_hours": 24,
    "weather_pattern": "Temperate & sunny",
    "temp_manual": 0,
    "precip_manual": 0.0,
    "event_intensity": 1.0,
    "marketing_boost_pct": 10,
    "snack_price_change": 0,
    "snack_promo": "Baseline offer",
}


def scenario_form(label: str, defaults: dict):
    with st.expander(label, expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            weather_pattern = st.selectbox(
                "Weather pattern",
                list(WEATHER_SCENARIOS.keys()),
                index=list(WEATHER_SCENARIOS.keys()).index(defaults["weather_pattern"]),
                key=f"{label}-weather",
            )
            temp_manual = st.slider("Manual temp shift (°C)", -8, 8, defaults["temp_manual"], key=f"{label}-temp")
            precip_manual = st.slider(
                "Manual precipitation shift (mm)",
                -2.0,
                2.0,
                defaults["precip_manual"],
                step=0.1,
                key=f"{label}-precip",
            )
            event_intensity = st.slider(
                "Event intensity",
                0.2,
                2.5,
                defaults["event_intensity"],
                step=0.1,
                key=f"{label}-event",
            )
        with col2:
            marketing_boost_pct = st.slider(
                "Marketing boost (%)",
                0,
                100,
                defaults["marketing_boost_pct"],
                step=5,
                key=f"{label}-marketing",
            )
            snack_price_change = st.slider(
                "Snack price change (%)",
                -30,
                40,
                defaults["snack_price_change"],
                step=5,
                key=f"{label}-price",
            )
            snack_promo = st.selectbox(
                "Snack activation",
                list(SNACK_PROMOS.keys()),
                index=list(SNACK_PROMOS.keys()).index(defaults["snack_promo"]),
                key=f"{label}-promo",
            )
            horizon_hours = st.slider(
                "Forecast horizon (hours)",
                6,
                72,
                defaults["horizon_hours"],
                step=6,
                key=f"{label}-horizon",
            )
    return {
        "horizon_hours": horizon_hours,
        "weather_pattern": weather_pattern,
        "temp_manual": temp_manual,
        "precip_manual": precip_manual,
        "event_intensity": event_intensity,
        "marketing_boost_pct": marketing_boost_pct,
        "snack_price_change": snack_price_change,
        "snack_promo": snack_promo,
        "use_live_weather": use_weather_api,
    }


scenario_a = scenario_form("Scenario A · Baseline", default_scenario)
scenario_b = scenario_form(
    "Scenario B · Experimental",
    {**default_scenario, "marketing_boost_pct": 30, "snack_promo": "Buy-one-get-one", "horizon_hours": 36},
)

forecast_a = build_scenario_forecast(data, models, scenario_a)
forecast_b = build_scenario_forecast(data, models, scenario_b)

def summarize_forecast(df: pd.DataFrame, label: str) -> dict:
    return {
        "label": label,
        "checkins": df["pred_checkins"].sum(),
        "snack_units": df["pred_snack_units"].sum(),
        "snack_revenue": df["pred_snack_revenue"].sum(),
    }


summary_df = pd.DataFrame(
    [summarize_forecast(forecast_a, "Scenario A"), summarize_forecast(forecast_b, "Scenario B")]
)
summary_df["snack_margin_proxy"] = summary_df["snack_revenue"] - 1.5 * summary_df["snack_units"]

st.subheader("Scenario comparison")
metric_cols = st.columns(3)
metric_cols[0].metric(
    "Snack revenue delta",
    f"€{summary_df.loc[1, 'snack_revenue'] - summary_df.loc[0, 'snack_revenue']:.0f}",
    f"{(summary_df.loc[1, 'snack_revenue'] / summary_df.loc[0, 'snack_revenue'] - 1)*100:.1f}%",
)
metric_cols[1].metric(
    "Check-ins delta",
    f"{summary_df.loc[1, 'checkins'] - summary_df.loc[0, 'checkins']:.0f}",
    f"{(summary_df.loc[1, 'checkins'] / summary_df.loc[0, 'checkins'] - 1)*100:.1f}%",
)
metric_cols[2].metric(
    "Snack margin proxy delta",
    f"€{summary_df.loc[1, 'snack_margin_proxy'] - summary_df.loc[0, 'snack_margin_proxy']:.0f}",
)

comp_fig = go.Figure()
for col in ["checkins", "snack_units", "snack_revenue"]:
    comp_fig.add_trace(
        go.Bar(
            x=summary_df["label"],
            y=summary_df[col],
            name=col.replace("_", " ").title(),
        )
    )
comp_fig.update_layout(barmode="group", title="Aggregate forecast comparison")
st.plotly_chart(comp_fig, width="stretch")

st.subheader("Hourly trajectory")
combined = pd.concat(
    [forecast_a.assign(scenario="A"), forecast_b.assign(scenario="B")],
    ignore_index=True,
)

line_fig = go.Figure()
for scenario, df_slice in combined.groupby("scenario"):
    line_fig.add_trace(
        go.Scatter(
            x=df_slice["timestamp"],
            y=df_slice["pred_checkins"],
            mode="lines",
            name=f"Check-ins {scenario}",
        )
    )
    line_fig.add_trace(
        go.Scatter(
            x=df_slice["timestamp"],
            y=df_slice["pred_snack_units"],
            mode="lines",
            name=f"Snack units {scenario}",
            yaxis="y2",
            line=dict(dash="dash"),
        )
    )
line_fig.update_layout(
    yaxis=dict(title="Check-ins"),
    yaxis2=dict(title="Snack units", overlaying="y", side="right"),
    title="Projected hourly demand profiles",
)
st.plotly_chart(line_fig, width="stretch")

st.subheader("Scenario inputs recap")
st.write("Scenario A", scenario_a)
st.write("Scenario B", scenario_b)
render_footer()
