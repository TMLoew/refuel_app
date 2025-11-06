from pathlib import Path
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import statsmodels.api as sm
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from frontend.streamlit_app.components.layout import render_top_nav, sidebar_info_block
from frontend.streamlit_app.services.data_utils import load_enriched_data

st.set_page_config(page_title="Statistics", page_icon="📈", layout="wide")

render_top_nav("6_Statistics.py")
st.title("Statistical Rundown")
st.caption("Understand how weather, gym attendance, and snack demand interact using descriptive and regression analytics.")

with st.sidebar:
    sidebar_info_block()
    st.subheader("Data slice")
    use_weather_api = st.toggle("Use live weather API", value=False)
    lookback_days = st.slider("History window (days)", 3, 30, 14)

with st.spinner("Loading telemetry..."):
    data = load_enriched_data(use_weather_api=use_weather_api)

if data.empty:
    st.error("Need telemetry data in `data/gym_badges.csv` (or `*_long.csv`).")
    st.stop()

latest = data["timestamp"].max()
window_df = data[data["timestamp"] >= latest - pd.Timedelta(days=lookback_days)].copy()

if window_df.empty:
    st.warning("Selected lookback window returns no rows. Try increasing the range.")
    st.stop()

metrics_row = window_df.tail(1).iloc[0]
col_a, col_b, col_c, col_d = st.columns(4)
col_a.metric("Latest temperature", f"{metrics_row['temperature_c']:.1f}°C")
col_b.metric("Daily rainfall", f"{window_df['precipitation_mm'].tail(96).sum():.1f} mm")
col_c.metric("Check-ins (24h)", f"{window_df['checkins'].tail(96).sum():.0f}")
col_d.metric("Snack units (24h)", f"{window_df['snack_units'].tail(96).sum():.0f}")

st.subheader("Correlation overview")
corr_cols = ["temperature_c", "precipitation_mm", "checkins", "snack_units"]
corr_matrix = window_df[corr_cols].corr()
corr_fig = px.imshow(
    corr_matrix,
    text_auto=".2f",
    aspect="auto",
    color_continuous_scale="RdBu",
    title="Pearson correlation matrix",
)
st.plotly_chart(corr_fig, use_container_width=True)

st.subheader("Pairwise relationships")
pair_cols = ["temperature_c", "precipitation_mm", "checkins", "snack_units"]
pair_fig = px.scatter_matrix(window_df, dimensions=pair_cols, color="weather_label", title="Scatter matrix with weather classes")
st.plotly_chart(pair_fig, use_container_width=True)

st.subheader("Weather → attendance → snacks")
scatter_cols = st.columns(2)
scatter_cols[0].plotly_chart(
    px.scatter(
        window_df,
        x="temperature_c",
        y="checkins",
        color="is_weekend",
        labels={"is_weekend": "Weekend?", "temperature_c": "Temperature (°C)", "checkins": "Check-ins"},
        trendline="ols",
        title="Temperature vs. gym attendance",
    ),
    use_container_width=True,
)
scatter_cols[1].plotly_chart(
    px.scatter(
        window_df,
        x="precipitation_mm",
        y="snack_units",
        color="weather_label",
        labels={"precipitation_mm": "Precipitation (mm)", "snack_units": "Snack units"},
        trendline="ols",
        title="Precipitation vs. snack demand",
    ),
    use_container_width=True,
)

st.subheader("Regression diagnostics")
reg_df = window_df[["checkins", "snack_units", "temperature_c", "precipitation_mm", "is_weekend"]].copy()
reg_df["is_weekend"] = reg_df["is_weekend"].astype(float)

X = sm.add_constant(reg_df[["temperature_c", "precipitation_mm", "is_weekend"]])
checkins_model = sm.OLS(reg_df["checkins"], X).fit()
snacks_model = sm.OLS(reg_df["snack_units"], X).fit()

coef_table = pd.DataFrame(
    {
        "Predictor": X.columns,
        "Check-ins β": checkins_model.params,
        "Check-ins p": checkins_model.pvalues,
        "Snack β": snacks_model.params,
        "Snack p": snacks_model.pvalues,
    }
).round(3)
st.dataframe(coef_table, use_container_width=True)

st.markdown(
    """
- **Coefficients (β)** quantify how much the outcome changes per unit shift in the predictor, controlling for the other variables.
- **p-values** < 0.05 usually indicate statistically meaningful relationships in the current sample.
"""
)

st.subheader("Daily decomposition")
daily = (
    window_df.set_index("timestamp")
    .resample("D")
    .agg({"temperature_c": "mean", "precipitation_mm": "sum", "checkins": "sum", "snack_units": "sum"})
    .reset_index()
)
daily_fig = px.line(
    daily,
    x="timestamp",
    y=["temperature_c", "checkins", "snack_units"],
    labels={"value": "Value", "variable": "Series"},
    title="Daily weather & demand trend",
)
st.plotly_chart(daily_fig, use_container_width=True)

st.info(
    "Use this page to validate demand hypotheses before configuring scenarios and automation on the other tabs."
)
