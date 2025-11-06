from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from frontend.streamlit_app.components.layout import render_top_nav, sidebar_info_block
from frontend.streamlit_app.services.data_utils import (
    CHECKIN_FEATURES,
    load_enriched_data,
    train_models,
)


st.set_page_config(page_title="Forecast Explorer", page_icon="🔮", layout="wide")

render_top_nav("2_Forecasts.py")
st.title("Forecast Explorer")
st.caption("Dig into the regression models, understand residuals, and inspect sensitivities before committing to a plan.")

with st.sidebar:
    sidebar_info_block()
    st.subheader("Data slice")
    use_weather_api = st.toggle("Use live weather API", value=False, key="forecast-weather")
    lookback_days = st.slider("History window (days)", 3, 14, 7)
    metric_focus = st.selectbox("Focus metric", ["checkins", "snack_units", "snack_revenue"])
    with st.expander("Snacks ↔ visits settings", expanded=False):
        snack_agg_mode = st.radio("Aggregation", ["Hourly", "Daily"], horizontal=True, key="snack-agg")
        color_dim = st.selectbox(
            "Color points by",
            ["weekday", "weather_label", "is_weekend"],
            index=0,
            key="snack-color",
        )
        min_checkins = st.slider("Min. check-ins to include", 0, 50, 5, key="snack-min-checkins")

data = load_enriched_data(use_weather_api=use_weather_api)
if data.empty:
    st.error("No telemetry CSV detected. Upload data via the Data Editor page first.")
    st.stop()

models = train_models(data)

latest_ts = data["timestamp"].max()
history = data[data["timestamp"] >= latest_ts - pd.Timedelta(days=lookback_days)]

col_a, col_b, col_c = st.columns(3)
col_a.metric("Latest check-ins/hr", f"{history['checkins'].iloc[-1]:.0f}")
col_b.metric("Snack revenue (24h)", f"€{history.tail(24)['snack_revenue'].sum():.0f}")
col_c.metric("Weather source", data.attrs.get("weather_source", "synthetic").title())

# --- Daily trend ----------------------------------------------------------------
daily = history.resample("D", on="timestamp").agg({"checkins": "sum", "snack_units": "sum", "snack_revenue": "sum"})
trend_fig = px.line(
    daily.reset_index(),
    x="timestamp",
    y=["checkins", "snack_units"],
    title="Daily totals",
    markers=True,
    labels={"value": "Value", "variable": "Series", "timestamp": "Date"},
)
trend_fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))

# --- Heatmap --------------------------------------------------------------------
pivot = history.pivot_table(index="weekday", columns="hour", values="checkins", aggfunc="mean").reindex(range(7))
heatmap_fig = go.Figure(
    data=go.Heatmap(
        z=pivot.values,
        x=pivot.columns,
        y=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
        colorscale=[
            [0.0, "#ffffff"],
            [0.3, "#ffe5e5"],
            [0.6, "#ff7b7b"],
            [1.0, "#a80000"],
        ],
        colorbar=dict(title="Avg check-ins/hr"),
    )
)
heatmap_fig.update_layout(title="Hourly utilization pattern")

col1, col2 = st.columns(2)
col1.plotly_chart(trend_fig, use_container_width=True)
col2.plotly_chart(heatmap_fig, use_container_width=True)

# --- Correlation matrix --------------------------------------------------------
corr_fields = ["checkins", "snack_units", "temperature_c", "precipitation_mm", "humidity_pct"]
corr_matrix = history[corr_fields].corr()
corr_fig = px.imshow(
    corr_matrix,
    text_auto=".2f",
    aspect="auto",
    color_continuous_scale="RdBu",
    title="Correlation matrix: weather vs. demand",
)
st.plotly_chart(corr_fig, use_container_width=True)

# --- Snack vs visit correlation explorer ---------------------------------------
if snack_agg_mode == "Daily":
    snack_df = history.resample("D", on="timestamp").agg(
        {"checkins": "sum", "snack_units": "sum", "temperature_c": "mean", "weekday": "first", "is_weekend": "max"}
    )
    snack_df["weather_label"] = history.resample("D", on="timestamp")["weather_label"].agg(lambda x: x.mode().iloc[0])
else:
    snack_df = history.copy()

snack_df = snack_df[snack_df["checkins"] >= min_checkins]

if not snack_df.empty:
    corr_value = snack_df["checkins"].corr(snack_df["snack_units"])
    st.subheader("Snack demand vs. visit load")
    st.caption("Tune the controls in the sidebar to slice the correlation by aggregation or filters.")
    st.metric("Pearson correlation", f"{corr_value:.2f}" if not pd.isna(corr_value) else "n/a")
    color_field = color_dim if color_dim in snack_df.columns else None
    scatter_fig = px.scatter(
        snack_df,
        x="checkins",
        y="snack_units",
        color=color_field,
        trendline="ols",
        labels={"checkins": "Gym check-ins", "snack_units": "Snack units"},
        title="Snacks vs. visits",
        hover_data=["temperature_c"],
    )
    st.plotly_chart(scatter_fig, use_container_width=True)

# --- Feature sensitivity --------------------------------------------------------
checkin_model, snack_model = models
if checkin_model is not None:
    coefs = checkin_model.named_steps["model"].coef_
    impact = (
        pd.DataFrame({"feature": CHECKIN_FEATURES, "coef": coefs})
        .sort_values("coef", key=lambda s: np.abs(s), ascending=False)
        .head(8)
    )
    impact_fig = px.bar(
        impact,
        x="coef",
        y="feature",
        orientation="h",
        color="coef",
        color_continuous_scale="RdBu",
        title="Model coefficient (standardized features)",
    )
    impact_fig.update_layout(coloraxis_showscale=False)

    st.plotly_chart(impact_fig, use_container_width=True)

# --- Residual diagnostics -------------------------------------------------------
if checkin_model is not None:
    feature_df = history.copy()
    feature_df["residuals_checkins"] = feature_df["checkins"] - checkin_model.predict(
        feature_df[CHECKIN_FEATURES]
    )
    residual_fig = px.scatter(
        feature_df,
        x="temperature_c",
        y="residuals_checkins",
        color="weather_label",
        title="Residuals vs. temperature",
        trendline="ols",
        labels={"residuals_checkins": "Residual (actual - predicted)"},
    )
    st.plotly_chart(residual_fig, use_container_width=True)

st.subheader("Raw data peek")
st.dataframe(history.tail(200).set_index("timestamp"), use_container_width=True, height=360)
