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

from frontend.streamlit_app.components.layout import (
    render_top_nav,
    sidebar_info_block,
    render_footer,
    get_logo_path,
    PRIMARY_GREEN,
    CORAL,
)
try:
    from frontend.streamlit_app.components.layout import hover_tip
except ImportError:
    try:
        from components.layout import hover_tip  # type: ignore
    except ImportError:
        def hover_tip(label: str, tooltip: str) -> None:
            st.caption(f"{label}: {tooltip}")
from frontend.streamlit_app.services.data_utils import load_enriched_data

PAGE_ICON = get_logo_path() or "📈"
st.set_page_config(page_title="Statistics", page_icon=PAGE_ICON, layout="wide")

render_top_nav("6_Statistics.py")
st.title("Trends & Drivers")
st.caption("Simple views to show how weather and visits affect snack sales.")

# Sidebar: pick live vs cached weather and how many days to analyze.
with st.sidebar:
    sidebar_info_block()
    st.subheader("Data slice")
    use_weather_api = st.toggle("Use live weather API", value=True)
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
col_a.metric("Latest temperature", f"{metrics_row['temperature_c']:.1f}°C", "Now")
col_b.metric("Daily rainfall", f"{window_df['precipitation_mm'].tail(96).sum():.1f} mm", "Past day")
col_c.metric("Check-ins (24h)", f"{window_df['checkins'].tail(96).sum():.0f}", "Past day")
col_d.metric("Snack units (24h)", f"{window_df['snack_units'].tail(96).sum():.0f}", "Past day")

overview_tab, drivers_tab, regression_tab, rhythm_tab = st.tabs(
    ["Overview", "Drivers", "Regression", "Daily rhythm"]
)

with overview_tab:
    # Quick correlations and highlights.
    st.subheader("Quick relationships")
    corr_cols = ["temperature_c", "precipitation_mm", "checkins", "snack_units"]
    corr_matrix = window_df[corr_cols].corr()
    corr_fig = px.imshow(
        corr_matrix,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale=[CORAL, "#FFFFFF", PRIMARY_GREEN],
        title="How variables move together",
    )
    st.plotly_chart(corr_fig, use_container_width=True)
    corr_pairs = (
        corr_matrix.where(~np.eye(len(corr_matrix), dtype=bool))
        .stack()
        .reset_index()
        .rename(columns={"level_0": "A", "level_1": "B", 0: "r"})
        .sort_values("r", key=lambda s: s.abs(), ascending=False)
    )
    top_corr = corr_pairs.head(3)
    bullets = [
        f"`{row['A']} ↗ {row['B']}` r = {row['r']:.2f}" for _, row in top_corr.iterrows()
    ]
    st.markdown("**Strongest moves in this window:** " + " · ".join(bullets))
    hover_tip(
        "How to read this",
        "Closer to +1 means they rise together, closer to -1 means they move opposite, around 0 means little link.",
    )

    st.info(
        "Use this to sanity-check: do warm, dry days lift visits and snacks as expected?"
    )

with drivers_tab:
    # Scatter views to eyeball relationships.
    st.subheader("How pairs of things move")
    pair_cols = ["temperature_c", "precipitation_mm", "checkins", "snack_units"]
    pair_fig = px.scatter_matrix(
        window_df,
        dimensions=pair_cols,
        color="weather_label",
        title="Raw pairs with weather classes",
    )
    st.plotly_chart(pair_fig, use_container_width=True)
    hover_tip(
        "What to look for",
        "Diagonal panels show distribution; off-diagonals show how clouds of points lean up or down.",
    )

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
    st.markdown(
        "- Warmer days mean more visits, especially on weekends.\n"
        "- Heavy rain usually lowers snack units unless it's just a light shower."
    )

with regression_tab:
    st.subheader("Quick impact readout")
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
        "- Larger numbers mean a bigger push up or down.\n"
        "- Small p-values hint the effect is likely real.\n"
        "- Warmer temps lift visits/snacks; steady rain pulls them down."
    )

with rhythm_tab:
    st.subheader("Day by day")
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
        title="Daily trend of weather, traffic, and snacks",
    )
    st.plotly_chart(daily_fig, use_container_width=True)
    st.markdown("Use this to line up staffing and replenishment with the next upswing.")

st.success(
    "Read this page top to bottom: check the quick moves, see the point clouds, confirm the impact table, then glance at day-by-day trends."
)
render_footer()
