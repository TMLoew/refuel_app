from pathlib import Path
import sys
import numpy as np
import plotly.express as px

# Ensure imports work both locally and on Streamlit Cloud
ROOT_DIR = Path(__file__).resolve().parents[3]
APP_DIR = Path(__file__).resolve().parent
for path in (ROOT_DIR, APP_DIR):
    if str(path) not in sys.path:
        sys.path.append(str(path))

import streamlit as st

from components.layout import render_top_nav, get_logo_bytes
from services.data_utils import load_enriched_data

st.set_page_config(page_title="Refuel Control Center", page_icon="🏠", layout="wide")

render_top_nav("Home.py")
logo_bytes = get_logo_bytes()
if logo_bytes:
    st.image(logo_bytes, width=140)
st.title("Refuel Control Center")
st.caption("Manage your snack availability subject to weather and gym attendance forecasts")

with st.sidebar:
    if logo_bytes:
        st.image(logo_bytes, width=120)
    st.markdown("Live telemetry cockpit")
    st.caption("Data updated every hour · Last refresh from notebook sync.")
    st.divider()
    st.subheader("Data slice")
    use_weather_api = st.toggle("Use live weather API", value=False, key="home-weather-toggle")

with st.spinner("Loading telemetry for preview..."):
    data = load_enriched_data(use_weather_api=use_weather_api)

if data.empty:
    st.sidebar.error("No telemetry data available yet. Upload a CSV to explore the cockpit.")
    st.stop()

total_days = max(1, int((data["timestamp"].max() - data["timestamp"].min()).days) or 1)
with st.sidebar:
    history_days = st.slider(
        "History window (days)",
        min_value=3,
        max_value=max(3, total_days),
        value=min(7, max(3, total_days)),
        key="home-history",
    )
    metric_focus = st.selectbox("Focus metric", ["checkins", "snack_units", "snack_revenue"], key="home-metric")
    weather_meta = data.attrs.get("weather_meta", {})
    if weather_meta:
        st.caption(
            f"Weather synced {weather_meta.get('updated_at', 'n/a')} UTC · coverage {weather_meta.get('coverage_start', '?')} → {weather_meta.get('coverage_end', '?')}"
        )

st.info(
    "Use the navigation bar to jump into specific tools. The sidebar controls above mirror the default data slice you can apply inside each module."
)

st.divider()
st.subheader("Snack Pricing & Elasticity Sandbox")

avg_price = float(data["snack_price"].mean())
avg_units = float(data["snack_units"].mean())
col_left, col_right = st.columns(2)
with col_left:
    elasticity = st.slider("Elasticity factor (negative means demand drops with price)", -3.0, 1.0, -1.2, step=0.1)
    price_range_pct = st.slider("Price adjustment range (%)", 10, 60, 30, step=5)
    promo_boost = st.slider("Promo boost on demand (%)", 0, 100, 15, step=5)

price_points = np.linspace(avg_price * (1 - price_range_pct / 100), avg_price * (1 + price_range_pct / 100), 40)
demand_curve = avg_units * (price_points / avg_price) ** elasticity * (1 + promo_boost / 100)
elasticity_fig = px.line(
    x=price_points,
    y=demand_curve,
    labels={"x": "Price (€)", "y": "Expected snack units"},
    title="Elasticity curve",
)
st.plotly_chart(elasticity_fig, use_container_width=True)

with col_right:
    current_stock = st.number_input("Current snack stock (units)", min_value=0.0, value=round(avg_units * 5, 1), step=10.0)
    safety_stock = st.number_input(
        "Safety stock threshold", min_value=0.0, value=round(avg_units * 2, 1), step=5.0
    )
    projected_units = demand_curve[-1]
    st.metric("Projected demand at highest price", f"{projected_units:.0f} units")
    if current_stock <= safety_stock:
        st.error("Low stock alert: inventory below safety threshold!")
    else:
        st.success("Stock level healthy. No alert triggered.")
