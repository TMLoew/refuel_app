from pathlib import Path
import sys

# Ensure repo root on sys.path for Streamlit Cloud
ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import streamlit as st

try:
    from frontend.streamlit_app.components.layout import render_top_nav
except ModuleNotFoundError:
    from components.layout import render_top_nav

st.set_page_config(page_title="Refuel Control Center", page_icon="🏠", layout="wide")

render_top_nav("Home.py")
st.title("Refuel Control Center")
st.caption("Manage your snack availability subject to weather and gym attendance forecasts")

with st.sidebar:
    st.image("https://static.streamlit.io/examples/dice.jpg", width=96)
    st.markdown("**Refuel Ops**\n\nLive telemetry cockpit")
    st.caption("Data updated every hour · Last refresh from notebook sync.")
    st.divider()
    st.subheader("Data slice")
    use_weather_api = st.toggle("Use live weather API", value=False, key="home-weather-toggle")
    history_days = st.slider("History window (days)", min_value=3, max_value=30, value=7, key="home-history")
    metric_focus = st.selectbox("Focus metric", ["checkins", "snack_units", "snack_revenue"], key="home-metric")

st.info(
    "Use the navigation bar to jump into specific tools. The sidebar controls above mirror the default data slice you can apply inside each module."
)
