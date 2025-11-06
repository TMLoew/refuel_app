from pathlib import Path
import sys
import json
from datetime import datetime

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import streamlit as st

from frontend.streamlit_app.components.layout import render_top_nav, sidebar_info_block
from frontend.streamlit_app.services.weather_pipeline import DEFAULT_LAT, DEFAULT_LON

st.set_page_config(page_title="Settings & APIs", page_icon="⚙️", layout="wide")

render_top_nav("5_Settings_APIs.py")
st.title("Settings & API Console")
st.caption("Manage external data hooks, monitor credentials, and run quick health checks.")

with st.sidebar:
    sidebar_info_block()
    st.subheader("Profiles")
    active_env = st.selectbox("Environment", ["Staging", "Production"])
    st.caption(f"Selected profile: **{active_env}**")

st.subheader("Weather API configuration")
with st.form("weather-form"):
    col1, col2 = st.columns(2)
    with col1:
        lat = st.number_input("Latitude", value=DEFAULT_LAT, step=0.1, format="%.4f")
        api_timeout = st.number_input("Timeout (seconds)", min_value=1, max_value=60, value=10)
    with col2:
        lon = st.number_input("Longitude", value=DEFAULT_LON, step=0.1, format="%.4f")
        cache_hours = st.slider("Cache horizon (hours)", 1, 24, 6)
    submitted = st.form_submit_button("Save weather profile", use_container_width=True)
    if submitted:
        st.success(
            f"Saved weather coordinates ({lat:.4f}, {lon:.4f}) with timeout={api_timeout}s and cache={cache_hours}h."
        )

st.subheader("API health")
health_cols = st.columns(3)
health_cols[0].metric("Weather API", "✅ OK", delta="latency 180 ms")
health_cols[1].metric("Gym sensors", "⚠️ Delay", delta="+18 min")
health_cols[2].metric("POS snacks", "✅ OK", delta="live")

st.subheader("Secrets & tokens")
with st.expander("Current tokens (redacted)", expanded=False):
    token_store = {
        "open_meteo": "sk-***meteo",
        "gym_webhook": "whsec-***123",
        "pos_service": "pat-***refuel",
        "last_rotation": datetime.utcnow().isoformat(),
    }
    st.json(token_store)
    st.info("Manage actual secrets via your deployment platform; this panel is a placeholder for ops runbooks.")

st.subheader("Webhooks")
webhooks = [
    {"name": "Gym turnstiles", "status": "active", "last_event": "2 min ago"},
    {"name": "Snack POS", "status": "active", "last_event": "5 min ago"},
    {"name": "Marketing automation", "status": "paused", "last_event": "3 days ago"},
]
st.table(webhooks)

st.subheader("Export settings")
export_blob = json.dumps({"env": active_env, "lat": float(lat), "lon": float(lon)}, indent=2)
st.download_button("Download config JSON", export_blob, file_name="refuel_config.json", mime="application/json")
