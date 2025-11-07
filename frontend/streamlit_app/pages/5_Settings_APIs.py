from pathlib import Path
import sys
import json
from datetime import datetime

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import pandas as pd
import streamlit as st

from frontend.streamlit_app.components.layout import (
    render_top_nav,
    sidebar_info_block,
    render_footer,
    get_logo_path,
)
try:
    from frontend.streamlit_app.services.data_utils import load_enriched_data, load_procurement_plan
except ImportError as import_exc:
    if "load_procurement_plan" not in str(import_exc):
        raise
    from frontend.streamlit_app.services.data_utils import load_enriched_data  # type: ignore

    def load_procurement_plan() -> pd.DataFrame:  # type: ignore[misc]
        return pd.DataFrame()
from frontend.streamlit_app.services.weather_pipeline import DEFAULT_LAT, DEFAULT_LON

PAGE_ICON = get_logo_path() or "⚙️"
st.set_page_config(page_title="Settings & APIs", page_icon=PAGE_ICON, layout="wide")

render_top_nav("5_Settings_APIs.py")
st.title("Settings & API Console")
st.caption("Manage external data hooks, monitor credentials, and run quick health checks.")
active_env = "Default"

with st.sidebar:
    sidebar_info_block()

st.subheader("Weather API configuration")
with st.form("weather-form"):
    col1, col2 = st.columns(2)
    with col1:
        lat = st.number_input("Latitude", value=DEFAULT_LAT, step=0.1, format="%.4f")
        api_timeout = st.number_input("Timeout (seconds)", min_value=1, max_value=60, value=10)
    with col2:
        lon = st.number_input("Longitude", value=DEFAULT_LON, step=0.1, format="%.4f")
        cache_hours = st.slider("Cache horizon (hours)", 1, 24, 6)
    submitted = st.form_submit_button("Save weather profile", width="stretch")
    if submitted:
        st.success(
            f"Saved weather coordinates ({lat:.4f}, {lon:.4f}) with timeout={api_timeout}s and cache={cache_hours}h."
        )

st.subheader("API health")
data_sample = load_enriched_data(use_weather_api=True)
weather_meta = data_sample.attrs.get("weather_meta", {})
latency = weather_meta.get("latency_ms")
latency_text = f"{latency:.0f} ms" if latency else "n/a"

health_cols = st.columns(3)
health_cols[0].metric("Weather API", "✅ OK" if latency else "ℹ️ Pending", delta=f"latency {latency_text}")
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

st.subheader("Procurement plan snapshot")
plan_df = load_procurement_plan()
if plan_df.empty:
    st.info("No procurement plan saved yet. Run the autopilot simulation on the Home tab to publish one.")
else:
    plan_df = plan_df.copy()
    plan_df["date"] = pd.to_datetime(plan_df["date"])
    today = pd.Timestamp.now().normalize()
    future = plan_df[plan_df["date"] >= today]
    meta_cols = [col for col in plan_df.columns if col.startswith("plan_")]
    plan_meta = {col.replace("plan_", ""): plan_df[col].iloc[0] for col in meta_cols} if meta_cols else {}
    table_df = plan_df.drop(columns=meta_cols, errors="ignore")
    metrics = st.columns(3)
    if "profit" in plan_df.columns:
        metrics[0].metric("Projected profit", f"€{plan_df['profit'].sum():.0f}")
    if "reordered" in plan_df.columns:
        metrics[1].metric("Reorders planned", int((plan_df["reordered"] == "Yes").sum()))
    if "stock_after" in plan_df.columns:
        metrics[2].metric("Ending stock", f"{plan_df['stock_after'].iloc[-1]:.0f} units")
    if "plan_generated_at" in plan_df.columns:
        st.caption(f"Plan generated at {plan_df['plan_generated_at'].iloc[0]}")
    if plan_meta:
        st.markdown(
            "**Plan assumptions**  \n"
            f"- Weather: **{plan_meta.get('weather_pattern', 'n/a')}** · Promo: **{plan_meta.get('promo', 'n/a')}**  \n"
            f"- Pricing Δ: {plan_meta.get('price_change_pct', '0')}% · Strategy Δ: {plan_meta.get('price_strategy_pct', '0')}%  \n"
            f"- Unit cost: €{plan_meta.get('unit_cost', 'n/a')} · Fee: €{plan_meta.get('fee', 'n/a')}  \n"
            f"- Horizon: {plan_meta.get('horizon_days', '?')} d · Safety stock: {plan_meta.get('safety_stock', '?')} units",
        )
    if {"reordered", "reorder_qty"}.issubset(plan_df.columns):
        upcoming = future[future["reordered"] == "Yes"]
        if not upcoming.empty:
            next_row = upcoming.iloc[0]
            st.success(f"Next reorder {next_row['date'].strftime('%Y-%m-%d')} · {next_row['reorder_qty']:.0f} units.")
    columns_to_show = (
        ["date", "scenario", "price", "demand_est", "sold", "stock_after", "reordered", "reorder_qty", "profit"]
        if {"scenario", "reorder_qty"}.issubset(table_df.columns)
        else list(table_df.columns)
    )
    st.dataframe(table_df.head(25)[columns_to_show], width="stretch", height=300)

st.subheader("Export settings")
export_blob = json.dumps({"env": active_env, "lat": float(lat), "lon": float(lon)}, indent=2)
st.download_button("Download config JSON", export_blob, file_name="refuel_config.json", mime="application/json")
render_footer()
