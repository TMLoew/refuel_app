from pathlib import Path
import sys
import io

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import pandas as pd
import streamlit as st

from frontend.streamlit_app.components.layout import render_top_nav, sidebar_info_block
from frontend.streamlit_app.services.data_utils import DATA_FILE, load_enriched_data

st.set_page_config(page_title="Data Workbench", page_icon="📝", layout="wide")

render_top_nav("4_Data_Editor.py")
st.title("Data Workbench")
st.caption("Inspect, annotate, and experiment with the telemetry tables feeding the dashboards.")

with st.sidebar:
    sidebar_info_block()
    st.subheader("Filters")
    use_weather_api = st.toggle("Rebuild with live weather", value=False, key="editor-weather")
    sample_rows = st.slider("Rows to display", 50, 500, 200, step=50)

data = load_enriched_data(use_weather_api=use_weather_api)
if data.empty:
    st.error("Upload `data/gym_badges.csv` to start exploring the dataset.")
    st.stop()

date_min, date_max = data["timestamp"].min(), data["timestamp"].max()
selected_range = st.slider(
    "Select date window",
    min_value=date_min,
    max_value=date_max,
    value=(date_max - pd.Timedelta(days=3), date_max),
    format="YYYY-MM-DD",
)
mask = data["timestamp"].between(*selected_range)

st.subheader("Editable grid")
edited = st.data_editor(
    data.loc[mask].head(sample_rows).set_index("timestamp"),
    use_container_width=True,
    num_rows="dynamic",
    height=420,
    key="editor-grid",
)

st.caption("Changes above are session-only; export below if you want to persist.")

buffer = io.StringIO()
edited.reset_index().to_csv(buffer, index=False)
st.download_button(
    "Download edited slice as CSV",
    buffer.getvalue(),
    file_name="refuel_edited_slice.csv",
    mime="text/csv",
)

with st.expander("Summary stats", expanded=True):
    summary = data.describe()[["checkins", "snack_units", "snack_revenue", "temperature_c"]]
    st.dataframe(summary, use_container_width=True)

st.subheader("Upload replacement data")
uploaded = st.file_uploader("Upload new `gym_badges.csv`", type=["csv"])
if uploaded:
    new_df = pd.read_csv(uploaded)
    st.success(f"Loaded {len(new_df)} rows. Replace `{DATA_FILE}` manually to use this dataset.")
    st.dataframe(new_df.head(20), use_container_width=True)
