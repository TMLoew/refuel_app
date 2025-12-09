from pathlib import Path
import sys
import io

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
from frontend.streamlit_app.services.data_utils import DATA_FILE, load_enriched_data, train_models
MODEL_DIR = ROOT_DIR / "model"
CHECKIN_MODEL_FILE = MODEL_DIR / "checkins_hgb.joblib"
SNACK_MODEL_FILE = MODEL_DIR / "snacks_hgb.joblib"

PAGE_ICON = get_logo_path() or "📝"
st.set_page_config(page_title="Data Workbench", page_icon=PAGE_ICON, layout="wide")

render_top_nav("4_Data_Editor.py")
st.title("Data Workbench")
st.caption("Swap in a new CSV, check a sample of rows, and retrain the simple forecasts.")

st.subheader("Upload replacement data")
uploaded = st.file_uploader("Drop a new `gym_badges.csv` to preview it", type=["csv"])
if uploaded:
    new_df = pd.read_csv(uploaded)
    st.success(f"Loaded {len(new_df)} rows from your upload.")
    st.dataframe(new_df.head(20), use_container_width=True)
    st.caption(f"This will replace: `{DATA_FILE}`")
    if st.button("Make this the live dataset", type="primary"):
        try:
            DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
            new_df.to_csv(DATA_FILE, index=False)
        except Exception as exc:  # pragma: no cover - streamlit interaction
            st.error(f"Could not save file: {exc}")
        else:
            st.success("Saved. Refresh the app to rebuild dashboards on this file.")

with st.sidebar:
    sidebar_info_block()
    st.subheader("Filters")
    use_weather_api = st.toggle("Rebuild with live weather", value=False, key="editor-weather")
    sample_rows = st.slider("Rows to display", 50, 500, 200, step=50)

data = load_enriched_data(use_weather_api=use_weather_api)
if data.empty:
    st.error("Upload `data/gym_badges.csv` to start exploring the dataset.")
    st.stop()

ts_min = data["timestamp"].min()
ts_max = data["timestamp"].max()
date_min, date_max = ts_min.to_pydatetime(), ts_max.to_pydatetime()
selected_range = st.slider(
    "Select date window",
    min_value=date_min,
    max_value=date_max,
    value=((ts_max - pd.Timedelta(days=3)).to_pydatetime(), date_max),
    format="YYYY-MM-DD",
)
mask = data["timestamp"].between(pd.Timestamp(selected_range[0]), pd.Timestamp(selected_range[1]))

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

st.subheader("Forecast models")
st.caption("Retrain the lightweight models so they match the current data.")
retrain_cols = st.columns([0.4, 0.6])
with retrain_cols[0]:
    retrain_clicked = st.button("🔄 Retrain models on current data", type="primary")
with retrain_cols[1]:
    st.caption(f"Model files: `{CHECKIN_MODEL_FILE.name}`, `{SNACK_MODEL_FILE.name}`")

if retrain_clicked:
    data_for_training = load_enriched_data(use_weather_api=True, cache_buster=pd.Timestamp.utcnow().timestamp())
    if data_for_training.empty:
        st.error("No data available to train. Upload telemetry first.")
    else:
        for model_path in (CHECKIN_MODEL_FILE, SNACK_MODEL_FILE):
            try:
                if model_path.exists():
                    model_path.unlink()
            except Exception as exc:  # pragma: no cover - streamlit interaction
                st.warning(f"Could not remove {model_path.name}: {exc}")
        try:
            train_models.clear()  # type: ignore[attr-defined]
        except Exception:
            pass
        with st.spinner("Training attendance and snack models..."):
            try:
                models = train_models(data_for_training)
            except Exception as exc:  # pragma: no cover - streamlit interaction
                st.error(f"Training failed: {exc}")
                models = (None, None)
        if models and all(models):
            st.success("Models retrained and saved. Forecasts will now use the new fit.")
        else:
            st.warning("Models were not produced. Check logs and dataset completeness.")

render_footer()
