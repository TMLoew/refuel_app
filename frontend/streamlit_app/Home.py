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
st.caption("Manage your snack avaailability subject to weather and gym attendance forecasts")
