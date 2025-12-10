from pathlib import Path
import sys

import streamlit as st

# Ensure imports work both locally and on Streamlit Cloud
ROOT_DIR = Path(__file__).resolve().parents[3]
APP_DIR = Path(__file__).resolve().parent
for path in (ROOT_DIR, APP_DIR):
    if str(path) not in sys.path:
        sys.path.append(str(path))

# Try absolute import first, then fall back so the app runs both locally and on Streamlit Cloud.
try:
    from frontend.streamlit_app.components.layout import get_logo_path
except ModuleNotFoundError:
    try:
        from components.layout import get_logo_path  # type: ignore
    except ModuleNotFoundError:
        def get_logo_path():  # type: ignore
            return None

PAGE_ICON = get_logo_path() or "📊"
st.set_page_config(page_title="Refuel Control Center", page_icon=PAGE_ICON, layout="wide")

# Keep this file tiny: set config, show a message, and redirect to the dashboard.
st.write("Redirecting to the dashboard...")
st.switch_page("1_Dashboard.py")
