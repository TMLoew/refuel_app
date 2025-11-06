import streamlit as st

try:
    from frontend.streamlit_app.components.layout import render_top_nav
except ModuleNotFoundError:
    from components.layout import render_top_nav

st.set_page_config(page_title="Refuel Control Center", page_icon="🏠", layout="wide")

render_top_nav("Home.py")
st.title("Refuel Control Center")
st.caption("Choose a workspace from the left-hand navigation to explore live telemetry and planning tools.")
