import streamlit as st

try:
    from frontend.streamlit_app.components.layout import render_top_nav
except ModuleNotFoundError:
    from components.layout import render_top_nav

st.set_page_config(page_title="Refuel Control Center", page_icon="🏠", layout="wide")

render_top_nav("Home.py")
st.title("Refuel Control Center")
st.caption("Choose a workspace from the left-hand navigation to explore live telemetry and planning tools.")

st.divider()
st.subheader("Where to start")
st.markdown(
    """
- **Integrated Performance Dashboard** → full view of weather, gym flow, and snack demand.
- **Forecast Explorer** → diagnose the ML/regression outputs behind the scenario slider.
- **What-if Simulator** → experiment with staffing, campaigns, and pricing knobs.
- **Data Editor & Settings** → update source files or API keys.
"""
)

st.info("Use the sidebar navigation to switch between modules. Start with the dashboard to see a live snapshot.")
