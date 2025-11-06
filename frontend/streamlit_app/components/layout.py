"""Reusable layout primitives (top navigation, sidebar sections, etc.)."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, List, Optional

import plotly.io as pio
import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx


@dataclass(frozen=True)
class NavItem:
    label: str
    emoji: str
    path: str


DEFAULT_NAV: List[NavItem] = [
    NavItem("Home", "🏠", "Home.py"),
    NavItem("Dashboard", "📊", "pages/1_Dashboard.py"),
    NavItem("Forecasts", "🔮", "pages/2_Forecasts.py"),
    NavItem("What-if Lab", "🧪", "pages/3_WhatIf_Sim.py"),
    NavItem("Data Editor", "📝", "pages/4_Data_Editor.py"),
    NavItem("Settings", "⚙️", "pages/5_Settings_APIs.py"),
    NavItem("Statistics", "📈", "pages/6_Statistics.py"),
]


def render_top_nav(
    active_page: str,
    nav_items: Iterable[NavItem] = DEFAULT_NAV,
    show_logo: bool = True,
) -> None:
    """Render a top nav bar that switches pages without opening new tabs."""
    st.markdown(
        "<style>[data-testid='stSidebarNav']{display:none !important;}</style>",
        unsafe_allow_html=True,
    )
    ctx = get_script_run_ctx()
    if ctx is None:
        return
    pages = ctx.pages_manager.get_pages()
    apply_theme(st.session_state.get("ui_theme_mode", DEFAULT_THEME_MODE))
    logo_bytes = get_logo_bytes() if show_logo else None

    def lookup(path: str) -> Optional[dict]:
        for page in pages.values():
            if page["script_path"].endswith(path):
                return page
        return None

    if logo_bytes:
        wrapper_cols = st.columns([0.2, 0.8])
        wrapper_cols[0].image(logo_bytes, width=90)
        cols = wrapper_cols[1].columns(len(nav_items))
    else:
        cols = st.columns(len(nav_items))

    for col, item in zip(cols, nav_items):
        page_meta = lookup(item.path)
        if not page_meta:
            continue
        label = f"{item.emoji} {item.label}"
        with col:
            if item.path.endswith(active_page):
                st.button(label, use_container_width=True, disabled=True)
            else:
                if st.button(label, use_container_width=True, key=f"nav-{item.path}"):
                    st.switch_page(item.path)


LOGO_CANDIDATES = [
    Path(__file__).resolve().parents[3] / "logo.webp",
    Path(__file__).resolve().parents[3] / "frontend" / "logo.webp",
]


@lru_cache(maxsize=1)
def _resolve_logo_path() -> Optional[Path]:
    for logo_path in LOGO_CANDIDATES:
        if logo_path.exists():
            return logo_path
    return None


def get_logo_path() -> Optional[str]:
    resolved = _resolve_logo_path()
    return str(resolved) if resolved else None


def get_logo_bytes() -> Optional[bytes]:
    resolved = _resolve_logo_path()
    if resolved is None:
        return None
    return resolved.read_bytes()


DEFAULT_THEME_MODE = "light"
THEME_CSS = {
    "light": """
:root {
    --bg-color: #f6f8fc;
    --card-bg: #ffffff;
    --text-color: #1c2434;
    --muted-color: #6b7280;
    --accent-color: #f97316;
}
body, .stApp {
    background-color: var(--bg-color);
    color: var(--text-color);
}
[data-testid="stAppViewContainer"] {
    background-color: var(--bg-color);
}
[data-testid="stHeader"] {
    background-color: var(--bg-color);
    color: var(--text-color);
}
[data-testid="stSidebar"] {
    background-color: #ffffff !important;
    color: var(--text-color);
}
section.main > div {
    background-color: transparent;
}
.stMarkdown p, .stMarkdown span, .stMetric, label, h1, h2, h3, h4, h5, h6 {
    color: var(--text-color) !important;
}
.stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
    background-color: rgba(249,115,22,0.15);
    color: var(--text-color);
}
.stPlotlyChart {
    background-color: var(--card-bg) !important;
    border-radius: 12px;
    padding: 8px;
}
.stPlotlyChart div {
    background-color: transparent !important;
}
.theme-toggle .stRadio > label, .theme-toggle .stRadio div {
    justify-content: center;
}
.theme-toggle .stRadio label {
    padding-bottom: 4px;
}
""",
    "dark": """
:root {
    --bg-color: #0b1120;
    --card-bg: #111827;
    --text-color: #f1f5f9;
    --muted-color: #94a3b8;
    --accent-color: #f97316;
}
body, .stApp {
    background-color: var(--bg-color);
    color: var(--text-color);
}
[data-testid="stAppViewContainer"] {
    background-color: var(--bg-color);
}
[data-testid="stHeader"] {
    background-color: var(--bg-color);
    color: var(--text-color);
}
[data-testid="stSidebar"] {
    background-color: #0f172a !important;
    color: var(--text-color);
}
.stMarkdown p, .stMarkdown span, .stMetric, label, h1, h2, h3, h4, h5, h6 {
    color: var(--text-color) !important;
}
.stTabs [data-baseweb="tab-list"] button {
    color: var(--muted-color);
    background-color: rgba(255,255,255,0.05);
}
.stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
    color: var(--text-color);
    background-color: rgba(249,115,22,0.15);
}
.stDataFrame, .stTable, .stPlotlyChart, .stMetric {
    background-color: transparent;
}
.stPlotlyChart {
    background-color: #0f172a !important;
    border-radius: 12px;
    padding: 8px;
}
.stPlotlyChart div {
    background-color: transparent !important;
}
.theme-toggle .stRadio > label, .theme-toggle .stRadio div {
    justify-content: center;
}
""",
}


def apply_theme(mode: str) -> None:
    css = THEME_CSS.get(mode, THEME_CSS[DEFAULT_THEME_MODE])
    st.markdown(f"<style id='refuel-theme'>{css}</style>", unsafe_allow_html=True)
    if mode == "dark":
        pio.templates.default = "plotly_dark"
    else:
        pio.templates.default = "plotly_white"


def render_theme_toggle() -> None:
    mode = st.session_state.get("ui_theme_mode", DEFAULT_THEME_MODE)
    selection = st.radio(
        "Theme toggle",
        options=("light", "dark"),
        index=0 if mode == "light" else 1,
        horizontal=True,
        label_visibility="collapsed",
        key="theme-toggle",
        help="Toggle between light and dark mode",
        format_func=lambda opt: "🌞" if opt == "light" else "🌙",
    )
    new_mode = selection
    if new_mode != mode:
        st.session_state["ui_theme_mode"] = new_mode
    apply_theme(st.session_state.get("ui_theme_mode", DEFAULT_THEME_MODE))


def sidebar_info_block() -> None:
    """Standard sidebar header with team + data refresh details."""
    logo_bytes = get_logo_bytes()
    if logo_bytes:
        st.sidebar.image(logo_bytes, width=120)
    st.sidebar.markdown("**Refuel Ops**\n\nLive telemetry cockpit")
    st.sidebar.caption("Data updated every hour · Last refresh from notebook sync.")
    with st.sidebar:
        render_theme_toggle()
    st.sidebar.divider()


def render_footer() -> None:
    st.markdown("---")
    st.markdown("University of St. Gallen (HSG) · Tristan · Alice · Benjamin · Marie · Solal")
