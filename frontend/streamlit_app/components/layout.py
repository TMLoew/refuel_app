"""Reusable layout primitives (top navigation, sidebar sections, etc.)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

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
]


def render_top_nav(active_page: str, nav_items: Iterable[NavItem] = DEFAULT_NAV) -> None:
    """Render a top nav bar that switches pages without opening new tabs."""
    st.markdown(
        "<style>[data-testid='stSidebarNav']{display:none !important;}</style>",
        unsafe_allow_html=True,
    )
    ctx = get_script_run_ctx()
    if ctx is None:
        return
    pages = ctx.pages_manager.get_pages()

    def lookup(path: str) -> Optional[dict]:
        for page in pages.values():
            if page["script_path"].endswith(path):
                return page
        return None

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


LOGO_PATH = Path(__file__).resolve().parents[2] / "logo.webp"


def sidebar_info_block() -> None:
    """Standard sidebar header with team + data refresh details."""
    if LOGO_PATH.exists():
        st.sidebar.image(str(LOGO_PATH), width=120)
    st.sidebar.markdown("**Refuel Ops**\n\nLive telemetry cockpit")
    st.sidebar.caption("Data updated every hour · Last refresh from notebook sync.")
    st.sidebar.divider()
