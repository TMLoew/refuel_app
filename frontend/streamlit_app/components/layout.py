"""Reusable layout primitives (top navigation, sidebar sections, etc.)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import streamlit as st


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
    """Render a top navigation bar using Streamlit's page links."""
    nav_items = list(nav_items)
    cols = st.columns(len(nav_items))
    for col, item in zip(cols, nav_items):
        with col:
            st.page_link(
                item.path,
                label=item.label,
                icon=item.emoji,
                use_container_width=True,
            )


def sidebar_info_block() -> None:
    """Standard sidebar header with team + data refresh details."""
    st.sidebar.image("https://static.streamlit.io/examples/dice.jpg", width=96)
    st.sidebar.markdown("**Refuel Ops**\n\nLive telemetry cockpit")
    st.sidebar.caption("Data updated every hour · Last refresh from notebook sync.")
    st.sidebar.divider()
