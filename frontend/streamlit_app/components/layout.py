"""Reusable layout primitives (top navigation, sidebar sections, etc.)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import streamlit as st


@dataclass(frozen=True)
class NavItem:
    label: str
    page: str
    emoji: str = ""

    @property
    def title(self) -> str:
        return f"{self.emoji} {self.label}".strip()


DEFAULT_NAV: List[NavItem] = [
    NavItem("Home", "frontend/streamlit_app/Home.py", "🏠"),
    NavItem("Dashboard", "frontend/streamlit_app/1_Dashboard.py", "📊"),
    NavItem("Forecasts", "frontend/streamlit_app/2_Forecasts.py", "🔮"),
    NavItem("What-if Lab", "frontend/streamlit_app/3_WhatIf_Sim.py", "🧪"),
    NavItem("Data Editor", "frontend/streamlit_app/4_Data_Editor.py", "📝"),
]


def render_top_nav(active_page: str, nav_items: Iterable[NavItem] = DEFAULT_NAV) -> None:
    """Render a top navigation bar with buttons that switch pages."""
    nav_items = list(nav_items)
    col_width = 1 / len(nav_items) if nav_items else 1
    cols = st.columns([col_width] * len(nav_items), gap="small")

    for col, item in zip(cols, nav_items):
        btn_type = "primary" if item.page.endswith(active_page) else "secondary"
        with col:
            if st.button(
                item.title,
                key=f"nav-{item.page}",
                use_container_width=True,
                type=btn_type,
            ):
                st.switch_page(item.page)


def sidebar_info_block() -> None:
    """Standard sidebar header with team + data refresh details."""
    st.sidebar.image("https://static.streamlit.io/examples/dice.jpg", width=96)
    st.sidebar.markdown("**Refuel Ops**\n\nLive telemetry cockpit")
    st.sidebar.caption("Data updated every hour · Last refresh from notebook sync.")
    st.sidebar.divider()
