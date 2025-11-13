"""Reusable layout primitives (top navigation, sidebar sections, etc.)."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
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
    NavItem("POS Console", "🧾", "pages/7_POS_Console.py"),
    NavItem("Price Manager", "💲", "pages/8_Price_Manager.py"),
    NavItem("Settings", "⚙️", "pages/5_Settings_APIs.py"),
    NavItem("Statistics", "📈", "pages/6_Statistics.py"),
]


def render_top_nav(
    active_page: str,
    nav_items: Iterable[NavItem] = DEFAULT_NAV,
    show_logo: bool = False,
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
                st.button(label, disabled=True, use_container_width=True)
            else:
                if st.button(label, key=f"nav-{item.path}", use_container_width=True):
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

def sidebar_info_block() -> None:
    """Standard sidebar header with team + data refresh details."""
    logo_bytes = get_logo_bytes()
    if logo_bytes:
        st.sidebar.image(logo_bytes, width=120)
    st.sidebar.markdown("**Refuel Ops**\n\nLive telemetry cockpit")
    st.sidebar.caption("Data updated every hour · Last refresh from notebook sync.")
    st.sidebar.divider()


def render_footer() -> None:
    st.markdown("---")
    st.markdown("University of St. Gallen (HSG) · Alice · Marie · Benjamin · Solal · Tristan")


def _ensure_tooltip_css() -> None:
    key = "_tooltip_css_injected"
    if st.session_state.get(key):
        return
    st.markdown(
        """
        <style>
        .tooltip-badge {
            position: relative;
            display: inline-block;
            cursor: help;
            color: #555;
            font-size: 0.9rem;
            border-bottom: 1px dotted #888;
            margin-left: 6px;
        }
        .tooltip-badge .tooltip-content {
            visibility: hidden;
            width: 280px;
            background-color: #262730;
            color: #fff;
            text-align: left;
            border-radius: 6px;
            padding: 8px 10px;
            position: absolute;
            z-index: 10;
            bottom: 125%;
            left: 0;
            opacity: 0;
            transition: opacity 0.2s;
            font-size: 0.8rem;
        }
        .tooltip-badge .tooltip-content::after {
            content: "";
            position: absolute;
            top: 100%;
            left: 12px;
            border-width: 5px;
            border-style: solid;
            border-color: #262730 transparent transparent transparent;
        }
        .tooltip-badge:hover .tooltip-content {
            visibility: visible;
            opacity: 1;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.session_state[key] = True


def hover_tip(label: str, tooltip: str) -> None:
    """Render a reusable hover tooltip badge."""
    _ensure_tooltip_css()
    html = f'<span class="tooltip-badge">{label}<span class="tooltip-content">{tooltip}</span></span>'
    st.markdown(html, unsafe_allow_html=True)
