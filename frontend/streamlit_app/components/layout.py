"""Reusable layout primitives (top navigation, sidebar sections, etc.)."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, List, Optional

import plotly.express as px
import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx

px.defaults.template = "plotly_white"
px.defaults.color_discrete_sequence = [
    "#0B7A1F",  # primary green
    "#E97874",  # coral
    "#78A7B2",  # muted teal
    "#F7E24B",  # bright yellow
    "#0B5B2C",  # deep green
    "#000000",  # black
]
px.defaults.color_continuous_scale = ["#E6D8C0", "#0B7A1F"]


@dataclass(frozen=True)
class NavItem:
    label: str
    emoji: str
    path: str


DEFAULT_NAV: List[NavItem] = [
    NavItem("Home", "🏠", "Home.py"),
    NavItem("Dashboard", "📊", "pages/1_Dashboard.py"),
    NavItem("Forecasts", "🔮", "pages/2_Forecasts.py"),
    NavItem("Data Editor", "📝", "pages/4_Data_Editor.py"),
    NavItem("POS Console", "🧾", "pages/7_POS_Console.py"),
    NavItem("Price Manager", "💲", "pages/8_Price_Manager.py"),
    NavItem("Statistics", "📈", "pages/6_Statistics.py"),
    NavItem("What-if Lab", "🧪", "pages/3_WhatIf_Sim.py"),
    NavItem("Settings", "⚙️", "pages/5_Settings_APIs.py"),
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
    _inject_theme_css()
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

    st.markdown("<div class='refuel-top-nav'>", unsafe_allow_html=True)
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
        with col:
            label = f"{item.emoji} {item.label}"
            if item.path.endswith(active_page):
                st.button(
                    label=label,
                    disabled=True,
                    use_container_width=True,
                    key=f"nav-active-{item.path}",
                )
            else:
                if st.button(label, key=f"nav-{item.path}", use_container_width=True):
                    st.switch_page(item.path)
    st.markdown("</div>", unsafe_allow_html=True)


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
    st.markdown("[University of St. Gallen (HSG)](https://www.unisg.ch/en/) · Alice · Marie · Benjamin · Solal · Tristan")


def _inject_theme_css() -> None:
    key = "_refuel_theme_css"
    if st.session_state.get(key):
        return
    st.markdown(
        """
        <style>
        :root {
            --refuel-primary: #0b7a1f;
            --refuel-surface: #ffffff;
            --refuel-text: #0b1f1a;
            --refuel-pill-bg: #e6d8c0;
            --refuel-pill-fg: #0b1f1a;
            --refuel-pill-border: #0b5b2c;
            --refuel-accent-coral: #e97874;
            --refuel-accent-teal: #78a7b2;
            --refuel-accent-yellow: #f7e24b;
        }
        body, .stApp, div, span, label, button {
            font-family: "Gill Sans", "Gill Sans MT", Calibri, "Trebuchet MS", sans-serif !important;
        }
        body, .stApp {
            background-color: var(--refuel-surface);
            color: var(--refuel-text);
        }
        .refuel-top-nav div[data-testid="column"] {
            flex: 0 0 auto !important;
        }
        .refuel-top-nav div[data-testid="stButton"] > button {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            min-width: 120px;
            margin: 0 6px 8px 0 !important;
            height: 64px;
            background-color: var(--refuel-pill-bg);
            color: var(--refuel-pill-fg);
            border: 1px solid var(--refuel-pill-border);
            border-radius: 18px;
            box-shadow: none;
            width: auto;
        }
        .refuel-top-nav div[data-testid="stButton"] > button span {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            white-space: nowrap;
        }
        .refuel-top-nav div[data-testid="stButton"] > button:hover:not(:disabled) {
            border-color: var(--refuel-primary);
            color: var(--refuel-primary);
        }
        .refuel-top-nav div[data-testid="stButton"] > button:disabled {
            background-color: var(--refuel-primary);
            color: #ffffff;
            border-color: var(--refuel-primary);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.session_state[key] = True


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
