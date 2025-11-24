"""Reusable layout primitives (top navigation, sidebar sections, etc.)."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, List, Optional
import html

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx

PRIMARY_GREEN = "#0B7A1F"
DEEP_GREEN = "#0B5B2C"
CORAL = "#E97874"
TEAL = "#78A7B2"
YELLOW = "#F7E24B"
SAND = "#E6D8C0"
INK = "#0B1F1A"

px.defaults.color_discrete_sequence = [PRIMARY_GREEN, CORAL, TEAL, YELLOW, DEEP_GREEN, "#000000"]
px.defaults.color_continuous_scale = [SAND, PRIMARY_GREEN]

refuel_template = go.layout.Template(
    layout=dict(
        font=dict(family="Gill Sans,Gill Sans MT,Calibri,Trebuchet MS,sans-serif", color=INK),
        title=dict(font=dict(family="Gill Sans,Gill Sans MT,Calibri,Trebuchet MS,sans-serif", color=INK)),
        colorway=px.defaults.color_discrete_sequence,
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        xaxis=dict(
            gridcolor=SAND,
            zerolinecolor=SAND,
            linecolor=DEEP_GREEN,
            title=dict(font=dict(color=INK)),
            tickfont=dict(color=INK),
        ),
        yaxis=dict(
            gridcolor=SAND,
            zerolinecolor=SAND,
            linecolor=DEEP_GREEN,
            title=dict(font=dict(color=INK)),
            tickfont=dict(color=INK),
        ),
        legend=dict(
            bgcolor="#FFFFFF",
            bordercolor=SAND,
            borderwidth=0.5,
            font=dict(color=INK),
        ),
    )
)
pio.templates["refuel"] = refuel_template
pio.templates.default = "refuel"
px.defaults.template = "refuel"


@dataclass(frozen=True)
class NavItem:
    label: str
    emoji: str
    path: str


DEFAULT_NAV: List[NavItem] = [
    NavItem("Home", "📊", "pages/1_Dashboard.py"),
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
            --refuel-primary: #0B7A1F;
            --refuel-surface: #ffffff;
            --refuel-text: #0B1F1A;
            --refuel-pill-bg: #E6D8C0;
            --refuel-pill-fg: #0B1F1A;
            --refuel-pill-border: #0B5B2C;
            --refuel-accent-coral: #E97874;
            --refuel-accent-teal: #78A7B2;
            --refuel-accent-yellow: #F7E24B;
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
            display: inline-flex;
            align-items: center;
            cursor: help;
            font-size: 0.9rem;
            border-bottom: 1px dotted var(--refuel-pill-border, #0b5b2c);
            margin-left: 6px;
            color: var(--refuel-text, #0b1f1a);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.session_state[key] = True


def hover_tip(label: str, tooltip: str) -> None:
    """Render a reusable hover tooltip badge."""
    _ensure_tooltip_css()
    safe_label = html.escape(label)
    safe_tip = html.escape(tooltip)
    html_snippet = f'<span class="tooltip-badge" title="{safe_tip}">{safe_label}</span>'
    st.markdown(html_snippet, unsafe_allow_html=True)
