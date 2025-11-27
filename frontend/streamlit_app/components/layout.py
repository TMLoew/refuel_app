"""Reusable layout primitives (top navigation, sidebar sections, etc.)."""

from __future__ import annotations

import base64
from dataclasses import dataclass
import os
from functools import lru_cache
from pathlib import Path
from typing import Iterable, List, Optional
import html

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

LIGHT_PRIMARY_GREEN = "#0B7A1F"
LIGHT_DEEP_GREEN = "#0B5B2C"
LIGHT_CORAL = "#E97874"
LIGHT_TEAL = "#78A7B2"
LIGHT_YELLOW = "#F7E24B"
LIGHT_SAND = "#E6D8C0"
LIGHT_INK = "#0B1F1A"

DARK_PRIMARY_GREEN = "#5AD38B"
DARK_DEEP_GREEN = "#2E8B57"
DARK_CORAL = "#F5A6A6"
DARK_TEAL = "#A3D3DC"
DARK_YELLOW = "#FFF68F"
DARK_SAND = "#3A4252"
DARK_INK = "#F5F7FA"

# Backwards-compatible aliases used by some pages
PRIMARY_GREEN = LIGHT_PRIMARY_GREEN
DEEP_GREEN = LIGHT_DEEP_GREEN
CORAL = LIGHT_CORAL
YELLOW = LIGHT_YELLOW

px.defaults.color_discrete_sequence = [LIGHT_PRIMARY_GREEN, LIGHT_CORAL, LIGHT_TEAL, LIGHT_YELLOW, LIGHT_DEEP_GREEN, "#000000"]
px.defaults.color_continuous_scale = [LIGHT_SAND, LIGHT_PRIMARY_GREEN]

refuel_template = go.layout.Template(
    layout=dict(
        font=dict(family="Gill Sans,Gill Sans MT,Calibri,Trebuchet MS,sans-serif", color=LIGHT_INK),
        title=dict(font=dict(family="Gill Sans,Gill Sans MT,Calibri,Trebuchet MS,sans-serif", color=LIGHT_INK)),
        colorway=px.defaults.color_discrete_sequence,
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        xaxis=dict(
            gridcolor=LIGHT_SAND,
            zerolinecolor=LIGHT_SAND,
            linecolor=LIGHT_DEEP_GREEN,
            title=dict(font=dict(color=LIGHT_INK)),
            tickfont=dict(color=LIGHT_INK),
        ),
        yaxis=dict(
            gridcolor=LIGHT_SAND,
            zerolinecolor=LIGHT_SAND,
            linecolor=LIGHT_DEEP_GREEN,
            title=dict(font=dict(color=LIGHT_INK)),
            tickfont=dict(color=LIGHT_INK),
        ),
        legend=dict(
            bgcolor="#FFFFFF",
            bordercolor=LIGHT_SAND,
            borderwidth=0.5,
            font=dict(color=LIGHT_INK),
        ),
    )
)
pio.templates["refuel_light"] = refuel_template

refuel_dark_template = go.layout.Template(
    layout=dict(
        font=dict(family="Gill Sans,Gill Sans MT,Calibri,Trebuchet MS,sans-serif", color=DARK_INK),
        title=dict(font=dict(family="Gill Sans,Gill Sans MT,Calibri,Trebuchet MS,sans-serif", color=DARK_INK)),
        colorway=[DARK_PRIMARY_GREEN, DARK_CORAL, DARK_TEAL, DARK_YELLOW, DARK_DEEP_GREEN, "#FFFFFF"],
        paper_bgcolor="#111219",
        plot_bgcolor="#111219",
        xaxis=dict(
            gridcolor=DARK_SAND,
            zerolinecolor=DARK_SAND,
            linecolor=DARK_DEEP_GREEN,
            title=dict(font=dict(color=DARK_INK)),
            tickfont=dict(color=DARK_INK),
        ),
        yaxis=dict(
            gridcolor=DARK_SAND,
            zerolinecolor=DARK_SAND,
            linecolor=DARK_DEEP_GREEN,
            title=dict(font=dict(color=DARK_INK)),
            tickfont=dict(color=DARK_INK),
        ),
        legend=dict(
            bgcolor="#111219",
            bordercolor=DARK_SAND,
            borderwidth=0.5,
            font=dict(color=DARK_INK),
        ),
    )
)
pio.templates["refuel_dark"] = refuel_dark_template

def _apply_plotly_theme() -> None:
    theme = st.get_option("theme.base") if hasattr(st, "get_option") else "light"
    if theme == "dark":
        px.defaults.template = "refuel_dark"
    else:
        px.defaults.template = "refuel_light"

_apply_plotly_theme()


@dataclass(frozen=True)
class NavItem:
    label: str
    emoji: str
    path: str


DEFAULT_NAV: List[NavItem] = [
    NavItem("Home", "📊", "1_Dashboard.py"),
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
    """Inject theme CSS and position the brand block; navigation happens via sidebar."""
    _inject_theme_css()
    _enforce_cookie_consent()


LOGO_CANDIDATES = [
    Path(__file__).resolve().parents[3] / "logo.webp",
    Path(__file__).resolve().parents[3] / "frontend" / "logo.webp",
    Path(__file__).resolve().parents[3] / "logo.jpeg",
    Path(__file__).resolve().parents[3] / "frontend" / "logo.jpeg",
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
    """Render the brand block at the sidebar top."""
    logo_bytes = get_logo_bytes()
    logo_html = ""
    if logo_bytes:
        encoded = base64.b64encode(logo_bytes).decode("utf-8")
        logo_html = f"<img src='data:image/png;base64,{encoded}' alt='Refuel logo' />"
    block = f"""
    <div class="refuel-sidebar-brand">
        {logo_html}
        <strong>Refuel Ops</strong>
        <span>Live telemetry cockpit</span>
        <span>Data updated every hour · Last refresh from notebook sync.</span>
    </div>
    """
    st.sidebar.markdown(block, unsafe_allow_html=True)


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
        [data-theme="dark"] {
            --refuel-surface: #111219;
            --refuel-text: #F5F7FA;
            --refuel-pill-bg: #1f2633;
            --refuel-pill-fg: #F5F7FA;
            --refuel-pill-border: #3a4252;
        }
        body, .stApp, div, span, label, button {
            font-family: "Gill Sans", "Gill Sans MT", Calibri, "Trebuchet MS", sans-serif !important;
        }
        body, .stApp {
            background-color: var(--refuel-surface);
            color: var(--refuel-text);
        }
        html[data-theme="dark"],
        html[data-theme="dark"] body,
        html[data-theme="dark"] .stApp,
        html[data-theme="dark"] .stAppViewContainer,
        html[data-theme="dark"] .main,
        html[data-theme="dark"] .block-container {
            background-color: #111219 !important;
            color: #F5F7FA !important;
        }
section[data-testid="stSidebar"] {
            position: relative;
            padding-top: 170px !important;
        }
.refuel-sidebar-brand {
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            padding: 16px 12px;
            text-align: center;
            background: var(--refuel-surface,#ffffff);
            border-bottom: 1px solid var(--refuel-pill-border,#0B5B2C);
            z-index: 2000;
        }
nav[data-testid="stSidebarNav"], div[data-testid="stSidebarNav"] {
            margin-top: 0 !important;
        }
.refuel-sidebar-brand img {
            width: 120px;
            display: block;
            margin: 0 auto 8px;
        }
.refuel-sidebar-brand span {
            display: block;
            font-size: 0.9rem;
            color: var(--refuel-text,#0B1F1A);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.session_state[key] = True


def _cookie_flag() -> tuple[bool, str]:
    secret_keys = [
        "COOKIE_POPUP_ENABLED",
        "COOKIE_POPUP",
        "cookie_popup_enabled",
        "cookie_popup",
        "cookie_banner",
    ]
    raw_flag = None
    source = "none"
    try:
        for key in secret_keys:
            if key in st.secrets:
                raw_flag = st.secrets.get(key)
                source = f"secret:{key}"
                break
    except Exception:
        raw_flag = None
    if raw_flag is None:
        for key in secret_keys:
            if key in os.environ:
                raw_flag = os.environ.get(key)
                source = f"env:{key}"
                break
    enabled = False
    if raw_flag is not None:
        enabled = str(raw_flag).strip().lower() not in {"0", "false", "no", "off", ""}
    return enabled, source


def _enforce_cookie_consent() -> None:
    enabled, _source = _cookie_flag()
    if not enabled:
        return
    consent_key = "cookie_popup_choice"
    if st.session_state.get(consent_key):
        return
    try:
        modal = st.modal("Cookies & telemetry", key="cookie-modal", width=520)
    except Exception:
        modal = None
    if modal:
        with modal:
            st.write(
                "We use essential cookies for session integrity and optional telemetry to improve the experience. "
                "You can decline non-essential tracking."
            )
            col_a, col_b = st.columns(2, gap="medium")
            accept = col_a.button("Accept all", key="cookie-accept", use_container_width=True)
            decline = col_b.button("Decline non-essential", key="cookie-decline", use_container_width=True)
            st.caption("Please choose an option to continue.")
            if accept:
                st.session_state[consent_key] = "accepted"
                st.rerun()
            if decline:
                st.session_state[consent_key] = "declined"
                st.rerun()
    else:
        banner = st.container(border=True)
        banner.subheader("Cookies & telemetry")
        banner.write(
            "We use essential cookies for session integrity and optional telemetry to improve the experience. "
            "You can decline non-essential tracking."
        )
        col_a, col_b = banner.columns(2)
        accept = col_a.button("Accept all", key="cookie-accept")
        decline = col_b.button("Decline non-essential", key="cookie-decline")
        if accept:
            st.session_state[consent_key] = "accepted"
            st.rerun()
        if decline:
            st.session_state[consent_key] = "declined"
            st.rerun()
        st.stop()


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
