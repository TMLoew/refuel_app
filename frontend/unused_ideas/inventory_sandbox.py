"""
Archived Inventory Sandbox experiment. Move this file back into an active page
if we ever want to revive the interactive game.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st


def render_inventory_sandbox(df: pd.DataFrame) -> None:
    st.subheader("Inventory sandbox")
    daily_usage = (
        df.resample("D", on="timestamp")["snack_units"]
        .sum()
        .reset_index()
        .rename(columns={"snack_units": "daily_snacks"})
    )
    if daily_usage.empty:
        st.info("Not enough data to simulate inventory yet.")
        return

    avg_daily = float(daily_usage["daily_snacks"].mean()) if not daily_usage.empty else 10.0
    default_start = max(10.0, round(avg_daily * 4, 1))
    with st.expander("Game controls", expanded=True):
        col_left, col_right = st.columns(2)
        with col_left:
            start_stock = st.number_input(
                "Starting stock (units)", min_value=0.0, value=default_start, step=10.0
            )
            reorder_days = st.slider(
                "Reorder coverage (days)",
                min_value=1,
                max_value=14,
                value=3,
                key="inventory_reorder_days",
                help="How many days of demand you want on hand before triggering a new order.",
            )
        with col_right:
            reorder_amount = st.number_input(
                "Reorder amount",
                min_value=0.0,
                value=max(10.0, round(avg_daily * 2, 1)),
                step=5.0,
            )

        low_threshold_key = "inventory_low_threshold"
        recommended_threshold = max(5.0, round(avg_daily * reorder_days, 1))
        if low_threshold_key not in st.session_state:
            st.session_state[low_threshold_key] = recommended_threshold
        last_reorder_days = st.session_state.get("_inventory_last_reorder_days")
        if last_reorder_days != reorder_days:
            st.session_state["_inventory_last_reorder_days"] = reorder_days
            st.session_state[low_threshold_key] = recommended_threshold

        low_threshold = st.number_input(
            "Low-stock alert threshold",
            min_value=0.0,
            value=st.session_state[low_threshold_key],
            key=low_threshold_key,
            help="Auto-updated from reorder coverage; tweak if you need extra buffer.",
        )
        st.caption(
            f"Recommended reorder point: {recommended_threshold:.0f} units "
            f"(avg {avg_daily:.1f}/day × {reorder_days} days)."
        )

        col_a, col_b = st.columns(2)
        if col_a.button("Reset inventory game"):
            st.session_state["stock_level"] = start_stock
            st.session_state["stock_day_idx"] = 0
            st.session_state["stock_history"] = []
            st.rerun()
        if col_b.button("Reorder now"):
            st.session_state["stock_level"] = st.session_state.get("stock_level", start_stock) + reorder_amount
            st.rerun()

    if "stock_level" not in st.session_state:
        st.session_state["stock_level"] = start_stock
    if "stock_day_idx" not in st.session_state:
        st.session_state["stock_day_idx"] = 0
    if "stock_history" not in st.session_state:
        st.session_state["stock_history"] = []

    idx = st.session_state["stock_day_idx"] % len(daily_usage)
    current_day = daily_usage.iloc[idx]

    col1, col2, col3 = st.columns(3)
    col1.metric("Simulated date", str(current_day["timestamp"].date()))
    col2.metric("Stock level", f"{st.session_state['stock_level']:.0f} units")
    col3.metric("Projected demand", f"{current_day["daily_snacks"]:.0f} units")

    if st.session_state["stock_level"] <= low_threshold:
        st.warning("Low stock! Consider reordering before the next day.")

    if st.button("Next day ➡️"):
        st.session_state["stock_level"] = max(
            0.0, st.session_state["stock_level"] - current_day["daily_snacks"]
        )
        st.session_state["stock_history"].append(
            {
                "date": current_day["timestamp"].date(),
                "stock_end": st.session_state["stock_level"],
                "consumption": current_day["daily_snacks"],
            }
        )
        st.session_state["stock_day_idx"] = (idx + 1) % len(daily_usage)
        st.rerun()

    if st.session_state["stock_history"]:
        hist_df = pd.DataFrame(st.session_state["stock_history"])
        stock_fig = px.line(hist_df, x="date", y="stock_end", title="Stock level over simulated days")
        stock_fig.update_traces(mode="lines+markers")
        stock_fig.add_hrect(
            y0=0,
            y1=low_threshold,
            fillcolor="rgba(231,76,60,0.12)",
            line_width=0,
            annotation_text="Low stock zone",
            annotation_position="top left",
        )
        stock_fig.add_hline(
            y=low_threshold,
            line_dash="dash",
            line_color="#E74C3C",
            annotation_text=f"Threshold ({low_threshold:.0f})",
            annotation_position="bottom right",
        )
        st.plotly_chart(stock_fig, use_container_width=True)


__all__ = ["render_inventory_sandbox"]
