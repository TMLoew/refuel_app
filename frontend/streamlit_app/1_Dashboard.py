
# --- Import bootstrap: make this file work from both local runs and Streamlit Cloud ---
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

# Try absolute import first, but capture the exception so we can display the real cause if it
# actually comes from inside data_utils (e.g., missing third‑party dependency like sklearn).
try:
    from frontend.streamlit_app.components.layout import render_top_nav, sidebar_info_block
    from frontend.streamlit_app.services.data_utils import (
        CHECKIN_FEATURES,
        SNACK_FEATURES,
        SNACK_PROMOS,
        WEATHER_SCENARIOS,
        DEFAULT_PRODUCT_PRICE,
        allocate_product_level_forecast,
        build_daily_forecast,
        build_scenario_forecast,
        build_daily_product_mix_view,
        compute_daily_actuals,
        get_product_price_map,
        get_weather_coordinates,
        load_enriched_data,
        load_product_mix_data,
        load_product_mix_snapshot,
        load_restock_policy,
        save_product_mix_snapshot,
        train_models,
    )
    from frontend.streamlit_app.services import weather_pipeline
except (ModuleNotFoundError, ImportError) as _abs_exc:
    # Prepare sys.path so that both `frontend.streamlit_app...` and local `components/services`
    # become importable regardless of whether we are in repo root or app subdir.
    _this = Path(__file__).resolve()

    # Candidate roots to add to sys.path (most specific first)
    _candidates = [
        _this.parent,                       # .../streamlit_app
        _this.parent.parent,                # .../frontend
        _this.parent.parent.parent,         # repo root (e.g., .../refuel_app)
    ]
    for _p in _candidates:
        _s = str(_p)
        if _s not in sys.path:
            sys.path.insert(0, _s)

    # Also ensure the explicit directories exist for sanity
    _frontend_dir = _this.parent.parent if (_this.parent.name == 'streamlit_app') else None
    if _frontend_dir and str(_frontend_dir) not in sys.path:
        sys.path.insert(0, str(_frontend_dir))

    try:
        # Retry absolute imports now that paths are primed
        from frontend.streamlit_app.components.layout import render_top_nav, sidebar_info_block
        from frontend.streamlit_app.services.data_utils import (
            CHECKIN_FEATURES,
            SNACK_FEATURES,
            SNACK_PROMOS,
            WEATHER_SCENARIOS,
            DEFAULT_PRODUCT_PRICE,
            allocate_product_level_forecast,
            build_daily_forecast,
            build_scenario_forecast,
            build_daily_product_mix_view,
            compute_daily_actuals,
            get_product_price_map,
            load_enriched_data,
            load_product_mix_data,
            load_product_mix_snapshot,
            load_restock_policy,
            save_product_mix_snapshot,
            train_models,
        )
        from frontend.streamlit_app.services import weather_pipeline
    except Exception as _retry_abs_exc:
        # Fall back to local package-style imports (components/, services/ under streamlit_app)
        try:
            from components.layout import hover_tip, render_top_nav, sidebar_info_block
            from services.data_utils import (
                CHECKIN_FEATURES,
                SNACK_FEATURES,
                SNACK_PROMOS,
                WEATHER_SCENARIOS,
                DEFAULT_PRODUCT_PRICE,
                allocate_product_level_forecast,
                build_daily_forecast,
                build_scenario_forecast,
                build_daily_product_mix_view,
                compute_daily_actuals,
                get_product_price_map,
                load_enriched_data,
                load_product_mix_data,
                load_product_mix_snapshot,
                load_restock_policy,
                save_product_mix_snapshot,
                train_models,
            )
            from services import weather_pipeline  # type: ignore
        except Exception as _local_exc:
            # Surface the true root cause to Streamlit UI for fast debugging
            import streamlit as _st
            _st.error("Failed to import app modules. See the exceptions below (absolute, retried absolute, local):")
            _st.exception(_abs_exc)
            _st.exception(_retry_abs_exc)
            _st.exception(_local_exc)
            raise

try:
    from frontend.streamlit_app.components.layout import hover_tip
except ImportError:
    try:
        from components.layout import hover_tip  # type: ignore
    except ImportError:
        def hover_tip(label: str, tooltip: str) -> None:
            st.caption(f"{label}: {tooltip}")
# --- End import bootstrap ---

st.set_page_config(
    page_title="Refuel Ops Dashboard",
    layout="wide",
    page_icon="💪",
)


def render_summary_cards(df: pd.DataFrame) -> None:
    recent = df.tail(24)
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Avg hourly check-ins (24h)", f"{recent['checkins'].mean():.1f}")
    col_b.metric("Snack units sold (24h)", f"{recent['snack_units'].sum():.0f}")
    col_c.metric("Snack revenue (24h)", f"€{recent['snack_revenue'].sum():.0f}")
    peak_hour = df.loc[df["checkins"].idxmax()]
    col_d.metric(
        "Peak load",
        f"{int(peak_hour['checkins'])} check-ins",
        f"{peak_hour['timestamp'].strftime('%a %H:%M')}",
    )


def render_history_charts(df: pd.DataFrame) -> None:
    history_window = df[df["timestamp"] >= df["timestamp"].max() - pd.Timedelta(days=5)]
    usage_fig = go.Figure()
    usage_fig.add_trace(
        go.Scatter(
            x=history_window["timestamp"],
            y=history_window["checkins"],
            mode="lines",
            name="Check-ins",
            line=dict(color="#2E86AB"),
        )
    )
    usage_fig.add_trace(
        go.Scatter(
            x=history_window["timestamp"],
            y=history_window["snack_units"],
            mode="lines",
            name="Snack units",
            yaxis="y2",
            line=dict(color="#F18F01"),
        )
    )
    usage_fig.update_layout(
        title="Gym traffic vs. snack demand",
        xaxis_title="Timestamp",
        yaxis_title="Check-ins",
        yaxis2=dict(title="Snack units", overlaying="y", side="right"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=380,
    )

    weather_fig = px.line(
        history_window,
        x="timestamp",
        y=["temperature_c", "precipitation_mm"],
        title="Weather trend (synthetic blend)",
        labels={"value": "Value", "variable": "Metric"},
    )
    weather_fig.update_layout(height=380, legend=dict(orientation="h", yanchor="bottom", y=1.02))

    col1, col2 = st.columns(2)
    col1.plotly_chart(usage_fig, width="stretch")
    col2.plotly_chart(weather_fig, width="stretch")


def render_weather_shotcast() -> None:
    lat, lon = get_weather_coordinates()
    iframe = f"""
    <iframe
        src="https://embed.windy.com/embed2.html?lat={lat:.3f}&lon={lon:.3f}&zoom=8&level=surface&overlay=rainAccu&menu=&message=true&marker=true&calendar=now&pressure=&type=map&location=coordinates&detail=true&detailLat={lat:.3f}&detailLon={lon:.3f}&metricWind=default&metricTemp=default&radarRange=-1"
        style="width:100%; height:340px; border:0; border-radius:12px;"
        allowfullscreen
    ></iframe>
    """
    components.html(iframe, height=340)


def render_forecast_section(history: pd.DataFrame, forecast: pd.DataFrame) -> None:
    if forecast.empty:
        st.warning("Not enough data to train the forecasting widgets yet.")
        return

    combined = history[
        history["timestamp"] >= history["timestamp"].max() - pd.Timedelta(hours=48)
    ][["timestamp", "checkins", "snack_units"]].copy()
    combined.rename(columns={"checkins": "actual_checkins", "snack_units": "actual_snacks"}, inplace=True)

    future_plot = forecast[["timestamp", "pred_checkins", "pred_snack_units"]].copy()
    combined = combined.merge(future_plot, on="timestamp", how="outer").sort_values("timestamp")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=combined["timestamp"],
            y=combined["actual_checkins"],
            mode="lines",
            name="Actual check-ins",
            line=dict(color="#2E86AB"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=combined["timestamp"],
            y=combined["pred_checkins"],
            mode="lines",
            name="Forecast check-ins",
            line=dict(color="#2E86AB", dash="dash"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=combined["timestamp"],
            y=combined["actual_snacks"],
            mode="lines",
            name="Actual snack units",
            line=dict(color="#F18F01"),
            yaxis="y2",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=combined["timestamp"],
            y=combined["pred_snack_units"],
            mode="lines",
            name="Forecast snack units",
            line=dict(color="#F18F01", dash="dash"),
            yaxis="y2",
        )
    )
    fig.update_layout(
        title="Scenario forecast",
        xaxis_title="Timestamp",
        yaxis=dict(title="Check-ins"),
        yaxis2=dict(title="Snack units", overlaying="y", side="right"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=420,
        shapes=[
            dict(
                type="rect",
                xref="x",
                yref="paper",
                x0=history["timestamp"].max(),
                x1=forecast["timestamp"].max(),
                y0=0,
                y1=1,
                fillcolor="rgba(200,200,200,0.15)",
                line=dict(width=0),
            )
        ],
    )
    st.plotly_chart(fig, width="stretch")

    kpi_cols = st.columns(3)
    kpi_cols[0].metric(
        "Forecast check-ins",
        f"{forecast['pred_checkins'].sum():.0f}",
        f"{forecast['pred_checkins'].max():.0f} peak/hr",
    )
    kpi_cols[1].metric(
        "Forecast snack units",
        f"{forecast['pred_snack_units'].sum():.0f}",
        f"{forecast['pred_snack_units'].max():.0f} peak/hr",
    )
    kpi_cols[2].metric(
        "Forecast snack revenue",
        f"€{forecast['pred_snack_revenue'].sum():.0f}",
        f"avg price €{forecast['snack_price'].mean():.2f}",
    )

    st.dataframe(
        forecast[
            [
                "timestamp",
                "temperature_c",
                "precipitation_mm",
                "pred_checkins",
                "pred_snack_units",
                "snack_price",
                "pred_snack_revenue",
            ]
        ]
        .rename(
            columns={
                "temperature_c": "Temp (°C)",
                "precipitation_mm": "Precip (mm)",
                "pred_checkins": "Check-ins",
                "pred_snack_units": "Snack units",
                "snack_price": "Snack price (€)",
                "pred_snack_revenue": "Snack revenue (€)",
            }
        )
        .set_index("timestamp"),
        width="stretch",
    )


def render_dashboard() -> None:
    render_top_nav("1_Dashboard.py", show_logo=False)
    st.title("Refuel Performance Cockpit")
    st.caption(
        "Blending weather mood, gym traffic, and snack behavior to guide staffing, procurement, and marketing."
    )

    with st.sidebar:
        sidebar_info_block()
        st.subheader("Scenario controls")
        use_weather_api = st.toggle("Use live weather API", value=False)

    with st.spinner("Loading telemetry and contextual data..."):
        data = load_enriched_data(use_weather_api=use_weather_api)
        product_mix_df = load_product_mix_data()
    price_map = get_product_price_map()
    if data.empty:
        st.error("No gym data found yet. Drop a CSV into `data/gym_badges.csv` to get started.")
        return

    weather_source = data.attrs.get("weather_source", "synthetic")
    st.caption(
        f"Weather source · {'Open-Meteo API' if weather_source == 'open-meteo' else 'synthetic fallback'}"
    )
    if use_weather_api and weather_source != "open-meteo":
        st.warning("Live weather API unreachable. Using synthetic fallback instead.")

    models = train_models(data)

    with st.sidebar:
        with st.expander("Weather & demand levers", expanded=True):
            horizon_hours = st.slider("Forecast horizon (hours)", min_value=6, max_value=72, value=24, step=6)
            weather_pattern = st.selectbox("Weather pattern", list(WEATHER_SCENARIOS.keys()))
            temp_manual = st.slider("Manual temperature shift (°C)", min_value=-8, max_value=8, value=0)
            precip_manual = st.slider(
                "Manual precipitation shift (mm)", min_value=-2.0, max_value=2.0, value=0.0, step=0.1
            )
        with st.expander("Campaign & pricing knobs", expanded=True):
            event_intensity = st.slider(
                "Gym event intensity",
                min_value=0.2,
                max_value=2.5,
                value=1.0,
                step=0.1,
                help="Represents corporate challenges or class launches (baseline ~0.5).",
            )
            marketing_boost_pct = st.slider(
                "Marketing reach boost (%)",
                min_value=0,
                max_value=80,
                value=10,
                step=5,
            )
            snack_price_change = st.slider(
                "Snack price change (%)",
                min_value=-30,
                max_value=40,
                value=0,
                step=5,
            )
            snack_promo = st.selectbox("Snack activation", list(SNACK_PROMOS.keys()))

    scenario = {
        "horizon_hours": horizon_hours,
        "weather_pattern": weather_pattern,
        "temp_manual": temp_manual,
        "precip_manual": precip_manual,
        "event_intensity": event_intensity,
        "marketing_boost_pct": marketing_boost_pct,
        "snack_price_change": snack_price_change,
        "snack_promo": snack_promo,
        "use_live_weather": use_weather_api,
    }
    restock_policy = load_restock_policy()
    restock_caption = (
        f"Auto restock ON · floor {restock_policy.get('threshold_units', 40)}u · lot {restock_policy.get('lot_size', 50)}u"
        if restock_policy.get("auto_enabled")
        else "Auto restock OFF · configure policy in POS Console"
    )
    st.caption(restock_caption)

    render_summary_cards(data)
    render_history_charts(data)
    st.subheader("Weather shotcast")
    st.caption("Open-Meteo radar/ cloud tiles centered on campus coordinates.")
    render_weather_shotcast()

    if isinstance(product_mix_df, pd.DataFrame) and not product_mix_df.empty:
        st.subheader("Product mix snapshot")
        latest_mix_date = product_mix_df["date"].max()
        latest_mix = product_mix_df[product_mix_df["date"] == latest_mix_date].copy()
        latest_mix["weight_pct"] = latest_mix["weight"] * 100
        latest_mix["unit_price"] = (
            latest_mix["product"].map(price_map).fillna(DEFAULT_PRODUCT_PRICE).round(2)
        )
        latest_mix["cost_estimate"] = latest_mix["suggested_qty"] * latest_mix["unit_price"]
        days_old = (pd.Timestamp.now().normalize() - latest_mix_date.normalize()).days
        if days_old > 1:
            st.warning(
                f"Product mix snapshot is {days_old} day(s) old. Update `data/product_mix_daily.csv` to reflect the latest plan."
            )
        mix_cols = st.columns(3)
        mix_cols[0].metric("Snapshot date", latest_mix_date.strftime("%Y-%m-%d"))
        mix_cols[1].metric("Visitors plan", f"{int(latest_mix['visitors'].iloc[0]):,}")
        mix_cols[2].metric(
            "Suggested units",
            f"{latest_mix['suggested_qty'].sum():.0f}",
            f"Est. cost €{latest_mix['cost_estimate'].sum():.0f}",
        )
        mix_fig = px.bar(
            latest_mix,
            x="product",
            y="weight_pct",
            title="Mix share by product",
            labels={"weight_pct": "Share (%)", "product": "Product"},
            color="hot_day",
            color_discrete_map={0: "#2E86AB", 1: "#E74C3C"},
        )
        mix_fig.update_layout(legend_title_text="Hot day?")
        st.plotly_chart(mix_fig, width="stretch")
        st.dataframe(
            latest_mix[["product", "suggested_qty", "weight_pct", "unit_price", "cost_estimate", "rainy_day"]]
            .rename(
                columns={
                    "product": "Product",
                    "suggested_qty": "Suggested Qty",
                    "weight_pct": "Mix Share (%)",
                    "unit_price": "Unit price (€)",
                    "cost_estimate": "Est. Cost (€)",
                    "rainy_day": "Rainy flag",
                }
            )
            .style.format(
                {
                    "Suggested Qty": "{:.0f}",
                    "Mix Share (%)": "{:.1f}",
                    "Unit price (€)": "€{:.2f}",
                    "Est. Cost (€)": "€{:.0f}",
                }
            ),
            width="stretch",
            height=260,
        )
    else:
        st.info("No product mix data detected. Upload `data/product_mix_daily.csv` to populate this snapshot.")

    merged_mix_view = build_daily_product_mix_view(data, product_mix_df)
    if not merged_mix_view.empty:
        st.subheader("Plan vs telemetry (daily)")
        st.caption("Daily rollup that compares suggested mix volume with actual snack demand.")
        recent_window = merged_mix_view[
            merged_mix_view["date"] >= merged_mix_view["date"].max() - pd.Timedelta(days=7)
        ].copy()
        if recent_window.empty:
            recent_window = merged_mix_view.copy()
        daily_rollup = (
            recent_window.groupby("date")
            .agg(
                planned_visitors=("visitors", "first"),
                actual_checkins=("actual_checkins", "first"),
                planned_units=("suggested_qty", "sum"),
                actual_units=("implied_units", "sum"),
                gap_units=("unit_gap", "sum"),
                avg_temp_c=("avg_temp_c", "first"),
            )
            .reset_index()
            .sort_values("date", ascending=False)
        )
        daily_rollup["actual_units"] = daily_rollup["actual_units"].fillna(daily_rollup["planned_units"])
        daily_rollup["gap_units"] = daily_rollup["gap_units"].fillna(0.0)
        saved_snapshot = load_product_mix_snapshot()
        last_saved_label = "Never"
        if not saved_snapshot.empty and "date" in saved_snapshot.columns:
            last_saved_label = saved_snapshot["date"].max().strftime("%Y-%m-%d")

        table_col, action_col = st.columns([4, 1])
        table_col.dataframe(
            daily_rollup.rename(
                columns={
                    "date": "Date",
                    "planned_visitors": "Planned visitors",
                    "actual_checkins": "Actual check-ins",
                    "planned_units": "Suggested units",
                    "actual_units": "Implied units",
                    "gap_units": "Unit gap",
                    "avg_temp_c": "Avg temp (°C)",
                }
            ),
            height=320,
        )
        action_col.markdown(f"Last snapshot:<br/>**{last_saved_label}**", unsafe_allow_html=True)
        if action_col.button("Save snapshot", key="save-mix-snapshot", use_container_width=True):
            metadata = {"snapshot_saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
            save_product_mix_snapshot(merged_mix_view, metadata=metadata)
            action_col.success("Saved to data/product_mix_enriched.csv")

    forecast_df = build_scenario_forecast(data, models, scenario)
    daily_actuals = compute_daily_actuals(data)
    daily_forecast = build_daily_forecast(forecast_df)
    if (not daily_actuals.empty) or (not daily_forecast.empty):
        st.subheader("Daily snack outlook")
        hover_tip(
            "ℹ️ Aggregation math",
            "Daily table sums hourly check-ins/snacks and compares them with forecasted sums. Forecast rows come from the same linear regressors but aggregated by date.",
        )
        merged_daily = daily_actuals.merge(
            daily_forecast,
            on="date",
            how="outer",
            suffixes=("_actual", "_forecast"),
        ).sort_values("date")
        if not merged_daily.empty:
            window_mask = merged_daily["date"] >= (merged_daily["date"].max() - pd.Timedelta(days=10))
            merged_window = merged_daily[window_mask].copy()
            merged_window["date"] = merged_window["date"].dt.strftime("%Y-%m-%d")
            display_cols = [
                "date",
                "actual_checkins",
                "pred_checkins",
                "actual_snack_units",
                "pred_snack_units",
                "actual_snack_revenue",
                "pred_snack_revenue",
            ]
            present_cols = [col for col in display_cols if col in merged_window.columns]
            st.dataframe(
                merged_window[present_cols].rename(
                    columns={
                        "date": "Date",
                        "actual_checkins": "Actual check-ins",
                        "pred_checkins": "Forecast check-ins",
                        "actual_snack_units": "Actual snacks",
                        "pred_snack_units": "Forecast snacks",
                        "actual_snack_revenue": "Actual revenue",
                        "pred_snack_revenue": "Forecast revenue",
                    }
                ),
                width="stretch",
                height=280,
            )
        product_forecast = allocate_product_level_forecast(daily_forecast, product_mix_df)
        if not product_forecast.empty:
            st.caption("Next 3 days · snack demand split by merchandise plan")
            hover_tip(
                "ℹ️ Mix allocation",
                "Product-level forecast multiplies each day's total snack prediction by its mix weight: units_i = weight_i × total_pred_snacks.",
            )
            upcoming_dates = sorted(product_forecast["date"].unique())[:3]
            product_window = product_forecast[product_forecast["date"].isin(upcoming_dates)].copy()
            product_window["date"] = product_window["date"].dt.strftime("%Y-%m-%d")
            st.dataframe(
                product_window[
                    ["date", "product", "forecast_units", "suggested_qty", "weight"]
                ].rename(
                    columns={
                        "date": "Date",
                        "product": "Product",
                        "forecast_units": "Forecast units",
                        "suggested_qty": "Plan units",
                        "weight": "Mix weight",
                    }
                ).style.format({"Forecast units": "{:.0f}", "Plan units": "{:.0f}", "Mix weight": "{:.2f}"}),
                width="stretch",
                height=280,
            )
    st.subheader("What-if forecast")
    hover_tip(
        "ℹ️ Forecast math",
        "Predictions come from two linear regressions: check-ins = β·features, snacks = γ·features (including weather shifts, events, and promo multipliers). Sliders alter the feature inputs before inference.",
    )
    render_forecast_section(data, forecast_df)


def _safe_render() -> None:
    try:
        render_dashboard()
    except Exception as exc:
        st.error("The dashboard crashed while rendering. See details below and please share this trace.")
        st.exception(exc)
        raise


_safe_render()
