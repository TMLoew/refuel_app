
# --- Import bootstrap: make this file work from both local runs and Streamlit Cloud ---
import sys
from pathlib import Path

# Try absolute import first, but capture the exception so we can display the real cause if it
# actually comes from inside data_utils (e.g., missing third‑party dependency like sklearn).
try:
    from frontend.streamlit_app.components.layout import render_top_nav, sidebar_info_block
    from frontend.streamlit_app.services.data_utils import (
        CHECKIN_FEATURES,
        SNACK_FEATURES,
        SNACK_PROMOS,
        WEATHER_SCENARIOS,
        build_scenario_forecast,
        load_enriched_data,
        train_models,
    )
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
            build_scenario_forecast,
            load_enriched_data,
            train_models,
        )
    except Exception as _retry_abs_exc:
        # Fall back to local package-style imports (components/, services/ under streamlit_app)
        try:
            from components.layout import render_top_nav, sidebar_info_block
            from services.data_utils import (
                CHECKIN_FEATURES,
                SNACK_FEATURES,
                SNACK_PROMOS,
                WEATHER_SCENARIOS,
                build_scenario_forecast,
                load_enriched_data,
                train_models,
            )
        except Exception as _local_exc:
            # Surface the true root cause to Streamlit UI for fast debugging
            import streamlit as _st
            _st.error("Failed to import app modules. See the exceptions below (absolute, retried absolute, local):")
            _st.exception(_abs_exc)
            _st.exception(_retry_abs_exc)
            _st.exception(_local_exc)
            raise
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
    col1.plotly_chart(usage_fig, use_container_width=True)
    col2.plotly_chart(weather_fig, use_container_width=True)


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
    st.plotly_chart(fig, use_container_width=True)

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
        use_container_width=True,
    )


def render_dashboard() -> None:
    render_top_nav("1_Dashboard.py")
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
    }

    render_summary_cards(data)
    render_history_charts(data)

    forecast_df = build_scenario_forecast(data, models, scenario)
    st.subheader("What-if forecast")
    render_forecast_section(data, forecast_df)


def _safe_render() -> None:
    try:
        render_dashboard()
    except Exception as exc:
        st.error("The dashboard crashed while rendering. See details below and please share this trace.")
        st.exception(exc)
        raise


_safe_render()
