from pathlib import Path
import sys
from datetime import datetime, timezone

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.pipeline import Pipeline

from frontend.streamlit_app.components.layout import (
    render_top_nav,
    sidebar_info_block,
    render_footer,
    get_logo_path,
    PRIMARY_GREEN,
    DEEP_GREEN,
    CORAL,
    YELLOW,
)
try:
    from frontend.streamlit_app.components.layout import hover_tip
except ImportError:
    try:
        from components.layout import hover_tip  # type: ignore
    except ImportError:
        def hover_tip(label: str, tooltip: str) -> None:
            st.caption(f"{label}: {tooltip}")
from frontend.streamlit_app.services.data_utils import (
    CHECKIN_FEATURES,
    WEATHER_SCENARIOS,
    DEFAULT_PRODUCT_PRICE,
    allocate_product_level_forecast,
    build_daily_forecast,
    build_scenario_forecast,
    get_product_price_map,
    load_enriched_data,
    load_procurement_plan,
    load_product_mix_data,
    load_restock_policy,
    save_procurement_plan,
    train_models,
)

PAGE_ICON = get_logo_path() or "🔮"
st.set_page_config(page_title="Forecast Explorer", page_icon=PAGE_ICON, layout="wide")

render_top_nav("2_Forecasts.py")
st.title("Forecast Explorer")
st.caption("Dig into the regression models, understand residuals, and inspect sensitivities before committing to a plan.")

with st.sidebar:
    sidebar_info_block()
    st.subheader("Data slice")
    use_weather_api = st.toggle("Use live weather API", value=True, key="forecast-weather")
    lookback_days = st.slider("History window (days)", 3, 14, 7)
    metric_focus = st.selectbox("Focus metric", ["checkins", "snack_units", "snack_revenue"])
    weather_profile = st.selectbox("Weather scenario override", list(WEATHER_SCENARIOS.keys()), key="forecast-weather-pattern")
    manual_temp_shift = st.slider("Manual temperature shift (°C)", -6, 6, 0, key="forecast-temp-shift")
    manual_precip_shift = st.slider("Manual precipitation shift (mm)", -2.0, 2.0, 0.0, step=0.1, key="forecast-precip-shift")
    max_horizon = 168  # allow longer runs; beyond live weather will use historical patterns
    horizon_hours = st.slider("Forecast horizon (hours)", 6, max_horizon, 72, step=6, key="forecast-horizon")
    with st.expander("Scenario levers", expanded=True):
        event_intensity = st.slider("Event intensity", 0.2, 2.5, 1.0, 0.1, key="forecast-event")
        marketing_boost_pct = st.slider("Marketing boost (%)", 0, 80, 10, 5, key="forecast-marketing")
        snack_price_change = st.slider("Snack price change (%)", -30, 40, 0, 5, key="forecast-price")
    with st.expander("Snacks ↔ visits settings", expanded=False):
        snack_agg_mode = st.radio("Aggregation", ["Hourly", "Daily"], horizontal=True, key="snack-agg")
        color_dim = st.selectbox(
            "Color points by",
            ["weekday", "weather_label", "is_weekend"],
            index=0,
            key="snack-color",
        )
        min_checkins = st.slider("Min. check-ins to include", 0, 50, 5, key="snack-min-checkins")

data = load_enriched_data(use_weather_api=use_weather_api)
if data.empty:
    st.error("No telemetry CSV detected. Upload data via the Data Editor page first.")
    st.stop()

models = train_models(data)
product_mix_df = load_product_mix_data()
restock_policy = load_restock_policy()
price_map = get_product_price_map()

latest_ts = data["timestamp"].max()
history = data[data["timestamp"] >= latest_ts - pd.Timedelta(days=lookback_days)]
scenario = WEATHER_SCENARIOS[weather_profile]
scenario_history = history.copy()
scenario_history["temperature_c"] += scenario["temp_offset"] + manual_temp_shift
scenario_history["precipitation_mm"] = np.clip(
    scenario_history["precipitation_mm"] * scenario["precip_multiplier"] + manual_precip_shift,
    0,
    None,
)
scenario_history["humidity_pct"] = np.clip(
    scenario_history["humidity_pct"] + scenario["humidity_offset"],
    0,
    100,
)
demand_factor = 1 + 0.015 * (scenario["temp_offset"] + manual_temp_shift) - 0.05 * (scenario["precip_multiplier"] - 1)
visit_factor = 1 + 0.01 * (scenario["temp_offset"] + manual_temp_shift) - 0.04 * (scenario["precip_multiplier"] - 1)
scenario_history["snack_units"] = np.clip(scenario_history["snack_units"] * demand_factor, 0, None)
scenario_history["checkins"] = np.clip(scenario_history["checkins"] * visit_factor, 0, None)
st.caption(
    f"Applied '{weather_profile}' scenario with temperature shift {manual_temp_shift:+}°C across the analysis views."
)

scenario_config = {
    "horizon_hours": horizon_hours,
    "weather_pattern": weather_profile,
    "temp_manual": manual_temp_shift,
    "precip_manual": manual_precip_shift,
    "event_intensity": event_intensity,
    "marketing_boost_pct": marketing_boost_pct,
    "snack_price_change": snack_price_change,
    "use_live_weather": use_weather_api,
}
forecast_df = build_scenario_forecast(data, models, scenario_config)
applied_horizon = forecast_df.attrs.get("applied_horizon_hours", horizon_hours) if not forecast_df.empty else 0
live_weather_hours = forecast_df.attrs.get("live_weather_hours")

col_a, col_b, col_c = st.columns(3)
col_a.metric("Latest check-ins/hr", f"{scenario_history['checkins'].iloc[-1]:.0f}")
col_b.metric("Snack revenue (24h)", f"CHF{scenario_history.tail(24)['snack_revenue'].sum():.0f}")
col_c.metric("Weather source", data.attrs.get("weather_source", "synthetic").title())
st.divider()

if forecast_df.empty:
    st.warning("Need more telemetry to compute the forward forecast. Upload additional history first.")
else:
    if use_weather_api and live_weather_hours:
        st.caption(f"Live weather used for the first {live_weather_hours} hours; remaining horizon uses historical patterns.")
    st.subheader(f"Scenario forecast · next {applied_horizon} hours")
    hover_tip(
        "ℹ️ Regression math",
        "Forecast lines come from two linear regressions: check-ins = β·features, snacks = γ·features. Slider tweaks shift the feature inputs before inference.",
    )
    history_window = data[data["timestamp"] >= data["timestamp"].max() - pd.Timedelta(hours=applied_horizon + 24)][
        ["timestamp", "checkins", "snack_units"]
    ].copy()
    history_window.rename(
        columns={"checkins": "actual_checkins", "snack_units": "actual_snacks"},
        inplace=True,
    )
    future_plot = forecast_df[["timestamp", "pred_checkins", "pred_snack_units"]].copy()
    combined = history_window.merge(future_plot, on="timestamp", how="outer").sort_values("timestamp")
    forecast_fig = go.Figure()
    forecast_fig.add_trace(
        go.Scatter(
            x=combined["timestamp"],
            y=combined["actual_checkins"],
            mode="lines",
            name="Actual check-ins",
            line=dict(color=PRIMARY_GREEN, width=2),
        )
    )
    forecast_fig.add_trace(
        go.Scatter(
            x=combined["timestamp"],
            y=combined["pred_checkins"],
            mode="lines",
            name="Forecast check-ins",
            line=dict(color=DEEP_GREEN, dash="dash", width=2),
        )
    )
    forecast_fig.add_trace(
        go.Scatter(
            x=combined["timestamp"],
            y=combined["actual_snacks"],
            mode="lines",
            name="Actual snack units",
            line=dict(color=CORAL, width=2),
            yaxis="y2",
        )
    )
    forecast_fig.add_trace(
        go.Scatter(
            x=combined["timestamp"],
            y=combined["pred_snack_units"],
            mode="lines",
            name="Forecast snack units",
            line=dict(color=YELLOW, dash="dash", width=2),
            yaxis="y2",
        )
    )
    forecast_fig.update_layout(
        xaxis_title="Timestamp",
        yaxis_title="Check-ins",
        yaxis2=dict(title="Snack units", overlaying="y", side="right"),
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        shapes=[
            dict(
                type="rect",
                xref="x",
                yref="paper",
                x0=data["timestamp"].max(),
                x1=forecast_df["timestamp"].max(),
                y0=0,
                y1=1,
                fillcolor="rgba(11,122,31,0.08)",
                line=dict(width=0),
            )
        ],
    )
    st.plotly_chart(forecast_fig, use_container_width=True)

    dl_col, stat_col = st.columns([1, 1])
    csv_bytes = forecast_df.to_csv(index=False).encode("utf-8")
    dl_col.download_button(
        "⬇️ Download forecast CSV",
        data=csv_bytes,
        file_name="forecast_hours.csv",
        mime="text/csv",
    )
    total_units = forecast_df["pred_snack_units"].sum()
    stat_col.metric("Total snacks forecast", f"{total_units:.0f} units", f"{forecast_df['pred_snack_units'].mean():.1f}/hr")

    daily_forecast = build_daily_forecast(forecast_df)
    if not daily_forecast.empty:
        st.subheader("Daily rollup & product mix impact")
        hover_tip(
            "ℹ️ Daily aggregation math",
            "Hourly predictions are summed per calendar day, then multiplied by each product's mix weight to allocate SKU-level units.",
        )
        st.caption("Merges the scenario forecast with the merchandising guidance.")
        st.dataframe(
            daily_forecast.assign(date=daily_forecast["date"].dt.strftime("%Y-%m-%d"))[
                ["date", "pred_checkins", "pred_snack_units", "pred_snack_revenue"]
            ].rename(
                columns={
                    "date": "Date",
                    "pred_checkins": "Forecast check-ins",
                    "pred_snack_units": "Forecast snacks",
                    "pred_snack_revenue": "Forecast revenue (CHF)",
                }
            ),
            use_container_width=True,
            height=260,
        )
        product_allocation = allocate_product_level_forecast(daily_forecast, product_mix_df)
        plan_payload = daily_forecast.assign(product="All snacks", forecast_units=daily_forecast["pred_snack_units"]).copy()
        plan_payload["unit_price"] = DEFAULT_PRODUCT_PRICE
        if not product_allocation.empty:
            st.caption("Next 3 days · forecasted units by product")
            upcoming_dates = sorted(product_allocation["date"].unique())[:3]
            mix_window = product_allocation[product_allocation["date"].isin(upcoming_dates)].copy()
            mix_window["date"] = mix_window["date"].dt.strftime("%Y-%m-%d")
            mix_window["unit_price"] = mix_window["product"].map(price_map).fillna(DEFAULT_PRODUCT_PRICE)
            st.dataframe(
                mix_window[
                    ["date", "product", "forecast_units", "suggested_qty", "weight", "unit_price"]
                ]
                .rename(
                    columns={
                        "date": "Date",
                        "product": "Product",
                        "forecast_units": "Forecast units",
                        "suggested_qty": "Plan units",
                        "weight": "Mix weight",
                        "unit_price": "Unit price (CHF)",
                    }
                )
                .style.format(
                    {"Forecast units": "{:.0f}", "Plan units": "{:.0f}", "Mix weight": "{:.2f}", "Unit price (CHF)": "CHF{:.2f}"}
                ),
                use_container_width=True,
                height=280,
            )
            product_allocation["unit_price"] = (
                product_allocation["product"].map(price_map).fillna(DEFAULT_PRODUCT_PRICE)
            )
            plan_payload = (
                product_allocation.rename(columns={"pred_snack_units": "forecast_units"})
                if "pred_snack_units" in product_allocation.columns
                else product_allocation.copy()
            )
        else:
            plan_payload["unit_price"] = DEFAULT_PRODUCT_PRICE
        columns = ["date", "product", "forecast_units"]
        if "suggested_qty" in plan_payload.columns:
            columns.append("suggested_qty")
        if "weight" in plan_payload.columns:
            columns.append("weight")
        columns.append("unit_price")
        plan_payload = plan_payload[columns]

    auto_caption = (
        f"Auto restock ON · floor {restock_policy.get('threshold_units', 40)}u · lot {restock_policy.get('lot_size', 50)}u"
        if restock_policy.get("auto_enabled")
        else "Auto restock OFF · configure on the POS Console to automate stock protection."
    )
    st.info(
        f"{auto_caption} · This scenario expects {total_units:.0f} snack units over the next {applied_horizon} hours.",
        icon="ℹ️",
    )
    if 'plan_payload' in locals() and not plan_payload.empty:
        with st.expander("Procurement actions", expanded=True):
            st.caption("Push this scenario into the shared procurement plan for downstream tabs.")
            hover_tip(
                "ℹ️ Plan math",
                "The exported CSV includes daily forecast units plus the scenario metadata (weather and price shifts) so downstream tabs can reproduce the assumptions.",
            )
            generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
            last_plan = load_procurement_plan()
            if not last_plan.empty and "plan_generated_at" in last_plan.columns:
                last_published = last_plan["plan_generated_at"].iloc[0]
                st.caption(f"Last published plan · {last_published}")
            plan_preview = plan_payload.copy()
            plan_preview["date"] = pd.to_datetime(plan_preview["date"]).dt.strftime("%Y-%m-%d")
            scenario_metadata = {
                "plan_generated_at": generated_at,
                "plan_source": "Forecast Explorer",
                "plan_weather_pattern": weather_profile,
                "plan_horizon_hours": f"{applied_horizon}",
                "plan_temp_manual": f"{manual_temp_shift}",
                "plan_precip_manual": f"{manual_precip_shift}",
                "plan_event_intensity": f"{event_intensity}",
                "plan_marketing_boost_pct": f"{marketing_boost_pct}",
                "plan_snack_price_change_pct": f"{snack_price_change}",
            }
            for key, value in scenario_metadata.items():
                plan_preview[key] = value
            plan_csv = plan_preview.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download plan CSV",
                data=plan_csv,
                file_name="procurement_plan_preview.csv",
                mime="text/csv",
                use_container_width=True,
            )
            publish = st.button("Publish scenario to procurement plan", use_container_width=True)
            if publish:
                plan_copy = plan_payload.copy()
                plan_copy["date"] = pd.to_datetime(plan_copy["date"])
                for key, value in scenario_metadata.items():
                    plan_copy[key] = value
                save_procurement_plan(plan_copy, metadata=scenario_metadata)
                st.success("Scenario published to procurement plan (data/procurement_plan.csv).")

# --- Daily trend ----------------------------------------------------------------
daily = scenario_history.resample("D", on="timestamp").agg({"checkins": "sum", "snack_units": "sum", "snack_revenue": "sum"})
trend_fig = px.line(
    daily.reset_index(),
    x="timestamp",
    y=["checkins", "snack_units"],
    title="Daily totals",
    markers=True,
    labels={"value": "Value", "variable": "Series", "timestamp": "Date"},
)
trend_fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))

# --- Heatmap --------------------------------------------------------------------
pivot = scenario_history.pivot_table(index="weekday", columns="hour", values="checkins", aggfunc="mean").reindex(range(7))
heatmap_fig = go.Figure(
    data=go.Heatmap(
        z=pivot.values,
        x=pivot.columns,
        y=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
        colorscale=[
            [0.0, "#ffffff"],
            [0.3, "#ffe5e5"],
            [0.6, "#ff7b7b"],
            [1.0, "#a80000"],
        ],
        colorbar=dict(title="Avg check-ins/hr"),
    )
)
heatmap_fig.update_layout(title="Hourly utilization pattern")

col1, col2 = st.columns(2)
col1.plotly_chart(trend_fig, use_container_width=True)
col2.plotly_chart(heatmap_fig, use_container_width=True)

# --- Correlation matrix --------------------------------------------------------
corr_fields = ["checkins", "snack_units", "temperature_c", "precipitation_mm", "humidity_pct"]
corr_matrix = scenario_history[corr_fields].corr()
corr_fig = px.imshow(
    corr_matrix,
    text_auto=".2f",
    aspect="auto",
    color_continuous_scale="RdBu",
    title="Correlation matrix: weather vs. demand",
)
st.plotly_chart(corr_fig, use_container_width=True)

# --- Snack vs visit correlation explorer ---------------------------------------
if snack_agg_mode == "Daily":
    snack_df = history.resample("D", on="timestamp").agg(
        {"checkins": "sum", "snack_units": "sum", "temperature_c": "mean", "weekday": "first", "is_weekend": "max"}
    )
    snack_df["weather_label"] = history.resample("D", on="timestamp")["weather_label"].agg(lambda x: x.mode().iloc[0])
else:
    snack_df = history.copy()

snack_df = snack_df[snack_df["checkins"] >= min_checkins]

if not snack_df.empty:
    corr_value = snack_df["checkins"].corr(snack_df["snack_units"])
    st.subheader("Snack demand vs. visit load")
    st.caption("Tune the controls in the sidebar to slice the correlation by aggregation or filters.")
    st.metric("Pearson correlation", f"{corr_value:.2f}" if not pd.isna(corr_value) else "n/a")
    color_field = color_dim if color_dim in snack_df.columns else None
    scatter_fig = px.scatter(
        snack_df,
        x="checkins",
        y="snack_units",
        color=color_field,
        trendline="ols",
        labels={"checkins": "Gym check-ins", "snack_units": "Snack units"},
        title="Snacks vs. visits",
        hover_data=["temperature_c"],
    )
    st.plotly_chart(scatter_fig, use_container_width=True)

# --- Feature sensitivity --------------------------------------------------------
checkin_model, snack_model = models
if checkin_model is not None:
    model_core = checkin_model.named_steps.get("model", checkin_model) if isinstance(checkin_model, Pipeline) else checkin_model
    importances = getattr(model_core, "feature_importances_", None)
    value_label = "importance"
    if importances is None and hasattr(model_core, "coef_"):
        importances = model_core.coef_
        value_label = "coef"
    if importances is not None:
        impact = (
            pd.DataFrame({"feature": CHECKIN_FEATURES, value_label: importances})
            .sort_values(value_label, key=lambda s: np.abs(s), ascending=False)
            .head(8)
        )
        impact_fig = px.bar(
            impact,
            x=value_label,
            y="feature",
            orientation="h",
            color=value_label,
            color_continuous_scale="RdBu",
            title="Model feature influence",
        )
        impact_fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(impact_fig, use_container_width=True)

# --- Residual diagnostics -------------------------------------------------------
if checkin_model is not None:
    feature_df = scenario_history.copy()
    feature_df["residuals_checkins"] = feature_df["checkins"] - checkin_model.predict(
        feature_df[CHECKIN_FEATURES]
    )
    residual_fig = px.scatter(
        feature_df,
        x="temperature_c",
        y="residuals_checkins",
        color="weather_label",
        title="Residuals vs. temperature",
        trendline="ols",
        labels={"residuals_checkins": "Residual (actual - predicted)"},
    )
    st.plotly_chart(residual_fig, use_container_width=True)

st.subheader("Raw data peek")
st.dataframe(scenario_history.tail(200).set_index("timestamp"), use_container_width=True, height=360)
render_footer()
