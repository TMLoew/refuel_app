from pathlib import Path
import sys
from typing import List
from datetime import datetime
import numpy as np
import pandas as pd
import plotly.express as px

# Ensure imports work both locally and on Streamlit Cloud
ROOT_DIR = Path(__file__).resolve().parents[3]
PROJECT_ROOT = ROOT_DIR
APP_DIR = Path(__file__).resolve().parent
for path in (ROOT_DIR, APP_DIR):
    if str(path) not in sys.path:
        sys.path.append(str(path))

import streamlit as st

try:
    from frontend.streamlit_app.components.layout import (
        render_top_nav,
        get_logo_path,
        get_logo_bytes,
        render_footer,
    )
    import frontend.streamlit_app.services.data_utils as data_utils_mod
except ModuleNotFoundError:
    from components.layout import render_top_nav, get_logo_path, get_logo_bytes, render_footer
    import services.data_utils as data_utils_mod

WEATHER_SCENARIOS = data_utils_mod.WEATHER_SCENARIOS
SNACK_PROMOS = data_utils_mod.SNACK_PROMOS
build_scenario_forecast = data_utils_mod.build_scenario_forecast
load_enriched_data = data_utils_mod.load_enriched_data
train_models = data_utils_mod.train_models
save_procurement_plan = getattr(data_utils_mod, "save_procurement_plan", lambda *_args, **_kwargs: None)
load_procurement_plan = getattr(data_utils_mod, "load_procurement_plan", lambda: pd.DataFrame())

PAGE_ICON = get_logo_path() or "🏠"
st.set_page_config(page_title="Refuel Control Center", page_icon=PAGE_ICON, layout="wide")

AUTOPILOT_STATE_FILE = PROJECT_ROOT / "data" / "autopilot_infinite.csv"


def load_autopilot_history_file() -> pd.DataFrame:
    if not AUTOPILOT_STATE_FILE.exists():
        return pd.DataFrame()
    df = pd.read_csv(AUTOPILOT_STATE_FILE)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df


def save_autopilot_history_file(history_df: pd.DataFrame) -> None:
    AUTOPILOT_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    export = history_df.copy()
    if not export.empty and "date" in export.columns:
        export["date"] = pd.to_datetime(export["date"]).dt.strftime("%Y-%m-%d")
    export.to_csv(AUTOPILOT_STATE_FILE, index=False)


def reset_autopilot_history_file() -> None:
    if AUTOPILOT_STATE_FILE.exists():
        AUTOPILOT_STATE_FILE.unlink()


def autopilot_anchor_timestamp(base_history: pd.DataFrame, autop_history: pd.DataFrame) -> pd.Timestamp:
    if autop_history is not None and not autop_history.empty and "date" in autop_history.columns:
        last_day = pd.to_datetime(autop_history["date"].max())
        return last_day + pd.Timedelta(hours=23)
    return base_history["timestamp"].max()

render_top_nav("Home.py", show_logo=False)
st.title("Refuel Control Center")
st.caption("Manage your snack availability subject to weather and gym attendance forecasts")

logo_bytes = get_logo_bytes()
with st.sidebar:
    if logo_bytes:
        st.image(logo_bytes, width=120)
    st.markdown("Live telemetry cockpit")
    st.caption("Data updated every hour · Last refresh from notebook sync.")
    st.divider()
    st.subheader("Data slice")
    use_weather_api = st.toggle("Use live weather API", value=False, key="home-weather-toggle")

with st.spinner("Loading telemetry for preview..."):
    data = load_enriched_data(use_weather_api=use_weather_api)

if data.empty:
    st.sidebar.error("No telemetry data available yet. Upload a CSV to explore the cockpit.")
    st.stop()

daily_summary = (
    data.set_index("timestamp")
    .resample("D")
    .agg({"checkins": "sum", "snack_units": "sum", "temperature_c": "mean", "snack_price": "mean"})
    .reset_index()
)
daily_summary["date"] = daily_summary["timestamp"].dt.date
models = train_models(data)
if "auto_results" not in st.session_state:
    cached_plan = load_procurement_plan()
    if not cached_plan.empty:
        st.session_state["auto_results"] = cached_plan

total_days = max(1, int((data["timestamp"].max() - data["timestamp"].min()).days) or 1)
with st.sidebar:
    history_days = st.slider(
        "History window (days)",
        min_value=3,
        max_value=max(3, total_days),
        value=min(7, max(3, total_days)),
        key="home-history",
    )
    metric_focus = st.selectbox("Focus metric", ["checkins", "snack_units", "snack_revenue"], key="home-metric")
    weather_meta = data.attrs.get("weather_meta", {})
    if weather_meta:
        st.caption(
            f"Weather synced {weather_meta.get('updated_at', 'n/a')} UTC · coverage {weather_meta.get('coverage_start', '?')} → {weather_meta.get('coverage_end', '?')}"
        )

st.info(
    "Use the navigation bar to jump into specific tools. The sidebar controls above mirror the default data slice you can apply inside each module."
)


def run_auto_simulation(
    forecast_hours: pd.DataFrame,
    starting_stock: float,
    safety_stock: float,
    reorder_qty: float,
    unit_cost: float,
    fee: float,
    elasticity: float,
    price_strategy_pct: float,
    scenario_label: str,
) -> pd.DataFrame:
    """Simulate multi-day stock + pricing autopilot decisions."""
    if forecast_hours.empty:
        return pd.DataFrame()

    forecast = forecast_hours.copy()
    forecast["timestamp"] = pd.to_datetime(forecast["timestamp"])
    daily = (
        forecast.assign(date=forecast["timestamp"].dt.date)
        .groupby("date")
        .agg(
            temperature_c=("temperature_c", "mean"),
            snack_price=("snack_price", "mean"),
            demand=("pred_snack_units", "sum"),
            checkins=("pred_checkins", "sum"),
        )
        .reset_index()
    )
    if daily.empty:
        return pd.DataFrame()

    base_price = float(daily["snack_price"].mean())
    temp_mean = float(daily["temperature_c"].mean())
    price_min = base_price * 0.8 if base_price else 1.5
    price_max = base_price * 1.25 if base_price else 5.0

    rows: List[dict] = []
    stock = starting_stock
    for _, row in daily.iterrows():
        temp_bias = 1 + 0.008 * (row["temperature_c"] - temp_mean)
        target_price = float(
            np.clip(row["snack_price"] * temp_bias * (1 + price_strategy_pct / 100), price_min, price_max)
        )
        demand_adj = max(0.0, row["demand"] * (target_price / max(base_price, 0.01)) ** elasticity)
        stock_before = stock
        sold = min(stock_before, demand_adj)
        stock_after = stock_before - sold
        profit = (target_price - unit_cost - fee) * sold
        reordered = ""
        reorder_qty_used = 0.0
        if stock_after <= safety_stock:
            stock_after += reorder_qty
            reordered = "Yes"
            reorder_qty_used = reorder_qty
        rows.append(
            {
                "date": pd.to_datetime(row["date"]),
                "scenario": scenario_label,
                "checkins_est": round(row["checkins"], 1),
                "temperature_c": round(row["temperature_c"], 1),
                "price": round(target_price, 2),
                "demand_est": round(demand_adj, 1),
                "sold": round(sold, 1),
                "profit": round(profit, 2),
                "stock_before": round(stock_before, 1),
                "stock_after": round(stock_after, 1),
                "reordered": reordered,
                "reorder_qty": reorder_qty_used,
            }
        )
        stock = stock_after
    return pd.DataFrame(rows)


def run_historic_replay(
    daily_df: pd.DataFrame,
    start_idx: int,
    total_days: int,
    start_stock: float,
    safety_stock: float,
    reorder_qty: float,
    auto_reorder: bool,
) -> pd.DataFrame:
    """Replay historic demand sequences to test manual policies."""
    if daily_df.empty:
        return pd.DataFrame()

    stock = start_stock
    rows: List[dict] = []
    for offset in range(total_days):
        row = daily_df.iloc[(start_idx + offset) % len(daily_df)]
        demand = float(row["snack_units"])
        sold = min(stock, demand)
        stock -= sold
        reordered = False
        if auto_reorder and stock <= safety_stock:
            stock += reorder_qty
            reordered = True
        rows.append(
            {
                "date": row["date"].isoformat(),
                "demand": round(demand, 1),
                "sold": round(sold, 1),
                "stock_after": round(stock, 1),
                "reordered": "Yes" if reordered else "",
            }
        )
    return pd.DataFrame(rows)


pricing_tab, inventory_tab = st.tabs(["Pricing & Elasticity", "Inventory Planner"])

with pricing_tab:
    st.subheader("Snack Pricing & Elasticity Sandbox")

    avg_price = float(data["snack_price"].mean())
    avg_units = float(data["snack_units"].mean())
    col_left, col_right = st.columns(2)
    with col_left:
        elasticity = st.slider("Elasticity factor (negative means demand drops with price)", -3.0, 1.0, -1.2, step=0.1)
        price_range_pct = st.slider("Price adjustment range (%)", 10, 60, 30, step=5)
        promo_boost = st.slider("Promo boost on demand (%)", 0, 100, 15, step=5)
        preferred_price = st.number_input("Test price (€)", min_value=0.5, max_value=10.0, value=round(avg_price, 2), step=0.1)

    price_points = np.linspace(avg_price * (1 - price_range_pct / 100), avg_price * (1 + price_range_pct / 100), 40)
    demand_curve = avg_units * (price_points / avg_price) ** elasticity * (1 + promo_boost / 100)
    elasticity_fig = px.line(
        x=price_points,
        y=demand_curve,
        labels={"x": "Price (€)", "y": "Expected snack units"},
        title="Elasticity curve",
    )
    st.plotly_chart(elasticity_fig, width="stretch")
    expected_at_pref = avg_units * (preferred_price / avg_price) ** elasticity * (1 + promo_boost / 100)
    st.metric("Expected demand at test price", f"{expected_at_pref:.0f} units")

    st.subheader("Profit maximizer")
    unit_cost = st.number_input("Unit cost (€)", min_value=0.1, value=round(avg_price * 0.6, 2), step=0.1)
    operating_fee = st.slider("Per-transaction fee (€)", 0.0, 2.0, 0.2, step=0.1)
    margin_curve = (price_points - unit_cost - operating_fee) * demand_curve
    optimal_idx = int(np.argmax(margin_curve))
    optimal_price = price_points[optimal_idx]
    optimal_units = demand_curve[optimal_idx]
    optimal_profit = margin_curve[optimal_idx]
    st.write(
        f"At €{optimal_price:.2f}, expected demand is {optimal_units:.0f} units and projected profit is €{optimal_profit:.0f} / period."
    )
    profit_fig = px.line(
        x=price_points,
        y=margin_curve,
        labels={"x": "Price (€)", "y": "Profit"},
        title="Profit vs. price",
    )
    profit_fig.add_vline(x=optimal_price, line_dash="dash", line_color="green", annotation_text="Optimal")
    profit_fig.add_vline(x=preferred_price, line_dash="dot", line_color="blue", annotation_text="Test price")
    st.plotly_chart(profit_fig, width="stretch")

with inventory_tab:
    sim_mode = st.selectbox("Simulation mode", ["Manual Planner", "Historic Replay", "Weather-aware Autopilot"])
    avg_price = float(data["snack_price"].mean())
    avg_units = float(data["snack_units"].mean())

    if sim_mode == "Manual Planner":
        st.subheader("Manual planner")
        current_stock = st.number_input("Current snack stock (units)", min_value=0.0, value=round(avg_units * 5, 1), step=10.0)
        safety_stock = st.number_input("Safety stock threshold", min_value=0.0, value=round(avg_units * 2, 1), step=5.0)
        lead_time_days = st.slider("Reorder lead time (days)", min_value=1, max_value=30, value=7, step=1)
        lead_time_demand = avg_units * lead_time_days
        reorder_point = safety_stock + lead_time_demand
        stock_fig = px.bar(x=["Current"], y=[current_stock], labels={"x": "", "y": "Units"}, title="Stock vs. safety bands")
        stock_fig.add_hrect(y0=safety_stock, y1=safety_stock, line_width=2, line_color="orange", annotation_text="Safety stock", annotation_position="top right")
        stock_fig.add_hrect(y0=reorder_point, y1=reorder_point, line_width=2, line_color="red", annotation_text="Reorder point", annotation_position="bottom right")
        st.plotly_chart(stock_fig, width="stretch")
        rolling_daily = daily_summary["snack_units"].rolling(7, min_periods=1).mean().iloc[-1]
        auto_stock = st.number_input("Auto-stock level (units)", min_value=0.0, value=current_stock, step=10.0, key="auto-stock")
        auto_lead = st.slider("Auto lead time (days)", 1, 21, lead_time_days, key="auto-lead")
        service_buffer = st.slider("Buffer after delivery (days)", 1, 14, 3, key="auto-buffer")
        days_until_out = auto_stock / max(rolling_daily, 1)
        recommended_order_in = max(0.0, days_until_out - auto_lead)
        recommended_qty = max(0.0, (auto_lead + service_buffer) * rolling_daily - auto_stock)
        col_a, col_b = st.columns(2)
        col_a.metric("Days until stockout", f"{days_until_out:.1f} d")
        col_b.metric("Recommended reorder in", f"{recommended_order_in:.1f} d")
        st.write(f"Order ~**{recommended_qty:.0f} units** to cover lead time + buffer at the current daily run rate of {rolling_daily:.0f} units.")

    elif sim_mode == "Historic Replay":
        st.subheader("Historic replay")
        if daily_summary.empty:
            st.info("Need daily history to run the simulator.")
        else:
            dates_sorted = sorted(daily_summary["date"].unique())
            start_date = st.select_slider("Simulation start date", options=dates_sorted, value=dates_sorted[0])
            sim_weeks = st.slider("Number of weeks", 1, 12, 4)
            sim_stock = st.number_input("Simulation starting stock", min_value=0.0, value=round(avg_units * 5, 1), step=10.0, key="sim-stock")
            sim_safety = st.number_input("Simulation safety stock", min_value=0.0, value=round(avg_units * 2, 1), step=5.0, key="sim-safety")
            sim_reorder_qty = st.number_input("Simulation reorder quantity", min_value=0.0, value=round(avg_units * 4, 1), step=10.0, key="sim-reorder")
            auto_reorder = st.checkbox("Auto reorder when below safety", value=True, key="sim-auto")
            if st.button("Run historic replay", key="run-historic"):
                start_idx = daily_summary.index[daily_summary["date"] == start_date][0]
                total_days_sim = sim_weeks * 7
                hist_df = run_historic_replay(
                    daily_summary,
                    start_idx=start_idx,
                    total_days=total_days_sim,
                    start_stock=sim_stock,
                    safety_stock=sim_safety,
                    reorder_qty=sim_reorder_qty,
                    auto_reorder=auto_reorder,
                )
                st.session_state["historic_results"] = hist_df
            hist_df = st.session_state.get("historic_results")
            if isinstance(hist_df, pd.DataFrame) and not hist_df.empty:
                sim_fig = px.line(hist_df, x="date", y="stock_after", title="Simulated stock over historic weeks")
                sim_fig.add_hline(y=sim_safety, line_dash="dot", line_color="orange", annotation_text="Safety stock")
                st.plotly_chart(sim_fig, width="stretch")
                st.dataframe(hist_df, width="stretch", height=300)

    else:
        st.subheader("Weather-aware autopilot")
        if models[0] is None or models[1] is None:
            st.warning("Need more telemetry to train the forecast models. Revisit after uploading additional data.")
        else:
            derived_conversion = float(data["snack_units"].sum()) / max(float(data["checkins"].sum()), 1.0)
            derived_conversion = float(np.clip(derived_conversion, 0.05, 0.9))
            auto_days = st.slider("Autopilot step (simulated days per run)", 3, 120, 21, key="auto-days")
            lead_time_auto = 7
            service_factor = 1.65
            demand_std = float(daily_summary["snack_units"].std() or avg_units * 0.1)
            mean_checkins = float(daily_summary["checkins"].mean())
            lead_time_demand = derived_conversion * mean_checkins * lead_time_auto
            safety_auto = max(0.0, lead_time_demand + service_factor * demand_std * np.sqrt(lead_time_auto))
            reorder_qty_auto = safety_auto + lead_time_demand
            starting_auto = reorder_qty_auto * 2
            auto_unit_cost = st.number_input("Sim unit cost (€)", min_value=0.1, value=unit_cost, step=0.1, key="auto-unit-cost")
            auto_fee = st.slider("Sim per-transaction fee (€)", 0.0, 2.0, operating_fee, step=0.1, key="auto-fee")
            st.caption("Autopilot now runs indefinitely: it generates weather, attendance, and snack demand while managing stock.")

            scenario_cols = st.columns(3)
            weather_pattern = scenario_cols[0].selectbox("Weather pattern", list(WEATHER_SCENARIOS.keys()), key="auto-weather")
            marketing_boost = scenario_cols[1].slider("Marketing boost (%)", 0, 80, 10, key="auto-marketing")
            promo_choice = scenario_cols[2].selectbox("Promo tactic", list(SNACK_PROMOS.keys()), key="auto-promo")

            manual_cols = st.columns(3)
            temp_manual = manual_cols[0].slider("Manual temp shift (°C)", -8, 8, 0, key="auto-temp")
            precip_manual = manual_cols[1].slider("Manual precipitation shift (mm)", -3.0, 3.0, 0.0, step=0.1, key="auto-precip")
            event_intensity = manual_cols[2].slider("Event intensity", 0.2, 2.5, 1.0, step=0.1, key="auto-event")

            price_change = st.slider("Baseline price change (%)", -20, 25, 0, key="auto-price-change")
            price_strategy = st.slider("Dynamic price aggressiveness (%)", -10, 15, 0, key="auto-price-strategy")
            sales_boost_pct = st.slider(
                "Sales boost (%)",
                min_value=0,
                max_value=200,
                value=0,
                step=5,
                key="auto-sales-boost",
                help="Apply an extra uplift to snack demand during the infinite simulation.",
            )

            if "autopilot_history" not in st.session_state:
                st.session_state["autopilot_history"] = load_autopilot_history_file()
            if "autopilot_running" not in st.session_state:
                st.session_state["autopilot_running"] = False

            autop_history = st.session_state["autopilot_history"]
            current_stock = starting_auto if autop_history.empty else float(autop_history["stock_after"].iloc[-1])
            cola, colb, colc = st.columns(3)
            cola.metric("Derived conversion", f"{derived_conversion:.2f}")
            colb.metric("Recommended safety stock", f"{safety_auto:.0f} units")
            colc.metric("Recommended reorder qty", f"{reorder_qty_auto:.0f} units")

            status_cols = st.columns(2)
            status_cols[0].metric("Autopilot status", "Running" if st.session_state["autopilot_running"] else "Paused")
            status_cols[1].metric("Current stock", f"{current_stock:.0f} units")

            scenario_payload = {
                "horizon_hours": auto_days * 24,
                "weather_pattern": weather_pattern,
                "temp_manual": temp_manual,
                "precip_manual": precip_manual,
                "event_intensity": event_intensity,
                "marketing_boost_pct": marketing_boost,
                "snack_price_change": price_change,
                "snack_promo": promo_choice,
            }

            action_cols = st.columns([0.4, 0.3, 0.3])
            play_clicked = action_cols[0].button("▶️ Play / Advance", key="auto-play", use_container_width=True)
            pause_clicked = action_cols[1].button(
                "⏸ Pause",
                key="auto-pause",
                use_container_width=True,
                disabled=not st.session_state["autopilot_running"],
            )
            reset_clicked = action_cols[2].button("♻️ Reset", key="auto-reset", use_container_width=True)

            if play_clicked:
                st.session_state["autopilot_running"] = True
                anchor_ts = autopilot_anchor_timestamp(data, autop_history)
                forecast_hours = build_scenario_forecast(
                    data,
                    models,
                    scenario_payload,
                    anchor_timestamp=anchor_ts,
                )
                if forecast_hours.empty:
                    st.warning("Forecast pipeline returned no rows. Adjust the scenario and try again.")
                else:
                    if sales_boost_pct:
                        forecast_hours["pred_snack_units"] *= 1 + sales_boost_pct / 100
                    auto_df = run_auto_simulation(
                        forecast_hours=forecast_hours,
                        starting_stock=current_stock,
                        safety_stock=safety_auto,
                        reorder_qty=reorder_qty_auto,
                        unit_cost=auto_unit_cost,
                        fee=auto_fee,
                        elasticity=elasticity,
                        price_strategy_pct=price_strategy,
                        scenario_label=weather_pattern,
                    )
                    if auto_df.empty:
                        st.warning("Simulation failed; need more historical telemetry.")
                    else:
                        block_id = (
                            1 if "sim_block" not in autop_history.columns or autop_history.empty else int(autop_history["sim_block"].max()) + 1
                        )
                        plan_id = datetime.utcnow().isoformat(timespec="seconds")
                        auto_df["sim_block"] = block_id
                        auto_df["sales_boost_pct"] = sales_boost_pct
                        auto_df["plan_generated_at"] = plan_id
                        scenario_metadata = {
                            "plan_weather_pattern": weather_pattern,
                            "plan_marketing_boost_pct": f"{marketing_boost}",
                            "plan_promo": promo_choice,
                            "plan_price_change_pct": f"{price_change}",
                            "plan_price_strategy_pct": f"{price_strategy}",
                            "plan_unit_cost": f"{auto_unit_cost:.2f}",
                            "plan_fee": f"{auto_fee:.2f}",
                            "plan_horizon_days": f"{auto_days}",
                            "plan_lead_time_days": f"{lead_time_auto}",
                            "plan_safety_stock": f"{safety_auto:.1f}",
                            "plan_reorder_qty": f"{reorder_qty_auto:.1f}",
                            "plan_temp_manual": f"{temp_manual}",
                            "plan_precip_manual": f"{precip_manual}",
                            "plan_event_intensity": f"{event_intensity}",
                            "plan_sales_boost_pct": f"{sales_boost_pct}",
                            "plan_block_id": f"{block_id}",
                        }
                        for meta_key, meta_val in scenario_metadata.items():
                            auto_df[meta_key] = meta_val
                        save_procurement_plan(auto_df, metadata=scenario_metadata)
                        st.session_state["auto_results"] = auto_df
                        autop_history = pd.concat([autop_history, auto_df], ignore_index=True)
                        st.session_state["autopilot_history"] = autop_history
                        save_autopilot_history_file(autop_history)
                        st.success(
                            f"Advanced {auto_days} days. Ending stock {auto_df['stock_after'].iloc[-1]:.0f} units · "
                            f"profit €{auto_df['profit'].sum():.0f}."
                        )

            if pause_clicked:
                st.session_state["autopilot_running"] = False
                st.info("Autopilot paused. Press Play to continue generating future days.")

            if reset_clicked:
                reset_autopilot_history_file()
                st.session_state["autopilot_history"] = pd.DataFrame()
                st.session_state["autopilot_running"] = False
                st.session_state.pop("auto_results", None)
                st.success("Autopilot state reset. You're back at the starting conditions.")
                st.experimental_rerun()

            autop_status = "Running" if st.session_state["autopilot_running"] else "Paused"
            st.caption(
                f"Status: **{autop_status}** · data persisted to `{AUTOPILOT_STATE_FILE.name}` "
                "(share this CSV or reload it later)."
            )

            if autop_history.empty:
                st.info("No autopilot history yet. Press Play to generate the first block of days.")
            else:
                history_view = autop_history.copy()
                history_view["date"] = pd.to_datetime(history_view["date"])
                metrics_cols = st.columns(3)
                metrics_cols[0].metric("Days simulated", f"{len(history_view):.0f}")
                metrics_cols[1].metric("Total profit", f"€{history_view['profit'].sum():.0f}")
                metrics_cols[2].metric("Reorders triggered", int((history_view["reordered"] == "Yes").sum()))

                auto_fig = px.line(history_view, x="date", y="stock_after", title="Autopilot stock trajectory (infinite run)")
                auto_fig.add_hline(y=safety_auto, line_dash="dot", line_color="orange", annotation_text="Safety stock")
                reorder_points = history_view[history_view["reordered"] == "Yes"]
                if not reorder_points.empty:
                    auto_fig.add_scatter(
                        x=reorder_points["date"],
                        y=reorder_points["stock_after"],
                        mode="markers",
                        marker=dict(color="green", size=10),
                        name="Reorders",
                    )
                st.plotly_chart(auto_fig, width="stretch")

                st.dataframe(
                    history_view.tail(60)[
                        ["date", "scenario", "price", "demand_est", "sold", "stock_after", "reordered", "reorder_qty", "profit"]
                    ],
                    width="stretch",
                    height=320,
                )
                download_blob = history_view.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download autopilot history (CSV)",
                    download_blob,
                    file_name="autopilot_infinite_history.csv",
                    mime="text/csv",
                )

st.subheader("Day-of-week pricing hints")
dow_stats = (
    data.groupby("weekday")
    .agg(checkins=("checkins", "mean"), snack_units=("snack_units", "mean"), price=("snack_price", "mean"))
    .reset_index()
)
dow_stats["weekday_name"] = dow_stats["weekday"].map(dict(enumerate(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])))
dow_stats["suggested_price"] = (
    dow_stats["price"]
    * (dow_stats["snack_units"] / dow_stats["snack_units"].mean()).clip(lower=0.8, upper=1.2)
)

dow_fig = px.bar(
    dow_stats,
    x="weekday_name",
    y=["snack_units", "checkins"],
    barmode="group",
    title="Average demand by weekday",
    labels={"value": "Average per hour", "variable": ""},
)
st.plotly_chart(dow_fig, width="stretch")
st.table(
    dow_stats[["weekday_name", "snack_units", "suggested_price"]]
    .rename(columns={"weekday_name": "Weekday", "snack_units": "Avg snack units", "suggested_price": "Suggested price (€)"})
    .style.format({"Avg snack units": "{:.1f}", "Suggested price (€)": "€{:.2f}"})
)


render_footer()
