from pathlib import Path
import sys
import numpy as np
import pandas as pd
import plotly.express as px

# Ensure imports work both locally and on Streamlit Cloud
ROOT_DIR = Path(__file__).resolve().parents[3]
APP_DIR = Path(__file__).resolve().parent
for path in (ROOT_DIR, APP_DIR):
    if str(path) not in sys.path:
        sys.path.append(str(path))

import streamlit as st

try:
    from frontend.streamlit_app.components.layout import render_top_nav, get_logo_bytes
    from frontend.streamlit_app.services.data_utils import load_enriched_data
except ModuleNotFoundError:
    from components.layout import render_top_nav, get_logo_bytes
    from services.data_utils import load_enriched_data

st.set_page_config(page_title="Refuel Control Center", page_icon="🏠", layout="wide")

render_top_nav("Home.py")
logo_bytes = get_logo_bytes()
if logo_bytes:
    st.image(logo_bytes, width=140)
st.title("Refuel Control Center")
st.caption("Manage your snack availability subject to weather and gym attendance forecasts")

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

st.divider()
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
st.plotly_chart(elasticity_fig, use_container_width=True)
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
st.plotly_chart(profit_fig, use_container_width=True)

with col_right:
    current_stock = st.number_input("Current snack stock (units)", min_value=0.0, value=round(avg_units * 5, 1), step=10.0)
    safety_stock = st.number_input(
        "Safety stock threshold", min_value=0.0, value=round(avg_units * 2, 1), step=5.0
    )
    lead_time_days = st.slider("Reorder lead time (days)", min_value=1, max_value=30, value=7, step=1)
    lead_time_demand = avg_units * lead_time_days
    reorder_point = safety_stock + lead_time_demand
    projected_units = demand_curve[-1]
    st.metric("Projected demand at highest price", f"{projected_units:.0f} units")
    stock_fig = px.bar(
        x=["Current"],
        y=[current_stock],
        labels={"x": "", "y": "Units"},
        title="Stock vs. safety bands",
    )
    stock_fig.add_hrect(
        y0=safety_stock,
        y1=safety_stock,
        line_width=2,
        line_color="orange",
        annotation_text="Safety stock",
        annotation_position="top right",
    )
    stock_fig.add_hrect(
        y0=reorder_point,
        y1=reorder_point,
        line_width=2,
        line_color="red",
        annotation_text="Reorder point",
        annotation_position="bottom right",
    )
    st.plotly_chart(stock_fig, use_container_width=True)
    if current_stock <= safety_stock:
        st.error("Low stock alert: inventory below safety threshold!")
    elif current_stock <= reorder_point:
        st.warning("Stock above safety but heading toward reorder point.")
    else:
        st.success("Stock level healthy. No alert triggered.")

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
st.plotly_chart(dow_fig, use_container_width=True)
st.table(
    dow_stats[["weekday_name", "snack_units", "suggested_price"]]
    .rename(columns={"weekday_name": "Weekday", "snack_units": "Avg snack units", "suggested_price": "Suggested price (€)"})
    .style.format({"Avg snack units": "{:.1f}", "Suggested price (€)": "€{:.2f}"})
)

st.subheader("Auto restock guidance")
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
st.write(
    f"Order ~**{recommended_qty:.0f} units** to cover lead time + buffer at the current daily run rate of {rolling_daily:.0f} units."
)

st.subheader("Historic week simulator")
if daily_summary.empty:
    st.info("Need daily history to run the simulator.")
else:
    dates_sorted = sorted(daily_summary["date"].unique())
    start_date = st.select_slider("Simulation start date", options=dates_sorted, value=dates_sorted[0])
    sim_weeks = st.slider("Number of weeks", 1, 12, 4)
    sim_stock = st.number_input("Simulation starting stock", min_value=0.0, value=current_stock, step=10.0, key="sim-stock")
    sim_safety = st.number_input("Simulation safety stock", min_value=0.0, value=safety_stock, step=5.0, key="sim-safety")
    sim_lead = st.slider("Simulation reorder lead (days)", 1, 21, lead_time_days, key="sim-lead")
    sim_reorder_qty = st.number_input(
        "Simulation reorder quantity", min_value=0.0, value=reorder_point - safety_stock, step=10.0, key="sim-reorder"
    )
    auto_reorder = st.checkbox("Auto reorder when below safety", value=True, key="sim-auto")

    start_idx = daily_summary.index[daily_summary["date"] == start_date][0]
    total_days_sim = sim_weeks * 7
    stock = sim_stock
    sim_rows: List[dict] = []
    for offset in range(total_days_sim):
        row = daily_summary.iloc[(start_idx + offset) % len(daily_summary)]
        stock -= row["snack_units"]
        reordered = False
        if auto_reorder and stock <= sim_safety:
            stock += sim_reorder_qty
            reordered = True
        sim_rows.append(
            {
                "date": row["date"].isoformat(),
                "demand": round(row["snack_units"], 1),
                "stock_after": round(stock, 1),
                "reordered": "Yes" if reordered else "",
            }
        )
    sim_df = pd.DataFrame(sim_rows)
    sim_fig = px.line(sim_df, x="date", y="stock_after", title="Simulated stock over historic weeks")
    sim_fig.add_hline(y=sim_safety, line_dash="dot", line_color="orange", annotation_text="Safety stock")
    st.plotly_chart(sim_fig, use_container_width=True)
    st.dataframe(sim_df, use_container_width=True, height=300)
