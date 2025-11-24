from pathlib import Path
import sys
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

load_enriched_data = data_utils_mod.load_enriched_data
load_product_mix_data = getattr(
    data_utils_mod,
    "load_product_mix_data",
    lambda *_args, **_kwargs: pd.DataFrame(),
)

PAGE_ICON = get_logo_path() or "🏠"
st.set_page_config(page_title="Refuel Control Center", page_icon=PAGE_ICON, layout="wide")

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
    use_weather_api = st.toggle("Use live weather API", value=True, key="home-weather-toggle")

with st.spinner("Loading telemetry for preview..."):
    data = load_enriched_data(use_weather_api=use_weather_api)

if data.empty:
    st.sidebar.error("No telemetry data available yet. Upload a CSV to explore the cockpit.")
    st.stop()

recent_window = data.tail(24)

hero_left, hero_right = st.columns([0.65, 0.35])

with hero_left:
    hero_btn_cols = st.columns(3)
    if hero_btn_cols[0].button("📊 Open Dashboard", use_container_width=True):
        st.switch_page("pages/1_Dashboard.py")
    if hero_btn_cols[1].button("🔮 Run Forecast", use_container_width=True):
        st.switch_page("pages/2_Forecasts.py")
    if hero_btn_cols[2].button("🧾 POS Console", use_container_width=True):
        st.switch_page("pages/7_POS_Console.py")

with hero_right:
    st.metric("Next 24h check-ins", f"{recent_window['checkins'].sum():.0f}")
    st.metric("Next 24h snack units", f"{recent_window['snack_units'].sum():.0f}")
    st.metric(
        "Avg snack price",
        f"CHF{recent_window['snack_price'].mean():.2f}",
        help="Weighted mean over the latest 24 hours.",
    )

daily_summary = (
    data.set_index("timestamp")
    .resample("D")
    .agg({
        "checkins": "sum",
        "snack_units": "sum",
        "temperature_c": "mean",
        "snack_price": "mean"
    })
    .reset_index()
)
daily_summary["date"] = daily_summary["timestamp"].dt.date

product_mix_df = load_product_mix_data()
total_days = max(1, int((data["timestamp"].max() - data["timestamp"].min()).days) or 1)

with st.sidebar:
    history_days = st.slider(
        "History window (days)",
        min_value=3,
        max_value=max(3, total_days),
        value=min(7, max(3, total_days)),
        key="home-history",
    )

weather_meta = data.attrs.get("weather_meta", {})

if weather_meta:
    st.caption(
        f"Weather synced {weather_meta.get('updated_at', 'n/a')} UTC · "
        f"coverage {weather_meta.get('coverage_start', '?')} → {weather_meta.get('coverage_end', '?')}"
    )

st.info(
    "Use the navigation bar or the call-to-action buttons above to jump into specific tools. "
    "The sidebar controls mirror the default data slice used inside each module."
)

st.subheader("How Refuel Works")
st.markdown(
    """
1. **Sync telemetry** – Drop your latest gym+snack CSV into data/, then toggle live weather to merge Open-Meteo forecasts.  
2. **Model demand** – Train lightweight regressors that power the Dashboard, Forecast Explorer, and POS alerts.  
3. **Act** – Use scenario sliders to publish procurement plans and adjust SKU prices in the Price Manager.
"""
)

st.subheader("Product mix outlook")

if isinstance(product_mix_df, pd.DataFrame) and not product_mix_df.empty:
    mix_dates = sorted(product_mix_df["date"].dt.date.unique())
    default_mix_date = mix_dates[-1]
    selected_mix_date = st.select_slider(
        "Product mix date",
        options=mix_dates,
        value=default_mix_date,
        key="mix-date",
        help="Choose a day to inspect the recommended assortment and quantities.",
    )
    mix_slice = product_mix_df[product_mix_df["date"].dt.date == selected_mix_date].copy()
    if not mix_slice.empty:
        mix_slice["weight_pct"] = mix_slice["weight"] * 100
        mix_cost = st.slider(
            "Assumed unit cost (CHF)",
            min_value=0.5,
            max_value=10.0,
            value=3.5,
            step=0.1,
            key="mix-cost-home",
        )
        mix_slice["cost_estimate"] = mix_slice["suggested_qty"] * mix_cost
        info_cols = st.columns(3)
        info_cols[0].metric("Visitors", f"{int(mix_slice['visitors'].iloc[0]):,}")
        info_cols[1].metric("Cardio share", f"{mix_slice['cardio_share'].iloc[0]*100:.1f}%")
        info_cols[2].metric(
            "Weather",
            f"{mix_slice['temp_max_c'].iloc[0]:.1f}°C · {mix_slice['precip_mm'].iloc[0]:.1f} mm",
        )
        mix_fig = px.bar(
            mix_slice,
            x="product",
            y="weight_pct",
            color="season",
            title=f"Recommended product share · {selected_mix_date}",
            labels={"weight_pct": "Mix share (%)", "product": "", "season": "Season"},
        )
        st.plotly_chart(mix_fig, use_container_width=True)
        st.dataframe(
            mix_slice[
                ["product", "suggested_qty", "weight_pct", "cost_estimate", "hot_day", "rainy_day"]
            ]
            .rename(
                columns={
                    "product": "Product",
                    "suggested_qty": "Suggested Qty",
                    "weight_pct": "Mix Share (%)",
                    "cost_estimate": "Est. Cost (CHF)",
                    "hot_day": "Hot?",
                    "rainy_day": "Rainy?",
                }
            )
            .style.format({
                "Suggested Qty": "{:.0f}",
                "Mix Share (%)": "{:.1f}",
                "Est. Cost (CHF)": "CHF{:.0f}",
            }),
            use_container_width=True,
            height=260,
        )
    else:
        st.info("No product mix rows for the selected date.")
else:
    st.info("Product mix file not found yet. Drop data/product_mix_daily.csv to unlock mix insights.")

st.subheader("Day-of-week pricing hints")

dow_stats = (
    data.groupby("weekday")
    .agg(
        checkins=("checkins", "mean"),
        snack_units=("snack_units", "mean"),
        price=("snack_price", "mean")
    )
    .reset_index()
)
dow_stats["weekday_name"] = dow_stats["weekday"].map(
    dict(enumerate(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]))
)
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
    .rename(
        columns={
            "weekday_name": "Weekday",
            "snack_units": "Avg snack units",
            "suggested_price": "Suggested price (CHF)",
        }
    )
    .style.format({"Avg snack units": "{:.1f}", "Suggested price (CHF)": "CHF{:.2f}"})
)

render_footer()
