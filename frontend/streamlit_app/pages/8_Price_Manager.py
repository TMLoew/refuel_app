from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import pandas as pd
import streamlit as st

from frontend.streamlit_app.components.layout import (
    render_top_nav,
    sidebar_info_block,
    render_footer,
    get_logo_path,
)
from frontend.streamlit_app.services.data_utils import (
    DEFAULT_PRODUCT_PRICE,
    load_product_mix_data,
    load_product_prices,
    save_product_prices,
)

PAGE_ICON = get_logo_path() or "💲"
st.set_page_config(page_title="Price Manager", page_icon=PAGE_ICON, layout="wide")

render_top_nav("8_Price_Manager.py")
st.title("Price Manager")
st.caption("Set per-product price points that flow into the dashboard, forecast, and procurement views.")

with st.sidebar:
    sidebar_info_block()
    st.info("Edits here persist to `data/product_prices.csv`.")

mix_df = load_product_mix_data()
price_df = load_product_prices()

# Ensure all known products are represented
known_products = sorted(set(price_df["product"]).union(set(mix_df["product"]) if "product" in mix_df.columns else set()))
if not known_products:
    st.warning("No products detected in the mix file yet. Add rows to `data/product_mix_daily.csv` first.")
    known_products = sorted(price_df["product"].unique())

if known_products:
    merged = (
        pd.DataFrame({"product": known_products})
        .merge(price_df, on="product", how="left")
        .fillna({"unit_price": DEFAULT_PRODUCT_PRICE})
    )
else:
    merged = price_df.copy()

st.subheader("Edit price list")
edited = st.data_editor(
    merged,
    num_rows="dynamic",
    use_container_width=True,
    column_config={
        "product": st.column_config.TextColumn("Product", required=True),
        "unit_price": st.column_config.NumberColumn("Unit price (€)", min_value=0.5, step=0.1),
    },
    key="price-manager-grid",
)

col_save, col_reset = st.columns(2)
if col_save.button("Save prices", use_container_width=True):
    save_product_prices(edited)
    st.success("Prices saved to data/product_prices.csv")
if col_reset.button("Reset to defaults", use_container_width=True):
    defaults = pd.DataFrame({"product": known_products, "unit_price": [DEFAULT_PRODUCT_PRICE] * len(known_products)})
    save_product_prices(defaults)
    st.experimental_rerun()

st.subheader("Current file snapshot")
st.dataframe(load_product_prices(), use_container_width=True, height=280)
render_footer()
