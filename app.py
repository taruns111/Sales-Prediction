import streamlit as st
import pandas as pd
import joblib

model = joblib.load("demand_model.pkl")
le_cat = joblib.load("category_encoder.pkl")
le_sub = joblib.load("sub_category_encoder.pkl")

df = pd.read_csv("sales_data.csv")

st.title("📦 Product Demand Prediction App")
st.write("Category → Sub-Category → Product based demand prediction")

category = st.selectbox(
    "Category",
    sorted(df["Category"].dropna().unique())
)

filtered_subcategories = (
    df[df["Category"] == category]["Sub-Category"]
    .dropna()
    .unique()
)

sub_category = st.selectbox(
    "Sub-Category",
    sorted(filtered_subcategories)
)

filtered_products = (
    df[
        (df["Category"] == category) &
        (df["Sub-Category"] == sub_category)
    ]["Product Name"]
    .dropna()
    .unique()
)

product = st.selectbox(
    "Product Name",
    sorted(filtered_products)
)


avg_price = (
    df[
        (df["Category"] == category) &
        (df["Sub-Category"] == sub_category) &
        (df["Product Name"] == product)
    ]["unit_price"]
    .mean()
)

st.info(f"📌 Average Historical Price (Product Level): ₹{round(avg_price, 2)}")


unit_price = st.number_input(
    "Unit Price (₹)",
    min_value=1.0,
    step=1.0
)

discount = st.slider(
    "Discount (%)",
    0, 80, 10
) / 100


price_ratio = round(unit_price / avg_price, 2)
st.write(f"📊 Auto Calculated Price Ratio: **{price_ratio}**")


if st.button("🔮 Predict Demand"):

    input_df = pd.DataFrame([{
        "unit_price": unit_price,
        "Discount": discount,
        "price_ratio": price_ratio,
        "category_enc": le_cat.transform([category])[0],
        "sub_category_enc": le_sub.transform([sub_category])[0]
    }])

    prediction = model.predict(input_df)[0]

    st.success(f"📈 Predicted Demand Level: **{prediction}**")
