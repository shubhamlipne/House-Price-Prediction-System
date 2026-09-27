"""
frontend/ui.py
----------------
Streamlit UI for the House Price Prediction project.
Collects area & rooms from the user, calls the Flask backend's
/predict endpoint, and displays the estimated price along with
a simple visualization of the training data.
"""

import os

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:5000")
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "house_price.csv")

st.set_page_config(page_title="House Price Prediction", page_icon="🏠", layout="centered")

st.title("🏠 House Price Prediction")
st.write(
    "Estimate a house's price from its **area (sq ft)** and **number of rooms** "
    "using a Linear Regression model served by a Flask API."
)

col1, col2 = st.columns(2)
with col1:
    area = st.number_input("Area (sq ft)", min_value=100, max_value=10000, value=1500, step=50)
with col2:
    rooms = st.number_input("Number of rooms", min_value=1, max_value=10, value=3, step=1)

if st.button("Predict Price", type="primary"):
    try:
        response = requests.post(
            f"{BACKEND_URL}/predict",
            json={"area": area, "rooms": rooms},
            timeout=5,
        )
        if response.status_code == 200:
            result = response.json()
            st.success(f"### Estimated Price: ₹{result['predicted_price']:,.0f}")
        else:
            st.error(f"Backend error: {response.json().get('error', 'unknown error')}")
    except requests.exceptions.ConnectionError:
        st.error(
            "Could not reach the backend API. Make sure it's running "
            f"at {BACKEND_URL} (see README for instructions)."
        )

st.divider()
st.subheader("Training Data Overview")

if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH)
    fig = px.scatter(
        df,
        x="area",
        y="price",
        color="rooms",
        title="House Price vs Area (colored by rooms)",
        labels={"area": "Area (sq ft)", "price": "Price", "rooms": "Rooms"},
    )
    st.plotly_chart(fig, use_container_width=True)
    with st.expander("View raw dataset"):
        st.dataframe(df, use_container_width=True)
else:
    st.info("Dataset not found — run this app from the project root.")
