import streamlit as st
import numpy as np
import joblib
import os

# Load pipeline
pipeline_path = "house_price_pipeline.pkl"
if not os.path.exists(pipeline_path):
    st.error("❌ Missing pipeline file!")
    st.stop()

pipeline = joblib.load(pipeline_path)

# Features order
features_order = ["floors", "waterfront", "lat", "bedrooms", "sqft_basement",
                  "view", "bathrooms", "sqft_living15", "sqft_above", "grade", "sqft_living"]

st.title("🏡 House Price Prediction App")

# Inputs
floors = st.number_input("Floors", 1.0, 5.0, 1.0)
waterfront = st.selectbox("Waterfront", [0, 1])
lat = st.number_input("Latitude", 47.0, 48.0, 47.5)
bedrooms = st.number_input("Bedrooms", 1, 10, 3)
sqft_basement = st.number_input("Basement sqft", 0, 3000, 0)
view = st.number_input("View (0-4)", 0, 4, 0)
bathrooms = st.number_input("Bathrooms", 1.0, 10.0, 2.0)
sqft_living15 = st.number_input("Living15 sqft", 500, 10000, 2000)
sqft_above = st.number_input("Above sqft", 500, 8000, 1500)
grade = st.number_input("Grade (1-13)", 1, 13, 7)
sqft_living = st.number_input("Living sqft", 500, 10000, 2000)

# Prepare input
input_data = np.array([[floors, waterfront, lat, bedrooms, sqft_basement, view,
                        bathrooms, sqft_living15, sqft_above, grade, sqft_living]])

# Prediction
if st.button("Predict Price"):
    price = pipeline.predict(input_data)[0]
    st.success(f"Estimated House Price: ${price:,.2f}")
