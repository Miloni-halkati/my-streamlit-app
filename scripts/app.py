import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os

st.title("🍽️ Zomato Restaurant Churn Prediction")
st.write("Enter restaurant details to predict if the restaurant is likely to churn or survive.")

# Check if model and scaler exist
model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
scaler_path = os.path.join(os.path.dirname(__file__), "scaler.pkl")

if os.path.exists(model_path) and os.path.exists(scaler_path):
    model = pickle.load(open(model_path, "rb"))
    scaler = pickle.load(open(scaler_path, "rb"))
    st.success("✅ Model Loaded Successfully!")
else:
    st.error("🚨 Model or Scaler file not found! Please upload `model.pkl` and `scaler.pkl`.")
    st.stop()  # Stop execution if files are missing

# User Inputs
avg_cost = st.number_input("Average Cost for Two People (₹)", min_value=50, max_value=5000, step=50)
rating = st.slider("Restaurant Rating (Out of 5)", 0.0, 5.0, step=0.1)
num_ratings = st.number_input("Number of Ratings", min_value=0, max_value=10000, step=10)
online_order = st.radio("Online Order Available?", ["Yes", "No"])
table_booking = st.radio("Table Booking Available?", ["Yes", "No"])

# Convert categorical inputs
online_order = 1 if online_order == "Yes" else 0
table_booking = 1 if table_booking == "Yes" else 0

# Prepare input for model
input_data = np.array([[avg_cost, rating, num_ratings, online_order, table_booking]])
input_data_scaled = scaler.transform(input_data)

# Predict on button click
if st.button("Predict Churn"):
    prediction = model.predict(input_data_scaled)
    result = "🚨 Churn Likely" if int(prediction[0]) == 1 else "✅ Restaurant is Stable"
    st.subheader(result)
