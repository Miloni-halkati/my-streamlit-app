import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load trained model and scaler
import os
<<<<<<< HEAD

model_path = os.path.join(os.path.dirname(__file__), "..", "models", "model.pkl")
model = pickle.load(open(model_path, "rb"))
=======
# Get the correct absolute path to model.pkl
model_path = os.path.join(os.path.dirname(__file__), "..", "models", "model.pkl")
# Load the model
model = pickle.load(open(model_path, "rb"))

>>>>>>> origin/main
scaler_path = os.path.join(os.path.dirname(__file__), "..", "models", "scaler.pkl")
scaler = pickle.load(open(scaler_path, "rb"))

# Streamlit UI
st.title("🍽️ Zomato Restaurant Churn Prediction")
st.write("Enter restaurant details to predict if the restaurant is likely to churn or survive.")

# User inputs
avg_cost = st.number_input("Average Cost for Two People (₹)", min_value=50, max_value=5000, step=50)
rating = st.slider("Restaurant Rating (Out of 5)", 0.0, 5.0, step=0.1)
num_ratings = st.number_input("Number of Ratings", min_value=0, max_value=10000, step=10)
online_order = st.radio("Online Order Available?", ["Yes", "No"])
table_booking = st.radio("Table Booking Available?", ["Yes", "No"])

# Convert categorical inputs
online_order = 1 if online_order == "Yes" else 0
table_booking = 1 if table_booking == "Yes" else 0

# Prepare input for model
feature_names = ["avg cost (two people)", "rate (out of 5)", "num of ratings", "online_order", "table booking"]
input_data = pd.DataFrame([[avg_cost, rating, num_ratings, online_order, table_booking]], columns=feature_names)

# Convert to NumPy array
input_data_np = input_data.to_numpy()

# Scale the input
input_data_scaled = scaler.transform(input_data_np)

# Predict
if st.button("Predict Churn"):
    prediction = model.predict(input_data_scaled)
    result = "🚨 Churn Likely" if prediction[0] == 1 else "✅ Restaurant is Stable"
    st.subheader(result)
