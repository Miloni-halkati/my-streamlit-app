import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("zomato.csv")

# Define features and target variable
features = ["avg cost (two people)", "rate (out of 5)", "num of ratings", "online_order", "table booking"]
df["online_order"] = df["online_order"].map({"Yes": 1, "No": 0})
df["table booking"] = df["table booking"].map({"Yes": 1, "No": 0})
df["churn"] = df.apply(lambda x: 1 if (x["rate (out of 5)"] < 3 or x["num of ratings"] < 50 or (x["avg cost (two people)"] > 1000 and x["rate (out of 5)"] < 4)) else 0, axis=1)

X = df[features]
y = df["churn"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_np = X_train.to_numpy()
scaler.fit(X_train_np)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_np, y_train)

# Save model and scaler
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))

print("✅ Model and scaler saved successfully!")
