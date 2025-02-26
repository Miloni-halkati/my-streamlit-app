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
df["churn"] = df.apply(
    lambda x: 1 if (
        x["rate (out of 5)"] < 3 or
        x["num of ratings"] < 30 or  # Reduced from 50 → 30
        (x["avg cost (two people)"] > 1500 and x["rate (out of 5)"] < 3.5)  # Adjusted cost & rating
    ) else 0,
    axis=1
)

X = df[features]
y = df["churn"]

print("Churn Distribution:\n", y.value_counts())

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Check for imbalance & Apply SMOTE if needed
churn_counts = y_train.value_counts()
if churn_counts[1] > 1.5 * churn_counts[0]:  # If churn cases are much higher than non-churn
    print("⚠️ Data imbalance detected. Applying SMOTE...")
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

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
