import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv(r"C:\Users\milon\OneDrive\Desktop\my-streamlit-app\data\zomato.csv")

# Define features and target variable
features = ["avg cost (two people)", "rate (out of 5)", "num of ratings", "online_order", "table booking"]
df["online_order"] = df["online_order"].map({"Yes": 1, "No": 0})
df["table booking"] = df["table booking"].map({"Yes": 1, "No": 0})

# Define churn condition
df["churn"] = df.apply(lambda x: 1 if (x["rate (out of 5)"] < 2.5 and x["num of ratings"] < 10) else 0, axis=1)

# Split Features & Target
X = df[features]
y = df["churn"]

# Initialize K-Fold
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
best_model = None
best_accuracy = 0

accuracies = []

for train_index, test_index in kf.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Scale separately for each fold
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    # Validate Model
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)

    # Store Best Model
    if acc > best_accuracy:
        best_accuracy = acc
        best_model = model
        best_scaler = scaler  # Save the scaler used for the best model

print(f"✅ Best Accuracy across folds: {best_accuracy:.4f}")

# Retrain on Full Dataset with Best Model
X_scaled_final = best_scaler.fit_transform(X)
best_model.fit(X_scaled_final, y)

# Save Model & Scaler
pickle.dump(best_model, open("model.pkl", "wb"))
pickle.dump(best_scaler, open("scaler.pkl", "wb"))

print("✅ Final Model and Scaler saved successfully!")
