import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv(r"C:\Users\milon\OneDrive\Desktop\my-streamlit-app\data\zomato.csv")

# Convert categorical features
df["online_order"] = df["online_order"].map({"Yes": 1, "No": 0})
df["table booking"] = df["table booking"].map({"Yes": 1, "No": 0})

# Define churn condition based on your new logic
def classify_churn(row):
    # Condition 1: Churn likely if avg cost > 800, rating < 3.5, num_ratings > 15, and no online/table booking
    if row["avg cost (two people)"] > 800 and row["rate (out of 5)"] < 3.5 and row["num of ratings"] > 15 and row["online_order"] == 0 and row["table booking"] == 0:
        return 1  # Churn Likely
    
    # Condition 2: Stable if rating > 3.5, num_ratings > 15, and either online ordering or table booking available
    elif row["rate (out of 5)"] >= 3.5 and row["num of ratings"] > 15 and (row["online_order"] == 1 or row["table booking"] == 1):
        return 0  # Stable Restaurant
    
    # New Condition: Churn likely if rating < 2.5 regardless of other conditions
    elif row["rate (out of 5)"] < 3.5:
        return 1  # Churn Likely
    
    # New Condition: Churn likely if number of ratings is high but rating is still bad (< 3)
    elif row["num of ratings"] > 50 and row["rate (out of 5)"] < 3:
        return 1  # Churn Likely
    
    # Default to stable if none of the churn conditions are met
    else:
        return 0  # Stable


df["churn"] = df.apply(classify_churn, axis=1)

# Features and labels
features = ["avg cost (two people)", "rate (out of 5)", "num of ratings", "online_order", "table booking"]
X = df[features]
y = df["churn"]

# Train-test split with stratification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train RandomForest with Cross Validation
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
best_model = None
best_accuracy = 0

for train_index, test_index in kf.split(X_train_scaled, y_train):
    X_train_fold, X_test_fold = X_train_scaled[train_index], X_train_scaled[test_index]
    y_train_fold, y_test_fold = y_train.iloc[train_index], y_train.iloc[test_index]

    model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    model.fit(X_train_fold, y_train_fold)

    y_pred_fold = model.predict(X_test_fold)
    acc = accuracy_score(y_test_fold, y_pred_fold)

    if acc > best_accuracy:
        best_accuracy = acc
        best_model = model

print(f"✅ Best Model Accuracy: {best_accuracy:.4f}")

# Train final model on full data
X_scaled_final = scaler.fit_transform(X)
best_model.fit(X_scaled_final, y)

# Save model and scaler
pickle.dump(best_model, open("model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))

print("✅ Final Model and Scaler saved successfully!")
