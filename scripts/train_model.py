import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

df = pd.read_csv(r"C:\Users\milon\OneDrive\Desktop\my-streamlit-app\data\zomato.csv")

features = ["avg cost (two people)", "rate (out of 5)", "num of ratings", "online_order", "table booking"]
df["online_order"] = df["online_order"].map({"Yes": 1, "No": 0})
df["table booking"] = df["table booking"].map({"Yes": 1, "No": 0})

df["churn"] = df.apply(lambda x: 1 if (x["rate (out of 5)"] < 2.5 and x["num of ratings"] < 10) else 0, axis=1)

X = df[features]
y = df["churn"]

scaler = StandardScaler()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)

accuracies = []

for train_index, test_index in kf.split(X_scaled, y):
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)

print(f"✅ Average Accuracy across folds: {np.mean(accuracies):.4f}")

model.fit(X_scaled, y)

pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))

print("✅ Final Model and Scaler saved successfully!")
