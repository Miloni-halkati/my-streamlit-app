## 🍽️ Zomato Restaurant Success Predictor  
![Powered by Streamlit](https://img.shields.io/badge/Powered%20by-Streamlit-red)  
![Built with Python](https://img.shields.io/badge/Built%20with-Python%203.x-blue)  
![License](https://img.shields.io/badge/License-MIT-green)  

🚀 **Predict whether a restaurant will be successful based on its ratings, cost, and other factors!**  

---

## 📌 Features  
✔️ **Predict restaurant success** based on key parameters  
✔️ **EDA & Data Cleaning** performed in Jupyter Notebook  
✔️ **Machine Learning Model** trained using `Scikit-learn`  
✔️ **Power BI Dashboard** for interactive data visualization  
✔️ **Deployed using Streamlit Cloud**  

---

## 📂 Project Structure  
```
my-streamlit-app/
│── notebooks/          # Jupyter Notebooks for EDA & Data Cleaning
│   ├── zomato_analysis.ipynb
│
│── dashboard/          # Power BI Dashboard
│   ├── Zomato.pbix    # Power BI report file
│
│── models/             # Machine Learning models
│   ├── model.pkl      # Trained ML model
│   ├── scaler.pkl     # Scaler used for preprocessing
│
│── scripts/            # Python scripts
│   ├── train_model.py  # Model training script
│   ├── app.py         # Streamlit app script
│
│── data/               # Raw and processed datasets
│   ├── zomato.csv     # Raw dataset
│   ├── cleaned_zomato.csv # Processed dataset
│
│── requirements.txt    # Dependencies  
│── README.md           # Documentation  
```

---

## 📊 Dataset  
The dataset used in this project is the **Zomato Restaurant Dataset** with features such as:  
- **Average Cost for Two**
- **Restaurant Ratings**
- **Number of Ratings**
- **Online Order Availability**
- **Table Booking Option**

---

## 📦 Installation  
To run the app locally, follow these steps:

### **Step 1: Clone the Repository**  
```bash
git clone https://github.com/Miloni-halkati/my-streamlit-app.git
cd my-streamlit-app
```

### **Step 2: Install Dependencies**  
```bash
pip install -r requirements.txt
```

### **Step 3: Run the App**  
```bash
streamlit run scripts/app.py
```

---

## 🎯 Power BI Dashboard  
📌 The **interactive dashboard** provides key business insights and data visualization.  
📂 Located in the **dashboard/** folder → `Zomato.pbix`  

---

## 🔗 Live Deployment  
🚀 The app is deployed on **Streamlit Cloud**:  
🔗 **(https://miloni-halkati-my-streamlit-app-app-mhdrqm.streamlit.app/)**  
