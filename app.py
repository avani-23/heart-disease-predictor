import streamlit as st
import pickle
import numpy as np

# Load model and scaler
with open("heart_disease_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

st.title("💓 Heart Disease Predictor")

st.markdown("Enter patient information below:")

# Input fields
age = st.number_input("Age", min_value=1, max_value=120)
sex = st.selectbox("Sex", ["Male", "Female"])
cp = st.selectbox("Chest Pain Type (0–3)", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure", min_value=80, max_value=200)
chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["Yes", "No"])
restecg = st.selectbox("Resting ECG Results (0–2)", [0, 1, 2])
thalach = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=220)
exang = st.selectbox("Exercise Induced Angina", ["Yes", "No"])
oldpeak = st.number_input("Oldpeak (ST depression)", format="%.1f")
slope = st.selectbox("Slope of Peak Exercise ST (0–2)", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels (0–3)", [0, 1, 2, 3])
thal = st.selectbox("Thalassemia (1=Normal, 2=Fixed Defect, 3=Reversible Defect)", [1, 2, 3])

# Encode categorical fields
sex = 1 if sex == "Male" else 0
fbs = 1 if fbs == "Yes" else 0
exang = 1 if exang == "Yes" else 0

# Create input array
input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                        thalach, exang, oldpeak, slope, ca, thal]])

# Scale input
input_data_scaled = scaler.transform(input_data)

# Predict
if st.button("Predict"):
    prediction = model.predict(input_data_scaled)
    if prediction[0] == 1:
        st.error("⚠️ High Risk: The person is likely to have heart disease.")
    else:
        st.success("✅ Low Risk: The person is unlikely to have heart disease.")
