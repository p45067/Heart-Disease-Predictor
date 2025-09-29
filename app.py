import streamlit as st
import pandas as pd
import joblib

# ------------------------------
# Load the trained Random Forest model
# ------------------------------
@st.cache_resource
def load_model():
    return joblib.load("random_forest_model.joblib")

model = load_model()

# ------------------------------
# Streamlit App UI
# ------------------------------
st.title("❤️ Heart Disease Prediction App")

st.write("Fill the details below to check the prediction:")

# Example input fields (you can add all features you trained the model on)
age = st.number_input("Age", min_value=20, max_value=100, value=30)
sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
cp = st.number_input("Chest Pain Type (0-3)", min_value=0, max_value=3, value=0)
trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, value=120)
chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1])
restecg = st.number_input("Resting ECG (0-2)", min_value=0, max_value=2, value=0)
thalach = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=220, value=150)
exang = st.selectbox("Exercise Induced Angina", options=[0, 1])
oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
slope = st.number_input("Slope of Peak Exercise ST Segment (0-2)", min_value=0, max_value=2, value=0)
ca = st.number_input("Number of Major Vessels (0-3)", min_value=0, max_value=3, value=0)
thal = st.number_input("Thalassemia (0=Normal, 1=Fixed Defect, 2=Reversible Defect)", min_value=0, max_value=2, value=0)

# ------------------------------
# Prepare Input
# ------------------------------
input_data = pd.DataFrame(
    [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]],
    columns=["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", 
             "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
)

# ------------------------------
# Prediction
# ------------------------------
if st.button("🔍 Predict"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][prediction]

    if prediction == 1:
        st.error(f"⚠️ The model predicts **Heart Disease** with probability {probability:.2f}")
    else:
        st.success(f"✅ The model predicts **No Heart Disease** with probability {probability:.2f}")
