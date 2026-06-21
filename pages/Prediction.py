import streamlit as st
import pandas as pd
import joblib

# -------------------------
# LOAD MODEL + SCALER
# -------------------------
model = joblib.load("../heart_model.pkl")
scaler = joblib.load("../scaler.pkl")

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="centered")

st.title("❤️ Heart Disease Prediction System")

st.markdown("""
This AI system predicts the likelihood of heart disease using machine learning.
Fill in the patient details below.
""")

# -------------------------
# INPUT FIELDS
# -------------------------

age = st.number_input("Age", 1, 120, 45)

sex = st.selectbox("Sex", ["Female", "Male"])
sex = 0 if sex == "Female" else 1

cp = st.selectbox(
    "Chest Pain Type",
    ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"]
)
cp = {"Typical Angina": 0, "Atypical Angina": 1, "Non-Anginal Pain": 2, "Asymptomatic": 3}[cp]

trestbps = st.number_input("Resting Blood Pressure", 50, 250, 120)
chol = st.number_input("Cholesterol", 50, 700, 200)

fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
fbs = 0 if fbs == "No" else 1

restecg = st.selectbox(
    "Resting ECG",
    ["Normal", "ST-T Abnormality", "Left Ventricular Hypertrophy"]
)
restecg = {"Normal": 0, "ST-T Abnormality": 1, "Left Ventricular Hypertrophy": 2}[restecg]

thalach = st.number_input("Max Heart Rate", 50, 250, 150)

exang = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
exang = 0 if exang == "No" else 1

oldpeak = st.number_input("Oldpeak", 0.0, 10.0, 1.0, step=0.1)

slope = st.selectbox("Slope", ["Upsloping", "Flat", "Downsloping"])
slope = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}[slope]

ca = st.selectbox("Major Vessels (0-4)", [0, 1, 2, 3, 4])

thal = st.selectbox("Thalassemia", ["Normal", "Fixed Defect", "Reversible Defect"])
thal = {"Normal": 1, "Fixed Defect": 2, "Reversible Defect": 3}[thal]

# -------------------------
# PREDICTION BUTTON
# -------------------------
if st.button("Predict Heart Disease"):

    # Create dataframe
    input_data = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs,
                                restecg, thalach, exang, oldpeak,
                                slope, ca, thal]],
                              columns=[
                                  'age','sex','cp','trestbps','chol','fbs',
                                  'restecg','thalach','exang','oldpeak',
                                  'slope','ca','thal'
                              ])

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)

    # -------------------------
    # OUTPUT
    # -------------------------
    st.subheader("Result")

    if prediction[0] == 0:
        st.success("✅ No Heart Disease Detected")
        risk = "Low Risk"
    else:
        st.error("⚠️ Heart Disease Detected")
        risk = "High Risk"

    # Confidence
    confidence = probability[0][prediction[0]] * 100

    st.write(f"### Risk Level: {risk}")
    st.write(f"### Confidence: {confidence:.2f}%")

    st.write("### Prediction Probabilities")
    st.write({
        "No Disease": f"{probability[0][0]*100:.2f}%",
        "Disease": f"{probability[0][1]*100:.2f}%"
    })

# -------------------------
# SIDEBAR
# -------------------------
st.sidebar.title("About")

st.sidebar.info("""
Heart Disease Prediction AI App

Models Tested:
- Logistic Regression
- Random Forest
- SVM
- Naive Bayes
- AdaBoost
- Stacking

Final Model:
Random Forest
""")
