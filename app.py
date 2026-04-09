#. /usr/local/bin/python3 -m streamlit run app.py

import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(
    page_title="Diabetes Prediction System", 
    layout="wide",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': None
    }
)

# Hide all Streamlit's default UI elements
st.markdown("""
<style>
#MainMenu {visibility: hidden !important;}
footer {visibility: hidden !important;}
.viewerBadge_container__1QSob {display: none !important;}
.css-1v3fvcr {display: none !important;}  /* Deploy button class fallback */
.css-1b7w8j8 {display: none !important;}  /* Other menu items */
[data-testid="stSidebar"] .css-1v3fvcr {display: none !important;}
button[aria-label="Open online demo"] {display: none !important;}
button[aria-label="Deploy"] {display: none !important;}
</style>
<script>
window.addEventListener('load', function() {
  const hideDeploy = () => {
    [...document.querySelectorAll('button'), ...document.querySelectorAll('div[role=button]')].forEach(el => {
      if (el.textContent.trim().toLowerCase().includes('deploy')) {
        el.style.display = 'none';
      }
    });
  };
  hideDeploy();
  setTimeout(hideDeploy, 1000);
  setTimeout(hideDeploy, 2000);
});
</script>
""", unsafe_allow_html=True)

# Load models
models = joblib.load("all_models.pkl")
label_encoder = joblib.load("label_encoder.pkl")

st.title("Diabetes Prediction System")

# Define best model
best_model_name = "Logistic Regression"

# Create model options with best model label
model_options = [f"{name} {'⭐ (Best Model)' if name == best_model_name else ''}" for name in models.keys()]
model_display = st.sidebar.selectbox("Select Model", model_options)

# Extract actual model name (remove the best model label)
model_choice = model_display.split(" ⭐")[0] if " ⭐" in model_display else model_display
model = models[model_choice]

st.subheader("Enter Patient Details")

# Input fields
age = st.number_input("Age", 1, 100)
gender = st.selectbox("Gender", ["Male", "Female"])
bmi = st.number_input("BMI", 10.0, 50.0)
bp = st.number_input("Blood Pressure", 80, 200)
glucose = st.number_input("Fasting Glucose Level", 50, 300)
insulin = st.number_input("Insulin Level", 0.0, 300.0)
hba1c = st.number_input("HbA1c Level", 3.0, 15.0)
cholesterol = st.number_input("Cholesterol Level", 100, 400)
triglycerides = st.number_input("Triglycerides Level", 50, 500)
activity = st.selectbox("Physical Activity Level", ["Low", "Moderate", "High"])
calories = st.number_input("Daily Calorie Intake", 1000, 5000)
sugar = st.number_input("Sugar Intake (grams/day)", 0.0, 300.0)
sleep = st.number_input("Sleep Hours", 1.0, 12.0)
stress = st.slider("Stress Level", 1, 10)
family = st.selectbox("Family History of Diabetes", ["Yes", "No"])
waist = st.number_input("Waist Circumference (cm)", 50.0, 150.0)

if st.button("Predict"):
    st.session_state["run_prediction"] = True

if "run_prediction" in st.session_state:

    input_data = pd.DataFrame([{
        "age": age,
        "gender": gender,
        "bmi": bmi,
        "blood_pressure": bp,
        "fasting_glucose_level": glucose,
        "insulin_level": insulin,
        "HbA1c_level": hba1c,
        "cholesterol_level": cholesterol,
        "triglycerides_level": triglycerides,
        "physical_activity_level": activity,
        "daily_calorie_intake": calories,
        "sugar_intake_grams_per_day": sugar,
        "sleep_hours": sleep,
        "stress_level": stress,
        "family_history_diabetes": family,
        "waist_circumference_cm": waist
    }])

    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)

    risk = label_encoder.inverse_transform(prediction)[0]

    st.success(f"Predicted Risk Category: {risk}")    
