import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("best_model.pkl")

st.set_page_config(page_title="Employee Salary Classification", page_icon="💼", layout="centered")

st.title("Employee Salary Classification App")
st.markdown("Predict whether an employee earns >50K or ≤50K based on input features.")

st.sidebar.header("Input Employee Details")

# Inputs (categorical + numeric)
age = st.sidebar.slider("Age", 18, 65, 30)
education = st.sidebar.selectbox("Education Level", ["Bachelors", "Masters", "PhD", "HS-grad", "Assoc", "Some-college"])
occupation = st.sidebar.selectbox("Occupation", ["Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial", "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical", "Farming-fishing", "Transport-moving", "Priv-house-serv", "Protective-serv", "Armed-Forces"])
hours_per_week = st.sidebar.slider("Hours per week", 1, 80, 40)
experience = st.sidebar.slider("Years of Experience", 0, 40, 5)

# Encoding categorical manually (MUST match training encoding)
education_map = {
    "Some-college": 10, "HS-grad": 9, "Assoc": 12,
    "Bachelors": 13, "Masters": 14, "PhD": 16
}
occupation_map = {
    "Tech-support": 0, "Craft-repair": 1, "Other-service": 2, "Sales": 3,
    "Exec-managerial": 4, "Prof-specialty": 5, "Handlers-cleaners": 6,
    "Machine-op-inspct": 7, "Adm-clerical": 8, "Farming-fishing": 9,
    "Transport-moving": 10, "Priv-house-serv": 11,
    "Protective-serv": 12, "Armed-Forces": 13
}

# Build input_df using encoded values
input_df = pd.DataFrame({
    'age': [age],
    'educational-num': [education_map[education]],
    'occupation': [occupation_map[occupation]],
    'hours-per-week': [hours_per_week],
    'experience': [experience]
})

st.write("### Input Data (Encoded)")
st.write(input_df)

# Predict
if st.button("Predict Salary Class"):
    expected_features = model.feature_names_in_
    for col in expected_features:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[expected_features]

    prediction = model.predict(input_df)
    st.success(f"Prediction: {prediction[0]}")

