import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Student GPA Predictor",
    layout="wide"
)

# Load model, scaler, and columns
model = joblib.load("gpa_prediction_model.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")

# Title
st.title("Student GPA Prediction System")

st.write("Predict a student's GPA using a Linear Regression Machine Learning model.")

st.write("---")

# Input Section
col1, col2 = st.columns(2)

with col1:
    study = st.number_input(
        "Study Hours Per Day",
        min_value=0.0,
        max_value=24.0,
        value=5.0,
        step=0.5
    )

    extra = st.number_input(
        "Extracurricular Hours Per Day",
        min_value=0.0,
        max_value=24.0,
        value=2.0,
        step=0.5
    )

    sleep = st.number_input(
        "Sleep Hours Per Day",
        min_value=0.0,
        max_value=24.0,
        value=7.0,
        step=0.5
    )

with col2:
    social = st.number_input(
        "Social Hours Per Day",
        min_value=0.0,
        max_value=24.0,
        value=2.0,
        step=0.5
    )

    physical = st.number_input(
        "Physical Activity Hours Per Day",
        min_value=0.0,
        max_value=24.0,
        value=1.0,
        step=0.5
    )

    stress = st.selectbox(
        "Stress Level",
        ["High", "Low", "Moderate"]
    )

# Prediction
if st.button("Predict GPA"):

    # Create input dataframe
    input_df = pd.DataFrame(0, index=[0], columns=columns)

    # Numerical features
    input_df["Study_Hours_Per_Day"] = study
    input_df["Extracurricular_Hours_Per_Day"] = extra
    input_df["Sleep_Hours_Per_Day"] = sleep
    input_df["Social_Hours_Per_Day"] = social
    input_df["Physical_Activity_Hours_Per_Day"] = physical

    # One-hot encoding for Stress Level
    for col in columns:
        if "Stress_Level_Low" in col and stress == "Low":
            input_df[col] = 1
        elif "Stress_Level_Moderate" in col and stress == "Moderate":
            input_df[col] = 1

    # Scale input
    input_scaled = scaler.transform(input_df)

    # Predict GPA
    prediction = model.predict(input_scaled)[0]

    st.success("Prediction Completed Successfully")
    st.metric("Predicted GPA", f"{prediction:.2f}")

    # Performance message
    if prediction >= 3.5:
        st.success("Excellent Academic Performance")
    elif prediction >= 3.0:
        st.info("Good Academic Performance")
    elif prediction >= 2.5:
        st.warning("Average Academic Performance")
    else:
        st.error("Needs Improvement")

st.write("---")

st.subheader("About")

st.write("Machine Learning Model: Linear Regression")
st.write("Target Variable: GPA")

st.write("Features Used:")
st.write("- Study Hours Per Day")
st.write("- Extracurricular Hours Per Day")
st.write("- Sleep Hours Per Day")
st.write("- Social Hours Per Day")
st.write("- Physical Activity Hours Per Day")
st.write("- Stress Level")

st.write("Developed by Achal Chaudhari using Streamlit and Scikit-Learn.")