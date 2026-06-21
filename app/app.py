import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ==========================
# Page Config
# ==========================

st.set_page_config(
    page_title="Road Accident Risk Prediction",
    page_icon="🚗",
    layout="centered"
)

# ==========================
# Load Model
# ==========================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "risk_model.pkl"

if not MODEL_PATH.exists():
    st.error(f"Model file not found: {MODEL_PATH}")
    st.stop()

model = joblib.load(MODEL_PATH)

# ==========================
# Title
# ==========================

st.title("🚗 Road Accident Risk Prediction System")

st.write(
    "Enter accident conditions below to predict the risk level."
)

# ==========================
# Inputs
# ==========================

lat = st.number_input(
    "Latitude",
    value=39.8,
    format="%.6f"
)

lng = st.number_input(
    "Longitude",
    value=-84.0,
    format="%.6f"
)

temp = st.number_input(
    "Temperature (F)",
    value=70.0
)

humidity = st.number_input(
    "Humidity (%)",
    value=50.0
)

pressure = st.number_input(
    "Pressure (in)",
    value=29.9
)

visibility = st.number_input(
    "Visibility (mi)",
    value=10.0
)

wind_speed = st.number_input(
    "Wind Speed (mph)",
    value=5.0
)

weather = st.number_input(
    "Weather Condition Encoded",
    value=0
)

hour = st.slider(
    "Hour",
    min_value=0,
    max_value=23,
    value=12
)

month = st.slider(
    "Month",
    min_value=1,
    max_value=12,
    value=6
)

day = st.slider(
    "Day Of Week",
    min_value=0,
    max_value=6,
    value=3
)

# ==========================
# Predict Button
# ==========================

if st.button("Predict Risk"):

    data = pd.DataFrame(
        [[
            lat,
            lng,
            temp,
            humidity,
            pressure,
            visibility,
            wind_speed,
            weather,
            hour,
            month,
            day
        ]],
        columns=[
            "Start_Lat",
            "Start_Lng",
            "Temperature(F)",
            "Humidity(%)",
            "Pressure(in)",
            "Visibility(mi)",
            "Wind_Speed(mph)",
            "Weather_Condition",
            "Hour",
            "Month",
            "DayOfWeek"
        ]
    )

    # Probability of High Risk (Class 1)
    probability = model.predict_proba(data)[0][1]

    st.subheader("Prediction Result")

    st.write(
        f"### High Risk Probability: {probability * 100:.2f}%"
    )

    st.progress(float(probability))

    # Custom threshold
    prediction = 1 if probability >= 0.30 else 0

    if prediction == 1:
        st.error("⚠️ HIGH ACCIDENT RISK")
    else:
        st.success("✅ LOW ACCIDENT RISK")

    st.write("---")

    st.write("Input Summary")

    st.dataframe(data)