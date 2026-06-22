import os
import pandas as pd
import streamlit as st
import joblib

# ---------------- MODEL PATH ----------------
MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "models",
    "risk_model.pkl"
)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# ---------------- CSS ----------------
st.markdown("""
<style>
.title {
    font-size: 42px;
    font-weight: bold;
    text-align: center;
    color: #00d4ff;
}
.subtitle {
    text-align: center;
    color: #cfcfcf;
    margin-bottom: 25px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown(
    '<div class="title">🚧 Accident Risk Prediction System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">AI-powered ML model for predicting road accident risk</div>',
    unsafe_allow_html=True
)

st.divider()

# ---------------- MODEL INFO ----------------
with st.expander("📋 Model Information"):
    st.write("Model Type:", type(model))

    if hasattr(model, "feature_names_in_"):
        st.write("Model Features:")
        st.write(list(model.feature_names_in_))

# ---------------- INPUT SECTION ----------------
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📊 Input Features")

    speed = st.slider("Speed (km/h)", 0, 150, 60)

    weather = st.selectbox(
        "Weather Condition",
        ["Clear", "Rainy", "Foggy", "Stormy"]
    )

    road = st.selectbox(
        "Road Type",
        ["Highway", "City", "Rural"]
    )

    traffic = st.slider(
        "Traffic Density",
        0,
        100,
        50
    )

with col2:
    st.markdown("### 🎯 Prediction Result")

    if st.button("Predict Accident Risk 🚨", use_container_width=True):

        weather_map = {
            "Clear": 0,
            "Rainy": 1,
            "Foggy": 2,
            "Stormy": 3
        }

        road_map = {
            "Highway": 0,
            "City": 1,
            "Rural": 2
        }

        input_data = pd.DataFrame({
            "Speed": [speed],
            "Weather": [weather_map[weather]],
            "RoadType": [road_map[road]],
            "TrafficDensity": [traffic]
        })

        prediction = model.predict(input_data)[0]

        if prediction == 1:
            st.error("🔴 High Accident Risk")
            st.write("Drive carefully and stay alert.")
        else:
            st.success("🟢 Low Accident Risk")
            st.write("Current conditions appear relatively safe.")

# ---------------- FOOTER ----------------
st.divider()
st.caption("Built with ❤️ using Streamlit | Accident Risk Prediction Project")