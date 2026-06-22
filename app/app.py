import os
import numpy as np
import joblib
import streamlit as st

MODEL_PATH = os.path.join(os.path.dirname(__file__), "../models/risk_model.pkl")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()
st.write("Features expected by model:")
st.write(model.feature_names_in_)

# DEBUG - shows on page load, no button needed
st.write("Feature names:", list(model.feature_names_in_))

# ---------------- CUSTOM CSS ----------------
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
    .box {
        background-color: #161b22;
        padding: 20px;
        border-radius: 12px;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown('<div class="title">🚧 Accident Risk Prediction System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-powered ML model for predicting road accident risk</div>', unsafe_allow_html=True)
st.divider()

# ---------------- INPUT SECTION ----------------
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📊 Input Features")
    speed = st.slider("Speed (km/h)", 0, 150, 60)
    weather = st.selectbox("Weather", ["Clear", "Rainy", "Foggy", "Stormy"])
    road = st.selectbox("Road Type", ["Highway", "City", "Rural"])
    traffic = st.slider("Traffic Density", 0, 100, 50)

with col2:
    st.markdown("### 🎯 Prediction Result")
    if st.button("Predict Accident Risk 🚨", use_container_width=True):
        # STOP prediction, just show feature names
        st.info("Your model needs these 11 features:")
        st.write(list(model.feature_names_in_))
        st.stop()

# ---------------- FOOTER ----------------
st.divider()
st.caption("Built with ❤️ using Streamlit | Accident Risk Prediction Project")
