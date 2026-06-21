import streamlit as st
import pickle
import numpy as np

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Accident Risk Predictor",
    page_icon="🚧",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------
model = pickle.load(open("risk_model.pkl", "rb"))

# ---------------- CUSTOM CSS ----------------
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .title {
        font-size: 40px;
        font-weight: bold;
        color: #00d4ff;
        text-align: center;
    }
    .subtitle {
        text-align: center;
        color: #cfcfcf;
        margin-bottom: 30px;
    }
    .card {
        background-color: #161b22;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0px 0px 10px rgba(0,0,0,0.5);
    }
    </style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown('<div class="title">🚧 Accident Risk Prediction System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-powered system to predict road accident risk level</div>', unsafe_allow_html=True)

st.divider()

# ---------------- LAYOUT ----------------
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📊 Input Features")

    speed = st.slider("Speed (km/h)", 0, 150, 60)
    weather = st.selectbox("Weather Condition", ["Clear", "Rainy", "Foggy", "Stormy"])
    road = st.selectbox("Road Type", ["Highway", "City", "Rural"])
    traffic = st.slider("Traffic Density", 0, 100, 50)

with col2:
    st.markdown("### 🎯 Prediction Panel")

    if st.button("Predict Risk 🚨", use_container_width=True):

        # Example encoding (adjust based on your training)
        weather_map = {"Clear": 0, "Rainy": 1, "Foggy": 2, "Stormy": 3}
        road_map = {"Highway": 0, "City": 1, "Rural": 2}

        features = np.array([[speed,
                              weather_map[weather],
                              road_map[road],
                              traffic]])

        prediction = model.predict(features)[0]

        st.markdown("### Result")

        if prediction == 1:
            st.error("🔴 High Accident Risk")
        elif prediction == 0:
            st.success("🟢 Low Accident Risk")
        else:
            st.warning("🟡 Medium Risk")

# ---------------- FOOTER ----------------
st.divider()
st.caption("Built with ❤️ using Streamlit | ML Project by Radhey Mohan Singh")

