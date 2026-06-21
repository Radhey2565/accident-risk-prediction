import os
import pickle
import streamlit as st

MODEL_PATH = os.path.join(os.path.dirname(__file__), "../models/risk_model.pkl")

@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()
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

        # Encoding (must match training)
        weather_map = {"Clear": 0, "Rainy": 1, "Foggy": 2, "Stormy": 3}
        road_map = {"Highway": 0, "City": 1, "Rural": 2}

        features = np.array([[
            speed,
            weather_map[weather],
            road_map[road],
            traffic
        ]])

        prediction = model.predict(features)[0]

        st.markdown("### Result:")

        if prediction == 1:
            st.error("🔴 High Accident Risk")
            st.write("Take extra caution while driving.")
        elif prediction == 0:
            st.success("🟢 Low Accident Risk")
            st.write("Conditions are safe for driving.")
        else:
            st.warning("🟡 Medium Risk")

# ---------------- FOOTER ----------------
st.divider()
st.caption("Built with ❤️ using Streamlit | Accident Risk Prediction Project")

