import streamlit as st
import joblib
import requests
import numpy as np

st.set_page_config(layout="wide")

st.title("📊 Score Predictor")

# 🔥 Load model from GitHub Release
@st.cache_resource
def load_model():
    url = "https://github.com/Jyotikasingh04/IPL_Analysis_Web_app/releases/download/v1.0/score_model_new.1.pkl"
    return joblib.load(requests.get(url, stream=True).raw)

model = load_model()

# ================= INPUTS =================
st.subheader("🏏 Enter Match Details")

col1, col2, col3 = st.columns(3)

with col1:
    current_score = st.number_input("Current Score", min_value=0, step=1)

with col2:
    overs = st.number_input("Overs Completed", min_value=0.0, max_value=20.0, step=0.1)

with col3:
    wickets = st.number_input("Wickets Fallen", min_value=0, max_value=10, step=1)

# ================= FEATURE ENGINEERING =================
balls_left = 120 - int(overs * 6)
run_rate = current_score / overs if overs > 0 else 0

# ================= PREDICTION =================
st.markdown("---")

if st.button("🚀 Predict Final Score"):

    # input format must match training
    input_data = np.array([[current_score, balls_left, wickets, run_rate]])

    prediction = model.predict(input_data)[0]

    st.success(f"🏆 Predicted Final Score: {int(prediction)}")

# ================= INSIGHT =================
st.markdown("---")

st.subheader("🧠 Insight")

st.markdown("""
- Higher current run rate increases final score  
- More wickets fallen reduces scoring potential  
- Fewer balls left = less opportunity to score  
""")