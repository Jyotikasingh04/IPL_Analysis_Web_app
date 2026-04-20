import streamlit as st
import joblib
import requests
import numpy as np
import io

st.set_page_config(page_title="IPL Score Predictor", layout="wide")

# ================= LOAD MODEL =================
@st.cache_resource
def load_model():
    url = "https://github.com/Jyotikasingh04/IPL_Analysis_Web_app/releases/download/v1.0/score_model_new.1.pkl"
    response = requests.get(url)
    return joblib.load(io.BytesIO(response.content))

model = load_model()

# ================= HEADER =================
st.markdown(
    """
    <h1 style='text-align: center;'>IPL Score Predictor</h1>
    <p style='text-align: center; color: gray;'>Predict final score based on current match situation</p>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# ================= INPUT SECTION =================
st.subheader("Enter Match Details")

col1, col2, col3 = st.columns(3)

with col1:
    current_score = st.number_input("Current Score", min_value=0, step=1)

with col2:
    overs = st.number_input("Overs Completed", min_value=0.0, max_value=20.0, step=0.1)

with col3:
    wickets = st.slider("Wickets Fallen", 0, 10, 0)

# ================= LIVE METRICS =================
balls_left = 120 - int(overs * 6)
run_rate = current_score / overs if overs > 0 else 0

st.markdown("### Live Match Metrics")

m1, m2, m3 = st.columns(3)

m1.metric("Balls Left", balls_left)
m2.metric("Current Run Rate", round(run_rate, 2))
m3.metric("Wickets in Hand", 10 - wickets)

st.markdown("---")

# ================= PREDICTION =================
if st.button("Predict Final Score"):

    input_data = np.array([[current_score, balls_left, wickets, run_rate]])
    prediction = model.predict(input_data)[0]

    # Highlight box
    st.markdown(
        f"""
        <div style='
            background-color:#1f77b4;
            padding:20px;
            border-radius:10px;
            text-align:center;
            color:white;
            font-size:28px;
            font-weight:bold;
        '>
            Predicted Final Score: {int(prediction)}
        </div>
        """,
        unsafe_allow_html=True
    )

# ================= INSIGHTS =================
st.markdown("---")

st.subheader("Match Insights")

col1, col2 = st.columns(2)

with col1:
    st.info("Higher run rate increases scoring potential")
    st.info("Fewer balls left reduces scoring opportunities")

with col2:
    st.info("More wickets fallen reduces momentum")
    st.info("Powerplay and death overs impact scoring")

