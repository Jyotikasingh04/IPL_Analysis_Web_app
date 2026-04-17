import streamlit as st
import joblib
import requests
import numpy as np
import io

st.set_page_config(layout="wide")

st.title("🏆 Win Probability Predictor")

@st.cache_resource
def load_model():
    url = "https://github.com/Jyotikasingh04/IPL_Analysis_Web_app/blob/main/model%20(1).pkl"
    
    response = requests.get(url)
    bytes_data = io.BytesIO(response.content)
    
    return joblib.load(bytes_data)

model = load_model()

# ================= INPUTS =================
teams = [
    'Mumbai Indians','Chennai Super Kings','Royal Challengers Bangalore',
    'Kolkata Knight Riders','Delhi Capitals','Punjab Kings',
    'Rajasthan Royals','Sunrisers Hyderabad'
]

st.subheader("🏏 Match Situation")

col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox("Batting Team", sorted(teams))
    target = st.number_input("Target Score", min_value=1)

with col2:
    bowling_team = st.selectbox("Bowling Team", sorted(teams))
    overs = st.number_input("Overs Completed", min_value=0.0, max_value=20.0, step=0.1)

score = st.number_input("Current Score", min_value=0)
wickets = st.number_input("Wickets Fallen", min_value=0, max_value=10)

# ================= FEATURE ENGINEERING =================
runs_left = target - score
balls_left = 120 - int(overs * 6)
wickets_left = 10 - wickets

crr = score / overs if overs > 0 else 0
rrr = (runs_left / balls_left) * 6 if balls_left > 0 else 0

# ================= PREDICTION =================
st.markdown("---")

if st.button("🚀 Predict Win Probability"):

    input_data = np.array([[runs_left, balls_left, wickets_left, crr, rrr]])

    prediction = model.predict_proba(input_data)[0]

    loss_prob = prediction[0]
    win_prob = prediction[1]

    st.success(f"🏆 {batting_team}: {round(win_prob * 100, 2)}% chance to win")
    st.error(f"❌ {bowling_team}: {round(loss_prob * 100, 2)}% chance to win")

    st.progress(int(win_prob * 100))