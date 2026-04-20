import streamlit as st
import joblib
import requests
import numpy as np
import io

st.title("Win Probability Predictor")

@st.cache_resource
def load_model():
    url = "https://raw.githubusercontent.com/Jyotikasingh04/IPL_Analysis_Web_app/main/model%20(1).pkl"
    response = requests.get(url)
    return joblib.load(io.BytesIO(response.content))

model = load_model()

teams = [
    'Mumbai Indians','Chennai Super Kings','Royal Challengers Bangalore',
    'Kolkata Knight Riders','Delhi Capitals','Punjab Kings',
    'Rajasthan Royals','Sunrisers Hyderabad'
]

batting_team = st.selectbox("Batting Team", sorted(teams))
bowling_team = st.selectbox("Bowling Team", sorted(teams))

target = st.number_input("Target Score", min_value=1)
score = st.number_input("Current Score", min_value=0)
overs = st.number_input("Overs Completed", min_value=0.0, max_value=20.0)
wickets = st.number_input("Wickets Fallen", min_value=0, max_value=10)

if st.button("Predict"):

    runs_left = target - score
    balls_left = 120 - int(overs * 6)
    wickets_left = 10 - wickets

    if balls_left <= 0:
        st.error("Match Over")
    else:
        # Required run rate
        rrr = (runs_left / balls_left) * 6 if balls_left > 0 else 0

        # Base probability logic
        win_prob = 0.5

        # 🔥 Key cricket factors
        if runs_left <= 0:
            win_prob = 0.99

        elif rrr > 12:
            win_prob -= 0.3

        elif rrr > 9:
            win_prob -= 0.15

        elif rrr < 6:
            win_prob += 0.2

        # wickets effect
        if wickets_left > 6:
            win_prob += 0.1
        elif wickets_left < 3:
            win_prob -= 0.2

        # balls pressure
        if balls_left < 30:
            win_prob -= 0.1

        # clamp
        win_prob = max(min(win_prob, 0.95), 0.05)
        loss_prob = 1 - win_prob

        st.success(f" {batting_team}: {round(win_prob * 100,2)}% chance to win")
        st.error(f"{bowling_team}: {round(loss_prob * 100,2)}% chance to win")

        st.progress(int(win_prob * 100))
