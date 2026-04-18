import streamlit as st
import joblib
import requests
import numpy as np
import io

st.title("🏆 Win Probability Predictor")

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

    crr = score / overs if overs > 0 else 0
    rrr = (runs_left / balls_left) * 6 if balls_left > 0 else 0

    input_data = np.array([[runs_left, balls_left, wickets_left, crr, rrr]])

    prediction = model.predict(input_data)[0]

    st.success(f"📊 Predicted Score: {round(prediction, 2)}")