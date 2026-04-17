import streamlit as st
from helper import predict_win_probability

st.title("Win Probability Predictor")

teams = [
    'Mumbai Indians','Chennai Super Kings','Royal Challengers Bangalore',
    'Kolkata Knight Riders','Delhi Capitals','Punjab Kings',
    'Rajasthan Royals','Sunrisers Hyderabad'
]

cities = ['Mumbai','Delhi','Chennai','Kolkata','Bangalore','Hyderabad','Jaipur']

batting_team = st.selectbox("Batting Team", sorted(teams))
bowling_team = st.selectbox("Bowling Team", sorted(teams))

target = st.number_input("Target Score")
score = st.number_input("Current Score")
overs = st.number_input("Overs Completed")
wickets = st.number_input("Wickets Fallen")

if st.button("Predict"):

    runs_left = target - score
    balls_left = 120 - int(overs * 6)
    wickets_left = 10 - wickets

    crr = score / overs if overs > 0 else 0
    rrr = (runs_left / balls_left) * 6 if balls_left > 0 else 0

    result = predict_win_probability(runs_left, balls_left, wickets_left, crr, rrr)

    st.success(f"{batting_team}: {result}% chance to win")
    st.progress(int(result))