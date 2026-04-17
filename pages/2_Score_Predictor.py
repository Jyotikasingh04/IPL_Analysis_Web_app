import streamlit as st
from helper import predict_score

st.title("📊 Score Predictor")

current_score = st.number_input("Current Score")
overs = st.number_input("Overs Completed")
wickets = st.number_input("Wickets Fallen")

balls_left = 120 - int(overs * 6)
run_rate = current_score / overs if overs > 0 else 0

if st.button("Predict Score"):

    result = predict_score(current_score, balls_left, wickets, run_rate)

    st.success(f"Predicted Score: {result}")