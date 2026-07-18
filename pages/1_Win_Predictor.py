import streamlit as st

from helper import predict_win_probability

st.title("Win Probability Predictor")

teams = [
    'Mumbai Indians', 'Chennai Super Kings', 'Royal Challengers Bangalore',
    'Kolkata Knight Riders', 'Delhi Capitals', 'Punjab Kings',
    'Rajasthan Royals', 'Sunrisers Hyderabad'
]

# A realistic ceiling for a T20 innings total. Used only to catch obviously
# invalid manual entries (e.g. typos like 700), not a hard cricketing limit.
MAX_REALISTIC_SCORE = 350

batting_team = st.selectbox("Batting Team", sorted(teams))
bowling_team = st.selectbox("Bowling Team", sorted(teams))

target = st.number_input("Target Score", min_value=1)
score = st.number_input("Current Score", min_value=0)
overs = st.number_input("Overs Completed", min_value=0.0, max_value=20.0, step=0.1)
wickets = st.number_input("Wickets Fallen", min_value=0, max_value=10)

if st.button("Predict"):

    # -------------------------------------------------------------------
    # 1. Team validation
    # -------------------------------------------------------------------
    if batting_team == bowling_team:
        st.error("Batting and Bowling team cannot be the same.")
        st.stop()

    # -------------------------------------------------------------------
    # 2. Score sanity checks
    # -------------------------------------------------------------------
    if target > MAX_REALISTIC_SCORE:
        st.error(f"Target Score looks invalid (over {MAX_REALISTIC_SCORE}). Please check the input.")
        st.stop()

    if score > MAX_REALISTIC_SCORE:
        st.error(f"Current Score looks invalid (over {MAX_REALISTIC_SCORE}). Please check the input.")
        st.stop()

    if score > target:
        st.error("Current Score cannot be greater than Target Score. Please check the inputs.")
        st.stop()

    # -------------------------------------------------------------------
    # 3. Overs validation.
    #    Cricket overs are X.Y where Y is balls bowled in the current over
    #    (0-5). X.6, X.7 etc. are not valid overs and would silently
    #    corrupt balls_left / run rates if left unchecked.
    #    int(round(...)) instead of round(...) so Streamlit's float
    #    representation (e.g. 19.5 stored as 19.499999999) doesn't get
    #    truncated to the wrong ball count.
    # -------------------------------------------------------------------
    completed_overs = int(overs)
    balls_in_current_over = int(round((overs - completed_overs) * 10))

    if balls_in_current_over > 5 or balls_in_current_over < 0:
        st.error("Invalid overs value. The decimal part must be between .0 and .5 "
                  "(e.g. 12.4 = 12 overs and 4 balls).")
        st.stop()

    balls_bowled = completed_overs * 6 + balls_in_current_over
    balls_left = 120 - balls_bowled
    wickets_left = 10 - wickets
    runs_left = target - score

    # -------------------------------------------------------------------
    # 4. Match-state edge cases, handled BEFORE calling the model.
    #    The model was trained only on live, in-progress second-innings
    #    situations, so these terminal states are handled explicitly
    #    rather than fed into predict_proba().
    # -------------------------------------------------------------------
    if wickets_left <= 0:
        st.error(f"{batting_team} are all out. {bowling_team} win the match.")
        st.stop()

    if runs_left <= 0:
        st.success(f"{batting_team} have already chased the target. {batting_team} win the match.")
        st.stop()

    if balls_left <= 0:
        st.error(f"Innings/over limit reached with target not chased. {bowling_team} win the match.")
        st.stop()

    # -------------------------------------------------------------------
    # 5. Compute the exact features the model was trained on.
    # -------------------------------------------------------------------
    current_run_rate = (score / balls_bowled) * 6 if balls_bowled > 0 else 0.0
    required_run_rate = (runs_left / balls_left) * 6

    # -------------------------------------------------------------------
    # 6. Call the trained Logistic Regression model. No heuristic logic
    #    remains in this page.
    # -------------------------------------------------------------------
    win_prob = predict_win_probability(
        runs_left=runs_left,
        balls_left=balls_left,
        wickets_left=wickets_left,
        crr=current_run_rate,
        rrr=required_run_rate
    )
    loss_prob = round(100 - win_prob, 2)

    st.success(f"{batting_team}: {win_prob}% chance to win")
    st.error(f"{bowling_team}: {loss_prob}% chance to win")
    st.progress(int(win_prob))
