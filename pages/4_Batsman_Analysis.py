import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

df = pd.read_csv('deliveries.csv')

st.title("🏏 Batsman Analysis")

# dropdown
players = sorted(df['batter'].unique())
selected_player = st.selectbox("Select Batsman", players)

player_df = df[df['batter'] == selected_player]

# metrics
runs = player_df['batsman_runs'].sum()
balls = player_df.shape[0]
sr = (runs / balls) * 100 if balls > 0 else 0

fours = player_df[player_df['batsman_runs'] == 4].shape[0]
sixes = player_df[player_df['batsman_runs'] == 6].shape[0]

col1, col2, col3, col4 = st.columns(4)

col1.metric("Runs", runs)
col2.metric("Balls", balls)
col3.metric("Strike Rate", round(sr, 2))
col4.metric("4s / 6s", f"{fours}/{sixes}")

# 🔥 Runs per match graph
st.subheader("📈 Performance Trend")

runs_per_match = player_df.groupby('match_id')['batsman_runs'].sum()

fig = plt.figure(figsize=(10,5))
plt.plot(runs_per_match.values)
plt.xlabel("Matches")
plt.ylabel("Runs")
plt.title("Runs per Match")

st.pyplot(fig)

# 🔥 Shot distribution
st.subheader("🎯 Shot Distribution")

singles = player_df[player_df['batsman_runs'] == 1].shape[0]

fig2 = plt.figure(figsize=(5,5))
plt.pie(
    [singles, fours, sixes],
    labels=["1s", "4s", "6s"],
    autopct="%1.1f%%"
)

st.pyplot(fig2)

# 🔥 Insight
st.subheader("🧠 Insight")

if sr > 140:
    st.success("Aggressive batter 🔥")
elif sr > 120:
    st.info("Balanced player ⚖️")
else:
    st.warning("Anchor type player 🧱")