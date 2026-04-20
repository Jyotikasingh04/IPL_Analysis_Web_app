import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

# ================= DATA LOAD (KEEPING YOUR METHOD) =================
df = pd.read_csv(
    'https://raw.githubusercontent.com/Jyotikasingh04/IPL_Analysis_Web_app/main/deliveries.csv',
    compression='gzip',
    encoding='latin1'
)


st.title("🏏 Advanced Batsman Analysis")

# ================= SELECT PLAYER =================
players = sorted(df['batter'].dropna().unique())
selected_player = st.selectbox("Select Batsman", players)

player_df = df[df['batter'] == selected_player]

# ================= METRICS =================
runs = player_df['batsman_runs'].sum()
balls = player_df.shape[0]

# ⚠️ Better strike rate (ignore wides)
valid_balls = player_df[player_df['wide_runs'] == 0].shape[0]
sr = (runs / valid_balls) * 100 if valid_balls > 0 else 0

# dismissals (for average)
dismissals = player_df[player_df['player_dismissed'] == selected_player].shape[0]
avg = runs / dismissals if dismissals > 0 else runs

fours = player_df[player_df['batsman_runs'] == 4].shape[0]
sixes = player_df[player_df['batsman_runs'] == 6].shape[0]

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Runs", runs)
col2.metric("Balls", balls)
col3.metric("Strike Rate", round(sr, 2))
col4.metric("Average", round(avg, 2))
col5.metric("4s / 6s", f"{fours}/{sixes}")

# ================= PERFORMANCE TREND =================
st.subheader("📈 Runs per Match")

runs_per_match = player_df.groupby('match_id')['batsman_runs'].sum()

fig = plt.figure(figsize=(10,5))
plt.plot(runs_per_match.index, runs_per_match.values)
plt.xlabel("Match ID")
plt.ylabel("Runs")
plt.title("Runs per Match")

st.pyplot(fig)

# ================= SHOT DISTRIBUTION =================
st.subheader("🎯 Shot Distribution")

dots = player_df[player_df['batsman_runs'] == 0].shape[0]
singles = player_df[player_df['batsman_runs'] == 1].shape[0]
doubles = player_df[player_df['batsman_runs'] == 2].shape[0]
boundaries = player_df[player_df['batsman_runs'] >= 4].shape[0]

fig2 = plt.figure(figsize=(5,5))
plt.pie(
    [dots, singles, doubles, boundaries],
    labels=["Dots", "1s", "2s", "4s/6s"],
    autopct="%1.1f%%"
)

st.pyplot(fig2)

# ================= TOP PERFORMANCES =================
st.subheader("🔥 Top Performances")

top_scores = runs_per_match.sort_values(ascending=False).head(5)
st.dataframe(top_scores.reset_index().rename(columns={'batsman_runs': 'Runs'}))

# ================= INSIGHTS =================
st.subheader("🧠 Insights")

if sr > 150:
    st.success("Highly aggressive batter 🔥")
elif sr > 120:
    st.info("Balanced attacking player ⚖️")
else:
    st.warning("Anchor-type batter 🧱")

if avg > 40:
    st.success("Consistent performer 💎")
elif avg > 25:
    st.info("Moderately consistent")
else:
    st.warning("Inconsistent performances ⚠️")

# ================= COMPARISON =================
st.subheader("⚔️ Compare with Another Batsman")

compare_player = st.selectbox("Select another batsman", players, key="compare")

compare_df = df[df['batter'] == compare_player]

compare_runs = compare_df['batsman_runs'].sum()
compare_balls = compare_df.shape[0]

compare_valid_balls = compare_df[compare_df['wide_runs'] == 0].shape[0]
compare_sr = (compare_runs / compare_valid_balls) * 100 if compare_valid_balls > 0 else 0

compare_dismissals = compare_df[compare_df['player_dismissed'] == compare_player].shape[0]
compare_avg = compare_runs / compare_dismissals if compare_dismissals > 0 else compare_runs

col1, col2 = st.columns(2)

col1.metric(selected_player, f"SR: {round(sr,2)} | Avg: {round(avg,2)}")
col2.metric(compare_player, f"SR: {round(compare_sr,2)} | Avg: {round(compare_avg,2)}")