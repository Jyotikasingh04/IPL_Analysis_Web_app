import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

df = pd.read_csv(
    'https://raw.githubusercontent.com/Jyotikasingh04/IPL_Analysis_Web_app/main/deliveries.csv',
    compression='gzip',
    encoding='latin1'
)

# FIX TEAM NAMES
df['batting_team'] = df['batting_team'].replace({
    'Delhi Daredevils': 'Delhi Capitals',
    'Kings XI Punjab': 'Punjab Kings',
    'Royal Challengers Bengaluru': 'Royal Challengers Bangalore'
})

st.title("Team Analysis")

# dropdown
teams = sorted(df['batting_team'].unique())
selected_team = st.selectbox("Select Team", teams)

team_df = df[df['batting_team'] == selected_team]

# metrics
total_runs = team_df['total_runs'].sum()
matches = team_df['match_id'].nunique()
avg_score = total_runs / matches if matches > 0 else 0

col1, col2, col3 = st.columns(3)

col1.metric("Total Runs", total_runs)
col2.metric("Matches Played", matches)
col3.metric("Avg Score", round(avg_score, 2))

# Runs per match graph
st.subheader("Runs Per Match")

runs_per_match = team_df.groupby('match_id')['total_runs'].sum()

fig = plt.figure(figsize=(10,5))
plt.plot(runs_per_match.values)
plt.xlabel("Matches")
plt.ylabel("Runs")
plt.title("Performance Trend")

st.pyplot(fig)

# Insight
st.subheader("Insight")

if avg_score > 160:
    st.success("High scoring team ")
elif avg_score > 140:
    st.info("Balanced team ")
else:
    st.warning("Struggles in scoring ")
