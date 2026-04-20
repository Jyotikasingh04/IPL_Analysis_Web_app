import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

# ================= DATA LOAD (AS YOU SAID - GZIP) =================
df = pd.read_csv(
    'https://raw.githubusercontent.com/Jyotikasingh04/IPL_Analysis_Web_app/main/deliveries.csv',
    compression='gzip',
    encoding='latin1'
)


st.title("Bowler Analysis")

# ================= SELECT BOWLER =================
bowlers = sorted(df['bowler'].dropna().unique())
selected_bowler = st.selectbox("Select Bowler", bowlers)

bowler_df = df[df['bowler'] == selected_bowler]

# ================= METRICS =================
runs_conceded = bowler_df['total_runs'].sum()
balls = bowler_df.shape[0]
overs = balls / 6

# ✅ FIXED wicket logic (exclude run-outs)
wickets = bowler_df[
    (bowler_df['is_wicket'] == 1) & 
    (bowler_df['dismissal_kind'] != 'run out')
].shape[0]

economy = runs_conceded / overs if overs > 0 else 0
avg = runs_conceded / wickets if wickets > 0 else 0


col1, col2, col3, col4 = st.columns(4)

col1.metric("Overs", round(overs, 1))
col2.metric("Runs", runs_conceded)
col3.metric("Wickets", wickets)
col4.metric("Economy", round(economy, 2))


# ================= PERFORMANCE TREND =================
st.subheader("Match-wise Performance")

match_perf = bowler_df.groupby('match_id').agg({
    'total_runs': 'sum',
    'dismissal_kind': lambda x: ((x != 'run out') & (x.notna())).sum()
}).reset_index().rename(columns={'dismissal_kind': 'wickets'})

fig = plt.figure(figsize=(10,5))
plt.plot(match_perf['match_id'], match_perf['wickets'], label="Wickets")
plt.plot(match_perf['match_id'], match_perf['total_runs'], label="Runs Conceded")
plt.legend()
plt.xlabel("Match ID")
plt.title("Performance Trend")

st.pyplot(fig)

# ================= RUN DISTRIBUTION =================
st.subheader("Ball Outcome Distribution")

dots = bowler_df[bowler_df['batsman_runs'] == 0].shape[0]
singles = bowler_df[bowler_df['batsman_runs'] == 1].shape[0]
boundaries = bowler_df[bowler_df['batsman_runs'] >= 4].shape[0]

fig2 = plt.figure(figsize=(5,5))
plt.pie(
    [dots, singles, boundaries],
    labels=["Dot Balls", "Singles", "Boundaries"],
    autopct="%1.1f%%"
)

st.pyplot(fig2)

# ================= TOP SPELLS =================
st.subheader("Best Bowling Spells")

top_spells = match_perf.sort_values(
    by=['wickets', 'total_runs'],
    ascending=[False, True]
).head(5)

st.dataframe(top_spells)

# ================= INSIGHTS =================
st.subheader("Performance Insights")

if economy < 6:
    st.success("Elite economical bowler ")
elif economy < 8:
    st.info("Decent control ")
else:
    st.warning("Expensive spells detected ")

# ================= COMPARISON =================
st.subheader("⚔️ Compare with Another Bowler")

compare_bowler = st.selectbox("Select another bowler", bowlers, key="compare")

compare_df = df[df['bowler'] == compare_bowler]

compare_runs = compare_df['total_runs'].sum()
compare_balls = compare_df.shape[0]
compare_overs = compare_balls / 6

compare_wickets = compare_df[
    (compare_df['is_wicket'] == 1) & 
    (compare_df['dismissal_kind'] != 'run out')
].shape[0]

compare_economy = compare_runs / compare_overs if compare_overs > 0 else 0

col1, col2 = st.columns(2)

col1.metric(selected_bowler, f"Econ: {round(economy,2)} | Wkts: {wickets}")
col2.metric(compare_bowler, f"Econ: {round(compare_economy,2)} | Wkts: {compare_wickets}")
