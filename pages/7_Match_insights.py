import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(layout="wide")

df = pd.read_csv(
    'https://raw.githubusercontent.com/Jyotikasingh04/IPL_Analysis_Web_app/main/deliveries.csv',
    compression='gzip',
    encoding='latin1'
)

st.title("📊 Match Insights")

# ================= AVERAGE SCORE =================
st.subheader("🏏 Average Match Score")

match_scores = df.groupby(['match_id','inning'])['total_runs'].sum().reset_index()
avg_score = match_scores['total_runs'].mean()

st.metric("Average Score (IPL)", round(avg_score, 2))


# ================= POWERPLAY VS DEATH =================
st.subheader("⚡ Powerplay vs Death Overs")

powerplay = df[df['over'] <= 6]['total_runs'].sum()
death = df[df['over'] > 15]['total_runs'].sum()

fig1 = plt.figure(figsize=(6,6))
plt.pie([powerplay, death], labels=['Powerplay','Death'], autopct="%1.1f%%")
plt.title("Run Distribution")

st.pyplot(fig1)


# ================= TEAM SCORING =================
st.subheader("📈 Team Scoring Trends")

team_runs = df.groupby('batting_team')['total_runs'].sum().sort_values(ascending=False)

fig2 = plt.figure(figsize=(12,6))   # 🔥 FIX SIZE
team_runs.plot(kind='bar')

plt.xticks(rotation=45, ha='right')  # 🔥 FIX LABELS
plt.ylabel("Runs")
plt.title("Total Runs by Team")

st.pyplot(fig2)


# ================= CORRELATION HEATMAP =================
st.subheader("🔥 Correlation Analysis")

corr_df = df[['total_runs','ball']].copy()
corr_df['is_boundary'] = (df['batsman_runs'] >= 4).astype(int)
corr_df['is_dot'] = (df['total_runs'] == 0).astype(int)

corr = corr_df.corr()

fig3 = plt.figure(figsize=(6,4))
sns.heatmap(corr, annot=True, cmap='coolwarm')

st.pyplot(fig3)


# ================= KEY INSIGHTS =================
st.subheader("🧠 Key Insights")

st.markdown("""
- Boundary hitting has strong positive impact on total runs  
- Dot balls increase pressure and reduce scoring rate  
- Death overs contribute significantly to total score  
- Teams with aggressive batting dominate IPL scoring trends  
""")