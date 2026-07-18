import json
import io

import requests
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="Model Evaluation", layout="wide")

# =============================================================
# LOAD METRICS  (model_metrics.json is the ONLY source of truth)
# No model is loaded, no data is retrained on this page.
# =============================================================
LOCAL_METRICS_PATH = "model_metrics.json"
REMOTE_METRICS_URL = (
    "https://github.com/Jyotikasingh04/IPL_Analysis_Web_app/"
    "releases/download/v1.0/model_metrics.json"
)
REQUEST_TIMEOUT = 20


@st.cache_data
def load_metrics():
    """Load model_metrics.json from the repo. Falls back to the
    GitHub raw URL if the file isn't found locally (e.g. Streamlit
    Cloud checkout path differences)."""
    try:
        with open(LOCAL_METRICS_PATH, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        pass

    response = requests.get(REMOTE_METRICS_URL, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return json.load(io.StringIO(response.text))


try:
    metrics = load_metrics()
except Exception as e:
    st.error(
        "Could not load `model_metrics.json`. This page only reads "
        "pre-computed metrics exported from the training notebook — "
        "it does not retrain anything. Make sure model_metrics.json "
        "is committed to the repo root."
    )
    st.exception(e)
    st.stop()

win = metrics.get("win_model", {})
score = metrics.get("score_model", {})

if not win or not score:
    st.error("model_metrics.json is missing the 'win_model' or 'score_model' key.")
    st.stop()

# =============================================================
# HEADER
# =============================================================
st.markdown(
    """
    <h1 style='text-align: center;'>Model Evaluation</h1>
    <p style='text-align: center; color: gray;'>
        Evaluation metrics computed directly from the trained models
        in the training notebook — nothing on this page is retrained
        or estimated live.
    </p>
    """,
    unsafe_allow_html=True,
)
st.markdown("---")

# =============================================================
# MODEL SUMMARY CARD
# =============================================================
win_total_rows = win.get("train_rows", 0) + win.get("test_rows", 0)
score_total_rows = score.get("train_rows", 0) + score.get("test_rows", 0)


def _split_label(total_rows, test_rows):
    if not total_rows:
        return "N/A"
    test_pct = round(test_rows / total_rows * 100)
    return f"{100 - test_pct}/{test_pct} Split"


win_split_label = _split_label(win_total_rows, win.get("test_rows", 0))
score_split_label = _split_label(score_total_rows, score.get("test_rows", 0))

sklearn_version = metrics.get("sklearn_version")

with st.container(border=True):
    st.markdown("### Model Summary")
    s1, s2, s3, s4 = st.columns(4)

    with s1:
        st.markdown("**Win Predictor**")
        st.markdown(f"Algorithm: `{win.get('algorithm', 'N/A')}`")

    with s2:
        st.markdown("**Score Predictor**")
        st.markdown(f"Algorithm: `{score.get('algorithm', 'N/A')}`")

    with s3:
        st.markdown("**Training Strategy**")
        st.markdown(f"Win: `{win_split_label}`  \nScore: `{score_split_label}`")

    with s4:
        st.markdown("**Framework**")
        st.markdown(f"`scikit-learn{f' {sklearn_version}' if sklearn_version else ''}`")

st.markdown("---")

# =============================================================
# 1. MODEL OVERVIEW
# =============================================================
st.subheader("Model Overview")

with st.container():
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Win Predictor**")
        st.metric("Algorithm", win.get("algorithm", "N/A"))

    with c2:
        st.markdown("**Score Predictor**")
        st.metric("Algorithm", score.get("algorithm", "N/A"))

    st.markdown("")

    o1, o2, o3, o4 = st.columns(4)

    with o1:
        st.metric("Win Model Train Samples", f"{win.get('train_rows', 0):,}")
        st.metric("Score Model Train Samples", f"{score.get('train_rows', 0):,}")

    with o2:
        st.metric("Win Model Test Samples", f"{win.get('test_rows', 0):,}")
        st.metric("Score Model Test Samples", f"{score.get('test_rows', 0):,}")

    with o3:
        win_total = win.get("train_rows", 0) + win.get("test_rows", 0)
        win_split = (
            f"{win.get('train_rows', 0) / win_total * 100:.0f}/"
            f"{win.get('test_rows', 0) / win_total * 100:.0f}"
            if win_total else "N/A"
        )
        st.metric("Win Model Train/Test Split", win_split)

        score_total = score.get("train_rows", 0) + score.get("test_rows", 0)
        score_split = (
            f"{score.get('train_rows', 0) / score_total * 100:.0f}/"
            f"{score.get('test_rows', 0) / score_total * 100:.0f}"
            if score_total else "N/A"
        )
        st.metric("Score Model Train/Test Split", score_split)

    with o4:
        win_rs = win.get("hyperparameters", {}).get("random_state", "N/A")
        score_rs = score.get("hyperparameters", {}).get("random_state", "N/A")
        st.metric("Win Model Random State", win_rs)
        st.metric("Score Model Random State", score_rs)

st.markdown("---")

# =============================================================
# 2. CLASSIFICATION METRICS (Win Predictor)
# =============================================================
st.subheader("Classification Metrics — Win Predictor")

cm1, cm2, cm3, cm4, cm5 = st.columns(5)

def _clamped(value):
    """Progress bars require a value in [0, 1]."""
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


with cm1:
    acc = win.get("accuracy", 0)
    st.metric("Accuracy", f"{acc:.4f}")
    st.progress(_clamped(acc))
    st.caption("Share of test-set predictions (win/loss) that matched the actual outcome.")

with cm2:
    prec = win.get("precision", 0)
    st.metric("Precision", f"{prec:.4f}")
    st.progress(_clamped(prec))
    st.caption("Of the matches predicted as a win, the fraction that were actually won.")

with cm3:
    rec = win.get("recall", 0)
    st.metric("Recall", f"{rec:.4f}")
    st.progress(_clamped(rec))
    st.caption("Of the matches actually won, the fraction the model correctly identified.")

with cm4:
    f1 = win.get("f1_score", 0)
    st.metric("F1 Score", f"{f1:.4f}")
    st.progress(_clamped(f1))
    st.caption("Harmonic mean of precision and recall — balances both in one number.")

with cm5:
    roc = win.get("roc_auc", 0)
    st.metric("ROC-AUC", f"{roc:.4f}")
    st.progress(_clamped(roc))
    st.caption("Model's ability to rank a random win above a random loss, across all thresholds.")

if "confusion_matrix" in win:
    with st.expander("Confusion Matrix (Win Predictor)"):
        cm = win["confusion_matrix"]
        fig_cm = go.Figure(
            data=go.Heatmap(
                z=cm,
                x=["Predicted: Loss", "Predicted: Win"],
                y=["Actual: Loss", "Actual: Win"],
                colorscale="Blues",
                text=cm,
                texttemplate="%{text}",
                showscale=False,
            )
        )
        fig_cm.update_layout(height=350, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_cm, use_container_width=True)

st.markdown("---")

# =============================================================
# 3. REGRESSION METRICS (Score Predictor)
# =============================================================
st.subheader("Regression Metrics — Score Predictor")

rm1, rm2, rm3 = st.columns(3)

with rm1:
    st.metric("MAE", f"{score.get('mae', 0):.2f} runs")
    st.caption("Average absolute difference between predicted and actual final score.")

with rm2:
    st.metric("RMSE", f"{score.get('rmse', 0):.2f} runs")
    st.caption("Similar to MAE but penalizes large errors more heavily — sensitive to outliers.")

with rm3:
    r2 = score.get("r2_score", 0)
    st.metric("R² Score", f"{r2:.4f}")
    st.progress(_clamped(r2))
    st.caption("Proportion of variance in final score explained by the model. 1.0 is a perfect fit.")

st.markdown("---")

# =============================================================
# 4. FEATURE IMPORTANCE / COEFFICIENTS
# =============================================================
st.subheader("Feature Importance & Coefficients")

fi_col, coef_col = st.columns(2)

with fi_col:
    st.markdown("**Feature Importance — Random Forest (Score Predictor)**")
    feature_names_score = score.get("feature_names", [])
    importances = score.get("feature_importances", [])

    if feature_names_score and importances:
        pairs = sorted(zip(feature_names_score, importances), key=lambda x: x[1])
        names_sorted, values_sorted = zip(*pairs)

        fig_fi = go.Figure(
            go.Bar(
                x=values_sorted,
                y=names_sorted,
                orientation="h",
                marker_color="#1f77b4",
            )
        )
        fig_fi.update_layout(
            xaxis_title="Importance",
            yaxis_title="Feature",
            height=350,
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig_fi, use_container_width=True)
    else:
        st.info("No feature importance data available in model_metrics.json.")

with coef_col:
    st.markdown("**Feature Coefficients — Logistic Regression (Win Predictor)**")
    st.caption(
        "Logistic Regression has no `feature_importances_`. These are raw "
        "model coefficients, not importance scores — sign and magnitude "
        "reflect direction and strength of each feature's linear effect "
        "on the log-odds of winning."
    )

    coefficients = win.get("coefficients")
    feature_names_win = win.get("feature_names", [])

    if coefficients and feature_names_win:
        pairs = sorted(zip(feature_names_win, coefficients), key=lambda x: x[1])
        names_sorted, values_sorted = zip(*pairs)

        fig_coef = go.Figure(
            go.Bar(
                x=values_sorted,
                y=names_sorted,
                orientation="h",
                marker_color="#ff7f0e",
            )
        )
        fig_coef.update_layout(
            xaxis_title="Coefficient",
            yaxis_title="Feature",
            height=350,
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig_coef, use_container_width=True)
    else:
        st.info(
            "`coefficients` not found in model_metrics.json. Add "
            "`model.coef_[0].tolist()` to the win-model evaluation cell "
            "in the notebook to enable this chart."
        )

st.markdown("---")

# =============================================================
# 5. HYPERPARAMETERS
# =============================================================
st.subheader("Hyperparameters")

hp1, hp2 = st.columns(2)

with hp1:
    with st.expander("Win Predictor — Logistic Regression Parameters"):
        st.json(win.get("hyperparameters", {}))

with hp2:
    with st.expander("Score Predictor — Random Forest Parameters"):
        st.json(score.get("hyperparameters", {}))

st.markdown("---")

# =============================================================
# 6. MODEL NOTES
# =============================================================
st.subheader("Model Notes")

with st.container():
    st.markdown(
        """
- **Logistic Regression** is used for **Win Prediction**.
- **Random Forest Regressor** is used for **Score Prediction**.
- All metrics on this page are generated from the training notebook
  (`ipl_analysis_web_app.ipynb`) and exported to `model_metrics.json` —
  none are computed live in this app.
- The **Win Predictor** and **Score Predictor** pages in this dashboard
  use these same trained models to generate predictions.
        """
    )

st.markdown("---")
st.caption(
    "Metrics reflect a random 80/20 split at the row level (not grouped "
    "by match_id). Values may be modestly optimistic versus a strict "
    "match-level split, since deliveries from the same match can appear "
    "in both train and test sets."
)
