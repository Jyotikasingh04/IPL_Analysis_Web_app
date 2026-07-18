import io

import joblib
import pandas as pd
import requests
import streamlit as st

# -----------------------------------------------------------------------------
# Model loading
# -----------------------------------------------------------------------------
# NOTE: model (1).pkl and score_model.pkl are in .gitignore, so they do NOT
# exist in the deployed repo on Streamlit Cloud. Loading them lazily (rather
# than at module import time) and caching with st.cache_resource means this
# module is safe to import from any page without crashing on import.
#
# Both models are now fetched via GitHub Releases (not raw.githubusercontent.com).
# Raw URLs are convenient for small files but are subject to GitHub's stricter,
# unauthenticated rate limits and are not meant for serving binary assets;
# Releases are the supported path for this and is what the score model
# already used.
#
# IMPORTANT: WIN_MODEL_URL below assumes you upload "model (1).pkl" as a
# release asset (same way score_model_new.1.pkl was uploaded). Replace the
# URL with the real release asset URL once it exists.

WIN_MODEL_URL = "https://github.com/Jyotikasingh04/IPL_Analysis_Web_app/releases/download/v1.0/model.pkl"
SCORE_MODEL_URL = "https://github.com/Jyotikasingh04/IPL_Analysis_Web_app/releases/download/v1.0/score_model_new.pkl"

REQUEST_TIMEOUT = 20  # seconds


def load_remote_model(url: str):
    """
    Shared loader: downloads a joblib-dumped sklearn model from a GitHub
    Release URL and loads it into memory. joblib is used instead of raw
    pickle since it is scikit-learn's recommended serialization path for
    estimators (better handling of numpy arrays, more robust across
    environments) - pickle still works, but joblib is the documented choice.
    """
    response = requests.get(url, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return joblib.load(io.BytesIO(response.content))


@st.cache_resource
def get_win_model():
    """Load the trained Logistic Regression win-probability model (cached)."""
    return load_remote_model(WIN_MODEL_URL)


@st.cache_resource
def get_score_model():
    """Load the trained RandomForestRegressor score model (cached)."""
    return load_remote_model(SCORE_MODEL_URL)


# -----------------------------------------------------------------------------
# Win prediction
# -----------------------------------------------------------------------------
WIN_MODEL_FEATURES = [
    'runs_left',
    'balls_left',
    'wickets_left',
    'current_run_rate',
    'required_run_rate',
]


def predict_win_probability(runs_left, balls_left, wickets_left, crr, rrr):
    """
    Run the trained Logistic Regression model and return the batting team's
    win probability as a percentage (0-100), rounded to 2 decimals.

    Column order/names are pinned explicitly via WIN_MODEL_FEATURES to match
    training exactly (see ipl_analysis_web_app.ipynb, cell 59). Relying on
    dict insertion order alone is fragile if this function is ever edited;
    passing `columns=` makes the required order impossible to break silently.
    """
    model = get_win_model()

    input_data = pd.DataFrame(
        [[runs_left, balls_left, wickets_left, crr, rrr]],
        columns=WIN_MODEL_FEATURES
    )

    # classes_ == [0, 1], and in training result=1 means batting_team == winner,
    # so index 1 is the batting team's win probability.
    prob = model.predict_proba(input_data)[0]

    return round(prob[1] * 100, 2)


# -----------------------------------------------------------------------------
# Score prediction (unchanged behaviour, only the loading mechanism changed)
# -----------------------------------------------------------------------------
SCORE_MODEL_FEATURES = ['current_score', 'balls_left', 'wickets', 'run_rate']


def predict_score(current_score, balls_left, wickets, run_rate):
    model = get_score_model()

    input_data = pd.DataFrame(
        [[current_score, balls_left, wickets, run_rate]],
        columns=SCORE_MODEL_FEATURES
    )

    return round(model.predict(input_data)[0])
