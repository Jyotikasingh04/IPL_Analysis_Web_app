import pandas as pd
import pickle

# load models
model = pickle.load(open('model (1).pkl', 'rb'))
score_model = pickle.load(open('score_model.pkl', 'rb'))

# win prediction
def predict_win_probability(runs_left, balls_left, wickets_left, crr, rrr):
    
    input_data = pd.DataFrame({
        'runs_left': [runs_left],
        'balls_left': [balls_left],
        'wickets_left': [wickets_left],
        'current_run_rate': [crr],
        'required_run_rate': [rrr]
    })
    
    prob = model.predict_proba(input_data)[0]
    
    return round(prob[1]*100, 2)


# score prediction
def predict_score(current_score, balls_left, wickets, run_rate):
    
    input_data = pd.DataFrame({
        'current_score': [current_score],
        'balls_left': [balls_left],
        'wickets': [wickets],
        'run_rate': [run_rate]
    })
    
    return round(score_model.predict(input_data)[0])