import streamlit as st
import pandas as pd
import joblib

# Load saved files
model = joblib.load("models/cricket_model.pkl")
scaler = joblib.load("models/scaler.pkl")
le = joblib.load("models/label_encoder.pkl")

# Load dataset for dummy structure
raw_data = pd.read_csv("data/matches.csv")
raw_data.dropna(inplace=True)
raw_data = raw_data.drop(["id", "Season", "city", "date", "player_of_match", "venue", "umpire1", "umpire2", "winner"], axis=1)
data = pd.get_dummies(raw_data, drop_first=True)

# App title
st.title("🏏 IPL Match Winner Predictor")

# Dropdowns
teams = sorted({col.replace("team1_", "") for col in data.columns if "team1_" in col})
team1 = st.selectbox("Team 1", teams)
team2 = st.selectbox("Team 2", teams)
toss_winner = st.selectbox("Toss Winner", [team1, team2])
toss_decision = st.selectbox("Toss Decision", ["field", "bat"])
match_result = st.selectbox("Match Result Type", ["normal", "tie", "no result"])

# Predict button
if st.button("Predict Winner"):
    input_dict = {col: 0 for col in data.columns}
    input_dict[f"team1_{team1}"] = 1
    input_dict[f"team2_{team2}"] = 1
    input_dict[f"toss_winner_{toss_winner}"] = 1
    input_dict[f"toss_decision_{toss_decision}"] = 1
    input_dict[f"result_{match_result}"] = 1

    input_df = pd.DataFrame([input_dict])
    scaled_input = scaler.transform(input_df)
    pred = model.predict(scaled_input)
    winner = le.inverse_transform(pred)[0]

    st.success(f"🎉 Predicted Winner: {winner}")
