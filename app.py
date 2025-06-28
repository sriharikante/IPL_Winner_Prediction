
# app.py
import streamlit as st
import pickle
import numpy as np

# Load saved model and encoders
model = pickle.load(open("models/cricket_model.pkl", "rb"))
encoders = pickle.load(open("models/encoders.pkl", "rb"))
scaler = pickle.load(open("models/scaler.pkl", "rb"))

st.title("🏏 IPL Match Winner Predictor")

teams = encoders['team1'].classes_.tolist()
venues = encoders['venue'].classes_.tolist()
toss_decisions = encoders['toss_decision'].classes_.tolist()

t1 = st.selectbox("Select Team 1", teams)
t2 = st.selectbox("Select Team 2", [team for team in teams if team != t1])
ven = st.selectbox("Select Venue", venues)
toss_winner = st.selectbox("Who won the toss?", [t1, t2])
toss_decision = st.selectbox("Toss Decision", toss_decisions)

if st.button("Predict Winner"):
    try:
        input_data = [
            encoders['team1'].transform([t1])[0],
            encoders['team2'].transform([t2])[0],
            encoders['toss_winner'].transform([toss_winner])[0],
            encoders['venue'].transform([ven])[0],
            encoders['toss_decision'].transform([toss_decision])[0]
        ]

        input_scaled = scaler.transform([input_data])
        pred = model.predict(input_scaled)
        winner = encoders['winner'].inverse_transform(pred)

        st.success(f"🏆 Predicted Winner: {winner[0]}")
    except Exception as e:
        st.error("Prediction failed. Ensure all options are valid.")
        st.exception(e)
