# app.py
import streamlit as st
import pickle
import numpy as np

# Load saved model and encoders
model = pickle.load(open("models/cricket_model.pkl", "rb"))
label_encoder = pickle.load(open("models/label_encoder.pkl", "rb"))
scaler = pickle.load(open("models/scaler.pkl", "rb"))

st.title("🏏 IPL Match Winner Predictor")

teams = ['Mumbai Indians', 'Chennai Super Kings', 'Royal Challengers Bangalore', 'Gujarat Titans',
         'Rajasthan Royals', 'Kolkata Knight Riders', 'Sunrisers Hyderabad', 'Lucknow Super Giants',
         'Punjab Kings', 'Delhi Capitals']

venue = ['Mumbai', 'Chennai', 'Bangalore', 'Kolkata', 'Delhi', 'Ahmedabad']

t1 = st.selectbox("Select Team 1", teams)
t2 = st.selectbox("Select Team 2", [team for team in teams if team != t1])
ven = st.selectbox("Select Venue", venue)
toss_winner = st.selectbox("Who won the toss?", [t1, t2])
toss_decision = st.selectbox("Toss Decision", ['bat', 'field'])

if st.button("Predict Winner"):
    try:
        input_data = [t1, t2, toss_winner, ven, toss_decision]
        input_encoded = label_encoder.transform(input_data)
        input_scaled = scaler.transform([input_encoded])
        pred = model.predict(input_scaled)
        winner = label_encoder.inverse_transform(pred)
        st.success(f"🏆 Predicted Winner: {winner[0]}")
    except Exception as e:
        st.error("Prediction failed. Ensure all options are valid.")
        st.exception(e)
