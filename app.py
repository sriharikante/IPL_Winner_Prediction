import streamlit as st
import pandas as pd
import pickle

# Load models
model = pickle.load(open("cricket_model.pkl", "rb"))
label_encoder = pickle.load(open("label_encoder.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Define inputs
teams = [
    'Mumbai Indians', 'Chennai Super Kings', 'Royal Challengers Bangalore',
    'Gujarat Titans', 'Rajasthan Royals', 'Kolkata Knight Riders',
    'Sunrisers Hyderabad', 'Lucknow Super Giants', 'Punjab Kings', 'Delhi Capitals'
]

venues = ['Mumbai', 'Chennai', 'Bangalore', 'Kolkata', 'Delhi', 'Ahmedabad']
toss_decisions = ['bat', 'field']

st.set_page_config(page_title="IPL Match Predictor", page_icon="🏏")
st.title("🏏 IPL Match Winner Predictor")

team1 = st.selectbox("Select Team 1", teams)
team2 = st.selectbox("Select Team 2", [t for t in teams if t != team1])
venue = st.selectbox("Select Venue", venues)
toss_winner = st.selectbox("Toss Won By", [team1, team2])
toss_decision = st.selectbox("Toss Decision", toss_decisions)

def predict_winner(team1, team2, toss_winner, venue, toss_decision):
    input_df = pd.DataFrame([[team1, team2, toss_winner, venue, toss_decision]],
                            columns=['team1', 'team2', 'toss_winner', 'venue', 'toss_decision'])

    for col in input_df.columns:
        input_df[col] = label_encoder.transform(input_df[col])

    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)
    return label_encoder.inverse_transform(prediction)[0]

if st.button("Predict Winner"):
    winner = predict_winner(team1, team2, toss_winner, venue, toss_decision)
    st.success(f"🏆 Predicted Winner: {winner}")
