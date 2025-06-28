import streamlit as st
import pandas as pd
import pickle

# Load models
model = pickle.load(open("models/cricket_model.pkl", "rb"))
label_encoder = pickle.load(open("models/label_encoder.pkl", "rb"))
scaler = pickle.load(open("models/scaler.pkl", "rb"))

# Define inputs
teams = ['MI', 'CSK', 'RCB', 'GT', 'RR', 'KKR', 'SRH', 'LSG', 'PBKS', 'DC']
venues = ['Mumbai', 'Chennai', 'Bangalore', 'Kolkata', 'Delhi', 'Ahmedabad']

st.set_page_config(page_title="IPL Match Predictor", page_icon="🏏")
st.title("🏏 IPL Match Winner Predictor")

team1 = st.selectbox("Select Team 1", teams)
team2 = st.selectbox("Select Team 2", [t for t in teams if t != team1])
venue = st.selectbox("Select Venue", venues)
toss_winner = st.selectbox("Toss Winner", [team1, team2])

def predict_winner(team1, team2, toss_winner, venue):
    input_df = pd.DataFrame([[team1, team2, toss_winner, venue]],
                            columns=['team1', 'team2', 'toss_winner', 'venue'])

    # Encode inputs
    for col in input_df.columns:
        input_df[col] = label_encoder.transform(input_df[col])

    # Scale inputs
    input_scaled = scaler.transform(input_df)

    # Predict
    pred = model.predict(input_scaled)
    return label_encoder.inverse_transform(pred)[0]

if st.button("Predict Winner"):
    winner = predict_winner(team1, team2, toss_winner, venue)
    st.success(f"🏆 Predicted Winner: {winner}")
