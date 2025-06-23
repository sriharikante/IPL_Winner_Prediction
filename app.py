import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
import os

# --- Load data and model ---
DATA_FILE = "matches.csv"
MODEL_FILE = "cricket_model.pkl"

# Check if required files exist
if not os.path.exists(DATA_FILE) or not os.path.exists(MODEL_FILE):
    st.error("Required file matches.csv or cricket_model.pkl not found.")
    st.stop()

# Load dataset and model
df = pd.read_csv(DATA_FILE)
model = joblib.load(MODEL_FILE)

# Encode categorical columns
le = LabelEncoder()
for col in ['team1', 'team2', 'winner', 'venue']:
    df[col] = le.fit_transform(df[col])

# --- Streamlit UI ---
st.title("🏏 IPL Match Winner Predictor")

teams = sorted(list(df['team1'].unique()))
team_names = le.inverse_transform(teams)

team1 = st.selectbox("Select Team 1", team_names)
team2 = st.selectbox("Select Team 2", team_names)
venue_input = st.selectbox("Select Venue", le.inverse_transform(df['venue'].unique()))

if st.button("Predict Winner"):
    if team1 == team2:
        st.warning("Team 1 and Team 2 must be different.")
    else:
        try:
            t1 = le.transform([team1])[0]
            t2 = le.transform([team2])[0]
            venue = le.transform([venue_input])[0]

            prediction = model.predict([[t1, t2, venue]])
            predicted_winner = le.inverse_transform(prediction)[0]

            st.success(f"🎉 Predicted Winner: {predicted_winner}")
            st.caption("Confidence: 90% (static)")
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
