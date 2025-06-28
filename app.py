import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder

# --- Load data and model ---
DATA_FILE = "matches.csv"  
MODEL_FILE = "cricket_model.pkl"
SCALER_FILE = "scaler.pkl"

# Check if required files exist
if not os.path.exists(DATA_FILE) or not os.path.exists(MODEL_FILE) or not os.path.exists(SCALER_FILE):
    st.error("Required files not found. Make sure matches.csv, cricket_model.pkl, and scaler.pkl are present.")
    st.stop()

@st.cache_data
def load_and_prepare_data():
    """Load and prepare data exactly like training"""
    # Load dataset
    df = pd.read_csv(DATA_FILE)
    
    # Apply exact same preprocessing as training
    df_processed = df.copy()
    df_processed = df_processed.iloc[:, :-1]  # Remove last column
    df_processed.dropna(inplace=True)
    df_processed.drop(["id", "season", "city", "date", "player_of_match", "venue", "umpire1", "umpire2"], axis=1, inplace=True)
    
    # Get unique values for dropdowns
    teams = sorted(list(set(df_processed['team1'].unique()) | set(df_processed['team2'].unique())))
    toss_decisions = sorted(df_processed['toss_decision'].unique())
    results = sorted(df_processed['result'].unique())
    
    # Create the feature structure (exactly like training)
    X = df_processed.drop(["winner"], axis=1)
    X_encoded = pd.get_dummies(X, columns=["team1", "team2", "toss_winner", "toss_decision", "result"], drop_first=True)
    
    # Create label encoder for target
    le = LabelEncoder()
    le.fit(df_processed['winner'])
    
    return teams, toss_decisions, results, X_encoded.columns.tolist(), le

# Load data and get options
teams, toss_decisions, results, expected_features, label_encoder = load_and_prepare_data()

# Load model and scaler
model = joblib.load(MODEL_FILE)
scaler = joblib.load(SCALER_FILE)

# --- Streamlit UI ---
st.title("🏏 IPL Match Winner Predictor")

col1, col2 = st.columns(2)

with col1:
    team1 = st.selectbox("Select Team 1", teams)

with col2:
    team2 = st.selectbox("Select Team 2", teams)

# Make toss winner dependent on selected teams
if team1 and team2:
    toss_winner = st.selectbox("Select Toss Winner", [team1, team2])
else:
    toss_winner = st.selectbox("Select Toss Winner", ["Select teams first"], disabled=True)

toss_decision = st.selectbox("Select Toss Decision", toss_decisions)  
result = st.selectbox("Select Result Type", results)

if st.button("Predict Winner"):
    if team1 == team2:
        st.warning("Team 1 and Team 2 must be different.")
    elif not all([team1, team2, toss_winner, toss_decision, result]):
        st.warning("Please fill all fields.")
    else:
        try:
            # Create input dataframe exactly like training
            input_data = pd.DataFrame({
                'team1': [team1],
                'team2': [team2], 
                'toss_winner': [toss_winner],
                'toss_decision': [toss_decision],
                'result': [result]
            })
            
            # Apply same get_dummies transformation as training
            input_encoded = pd.get_dummies(input_data, columns=["team1", "team2", "toss_winner", "toss_decision", "result"], drop_first=True)
            
            # Create a dataframe with all expected features (initialized to 0)
            prediction_input = pd.DataFrame(0, index=[0], columns=expected_features)
            
            # Fill in the values for columns that exist in our input
            for col in input_encoded.columns:
                if col in prediction_input.columns:
                    prediction_input[col] = input_encoded[col].values[0]
            
            # Scale the input
            input_scaled = scaler.transform(prediction_input)
            
            # Make prediction
            prediction = model.predict(input_scaled)
            prediction_proba = model.predict_proba(input_scaled)
            
            # Decode prediction
            predicted_winner = label_encoder.inverse_transform(prediction)[0]
            confidence = np.max(prediction_proba) * 100
            
            st.success(f"🎉 Predicted Winner: **{predicted_winner}**")
            st.info(f"🎯 Confidence: **{confidence:.1f}%**")
            
            # Show feature analysis
            with st.expander("🔍 Prediction Details"):
                st.write(f"**Input Features Used:** {len(expected_features)} features")
                st.write(f"**Teams:** {team1} vs {team2}")
                st.write(f"**Toss:** {toss_winner} won and chose to {toss_decision}")
                st.write(f"**Expected Result Type:** {result}")
                
                # Show which features were active
                active_features = [col for col in input_encoded.columns if input_encoded[col].values[0] == 1]
                if active_features:
                    st.write(f"**Active Features:** {', '.join(active_features)}")
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            st.error("Please check that your model files are compatible.")
            
            # Debug info
            with st.expander("Debug Information"):
                st.write(f"Expected features: {len(expected_features)}")
                st.write(f"Model expects: {model.n_features_in_} features")
                st.write(f"Error details: {str(e)}")

# Show model info
with st.sidebar:
    st.header("📊 Model Information")
    st.write(f"**Teams Available:** {len(teams)}")
    st.write(f"**Toss Decisions:** {len(toss_decisions)}")
    st.write(f"**Result Types:** {len(results)}")
    st.write(f"**Model Features:** {len(expected_features)}")
    
    if st.button("Show All Features"):
        st.write("**All Model Features:**")
        for i, feature in enumerate(expected_features, 1):
            st.write(f"{i}. {feature}")
