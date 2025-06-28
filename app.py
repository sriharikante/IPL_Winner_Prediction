import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- Load all saved objects ---
@st.cache_data
def load_model_objects():
    """Load all model objects and cache them"""
    try:
        model = joblib.load("cricket_model.pkl")
        scaler = joblib.load("scaler.pkl")
        target_encoder = joblib.load("target_encoder.pkl") 
        feature_encoders = joblib.load("feature_encoders.pkl")
        feature_columns = joblib.load("feature_columns.pkl")
        model_info = joblib.load("model_info.pkl")
        
        return model, scaler, target_encoder, feature_encoders, feature_columns, model_info
    except Exception as e:
        st.error(f"Error loading model files: {e}")
        st.error("Make sure you've run the training script first!")
        st.stop()

# Load everything
model, scaler, target_encoder, feature_encoders, feature_columns, model_info = load_model_objects()

# --- Streamlit UI ---
st.title("🏏 IPL Match Winner Predictor")
st.write("**Simple model using: Team1, Team2, Venue, Toss Winner, Toss Decision**")

# Show model info
with st.expander("📊 Model Information"):
    st.write(f"**Accuracy**: {model_info['accuracy']*100:.1f}%")
    st.write(f"**Features**: {', '.join(model_info['feature_columns'])}")
    st.write(f"**Teams**: {len(model_info['feature_classes']['team1'])} teams")
    st.write(f"**Venues**: {len(model_info['feature_classes']['venue'])} venues")

# Get options from saved encoders
teams = sorted(feature_encoders['team1'].classes_)
venues = sorted(feature_encoders['venue'].classes_)
toss_decisions = sorted(feature_encoders['toss_decision'].classes_)

# Create input form
st.subheader("🏏 Match Details")

col1, col2 = st.columns(2)

with col1:
    team1 = st.selectbox("**Team 1**", teams, key="team1")
    venue = st.selectbox("**Venue**", venues, key="venue")
    toss_decision = st.selectbox("**Toss Decision**", toss_decisions, key="toss_decision")

with col2:
    team2 = st.selectbox("**Team 2**", teams, key="team2")
    
    # Toss winner should be one of the selected teams
    toss_options = [team1, team2] if team1 and team2 and team1 != team2 else ["Select teams first"]
    toss_winner = st.selectbox("**Toss Winner**", toss_options, key="toss_winner")

# Prediction
st.subheader("🎯 Prediction")

if st.button("Predict Winner", type="primary", use_container_width=True):
    if team1 == team2:
        st.error("⚠️ Please select different teams!")
    elif toss_winner == "Select teams first":
        st.error("⚠️ Please select teams first!")
    else:
        try:
            # Create input in exact same format as training
            input_data = pd.DataFrame({
                'team1': [team1],
                'team2': [team2],
                'venue': [venue], 
                'toss_winner': [toss_winner],
                'toss_decision': [toss_decision]
            })
            
            # Show input
            with st.expander("📋 Input Data"):
                st.dataframe(input_data, use_container_width=True)
            
            # Encode using SAME encoders from training
            input_encoded = input_data.copy()
            for col in feature_columns:
                input_encoded[col] = feature_encoders[col].transform([input_data[col].iloc[0]])
            
            # Ensure correct order
            input_encoded = input_encoded[feature_columns]
            
            # Scale using SAME scaler from training  
            input_scaled = scaler.transform(input_encoded)
            
            # Make prediction
            prediction = model.predict(input_scaled)
            prediction_proba = model.predict_proba(input_scaled)
            
            # Decode prediction
            predicted_winner = target_encoder.inverse_transform(prediction)[0]
            confidence = np.max(prediction_proba) * 100
            
            # Display results
            st.success(f"🏆 **Predicted Winner: {predicted_winner}**")
            st.info(f"📊 **Confidence: {confidence:.1f}%**")
            
            # Show all team probabilities
            prob_data = []
            for i, team in enumerate(target_encoder.classes_):
                prob_data.append({
                    'Team': team,
                    'Probability': f"{prediction_proba[0][i]*100:.1f}%"
                })
            
            prob_df = pd.DataFrame(prob_data).sort_values('Probability', ascending=False, key=lambda x: x.str.rstrip('%').astype(float))
            
            st.subheader("📈 All Team Probabilities")
            st.dataframe(prob_df, use_container_width=True, hide_index=True)
            
        except ValueError as ve:
            if "not in" in str(ve):
                st.error(f"❌ Invalid input: {str(ve)}")
                st.error("This combination of teams/venue/toss might not exist in training data.")
            else:
                st.error(f"❌ Error: {str(ve)}")
        except Exception as e:
            st.error(f"❌ Prediction error: {str(e)}")

# Consistency test
st.subheader("🧪 Consistency Test")
if st.button("Test Same Input 5 Times"):
    if team1 == team2:
        st.error("⚠️ Please select valid inputs first!")
    elif toss_winner == "Select teams first":
        st.error("⚠️ Please select teams first!")
    else:
        results = []
        confidences = []
        
        for i in range(5):
            # Same prediction process
            input_data = pd.DataFrame({
                'team1': [team1], 'team2': [team2], 'venue': [venue],
                'toss_winner': [toss_winner], 'toss_decision': [toss_decision]
            })
            
            input_encoded = input_data.copy()
            for col in feature_columns:
                input_encoded[col] = feature_encoders[col].transform([input_data[col].iloc[0]])
            
            input_encoded = input_encoded[feature_columns]
            input_scaled = scaler.transform(input_encoded)
            
            prediction = model.predict(input_scaled)
            prediction_proba = model.predict_proba(input_scaled)
            
            predicted_winner = target_encoder.inverse_transform(prediction)[0]
            confidence = np.max(prediction_proba) * 100
            
            results.append(predicted_winner)
            confidences.append(f"{confidence:.1f}%")
        
        # Show results
        test_df = pd.DataFrame({
            'Test': range(1, 6),
            'Predicted Winner': results,
            'Confidence': confidences
        })
        
        st.dataframe(test_df, use_container_width=True, hide_index=True)
        
        if len(set(results)) == 1:
            st.success("✅ **CONSISTENT**: All 5 predictions are identical!")
        else:
            st.error("❌ **INCONSISTENT**: Predictions vary - there's still a bug!")
            
# Show available options
with st.expander("📝 Available Options"):
    st.write("**Teams:**", ", ".join(teams[:10]) + f"... ({len(teams)} total)")
    st.write("**Venues:**", ", ".join(venues[:5]) + f"... ({len(venues)} total)")
    st.write("**Toss Decisions:**", ", ".join(toss_decisions))
