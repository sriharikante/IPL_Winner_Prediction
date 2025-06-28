import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder

# Configure page
st.set_page_config(
    page_title="🏏 IPL Match Predictor",
    page_icon="🏏",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 2rem;
    }
    .team-vs {
        text-align: center;
        font-size: 2rem;
        color: #ff6b35;
        font-weight: bold;
        margin: 1rem 0;
    }
    .prediction-box {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .confidence-box {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# --- Load required files ---
@st.cache_resource
def load_model_files():
    """Load all model files with error handling"""
    required_files = {
        "model": "cricket_model.pkl",
        "scaler": "scaler.pkl", 
        "label_encoder": "label_encoder.pkl",
        "feature_columns": "feature_columns.pkl",
        "sample_data": "sample_data.pkl"
    }
    
    # Check if files exist
    missing_files = [file for file in required_files.values() if not os.path.exists(file)]
    if missing_files:
        st.error(f"❌ Missing files: {', '.join(missing_files)}")
        st.error("Please run the training script first to generate all required files.")
        st.stop()
    
    # Load files
    try:
        loaded_files = {}
        for key, filename in required_files.items():
            loaded_files[key] = joblib.load(filename)
        return loaded_files
    except Exception as e:
        st.error(f"❌ Error loading model files: {str(e)}")
        st.stop()

# Load all files
files = load_model_files()
model = files["model"]
scaler = files["scaler"]
label_encoder = files["label_encoder"]
feature_columns = files["feature_columns"]
sample_data = files["sample_data"]

# --- Streamlit UI ---
st.markdown('<h1 class="main-header">🏏 IPL Match Winner Predictor</h1>', unsafe_allow_html=True)

# Create columns for team selection
col1, col2, col3 = st.columns([1, 0.3, 1])

with col1:
    st.subheader("Team 1")
    team1 = st.selectbox("Select Team 1", sample_data['teams'], key="team1")

with col2:
    st.markdown('<div class="team-vs">VS</div>', unsafe_allow_html=True)

with col3:
    st.subheader("Team 2")
    team2 = st.selectbox("Select Team 2", sample_data['teams'], key="team2")

# Additional parameters
st.subheader("Match Details")

col4, col5 = st.columns(2)
with col4:
    venue = st.selectbox("🏟️ Venue", sample_data['venues'])
    toss_decision = st.selectbox("🪙 Toss Decision", sample_data['toss_decisions'])

with col5:
    # Toss winner should be one of the selected teams
    if team1 and team2:
        toss_winner = st.selectbox("🏆 Toss Winner", [team1, team2])
    else:
        toss_winner = st.selectbox("🏆 Toss Winner", ["Select teams first"])
    
    result_type = st.selectbox("📊 Result Type", sample_data['result_types'])

# Prediction button
if st.button("🎯 Predict Winner", type="primary", use_container_width=True):
    if team1 == team2:
        st.warning("⚠️ Please select different teams!")
    elif not all([team1, team2, venue, toss_winner, toss_decision, result_type]):
        st.warning("⚠️ Please fill all fields!")
    else:
        try:
            with st.spinner("🔮 Analyzing match conditions..."):
                # Create input dataframe matching training format
                input_data = pd.DataFrame({
                    'team1': [team1],
                    'team2': [team2],
                    'venue': [venue],
                    'toss_winner': [toss_winner],
                    'toss_decision': [toss_decision],
                    'result': [result_type]
                })
                
                # Apply same preprocessing as training
                input_encoded = pd.get_dummies(
                    input_data, 
                    columns=['team1', 'team2', 'venue', 'toss_winner', 'toss_decision', 'result'], 
                    drop_first=True
                )
                
                # Align columns with training data
                for col in feature_columns:
                    if col not in input_encoded.columns:
                        input_encoded[col] = 0
                
                # Reorder columns to match training
                input_encoded = input_encoded.reindex(columns=feature_columns, fill_value=0)
                
                # Scale the input
                input_scaled = scaler.transform(input_encoded)
                
                # Make prediction
                prediction = model.predict(input_scaled)
                prediction_proba = model.predict_proba(input_scaled)
                
                # Get winner name
                predicted_winner = label_encoder.inverse_transform(prediction)[0]
                confidence = np.max(prediction_proba) * 100
                
                # Display results
                st.markdown(
                    f'<div class="prediction-box">'
                    f'<h2>🏆 Predicted Winner</h2>'
                    f'<h1>{predicted_winner}</h1>'
                    f'</div>', 
                    unsafe_allow_html=True
                )
                
                col_conf1, col_conf2 = st.columns(2)
                with col_conf1:
                    st.markdown(
                        f'<div class="confidence-box">'
                        f'<h4>🎯 Confidence</h4>'
                        f'<h3>{confidence:.1f}%</h3>'
                        f'</div>', 
                        unsafe_allow_html=True
                    )
                
                with col_conf2:
                    st.markdown(
                        f'<div class="confidence-box">'
                        f'<h4>🏟️ Venue Advantage</h4>'
                        f'<p>{venue}</p>'
                        f'</div>', 
                        unsafe_allow_html=True
                    )
                
                # Show prediction probabilities for all teams
                with st.expander("📊 Detailed Probabilities"):
                    prob_df = pd.DataFrame({
                        'Team': label_encoder.classes_,
                        'Win Probability': prediction_proba[0] * 100
                    }).sort_values('Win Probability', ascending=False)
                    
                    st.dataframe(prob_df, use_container_width=True)
                
        except Exception as e:
            st.error(f"❌ Prediction failed: {str(e)}")
            st.error("Please check if the model was trained properly.")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>🏏 IPL Match Predictor | Built with Machine Learning</p>
        <p>Accuracy based on historical match data and team performance</p>
    </div>
    """, 
    unsafe_allow_html=True
)

# Sidebar with information
with st.sidebar:
    st.header("ℹ️ About")
    st.write("""
    This app predicts IPL match winners using machine learning.
    
    **Features considered:**
    - Team strength and form
    - Venue conditions
    - Toss advantage
    - Historical performance
    
    **Model:** Random Forest Classifier
    """)
    
    if st.button("🔄 Refresh Model"):
        st.cache_resource.clear()
        st.rerun()
