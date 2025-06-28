# ipl_model_training.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import os

# Load the dataset
df = pd.read_csv("/content/matches.csv")

# Drop rows with missing important values
df.dropna(subset=['team1', 'team2', 'toss_winner', 'venue', 'toss_decision', 'winner'], inplace=True)

# Encode categorical columns
le = LabelEncoder()
for col in ['team1', 'team2', 'toss_winner', 'venue', 'toss_decision', 'winner']:
    df[col] = le.fit_transform(df[col])

# Select features and target
X = df[['team1', 'team2', 'toss_winner', 'venue', 'toss_decision']]
y = df['winner']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Save model files
os.makedirs("models", exist_ok=True)
pickle.dump(model, open("models/cricket_model.pkl", "wb"))
pickle.dump(le, open("models/label_encoder.pkl", "wb"))
pickle.dump(scaler, open("models/scaler.pkl", "wb"))

print("Model training and saving complete.")


