import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle

# Load and clean data
df = pd.read_csv("data/matches.csv")
df = df[['team1', 'team2', 'toss_winner', 'venue', 'toss_decision', 'winner']]
df.dropna(inplace=True)

# Encode categorical features
le = LabelEncoder()
for col in df.columns:
    df[col] = le.fit_transform(df[col])

X = df.drop('winner', axis=1)
y = df['winner']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model, encoder, scaler
with open("models/cricket_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("models/label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

with open("models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("✅ Model, LabelEncoder, and Scaler saved.")
