import numpy as np 
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler 
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score 
import joblib

print("Starting training process...")

# Load the dataset 
data = pd.read_csv("matches.csv") 
print(f"Dataset loaded: {data.shape}")

# Select only the features we need + target
columns_needed = ['team1', 'team2', 'venue', 'toss_winner', 'toss_decision', 'winner']
data_clean = data[columns_needed].copy()

# Remove rows with missing values
data_clean.dropna(inplace=True)
print(f"After cleaning: {data_clean.shape}")

# Separate features and target
feature_cols = ['team1', 'team2', 'venue', 'toss_winner', 'toss_decision']
X = data_clean[feature_cols].copy()
y = data_clean['winner'].copy()

print("Unique values in each feature:")
for col in feature_cols:
    print(f"{col}: {len(X[col].unique())} unique values")
    print(f"  Sample values: {sorted(X[col].unique())[:5]}")

# Create and save encoders for each feature
encoders = {}
X_encoded = X.copy()

for col in feature_cols:
    encoders[col] = LabelEncoder()
    X_encoded[col] = encoders[col].fit_transform(X[col])
    print(f"Encoded {col}: {encoders[col].classes_[:5]}...")

# Encode target
target_encoder = LabelEncoder()
y_encoded = target_encoder.fit_transform(y)
print(f"Target classes: {target_encoder.classes_}")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f'\nModel Accuracy: {accuracy * 100:.2f}%')

# Feature importance
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
print("\nFeature Importance:")
print(feature_importance)

# Save everything
print("\nSaving model and preprocessing objects...")

joblib.dump(model, "cricket_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(target_encoder, "target_encoder.pkl")
joblib.dump(encoders, "feature_encoders.pkl")
joblib.dump(feature_cols, "feature_columns.pkl")

# Save model info
model_info = {
    'feature_columns': feature_cols,
    'target_classes': target_encoder.classes_.tolist(),
    'feature_classes': {col: encoders[col].classes_.tolist() for col in feature_cols},
    'accuracy': accuracy
}
joblib.dump(model_info, "model_info.pkl")

print("✅ Training completed successfully!")
print("Files created:")
for file in ["cricket_model.pkl", "scaler.pkl", "target_encoder.pkl", "feature_encoders.pkl", "feature_columns.pkl", "model_info.pkl"]:
    print(f"- {file}")

# Test prediction consistency
print("\n🧪 Testing prediction consistency...")
test_input = X_encoded.iloc[0:1]  # First row
test_scaled = scaler.transform(test_input)

predictions = []
for i in range(5):
    pred = model.predict(test_scaled)[0]
    pred_winner = target_encoder.inverse_transform([pred])[0]
    predictions.append(pred_winner)

print(f"Same input 5 times: {predictions}")
if len(set(predictions)) == 1:
    print("✅ Model gives consistent predictions!")
else:
    print("❌ Model predictions are inconsistent!")
