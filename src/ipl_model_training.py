import numpy as np 
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler 
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score 
import joblib
import pickle

print("Starting training process...")

# Load the dataset 
data = pd.read_csv("matches.csv") 
print(f"Dataset loaded: {data.shape}")

# Data preprocessing - EXACT same steps every time
data_clean = data.copy()
data_clean = data_clean.iloc[:, :-1]  # Remove last column
data_clean.dropna(inplace=True) 
data_clean.drop(["id", "season", "city", "date", "player_of_match", "venue", "umpire1", "umpire2"], axis=1, inplace=True)

print(f"After cleaning: {data_clean.shape}")
print("Columns:", data_clean.columns.tolist())

# Separate features and target
X = data_clean.drop(["winner"], axis=1) 
y = data_clean["winner"]

print("Unique values in each column:")
for col in X.columns:
    print(f"{col}: {sorted(X[col].unique())}")

# Create and save all encoders BEFORE any transformations
encoders = {}
for col in X.columns:
    if X[col].dtype == 'object':
        encoders[col] = LabelEncoder()
        X[col] = encoders[col].fit_transform(X[col])
        print(f"Encoded {col}: {len(encoders[col].classes_)} unique values")

# Target encoder
target_encoder = LabelEncoder()
y_encoded = target_encoder.fit_transform(y)
print(f"Target classes: {target_encoder.classes_}")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
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
print(f'Model Accuracy: {accuracy * 100:.2f}%')

# Save everything we need for predictions
print("\nSaving model and preprocessing objects...")

# Save the trained model
joblib.dump(model, "cricket_model.pkl")
print("✓ Model saved")

# Save the scaler
joblib.dump(scaler, "scaler.pkl")
print("✓ Scaler saved")

# Save target encoder
joblib.dump(target_encoder, "target_encoder.pkl")
print("✓ Target encoder saved")

# Save all feature encoders
joblib.dump(encoders, "feature_encoders.pkl")
print("✓ Feature encoders saved")

# Save feature columns order
feature_columns = X.columns.tolist()
joblib.dump(feature_columns, "feature_columns.pkl")
print("✓ Feature columns saved")

# Save a sample of processed data for reference
sample_data = {
    'feature_columns': feature_columns,
    'target_classes': target_encoder.classes_.tolist(),
    'feature_encoders_classes': {col: enc.classes_.tolist() for col, enc in encoders.items()}
}
with open('model_info.pkl', 'wb') as f:
    pickle.dump(sample_data, f)
print("✓ Model info saved")

# Test the saved model to make sure it works
print("\nTesting saved model...")
loaded_model = joblib.load("cricket_model.pkl")
loaded_scaler = joblib.load("scaler.pkl")
test_pred = loaded_model.predict(loaded_scaler.transform(X_test[:1]))
print(f"Test prediction: {target_encoder.inverse_transform(test_pred)[0]}")

print("\n✅ Training completed successfully!")
print("Files created:")
print("- cricket_model.pkl")
print("- scaler.pkl") 
print("- target_encoder.pkl")
print("- feature_encoders.pkl")
print("- feature_columns.pkl")
print("- model_info.pkl")
