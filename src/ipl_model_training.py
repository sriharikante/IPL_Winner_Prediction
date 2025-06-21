# ipl_model_training.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import squarify
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import joblib

# Load the dataset
data = pd.read_csv("matches.csv")

# Preprocess the dataset
data = data.iloc[:, :-1]  # Remove last column (if unnecessary)
data.dropna(inplace=True)
data.drop(["id", "season", "city", "date", "player_of_match", "venue", "umpire1", "umpire2"], axis=1, inplace=True)

# Encode features and target
X = data.drop(["winner"], axis=1)
y = data["winner"]
X = pd.get_dummies(X, columns=["team1", "team2", "toss_winner", "toss_decision", "result"], drop_first=True)

le = LabelEncoder()
y = le.fit_transform(y)

# Save label encoder
joblib.dump(le, "label_encoder.pkl")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save model and scaler
joblib.dump(model, "cricket_model.pkl")
joblib.dump(sc, "scaler.pkl")
