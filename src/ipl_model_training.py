# Complete Cricket Match Predictor Training Script
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.preprocessing import LabelEncoder, StandardScaler 
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import joblib
import warnings
warnings.filterwarnings('ignore')

print("🏏 Cricket Match Predictor - Training Script")
print("=" * 50)

# Step 1: Load the dataset 
try:
    data = pd.read_csv("matches.csv")
    print(f"✅ Dataset loaded successfully!")
    print(f"📊 Dataset shape: {data.shape}")
    print(f"📋 Columns: {list(data.columns)}")
except FileNotFoundError:
    print("❌ Error: matches.csv file not found!")
    exit()

# Step 2: Data Pre-Processing & Feature selection
print("\n🔧 Data Preprocessing...")

# Remove unnecessary columns but keep venue
data_processed = data.copy()
columns_to_drop = ["id", "season", "city", "date", "player_of_match", "umpire1", "umpire2"]

# Drop columns that exist
existing_cols_to_drop = [col for col in columns_to_drop if col in data_processed.columns]
data_processed.drop(existing_cols_to_drop, axis=1, inplace=True)

# Handle missing values
print(f"📝 Missing values before cleaning:")
print(data_processed.isnull().sum())

data_processed.dropna(inplace=True)
print(f"✅ Cleaned dataset shape: {data_processed.shape}")

# Select features and target
feature_columns = ['team1', 'team2', 'venue', 'toss_winner', 'toss_decision', 'result']
target_column = 'winner'

# Check if all required columns exist
missing_cols = [col for col in feature_columns + [target_column] if col not in data_processed.columns]
if missing_cols:
    print(f"❌ Missing columns: {missing_cols}")
    print(f"Available columns: {list(data_processed.columns)}")
    exit()

X = data_processed[feature_columns]
y = data_processed[target_column]

print(f"📈 Features selected: {feature_columns}")
print(f"🎯 Target: {target_column}")

# Step 3: Encode categorical features
print("\n🔢 Encoding categorical features...")

# One-hot encode all categorical features
X_encoded = pd.get_dummies(X, columns=feature_columns, drop_first=True)
print(f"✅ Features after encoding: {X_encoded.shape}")
print(f"📊 Total features: {len(X_encoded.columns)}")

# Encode target variable
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
print(f"✅ Target classes: {list(label_encoder.classes_)}")

# Step 4: Data Visualization
print("\n📊 Generating visualizations...")

# Winner distribution
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
data_processed['winner'].value_counts().plot(kind='bar', color='skyblue')
plt.title('Match Winners Distribution')
plt.xlabel('Teams')
plt.ylabel('Number of Wins')
plt.xticks(rotation=45)

# Toss decision distribution
plt.subplot(1, 2, 2)
data_processed['toss_decision'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=['lightcoral', 'lightgreen'])
plt.title('Toss Decision Distribution')
plt.ylabel('')

plt.tight_layout()
plt.savefig('cricket_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Step 5: Split the data
print("\n🔀 Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"✅ Training set: {X_train.shape}")
print(f"✅ Test set: {X_test.shape}")

# Step 6: Scale the features
print("\n⚖️ Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 7: Train the model
print("\n🤖 Training Random Forest model...")
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    n_jobs=-1
)

model.fit(X_train_scaled, y_train)
print("✅ Model training completed!")

# Step 8: Evaluate the model
print("\n📊 Model Evaluation...")
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print(f"🎯 Model Accuracy: {accuracy * 100:.2f}%")
print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Confusion Matrix
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=label_encoder.classes_, 
            yticklabels=label_encoder.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# Feature Importance
feature_importance = pd.DataFrame({
    'feature': X_encoded.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(12, 8))
top_features = feature_importance.head(15)
plt.barh(range(len(top_features)), top_features['importance'])
plt.yticks(range(len(top_features)), top_features['feature'])
plt.xlabel('Feature Importance')
plt.title('Top 15 Most Important Features')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n💾 Saving model and preprocessing objects...")

# Step 9: Save all necessary objects
joblib.dump(model, "cricket_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")
joblib.dump(X_encoded.columns.tolist(), "feature_columns.pkl")

# Save sample data for reference
sample_data = {
    'teams': sorted(data_processed['team1'].unique().tolist()),
    'venues': sorted(data_processed['venue'].unique().tolist()),
    'toss_decisions': sorted(data_processed['toss_decision'].unique().tolist()),
    'result_types': sorted(data_processed['result'].unique().tolist())
}
joblib.dump(sample_data, "sample_data.pkl")

print("✅ Model saved as: cricket_model.pkl")
print("✅ Scaler saved as: scaler.pkl")
print("✅ Label encoder saved as: label_encoder.pkl")
print("✅ Feature columns saved as: feature_columns.pkl")
print("✅ Sample data saved as: sample_data.pkl")

print(f"\n🎉 Training completed successfully!")
print(f"📊 Final Model Accuracy: {accuracy * 100:.2f}%")
print(f"🏏 Total teams: {len(sample_data['teams'])}")
print(f"🏟️ Total venues: {len(sample_data['venues'])}")
print("🚀 Ready for predictions!")
