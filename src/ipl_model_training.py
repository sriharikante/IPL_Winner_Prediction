# Step 1: Import required libraries 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.preprocessing import LabelEncoder, StandardScaler 
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import confusion_matrix, accuracy_score 

# Step 2: Load the dataset 
data = pd.read_csv("matches.csv") 
data.head() 

# Step 3: Data Pre-Processing & Feature selection 
data = data.iloc[:, :-1]  # Remove the last column which seems unnecessary 
data.dropna(inplace=True) 
data.drop(["id", "Season", "city", "date", "player_of_match", "venue", "umpire1", "umpire2"], axis=1, inplace=True) 

X = data.drop(["winner"], axis=1) 
y = data["winner"] 
X = pd.get_dummies(X, columns=["team1", "team2", "toss_winner", "toss_decision", "result"], drop_first=True) 

label_encode = LabelEncoder() 
y = label_encode.fit_transform(y) 

# Step 4: Data Visualization 
plt.figure(figsize=(10, 6)) 
data['winner'].value_counts().plot(kind='bar') 
plt.title('Number of Wins by Team') 
plt.xlabel('Teams') 
plt.ylabel('Number of Wins') 
plt.show() 

numeric_data = data.select_dtypes(include=[np.number]) 
plt.figure(figsize=(12, 10)) 
sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm') 
plt.title('Heat Map of Feature Correlations') 
plt.show() 

plt.figure(figsize=(10, 6)) 
data['team1'].value_counts().plot(kind='bar', alpha=0.5, color='blue', label='Team 1') 
data['team2'].value_counts().plot(kind='bar', alpha=0.5, color='red', label='Team 2') 
plt.title('Number of Matches per Team') 
plt.xlabel('Teams') 
plt.ylabel('Number of Matches') 
plt.legend() 
plt.show() 

plt.figure(figsize=(8, 8)) 
data['toss_decision'].value_counts().plot(kind='pie', autopct='%1.1f%%') 
plt.title('Toss Decision Distribution') 
plt.ylabel('') 
plt.show() 

import squarify 
plt.figure(figsize=(12, 8)) 
result_counts = data['result'].value_counts() 
squarify.plot(sizes=result_counts, label=result_counts.index, alpha=0.8) 
plt.title('Treemap of Match Results') 
plt.axis('off') 
plt.show() 

# Step 5: Splitting and Training the data 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0) 

# Step 6: Scaling the data 
sc = StandardScaler() 
X_train = sc.fit_transform(X_train) 
X_test = sc.transform(X_test) 

# Step 7: Load and Train the model 
model = RandomForestClassifier(n_estimators=100, random_state=42) 
model.fit(X_train, y_train) 

# Step 8: Evaluate the model 
y_pred = model.predict(X_test) 
cm = confusion_matrix(y_test, y_pred) 
plt.figure(figsize=(8, 6)) 
sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', cbar=False) 
plt.title('Confusion Matrix') 
plt.xlabel('Predicted Labels') 
plt.ylabel('True Labels') 
plt.show() 

accuracy = accuracy_score(y_pred, y_test) 
print(f'Accuracy of the model: {accuracy * 100:.2f}%')
