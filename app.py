from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__, template_folder='templates')

# Load model, scaler, encoder
model = joblib.load("models/cricket_model.pkl")
scaler = joblib.load("models/scaler.pkl")
le = joblib.load("models/label_encoder.pkl")

# Load structure from original data
data = pd.read_csv("data/matches.csv")
data.dropna(inplace=True)
data.drop(["id", "Season", "city", "date", "player_of_match", "venue", "umpire1", "umpire2", "winner"], axis=1, inplace=True)
base_columns = pd.get_dummies(data, drop_first=True).columns

# Teams list
teams = sorted({col.replace("team1_", "") for col in base_columns if "team1_" in col})

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    team1 = request.form.get("team1")
    team2 = request.form.get("team2")
    toss_winner_choice = request.form.get("toss_winner")
    toss_decision = request.form.get("toss_decision")

    if toss_winner_choice == "team1":
        toss_winner = team1
    elif toss_winner_choice == "team2":
        toss_winner = team2
    else:
        return render_template("index.html", error="Toss winner not selected properly")

    result = "normal"  # Default assumption

    # Create dummy input
    input_data = {col: 0 for col in base_columns}
    input_data[f"team1_{team1}"] = 1
    input_data[f"team2_{team2}"] = 1
    input_data[f"toss_winner_{toss_winner}"] = 1
    input_data[f"toss_decision_{toss_decision}"] = 1
    input_data[f"result_{result}"] = 1

    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])
    scaled_input = scaler.transform(input_df)

    prediction = model.predict(scaled_input)
    winner = le.inverse_transform(prediction)[0]

    return render_template("index.html", prediction=winner)

if __name__ == "__main__":
    app.run(debug=True)
