import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor
import joblib

model = joblib.load("models/astronaut_score_model.pkl")

# Load encoding columns from training
with open("models/encoding_columns.pkl", "rb") as f:
    encoding_cols = pickle.load(f)

undergrad_columns = encoding_cols["undergrad"]
grad_columns = encoding_cols["grad"]
alma_columns = encoding_cols["alma"]

missions = pd.read_csv("data/raw/mission.csv", quotechar='"')

# Process each mission
for idx, mission in missions.iterrows():
    print(f"\nMission: {mission['Mission Name & Program']}")

    member_names = mission["Members"].replace('"', '').split(",")
    astronauts = pd.read_csv("data/processed/astronauts_scores.csv")

    # Select only the astronauts in this mission
    mission_astronauts = astronauts[astronauts["Name"].isin([n.strip() for n in member_names])].copy()
    if mission_astronauts.empty:
        print("Warning: No matching astronauts found for this mission.")
        continue

    # Reindex encoded columns to match training
    # Fill missing columns with zeros if any astronaut didn't have a feature seen in training
    for col_group in [undergrad_columns, grad_columns, alma_columns]:
        for col in col_group:
            if col not in mission_astronauts:
                mission_astronauts[col] = 0
    mission_astronauts = mission_astronauts[model.feature_names_in_]  # enforce exact column order

    # Predict scores
    X = mission_astronauts
    predicted_scores = model.predict(X)

    # Compute average predicted score for the mission
    avg_score = predicted_scores.mean()
    print(f"Predicted average mission score: {avg_score:.2f}")
    print(f"Mission success (real world): {mission['Success T/F']}")
    for name, score in zip(member_names, predicted_scores):
        print(f"{name}: {score:.2f}")