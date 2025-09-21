import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

df = pd.read_csv("data/processed/astronauts_scores.csv")
drop_cols = ["Name", "UndergradMajorList", "GradMajorList", "AlmaMaterList", "OverallScore"]

# Feature matrix
X = df.drop(columns=[col for col in drop_cols if col in df.columns])

# Target variable
y = df["OverallScore"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

score = model.score(X_test, y_test)
print(f"Model R^2 score: {score:.3f}")

joblib.dump(model, "models/astronaut_score_model.pkl")
print("Trained model saved as 'models/astronaut_score_model.pkl'")

importances = pd.Series(model.feature_importances_, index=X.columns)
importances = importances.sort_values(ascending=False)
print("\nTop 10 most important features:")
print(importances.head(10))