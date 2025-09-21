import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# Analyze gender differences in astronaut experience and mission scores

# Load processed astronaut scores
df_scores = pd.read_csv("data/processed/astronauts_scores.csv")

# Load gender info from master.csv
df_master = pd.read_csv("data/raw/master.csv", quotechar='"')
df_scores["Gender"] = df_master.iloc[:, 3]  # column 4 is Gender

# Quick summary of gender counts
print("Gender distribution:")
print(df_scores["Gender"].value_counts(), "\n")

# Compare average OverallScore by gender
avg_scores = df_scores.groupby("Gender")["OverallScore"].mean()
print("Average OverallScore by gender:")
print(avg_scores, "\n")

# Optional: visualize distribution
sns.boxplot(data=df_scores, x="Gender", y="OverallScore")
plt.title("Distribution of OverallScore by Gender")
plt.show()

# Regression to estimate conditional effect of gender
# Using SpaceFlights, FlightHours, Spacewalks, SpacewalkHours, Achievements as controls
control_cols = ["SpaceFlights_Score", "FlightHours_Score", "Spacewalks_Score", "SpacewalkHours_Score", "Achievements_Score"]
X = df_scores[control_cols].copy()
X = sm.add_constant(X)
# Encode gender as binary (F=1, M=0)
y_gender = df_scores["Gender"].apply(lambda x: 1 if x.upper() == "F" else 0)

# Logistic regression to see if high OverallScore predicts gender representation
logit_model = sm.Logit(y_gender, X)
result = logit_model.fit(disp=False)
print("Gender vs experience regression summary:")
print(result.summary())