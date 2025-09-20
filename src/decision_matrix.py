#Load dataset, normalize qualitative and quantitative data, encode quantitative data, compute weights

import pandas as pd

#Loading csv with pandas
df = pd.read_csv("data/NASA Astronauts 1959-Present.csv")

#Drop everything but what we need
# Column 0 = Name, Columns 13–16 = SpaceFlights, FlightHours, Spacewalks, SpacewalkHours
df = df.iloc[:, [0, 13, 14, 15, 16]]
df.columns = ["Name", "SpaceFlights", "FlightHours", "Spacewalks", "SpacewalkHours"]

def normalize(value, min_val, max_val, min_target=50, max_target=100):
    if max_val == min_val:
        return max_target
    return min_target + (value - min_val) / (max_val - min_val) * (max_target - min_target)

#Convert columns to numeric, handle missing values
quant_cols = ["SpaceFlights", "FlightHours", "Spacewalks", "SpacewalkHours"]
for col in quant_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df[quant_cols] = df[quant_cols].fillna(0)

#Normalize the values
for col in quant_cols:
    min_val = df[col].min()
    max_val = df[col].max()
    df[col + "_Score"] = df[col].apply(lambda x: round(normalize(x, min_val, max_val)))

#Weighted criteria
weights = {
    "SpaceFlights_Score": 0.3,
    "FlightHours_Score": 0.3,
    "Spacewalks_Score": 0.2,
    "SpacewalkHours_Score": 0.2
}

df["OverallScore"] = sum(df[col] * weight for col, weight in weights.items())
df["OverallScore"] = round(normalize(df["OverallScore"], df["OverallScore"].min(), df["OverallScore"].max()))

#Convert normalized scores to file
df.to_csv("data/astronauts_scores.csv", index=False)
print("Adrian")
