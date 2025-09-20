#Load dataset, normalize qualitative and quantitative data, encode quantitative data, compute weights

import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

#Loading csv with pandas
df = pd.read_csv("data/NASA Astronauts 1959-Present.csv", quotechar='"')

#Drop everything but what we need
# Column 0 = Name, Columns 12–15 = SpaceFlights, FlightHours, Spacewalks, SpacewalkHours
df_numeric = df.iloc[:, [0, 12, 13, 14, 15]].copy()
df_numeric.columns = ["Name", "SpaceFlights", "FlightHours", "Spacewalks", "SpacewalkHours"]

#Convert columns to numeric, handle missing values
quant_cols = ["SpaceFlights", "FlightHours", "Spacewalks", "SpacewalkHours"]
for col in quant_cols:
    df_numeric[col] = pd.to_numeric(df_numeric[col], errors='coerce')
df_numeric[quant_cols] = df_numeric[quant_cols].fillna(0)

#Normalize the values
def normalize(value, min_val, max_val, min_target=50, max_target=100):
    if max_val == min_val:
        return max_target
    return min_target + (value - min_val) / (max_val - min_val) * (max_target - min_target)

for col in quant_cols:
    min_val = df_numeric[col].min()
    max_val = df_numeric[col].max()
    df_numeric[col + "_Score"] = df_numeric[col].apply(lambda x: round(normalize(x, min_val, max_val)))

#Weighted criteria
weights = {
    "SpaceFlights_Score": 0.3,
    "FlightHours_Score": 0.3,
    "Spacewalks_Score": 0.2,
    "SpacewalkHours_Score": 0.2
}

df_numeric["OverallScore"] = sum(df_numeric[col] * weight for col, weight in weights.items())
df_numeric["OverallScore"] = round(normalize(df_numeric["OverallScore"],
                                             df_numeric["OverallScore"].min(),
                                             df_numeric["OverallScore"].max()))

#Qualitative data encoding
df["UndergradMajor"] = df["Undergraduate Major"].fillna("").str.split(";")
df["GradMajor"] = df["Graduate Major"].fillna("").str.split(";")
df["AlmaMater"] = df["Alma Mater"].fillna("").str.replace('"', '').str.strip()

mlb_undergrad = MultiLabelBinarizer()
undergrad_encoded = pd.DataFrame(mlb_undergrad.fit_transform(df["UndergradMajor"]),
                                 columns=[f"Undergrad_{c}" for c in mlb_undergrad.classes_],
                                 index=df.index)

mlb_grad = MultiLabelBinarizer()
grad_encoded = pd.DataFrame(mlb_grad.fit_transform(df["GradMajor"]),
                            columns=[f"Grad_{c}" for c in mlb_grad.classes_],
                            index=df.index)

# For Alma Mater, we can do simple one-hot encoding (usually one per astronaut)
alma_encoded = pd.get_dummies(df["AlmaMater"], prefix="Alma", dtype=int)

#Combine numeric and encoded qualitative data
df_final = pd.concat([df_numeric, undergrad_encoded, grad_encoded, alma_encoded], axis=1)

#Convert normalized scores to file
df_final.to_csv("data/astronauts_scores.csv", index=False)
print("Adrian")