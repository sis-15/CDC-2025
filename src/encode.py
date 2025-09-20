# Load dataset, normalize quantitative data, encode qualitative data, compute weighted overall score

import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

# Load CSV
df = pd.read_csv("data/raw/master.csv", quotechar='"')

# Numeric columns
df_numeric = df[["Full Name", "Space Flights", "Space Flight Hours", "Spacewalks", "Spacewalk Hours", "Achievement Count"]].copy()
df_numeric.columns = ["Name", "SpaceFlights", "FlightHours", "Spacewalks", "SpacewalkHours", "Achievements"]

# Convert to numeric and fill missing
quant_cols = ["SpaceFlights", "FlightHours", "Spacewalks", "SpacewalkHours", "Achievements"]
for col in quant_cols:
    df_numeric[col] = pd.to_numeric(df_numeric[col], errors='coerce').fillna(0)

# Normalize function
def normalize(value, min_val, max_val, min_target=50, max_target=100):
    if max_val == min_val:
        return max_target
    return min_target + (value - min_val) / (max_val - min_val) * (max_target - min_target)

# Compute individual scores
for col in quant_cols:
    min_val = df_numeric[col].min()
    max_val = df_numeric[col].max()
    df_numeric[col + "_Score"] = df_numeric[col].apply(lambda x: round(normalize(x, min_val, max_val)))

# Weighted overall score (including achievements)
weights = {
    "SpaceFlights_Score": 0.25,
    "FlightHours_Score": 0.25,
    "Spacewalks_Score": 0.2,
    "SpacewalkHours_Score": 0.2,
    "Achievements_Score": 0.1
}
df_numeric["OverallScore"] = sum(df_numeric[col] * weight for col, weight in weights.items())
df_numeric["OverallScore"] = df_numeric["OverallScore"].apply(lambda x: round(normalize(x, df_numeric["OverallScore"].min(), df_numeric["OverallScore"].max())))

# Qualitative columns
# Handle lists separated by semicolons
df["UndergradMajorList"] = df["Undergrad Major"].fillna("").str.split(";").apply(lambda lst: [s.strip() for s in lst if s])
df["GradMajorList"] = df["Graduate Major"].fillna("").str.split(";").apply(lambda lst: [s.strip() for s in lst if s])
df["AlmaMaterList"] = df["Alma Mater"].fillna("").str.split(";").apply(lambda lst: [s.strip() for s in lst if s])

# One-hot encoding for majors
mlb_undergrad = MultiLabelBinarizer()
undergrad_encoded = pd.DataFrame(mlb_undergrad.fit_transform(df["UndergradMajorList"]),
                                 columns=[f"Undergrad_{c}" for c in mlb_undergrad.classes_],
                                 index=df.index)

mlb_grad = MultiLabelBinarizer()
grad_encoded = pd.DataFrame(mlb_grad.fit_transform(df["GradMajorList"]),
                            columns=[f"Grad_{c}" for c in mlb_grad.classes_],
                            index=df.index)

# One-hot encoding for Alma Mater
mlb_alma = MultiLabelBinarizer()
alma_encoded = pd.DataFrame(mlb_alma.fit_transform(df["AlmaMaterList"]),
                            columns=[f"Alma_{c}" for c in mlb_alma.classes_],
                            index=df.index)

# Combine all
df_final = pd.concat([df_numeric, undergrad_encoded, grad_encoded, alma_encoded,
                      df[["UndergradMajorList", "GradMajorList", "AlmaMaterList"]]], axis=1)

# Save to CSV
df_final.to_csv("data/processed/astronauts_scores.csv", index=False)
print("Encoding complete.")