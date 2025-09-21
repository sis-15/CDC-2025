import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
import ast
import re
import difflib
from collections import Counter

# Visualize flight hours vs individual scores, color-coded by degree

# Load processed astronaut scores
df = pd.read_csv("data/processed/astronauts_scores.csv")

# Check if the columns exist
required_columns = ["FlightHours_Score", "OverallScore", "UndergradMajorList"]
for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Required column '{col}' not found in dataframe. Found columns: {list(df.columns)}")

# Lines 23-41 written by ChatGPT
# For plotting, we can assign a "primary degree" to each astronaut (first listed undergrad major)
# Robust extractor: handles real lists, stringified lists, semicolon-separated strings, comma-separated strings.
def extract_primary_degree(cell):
    # if it's already a list type
    if isinstance(cell, list):
        return cell[0].strip() if cell else "Unknown"
    if pd.isna(cell) or str(cell).strip() == "":
        return "Unknown"
    s = str(cell).strip()
    # try to parse stringified list like "['Physics']"
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, list) and parsed:
            return str(parsed[0]).strip()
    except Exception:
        pass
    # fallback: split on semicolon (preferred) or comma
    parts = re.split(r"\s*;\s*|\s*,\s*", s)
    return parts[0].strip() if parts and parts[0] != "" else s

df["PrimaryDegree"] = df["UndergradMajorList"].apply(extract_primary_degree)

# normalize for display (but keep original PrimaryDegree too)
df["PrimaryDegree"] = df["PrimaryDegree"].astype(str).str.strip()

#Grouping each major into categories

with open("data/raw/degree_categories.json", "r") as f:
    degree_categories = json.load(f)

# Lines 54-68 written by ChatGPT
# Normalization helper that we apply to both the CSV degrees and JSON degrees
def normalize_degree_text(s):
    if s is None:
        return ""
    s = str(s)
    s = s.strip()
    # replace & with 'and'
    s = s.replace("&", " and ")
    # remove quotes and brackets
    s = re.sub(r"[\[\]\"']", " ", s)
    # remove punctuation except spaces and ampersands already handled
    s = re.sub(r"[^0-9a-zA-Z\s]", " ", s)
    # collapse whitespace and lowercase
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s

# Build a normalized flat lookup from normalized major -> category
degree_to_category = {}
for category, majors in degree_categories.items():
    for major in majors:
        norm = normalize_degree_text(major)
        if norm:  # avoid empty keys
            degree_to_category[norm] = category

# Lines 79-99 written by ChatGPT
# Map a degree string into a category using multiple strategies
def map_degree_to_category(degree):
    if not degree or degree.lower() in ("nan", "none"):
        return "Other/Unknown"
    norm = normalize_degree_text(degree)
    # Exact normalized match
    if norm in degree_to_category:
        return degree_to_category[norm]
    # Fuzzy match against known normalized majors
    candidates = difflib.get_close_matches(norm, degree_to_category.keys(), n=1, cutoff=0.72)
    if candidates:
        return degree_to_category[candidates[0]]
    # Token-substring heuristic: if any token in norm appears inside a known major or vice versa
    norm_tokens = set(norm.split())
    for key in degree_to_category:
        key_tokens = set(key.split())
        # overlap of tokens
        if norm_tokens & key_tokens:
            return degree_to_category[key]
    # give up
    return "Other/Unknown"

# Apply mapping and add a debug summary
df["DegreeCategory"] = df["PrimaryDegree"].apply(map_degree_to_category)

# Debugging output to verify mapping quality
total = len(df)
mapped = (df["DegreeCategory"] != "Other/Unknown").sum()
unknown = total - mapped
# print(f"Degree mapping: total={total}, mapped={mapped}, unknown={unknown}")

# Show the top unknown PrimaryDegree values for manual inspection
unknown_samples = df[df["DegreeCategory"] == "Other/Unknown"]["PrimaryDegree"].value_counts().head(30)
if not unknown_samples.empty:
    print("\nTop unknown PrimaryDegree values (for debugging):")
    print(unknown_samples.to_string())

# Create a color palette by category
categories = sorted(df["DegreeCategory"].unique())
palette = sns.color_palette("Set2", len(categories))  # Or any palette you like
color_map = dict(zip(categories, palette))

# Create figure with constrained layout
fig, ax = plt.subplots(figsize=(16, 10), constrained_layout=False)
fig.tight_layout(rect=[0, 0, 0.78, 1])

# Scatter each degree (add small jitter scaled to score ranges)
x_jitter_scale = 0.5
y_jitter_scale = 0.4

for cat in categories:
    subset = df[df["DegreeCategory"] == cat]
    if subset.empty:
        continue
    x_vals = subset["FlightHours_Score"].astype(float).values
    y_vals = subset["OverallScore"].astype(float).values
    plt.scatter(
        x_vals + np.random.uniform(-x_jitter_scale, x_jitter_scale, size=len(subset)),  # jitter X
        y_vals + np.random.uniform(-y_jitter_scale, y_jitter_scale, size=len(subset)),  # jitter Y
        label=cat,
        alpha=0.75,
        s=25,
        color=color_map[cat]
    )

# Axis labels, title, legend
ax.set_xlabel("Flight Hours Score", fontsize=12)
ax.set_ylabel("Individual Score (OverallScore)", fontsize=14)
ax.set_title("Flight Hours vs Individual Score by Degree Category", fontsize=14)
plt.subplots_adjust(top = 0.92, right = 0.78, left = 0.08, bottom = 0.1)
ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., title="Degree Category", fontsize=14, ncol=1)

plt.savefig("data/analysis/flight_hours_vs_score.png", dpi=300, bbox_inches="tight")
plt.show()

# Debug: print a few mapped examples for visual verification
# print("\nExamples (PrimaryDegree -> DegreeCategory):")
# print(df[["PrimaryDegree", "DegreeCategory"]].drop_duplicates().head(50).to_string(index=False))