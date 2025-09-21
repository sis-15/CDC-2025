import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression

# Visualize flight hours vs individual scores, color-coded by degree

# Load processed astronaut scores
df = pd.read_csv("data/processed/astronauts_scores.csv")

# Check if the columns exist
required_columns = ["FlightHours_Score", "OverallScore", "UndergradMajorList"]
for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Required column '{col}' not found in dataframe")

# For plotting, we can assign a "primary degree" to each astronaut (first listed undergrad major)
df["PrimaryDegree"] = df["UndergradMajorList"].apply(lambda lst: lst.split(";")[0].strip() if pd.notna(lst) and lst != "" else "Unknown")

# Generate a color palette for each degree dynamically
unique_degrees = df["PrimaryDegree"].unique()
palette = sns.color_palette("hls", len(unique_degrees))
degree_colors = dict(zip(unique_degrees, palette))

# Create figure with constrained layout
fig, ax = plt.subplots(figsize=(16, 10), constrained_layout=True)

# Scatter each degree
for degree, color in degree_colors.items():
    subset = df[df["PrimaryDegree"] == degree]
    jitter_strength = 0.5  # How close the dots are
    subset_x = subset["FlightHours_Score"] + np.random.uniform(-jitter_strength, jitter_strength, size=len(subset))
    subset_y = subset["OverallScore"] + np.random.uniform(-jitter_strength, jitter_strength, size=len(subset))

    # Code without jitter
    # ax.scatter(subset["FlightHours_Score"], subset["OverallScore"],
    #            label=degree, color=color, s=50, alpha=0.7)
    ax.scatter(subset_x, subset_y, label=degree, color=color, s=40, alpha=0.7)
    
# Trend line and standard deviation stuff
# X = df["FlightHours_Score"].values.reshape(-1, 1)
# y = df["OverallScore"].values
# reg = LinearRegression().fit(X, y)
# y_pred = reg.predict(X)
# residuals = y - y_pred
# std = np.std(residuals)
# plt.fill_between(df["FlightHours_Score"], y_pred - std, y_pred + std, color='gray', alpha=0.2, label='±1 SD')
# plt.plot(df["FlightHours_Score"], y_pred, color="black", linestyle="--", label="Trendline")

min_val = df["FlightHours_Score"].min()
max_val = df["FlightHours_Score"].max()
num_bins = 6  # adjust for granularity - 6 looks pretty good
bins = np.linspace(min_val - 1, max_val + 1, num_bins)  # slight buffer to include all points
df["FlightBin"] = pd.cut(df["FlightHours_Score"], bins, include_lowest=True)

# Compute mean and std for each bin
agg = df.groupby("FlightBin").agg(
    mean_score=("OverallScore", "mean"),
    std_score=("OverallScore", "std")
).reset_index()

# Compute bin centers
bin_centers = [interval.mid for interval in agg["FlightBin"]]

# --- Plot trendline ---
plt.plot(bin_centers, agg["mean_score"], color="blue", marker="o", label="Mean Score")
plt.fill_between(bin_centers,
                 agg["mean_score"] - agg["std_score"],
                 agg["mean_score"] + agg["std_score"],
                 color="blue", alpha=0.2, label="±1 SD")

# Axis labels and title
ax.set_xlabel("Flight Hours Score", fontsize=14)
ax.set_ylabel("Individual Score (OverallScore)", fontsize=14)
ax.set_title("Flight Hours vs Individual Score by Degree", fontsize=16)

# Legend outside plot
ax.legend(bbox_to_anchor=(1.05, 1), loc='best', borderaxespad=0., title="Primary Degree", fontsize=4, ncol=2)

# Save plot
plt.savefig("data/analysis/flight_hours_vs_score.png", dpi=300)

plt.show()