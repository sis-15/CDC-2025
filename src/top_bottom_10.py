import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/processed/astronauts_scores.csv")

# Sort by OverallScore
top_10 = df.sort_values("OverallScore", ascending=False).head(10)
bottom_10 = df.sort_values("OverallScore", ascending=True).head(10)

fig, axes = plt.subplots(2, 1, figsize=(12, 8))
fig.suptitle("Top and Bottom 10 Astronauts by Overall Score", fontsize=16)

# Top 10 table
axes[0].axis("off")
axes[0].table(cellText=top_10[["Name", "OverallScore"]].values,
              colLabels=["Name", "OverallScore"],
              cellLoc="center",
              loc="center")
axes[0].set_title("Top 10", fontsize=14)

# Bottom 10 table
axes[1].axis("off")
axes[1].table(cellText=bottom_10[["Name", "OverallScore"]].values,
              colLabels=["Name", "OverallScore"],
              cellLoc="center",
              loc="center")
axes[1].set_title("Bottom 10", fontsize=14)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("data/analysis/top_bottom_10.png", dpi=300)
plt.show()