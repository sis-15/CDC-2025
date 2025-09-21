import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from draft import draft_fantasy_team

plt.style.use('bmh')
sns.set_theme(font="DejaVu Sans") 
# ['Solarize_Light2', '_classic_test_patch', '_mpl-gallery', '_mpl-gallery-nogrid', 'bmh', 'classic', 'dark_background', 'fast', 'fivethirtyeight', 'ggplot', 'grayscale', 'petroff10', 'seaborn-v0_8', 'seaborn-v0_8-bright', 'seaborn-v0_8-colorblind', 'seaborn-v0_8-dark', 'seaborn-v0_8-dark-palette', 'seaborn-v0_8-darkgrid', 'seaborn-v0_8-deep', 'seaborn-v0_8-muted', 'seaborn-v0_8-notebook', 'seaborn-v0_8-paper', 'seaborn-v0_8-pastel', 'seaborn-v0_8-poster', 'seaborn-v0_8-talk', 'seaborn-v0_8-ticks', 'seaborn-v0_8-white', 'seaborn-v0_8-whitegrid', 'tableau-colorblind10']

df = pd.read_csv("data/processed/astronauts_scores.csv")

team_size = 5

criteria_columns = ["SpaceFlights_Score", "FlightHours_Score", "Spacewalks_Score",
                    "SpacewalkHours_Score", "Achievements_Score", "OverallScore"]

top_team = draft_fantasy_team(df, team_size=team_size, strategy="top")

# Evil meta: try to get the worst team by sorting ascending and using top script
bottom_team = draft_fantasy_team(df.sort_values("OverallScore", ascending=True),
                                 team_size=team_size, strategy="top")

fig, axes = plt.subplots(2, 1, figsize=(14, 8))
fig.suptitle(f"Top and Bottom {team_size}-Member Teams with Individual Attributes", fontsize=16)


axes[0].axis("off")
axes[0].table(cellText=top_team[["Name"] + criteria_columns].values,
              colLabels=["Name"] + criteria_columns,
              cellLoc="center",
              loc="center")
axes[0].set_title("Top Team", fontsize=14)

axes[1].axis("off")
axes[1].table(cellText=bottom_team[["Name"] + criteria_columns].values,
              colLabels=["Name"] + criteria_columns,
              cellLoc="center",
              loc="center")
axes[1].set_title("Bottom Team", fontsize=14)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("data/analysis/top_bottom_teams.png", dpi=300)
plt.show()