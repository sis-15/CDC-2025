# Draft fantasy team with top or balanced strategy

import pandas as pd

def draft_fantasy_team(df, team_size=5, strategy="top"):
    if strategy == "top":
        # Just pick top N by OverallScore
        return df.sort_values("OverallScore", ascending=False).head(team_size)
    
    # Instead of just reversing order and using top strategy we can guarantee the bottom
    elif strategy == "bottom":
        return df.nsmallest(team_size, "OverallScore")

    elif strategy == "balanced":
        # Ensure mix of STEM vs non-STEM majors, different colleges, etc.
        team = []
        remaining = df.sort_values("OverallScore", ascending=False).copy()

        # Always pick the top scorer first
        team.append(remaining.iloc[0])
        remaining = remaining.drop(remaining.index[0])

        # Keep track of already chosen majors and colleges
        chosen_undergrad = set(team[0]["UndergradMajorList"])
        chosen_alma = set(team[0]["AlmaMaterList"])

        for _ in range(team_size - 1):
            # Lines 25-32 written by ChatGPT
            # Prefer candidates that maximize diversity
            def diversity_score(row):
                undergrad_overlap = len(set(row["UndergradMajorList"]) & chosen_undergrad)
                alma_overlap = len(set(row["AlmaMaterList"]) & chosen_alma)
                return undergrad_overlap + alma_overlap  # lower is better

            remaining["DiversityScore"] = remaining.apply(diversity_score, axis=1)
            candidate = remaining.sort_values(["DiversityScore", "OverallScore"]).iloc[0]

            # Add candidate to team
            team.append(candidate)
            chosen_undergrad.update(candidate["UndergradMajorList"])
            chosen_alma.update(candidate["AlmaMaterList"])
            remaining = remaining.drop(candidate.name)

        return pd.DataFrame(team)

# Run draft
df = pd.read_csv("data/processed/astronauts_scores.csv")

print("Top strategy draft:")
print(draft_fantasy_team(df, team_size=5, strategy="top")[["Name", "OverallScore"]])

print("Bottom strategy draft:")
print(draft_fantasy_team(df, team_size=5, strategy="bottom")[["Name", "OverallScore"]])

print("\nBalanced strategy draft:")
balanced_team = draft_fantasy_team(df, team_size=5, strategy="balanced")
print(balanced_team[["Name", "OverallScore", "UndergradMajorList", "AlmaMaterList"]])