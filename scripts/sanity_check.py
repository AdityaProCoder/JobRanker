"""Sanity check the top-100 submission."""
import pandas as pd

df = pd.read_csv("D:/Project/redrob/submission.csv")
print("Top 100 reasoning preview (rank 1, 50, 100):")
for r in [0, 49, 99]:
    print(f"  rank {r+1}: {df.iloc[r]['reasoning'][:200]}")
print()

non_eng = [
    "marketing manager", "accountant", "sales executive", "graphic designer",
    "content writer", "civil engineer", "mechanical engineer",
    "customer support", "operations manager", "project manager",
    "business analyst", "hr manager",
]
hits_non = 0
for r in df["reasoning"]:
    rl = r.lower()
    if any(k in rl for k in non_eng):
        hits_non += 1
print(f"rows with non-eng keywords in reasoning: {hits_non}/100")

# Score stats
print(f"\nScore stats: min={df['score'].min():.4f}, max={df['score'].max():.4f}")
print(f"All scores strictly decreasing: {(df['score'].diff().dropna() <= 0).all()}")
print(f"Rank 1-100 all unique: {df['rank'].nunique() == 100}")
print(f"All candidate_ids unique: {df['candidate_id'].nunique() == 100}")
print(f"All candidate_ids well-formed: {df['candidate_id'].str.match(r'^CAND_[0-9]{7}$').all()}")
