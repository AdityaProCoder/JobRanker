"""Independent top-100 quality check.

Counts:
- senior/staff/lead/principal engineer/scientist titles
- junior/intern titles (should be 0)
- non-engineering titles (should be 0)
- top-tier product company mentions (more = better)
- score monotonicity check
"""
import sys
from pathlib import Path
import pandas as pd

CSV = Path("D:/Project/redrob/submission.csv")
df = pd.read_csv(CSV)

print(f"Loaded {len(df)} rows")
print()

NON_ENG = ["marketing manager", "hr manager", "accountant", "sales executive",
           "graphic designer", "content writer", "civil engineer",
           "mechanical engineer", "customer support", "operations manager",
           "project manager", "business analyst"]
ENG_KW = ["senior", "staff", "lead", "principal", "engineer", "scientist",
          "architect", "head of", "director"]
JUNIOR_KW = ["junior", "jr.", "intern", "trainee", "associate"]
TOP_COMPANIES = ["google", "meta", "microsoft", "amazon", "apple", "nvidia",
                 "openai", "anthropic", "razorpay", "cred", "phonepe", "paytm",
                 "swiggy", "zomato", "flipkart", "freshworks", "postman",
                 "niramai", "sarvam", "rephrase", "databricks", "snowflake",
                 "linkedin", "uber", "stripe"]

eng_count = 0
junior_count = 0
non_eng_count = 0
for _, row in df.iterrows():
    txt = (row["reasoning"] or "").lower()
    if any(k in txt for k in JUNIOR_KW):
        junior_count += 1
    elif any(k in txt for k in NON_ENG) and not any(k in txt for k in ENG_KW):
        non_eng_count += 1
    elif any(k in txt for k in ENG_KW):
        eng_count += 1

print(f"Title-class distribution:")
print(f"  Senior/Staff/Eng titles: {eng_count}/100")
print(f"  Junior titles:           {junior_count}/100")
print(f"  Non-eng titles:          {non_eng_count}/100")
print()

print("Top-tier product company mentions (in reasoning):")
for c in TOP_COMPANIES:
    n = df["reasoning"].str.lower().str.contains(c).sum()
    if n > 0:
        print(f"  {c:14s}: {n} mentions")
print()

# Score monotonicity
diffs = df["score"].diff().dropna()
print(f"Score strictly decreasing: {(diffs < 0).all()}")
print(f"All ranks unique:          {df['rank'].nunique() == 100}")
print(f"Min score:                 {df['score'].min():.4f}")
print(f"Max score:                 {df['score'].max():.4f}")
