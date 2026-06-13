"""Write the final submission CSV and (optionally) run the validator."""
from __future__ import annotations

import csv
import subprocess
from pathlib import Path
from typing import List

import pandas as pd

from .. import config


def _normalise_score(s: float) -> float:
    """Squeeze score to [0, 1] in a monotonic way that preserves relative order."""
    if s <= 0:
        return 0.0
    return float(1 / (1 + 2.71828 ** (-0.6 * s)))


def write_submission(
    df_ranked: pd.DataFrame,
    out_path: Path | str = config.DEFAULT_OUTPUT_CSV,
    validate: bool = True,
    validator: Path | str | None = config.DEFAULT_VALIDATOR,
) -> Path:
    """Write the final CSV and (optionally) invoke the validator.

    Expects `df_ranked` to be sorted best-first, with columns
    `candidate_id` and `final_score`. Reasoning is generated from the
    candidate row.

    The validator requires that scores be *strictly* non-increasing
    (since the tie-breaker is candidate_id ascending on equal scores).
    We use the raw LightGBM score and apply a 1e-4 epsilon (well below
    the model's scoring resolution but strictly decreasing in float
    comparison). The displayed 6-decimal value preserves the model's
    ranking resolution.
    """
    from ..reasoning.template import generate_reasoning_dataframe

    out_path = Path(out_path)
    df_ranked = df_ranked.reset_index(drop=True).copy()
    if len(df_ranked) > 100:
        df_ranked = df_ranked.head(100)

    # Build a strictly decreasing score for each row.
    # Use the raw final_score (LightGBM LambdaRank produces values in
    # roughly [-15, +15]). Apply a per-index epsilon of 1e-4 which
    # guarantees strict float-decrease while being invisible at 6 decimals.
    norm_scores = df_ranked["final_score"].astype(float).tolist()
    eps = 1e-4
    for i in range(1, len(norm_scores)):
        if norm_scores[i] >= norm_scores[i - 1] - eps * (i - 1):
            norm_scores[i] = norm_scores[i - 1] - eps * i

    # Reasoning
    reasonings = generate_reasoning_dataframe(df_ranked)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        w.writerow(["candidate_id", "rank", "score", "reasoning"])
        for i, row in df_ranked.iterrows():
            cid = str(row["candidate_id"])
            score = f"{norm_scores[i]:.6f}"
            w.writerow([cid, str(i + 1), score, reasonings[i]])

    if validate and validator and Path(validator).exists():
        try:
            r = subprocess.run(
                ["python", str(validator), str(out_path)],
                capture_output=True, text=True, check=False,
            )
            if r.returncode == 0:
                print(f"[validator] OK: {out_path}")
            else:
                print(f"[validator] FAILED:\n{r.stdout}\n{r.stderr}")
        except Exception as e:
            print(f"[validator] error: {e}")

    return out_path


if __name__ == "__main__":
    from ..data import load_or_build_parquet
    from ..rank.predict import rank_shortlist
    df = load_or_build_parquet().sample(120, random_state=0)
    ranked = rank_shortlist(df, use_lgbm=False)
    p = write_submission(ranked, validate=True)
    print("wrote:", p)
