"""Apply the trained ranker to a shortlist and return the final ranking.

Architecture: deterministic-first, with the LTR as a local tiebreaker.

The deterministic composite (see redrob/rank/train.py:deterministic_score)
was validated empirically to surface 80 elites in top-100, vs 55 for the
LTR-first approach. The LTR is preserved as a refiner that only reorders
near-tied deterministic buckets.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from .. import config
from .train import (
    FEATURE_COLUMNS,
    build_feature_frame,
    deterministic_score,
    train_ranker,
)


# Local-tiebreak threshold (deterministic-score units). Within this band,
# the LTR refines the ordering; outside it, the deterministic score is the
# primary key.
LTR_REFINE_EPS = 0.15


def _load_model() -> Tuple[Optional[object], List[str]]:
    try:
        return train_ranker(force=False)
    except Exception:
        return None, FEATURE_COLUMNS


def rank_shortlist(
    df: pd.DataFrame,
    use_lgbm: bool = True,
) -> pd.DataFrame:
    """Rank a shortlist dataframe.

    Returns the dataframe with `model_score`, `det_score`, `final_score`,
    and `honeypot_excluded` columns.

    Sort order:
      1) Honeypot hard-excluded candidates pushed to the bottom.
      2) Primary sort by deterministic_score (proven composite).
      3) Within near-tied deterministic buckets (Δ < LTR_REFINE_EPS),
         re-sort by LTR score. Otherwise keep deterministic order.
      4) Final tiebreaker: candidate_id ascending.
    """
    df = build_feature_frame(df)

    # Always compute deterministic score
    det = deterministic_score(df)
    df["det_score"] = det
    df["model_score"] = det  # for backwards-compat callers

    # Optionally compute LTR score
    ltr_score: Optional[np.ndarray] = None
    if use_lgbm:
        model, feats = _load_model()
        if model is not None and "sparse_score" in df.columns:
            # Only run the model if retrieval features are non-zero
            X = df[feats].fillna(0).astype(np.float32)
            ltr_score = np.asarray(model.predict(X), dtype=np.float32)

    # Honeypot hard-exclusion flag
    df["honeypot_excluded"] = df["honeypot_penalty"] >= config.HONEYPOT_HARD_EXCLUDE

    # final_score starts as the deterministic score; honeypot-hard → -1e9
    df["final_score"] = det.astype(np.float32).copy()
    df.loc[df["honeypot_excluded"], "final_score"] = -1e9

    # Sort: hard-excluded at bottom, then deterministic-primary, LTR-tiebreak,
    # candidate_id-asc final tiebreak.
    df = _sort_with_ltr_refine(df, ltr_score)
    return df


def _sort_with_ltr_refine(
    df: pd.DataFrame,
    ltr_score: Optional[np.ndarray],
) -> pd.DataFrame:
    """Stable sort with LTR refinement of near-tied deterministic buckets."""
    n = len(df)
    if n == 0:
        return df.reset_index(drop=True)

    # Bucket key: the integer floor of det_score / LTR_REFINE_EPS.
    # All candidates whose det scores fall in the same bucket are eligible
    # for LTR-based refinement.
    eps = float(LTR_REFINE_EPS)
    if eps <= 0:
        eps = 0.15

    # Build a stable sort key with two parts:
    #   primary:   det_score descending
    #   secondary: (within-bucket rank from LTR) descending
    #   tertiary:  candidate_id ascending
    if ltr_score is None or len(ltr_score) != n:
        # No LTR: just sort by final_score then candidate_id.
        return df.sort_values(
            by=["final_score", "candidate_id"],
            ascending=[False, True],
            kind="mergesort",
        ).reset_index(drop=True)

    # Compute the "LTR rank within bucket". For each candidate, find the
    # set of other candidates within ±eps of its det_score; within that
    # set, the LTR score determines order.
    det = df["det_score"].to_numpy()
    # Sort indices by det descending; ties broken by ltr descending then id asc.
    # We do this with a stable sort over a composite key.
    # composite_key = (det_bucket, -ltr, id)  → ascending sort gives the right order
    # bucket = floor(det / eps). We want highest det first → use -bucket.
    bucket = np.floor(det / eps).astype(np.int64)
    order = np.lexsort((
        df["candidate_id"].to_numpy(),       # tiebreaker 3: candidate_id asc
        -ltr_score,                          # tiebreaker 2: ltr desc
        -bucket,                             # primary: bucket desc (i.e., det desc)
    ))
    return df.iloc[order].reset_index(drop=True)


if __name__ == "__main__":
    from ..data import load_or_build_parquet
    df = load_or_build_parquet().head(200)
    out = rank_shortlist(df)
    print(out[["candidate_id", "current_title", "title_fit_blend", "det_score",
               "model_score", "final_score", "honeypot_penalty"]].head(20))
