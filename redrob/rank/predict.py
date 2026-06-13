"""Apply the trained ranker to a shortlist and return the final ranking."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .. import config
from .train import (
    FEATURE_COLUMNS,
    build_feature_frame,
    deterministic_score,
    train_ranker,
)


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

    Returns the dataframe with two new columns: `model_score` and `final_score`.
    """
    df = build_feature_frame(df)

    if use_lgbm:
        model, feats = _load_model()
        if model is not None:
            X = df[feats].fillna(0).astype(np.float32)
            df["model_score"] = model.predict(X)
        else:
            df["model_score"] = deterministic_score(df)
    else:
        df["model_score"] = deterministic_score(df)

    # Apply honeypot hard exclusion as a hard floor
    df["honeypot_excluded"] = df["honeypot_penalty"] >= config.HONEYPOT_HARD_EXCLUDE

    # Final score = model score, with honeypot penalty pushed to -inf for hard excludes
    df["final_score"] = df["model_score"].astype(np.float32)
    df.loc[df["honeypot_excluded"], "final_score"] = -1e9

    # Deterministic tiebreaker
    df = df.sort_values(
        by=["final_score", "candidate_id"],
        ascending=[False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    return df


if __name__ == "__main__":
    from ..data import load_or_build_parquet
    df = load_or_build_parquet().head(200)
    out = rank_shortlist(df)
    print(out[["candidate_id", "current_title", "title_fit_blend", "model_score", "final_score", "honeypot_penalty"]].head(20))
