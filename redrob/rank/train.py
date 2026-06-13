"""Feature assembly, weak labels, and ranker training.

The ranker is a LightGBM LambdaRank model trained on synthetic weak labels
derived from the role blueprint. A deterministic scoring function is also
exposed as a robust fallback that does not depend on the trained model.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from .. import config
from ..blueprint import build_blueprint
from ..data import load_or_build_parquet
from ..graph.coherence import career_coherence_scores, skill_community_features
from ..graph.propagate import personalised_pagerank
from ..features import (
    title_fit_features,
    recruitability_features,
    honeypot_features,
    skill_features,
)


# ---------------------------------------------------------------------------
# Feature assembly
# ---------------------------------------------------------------------------

def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Compute every feature the ranker needs for the given dataframe."""
    bp = build_blueprint()

    # 1) Graph PPR over the full 100k pool (cached inside the graph module)
    ppr = personalised_pagerank()
    df = df.copy()
    df["ppr_score"] = df["candidate_id"].map(lambda c: ppr.get(f"cand::{c}", 0.0))

    # 2) Skill community features
    purity, rarity, degree = skill_community_features(df)
    df["skill_community_purity"] = purity
    df["skill_rarity"] = rarity
    df["graph_degree"] = degree

    # 3) Career coherence
    df["career_coherence"] = career_coherence_scores(df)

    # 4) Title features
    tfeat = title_fit_features(df).reset_index(drop=True)
    df = pd.concat([df.reset_index(drop=True), tfeat], axis=1)

    # 5) Behavioral
    bfeat = recruitability_features(df).reset_index(drop=True)
    df = pd.concat([df, bfeat], axis=1)

    # 6) Honeypot
    hfeat = honeypot_features(df).reset_index(drop=True)
    df = pd.concat([df, hfeat], axis=1)

    # 7) Skill features
    sfeat = skill_features(df).reset_index(drop=True)
    df = pd.concat([df, sfeat], axis=1)

    # 8) Seniority / location / industry features
    df["years_in_ideal_band"] = (
        (df["yoe"] >= config.YOE_IDEAL_LOW) & (df["yoe"] <= config.YOE_IDEAL_HIGH)
    ).astype(np.float32)
    df["yoe_log"] = np.log1p(df["yoe"].clip(lower=0))
    df["preferred_location"] = df["location"].fillna("").apply(
        lambda s: int(any(loc.lower() in (s or "").lower() for loc in config.PREFERRED_LOCATIONS))
    )
    df["country_ok"] = df["country"].fillna("").apply(
        lambda s: int(s.strip() in config.COUNTRY_OK)
    )

    def _as_dict(s):
        if s is None:
            return {}
        if hasattr(s, "tolist"):
            s = s.tolist() if s else {}
        return s if isinstance(s, dict) else {}

    df["willing_to_relocate"] = df["signals"].apply(
        lambda s: int(bool(_as_dict(s).get("willing_to_relocate", False)))
    )
    df["notice_period_days"] = df["signals"].apply(
        lambda s: int(_as_dict(s).get("notice_period_days", 90))
    )
    df["notice_log"] = np.log1p(df["notice_period_days"].clip(lower=0))
    df["work_mode_remote"] = df["signals"].apply(
        lambda s: int(_as_dict(s).get("preferred_work_mode", "") in ("remote", "flexible", "hybrid"))
    )

    # 9) Industry / company features
    def _as_list(v):
        if v is None:
            return []
        if hasattr(v, "tolist"):
            return v.tolist()
        return v

    df["is_services_company"] = df["career_companies"].apply(
        lambda lst: int(any((c or "").strip() in config.IT_SERVICES_PURE_PLAY for c in (_as_list(lst) or [])))
    )
    df["is_product_company"] = df["career_companies"].apply(
        lambda lst: int(any((c or "").strip() in config.PRODUCT_COMPANY_HINTS for c in (_as_list(lst) or [])))
    )
    df["n_industries"] = df["career_industries"].apply(
        lambda lst: len(set(_as_list(lst) or []))
    )

    return df


FEATURE_COLUMNS: List[str] = [
    # retrieval
    "sparse_score", "dense_score", "rrf_score", "structured_hit",
    # graph
    "ppr_score", "skill_community_purity", "skill_rarity", "graph_degree",
    "career_coherence",
    # title
    "title_weight", "title_history_max_weight", "is_target_title",
    "is_noneng_title", "is_data_platform", "is_data_science", "is_generic_swe",
    "n_career_titles", "avg_tenure_months", "title_fit_blend",
    # skills
    "skill_count", "n_core_skills", "n_adj_skills", "n_neg_skills",
    "n_advanced_skills", "n_expert_skills", "must_have_coverage",
    "adjacent_coverage", "ai_signal_strength", "assessment_max",
    "assessment_mean", "endorsement_log_mean", "duration_log_mean",
    "jaccard_core",
    # behavioral
    "recruit_open_to_work", "recruit_response_rate", "recruit_verified",
    "recruit_completeness", "recruit_recency", "recruit_notice_ok",
    "recruit_recruiter_saves", "recruit_interview_completion",
    "recruit_offer_acceptance", "recruit_github", "recruitability",
    # honeypot
    "honeypot_penalty",
    # seniority / location / industry
    "yoe", "years_in_ideal_band", "yoe_log", "preferred_location",
    "country_ok", "willing_to_relocate", "notice_period_days", "notice_log",
    "work_mode_remote", "is_services_company", "is_product_company",
    "n_industries",
]


# ---------------------------------------------------------------------------
# Weak labels (used to train the LambdaRank model)
# ---------------------------------------------------------------------------

def weak_relevance_label(df: pd.DataFrame) -> np.ndarray:
    """A 0..4 relevance grade derived from a transparent composite.

    The grade is intentionally not the same as `deterministic_score` — it
    is a *learning target* that emphasises the role blueprint's
    hard requirements (target title, must-have coverage, low honeypot
    penalty) so the ranker can calibrate.
    """
    title_w = df["title_fit_blend"].astype(float).to_numpy()
    must = df["must_have_coverage"].astype(float).to_numpy()
    ppr = df["ppr_score"].astype(float).to_numpy()
    coh = df["career_coherence"].astype(float).to_numpy()
    rec = df["recruitability"].astype(float).to_numpy()
    hp = df["honeypot_penalty"].astype(float).to_numpy()
    jac = df["jaccard_core"].astype(float).to_numpy()
    yoe = df["yoe"].astype(float).to_numpy()
    in_band = df["years_in_ideal_band"].astype(float).to_numpy()

    score = (
        5.0 * title_w
        + 2.0 * must
        + 1.5 * np.log1p(ppr * 1e5)  # log-scaled, since PPR is small
        + 1.0 * coh
        + 1.0 * rec
        - 3.0 * hp
        + 0.5 * jac
        + 0.5 * np.log1p(yoe) * in_band
        + 0.3 * df["is_product_company"].astype(float).to_numpy()
    )
    # Convert to 5-point Likert
    grade = np.zeros(len(df), dtype=np.int32)
    grade[score > 2.0] = 1
    grade[score > 3.5] = 2
    grade[score > 5.0] = 3
    grade[score > 6.5] = 4
    return grade


# ---------------------------------------------------------------------------
# Deterministic fallback scorer (works without training)
# ---------------------------------------------------------------------------

def deterministic_score(df: pd.DataFrame) -> np.ndarray:
    """A robust, fully-deterministic score combining the same features."""
    s = np.zeros(len(df), dtype=np.float32)

    # Title (heavy weight — JD emphasises this)
    s += 4.5 * df["title_fit_blend"].astype(float).to_numpy()
    s += 1.5 * df["is_target_title"].astype(float).to_numpy()
    s -= 2.0 * df["is_noneng_title"].astype(float).to_numpy()

    # Skills
    s += 2.0 * df["must_have_coverage"].astype(float).to_numpy()
    s += 0.5 * df["adjacent_coverage"].astype(float).to_numpy()
    s += 0.3 * df["jaccard_core"].astype(float).to_numpy()
    s += 0.4 * (df["assessment_max"].fillna(0).astype(float).to_numpy() / 100.0)
    s += 0.4 * (df["assessment_mean"].fillna(0).astype(float).to_numpy() / 100.0)

    # Graph
    s += 1.5 * np.log1p(df["ppr_score"].astype(float).to_numpy() * 1e5)
    s += 0.8 * df["career_coherence"].astype(float).to_numpy()
    s += 0.5 * df["skill_community_purity"].astype(float).to_numpy()
    s += 0.2 * df["skill_rarity"].astype(float).to_numpy()

    # Behavior
    s += 1.2 * df["recruitability"].astype(float).to_numpy()

    # Honeypot penalty (multiplicative, applied later; here we just subtract)
    s -= 4.0 * df["honeypot_penalty"].astype(float).to_numpy()

    # Seniority & location
    s += 0.4 * df["years_in_ideal_band"].astype(float).to_numpy()
    s += 0.3 * df["country_ok"].astype(float).to_numpy()
    s += 0.2 * df["preferred_location"].astype(float).to_numpy()
    s += 0.2 * df["willing_to_relocate"].astype(float).to_numpy()
    s -= 0.4 * (df["notice_period_days"].fillna(90).clip(lower=0).astype(float).to_numpy() / 180.0)
    s += 0.3 * df["is_product_company"].astype(float).to_numpy()
    s -= 0.2 * df["is_services_company"].astype(float).to_numpy() * (1.0 - df["must_have_coverage"].astype(float).to_numpy())

    # Retrieval signals (additive)
    if "rrf_score" in df.columns:
        s += 0.6 * df["rrf_score"].fillna(0).astype(float).to_numpy()
    if "sparse_score" in df.columns:
        s += 0.05 * (df["sparse_score"].fillna(0).astype(float).to_numpy() / 30.0)
    if "dense_score" in df.columns:
        s += 0.6 * df["dense_score"].fillna(0).astype(float).to_numpy()

    return s


# ---------------------------------------------------------------------------
# Train LightGBM ranker
# ---------------------------------------------------------------------------

def train_ranker(force: bool = False) -> Tuple[object, List[str]]:
    """Train a LambdaRank model on the full 100k pool using weak labels.

    Returns (model, feature_columns).
    """
    if not force and config.RANKER_TXT.exists():
        import lightgbm as lgb
        return lgb.Booster(model_file=str(config.RANKER_TXT)), FEATURE_COLUMNS

    df = load_or_build_parquet()
    df = build_feature_frame(df)
    # The retrieval-score features are only set when the ranker is invoked
    # via the run_ranking pipeline. During standalone training we fill them
    # with 0 so the schema matches.
    for col in ("sparse_score", "dense_score", "rrf_score", "structured_hit"):
        if col not in df.columns:
            df[col] = 0.0
    df["label"] = weak_relevance_label(df)

    # LambdaRank requires per-group rows <= 10000. We split the 100k pool
    # into ~20 pseudo-queries (groups of 5000 random candidates) so the
    # ranker can learn a global ordering while respecting the limit.
    n = len(df)
    rng = np.random.default_rng(42)
    perm = rng.permutation(n)
    group_size = 5000
    n_groups = (n + group_size - 1) // group_size
    groups: list = []
    for g in range(n_groups):
        start = g * group_size
        end = min(n, start + group_size)
        groups.append(end - start)
    # Reorder dataframe by the permutation so groups are contiguous
    df = df.iloc[perm].reset_index(drop=True)
    y = df["label"].astype(np.int32).to_numpy()
    X = df[FEATURE_COLUMNS].fillna(0).astype(np.float32)

    import lightgbm as lgb  # type: ignore

    train_data = lgb.Dataset(X, label=y, group=groups)
    model = lgb.train(
        config.RANKER_PARAMS,
        train_data,
        valid_sets=[train_data],
        valid_names=["train"],
    )
    model.save_model(str(config.RANKER_TXT))
    return model, FEATURE_COLUMNS


if __name__ == "__main__":
    import sys, time
    t = time.time()
    model, feats = train_ranker(force=True)
    print(f"trained in {time.time()-t:.1f}s, features={len(feats)}")
    df = load_or_build_parquet().head(5)
    df = build_feature_frame(df)
    X = df[feats].fillna(0).astype(np.float32)
    print("scores:", model.predict(X))
    print("det:", deterministic_score(df))
