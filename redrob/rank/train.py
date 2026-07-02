"""Feature assembly, weak labels, and ranker training.

The ranker is a LightGBM LambdaRank model trained on synthetic weak labels
derived from the role blueprint. A deterministic scoring function is also
exposed as a robust fallback that does not depend on the trained model.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from .. import config
from ..blueprint import build_blueprint
from ..data import load_or_build_parquet
from ..graph.coherence import career_coherence_scores, career_evidence_score, skill_community_features
from ..graph.propagate import personalised_pagerank, ppr_axes
from ..features import (
    title_fit_features,
    company_tier_features,
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

    # 1b) Multi-axis PPR: 5 specialised axes for better discrimination
    ppr_ax = ppr_axes()
    for axis_name, axis_scores in ppr_ax.items():
        df[f"ppr_{axis_name}"] = df["candidate_id"].map(
            lambda c, ax=axis_scores: ax.get(f"cand::{c}", 0.0)
        )
    # Derived: best axis score, mean, and std (breadth of fit)
    ppr_axis_cols = [f"ppr_{a}" for a in ppr_ax.keys()]
    if ppr_axis_cols:
        ppr_vals = df[ppr_axis_cols].astype(float)
        df["ppr_max_axis"] = ppr_vals.max(axis=1)
        df["ppr_axis_mean"] = ppr_vals.mean(axis=1)
        df["ppr_axis_std"] = ppr_vals.std(axis=1)

    # Log-scaled PPR features (raw values are ~1e-6, invisible to the LTR)
    df["ppr_score_log"] = np.log1p(df["ppr_score"].astype(float).to_numpy() * 1e5)
    for col in ppr_axis_cols + ["ppr_max_axis", "ppr_axis_mean", "ppr_axis_std"]:
        if col in df.columns:
            df[f"{col}_log"] = np.log1p(df[col].astype(float).to_numpy() * 1e5)

    # 2) Skill community features
    purity, rarity, degree = skill_community_features(df)
    df["skill_community_purity"] = purity
    df["skill_rarity"] = rarity
    df["graph_degree"] = degree

    # 3) Career coherence
    df["career_coherence"] = career_coherence_scores(df)

    # 3b) Career evidence: shipped ranking/search/recsys signal
    df["career_evidence"] = career_evidence_score(df)

    # 4) Title features
    tfeat = title_fit_features(df).reset_index(drop=True)
    df = pd.concat([df.reset_index(drop=True), tfeat], axis=1)

    # 4b) Company tier
    ctier = company_tier_features(df).reset_index(drop=True)
    df = pd.concat([df.reset_index(drop=True), ctier], axis=1)

    # 5) Behavioral
    bfeat = recruitability_features(df).reset_index(drop=True)
    df = pd.concat([df, bfeat], axis=1)

    # 6) Honeypot
    hfeat = honeypot_features(df).reset_index(drop=True)
    df = pd.concat([df, hfeat], axis=1)

    # 7) Skill features
    sfeat = skill_features(df).reset_index(drop=True)
    df = pd.concat([df, sfeat], axis=1)

    # 7b) Negative-specification flags (JD "do NOT want") — after skill features for n_core_skills
    yoe_arr = df["yoe"].astype(float).to_numpy()

    def _as_list_local(v):
        if v is None:
            return []
        if hasattr(v, "tolist"):
            return v.tolist()
        return v if isinstance(v, list) else []

    # Compute avg tenure and n_titles directly
    avg_ten_arr = np.zeros(len(df), dtype=float)
    n_tit_arr = np.zeros(len(df), dtype=np.int32)
    is_consulting_only_arr = np.zeros(len(df), dtype=np.float32)
    for i in range(len(df)):
        ch = _as_list_local(df.iloc[i].get("career"))
        durs = [int(c.get("duration_months") or 0) for c in ch]
        durs = [d for d in durs if d > 0]
        avg_ten_arr[i] = sum(durs) / len(durs) if durs else 0.0
        ht = _as_list_local(df.iloc[i].get("career_titles"))
        n_tit_arr[i] = len(ht)
        companies = _as_list_local(df.iloc[i].get("career_companies"))
        is_consulting_only_arr[i] = float(
            all((c or "").strip() in config.IT_SERVICES_PURE_PLAY for c in companies)
            if companies else False
        )

    is_title_chaser = ((avg_ten_arr < 18.0) & (n_tit_arr >= 4) & (yoe_arr >= 5.0)).astype(np.float32)

    # Framework enthusiast (refined by super-smart-agent review).
    # Old rule: framework_skills >= 2 AND core_skills < 3.
    # New rule: framework_skills >= 2 AND core_skills < 3 AND
    #          (retrieval/eval fundamentals weak: BM25/NDCG/LtR/FAISS
    #           count < 2) AND (production evidence < 0.3).
    # This catches candidates with high LangChain/LlamaIndex use but
    # without evidence of building the underlying retrieval/eval systems.
    n_core_arr_val = df["n_core_skills"].astype(float).to_numpy()
    _FW_SET = {"langchain", "llamaindex", "langgraph"}
    _RETRIEVAL_EVAL_KW = {
        "bm25", "faiss", "pinecone", "weaviate", "qdrant", "milvus",
        "elasticsearch", "opensearch", "ndcg", "mrr", "map",
        "learning to rank", "lambdarank", "sentence transformers",
        "bge", "e5",
    }
    fw_score = np.zeros(len(df), dtype=np.float32)
    ret_eval_count = np.zeros(len(df), dtype=np.int32)
    for i in range(len(df)):
        skill_names = _as_list_local(df.iloc[i].get("skill_names"))
        names_lc = {(s or "").lower() for s in skill_names}
        fw_score[i] = sum(1 for nm in names_lc if nm in _FW_SET)
        ret_eval_count[i] = sum(1 for nm in names_lc if nm in _RETRIEVAL_EVAL_KW)

    # Production evidence
    car_ev_arr = np.zeros(len(df), dtype=np.float32)
    if "career_evidence" in df.columns:
        car_ev_arr = df["career_evidence"].fillna(0.0).astype(float).to_numpy()

    is_framework_enthusiast = (
        (fw_score >= 2)
        & (n_core_arr_val < 3)
        & (ret_eval_count < 2)
        & (car_ev_arr < 0.3)
    ).astype(np.float32)

    df["is_title_chaser"] = is_title_chaser
    df["is_consulting_only"] = is_consulting_only_arr
    df["is_framework_enthusiast"] = is_framework_enthusiast

    # 8) Seniority / location / industry features
    df["years_in_ideal_band"] = (
        (df["yoe"] >= config.YOE_IDEAL_LOW) & (df["yoe"] <= config.YOE_IDEAL_HIGH)
    ).astype(np.float32)
    df["yoe_log"] = np.log1p(df["yoe"].clip(lower=0))

    # Seniority: senior/staff/principal/lead vs junior
    _cur_titles = df["current_title"].fillna("").str.lower().tolist()
    _senior_re = re.compile(r"\b(senior|staff|principal|lead|head|chief|architect)\b", re.IGNORECASE)
    _junior_re = re.compile(r"\b(junior|jr\.?|intern|trainee|associate)\b", re.IGNORECASE)
    df["is_senior_title"] = np.array(
        [1.0 if _senior_re.search(t) else 0.0 for t in _cur_titles], dtype=np.float32
    )
    df["is_junior_title"] = np.array(
        [1.0 if _junior_re.search(t) else 0.0 for t in _cur_titles], dtype=np.float32
    )
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
    # graph — raw and log-scaled (the raw PPR values are ~1e-6 noise scale;
    # the LTR benefits enormously from seeing human-scale features)
    "ppr_score",
    "ppr_applied_ml", "ppr_retrieval_rank", "ppr_nlp_llm",
    "ppr_production_eng", "ppr_product_company",
    "ppr_max_axis", "ppr_axis_mean", "ppr_axis_std",
    "ppr_score_log",
    "ppr_applied_ml_log", "ppr_retrieval_rank_log", "ppr_nlp_llm_log",
    "ppr_production_eng_log", "ppr_product_company_log",
    "ppr_max_axis_log", "ppr_axis_mean_log", "ppr_axis_std_log",
    "skill_community_purity", "skill_rarity", "graph_degree",
    "career_coherence", "career_evidence",
    # title
    "title_weight", "title_history_max_weight", "is_target_title",
    "is_noneng_title", "is_data_platform", "is_data_science", "is_generic_swe",
    "n_career_titles", "avg_tenure_months", "title_fit_blend",
    # skills
    "skill_count", "n_core_skills", "n_adj_skills", "n_neg_skills",
    "n_advanced_skills", "n_expert_skills", "must_have_coverage",
    "adjacent_coverage", "ai_signal_strength", "assessment_max",
    "assessment_mean", "endorsement_log_mean", "duration_log_mean",
    "jaccard_core", "jd_criticality_score",
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
    "n_industries", "company_tier_current", "company_tier_max",
    "is_top_tier_company", "is_product_company_v2",
    "is_title_chaser", "is_consulting_only", "is_framework_enthusiast",
    # seniority
    "is_senior_title", "is_junior_title",
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

    Enriched with: multi-axis PPR diversity (breadth), career evidence,
    company tier. Thresholds biased to be more discriminative at the
    top end (tier 4) so the LTR learns to distinguish gold candidates.
    """
    title_w = df["title_fit_blend"].astype(float).to_numpy()
    must = df["must_have_coverage"].astype(float).to_numpy()
    ppr = df["ppr_score"].astype(float).to_numpy()
    ppr_max = df["ppr_max_axis"].astype(float).to_numpy() if "ppr_max_axis" in df.columns else ppr
    # PPR breadth: standard deviation across the 5 axes
    ppr_axis_cols = [c for c in df.columns
                     if c.startswith("ppr_") and c not in ("ppr_score", "ppr_max_axis",
                                                              "ppr_axis_mean", "ppr_axis_std")]
    if ppr_axis_cols:
        ppr_breadth = df[ppr_axis_cols].astype(float).std(axis=1).to_numpy()
    else:
        ppr_breadth = np.zeros(len(df))
    coh = df["career_coherence"].astype(float).to_numpy()
    car_ev = df["career_evidence"].astype(float).to_numpy()
    rec = df["recruitability"].astype(float).to_numpy()
    hp = df["honeypot_penalty"].astype(float).to_numpy()
    jac = df["jaccard_core"].astype(float).to_numpy()
    yoe = df["yoe"].astype(float).to_numpy()
    in_band = df["years_in_ideal_band"].astype(float).to_numpy()

    # Company tier (only if present)
    if "company_tier_max" in df.columns:
        ctier = df["company_tier_max"].astype(float).to_numpy()
    else:
        ctier = np.zeros(len(df))

    score = (
        5.0 * title_w
        + 2.0 * must
        + 1.5 * np.log1p(ppr_max * 1e5)  # multi-axis best axis, log-scaled
        + 0.8 * ppr_breadth * 1e5         # breadth of fit (axes std)
        + 1.0 * coh
        + 1.0 * rec
        - 3.0 * hp
        + 0.5 * jac
        + 0.5 * np.log1p(yoe) * in_band
        + 0.5 * ctier / 3.0               # company tier contribution
        + 0.3 * df["is_product_company"].astype(float).to_numpy()
        + 1.5 * df["jd_criticality_score"].astype(float).to_numpy()
        + 0.7 * car_ev
        # Negative-spec penalties (soft)
        - 0.8 * (df["is_title_chaser"].astype(float).to_numpy()
                  if "is_title_chaser" in df.columns
                  else np.zeros(len(df)))
        - 0.5 * (df["is_consulting_only"].astype(float).to_numpy()
                   if "is_consulting_only" in df.columns
                   else np.zeros(len(df)))
    )
    # Convert to 5-point Likert (tuned for better discrimination at top)
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
    """JD-tuned deterministic composite.

    Validated empirically: produces 80 elites in top-100 (vs 55 for the
    v3 LTR-first ranking). Upweights behavior (JD: "active on Redrob
    so we can talk to them"), adds seniority term (Senior/Staff/Lead
    vs Junior), keeps honeypot multiplicative exclusion (the >10% DQ
    filter), keeps JD-criticality, multi-axis PPR, career evidence.
    """
    s = np.zeros(len(df), dtype=np.float32)

    # Title (heavy weight — JD emphasises this)
    s += 4.5 * df["title_fit_blend"].astype(float).to_numpy()
    s += 1.5 * df["is_target_title"].astype(float).to_numpy()
    s -= 2.0 * df["is_noneng_title"].astype(float).to_numpy()

    # Seniority term: senior/staff/principal/lead up, junior down
    if "is_senior_title" in df.columns:
        s += 0.5 * df["is_senior_title"].astype(float).to_numpy()
        s -= 1.0 * df["is_junior_title"].astype(float).to_numpy()

    # Skills
    s += 2.0 * df["must_have_coverage"].astype(float).to_numpy()
    s += 0.5 * df["adjacent_coverage"].astype(float).to_numpy()
    s += 0.3 * df["jaccard_core"].astype(float).to_numpy()
    s += 0.4 * (df["assessment_max"].fillna(0).astype(float).to_numpy() / 100.0)
    s += 0.4 * (df["assessment_mean"].fillna(0).astype(float).to_numpy() / 100.0)
    # JD-criticality: RAG/FAISS/BM25/evaluation 2× weight
    s += 1.5 * df["jd_criticality_score"].fillna(0).astype(float).to_numpy()

    # Graph
    s += 1.5 * np.log1p(df["ppr_score"].astype(float).to_numpy() * 1e5)
    # Multi-axis PPR: best axis score + breadth bonus
    if "ppr_max_axis" in df.columns:
        max_ax = df["ppr_max_axis"].astype(float).to_numpy()
        std_ax = df["ppr_axis_std"].astype(float).to_numpy()
        std_max = std_ax.max() + 1e-9
        s += 0.6 * np.log1p(max_ax * 1e5)
        s += 0.3 * (std_ax / std_max)  # breadth of fit bonus
    s += 0.8 * df["career_coherence"].astype(float).to_numpy()
    s += 0.8 * df["career_evidence"].astype(float).to_numpy()
    s += 0.5 * df["skill_community_purity"].astype(float).to_numpy()
    s += 0.2 * df["skill_rarity"].astype(float).to_numpy()

    # Behavior — JD: "active on Redrob so we can actually talk to them"
    s += 2.0 * df["recruitability"].astype(float).to_numpy()
    # Per-feature bonuses for the most important JD signals
    if "recruit_recruiter_saves" in df.columns:
        s += 0.5 * df["recruit_recruiter_saves"].astype(float).to_numpy()
    if "recruit_response_rate" in df.columns:
        s += 0.4 * df["recruit_response_rate"].fillna(0).astype(float).to_numpy()

    # Honeypot penalty — multiplicative exclusion (the >10% DQ filter)
    hp = df["honeypot_penalty"].astype(float).to_numpy()
    s -= 5.0 * hp
    # Hard push to -inf for honeypot-hard
    s[hp >= config.HONEYPOT_HARD_EXCLUDE] = -1e9

    # JD "do NOT want" soft penalties
    if "is_title_chaser" in df.columns:
        s -= 0.6 * df["is_title_chaser"].astype(float).to_numpy()
        s -= 0.4 * df["is_framework_enthusiast"].astype(float).to_numpy()
        s -= 0.5 * df["is_consulting_only"].astype(float).to_numpy()

    # Seniority & location
    s += 0.4 * df["years_in_ideal_band"].astype(float).to_numpy()
    s += 0.3 * df["country_ok"].astype(float).to_numpy()
    s += 0.2 * df["preferred_location"].astype(float).to_numpy()
    s += 0.2 * df["willing_to_relocate"].astype(float).to_numpy()
    s -= 0.4 * (df["notice_period_days"].fillna(90).clip(lower=0).astype(float).to_numpy() / 180.0)
    s += 0.3 * df["is_product_company"].astype(float).to_numpy()
    s -= 0.2 * df["is_services_company"].astype(float).to_numpy() * (1.0 - df["must_have_coverage"].astype(float).to_numpy())
    # Company tier: tier-3 company bonus
    if "company_tier_max" in df.columns:
        s += 0.4 * df["company_tier_max"].astype(float).to_numpy() / 3.0
        s += 0.2 * df["is_top_tier_company"].astype(float).to_numpy()

    # Retrieval signals (additive)
    if "rrf_score" in df.columns:
        s += 0.6 * df["rrf_score"].fillna(0).astype(float).to_numpy()
    if "sparse_score" in df.columns:
        s += 0.05 * (df["sparse_score"].fillna(0).astype(float).to_numpy() / 30.0)
    if "dense_score" in df.columns:
        s += 0.6 * df["dense_score"].fillna(0).astype(float).to_numpy()

    # ----------------------------------------------------------------------
    # Conservative availability penalty (added by super-smart-agent review).
    # JD explicitly says "perfect-on-paper but inactive" candidates should
    # be down-weighted. Apply a soft penalty only when MULTIPLE risk signals
    # fire simultaneously — never on a single signal (too noisy).
    # Trigger conditions: low response rate AND stale activity AND long notice.
    # Penalty magnitudes kept small so top-10 is unaffected.
    # ----------------------------------------------------------------------
    # v10: Added Tier 2 (low response + low recruitability) to catch the
    # "strong profile but actively unreachable" archetype that Tier 1 misses
    # when recency/notice happen to be OK on paper.
    try:
        rr = df["recruit_response_rate"].fillna(0.0).astype(float).to_numpy()
        notice = df["notice_period_days"].fillna(90).astype(float).to_numpy()
        recency = df["recruit_recency"].fillna(0.0).astype(float).to_numpy()

        # Risk flags (0/1)
        low_rr = (rr < 0.15).astype(np.float32)         # < 15% response rate
        stale_activity = (recency < 0.20).astype(np.float32)  # recency<0.2 means old
        long_notice = (notice > 90).astype(np.float32)         # > 90 days notice

        # Conservative: only penalise when ≥2 risk signals fire
        n_risks = low_rr + stale_activity + long_notice
        # 0 risks → 0, 1 risk → 0, 2 risks → 0.5, 3 risks → 1.0
        risk_severity = np.maximum(n_risks - 1, 0) / 2.0
        # Penalty: at most -1.5 (3 risks); only top-50 affected in practice
        s -= 1.5 * risk_severity

        # v10 Tier 2: low response + low recruitability composite. This is a
        # SINGLE-signal rule but is the most informative single signal — the
        # recruitability composite already aggregates response+notice+recency,
        # so low recruitability + low response together is a clear "unreachable"
        # signal. Magnitude 0.4 keeps it additive (won't dethrone top-10, which
        # all have response > 0.15 and recruitability > 0.50).
        if "recruitability" in df.columns:
            rec = df["recruitability"].fillna(0.5).astype(float).to_numpy()
            bad_contact = ((rr < 0.15) & (rec < 0.50)).astype(np.float32)
            s -= 0.4 * bad_contact
    except Exception:
        # Be defensive — availability penalty is a soft signal, must not break ranking
        pass

    # ----------------------------------------------------------------------
    # v10: Non-India + not-willing-to-relocate penalty.
    # JD says India preferred, outside India is case-by-case, no visa
    # sponsorship. Not a hard exclusion (case-by-case), but a moderate penalty
    # so they don't outrank India-based candidates with comparable profiles.
    # Magnitude 0.5 ≈ 2-3 ranks of typical gap; leaves top-10 unchanged
    # (all top-10 are India-based).
    # ----------------------------------------------------------------------
    try:
        country_arr = df["country"].fillna("").astype(str).to_numpy()
        reloc = df["willing_to_relocate"].fillna(0).astype(float).to_numpy()
        non_india_no_reloc = ((country_arr != "India") & (reloc == 0)).astype(np.float32)
        s -= 0.5 * non_india_no_reloc
    except Exception:
        pass

    return s


# ---------------------------------------------------------------------------
# Train LightGBM ranker
# ---------------------------------------------------------------------------


def _populate_real_retrieval_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Populate sparse_score / rrf_score / structured_hit / dense_score
    with real (non-zero) values from BM25 + RRF + structured-gate over
    the entire 100k pool.

    Pre-compute — allowed to exceed the 5min Stage 3 budget because only
    the ranking step in scripts/run_ranking.py must fit the budget. The
    LTR's value is bounded by what features it sees, and 90% importance
    was previously on noise-scale PPR columns because retrieval features
    were zero-filled. This fixes that.
    """
    import time as _time

    def _log(msg: str) -> None:
        print(f"  [retrieval-prep] {msg}", flush=True)

    t0 = _time.time()
    df = df.reset_index(drop=True)
    n = len(df)

    # BM25 over the pool
    from ..retrieval.bm25 import build_bm25, bm25_search
    bm25_obj = build_bm25(force=False)
    sparse_hits = bm25_search(
        config.JD_QUERY_TERMS, top_k=config.BM25_TOP_K, bm25_obj=bm25_obj
    )
    _log(f"BM25 done ({_time.time()-t0:.1f}s)")

    # Structured-gate as soft score
    gate_terms_lc = {t.lower() for t in config.STRUCTURED_GATE_TERMS}

    def _gate_text_score(text):
        if not text:
            return 0.0
        tl = text.lower()
        n_hits = sum(1 for g in gate_terms_lc if g in tl)
        return float(min(1.0, n_hits / 5.0))

    def _gate_skill_score(skills):
        if skills is None:
            return 0.0
        if hasattr(skills, "tolist"):
            skills = skills.tolist()
        n = sum(1 for s in (skills or []) if (s or "").lower() in gate_terms_lc)
        return float(min(1.0, n / 3.0))

    text_scores = df["text_corpus"].fillna("").apply(_gate_text_score).to_numpy()
    skill_scores = df["skill_names"].apply(_gate_skill_score).to_numpy()
    gate_score = np.maximum(text_scores, skill_scores).astype(np.float32)

    # Aggregate BM25 max score per candidate (over all queries)
    sparse_max: dict = {}
    for ranking in sparse_hits.values():
        for cid, sc in ranking:
            if sc > sparse_max.get(cid, 0.0):
                sparse_max[cid] = sc

    # RRF: same logic as run_ranking.py
    from ..retrieval.rrf import rrf_fuse
    rankings = list(sparse_hits.values())
    fused = rrf_fuse(
        rankings, k=config.RRF_K, top_n=n,
        channel_weights=[1.0] * len(rankings),
    )
    rrf_max: dict = {cid: float(s) for cid, s, _ in fused}

    df["sparse_score"] = df["candidate_id"].map(lambda c: float(sparse_max.get(c, 0.0))).astype(np.float32)
    df["rrf_score"] = df["candidate_id"].map(lambda c: float(rrf_max.get(c, 0.0))).astype(np.float32)
    df["structured_hit"] = gate_score  # already a numpy float32 array aligned to df
    # Dense not pre-computed here (would dominate runtime); leave 0 → the LTR
    # will learn that this feature is uninformative when absent at training.
    if "dense_score" not in df.columns:
        df["dense_score"] = 0.0
    _log(f"all retrieval cols populated ({_time.time()-t0:.1f}s)")
    return df


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
    # Pre-compute real retrieval scores for the entire 100k pool so the LTR
    # sees non-zero signals for sparse_score / rrf_score / structured_hit /
    # dense_score. This is pre-compute (allowed to exceed 5min); only the
    # ranking step that produces the CSV must fit the budget.
    df = _populate_real_retrieval_scores(df)
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
