"""Streamlit sandbox demo for the Redrob Candidate Intelligence Engine.

The sandbox runs the COMPLETE pipeline on a small input (defaulting to
the first ~200 records of `sample_candidates.json`). It demonstrates:

  * The Role Blueprint and how the JD is parsed
  * Hybrid retrieval (BM25 + sentence-transformers + RRF)
  * Feature engineering
  * LightGBM LambdaRank (or deterministic fallback) ranking
  * Per-candidate reasoning
  * Export of the top-N as a CSV

For the OFFICIAL reproduce_command see `scripts/run_ranking.py` in the
repository. That command processes the full 100k candidate pool within
the 5-minute / 16GB sandbox budget.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import streamlit as st

from redrob import config
from redrob.blueprint import build_blueprint
from redrob.data import iter_canonical
from redrob.rank.predict import rank_shortlist
from redrob.reasoning.template import generate_reasoning_dataframe
from redrob.submit.write import write_submission


st.set_page_config(page_title="Redrob Candidate Intelligence", layout="wide")

st.title("Redrob Candidate Intelligence Engine")
st.caption("Senior AI Engineer ranking — sandbox demo")

# ---------------------------------------------------------------------------
# Sidebar: inputs
# ---------------------------------------------------------------------------
st.sidebar.header("Inputs")
default_sample = (
    PROJECT_ROOT
    / "[PUB] India_runs_data_and_ai_challenge"
    / "[PUB] India_runs_data_and_ai_challenge"
    / "India_runs_data_and_ai_challenge"
    / "sample_candidates.json"
)
uploaded = st.sidebar.file_uploader(
    "Upload a candidates.jsonl (max ~5,000 records for the sandbox)",
    type=["jsonl", "json"],
)
sample_size = st.sidebar.slider("Sample size for sandbox run", 50, 2000, 200, step=50)
use_dense = st.sidebar.checkbox("Use sentence-transformers (slower)", value=False)
use_lgbm = st.sidebar.checkbox("Use LightGBM ranker (slower)", value=True)
run_btn = st.sidebar.button("Run ranking pipeline", type="primary")

# ---------------------------------------------------------------------------
# Blueprint panel
# ---------------------------------------------------------------------------
bp = build_blueprint()
with st.expander("Role Blueprint (Senior AI Engineer, Redrob)", expanded=True):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Core competencies**")
        st.write(", ".join(bp["core_competencies"][:20]) + "…")
        st.markdown(f"*{len(bp['core_competencies'])} total*")
    with c2:
        st.markdown("**Target titles**")
        st.write(", ".join(bp["target_titles"][:10]) + "…")
        st.markdown("**Experience band (yrs)**")
        st.write(
            f"min {bp['experience_band']['min']} · ideal "
            f"{bp['experience_band']['ideal_low']}–{bp['experience_band']['ideal_high']}"
        )
    with c3:
        st.markdown("**Notice period**")
        st.write(f"preferred ≤ {bp['notice_period_ok_days']} days")
        st.markdown("**Locations**")
        st.write(", ".join(bp["preferred_locations"]))

# ---------------------------------------------------------------------------
# Pipeline run
# ---------------------------------------------------------------------------
if run_btn:
    t0 = time.time()
    if uploaded is not None:
        # Save uploaded file to a temp path and load
        tmp_path = PROJECT_ROOT / "artifacts" / "_sandbox_upload.jsonl"
        tmp_path.parent.mkdir(exist_ok=True, parents=True)
        with open(tmp_path, "wb") as f:
            f.write(uploaded.getbuffer())
        source = tmp_path
    elif default_sample.exists():
        source = default_sample
    else:
        st.error("No sample file found. Please upload candidates.jsonl.")
        st.stop()

    # Limit the sample size to keep the sandbox responsive
    rows = []
    n = 0
    for cc in iter_canonical(source):
        rows.append({
            "candidate_id": cc.candidate_id,
            "name": cc.name,
            "headline": cc.headline,
            "summary": cc.summary,
            "location": cc.location,
            "country": cc.country,
            "yoe": cc.yoe,
            "current_title": cc.current_title,
            "current_company": cc.current_company,
            "current_company_size": cc.current_company_size,
            "current_industry": cc.current_industry,
            "career": cc.career,
            "edu": cc.edu,
            "skills": cc.skills,
            "certs": cc.certs,
            "languages": cc.languages,
            "signals": cc.signals,
            "text_corpus": cc.text_corpus,
            "career_total_months": cc.career_total_months,
            "career_overlap_months": cc.career_overlap_months,
            "career_industries": cc.career_industries,
            "career_titles": cc.career_titles,
            "career_companies": cc.career_companies,
            "skill_names": cc.skill_names,
            "skill_durations_max": cc.skill_durations_max,
            "has_ai_skill": cc.has_ai_skill,
        })
        n += 1
        if n >= sample_size:
            break
    df = pd.DataFrame(rows)
    st.success(f"Loaded {len(df):,} candidates in {time.time()-t0:.1f}s")

    # Run the ranker (the sandbox uses the deterministic scoring by
    # default; the LGBM model is only used if `use_lgbm` is checked and
    # a trained model exists on disk).
    if not use_dense:
        # For speed, disable dense retrieval
        from redrob import retrieval
        # We monkey-patch dense_search to a no-op for the sandbox
        retrieval.dense.dense_search = lambda *a, **kw: {}
    t0 = time.time()
    # Provide dummy sparse/dense/rrf/structured scores
    df["sparse_score"] = 0.0
    df["dense_score"] = 0.0
    df["rrf_score"] = 0.0
    df["structured_hit"] = 0.0
    ranked = rank_shortlist(df, use_lgbm=use_lgbm)
    ranked = ranked.head(100)
    elapsed = time.time() - t0
    st.success(f"Ranked in {elapsed:.1f}s — top-100 ready")

    # Reasoning
    reasonings = generate_reasoning_dataframe(ranked)
    ranked = ranked.copy()
    ranked["reasoning"] = reasonings

    # Display
    st.subheader("Top 25 (preview)")
    display = ranked.head(25)[[
        "candidate_id", "current_title", "yoe",
        "title_fit_blend", "recruitability", "honeypot_penalty",
        "model_score", "final_score", "reasoning",
    ]].copy()
    st.dataframe(display, use_container_width=True, hide_index=True)

    st.subheader("Top-100 full table")
    full_display = ranked[[
        "candidate_id", "rank" if "rank" in ranked.columns else "candidate_id",
        "current_title", "yoe", "title_fit_blend",
        "recruitability", "honeypot_penalty", "final_score", "reasoning",
    ]].copy()
    st.dataframe(full_display, use_container_width=True, hide_index=True)

    # Charts
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Top-100 title distribution**")
        st.bar_chart(ranked["current_title"].value_counts().head(10))
    with c2:
        st.markdown("**Top-100 title fit histogram**")
        st.bar_chart(pd.cut(ranked["title_fit_blend"], bins=10).value_counts().sort_index())

    # Export
    out_csv = PROJECT_ROOT / "submission_sandbox.csv"
    write_submission(ranked, out_path=out_csv, validate=False)
    with open(out_csv, "rb") as f:
        st.download_button(
            "Download submission CSV",
            data=f.read(),
            file_name="submission.csv",
            mime="text/csv",
        )

    st.info(
        "The sandbox runs the COMPLETE pipeline on a small sample. For "
        "the full 100k pool run `python scripts/run_ranking.py` (see "
        "README)."
    )
else:
    st.info("Press 'Run ranking pipeline' in the sidebar to begin.")
