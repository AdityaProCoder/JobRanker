"""End-to-end ranking pipeline.

This is THE reproduce command listed in submission_metadata.yaml.

Usage:
    python scripts/run_ranking.py \\
        --candidates "<path-to-candidates.jsonl>" \\
        --out        "<path-to-submission.csv>"

It streams the candidate pool, builds BM25 + (optional) dense indices,
fuses them with RRF, computes the full feature set on the shortlist, and
applies the trained ranker (or the deterministic fallback). The result
is a 100-row CSV that the validator checks for compliance.

If a precomputed artifact is missing, the pipeline rebuilds it. All
artifacts live under `artifacts/`.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Determinism
os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from redrob import config
from redrob.blueprint import build_blueprint, save_blueprint
from redrob.data import load_or_build_parquet
from redrob.retrieval.bm25 import build_bm25, bm25_search
from redrob.retrieval.dense import (
    build_dense_index, dense_search, dense_model_name,
)
from redrob.retrieval.rrf import rrf_fuse
from redrob.rank.predict import rank_shortlist
from redrob.submit.write import write_submission


def _save_id_map(path: Path, ids: list) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(ids, f)


def _load_id_map(path: Path) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run(args) -> None:
    t_start = time.time()
    print(f"[{time.time()-t_start:5.1f}s] starting pipeline")

    # 1) Blueprint
    save_blueprint()
    bp = build_blueprint()
    queries = bp["query_terms"]
    print(f"[{time.time()-t_start:5.1f}s] blueprint saved ({len(config.CORE_COMPETENCIES)} core skills)")

    # 2) Load or build the canonicalised parquet
    df = load_or_build_parquet(args.candidates)
    print(f"[{time.time()-t_start:5.1f}s] canonicalised: {len(df):,} candidates")

    # 3) BM25 sparse retrieval
    bm25, _ = build_bm25(force=False)
    sparse_hits = bm25_search(queries, top_k=config.BM25_TOP_K, bm25_obj=(bm25, _))
    sparse_rankings = list(sparse_hits.values())
    print(f"[{time.time()-t_start:5.1f}s] BM25 ready (top-{config.BM25_TOP_K} per query)")

    # 4) Dense retrieval (or fallback)
    dense_rankings: list = []
    if not args.no_dense:
        try:
            t_dense = time.time()
            print(f"[{time.time()-t_start:5.1f}s] building dense index (model={dense_model_name()}) ...")
            # If the cache exists, use it (fast). Otherwise encode and cache.
            emb, ids = build_dense_index(force=False)
            dense_hits = dense_search(queries, top_k=config.DENSE_TOP_K, matrix=(emb, ids))
            dense_rankings = list(dense_hits.values())
            print(f"[{time.time()-t_start:5.1f}s] dense ready in {time.time()-t_dense:.1f}s")
        except Exception as e:  # noqa: BLE001
            print(f"[{time.time()-t_start:5.1f}s] dense unavailable ({e}); using BM25-only")
            dense_rankings = []
    else:
        print(f"[{time.time()-t_start:5.1f}s] dense path disabled by --no_dense")

    # 5) Structured high-recall gate as SOFT score (saturating)
    gate_terms_lc = {t.lower() for t in config.STRUCTURED_GATE_TERMS}
    df_reset = df.reset_index(drop=True)

    def _gate_text_score(text):
        if not text:
            return 0
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

    text_scores = df_reset["text_corpus"].fillna("").apply(_gate_text_score).to_numpy()
    skill_scores = df_reset["skill_names"].apply(_gate_skill_score).to_numpy()
    gate_score = np.maximum(text_scores, skill_scores)
    structured_hit = (gate_score > 0.0).astype(np.float32)
    print(f"[{time.time()-t_start:5.1f}s] structured gate: median score {np.median(gate_score):.2f}, "
          f"hits {int(structured_hit.sum()):,}/{len(df):,}")

    # 6) RRF over all channels with per-channel weights
    # BM25 (reliable) > structured-gate (binary OK) > dense (noisy in this domain)
    rankings = list(sparse_rankings) + list(dense_rankings)
    if dense_rankings:
        channel_weights = [1.2] * len(sparse_rankings) + [0.8] * len(dense_rankings)
    else:
        channel_weights = [1.0] * len(sparse_rankings)
    fused = rrf_fuse(rankings, k=config.RRF_K, top_n=config.SHORTLIST_N,
                     channel_weights=channel_weights)
    shortlist_ids = [cid for cid, _score, _r in fused]
    shortlist_set = set(shortlist_ids)
    if args.verbose:
        print(f"[{time.time()-t_start:5.1f}s] RRF shortlist: {len(shortlist_ids):,}")

    # Add top structured-gate candidates not already in shortlist (sorted by gate_score)
    struct_candidates = df_reset.loc[gate_score > 0.0].copy()
    struct_candidates = struct_candidates.assign(_g=gate_score[gate_score > 0.0])
    struct_candidates = struct_candidates.sort_values("_g", ascending=False)
    added = 0
    for cid in struct_candidates["candidate_id"].tolist():
        if cid not in shortlist_set:
            shortlist_ids.append(cid)
            shortlist_set.add(cid)
            added += 1
            if added >= args.structured_top_up:
                break
    if args.verbose:
        print(f"[{time.time()-t_start:5.1f}s] +{added:,} from structured gate (total shortlist: {len(shortlist_ids):,})")
    struct_ids = df_reset.loc[gate_score > 0.0, "candidate_id"].tolist()
    gate_score_map = {cid: float(s) for cid, s in zip(df_reset["candidate_id"].tolist(), gate_score)}

    # 7) Build the shortlist dataframe
    shortlist_df = df_reset[df_reset["candidate_id"].isin(shortlist_set)].copy()

    # 8) Per-channel scores for the shortlist (sparse/dense/rrf)
    sparse_score_by_id: dict = {}
    for ranking in sparse_rankings:
        for cid, sc in ranking:
            if cid in shortlist_set:
                sparse_score_by_id[cid] = max(sparse_score_by_id.get(cid, 0.0), sc)
    dense_score_by_id: dict = {}
    for ranking in dense_rankings:
        for cid, sc in ranking:
            if cid in shortlist_set:
                dense_score_by_id[cid] = max(dense_score_by_id.get(cid, 0.0), sc)
    rrf_score_by_id: dict = {cid: s for cid, s, _r in fused}

    shortlist_df["sparse_score"] = shortlist_df["candidate_id"].map(lambda c: sparse_score_by_id.get(c, 0.0))
    shortlist_df["dense_score"] = shortlist_df["candidate_id"].map(lambda c: dense_score_by_id.get(c, 0.0))
    shortlist_df["rrf_score"] = shortlist_df["candidate_id"].map(lambda c: rrf_score_by_id.get(c, 0.0))
    shortlist_df["structured_hit"] = shortlist_df["candidate_id"].map(lambda c: gate_score_map.get(c, 0.0))

    print(f"[{time.time()-t_start:5.1f}s] computing features for {len(shortlist_df):,} shortlist candidates")

    # 9) Rank (uses LightGBM if model exists, else deterministic fallback)
    use_lgbm = not args.no_lgbm
    if not use_lgbm:
        print(f"[{time.time()-t_start:5.1f}s] ranker: deterministic fallback (--no_lgbm)")
    else:
        print(f"[{time.time()-t_start:5.1f}s] ranker: LightGBM (auto-trains if missing)")
    ranked = rank_shortlist(shortlist_df, use_lgbm=use_lgbm)

    # 10) Top-100
    top = ranked.head(100).copy()
    print(f"[{time.time()-t_start:5.1f}s] top-100 selected")

    # 11) Honeypot self-check
    n_honeypot = int((top["honeypot_penalty"] >= config.HONEYPOT_HARD_EXCLUDE).sum())
    n_low_title = int((top["title_fit_blend"] < 0.10).sum())
    n_services_top50 = int(top.head(50)["is_services_company"].astype(int).sum()) if "is_services_company" in top.columns else 0
    print(f"[{time.time()-t_start:5.1f}s] self-check: honeypot_hard={n_honeypot}, low_title_top100={n_low_title}, services_top50={n_services_top50}")

    # 12) Write CSV
    out = write_submission(top, out_path=args.out, validate=True)
    print(f"[{time.time()-t_start:5.1f}s] submission written: {out}")
    print(f"[{time.time()-t_start:5.1f}s] DONE")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--candidates",
        default=str(config.DEFAULT_CANDIDATES_JSONL),
        help="Path to candidates.jsonl",
    )
    ap.add_argument(
        "--out",
        default=str(config.DEFAULT_OUTPUT_CSV),
        help="Path to submission CSV",
    )
    ap.add_argument(
        "--no_dense",
        action="store_true",
        help="Disable sentence-transformers (use BM25-only retrieval)",
    )
    ap.add_argument(
        "--no_lgbm",
        action="store_true",
        help="Use the deterministic scorer instead of the LightGBM ranker",
    )
    ap.add_argument(
        "--structured_top_up",
        type=int,
        default=2000,
        help="How many structured-gate candidates to add beyond the RRF shortlist",
    )
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
