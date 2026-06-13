"""Career coherence and skill community features."""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
import pandas as pd

from .. import config
from .build import build_graph


# --- title embeddings (cached) -------------------------------------------------

def _title_centroid_embeddings() -> Dict[str, np.ndarray]:
    """Encode every unique title once. Uses sentence-transformers if available,
    otherwise a one-hot over known role groups.
    """
    if config.TITLE_EMB_NPY.exists():
        try:
            with open(str(config.TITLE_EMB_NPY) + ".ids.txt", "r", encoding="utf-8") as f:
                ids = [line.strip() for line in f if line.strip()]
            arr = np.load(config.TITLE_EMB_NPY)
            return {ids[i]: arr[i] for i in range(len(ids))}
        except Exception:
            pass

    from ..data import load_or_build_parquet
    df = load_or_build_parquet()
    titles = sorted({(t or "").strip() for t in df["career_titles"].explode().tolist() if t})
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
        from ..retrieval.dense import _load_model
        model = _load_model()
        embs = model.encode(
            titles,
            batch_size=128,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)
    except Exception:
        # Deterministic fallback: bag-of-words over known role keywords
        vocab: Dict[str, int] = {}
        for t in titles:
            for tok in re.findall(r"[a-z]+", t.lower()):
                if tok not in vocab:
                    vocab[tok] = len(vocab)
        embs = np.zeros((len(titles), max(len(vocab), 1)), dtype=np.float32)
        for i, t in enumerate(titles):
            for tok in re.findall(r"[a-z]+", t.lower()):
                embs[i, vocab[tok]] = 1.0
        n = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8
        embs = embs / n

    np.save(config.TITLE_EMB_NPY, embs)
    with open(str(config.TITLE_EMB_NPY) + ".ids.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(titles))
    return {titles[i]: embs[i] for i in range(len(titles))}


def career_coherence_scores(df: pd.DataFrame) -> np.ndarray:
    """Mean cosine similarity of consecutive title embeddings for each row."""
    embs = _title_centroid_embeddings()
    out = np.zeros(len(df), dtype=np.float32)
    df_reset = df.reset_index(drop=True)
    for i in range(len(df_reset)):
        titles = df_reset.iloc[i]["career_titles"]
        if titles is None:
            titles = []
        if hasattr(titles, "tolist"):
            titles = titles.tolist()
        if not titles or len(titles) < 2:
            out[i] = 0.5  # neutral default for single-title records
            continue
        vecs = []
        for t in titles:
            v = embs.get((t or "").strip())
            if v is not None:
                vecs.append(v)
        if len(vecs) < 2:
            out[i] = 0.5
            continue
        sims = []
        for j in range(1, len(vecs)):
            s = float(np.dot(vecs[j - 1], vecs[j]))
            sims.append(s)
        out[i] = float(np.mean(sims))
    return out


# --- skill community features (Louvain over skill-skill subgraph) ------------

def _louvain_communities(G: nx.Graph) -> Dict[str, int]:
    try:
        from networkx.algorithms.community import louvain_communities  # type: ignore
        comms = louvain_communities(G, weight="weight", resolution=1.0, seed=42)
        out: Dict[str, int] = {}
        for idx, c in enumerate(comms):
            for n in c:
                out[n] = idx
        return out
    except Exception:
        # Greedy modularity fallback
        try:
            from networkx.algorithms.community import greedy_modularity_communities
            comms = list(greedy_modularity_communities(G, weight="weight"))
            out = {}
            for idx, c in enumerate(comms):
                for n in c:
                    out[n] = idx
            return out
        except Exception:
            return {}


def skill_community_features(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (community_purity, skill_rarity, graph_degree) per row.

    - community_purity: fraction of candidate's skills that share a community
      with at least one JD-core skill.
    - skill_rarity: sum of inverse-skill-frequency * jd_match over skills.
    - graph_degree: log1p of candidate's degree in the candidate-skill subgraph.
    """
    G = build_graph()
    skill_nodes = [n for n, d in G.nodes(data=True) if d.get("kind") == "skill"]
    skill_subg = G.subgraph(skill_nodes).copy()
    communities = _louvain_communities(skill_subg)

    jd_skill_nodes = {f"skill::{t.strip()}" for t in config.CORE_COMPETENCIES}
    jd_community_ids = {communities.get(n) for n in jd_skill_nodes if n in communities and communities.get(n) is not None}
    jd_community_ids.discard(None)

    # Skill frequency (for rarity)
    skill_freq: Dict[str, int] = defaultdict(int)
    for u, v, d in G.edges(data=True):
        if d.get("kind") == "has_skill":
            skill_freq[v] += 1
    total_cands = max(1, len(df))
    idf = {s: float(np.log((1 + total_cands) / (1 + f))) for s, f in skill_freq.items()}

    purity = np.zeros(len(df), dtype=np.float32)
    rarity = np.zeros(len(df), dtype=np.float32)
    degree = np.zeros(len(df), dtype=np.float32)

    df_reset = df.reset_index(drop=True)
    for i, row in df_reset.iterrows():
        skill_names = row["skill_names"]
        if skill_names is None:
            skill_names = []
        if hasattr(skill_names, "tolist"):
            skill_names = skill_names.tolist()
        skills = [f"skill::{(s or '').strip()}" for s in skill_names if s]
        if not skills:
            continue
        n = len(skills)
        n_jd_aligned = 0
        rsum = 0.0
        for s in skills:
            if s in communities and communities[s] in jd_community_ids:
                n_jd_aligned += 1
            rsum += idf.get(s, 0.0)
        purity[i] = n_jd_aligned / n
        rarity[i] = rsum / n
        cnode = f"cand::{row['candidate_id']}"
        deg = G.degree(cnode) if cnode in G else 0
        degree[i] = float(np.log1p(deg))
    return purity, rarity, degree


if __name__ == "__main__":
    from ..data import load_or_build_parquet
    df = load_or_build_parquet().head(1000)
    coh = career_coherence_scores(df)
    purity, rarity, deg = skill_community_features(df)
    print("coherence range:", coh.min(), coh.max())
    print("purity range:", purity.min(), purity.max())
    print("rarity range:", rarity.min(), rarity.max())
    print("degree range:", deg.min(), deg.max())
