"""Skill-related features: must-have coverage, assessments, durations."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config


_CORE = {s.lower() for s in config.CORE_COMPETENCIES}
_ADJ = {s.lower() for s in config.ADJACENT_COMPETENCIES}
_NEG = {s.lower() for s in config.NEGATIVE_COMPETENCIES}
_CRIT = {s.lower() for s in config.JD_CRITICAL}
_NICE = {s.lower() for s in config.JD_NICE_TO_HAVE}
_SKILL_WEIGHT = {s: 2.0 for s in _CRIT}
_SKILL_WEIGHT.update({s: 1.0 for s in _NICE})
for s in _CORE - _CRIT - _NICE:
    _SKILL_WEIGHT.setdefault(s, 0.5)
# Max possible weighted score (6 critical + 4 nice-to-have)
_MAX_WEIGHT = sum(sorted(_SKILL_WEIGHT.values(), reverse=True)[:10])


# ----------------------------------------------------------------------
# Skill alias / canonicalization map (added by agent review, Exp #8)
# ----------------------------------------------------------------------
# Maps variant skill names (as they appear in candidates.jsonl and
# Redrob assessments) to their canonical JD-core counterparts.
# This boosts recall for candidates who list the same skill under a
# slightly different name (e.g. "Fine-tuning LLMs" → "Fine-tuning",
# "pgvector" → vector DB family, "Search Relevance" → BM25 family).
_SKILL_ALIAS = {
    # Retrieval / vector DB family
    "pgvector": "vector database",
    "pgvector (postgres vector)": "vector database",
    "vector database": "vector database",
    "vector databases": "vector database",
    "vector db": "vector database",
    "dense retrieval": "dense retrieval",
    "dense passage retrieval": "dense retrieval",
    "dense passage retrieval (dpr)": "dense retrieval",
    "approximate nearest neighbor": "vector database",
    "faiss": "faiss",
    "ann": "faiss",
    "hnsw": "faiss",
    "scaNN": "faiss",
    "qdrant": "qdrant",
    # Search ranking family
    "search relevance": "bm25",
    "search ranking": "bm25",
    "relevance ranking": "learning to rank",
    "relevance": "bm25",
    "search engineer": "bm25",
    "lexical search": "bm25",
    "tf-idf": "bm25",
    "tfidf": "bm25",
    # LLM / fine-tuning family
    "fine-tuning llms": "fine-tuning",
    "llm fine-tuning": "fine-tuning",
    "llm finetuning": "fine-tuning",
    "fine-tuning": "fine-tuning",
    "finetuning": "fine-tuning",
    "instruction tuning": "fine-tuning",
    "rlhf": "fine-tuning",
    "rlhf (reinforcement learning from human feedback)": "fine-tuning",
    "prompt tuning": "prompt engineering",
    "prompt engineering": "prompt engineering",
    "prompt design": "prompt engineering",
    "prompt crafting": "prompt engineering",
    "rag (retrieval-augmented generation)": "rag",
    "retrieval augmented generation (rag)": "rag",
    "retrieval-augmented generation (rag)": "rag",
    "retrieval augmented generation": "rag",
    "rag": "rag",
    # Evaluation family
    "learning-to-rank": "learning to rank",
    "learning to rank (ltr)": "learning to rank",
    "ltr (learning to rank)": "learning to rank",
    "learning2rank": "learning to rank",
    "lambdarank": "lambdarank",
    "lambda rank": "lambdarank",
    "lambdamart": "lambdarank",
    "ndcg (normalised discounted cumulative gain)": "ndcg",
    "mrr (mean reciprocal rank)": "mrr",
    "map (mean average precision)": "map",
    "mean reciprocal rank": "mrr",
    "mean average precision": "map",
    "normalised discounted cumulative gain": "ndcg",
    # Embeddings / encoders
    "sentence-transformer": "sentence transformers",
    "sentence transformer": "sentence transformers",
    "sentence-transformers": "sentence transformers",
    "sbert": "sentence transformers",
    "use (universal sentence encoder)": "sentence transformers",
    "bge (bge embedding)": "bge",
    "e5 (text embeddings)": "e5",
    # Hybrid search
    "hybrid retrieval": "hybrid search",
    "hybrid ranking": "hybrid search",
    "hybrid information retrieval": "hybrid search",
    # OpenSearch / Elasticsearch
    "elastic search": "elasticsearch",
    "opensearch (managed elasticsearch)": "opensearch",
    # Pinecone / Weaviate
    "pinecone (vector database)": "pinecone",
    "weaviate (vector database)": "weaviate",
    # LLM
    "large language models": "llm",
    "large language model (llm)": "llm",
    "llm (large language models)": "llm",
    "llms": "llm",
    # NLP
    "natural language processing": "nlp",
    "nlp (natural language processing)": "nlp",
    # Vector Search alternative
    "vector search (semantic search)": "embeddings",
    "vector search": "embeddings",
    "semantic search": "embeddings",
    "approximate nearest neighbours": "vector database",
}


def _canon_skill(name: str) -> str:
    """Return the canonical form of a skill name. Falls back to lower(name)."""
    n = (name or "").strip().lower()
    if not n:
        return ""
    return _SKILL_ALIAS.get(n, n)


def _canon_skill_set(names: set) -> set:
    """Apply canonicalization to a set of skill names."""
    return {_canon_skill(n) for n in names if n}


def skill_features(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)

    def _as_list(v):
        if v is None:
            return []
        if hasattr(v, "tolist"):
            return v.tolist()
        return v

    skills_full_col = [_as_list(x) for x in df["skills"].tolist()]
    sigs_col = [_as_list(x) for x in df["signals"].tolist()]
    n_core_hits = np.zeros(n, dtype=np.int32)
    n_adj_hits = np.zeros(n, dtype=np.int32)
    n_neg_hits = np.zeros(n, dtype=np.int32)
    n_advanced = np.zeros(n, dtype=np.int32)
    n_expert = np.zeros(n, dtype=np.int32)
    assessment_max = np.zeros(n, dtype=np.float32)
    assessment_mean = np.zeros(n, dtype=np.float32)
    endorsement_log_mean = np.zeros(n, dtype=np.float32)
    duration_log_mean = np.zeros(n, dtype=np.float32)
    skill_count = np.zeros(n, dtype=np.int32)
    jaccard_core = np.zeros(n, dtype=np.float32)
    jd_crit_score = np.zeros(n, dtype=np.float32)
    # Alias-canonicalized sets (for matching against core/jd-critical)
    names_canon_lc_col = []

    for i in range(n):
        skills_full = skills_full_col[i]
        if skills_full is None:
            skills_full = []
        names = [s.get("name", "").strip() for s in skills_full if s.get("name")]
        names_lc = {n_.lower() for n_ in names if n_}
        names_canon_lc = _canon_skill_set(names_lc)
        names_canon_lc_col.append(names_canon_lc)
        skill_count[i] = len(names)
        n_core_hits[i] = len(names_canon_lc & _CORE)
        n_adj_hits[i] = len(names_canon_lc & _ADJ)
        n_neg_hits[i] = len(names_canon_lc & _NEG)
        n_advanced[i] = sum(1 for s in skills_full if s.get("proficiency") in ("advanced",))
        n_expert[i] = sum(1 for s in skills_full if s.get("proficiency") == "expert")

        # JD-criticality weighted score (uses canonical names)
        w_sum = sum(_SKILL_WEIGHT.get(n, 0.0) for n in names_canon_lc)
        jd_crit_score[i] = min(w_sum / max(_MAX_WEIGHT, 1e-9), 1.0)

        # Assessments — canonicalize keys to merge aliases (e.g. "pgvector" and
        # "vector database" both map to the same canonical key for aggregation)
        sig = sigs_col[i]
        if sig is None:
            sig = {}
        if isinstance(sig, dict):
            sas = sig.get("skill_assessment_scores") or {}
            # Canonicalize keys and merge values (take max for same canonical)
            canon_sas: dict = {}
            for k, v in sas.items():
                if v is None or not isinstance(v, (int, float)):
                    continue
                ck = _canon_skill(k)
                if ck in canon_sas:
                    if v > canon_sas[ck]:
                        canon_sas[ck] = float(v)
                else:
                    canon_sas[ck] = float(v)
            if canon_sas:
                vals = list(canon_sas.values())
                assessment_max[i] = max(vals)
                assessment_mean[i] = sum(vals) / len(vals)
        # Endorsement / duration log means
        e_vals = [int(s.get("endorsements") or 0) for s in skills_full]
        d_vals = [int(s.get("duration_months") or 0) for s in skills_full]
        if e_vals:
            endorsement_log_mean[i] = float(np.mean(np.log1p(e_vals)))
        if d_vals:
            duration_log_mean[i] = float(np.mean(np.log1p(d_vals)))

        # Jaccard of candidate's skills with core (canonicalized)
        if names_canon_lc and _CORE:
            inter = len(names_canon_lc & _CORE)
            union = len(names_canon_lc | _CORE)
            jaccard_core[i] = inter / max(union, 1)

    must_have_coverage = np.minimum(n_core_hits / 8.0, 1.0)  # saturates at 8 core skills
    adj_coverage = np.minimum(n_adj_hits / 5.0, 1.0)
    ai_signal_strength = (n_core_hits + 0.5 * n_adj_hits) / np.maximum(1, skill_count)

    return pd.DataFrame({
        "skill_count": skill_count,
        "n_core_skills": n_core_hits,
        "n_adj_skills": n_adj_hits,
        "n_neg_skills": n_neg_hits,
        "n_advanced_skills": n_advanced,
        "n_expert_skills": n_expert,
        "must_have_coverage": must_have_coverage,
        "adjacent_coverage": adj_coverage,
        "ai_signal_strength": ai_signal_strength,
        "assessment_max": assessment_max,
        "assessment_mean": assessment_mean,
        "endorsement_log_mean": endorsement_log_mean,
        "duration_log_mean": duration_log_mean,
        "jaccard_core": jaccard_core,
        "jd_criticality_score": jd_crit_score,
    })
