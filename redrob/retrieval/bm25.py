"""BM25 sparse retrieval over candidate text corpora."""
from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .. import config
from ..data import _tokenize, load_or_build_parquet


def _query_tokens(query: str) -> List[str]:
    return _tokenize(query)


def build_bm25(force: bool = False) -> Tuple[object, List[str]]:
    """Build (or load) the BM25 index over the parquet corpus."""
    if not force and config.BM25_PICKLE.exists() and config.BM25_TOKENS_PICKLE.exists():
        with open(config.BM25_PICKLE, "rb") as f:
            bm25 = pickle.load(f)
        with open(config.BM25_TOKENS_PICKLE, "rb") as f:
            tokens_list = pickle.load(f)
        return bm25, tokens_list

    from rank_bm25 import BM25Okapi  # type: ignore

    df = load_or_build_parquet()
    tokens_list: List[List[str]] = []
    for txt in df["text_corpus"].tolist():
        tokens_list.append(_tokenize(txt or ""))

    bm25 = BM25Okapi(tokens_list)

    with open(config.BM25_PICKLE, "wb") as f:
        pickle.dump(bm25, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(config.BM25_TOKENS_PICKLE, "wb") as f:
        pickle.dump(tokens_list, f, protocol=pickle.HIGHEST_PROTOCOL)
    return bm25, tokens_list


def bm25_search(
    queries: List[str],
    top_k: int = config.BM25_TOP_K,
    bm25_obj: Optional[Tuple[object, List[str]]] = None,
    candidate_ids: Optional[List[str]] = None,
) -> Dict[str, List[Tuple[str, float]]]:
    """Run BM25 for each query and return top-k per query.

    Returns a dict mapping query -> list of (candidate_id, score).
    """
    bm25, _ = bm25_obj or build_bm25()
    df = load_or_build_parquet()
    if candidate_ids is None:
        candidate_ids = df["candidate_id"].tolist()
    n_docs = len(candidate_ids)
    out: Dict[str, List[Tuple[str, float]]] = {}
    for q in queries:
        qtoks = _query_tokens(q)
        if not qtoks:
            out[q] = []
            continue
        scores = bm25.get_scores(qtoks)
        # Top-k by argpartition (O(n)) + small sort
        if top_k >= n_docs:
            order = np.argsort(-scores)
        else:
            idx = np.argpartition(-scores, top_k)[:top_k]
            order = idx[np.argsort(-scores[idx])]
        out[q] = [(candidate_ids[int(i)], float(scores[int(i)])) for i in order]
    return out


if __name__ == "__main__":
    bm25, toks = build_bm25(force=False)
    print(f"BM25 docs={len(toks):,}")
    res = bm25_search(["PyTorch transformers LoRA RAG vector search"], top_k=10)
    for q, hits in res.items():
        print(q)
        for cid, sc in hits[:10]:
            print(f"  {cid}\t{sc:.3f}")
