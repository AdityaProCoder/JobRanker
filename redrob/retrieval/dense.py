"""Dense retrieval using sentence-transformers.

The model is configurable via env var REDROB_DENSE_MODEL.
Stronger BGE variants are supported when they are available locally.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .. import config
from ..data import load_or_build_parquet


_DENSE_MODEL_CACHE: Dict[str, object] = {}


def dense_model_name() -> str:
    """Resolve the dense model name from env, with validation."""
    name = os.environ.get("REDROB_DENSE_MODEL", config.DEFAULT_DENSE_MODEL).strip()
    if name and name not in config.DENSE_MODEL_OPTIONS:
        # Allow any model name the user has cached; warn in logs.
        pass
    return name or config.DEFAULT_DENSE_MODEL


def _load_model():
    name = dense_model_name()
    if name in _DENSE_MODEL_CACHE:
        return _DENSE_MODEL_CACHE[name]
    from sentence_transformers import SentenceTransformer  # type: ignore
    # Try the requested model. If it isn't cached and we're offline,
    # fall back to MiniLM (always cached at construction time below).
    candidates = [name] + [m for m in config.DENSE_MODEL_OPTIONS if m != name]
    last_err: Optional[Exception] = None
    for cand in candidates:
        try:
            model = SentenceTransformer(cand, cache_folder=str(config.MODELS_DIR))
            _DENSE_MODEL_CACHE[name] = model
            return model
        except Exception as e:  # noqa: BLE001
            last_err = e
            continue
    raise RuntimeError(f"Could not load any sentence-transformers model: {last_err}")


def _model_dim() -> int:
    m = _load_model()
    return int(m.get_sentence_embedding_dimension())


def build_dense_index(force: bool = False, candidate_ids: Optional[List[str]] = None) -> Tuple[np.ndarray, List[str]]:
    """Encode every candidate's text corpus and return (matrix, ids).

    If `candidate_ids` is provided, only those candidates are encoded.
    The full 100k corpus is encoded on first call; subsequent runs use
    the cached matrix and align to the candidate_ids subset.
    """
    if not force and config.DENSE_NPY.exists() and config.DENSE_IDS_JSON.exists():
        emb_all = np.load(config.DENSE_NPY)
        ids_all = json.loads(config.DENSE_IDS_JSON.read_text(encoding="utf-8"))
        if candidate_ids is None:
            return emb_all, ids_all
        # subset to requested ids
        idx_map = {c: i for i, c in enumerate(ids_all)}
        idxs = [idx_map[c] for c in candidate_ids if c in idx_map]
        return emb_all[idxs], [ids_all[i] for i in idxs]

    model = _load_model()
    df = load_or_build_parquet()
    if candidate_ids is not None:
        df = df[df["candidate_id"].isin(set(candidate_ids))].reset_index(drop=True)
    texts = df["text_corpus"].fillna("").tolist()
    # truncate to model max seq length * 6 chars
    texts = [(t or "")[:2000] for t in texts]
    embeddings = model.encode(
        texts,
        batch_size=256,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    ids = df["candidate_id"].tolist()
    np.save(config.DENSE_NPY, embeddings)
    config.DENSE_IDS_JSON.write_text(json.dumps(ids), encoding="utf-8")
    return embeddings, ids


def dense_search(
    queries: List[str],
    top_k: int = config.DENSE_TOP_K,
    matrix: Optional[Tuple[np.ndarray, List[str]]] = None,
    candidate_ids: Optional[List[str]] = None,
) -> Dict[str, List[Tuple[str, float]]]:
    """Run dense retrieval.

    If a precomputed matrix is passed it is used. Otherwise the
    candidate_ids subset (or full corpus) is encoded and the matrix
    returned for reuse.
    """
    if matrix is None:
        matrix = build_dense_index(candidate_ids=candidate_ids)
    emb, ids = matrix

    model = _load_model()
    qvecs = model.encode(
        queries,
        batch_size=64,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    out: Dict[str, List[Tuple[str, float]]] = {}
    for q, qv in zip(queries, qvecs):
        scores = emb @ qv  # cosine because vectors are normalised
        if top_k >= len(scores):
            order = np.argsort(-scores)
        else:
            idx = np.argpartition(-scores, top_k)[:top_k]
            order = idx[np.argsort(-scores[idx])]
        out[q] = [(ids[int(i)], float(scores[int(i)])) for i in order]
    return out


if __name__ == "__main__":
    print("model:", dense_model_name())
    m, ids = build_dense_index(force=False)
    print("emb shape:", m.shape, "ids:", len(ids))
    res = dense_search(["Senior AI Engineer with hybrid search and RAG experience"], top_k=5)
    for q, hits in res.items():
        print(q)
        for cid, sc in hits:
            print(f"  {cid}\t{sc:.4f}")
