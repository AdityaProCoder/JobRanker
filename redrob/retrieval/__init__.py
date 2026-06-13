"""Hybrid retrieval layer: BM25 + sentence-transformers + structured gate + RRF."""
from .bm25 import build_bm25, bm25_search
from .dense import build_dense_index, dense_search, dense_model_name
from .rrf import rrf_fuse

__all__ = [
    "build_bm25", "bm25_search",
    "build_dense_index", "dense_search", "dense_model_name",
    "rrf_fuse",
]
