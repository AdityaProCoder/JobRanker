"""Reciprocal Rank Fusion of multiple ranked lists."""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Tuple


def rrf_fuse(
    rankings: List[List[Tuple[str, float]]],
    k: int = 60,
    top_n: int | None = None,
) -> List[Tuple[str, float, Dict[str, int]]]:
    """Combine multiple ranked lists using RRF.

    Each ranking is a list of (id, score). Higher score is better.
    Returns a single fused list of (id, rrf_score, per_channel_rank).
    """
    fused: Dict[str, float] = defaultdict(float)
    ranks: Dict[str, Dict[str, int]] = defaultdict(dict)
    for ch_idx, ranking in enumerate(rankings):
        ch_name = f"ch{ch_idx}"
        for r, (cid, _score) in enumerate(ranking, start=1):
            fused[cid] += 1.0 / (k + r)
            ranks[cid][ch_name] = r
    items = sorted(fused.items(), key=lambda x: -x[1])
    out: List[Tuple[str, float, Dict[str, int]]] = []
    for cid, s in items:
        out.append((cid, float(s), ranks[cid]))
        if top_n is not None and len(out) >= top_n:
            break
    return out


if __name__ == "__main__":
    a = [("A", 1.0), ("B", 0.9), ("C", 0.8)]
    b = [("B", 0.95), ("A", 0.7), ("D", 0.5)]
    fused = rrf_fuse([a, b], k=60)
    for cid, s, r in fused:
        print(cid, s, r)
