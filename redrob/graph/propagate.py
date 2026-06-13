"""Personalised PageRank over the heterogeneous graph, seeded at JD skills."""
from __future__ import annotations

from typing import Dict, Iterable, List

import networkx as nx
import numpy as np

from .. import config
from .build import build_graph


# Normalise core competencies into the same node-label scheme the graph uses.
def _seed_skill_nodes(jd_terms: Iterable[str]) -> List[str]:
    out: List[str] = []
    for term in jd_terms:
        # exact match against "skill::<Term>"
        key = "skill::" + term.strip()
        out.append(key)
    return out


def personalised_pagerank(
    jd_terms: Iterable[str] | None = None,
    alpha: float = 0.85,
    max_iter: int = 60,
) -> Dict[str, float]:
    """Return a dict of {candidate_node: ppr_score}."""
    G = build_graph()
    seeds = _seed_skill_nodes(jd_terms or config.CORE_COMPETENCIES)
    seeds = [s for s in seeds if s in G]
    if not seeds:
        return {}
    personalization = {n: 0.0 for n in G.nodes()}
    for s in seeds:
        personalization[s] = 1.0 / len(seeds)
    pr = nx.pagerank(
        G,
        alpha=alpha,
        personalization=personalization,
        max_iter=max_iter,
        tol=1e-6,
        weight="weight",
    )
    # Return only candidate nodes
    return {n: float(v) for n, v in pr.items() if isinstance(n, str) and n.startswith("cand::")}


if __name__ == "__main__":
    ppr = personalised_pagerank()
    top = sorted(ppr.items(), key=lambda x: -x[1])[:10]
    for n, s in top:
        print(n, round(s, 5))
