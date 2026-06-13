"""Personalised PageRank over the heterogeneous graph.

Supports both the original single-axis PPR and a multi-axis PPR
(5 JD axes: applied_ml, retrieval_rank, nlp_llm, production_eng, product_company).
Multi-axis results are cached on disk at artifacts/ppr_axes.pkl.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, Iterable, List

import networkx as nx

from .. import config
from .build import build_graph


def _seed_nodes(jd_terms: Iterable[str], prefix: str) -> List[str]:
    """Map JD terms to graph node labels with the given prefix."""
    out: List[str] = []
    for term in jd_terms:
        key = f"{prefix}::{term.strip()}"
        out.append(key)
    return out


def personalised_pagerank(
    jd_terms: Iterable[str] | None = None,
    alpha: float = 0.85,
    max_iter: int = 60,
) -> Dict[str, float]:
    """Return a dict of {candidate_node: ppr_score}."""
    G = build_graph()
    seeds = _seed_nodes(jd_terms or config.CORE_COMPETENCIES, "skill")
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
    return {n: float(v) for n, v in pr.items()
            if isinstance(n, str) and n.startswith("cand::")}


# Disk cache path for multi-axis PPR
_PPR_AXES_CACHE = config.ARTIFACTS_DIR / "ppr_axes.pkl"


def ppr_axes(
    axes: Dict[str, List[str]] | None = None,
    alpha: float = 0.85,
    max_iter: int = 60,
) -> Dict[str, Dict[str, float]]:
    """Multi-axis PPR: one PPR per JD axis.

    Returns Dict[axis_name, Dict[cand_node, ppr_score]].
    Results are cached at artifacts/ppr_axes.pkl.

    Axes:
      applied_ml      -> seeds from config.JD_AXES["applied_ml"] on skill:: nodes
      retrieval_rank  -> seeds from config.JD_AXES["retrieval_rank"] on skill:: nodes
      nlp_llm         -> seeds from config.JD_AXES["nlp_llm"] on skill:: nodes
      production_eng   -> seeds from config.JD_AXES["production_eng"] on skill:: nodes
      product_company  -> seeds from config.JD_AXES["product_company"] on company:: nodes
    """
    if axes is None:
        axes = config.JD_AXES

    # Load cache if present
    if _PPR_AXES_CACHE.exists():
        with open(_PPR_AXES_CACHE, "rb") as f:
            cached: Dict[str, Dict[str, float]] = pickle.load(f)
        # Validate all axes are present
        if all(ax in cached for ax in axes):
            return cached

    G = build_graph()
    result: Dict[str, Dict[str, float]] = {}

    for axis_name, terms in axes.items():
        # product_company axis uses company:: nodes; all others use skill::
        prefix = "company" if axis_name == "product_company" else "skill"
        seeds = _seed_nodes(terms, prefix)
        seeds = [s for s in seeds if s in G]
        if not seeds:
            result[axis_name] = {}
            print(f"  ppr_axes[{axis_name}]: 0 seeds found in graph")
            continue
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
        result[axis_name] = {
            n: float(v)
            for n, v in pr.items()
            if isinstance(n, str) and n.startswith("cand::")
        }
        top_score = max(result[axis_name].values()) if result[axis_name] else 0.0
        print(f"  ppr_axes[{axis_name}]: {len(seeds)} seeds, top_score={top_score:.6f}")

    # Persist to disk
    with open(_PPR_AXES_CACHE, "wb") as f:
        pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)

    return result


if __name__ == "__main__":
    axes = ppr_axes()
    for name, scores in axes.items():
        top = sorted(scores.items(), key=lambda x: -x[1])[:3]
        print(f"{name}: {top}")
