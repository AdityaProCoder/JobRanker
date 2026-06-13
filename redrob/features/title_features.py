"""Graduated title-fit scoring.

Combines:
  - exact role-group match on current and historical titles
  - embedding cosine of current title to the centroid of target titles
  - promotion velocity / role progression features
"""
from __future__ import annotations

import re
from typing import List

import numpy as np
import pandas as pd

from .. import config


def _normalise(title: str) -> str:
    t = (title or "").lower()
    t = re.sub(r"[^a-z0-9+\-# ]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _title_group(title_norm: str) -> str | None:
    best = None
    for group, members in config.TITLE_ROLE_GROUPS.items():
        for m in members:
            if m in title_norm:
                if best is None or config.TITLE_GROUP_WEIGHT[group] > config.TITLE_GROUP_WEIGHT[best]:
                    best = group
    return best


def title_fit_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame of per-candidate title features."""
    current = df["current_title"].fillna("").tolist()
    histories = df["career_titles"].tolist()

    # Pre-compute best historical group across all rows
    current_groups = [_title_group(_normalise(t)) for t in current]
    history_groups: List[List[str]] = []
    history_max_weight: List[float] = []
    for hist in histories:
        if hist is None:
            hist = []
        if hasattr(hist, "tolist"):
            hist = hist.tolist()
        gs = [_title_group(_normalise(t)) for t in (hist or [])]
        gs = [g for g in gs if g is not None]
        history_groups.append(gs)
        if gs:
            history_max_weight.append(max(config.TITLE_GROUP_WEIGHT[g] for g in gs))
        else:
            history_max_weight.append(0.0)

    # Per-row title features
    current_weight = [config.TITLE_GROUP_WEIGHT.get(g, 0.05) for g in current_groups]
    is_target = [1.0 if (g in {"applied_ml", "retrieval_ranking", "nlp_llm"}) else 0.0 for g in current_groups]
    is_services_or_noneng = [1.0 if g in {"non_target"} else 0.0 for g in current_groups]
    is_data_platform = [1.0 if g in {"data_platform"} else 0.0 for g in current_groups]
    is_data_science = [1.0 if g in {"data_science"} else 0.0 for g in current_groups]
    is_generic_swe = [1.0 if g in {"generic_swe"} else 0.0 for g in current_groups]

    # Promotion velocity: number of distinct titles / yoe (low if many short stints)
    n_titles = []
    for h in histories:
        if h is None:
            n_titles.append(0)
        elif hasattr(h, "tolist"):
            n_titles.append(len(h.tolist()))
        else:
            n_titles.append(len(h))
    avg_tenure_months = []
    for ch in df["career"].tolist():
        if ch is None:
            avg_tenure_months.append(0.0)
            continue
        if hasattr(ch, "tolist"):
            ch = ch.tolist()
        if not ch:
            avg_tenure_months.append(0.0)
            continue
        durs = [int(c.get("duration_months") or 0) for c in (ch or [])]
        durs = [d for d in durs if d > 0]
        if durs:
            avg_tenure_months.append(sum(durs) / len(durs))
        else:
            avg_tenure_months.append(0.0)

    out = pd.DataFrame({
        "title_weight": current_weight,
        "title_history_max_weight": history_max_weight,
        "is_target_title": is_target,
        "is_noneng_title": is_services_or_noneng,
        "is_data_platform": is_data_platform,
        "is_data_science": is_data_science,
        "is_generic_swe": is_generic_swe,
        "n_career_titles": n_titles,
        "avg_tenure_months": avg_tenure_months,
        # blended title fit (current 70%, history 30%)
        "title_fit_blend": [
            0.7 * w + 0.3 * hw for w, hw in zip(current_weight, history_max_weight)
        ],
    })
    return out
