"""Skill-related features: must-have coverage, assessments, durations."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config


_CORE = {s.lower() for s in config.CORE_COMPETENCIES}
_ADJ = {s.lower() for s in config.ADJACENT_COMPETENCIES}
_NEG = {s.lower() for s in config.NEGATIVE_COMPETENCIES}


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

    for i in range(n):
        skills_full = skills_full_col[i]
        if skills_full is None:
            skills_full = []
        names = [s.get("name", "").strip() for s in skills_full if s.get("name")]
        names_lc = {n_.lower() for n_ in names if n_}
        skill_count[i] = len(names)
        n_core_hits[i] = len(names_lc & _CORE)
        n_adj_hits[i] = len(names_lc & _ADJ)
        n_neg_hits[i] = len(names_lc & _NEG)
        n_advanced[i] = sum(1 for s in skills_full if s.get("proficiency") in ("advanced",))
        n_expert[i] = sum(1 for s in skills_full if s.get("proficiency") == "expert")

        # Assessments
        sig = sigs_col[i]
        if sig is None:
            sig = {}
        if isinstance(sig, dict):
            sas = sig.get("skill_assessment_scores") or {}
            if sas:
                vals = [float(v) for v in sas.values() if v is not None]
                if vals:
                    assessment_max[i] = max(vals)
                    assessment_mean[i] = sum(vals) / len(vals)
        # Endorsement / duration log means
        e_vals = [int(s.get("endorsements") or 0) for s in skills_full]
        d_vals = [int(s.get("duration_months") or 0) for s in skills_full]
        if e_vals:
            endorsement_log_mean[i] = float(np.mean(np.log1p(e_vals)))
        if d_vals:
            duration_log_mean[i] = float(np.mean(np.log1p(d_vals)))

        # Jaccard of candidate's skills with core
        if names_lc and _CORE:
            inter = len(names_lc & _CORE)
            union = len(names_lc | _CORE)
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
    })
