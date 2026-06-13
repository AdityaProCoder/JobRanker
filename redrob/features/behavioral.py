"""Behavioral recruitability features from the 23 Redrob signals."""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd


def _to_dt(s):
    if not s:
        return None
    try:
        return datetime.strptime(str(s)[:10], "%Y-%m-%d")
    except Exception:
        return None


def recruitability_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute a 0..1 recruitability composite plus sub-features."""
    n = len(df)
    sig_col = "signals"
    if sig_col not in df.columns:
        return pd.DataFrame({
            "recruit_open_to_work": np.zeros(n),
            "recruit_response_rate": np.zeros(n),
            "recruit_verified": np.zeros(n),
            "recruit_completeness": np.zeros(n),
            "recruit_recency": np.zeros(n),
            "recruit_notice_ok": np.zeros(n),
            "recruit_recruiter_saves": np.zeros(n),
            "recruit_interview_completion": np.zeros(n),
            "recruit_offer_acceptance": np.zeros(n),
            "recruit_github": np.zeros(n),
            "recruitability": np.zeros(n),
        })

    def get(row, key, default=0):
        d = row.get(sig_col)
        if d is None:
            d = {}
        elif hasattr(d, "tolist"):
            d = d.tolist() if d else {}
        if not isinstance(d, dict):
            return default
        v = d.get(key, default)
        return v if v is not None else default

    open_to_work = np.array([float(get(r, "open_to_work_flag", 0)) for _, r in df.iterrows()])
    response_rate = np.array([float(get(r, "recruiter_response_rate", 0.0)) for _, r in df.iterrows()])
    verified_email = np.array([float(get(r, "verified_email", 0)) for _, r in df.iterrows()])
    verified_phone = np.array([float(get(r, "verified_phone", 0)) for _, r in df.iterrows()])
    linkedin = np.array([float(get(r, "linkedin_connected", 0)) for _, r in df.iterrows()])
    completeness = np.array([float(get(r, "profile_completeness_score", 0.0)) for _, r in df.iterrows()]) / 100.0
    notice = np.array([float(get(r, "notice_period_days", 90)) for _, r in df.iterrows()])
    saves = np.array([float(get(r, "saved_by_recruiters_30d", 0)) for _, r in df.iterrows()])
    interview_completion = np.array([float(get(r, "interview_completion_rate", 0.0)) for _, r in df.iterrows()])
    offer_acceptance = np.array([float(get(r, "offer_acceptance_rate", -1.0)) for _, r in df.iterrows()])
    github = np.array([float(get(r, "github_activity_score", -1.0)) for _, r in df.iterrows()])

    # Recency: anchored to fixed reference date for determinism.
    # Reference = 2026-06-01 (near dataset creation date).
    # 90-day half-life exponential decay.
    from datetime import datetime as _dt
    _REF_DATE = _dt(2026, 6, 1)
    recency = np.zeros(n)
    for i, (_, r) in enumerate(df.iterrows()):
        d = _to_dt(get(r, "last_active_date", None))
        if d is None:
            recency[i] = 0.0
            continue
        days = (_REF_DATE - d).days
        import math
        recency[i] = max(0.0, math.exp(-max(0, days) / 90.0))

    # Notice period
    notice_ok = np.where(notice <= 30, 1.0, np.maximum(0.0, 1.0 - (notice - 30) / 150.0))
    recruiter_saves = np.minimum(saves / 20.0, 1.0)
    offer_acc = np.where(offer_acceptance < 0, 0.3, offer_acceptance)
    offer_acc = np.clip(offer_acc, 0.0, 1.0)
    github_norm = np.where(github < 0, 0.0, github / 100.0)
    verified = (verified_email + verified_phone + linkedin) / 3.0

    composite = (
        0.18 * open_to_work
        + 0.10 * verified
        + 0.10 * response_rate
        + 0.06 * interview_completion
        + 0.06 * offer_acc
        + 0.10 * completeness
        + 0.08 * recency
        + 0.10 * notice_ok
        + 0.10 * recruiter_saves
        + 0.06 * github_norm
        + 0.06 * open_to_work * verified  # bonus: open AND verified
    )
    composite = np.clip(composite, 0.0, 1.0)

    return pd.DataFrame({
        "recruit_open_to_work": open_to_work,
        "recruit_response_rate": response_rate,
        "recruit_verified": verified,
        "recruit_completeness": completeness,
        "recruit_recency": recency,
        "recruit_notice_ok": notice_ok,
        "recruit_recruiter_saves": recruiter_saves,
        "recruit_interview_completion": interview_completion,
        "recruit_offer_acceptance": offer_acc,
        "recruit_github": github_norm,
        "recruitability": composite,
    })
