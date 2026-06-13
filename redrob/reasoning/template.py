"""Deterministic reasoning template generator."""
from __future__ import annotations

import math
from typing import Dict, List

import numpy as np
import pandas as pd

from .. import config


# ---------------------------------------------------------------------------
# Phrase banks
# ---------------------------------------------------------------------------

ROLE_PHRASES = {
    "applied_ml": "applied ML",
    "retrieval_ranking": "retrieval / ranking",
    "nlp_llm": "NLP / LLM",
    "data_science": "data science",
    "data_platform": "data platform",
    "generic_swe": "software engineering",
    "non_target": "non-engineering",
}


def _fmt_skill(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return ""
    if "/" in s:
        return s
    if " " in s:
        return s
    return s


def _top_skills(df_row: pd.Series, k: int = 4) -> List[str]:
    """Pick the top-k skills that intersect the JD core, falling back to
    whatever advanced skills the candidate has."""
    skills_full = df_row.get("skills")
    if skills_full is None:
        skills_full = []
    elif hasattr(skills_full, "tolist"):
        skills_full = skills_full.tolist()
    core_lc = {s.lower() for s in config.CORE_COMPETENCIES}
    adj_lc = {s.lower() for s in config.ADJACENT_COMPETENCIES}
    ranked: List[tuple] = []
    for s in skills_full:
        if not isinstance(s, dict):
            continue
        n = (s.get("name") or "").strip()
        if not n:
            continue
        nl = n.lower()
        if nl in core_lc:
            ranked.append((0, n))
        elif nl in adj_lc:
            ranked.append((1, n))
        else:
            prof = s.get("proficiency", "")
            if prof in ("expert", "advanced"):
                ranked.append((2, n))
    ranked.sort()
    return [n for _, n in ranked[:k]]


def _assessment_str(df_row: pd.Series) -> str:
    sig = df_row.get("signals")
    if sig is None:
        sig = {}
    elif hasattr(sig, "tolist"):
        sig = sig.tolist() if sig else {}
    if not isinstance(sig, dict):
        return ""
    sas = sig.get("skill_assessment_scores") or {}
    if not sas:
        return ""
    vals = [(k, v) for k, v in sas.items() if v is not None]
    vals = [(k, v) for k, v in vals if isinstance(v, (int, float))]
    if not vals:
        return ""
    top = sorted(vals, key=lambda x: -x[1])[:3]
    return ", ".join(f"{k}={int(v)}" for k, v in top)


def _yoe_band(yoe: float) -> str:
    if yoe < config.YOE_IDEAL_LOW:
        return f"{yoe:.1f} yrs (below ideal band {config.YOE_IDEAL_LOW}-{config.YOE_IDEAL_HIGH})"
    if yoe > config.YOE_IDEAL_HIGH:
        return f"{yoe:.1f} yrs (above ideal band {config.YOE_IDEAL_LOW}-{config.YOE_IDEAL_HIGH})"
    return f"{yoe:.1f} yrs (in ideal {config.YOE_IDEAL_LOW}-{config.YOE_IDEAL_HIGH} band)"


def _location_str(df_row: pd.Series) -> str:
    loc = (df_row.get("location") or "").strip()
    country = (df_row.get("country") or "").strip()
    if loc and country:
        return f"{loc}, {country}"
    if loc:
        return loc
    if country:
        return country
    return "unspecified location"


def _company_str(df_row: pd.Series) -> str:
    cur = (df_row.get("current_company") or "").strip()
    if not cur:
        return "current company not listed"
    return f"currently at {cur}"


def _signals_brief(df_row: pd.Series) -> str:
    sig = df_row.get("signals")
    if sig is None:
        sig = {}
    elif hasattr(sig, "tolist"):
        sig = sig.tolist() if sig else {}
    if not isinstance(sig, dict):
        return ""
    parts = []
    if sig.get("open_to_work_flag"):
        parts.append("open-to-work")
    if sig.get("verified_email") and sig.get("verified_phone"):
        parts.append("verified")
    notice = sig.get("notice_period_days")
    if notice is not None and notice <= 30:
        parts.append(f"notice {notice}d")
    elif notice is not None and notice >= 90:
        parts.append(f"notice {notice}d (extended)")
    rr = sig.get("recruiter_response_rate")
    if isinstance(rr, (int, float)) and rr > 0:
        parts.append(f"response {rr:.0%}")
    return ", ".join(parts)


# ---------------------------------------------------------------------------
# Top-level reasoner
# ---------------------------------------------------------------------------

def _choose_template(rank: int, n: int, df_row: pd.Series) -> List[str]:
    """Pick a template family by rank and candidate characteristics."""
    is_top = rank <= 10
    is_mid = 10 < rank <= 60
    title_fit = float(df_row.get("title_fit_blend") or 0.0)
    rec = float(df_row.get("recruitability") or 0.0)
    honeypot = float(df_row.get("honeypot_penalty") or 0.0)
    yoe = float(df_row.get("yoe") or 0.0)
    notice = int((df_row.get("notice_period_days") or 0))

    out: List[str] = []

    # Clause 1: who they are and why this role
    if title_fit >= 0.7 and yoe >= config.YOE_IDEAL_LOW:
        out.append(
            f"Senior AI Engineer fit: {_yoe_band(yoe)}; current title "
            f"'{df_row.get('current_title')}' maps cleanly to the JD's applied-ML "
            f"archetype."
        )
    elif title_fit >= 0.55:
        out.append(
            f"Strong adjacent fit: '{df_row.get('current_title')}' with career "
            f"trajectory in {ROLE_PHRASES.get('data_science', 'data')}/ML — "
            f"covers most of the JD's core requirements."
        )
    elif title_fit >= 0.30:
        out.append(
            f"Adjacent fit: '{df_row.get('current_title')}' with demonstrable "
            f"applied-ML work in career history; the JD explicitly welcomes "
            f"people who can grow into the role."
        )
    else:
        out.append(
            f"Lower-confidence fit: '{df_row.get('current_title')}' — kept in "
            f"the top-100 as a long-shot given the title–skill gap."
        )

    # Clause 2: specific evidence
    skills = _top_skills(df_row, k=4)
    if skills:
        out.append(
            "Direct match on JD skills: " + ", ".join(_fmt_skill(s) for s in skills) + "."
        )
    assess = _assessment_str(df_row)
    if assess:
        out.append(f"Redrob assessment scores: {assess}.")
    ppr = float(df_row.get("ppr_score") or 0.0)
    if ppr > 0 and not skills:
        out.append(
            f"Sits in a JD-aligned skill community in the candidate-skill graph."
        )

    # Clause 3: concern / honest caveat
    if notice >= 90:
        out.append(
            f"Honest concern: {notice}-day notice period — above the JD's "
            f"30-day preference, raises the bar."
        )
    elif notice >= 30:
        out.append(
            f"Notice period {notice} days — within JD tolerance, may delay start."
        )
    if honeypot >= 0.25:
        out.append(
            f"Profile shows some inconsistency signals (honeypot penalty "
            f"{honeypot:.2f}); reviewer should verify the listed skills/dates."
        )
    if rec < 0.3:
        out.append(
            f"Behavioral signals are muted (recruitability {rec:.2f}); verify "
            f"interest before outreach."
        )

    return out


def generate_reasoning(df_row: pd.Series, rank: int, n_total: int = 100) -> str:
    """Compose a 1–3 sentence reasoning for one candidate.

    `rank` and `n_total` allow the wording to vary with rank position so
    that the reasoning column is not a single template.
    """
    clauses = _choose_template(rank, n_total, df_row)
    # Ensure at least 2 clauses
    if len(clauses) < 2:
        clauses.append(_location_str(df_row))
    # Truncate to first 3 clauses
    text = " ".join(clauses[:3])
    # Ensure no embedded newlines (validator tolerates them but cleaner)
    text = text.replace("\n", " ").replace("\r", " ")
    # CSV-safe: drop any double quotes (we'll wrap in CSV writer with quoting)
    text = text.replace('"', "'")
    return text


def generate_reasoning_dataframe(df: pd.DataFrame) -> List[str]:
    """Return a list of reasoning strings, one per row, in input order."""
    out: List[str] = []
    n = len(df)
    for i in range(n):
        row = df.iloc[i]
        rank = i + 1
        out.append(generate_reasoning(row, rank, n))
    return out
