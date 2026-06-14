"""Explicit honeypot detection.

We do not learn the contradictions. We encode ten hard rules and
combine them into a single penalty ∈ [0, 1].
"""
from __future__ import annotations

import re
from datetime import datetime

import numpy as np
import pandas as pd

from .. import config


_SKILL_AI = re.compile(
    r"\b(pytorch|tensorflow|transformers?|hugging|llm|rag|retrieval|recommend"
    r"|ranking|ranker|search|embedding|vector|faiss|pinecone|weaviate|qdrant"
    r"|milvus|opensearch|elasticsearch|bm25|xgboost|lightgbm|catboost"
    r"|learning[\s-]?to[\s-]?rank|lambdarank|lora|qlora|peft|fine[\s-]?tun"
    r"|sentence[\s-]?transformer|bge|e5|mlops|kubeflow|mlflow|triton|onnx)\b",
    re.IGNORECASE,
)


def _is_ai_skill(name: str) -> bool:
    return bool(_SKILL_AI.search(name or ""))


def honeypot_features(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    p = np.zeros(n)
    rule_flags = {f"hp_rule_{i}": np.zeros(n) for i in range(10)}

    # Pre-extract per-row arrays
    skill_names = []
    for x in df["skill_names"].tolist():
        if x is None:
            skill_names.append([])
        elif hasattr(x, "tolist"):
            skill_names.append(x.tolist())
        else:
            skill_names.append(x)
    skills_full = []
    for x in df["skills"].tolist():
        if x is None:
            skills_full.append([])
        elif hasattr(x, "tolist"):
            skills_full.append(x.tolist())
        else:
            skills_full.append(x)
    yoe = df["yoe"].astype(float).to_numpy()
    career_total = df["career_total_months"].astype(float).to_numpy() / 12.0
    overlap = df["career_overlap_months"].astype(float).to_numpy()
    edu = []
    for x in df["edu"].tolist():
        if x is None:
            edu.append([])
        elif hasattr(x, "tolist"):
            edu.append(x.tolist())
        else:
            edu.append(x)
    sigs = []
    for s in df["signals"].tolist():
        if s is None:
            sigs.append({})
        elif hasattr(s, "tolist"):
            sigs.append(s.tolist() if s else {})
        else:
            sigs.append(s)
    profile_completeness = np.array([
        (s or {}).get("profile_completeness_score", 50.0) if isinstance(s, dict) else 50.0
        for s in sigs
    ])
    salary_min = np.array([
        ((s or {}).get("expected_salary_range_inr_lpa") or {}).get("min", 0.0) if isinstance(s, dict) else 0.0
        for s in sigs
    ])
    salary_max = np.array([
        ((s or {}).get("expected_salary_range_inr_lpa") or {}).get("max", 0.0) if isinstance(s, dict) else 0.0
        for s in sigs
    ])
    search_appearance = np.array([
        (s or {}).get("search_appearance_30d", 0) if isinstance(s, dict) else 0
        for s in sigs
    ])
    saved_by_recruiters = np.array([
        (s or {}).get("saved_by_recruiters_30d", 0) if isinstance(s, dict) else 0
        for s in sigs
    ])

    for i in range(n):
        contrib = 0.0
        _skf = skills_full[i]
        if _skf is None:
            _skf = []
        elif hasattr(_skf, "tolist"):
            _skf = _skf.tolist()
        _sn = skill_names[i]
        if _sn is None:
            _sn = []
        elif hasattr(_sn, "tolist"):
            _sn = _sn.tolist()

        # Rule 1: expert + duration_months < 6 (capped)
        cnt = 0
        for s in _skf:
            if s.get("proficiency") == "expert" and (s.get("duration_months") or 0) < 6 and _is_ai_skill(s.get("name", "")):
                cnt += 1
        rule_flags["hp_rule_0"][i] = min(cnt * 0.25, 0.5)
        contrib += rule_flags["hp_rule_0"][i]

        # Rule 2: advanced/expert + 0 months
        cnt2 = 0
        for s in _skf:
            if s.get("proficiency") in ("advanced", "expert") and (s.get("duration_months") or 0) == 0 and _is_ai_skill(s.get("name", "")):
                cnt2 += 1
        rule_flags["hp_rule_1"][i] = min(cnt2 * 0.15, 0.3)
        contrib += rule_flags["hp_rule_1"][i]

        # Rule 3: career duration > YOE + 1.5 years
        if yoe[i] > 0 and career_total[i] > yoe[i] + 1.5:
            rule_flags["hp_rule_2"][i] = 0.25
            contrib += 0.25

        # Rule 4: concurrent overlap > 12 months
        if overlap[i] >= 12:
            rule_flags["hp_rule_3"][i] = 0.20
            contrib += 0.20

        # Rule 5: education anomalies
        bad_edu = 0
        _edu = edu[i]
        if _edu is None:
            _edu = []
        elif hasattr(_edu, "tolist"):
            _edu = _edu.tolist()
        for e in _edu:
            sy = int(e.get("start_year") or 0)
            ey = int(e.get("end_year") or 0)
            if ey and sy and ey < sy:
                bad_edu += 1
            elif sy and ey:
                deg = (e.get("degree") or "").lower()
                expected = 2 if "phd" in deg or "doctor" in deg else (1 if "master" in deg or "m.sc" in deg or "m.s." in deg or "m.e" in deg or "m.tech" in deg or "mba" in deg else 4)
                if (ey - sy) < expected - 1 and ey >= 2020:
                    bad_edu += 1
        if bad_edu:
            rule_flags["hp_rule_4"][i] = min(bad_edu * 0.10, 0.20)
            contrib += rule_flags["hp_rule_4"][i]

        # Rule 6: many advanced skills with low YOE
        adv = sum(1 for s in _skf if s.get("proficiency") in ("advanced", "expert"))
        if adv >= 6 and yoe[i] < 3:
            rule_flags["hp_rule_5"][i] = 0.25
            contrib += 0.25

        # Rule 7 v2: title-skill contradiction (loosened).
        # Fires when current_title is non-eng AND
        #   (a) every historical title is non-eng OR
        #   (b) max assessment >= 80 (unrealistic for non-eng) OR
        #   (c) career_total_months < yoe * 12 * 0.85 (implausibly short career).
        if adv >= 5:
            non_eng_kw = (
                "marketing", "hr ", "accountant", "sales ", "graphic", "content writer",
                "civil", "mechanical", "customer support", "operations manager",
                "project manager", "business analyst", "qa ", "frontend", "java developer",
                ".net", "mobile developer", "devops", "cloud engineer", "data analyst",
            )
            cur = (df.iloc[i]["current_title"] or "").lower()
            hist_titles = df.iloc[i]["career_titles"]
            if hist_titles is None:
                hist_titles = []
            elif hasattr(hist_titles, "tolist"):
                hist_titles = hist_titles.tolist()
            non_eng_cur = any(k in cur for k in non_eng_kw)
            all_non_eng_hist = all(any(k in (t or "").lower() for k in non_eng_kw)
                                    for t in hist_titles if t)
            # Compute max assessment score
            max_assess = 0
            sig = sigs[i]
            if isinstance(sig, dict):
                sas = sig.get("skill_assessment_scores") or {}
                for v in (sas or {}).values():
                    if isinstance(v, (int, float)) and v > max_assess:
                        max_assess = v
            # Career-impossible
            impossible_career = (
                yoe[i] > 0 and career_total[i] < yoe[i] * 0.85
            )
            if non_eng_cur and (all_non_eng_hist or max_assess >= 80 or impossible_career):
                rule_flags["hp_rule_6"][i] = 0.30
                contrib += 0.30

        # Rule 8: profile stub with rich skills
        if profile_completeness[i] < 35 and adv >= 8:
            rule_flags["hp_rule_7"][i] = 0.15
            contrib += 0.15

        # Rule 9: salary inversion
        if salary_min[i] > salary_max[i] and salary_min[i] > 0:
            rule_flags["hp_rule_8"][i] = 0.10
            contrib += 0.10

        # Rule 10: signals fakery
        if search_appearance[i] > 500 and saved_by_recruiters[i] == 0:
            rule_flags["hp_rule_9"][i] = 0.10
            contrib += 0.10

        p[i] = min(contrib, 1.0)

    out = pd.DataFrame({"honeypot_penalty": p})
    for k, v in rule_flags.items():
        out[k] = v
    return out
