"""Role Blueprint extraction.

The blueprint is largely static (already curated in `config.py`) but the
public entry point also writes `artifacts/blueprint.json` for the sandbox
to display.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List

from . import config


def build_blueprint() -> Dict[str, Any]:
    """Return the role blueprint as a plain dict (also JSON-serializable)."""
    bp: Dict[str, Any] = {
        "role": "Senior AI Engineer — Redrob AI (Series A)",
        "company": "Redrob AI",
        "location_preference": "Pune / Noida (Hybrid)",
        "country": "India",
        "experience_band": {
            "min": config.YOE_MIN,
            "ideal_low": config.YOE_IDEAL_LOW,
            "ideal_high": config.YOE_IDEAL_HIGH,
            "max_useful": config.YOE_MAX_USEFUL,
        },
        "core_competencies": sorted(config.CORE_COMPETENCIES),
        "adjacent_competencies": sorted(config.ADJACENT_COMPETENCIES),
        "negative_competencies": sorted(config.NEGATIVE_COMPETENCIES),
        "target_titles": sorted(config.TARGET_TITLES),
        "title_role_groups": config.TITLE_ROLE_GROUPS,
        "title_group_weight": config.TITLE_GROUP_WEIGHT,
        "notice_period_ok_days": config.NOTICE_OK_DAYS,
        "notice_period_hard_days": config.NOTICE_HARD_DAYS,
        "preferred_locations": sorted(config.PREFERRED_LOCATIONS),
        "country_ok": sorted(config.COUNTRY_OK),
        "it_services_pure_play": sorted(config.IT_SERVICES_PURE_PLAY),
        "product_company_hints": sorted(config.PRODUCT_COMPANY_HINTS),
        "query_terms": list(config.JD_QUERY_TERMS),
        "structured_gate_terms": sorted(config.STRUCTURED_GATE_TERMS),
    }
    return bp


def save_blueprint(path=None) -> Dict[str, Any]:
    bp = build_blueprint()
    out = config.BLUEPRINT_JSON if path is None else path
    with open(out, "w", encoding="utf-8") as f:
        json.dump(bp, f, indent=2, ensure_ascii=False)
    return bp


if __name__ == "__main__":
    save_blueprint()
    print(f"blueprint -> {config.BLUEPRINT_JSON}")
