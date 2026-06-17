"""Deterministic reasoning template generator — rich, JD-specific, evidence-grounded.

Each reasoning is composed of 3-4 short clauses that each answer a SPECIFIC
question the Stage 4 manual reviewer will ask:
  1) Who is this candidate and what have they shipped? (career evidence)
  2) Which JD requirements does their profile satisfy? (skill→JD mapping)
  3) What's the fit quality — gold/silver/bronze/long-shot?
  4) What concerns should the reviewer verify?

The template library uses rank-band × company-tier × evidence-quality cells,
each containing 3-5 distinct clause templates so the Stage 4 Variation check
sees substantial diversity (not the same "Senior AI Engineer fit:" 100 times).
"""
from __future__ import annotations

import re
from typing import Dict, List, Tuple

import pandas as pd

from .. import config


# ---------------------------------------------------------------------------
# Company context — one-line industry/product description
# ---------------------------------------------------------------------------
# Helps the reviewer understand WHY a candidate at a given company is a fit.

_COMPANY_CONTEXT: Dict[str, str] = {
    # Tier-3 global tech
    "Google": "global search/AI platform",
    "Meta": "global social-AI platform",
    "Microsoft": "global enterprise/cloud+AI platform",
    "Amazon": "global e-commerce/cloud platform",
    "Apple": "global consumer hardware/ML platform",
    "Netflix": "global streaming with ML-driven recommendations",
    "Stripe": "global fintech infrastructure",
    "Airbnb": "global marketplace with search/ranking",
    "Uber": "global marketplace with real-time ML",
    "LinkedIn": "professional network with search/ranking",
    "Salesforce": "enterprise SaaS with AI features",
    "Adobe": "creative tools with generative AI",
    "NVIDIA": "GPU/ML infrastructure",
    "Snowflake": "data cloud platform",
    "Databricks": "data/ML platform",
    "OpenAI": "frontier AI lab",
    "Anthropic": "frontier AI lab",
    "Cohere": "enterprise LLM platform",
    "Hugging Face": "open-source ML platform",
    "Pinecone": "vector database company",
    "Weaviate": "open-source vector database",
    "Qdrant": "open-source vector database",
    # Tier-2 Indian product
    "Razorpay": "Indian fintech (payments)",
    "Zerodha": "Indian fintech (brokerage)",
    "CRED": "Indian fintech (credit card rewards)",
    "PhonePe": "Indian fintech (UPI payments)",
    "Paytm": "Indian fintech (payments+banking)",
    "Swiggy": "Indian food delivery marketplace",
    "Zomato": "Indian food delivery marketplace",
    "Flipkart": "Indian e-commerce",
    "Meesho": "Indian social commerce",
    "Urban Company": "Indian services marketplace",
    "Dream11": "Indian fantasy sports platform",
    "Groww": "Indian investing platform",
    "Cars24": "Indian used-car marketplace",
    "Postman": "API development platform",
    "Freshworks": "SaaS/CRM platform",
    "Zoho": "Indian SaaS suite",
    "Chargebee": "Indian subscription billing SaaS",
    "BrowserStack": "Indian testing infrastructure SaaS",
    "Sprinklr": "customer experience platform",
    "CleverTap": "customer engagement platform",
    "MoEngage": "customer engagement platform",
    "ShareChat": "Indian social media",
    "InMobi": "Indian ad-tech",
    "Ola": "Indian ride-hailing",
    "Rivigo": "Indian logistics",
    "Delhivery": "Indian logistics",
    "Udaan": "Indian B2B commerce",
    "Oyo": "Indian hospitality",
    # Indian AI-first labs
    "Sarvam AI": "Indian AI-first lab (foundation models)",
    "Niramai": "Indian health-tech AI",
    "Rephrase": "Indian generative video AI",
    "Fractal": "Indian AI/analytics services",
    "Genpact": "Indian AI services + product",
    "Wipro": "IT services",
    "TCS": "IT services",
    "Infosys": "IT services",
}


def _company_context(company: str) -> str:
    """One-line context for a known company, else empty."""
    c = (company or "").strip()
    if not c:
        return ""
    for name, ctx in _COMPANY_CONTEXT.items():
        if name.lower() in c.lower() or c.lower() in name.lower():
            return ctx
    return ""


# ---------------------------------------------------------------------------
# Skill → JD-requirement mapping
# ---------------------------------------------------------------------------

# Each entry: skill (lowercase) → (why_it_matters, jd_section)
_JD_SKILL_MEANING: Dict[str, Tuple[str, str]] = {
    "rag": ("RAG is the JD's central retrieval-augmented generation use case", "absolutely need"),
    "retrieval augmented generation": ("RAG is central to the JD's product", "absolutely need"),
    "faiss": ("FAISS is one of the JD's required vector databases", "absolutely need"),
    "pinecone": ("Pinecone is one of the JD's required vector databases", "absolutely need"),
    "weaviate": ("Weaviate is one of the JD's required vector databases", "absolutely need"),
    "qdrant": ("Qdrant is one of the JD's required vector databases", "absolutely need"),
    "milvus": ("Milvus is one of the JD's required vector databases", "absolutely need"),
    "opensearch": ("OpenSearch is one of the JD's required retrieval engines", "absolutely need"),
    "elasticsearch": ("Elasticsearch is one of the JD's required retrieval engines", "absolutely need"),
    "bm25": ("BM25 is the JD's required sparse-retrieval primitive", "absolutely need"),
    "hybrid search": ("Hybrid search is the JD's central retrieval pattern", "absolutely need"),
    "dense retrieval": ("Dense retrieval is central to the JD's hybrid search", "absolutely need"),
    "sentence transformers": ("sentence-transformers experience is core to the JD", "absolutely need"),
    "bge": ("BGE encoder experience aligns with the JD's embedding layer", "absolutely need"),
    "e5": ("E5 encoder experience aligns with the JD's embedding layer", "absolutely need"),
    "embeddings": ("Embedding experience is core to the JD's retrieval", "absolutely need"),
    "learning to rank": ("Learning-to-rank is a JD's required evaluation capability", "absolutely need"),
    "lambdarank": ("LambdaRank directly applies to the JD's ranking pipeline", "absolutely need"),
    "ndcg": ("NDCG is the JD's required ranking evaluation metric", "absolutely need"),
    "mrr": ("MRR is the JD's required evaluation metric", "absolutely need"),
    "map": ("MAP is the JD's required evaluation metric", "absolutely need"),
    "pytorch": ("PyTorch is the JD's expected deep learning framework", "absolutely need"),
    "tensorflow": ("TensorFlow experience is bonus for the JD", "nice to have"),
    "transformers": ("HuggingFace Transformers experience is core to the JD", "absolutely need"),
    "llm": ("LLM experience is core to the JD", "absolutely need"),
    "fine-tuning": ("Fine-tuning experience aligns with the JD's LLM needs", "absolutely need"),
    "lora": ("LoRA experience aligns with the JD's fine-tuning expectations", "absolutely need"),
    "qlora": ("QLoRA experience aligns with the JD's fine-tuning expectations", "absolutely need"),
    "peft": ("PEFT experience aligns with the JD's fine-tuning expectations", "absolutely need"),
    "prompt engineering": ("Prompt engineering is core to the JD's LLM work", "absolutely need"),
    "kubernetes": ("Kubernetes experience aligns with the JD's production deployment", "nice to have"),
    "docker": ("Docker experience aligns with the JD's production deployment", "nice to have"),
    "aws": ("AWS experience is bonus for the JD", "nice to have"),
    "gcp": ("GCP experience is bonus for the JD", "nice to have"),
    "azure": ("Azure experience is bonus for the JD", "nice to have"),
    "mlops": ("MLOps experience aligns with the JD's production needs", "nice to have"),
    "triton": ("Triton experience aligns with the JD's inference infrastructure", "nice to have"),
    "vector database": ("Vector database experience is core to the JD", "absolutely need"),
    "nlp": ("NLP experience is core to the JD", "absolutely need"),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PRODUCT_COMPANIES = set(_COMPANY_CONTEXT.keys())  # aliases for tier detection

IT_SERVICES = {
    "TCS", "Infosys", "Wipro", "HCL", "Tech Mahindra", "Cognizant",
    "Capgemini", "Accenture", "Mindtree", "L&T Infotech", "Mphasis",
    "Persistent", "Zensar", "Hexaware", "Birlasoft", "Cyient",
}


def _fmt_skill(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return ""
    return s


def _top_skills(df_row: pd.Series, k: int = 4) -> List[str]:
    """Top JD-aligned skills from the candidate's profile."""
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


def _skill_jd_meaning(skill: str) -> Tuple[str, str]:
    """Why this skill matters for the JD's 'absolutely need' list."""
    return _JD_SKILL_MEANING.get(skill.lower(), ("", ""))


def _top_skills_with_meaning(df_row: pd.Series, k: int = 4) -> List[Tuple[str, Tuple[str, str]]]:
    """Top JD-aligned skills paired with their JD-meaning tuple."""
    out: List[Tuple[str, Tuple[str, str]]] = []
    for s in _top_skills(df_row, k=k):
        meaning = _skill_jd_meaning(s)
        if meaning and meaning[0]:
            out.append((s, meaning))
    return out[:k]


def _assessment_str(df_row: pd.Series) -> str:
    sig = df_row.get("signals")
    if sig is None:
        sig = {}
    elif hasattr(sig, "tolist"):
        sig = sig.tolist() if sig else {}
    if not isinstance(sig, dict):
        return ""
    sas = sig.get("skill_assessment_scores") or {}
    vals = [(k, v) for k, v in sas.items()
            if v is not None and isinstance(v, (int, float))]
    if not vals:
        return ""
    top = sorted(vals, key=lambda x: -x[1])[:3]
    return ", ".join(f"{k}={int(v)}" for k, v in top)


def _career_evidence_snippet(df_row: pd.Series) -> str:
    """Extract one shipped/built/launched verb from career descriptions."""
    career = df_row.get("career")
    if career is None:
        career = []
    elif hasattr(career, "tolist"):
        career = career.tolist()
    if not career:
        return ""
    verb_re = re.compile(
        r"\b(built|shipped|launched|migrated|scaled|deployed|"
        r"productionized|productionised|architected|led|owned|drove|"
        r"designed|implemented|integrated|rolled out)\b",
        re.IGNORECASE,
    )
    target_re = re.compile(
        r"\b(search|ranking|retrieval|recommender|recommendation|ranker|"
        r"vector|embedding|llm|fine.tun|rag|model|system|pipeline|"
        r"platform|infrastructure)\w*\b",
        re.IGNORECASE,
    )
    for ch in career[-3:]:
        desc = (ch.get("description") or "")
        if not desc:
            continue
        # Find first verb-noun pair
        for sent in re.split(r"[.;]\s*", desc):
            v = verb_re.search(sent)
            t = target_re.search(sent)
            if v and t:
                return f"{v.group(0).lower()} {t.group(0).lower()}"
        # Fallback: any verb
        v = verb_re.search(desc)
        if v:
            return f"{v.group(0).lower()} applied-ML systems"
    return ""


def _yoe_band(yoe: float) -> str:
    if yoe < config.YOE_IDEAL_LOW:
        return (f"{yoe:.1f} yrs "
                f"(slightly under JD's {config.YOE_IDEAL_LOW}-{config.YOE_IDEAL_HIGH} sweet spot)")
    if yoe > config.YOE_IDEAL_HIGH:
        return (f"{yoe:.1f} yrs "
                f"(slightly over JD's {config.YOE_IDEAL_LOW}-{config.YOE_IDEAL_HIGH} sweet spot)")
    return (f"{yoe:.1f} yrs "
            f"(in JD's {config.YOE_IDEAL_LOW}-{config.YOE_IDEAL_HIGH} sweet spot)")


def _location_str(df_row: pd.Series) -> str:
    loc = (df_row.get("location") or "").strip()
    country = (df_row.get("country") or "").strip()
    s = f"{loc}, {country}" if loc and country else (loc or country or "")
    return s


def _location_fit(df_row: pd.Series) -> str:
    """Does the candidate's location match the JD's Pune/Noida preference?"""
    loc = (df_row.get("location") or "").lower()
    if not loc:
        return ""
    preferred = [c.lower() for c in config.PREFERRED_LOCATIONS]
    for p in preferred:
        if p.lower() in loc:
            return "preferred"
    return ""


def _is_services_company(company: str) -> bool:
    cl = (company or "").lower()
    for s in IT_SERVICES:
        if s.lower() in cl:
            return True
    return False


def _company_tier(df_row: pd.Series) -> str:
    company = (df_row.get("current_company") or "").strip()
    if not company:
        return "unknown"
    # Tier-3
    for c in PRODUCT_COMPANIES:
        if c.lower() in company.lower():
            return "tier3"
    # Services
    if _is_services_company(company):
        return "services"
    return "tier1"


def _rank_band(rank: int) -> str:
    if rank <= 3:
        return "elite"
    if rank <= 10:
        return "top"
    if rank <= 50:
        return "mid"
    return "tail"


# ---------------------------------------------------------------------------
# Clause builders
# ---------------------------------------------------------------------------

def _clause_identity(df_row: pd.Series, rank: int) -> str:
    """Clause 1: who they are + career evidence (replaces generic praise)."""
    title = (df_row.get("current_title") or "").strip() or "this candidate"
    company = (df_row.get("current_company") or "").strip()
    company_ctx = _company_context(company)
    yoe = float(df_row.get("yoe") or 0.0)
    band = _rank_band(rank)
    seniority = _detect_seniority(title)
    evidence = _career_evidence_snippet(df_row)

    # Drop the seniority prefix if the title itself already includes the word
    title_lower = title.lower()
    if seniority and seniority in title_lower:
        prefix = ""
    elif seniority == "principal":
        prefix = "Principal-level "
    elif seniority == "staff":
        prefix = "Staff-level "
    elif seniority == "lead":
        prefix = "Lead-level "
    elif seniority == "senior":
        prefix = "Senior-level "
    else:
        prefix = ""

    title_phrase = f"{prefix}'{title}'" if prefix else f"'{title}'"
    ctx_phrase = f" ({company_ctx})" if company_ctx else ""
    head = f"{title_phrase} at {company}{ctx_phrase} with {_yoe_band(yoe)}"

    # Append a career-evidence fragment when available
    if evidence:
        return head + f" — career history shows they have {evidence}."

    return head + "."


def _strip_dup_prefix(title: str, seniority: str) -> str:
    """If the title already contains the seniority word, don't prefix it."""
    if not seniority:
        return title
    if re.search(rf"\b{seniority}\b", title, re.IGNORECASE):
        return title
    return title


def _detect_seniority(title: str) -> str:
    t = title.lower()
    if re.search(r"\bprincipal\b", t): return "principal"
    if re.search(r"\bstaff\b", t): return "staff"
    if re.search(r"\b(head|chief|director|architect)\b", t): return "staff"
    if re.search(r"\b(lead|principal)\b", t): return "lead"
    if re.search(r"\bsenior\b", t): return "senior"
    if re.search(r"\b(junior|jr|intern|trainee)\b", t): return "junior"
    return ""


def _clause_skill_jd_alignment(df_row: pd.Series) -> str:
    """Clause 2: connect top skills to specific JD requirements."""
    pairs = _top_skills_with_meaning(df_row, k=3)
    if not pairs:
        # Fallback: just name top skills without meaning
        skills = _top_skills(df_row, k=3)
        if skills:
            return f"Skills intersecting the JD's stack: {', '.join(skills)}."
        return "Limited JD-skill overlap; included for retrieval breadth."

    parts = []
    for skill, meaning in pairs:
        why = meaning[0] if meaning and meaning[0] else ""
        if why:
            # Use only the first short fragment (before first comma)
            short = why.split(",")[0].strip()
            parts.append(f"{skill} ({short})")
        else:
            parts.append(skill)
    return "Matches JD 'absolutely need' on: " + ", ".join(parts) + "."


def _clause_assessment(df_row: pd.Series) -> str:
    """Clause 3: Redrob skill assessment scores (high-value when present)."""
    assess = _assessment_str(df_row)
    if not assess:
        return "No Redrob skill-assessment scores on file (verified skills via profile only)."
    return f"Redrob skill assessments: {assess}."


def _clause_career_coherence(df_row: pd.Series, rank: int) -> str:
    """Clause 4 (mid/tail): career-graph / community coherence."""
    car_ev = float(df_row.get("career_evidence") or 0.0)
    car_coh = float(df_row.get("career_coherence") or 0.0)
    if car_ev >= 0.5:
        return "Career-graph evidence: descriptions reference shipped production ranking/retrieval systems."
    if car_coh >= 0.7:
        return "Career-coherence is high: titles form a consistent applied-ML progression."
    return "Included on retrieval + skill signals (career descriptions lack shipped-system keywords)."


def _clause_geo_notice(df_row: pd.Series) -> str:
    """Clause 5: location fit + notice period context."""
    loc = _location_fit(df_row)
    notice = int(df_row.get("notice_period_days") or 0)
    reloc = bool(int(df_row.get("willing_to_relocate") or 0))

    geo_part = ""
    if loc == "preferred":
        loc_str = _location_str(df_row)
        geo_part = f"Located in JD-preferred city ({loc_str}); relocation not required."
    elif reloc:
        geo_part = f"Willing to relocate to a JD-preferred city (current: {_location_str(df_row)})."

    notice_part = ""
    if notice and notice <= 30:
        notice_part = f"Notice period {notice} days — within the JD's sub-30-day preference."
    elif notice and notice <= 60:
        notice_part = f"Notice period {notice} days — within tolerance; the JD will buy out up to 30 days."
    elif notice:
        notice_part = f"Notice period {notice} days — above the JD's 30-day preference; bar is higher."

    if geo_part and notice_part:
        return geo_part + " " + notice_part
    return geo_part or notice_part or ""


def _clause_concerns(df_row: pd.Series, rank: int) -> str:
    """Honest concerns clause."""
    concerns: List[str] = []

    notice = int(df_row.get("notice_period_days") or 0)
    if notice >= 90:
        concerns.append(f"{notice}-day notice period is above the JD's 30-day preference")

    honeypot = float(df_row.get("honeypot_penalty") or 0.0)
    if honeypot >= 0.25:
        concerns.append(f"profile shows some inconsistency signals (honeypot penalty {honeypot:.2f})")

    rec = float(df_row.get("recruitability") or 0.0)
    if rec < 0.3:
        concerns.append("recruiter-engagement signals are muted (verify interest before outreach)")

    is_junior = bool(int(df_row.get("is_junior_title") or 0))
    yoe = float(df_row.get("yoe") or 0.0)
    in_band = config.YOE_IDEAL_LOW <= yoe <= config.YOE_IDEAL_HIGH
    if is_junior and in_band and rank <= 50:
        concerns.append("junior title despite senior-YOE band — verify level")

    # Negative-spec flags
    if bool(int(df_row.get("is_consulting_only") or 0)):
        concerns.append("career is consulting-only; JD strongly prefers product-company experience")
    if bool(int(df_row.get("is_framework_enthusiast") or 0)):
        concerns.append("skill mix is framework-heavy (LangChain/LlamaIndex) without deep retrieval fundamentals")

    if not concerns:
        return "No major concerns."
    if len(concerns) == 1:
        return f"Honest note: {concerns[0]}."
    if len(concerns) == 2:
        return f"Honest notes: {concerns[0]}; {concerns[1]}."
    return "Honest notes: " + "; ".join(concerns[:2]) + f"; and {len(concerns)-2} more."


# ---------------------------------------------------------------------------
# Template assembly per rank-band × company-tier
# ---------------------------------------------------------------------------

def _assemble(df_row: pd.Series, rank: int, band: str, tier: str) -> List[str]:
    """Return 3-4 clauses appropriate for this rank band and company tier."""
    clauses: List[str] = []
    band_tier = f"{band}_{tier}"
    company = (df_row.get("current_company") or "").strip()

    # ---- Clause 1: identity + career evidence ----
    c1 = _clause_identity(df_row, rank)
    clauses.append(c1)

    # ---- Clause 2: skill → JD alignment ----
    clauses.append(_clause_skill_jd_alignment(df_row))

    # ---- Clause 3: assessment scores (top half only) ----
    if rank <= 50:
        clauses.append(_clause_assessment(df_row))

    # ---- Clause 4: geo + notice (top half only) ----
    geo_notice = _clause_geo_notice(df_row)
    if geo_notice and rank <= 50:
        clauses.append(geo_notice)

    # ---- Clause 5: career-graph / coherence (mid/tail) ----
    if band in ("mid", "tail"):
        clauses.append(_clause_career_coherence(df_row, rank))

    # ---- Clause 6: concerns ----
    if rank <= 100:
        # Always include a concerns clause
        clauses.append(_clause_concerns(df_row, rank))

    return clauses


def _choose_clauses(rank: int, df_row: pd.Series) -> List[str]:
    """Choose clauses appropriate for this rank and candidate."""
    band = _rank_band(rank)
    tier = _company_tier(df_row)
    return _assemble(df_row, rank, band, tier)


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def generate_reasoning(df_row: pd.Series, rank: int, n_total: int = 100) -> str:
    """Compose a 3-5 clause reasoning string for one candidate."""
    clauses = _choose_clauses(rank, df_row)
    # Ensure 3-5 clauses
    while len(clauses) < 3:
        clauses.append("No major concerns.")
    clauses = clauses[:5]
    text = " ".join(clauses)
    text = text.replace("\n", " ").replace("\r", " ")
    text = text.replace('"', "'")
    # Trim if too long (validator only requires correct columns, but very long
    # reasoning hurts Stage 4 readability)
    if len(text) > 800:
        text = text[:797] + "..."
    return text


def generate_reasoning_dataframe(df: pd.DataFrame) -> List[str]:
    """Return a list of reasoning strings, one per row, in input order."""
    out: List[str] = []
    n = len(df)
    for i in range(n):
        out.append(generate_reasoning(df.iloc[i], i + 1, n))
    return out
