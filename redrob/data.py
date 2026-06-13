"""Candidate canonicalisation.

Streams the 100k JSONL, normalises each record into a compact dict, and
optionally writes a parquet cache for fast reloads.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import pandas as pd

from . import config


# ---------------------------------------------------------------------------
# Canonical record
# ---------------------------------------------------------------------------

@dataclass
class CanonicalCandidate:
    candidate_id: str
    name: str
    headline: str
    summary: str
    location: str
    country: str
    yoe: float
    current_title: str
    current_company: str
    current_company_size: str
    current_industry: str
    career: List[Dict] = field(default_factory=list)
    edu: List[Dict] = field(default_factory=list)
    skills: List[Dict] = field(default_factory=list)
    certs: List[Dict] = field(default_factory=list)
    languages: List[Dict] = field(default_factory=list)
    signals: Dict = field(default_factory=dict)
    # Derived
    text_corpus: str = ""
    career_total_months: int = 0
    career_overlap_months: int = 0
    career_industries: List[str] = field(default_factory=list)
    career_titles: List[str] = field(default_factory=list)
    career_companies: List[str] = field(default_factory=list)
    skill_names: List[str] = field(default_factory=list)
    skill_durations_max: Dict[str, int] = field(default_factory=dict)
    has_ai_skill: int = 0


# ---------------------------------------------------------------------------
# Text tokenisation (BM25 corpus)
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[a-z0-9+#\.]+")
# Pre-compile lowercasing noop; use str.translate for speed
_TR_TABLE = str.maketrans("", "", ".,;:!?'\"()[]{}<>")
_SKILL_HINT_RE = re.compile(
    r"\b("
    r"pytorch|tensorflow|transformers?|hugging|llm|rag|retrieval|recommend"
    r"|ranking|ranker|search|embedding|vector|faiss|pinecone|weaviate|qdrant"
    r"|milvus|opensearch|elasticsearch|bm25|xgboost|lightgbm|catboost"
    r"|learning[\s-]?to[\s-]?rank|lambdarank|lora|qlora|peft|fine[\s-]?tun"
    r"|sentence[\s-]?transformer|bge|e5|mlops|kubeflow|mlflow|triton|onnx"
    r"|nlp|computer vision|cv|numpy|pandas|spark|airflow|kafka|databricks"
    r"|snowflake|distributed|production|deploy|inference|eval|ndcg|mrr|map"
    r"|a/b|ab test|hybrid|recsys|recommender|search relevance|rank model"
    r"|promotion velocity"
    r")\b",
    re.IGNORECASE,
)

MAX_TEXT_CHARS = 4000  # BM25 memory control


def _tokenize(text: str) -> List[str]:
    """Unigram tokenization only. Bigrams removed because they reward
    keyword-stuffing honeypot profiles that pack AI buzzwords densely."""
    if not text:
        return []
    text = text.lower().translate(_TR_TABLE)
    return _TOKEN_RE.findall(text)


def _build_text_corpus(c: Dict) -> str:
    parts: List[str] = []
    if c.get("headline"):
        parts.append(c["headline"])
    if c.get("summary"):
        parts.append(c["summary"])
    for ch in c.get("career_history", []):
        if ch.get("title"):
            parts.append(ch["title"])
        if ch.get("company"):
            parts.append(ch["company"])
        if ch.get("description"):
            parts.append(ch["description"])
    for s in c.get("skills", []):
        parts.append(s.get("name", ""))
    for cert in c.get("certifications", []):
        parts.append(cert.get("name", ""))
    for lang in c.get("languages", []):
        parts.append(lang.get("language", ""))
    for edu in c.get("education", []):
        parts.append(edu.get("degree", ""))
        parts.append(edu.get("field_of_study", ""))
    text = " \n ".join([p for p in parts if p])
    return text[:MAX_TEXT_CHARS]


# ---------------------------------------------------------------------------
# Career timeline integrity
# ---------------------------------------------------------------------------

def _months_overlap(a_start: str, a_end: Optional[str], b_start: str, b_end: Optional[str]) -> int:
    """Approx months overlap of two (start,end) date pairs in YYYY-MM-DD."""
    if not a_start or not b_start:
        return 0
    try:
        a1 = _parse_date(a_start)
        a2 = _parse_date(a_end) if a_end else _TODAY
        b1 = _parse_date(b_start)
        b2 = _parse_date(b_end) if b_end else _TODAY
    except Exception:
        return 0
    if a1 > a2 or b1 > b2:
        return 0
    start = a1 if a1 > b1 else b1
    end = a2 if a2 < b2 else b2
    if end < start:
        return 0
    return max(0, int((end - start) // 30))


# Pre-parse today's date once.
_TODAY_DT = __import__("datetime").datetime.now()
_TODAY = _TODAY_DT.year * 12 + _TODAY_DT.month


def _parse_date(s: str) -> int:
    """Return YYYY*12+MM for a YYYY-MM-DD string. Very fast path."""
    if not s or len(s) < 7:
        return 0
    try:
        y = int(s[0:4])
        m = int(s[5:7])
        if not (1970 <= y <= 2099 and 1 <= m <= 12):
            return 0
        return y * 12 + m
    except Exception:
        return 0


def _career_timeline(career: List[Dict]) -> Tuple[int, int, int, int]:
    """Return (sum_months, overlap_months, has_overlap, is_impossible)."""
    total = 0
    overlap = 0
    n_impossible = 0
    n = len(career)
    spans = []
    for ch in career:
        dur = int(ch.get("duration_months") or 0)
        if dur < 0 or dur > 600:
            n_impossible += 1
        total += dur
        a1 = _parse_date(ch.get("start_date", ""))
        a2 = _parse_date(ch.get("end_date")) if ch.get("end_date") else _TODAY
        is_cur = bool(ch.get("is_current"))
        spans.append((a1, a2, is_cur))
    # pairwise overlap of roles where at least one is not "current"
    for i in range(n):
        a1, a2, ac = spans[i]
        if a1 == 0 or a2 == 0:
            continue
        for j in range(i + 1, n):
            b1, b2, bc = spans[j]
            if b1 == 0 or b2 == 0:
                continue
            if ac and bc:
                continue
            start = a1 if a1 > b1 else b1
            end = a2 if a2 < b2 else b2
            if end > start and (end - start) >= 6:
                overlap += (end - start)
    return total, overlap, int(overlap >= 6), n_impossible


# ---------------------------------------------------------------------------
# Canonicalisation
# ---------------------------------------------------------------------------

def canonicalise(c: Dict) -> CanonicalCandidate:
    p = c.get("profile", {}) or {}
    career = c.get("career_history", []) or []
    skills = c.get("skills", []) or []
    sig = c.get("redrob_signals", {}) or {}

    cc = CanonicalCandidate(
        candidate_id=c.get("candidate_id", ""),
        name=p.get("anonymized_name", ""),
        headline=p.get("headline", ""),
        summary=p.get("summary", ""),
        location=p.get("location", ""),
        country=p.get("country", ""),
        yoe=float(p.get("years_of_experience") or 0.0),
        current_title=p.get("current_title", ""),
        current_company=p.get("current_company", ""),
        current_company_size=p.get("current_company_size", ""),
        current_industry=p.get("current_industry", ""),
        career=career,
        edu=c.get("education", []) or [],
        skills=skills,
        certs=c.get("certifications", []) or [],
        languages=c.get("languages", []) or [],
        signals=sig,
    )

    # Text corpus
    cc.text_corpus = _build_text_corpus(c)

    # Career timeline metrics
    total, overlap, _has_overlap, _imp = _career_timeline(career)
    cc.career_total_months = total
    cc.career_overlap_months = overlap
    cc.career_industries = list({(ch.get("industry") or "").strip() for ch in career if ch.get("industry")})
    cc.career_titles = [(ch.get("title") or "").strip() for ch in career if ch.get("title")]
    cc.career_companies = [(ch.get("company") or "").strip() for ch in career if ch.get("company")]

    # Skills
    cc.skill_names = [s.get("name", "").strip() for s in skills if s.get("name")]
    for s in skills:
        n = (s.get("name") or "").strip()
        d = int(s.get("duration_months") or 0)
        if n:
            cc.skill_durations_max[n] = max(cc.skill_durations_max.get(n, 0), d)
    # has_ai_skill flag
    skills_lower = {n.lower() for n in cc.skill_names}
    if skills_lower & {t.lower() for t in config.STRUCTURED_GATE_TERMS}:
        cc.has_ai_skill = 1
    if _SKILL_HINT_RE.search(cc.text_corpus):
        cc.has_ai_skill = 1

    return cc


# ---------------------------------------------------------------------------
# Streaming loader
# ---------------------------------------------------------------------------

def iter_canonical(path: Optional[Path] = None) -> Iterator[CanonicalCandidate]:
    """Yield canonical records from a JSONL file.

    Reads line-by-line. The 100k file is ~487MB; line-based parsing with
    json.loads is ~4x faster than ijson on this file because ijson chokes
    on a small fraction of records with non-standard whitespace, and the
    total memory is fine since we only hold one record at a time.
    """
    p = Path(path) if path else config.DEFAULT_CANDIDATES_JSONL
    if not p.exists():
        raise FileNotFoundError(f"candidates file not found: {p}")

    with open(p, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield canonicalise(json.loads(line))
            except json.JSONDecodeError:
                # skip malformed records
                continue


def load_or_build_parquet(path: Optional[Path] = None) -> pd.DataFrame:
    """Return a pandas DataFrame of canonical candidates, building the
    parquet cache on first call."""
    p = Path(path) if path else config.DEFAULT_CANDIDATES_JSONL
    if config.CANDIDATES_PARQUET.exists():
        return pd.read_parquet(config.CANDIDATES_PARQUET)

    rows: List[Dict] = []
    for cc in iter_canonical(p):
        rows.append(asdict(cc))
    df = pd.DataFrame(rows)
    df.to_parquet(config.CANDIDATES_PARQUET, index=False)
    return df


if __name__ == "__main__":
    df = load_or_build_parquet()
    print(f"loaded {len(df):,} candidates -> {config.CANDIDATES_PARQUET}")
    print("columns:", list(df.columns)[:20])
