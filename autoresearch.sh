#!/usr/bin/env python3
"""autoresearch.sh — Redrob v2 benchmark runner.

Outputs METRIC name=value lines for the autoresearch helper.
Usage: python autoresearch.sh
"""
from __future__ import annotations

import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

VENV_PYTHON = Path("D:/Project/redrob/.venv/Scripts/python.exe")
PIPELINE = Path("D:/Project/redrob/scripts/run_ranking.py")
VALIDATOR = Path("D:/Project/redrob/[PUB] India_runs_data_and_ai_challenge/[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/validate_submission.py")
OUT_CSV = Path("D:/Project/redrob/submission.csv")

# ---- Constants ----
_NON_ENG_RE = re.compile(
    r"\b(marketing manager|hr manager|accountant|sales executive|"
    r"graphic designer|content writer|civil engineer|mechanical engineer|"
    r"customer support|operations manager|project manager|business analyst|"
    r"qa engineer|frontend developer|java developer|.net developer)\b",
    re.IGNORECASE,
)
_JD_CORE_RE = re.compile(
    r"\b(rag|faiss|bm25|pytorch|tensorflow|transformers|llm|lora|qlora|peft|"
    r"elasticsearch|opensearch|nlp|learning to rank|ndcg|mrr|map|embeddings|"
    r"sentence transformers|bge|hybrid search|dense retrieval|vector database|"
    r"pinecone|weaviate|qdrant|milvus|evaluation|a/?b testing|"
    r"lambdarank|xgboost|lightgbm|catboost|distributed systems|mlops|"
    r"kubeflow|triton|onnx|distributed training|inference|peft)\b",
    re.IGNORECASE,
)
_PRODUCT_RE = re.compile(
    r"\b(razorpay|zerodha|cred|phonepe|paytm|swiggy|zomato|flipkart|meesho|"
    r"urban company|dream11|groww|cars24|postman|freshworks|zoho|chargebee|"
    r"browserstack|sprinklr|clevertap|moengage|sharechat|inmobi|microsoft|google|"
    r"meta|amazon|apple|netflix|stripe|airbnb|uber|linkedin|salesforce|adobe|"
    r"nvidia|snowflake|databricks|openai|anthropic|cohere|hugging face|"
    r"pinecone|weaviate|qdrant|redrob)\b",
    re.IGNORECASE,
)
_SHIP_RE = re.compile(
    r"\b(built|shipped|launched|migrated|scaled|deployed|productionized|"
    r"productionised|architected|led|owned|drove|designed|implemented|"
    r"integrated|rolled out|from zero|to millions|million users|100x|10x)\b",
    re.IGNORECASE,
)
_YOE_RE = re.compile(r"(\d+\.?\d*)\s*yrs?")


def _grade_row(reasoning: str) -> int:
    """Grade 0-4 based on reasoning text — the most honest proxy available."""
    text = (reasoning or "").lower()

    # Honeypot: explicit inconsistency signal
    if any(k in text for k in ["honeypot penalty", "inconsistency signals",
                                "verify the listed", "profile shows some inconsistency"]):
        return 0

    # Title-skill contradiction
    has_non_eng = bool(_NON_ENG_RE.search(text))
    has_ai_kw = bool(_JD_CORE_RE.search(text))
    if has_non_eng and has_ai_kw:
        return 0

    # Pure non-eng title (no AI keywords to contradict)
    if has_non_eng and not has_ai_kw:
        return 1

    # Extract YOE
    m = _YOE_RE.search(reasoning or "")
    yoe = float(m.group(1)) if m else 0.0

    # Count JD-core skills in reasoning
    core_hits = len(_JD_CORE_RE.findall(text))

    # Ship/product evidence
    has_ship = bool(_SHIP_RE.search(text))
    has_product = bool(_PRODUCT_RE.search(text))

    # Retrieval keywords in reasoning (retrieval is the JD's CORE requirement)
    retrieval_kw = ["rag", "bm25", "faiss", "elasticsearch", "opensearch",
                    "vector search", "hybrid search", "sentence transformers",
                    "pinecone", "weaviate", "qdrant", "milvus",
                    "learning to rank", "dense retrieval"]
    n_retrieval = sum(1 for k in retrieval_kw if k in text)

    # Band
    ideal_band = 5.0 <= yoe <= 9.0
    decent_band = 4.0 <= yoe <= 11.0

    # Target titles present
    target_kw = ["ml engineer", "ai engineer", "applied scientist", "research engineer",
                  "search engineer", "ranking engineer", "recommendation systems engineer"]
    has_target = any(k in text for k in target_kw)

    # --- Grade 4: gold ---
    # JD says: 5-9 yrs, applied-ML title, product company, retrieval skills, shipped evidence
    if (ideal_band and has_target and has_ship and has_product and n_retrieval >= 3):
        return 4
    if (ideal_band and has_target and core_hits >= 6 and has_product and n_retrieval >= 2):
        return 4

    # --- Grade 3: strong ---
    # JD says: 4-11 yrs, applied ML, 4+ JD skills
    if (decent_band and has_target and core_hits >= 5):
        return 3
    if (has_target and n_retrieval >= 2 and core_hits >= 4):
        return 3

    # --- Grade 2: decent ---
    if has_target and core_hits >= 3:
        return 2
    if n_retrieval >= 1 and core_hits >= 2:
        return 2

    # --- Grade 1: weak ---
    if core_hits >= 1:
        return 1

    return 0


def _compute_proxy_ndcg10(csv_path: Path) -> float:
    """Compute proxy NDCG@10 using strict grading from reasoning text."""
    import csv
    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    grades = [_grade_row(r.get("reasoning", "")) for r in rows[:10]]

    def _dcg(g_list):
        return sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(g_list))

    dcg = _dcg(grades)
    ideal = _dcg(sorted(grades, reverse=True))
    return round(dcg / ideal if ideal > 0 else 0.0, 4)


def _compute_grade_distribution(csv_path: Path) -> dict:
    """Return grade distribution for top-50 for diagnostics."""
    import csv
    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    dist = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    for r in rows[:50]:
        dist[_grade_row(r.get("reasoning", ""))] += 1
    return dist


def _compute_hp_rate(csv_path: Path) -> float:
    import csv
    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    hp = sum(1 for r in rows if _grade_row(r.get("reasoning", "")) == 0)
    return round(100.0 * hp / len(rows), 1)


def _compute_reasoning_variation(csv_path: Path) -> float:
    """Mean Jaccard dissimilarity across evenly-spaced pairs from top-100."""
    import csv
    with open(csv_path, encoding="utf-8") as f:
        texts = [r.get("reasoning", "") or "" for r in csv.DictReader(f)]

    def _tokens(t: str):
        return set(re.findall(r"\w+", t.lower()))

    pairs: list = []
    for i in range(0, 100, 10):
        for j in range(i + 1, 100, 10):
            ti, tj = _tokens(texts[i]), _tokens(texts[j])
            u = len(ti | tj)
            if u > 0:
                pairs.append(len(ti & tj) / u)
    mean_jac = sum(pairs) / len(pairs) if pairs else 0.0
    return round(1.0 - mean_jac, 4)


def _count_unique_template_shapes(csv_path: Path) -> int:
    """Count distinct template shapes in top-100."""
    import csv
    with open(csv_path, encoding="utf-8") as f:
        texts = [r.get("reasoning", "") or "" for r in csv.DictReader(f)]
    shapes = set()
    for t in texts:
        t = t.strip()
        if t.startswith("Senior AI Engineer fit:"): shapes.add("senior_fit")
        elif t.startswith("Strong adjac"): shapes.add("strong_adj")
        elif t.startswith("Adjacent fit:"): shapes.add("adj_fit")
        elif t.startswith("Lower-confid"): shapes.add("low_conf")
        elif t.startswith("Long-shot:"): shapes.add("long_shot")
        elif t.startswith("Top of the shortlist"): shapes.add("top_tier")
        else: shapes.add("other")
    return len(shapes)


def _run_pipeline() -> float:
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = "0"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    t0 = time.time()
    r = subprocess.run(
        [str(VENV_PYTHON), str(PIPELINE), "--no_dense"],
        env=env, capture_output=True, text=True,
        cwd=str(Path(__file__).parent),
    )
    runtime = time.time() - t0
    if r.returncode != 0:
        print(f"PIPELINE ERROR:\n{r.stdout}\n{r.stderr}", file=sys.stderr)
        sys.exit(1)
    return round(runtime, 1)


def _run_validator(csv_path: Path) -> int:
    r = subprocess.run(
        [str(VENV_PYTHON), str(VALIDATOR), str(csv_path)],
        capture_output=True, text=True,
    )
    return 1 if "Submission is valid" in r.stdout else 0


def main():
    print("=" * 60)
    print("AUTORESEARCH BENCHMARK — Redrob v2")
    print("=" * 60)

    print("[1/4] Running pipeline...", flush=True)
    runtime = _run_pipeline()
    print(f"      done in {runtime:.1f}s")

    print("[2/4] Validating...", flush=True)
    vp = _run_validator(OUT_CSV)
    print(f"      validator_pass={vp}")

    print("[3/4] Computing proxy NDCG@10...", flush=True)
    proxy_ndcg = _compute_proxy_ndcg10(OUT_CSV)
    dist = _compute_grade_distribution(OUT_CSV)
    print(f"      proxy_ndcg10={proxy_ndcg}")
    print(f"      grade_dist(top50): {dist}")

    print("[4/4] Computing quality metrics...", flush=True)
    hp_rate = _compute_hp_rate(OUT_CSV)
    reason_var = _compute_reasoning_variation(OUT_CSV)
    n_templates = _count_unique_template_shapes(OUT_CSV)

    print()
    print("=" * 60)
    print("METRIC pipeline_runtime_s={:.1f}".format(runtime))
    print("METRIC validator_pass={}".format(vp))
    print("METRIC proxy_ndcg10={:.4f}".format(proxy_ndcg))
    print("METRIC honeypot_rate_pct={:.1f}".format(hp_rate))
    print("METRIC reasoning_variation={:.4f}".format(reason_var))
    print("METRIC n_unique_templates={}".format(n_templates))
    print("=" * 60)


if __name__ == "__main__":
    main()
