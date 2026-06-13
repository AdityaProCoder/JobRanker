#!/usr/bin/env python3
"""autoresearch.sh equivalent — pure Python runner for Redrob v2 benchmark.

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

# ---- Proxy NDCG@10 ----
JOB_TITLE_KW = [
    "ml engineer", "machine learning engineer", "ai engineer", "applied scientist",
    "research engineer", "search engineer", "ranking engineer",
    "recommendation systems engineer", "nlp engineer", "data scientist",
]
NON_ENG_KW = [
    "marketing manager", "hr manager", "accountant", "sales executive",
    "graphic designer", "content writer", "civil engineer", "mechanical engineer",
    "customer support", "operations manager", "project manager", "business analyst",
]
PRODUCT_COMPANIES = {
    "razorpay", "zerodha", "cred", "phonepe", "paytm", "swiggy", "zomato",
    "flipkart", "meesho", "urban company", "dream11", "groww", "cars24",
    "postman", "freshworks", "zoho", "chargebee", "browserstack", "sprinklr",
    "clevertap", "moengage", "sharechat", "inmobi", "microsoft", "google",
    "meta", "amazon", "apple", "netflix", "stripe", "airbnb", "uber",
    "linkedin", "salesforce", "adobe", "nvidia", "snowflake", "databricks",
    "openai", "anthropic", "cohere", "hugging face", "pinecone", "weaviate",
    "qdrant",
}
SHIP_VERBS = {
    "built", "shipped", "launched", "migrated", "scaled", "deployed",
    "productionized", "productionised", "architected", "led", "owned",
    "drove", "designed", "implemented", "integrated", "rolled out",
}
_JD_CORE_RE = re.compile(
    r"\b(rag|faiss|bm25|pytorch|tensorflow|transformers|llm|lora|qlora|peft|"
    r"elasticsearch|opensearch|nlp|learning to rank|ndcg|mrr|map|embeddings|"
    r"sentence transformers|bge|hybrid search|dense retrieval|vector database|"
    r"pinecone|weaviate|qdrant|milvus|evaluation|a/b testing|"
    r"lambdarank|xgboost|lightgbm|catboost)\b",
    re.IGNORECASE,
)


def _grade_row(candidate_id: str, reasoning: str, score: str) -> int:
    """Grade a candidate 0-4 based on reasoning text (proxy for full profile)."""
    text = (reasoning or "").lower()
    score_str = score or "0"
    try:
        sc = float(score_str)
    except Exception:
        sc = 0.0

    # Honeypot signals in reasoning
    hp_kw = ["honeypot", "inconsistency signals", "verify the listed"]
    if any(k in text for k in hp_kw):
        return 0

    # Title-skill contradiction in reasoning
    if any(t in text for t in NON_ENG_KW) and any(
        s in text for s in ["rag", "llm", "faiss", "pytorch", "bm25", "transformers"]
    ):
        return 0

    # Non-eng title in reasoning text
    non_eng_in_text = any(t in text for t in NON_ENG_KW)
    eng_in_text = any(t in text for t in ["ml engineer", "ai engineer", "applied scientist",
                                            "research engineer", "search engineer", "nlp engineer",
                                            "data scientist", "recommendation"])
    if non_eng_in_text and not eng_in_text:
        return 0

    # Count JD-core skills mentioned in reasoning
    core_hits = len(_JD_CORE_RE.findall(text))
    # Count ship verbs
    ship_hits = sum(1 for v in SHIP_VERBS if v in text)
    # Product company
    product_hit = any(c in text for c in ["razorpay", "google", "amazon", "microsoft",
                                           "meta", "netflix", "stripe", "swiggy", "zomato",
                                           "flipkart", "linkedin", "hugging face"])

    # Band from reasoning text
    import re as _re
    yoe_match = _re.search(r"(\d+\.?\d*)\s*yrs", reasoning or "")
    if yoe_match:
        try:
            yoe = float(yoe_match.group(1))
        except Exception:
            yoe = 0.0
    else:
        yoe = 0.0

    if eng_in_text and 5.0 <= yoe <= 9.0 and core_hits >= 6 and (ship_hits >= 1 or product_hit):
        return 4
    if eng_in_text and 4.0 <= yoe <= 11.0 and core_hits >= 4:
        return 3
    if eng_in_text and core_hits >= 2:
        return 2
    if core_hits >= 1:
        return 1
    return 0


def _compute_proxy_ndcg10(csv_path: Path) -> float:
    """Compute proxy NDCG@10 by grading candidates from reasoning text."""
    import csv
    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    grades = [_grade_row(r["candidate_id"], r.get("reasoning", ""), r.get("score", ""))
              for r in rows[:10]]

    def _dcg(g_list):
        return sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(g_list))

    dcg = _dcg(grades)
    ideal = _dcg(sorted(grades, reverse=True))
    return dcg / ideal if ideal > 0 else 0.0


def _compute_hp_rate(csv_path: Path) -> float:
    """Compute approximate honeypot rate from reasoning text."""
    import csv
    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    hp_count = 0
    for r in rows:
        text = (r.get("reasoning", "") or "").lower()
        hp_kw = ["honeypot", "inconsistency signals", "verify the listed"]
        if any(k in text for k in hp_kw):
            hp_count += 1
        # Title-skill contradiction
        if any(t in text for t in NON_ENG_KW) and any(
            s in text for s in ["rag", "llm", "faiss", "pytorch", "bm25", "transformers"]
        ):
            hp_count += 1
    return round(100.0 * hp_count / len(rows), 1)


def _compute_reasoning_variation(csv_path: Path) -> float:
    """Compute mean pairwise Jaccard dissimilarity across top-100 reasonings."""
    import csv
    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    texts = [r.get("reasoning", "") or "" for r in rows]

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


def _run_pipeline() -> float:
    """Run the ranking pipeline and return runtime in seconds."""
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = "0"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"

    t0 = time.time()
    r = subprocess.run(
        [str(VENV_PYTHON), str(PIPELINE), "--no_dense"],
        env=env,
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).parent),
    )
    runtime = time.time() - t0
    if r.returncode != 0:
        print(f"PIPELINE ERROR:\n{r.stdout}\n{r.stderr}", file=sys.stderr)
        sys.exit(1)
    return round(runtime, 1)


def _run_validator(csv_path: Path) -> int:
    """Run the validator. Returns 1 if pass, 0 if fail."""
    r = subprocess.run(
        [str(VENV_PYTHON), str(VALIDATOR), str(csv_path)],
        capture_output=True,
        text=True,
    )
    if "Submission is valid" in r.stdout:
        return 1
    print(f"VALIDATOR FAIL:\n{r.stdout}\n{r.stderr}", file=sys.stderr)
    return 0


def main():
    print("=" * 60, flush=True)
    print("AUTORESEARCH BENCHMARK", flush=True)
    print("=" * 60, flush=True)

    # Run pipeline
    print("[1/4] Running pipeline...", flush=True)
    runtime = _run_pipeline()
    print(f"      pipeline completed in {runtime:.1f}s", flush=True)

    # Validate
    print("[2/4] Validating...", flush=True)
    validator_pass = _run_validator(OUT_CSV)
    print(f"      validator_pass={validator_pass}", flush=True)

    # Proxy NDCG@10
    print("[3/4] Computing proxy NDCG@10...", flush=True)
    proxy_ndcg = _compute_proxy_ndcg10(OUT_CSV)
    print(f"      proxy_ndcg10={proxy_ndcg:.4f}", flush=True)

    # Honeypot rate
    print("[4/4] Computing metrics...", flush=True)
    hp_rate = _compute_hp_rate(OUT_CSV)
    reasoning_var = _compute_reasoning_variation(OUT_CSV)

    # Output METRIC lines
    print(flush=True)
    print("=" * 60, flush=True)
    print("METRIC pipeline_runtime_s={:.1f}".format(runtime), flush=True)
    print("METRIC validator_pass={}".format(validator_pass), flush=True)
    print("METRIC proxy_ndcg10={:.4f}".format(proxy_ndcg), flush=True)
    print("METRIC honeypot_rate_pct={:.1f}".format(hp_rate), flush=True)
    print("METRIC reasoning_variation={:.4f}".format(reasoning_var), flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
