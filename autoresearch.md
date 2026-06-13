# Autoresearch: Redrob Hackathon Composite Score Maximization

## Objective
Maximize the Redrob hackathon composite score:
`0.50 × NDCG@10 + 0.30 × NDCG@50 + 0.15 × MAP + 0.05 × P@10`

The ground truth is hidden. We optimize proxy metrics that correlate with the composite:
1. **proxy_ndcg10** (higher is better) — internal proxy NDCG@10 using a transparent 0-4 relevance grading
2. **honeypot_rate_pct** (lower is better) — % of top-100 flagged as honeypots
3. **reasoning_variation** (higher is better) — mean pairwise Jaccard dissimilarity across top-100 reasonings
4. **pipeline_runtime_s** (lower is better) — end-to-end runtime, must stay ≤ 90s
5. **validator_pass** (must be 1) — binary, submission must pass validate_submission.py

## How to Run
```bash
bash autoresearch.sh
```

## Files in Scope
- `redrob/data.py` — tokenization (bigram fix = P0)
- `redrob/retrieval/rrf.py` — RRF with per-channel weights
- `redrob/retrieval/dense.py` — dense retrieval
- `redrob/graph/propagate.py` — PPR and multi-axis PPR
- `redrob/graph/coherence.py` — career evidence scoring
- `redrob/features/honeypot.py` — honeypot rules and negative-spec flags
- `redrob/features/skill_features.py` — JD-criticality weighted skills
- `redrob/features/behavioral.py` — anchored recency
- `redrob/features/title_features.py` — company tier features
- `redrob/rank/train.py` — LTR features, synthetic labels, 5-fold CV
- `redrob/reasoning/template.py` — rank-conditional reasoning
- `scripts/run_ranking.py` — pipeline orchestration
- `redrob/config.py` — constants and knobs
- `redrob/__init__.py`

## Off Limits
- `validate_submission.py` — do not modify
- `candidates.jsonl` — do not modify
- `[PUB] India_runs_data_and_ai_challenge/` — do not modify
- `requirements.txt` — do not add packages
- `.venv/` — do not modify
- `artifacts/` — do not modify (build outputs only)

## Constraints
- Pipeline must complete in ≤ 90 seconds on CPU (hard cap at Stage 3: 5 min)
- Memory ≤ 12 GB
- No network calls during ranking
- No new pip packages
- Submission must pass validator on every experiment
- Byte-identical reproducible across runs (set PYTHONHASHSEED=0)

## Termination
Run until: 30 experiments or user interrupts.

## Proxy NDCG Grading Rubric (for internal evaluation only)
Grade 4 (gold): Title ∈ {Senior/Staff ML Engineer, Senior/Staff AI Engineer, Applied Scientist, Senior Applied Scientist, Search Engineer, Recommendation Systems Engineer} AND YOE 5-9 AND ≥6 JD-core skills AND career has shipped/built evidence AND company ∈ product-company list
Grade 3 (strong): Title ∈ target titles AND YOE 4-11 AND ≥4 JD-core skills AND career evidence
Grade 2 (decent): Title ∈ adjacent titles (NLP Engineer, Data Scientist, ML Engineer) AND ≥2 core skills
Grade 1 (weak): Adjacent only, limited evidence
Grade 0 (reject): Honeypot OR current_title is non-eng AND all hist titles non-eng AND ≥5 AI skills; OR profile stub with rich skills AND completeness<35

## What's Been Tried

### Exp #1 — Baseline
- Current pipeline: BM25+bigrams + MiniLM + PPR (1-axis) + LambdaRank(weak labels) + 1-template reasoning
- proxy_ndcg10: ~0.52 (estimated)
- honeypot_rate_pct: 0.0
- reasoning_variation: 0.08 (very low — 100% template identical)
- pipeline_runtime_s: 30s
- validator_pass: 1

### Exp #2 — Remove BM25 bigrams (biggest bang for buck)
- Hypothesis: bigrams reward keyword-stuffing honeypots; removing them reduces honeypot BM25 scores to median
- Expected: +0.05 NDCG@10, 0 honeypot rate maintained
- Status: PENDING

### Exp #3 — JD-criticality weighted skills
- Hypothesis: RAG/FAISS/BM25 should count 2× vs Python/AWS 0.5×
- Expected: +0.03 NDCG@10

### Exp #4 — Multi-axis PPR (5 axes)
- Hypothesis: single PPR on 64 skills gives no discrimination; 5-axis PPR distinguishes applied_ml vs retrieval_rank vs nlp_llm vs production_eng vs product_company
- Expected: +0.06 NDCG@10

### Exp #5 — Company-tier feature
- Hypothesis: Razorpay/Google Engineer >> Wipro/TCS Engineer
- Expected: +0.03 NDCG@10

### Exp #6 — Career-evidence score
- Hypothesis: JD explicitly asks for "shipped ranking/search/recsys at scale" — descriptions have this evidence
- Expected: +0.04 NDCG@50

### Exp #7 — Negative-spec flags
- Hypothesis: JD's "do NOT want" list should be explicit negative features
- Expected: +0.02 NDCG@10

### Exp #8 — Honeypot rule 7 v2
- Hypothesis: current rule too strict; loosened version catches more traps
- Expected: +0.02 NDCG@10

### Exp #9 — Per-channel RRF weights
- Hypothesis: BM25 more reliable than dense in this dataset; weight BM25×1.2, dense×0.8
- Expected: +0.01 NDCG@10

### Exp #10 — Anchored recency
- Hypothesis: datetime.utcnow() is non-deterministic; anchor to dataset max date
- Expected: reproducibility +0.01 NDCG@10

### Exp #11 — Synthetic CV labels
- Hypothesis: ranker trains on its own weak labels = no learning; use held-out feature subset for labels
- Expected: +0.04 NDCG@10

### Exp #12 — Rank-conditional reasoning
- Hypothesis: Stage 4 manual review penalizes identical templates
- Expected: Stage 4 check 6/6, composite uplift

## Key Insights from Deep Research
- The challenge has 3 explicit traps: keyword-bombing honeypots, "perfect on paper inactive", "do NOT want" list
- NDCG@10 = 50% of composite; top-5 positions are worth 10× top-50 positions
- Ground truth is tier-graded 0-4; our proxy should match this rubric
- The ranker is currently a polynomial of its own features; it learns nothing new
- Reasoning variation = 0.08 means 100% identical structure — guaranteed Stage 4 fail on Variation check
