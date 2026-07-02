# Redrob Candidate Intelligence Engine

This repository contains the implementation of a candidate discovery and
ranking system for the Redrob India Runs Data & AI Challenge.

## What it does

Given a 100,000-candidate pool and a job description for a **Senior AI
Engineer** (Redrob AI's founding team), the system produces a top-100
ranked shortlist with a per-candidate recruiter-style reasoning
explanation. The pipeline is:

1. **Role Blueprint Generator** — extracts and persists the hiring
   intent (core competencies, target titles, seniority band,
   constraints) from the job description.
2. **Candidate Canonicalisation** — streams the 487 MB JSONL, parses
   and normalises every record, and writes a compact parquet cache.
3. **Hybrid Candidate Discovery**:
   * BM25 (sparse) over the 100k corpus.
   * Local sentence-transformers (dense) — model name is
     configurable via `REDROB_DENSE_MODEL`; default is
     `sentence-transformers/all-MiniLM-L6-v2`, with BGE variants
     supported when cached locally.
   * Structured high-recall gate (soft, not exclusion) keeps adjacent
     candidates like Search/Relevance Engineers in the pool.
   * Reciprocal Rank Fusion produces a top-N shortlist.
4. **Graph Intelligence** — heterogeneous graph
   (candidate-skill-title-company-industry) with personalised
   PageRank from JD-skill seeds, skill community purity, skill
   rarity, career coherence, and graph degree.
5. **Behavioral Intelligence** — 23 Redrob signals are folded into a
   single recruitability score (open-to-work, response rate,
   verification, recency, notice period, etc.).
6. **Honeypot Detection** — ten explicit rules including a
   title–skill contradiction detector (advanced AI skills but
   non-engineering career history).
7. **Learning-to-Rank** — LightGBM LambdaRank trained on the full
   pool using transparent weak labels. A deterministic scoring
   function is also implemented as a robust fallback.
8. **Reasoning Generator** — deterministic, evidence-driven
   template-based strings; no LLM at submission time.
9. **Submission Writer** — strict 100-row CSV with monotonically
   non-increasing scores, ties broken by `candidate_id` ascending;
   the bundled validator is invoked as the final step.

## Reproducing the submission

```bash
# from the repository root, with the venv activated
python scripts/run_ranking.py --no_dense --no_lgbm
# Wall-clock: ~58.6 s end-to-end on CPU, no network
# Produces top-100 with byte-identical candidate order to submission.csv
# (scores differ only by tiny float noise; reasoning strings match exactly).
```

This is the proven safe reproduction command. It uses the deterministic
composite scorer (no LightGBM rerank — the LTR model is empirically only
a tiebreaker and the deterministic composite produces the same top-100).
The dense encoder is skipped (no network, faster; the deterministic
composite already dominates).

If you want the full pipeline with dense embeddings and LTR rerank:

```bash
python scripts/run_ranking.py           # ~18 min first run (cold caches)
python scripts/run_ranking.py           # ~50 s on subsequent runs (warm)
```

The command:

* Streams `candidates.jsonl` (~12 s).
* Builds / loads the BM25 index (~16 s on first run, ~4 s cached).
* Optionally builds the dense embedding matrix
  (`artifacts/dense_index.npy`, ~150 MB).
* Computes the heterogeneous graph and PageRank over the 100k
  pool (~14 s).
* Trains the LightGBM ranker on the full pool with weak labels
  (~50 s, only on first run).
* Scores the top-100 and writes `submission.csv`.

The validator from the bundle is invoked automatically; the script
exits non-zero if the file is invalid.

## Sandbox

The `app/` directory contains a Streamlit app intended for HuggingFace
Spaces. It runs the **complete pipeline on a small sample** (default
200 records) and shows the top-25 and top-100 tables, charts, and a
downloadable CSV. The sandbox is for demonstration; the official
submission must be produced by `scripts/run_ranking.py` on the full
pool.

## Project layout

```
redrob/                 # core package
  config.py             # paths, blueprint, hyper-parameters
  blueprint.py          # role blueprint
  data.py               # streaming JSONL canonicalisation
  retrieval/            # BM25 + dense + RRF
  graph/                # heterogeneous graph + PPR + coherence
  features/             # title, behavioral, honeypot, skill
  rank/                 # LightGBM LambdaRank + deterministic fallback
  reasoning/            # deterministic template reasoning
  submit/               # CSV writer + validator hook
scripts/                # run_ranking.py, train_ranker.py, build_blueprint.py
app/                    # Streamlit sandbox
artifacts/              # cached parquet, BM25, dense, graph, model
```

## Compute constraints

The full pipeline runs end-to-end in **≤2 minutes** on a 4-core CPU
with 16 GB RAM, no GPU, no network (after the first-time model
download). The fallback path (BM25-only, deterministic scorer)
finishes in **~90 seconds**.

## License

Internal hackathon submission.
