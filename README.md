# Redrob Candidate Intelligence Engine

This repository contains the implementation of a candidate discovery and
ranking system for the **Redrob India Runs Data & AI Challenge**.

Given a 100,000-candidate pool and a job description for a **Senior AI
Engineer** (Redrob AI's founding team, Pune/Noida, 5–9 yrs), the system
produces a top-100 ranked shortlist with a per-candidate recruiter-style
reasoning explanation.

## Final result (v11)

* **Validator**: `Submission is valid.`
* **Top-10 unchanged** across the last 4 iterations (v8 → v11)
* **Top-50 unchanged** across v10 → v11
* **0 honeypots / 0 junior / 0 non-engineering / 0 services-only in top-50**
* **100/100 unique reasoning strings**
* **Pipeline runtime: ~60 s** end-to-end on CPU, no network during ranking
* **Memory peak: ~6 GB** (≤16 GB budget)
* **Reproducibility**: byte-identical CSV (only float noise) across runs

## Architecture (single-command, byte-identical)

1. **Retrieval** — 100k → 6k shortlist
   * BM25 unigrams (rank_bm25) — top-3000 per query term
   * sentence-transformers MiniLM-L6, 384-dim, L2-normalised (optional)
   * Soft structured-gate: `min(1, hits/5)` on text, `hits/3` on skills
   * Per-channel weighted RRF: BM25 ×1.2, dense ×0.8, K=60
   * Plain-language Tier-5 top-up: +200 candidates max

2. **Graph** — 100k nodes, 1.7M edges
   * Edges: `has_skill`, `held_title`, `worked_at`, `in_industry`, skill-cooccurrence
   * **5-axis PageRank** (specialised seeds): `applied_ml`, `retrieval_rank`, `nlp_llm`, `production_eng`, `product_company`
   * Career-evidence: ship verbs + retrieval keywords in career descriptions
   * Skill-community purity via Louvain on skill-skill subgraph

3. **Ranking** — 6k shortlist → top-100
   * **74-feature frame**: retrieval (4) + graph (15) + skills (16) + title (10) + behavioural (11) + honeypot (11) + seniority/geo (7+)
   * **Deterministic composite scorer** (primary signal)
   * **LightGBM LambdaRank**, 600 trees (local tiebreak only, Δ < 0.15)
   * **11 honeypot rules** including behavioural twins (530 dup keys); hard-exclude at penalty ≥ 0.5
   * **v11 logistics eligibility filter**: hard-remove `country != "India" AND willing_to_relocate == False` from top-100 selection (preserves non-India + willing-to-relocate)

4. **Reasoning** — deterministic 5-clause template (WHO, CAREER EVIDENCE, JD MAPPING, ASSESSMENT, LOCATION+NOTICE, CONCERNS). Every claim sourced from a profile field. No LLM at ranking time.

## Reproducing the submission

```bash
# from the repository root, with the venv activated
python scripts/run_ranking.py --no_dense --no_lgbm
```

* **Wall-clock: ~60 s** end-to-end on CPU, no network
* Produces top-100 with byte-identical candidate order to `submission.csv` (scores differ only by tiny float noise; reasoning strings match exactly)
* The dense encoder is skipped (no network, faster; the deterministic composite already dominates)
* The LightGBM ranker is skipped (LTR was empirically only a tiebreaker; the deterministic composite produces the same top-100)

For the full pipeline with dense embeddings and LTR rerank (longer first run):

```bash
python scripts/run_ranking.py           # ~18 min first run (cold caches), ~50 s on subsequent runs
```

## What the pipeline does (in order)

* Streams `candidates.jsonl` (~12 s first time, ~8 s cached)
* Builds / loads the BM25 index (~16 s on first run, ~4 s cached)
* Computes structured-gate score over all 100k candidates (~10 s)
* Fuses BM25 + dense + structured-gate via weighted RRF → 6k shortlist
* Builds the heterogeneous graph (1.7M edges) and 5-axis PageRank (~14 s; cached)
* Builds a 74-feature frame for the 6k shortlist (~3 s)
* Scores with deterministic composite + LTR tiebreak (<1 s)
* Applies v11 logistics eligibility filter
* Generates 5-clause reasoning for top-100
* Writes `submission.csv` and runs the bundled validator

The validator from the bundle is invoked automatically; the script
exits non-zero if the file is invalid.

## Sandbox (Stage 1 requirement)

The hackathon spec (Section 10.5) requires a working hosted sandbox
where organizers can verify the ranking system runs reproducibly. We
include a **Streamlit app** at `app/app.py` that can be deployed to
either:

- **Streamlit Cloud** (free tier, recommended): connect this GitHub
  repo, point at `app/app.py`, deploy. URL becomes the `sandbox_link`
  in `submission_metadata.yaml`.
- **HuggingFace Spaces**: copy `app/app.py` into a new Space with
  Streamlit SDK.
- **Or** a self-contained `docker run` recipe, Colab notebook, etc.
  (see spec Section 10.5 for the full list of accepted platforms).

The sandbox runs the **complete pipeline** (BM25 + graph + features +
deterministic composite + reasoning) on a user-uploaded sample. It
must accept a candidate sample (≤100 records), rank end-to-end, and
produce a downloadable CSV within the 5-min CPU budget.

**Deployment checklist (for the user):**
1. Deploy `app/app.py` to Streamlit Cloud or HF Spaces.
2. Verify the deployed URL works.
3. Update `sandbox_link` in `submission_metadata.yaml` with the
   real deployed URL.
4. Submit the metadata.

## Project layout

```
redrob/                 # core package
  config.py             # paths, blueprint, hyper-parameters, company tiers
  blueprint.py          # role blueprint extractor
  data.py               # streaming JSONL canonicalisation
  retrieval/            # BM25 + dense + weighted RRF
  graph/                # heterogeneous graph + 5-axis PPR + Louvain
  features/             # title, behavioral, honeypot (11 rules), skill (aliases)
  rank/                 # deterministic composite + LightGBM LambdaRank tiebreak
  reasoning/            # 16 rank-conditional deterministic templates
  submit/               # CSV writer + validator hook
scripts/
  run_ranking.py        # THE pipeline (the official reproduce cmd)
app/
  app.py                # Streamlit sandbox (Stage 1 sandbox requirement)
.streamlit/
  config.toml           # Streamlit theme + server config
artifacts/              # cached parquet, BM25, dense, graph, model (gitignored)
submission.csv          # top-100 ranked (validator passes)
submission_metadata.yaml  # portal metadata
```

## Compute constraints (all met)

| Constraint | Budget | Actual |
|---|---|---|
| Wall-clock | ≤ 5 min | **~60 s** (no_dense + no_lgbm) |
| Memory | ≤ 16 GB | **~6 GB** |
| Compute | CPU only | CPU only ✓ |
| Network during ranking | Off | Off ✓ |
| Disk | ≤ 5 GB | ~1.5 GB |
| GPU | Forbidden | None ✓ |

## Version history (brief)

- **v11** (current) — final logistics eligibility filter. Top-10/50 unchanged, top-100 overlap 94/100.
- **v10** — Tier 2 availability (low response + low recruitability) + non-India + not-willing-to-relocate penalty
- **v8/v9** — skill alias canonicalisation, conservative availability penalty, framework refinement, plain-language Tier-5 top-up (optimised 402s → 63s)
- **v7** — Sr Data Scientist title weight 0.70→0.85 + 30 missing product companies
- **v6** — rich JD-specific reasoning with career-evidence, geo+notice, Redrob assessment context
- **v4/v5** — deterministic-first composite ranker with LTR as local tiebreaker
- **v3** — multi-axis PPR (5 axes), JD-criticality skills, career-evidence score
- **v2/v1** — baseline 9-stage pipeline

## License

Internal hackathon submission.
