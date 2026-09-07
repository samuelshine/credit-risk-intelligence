# Task Breakdown

Every task the build needs, grouped by phase. This is the backlog — granular
and checkable, not a summary. [x] = done and verified, [~] = in progress,
[ ] = not started. For where we are *right now*, see `docs/PROGRESS.md`;
this file is the full map, that one is the pin on it.

Update rule: check an item off only after it has been verified (a test run,
a real command executed, real output inspected) — not on "written but untested."

---

## Phase 0 — Foundation

- [x] Repo scaffold: `src/`, `sql/`, `docs/`, `models/`, `ui/`, `tests/`, `scripts/`
- [x] `requirements.txt` / `requirements-dev.txt`, resolved and smoke-tested on Python 3.12
- [x] `.gitignore` (anchored with leading `/` so `src/data/` isn't swept by `data/`)
- [x] `.env.example` documenting every variable
- [x] `src/utils/config.py` — typed settings, container/local path re-anchoring
- [x] `src/utils/logger.py` — stdout logging, `log_duration` context manager
- [x] `src/utils/helpers.py` — JSON-safe conversion, currency/pct/day formatting
- [x] `docs/TASKS.md` and `docs/PROGRESS.md` (this pair)

## Phase 1 — Data acquisition & ETL

- [x] `src/data/acquire.py` — 3-route dataset acquisition (extracted / zip / Kaggle API)
- [x] `scripts/download_data.py` — CLI wrapper
- [x] `src/data/database.py` — read-write + read-only DuckDB connections
  - [x] read-only connection opened with `enable_external_access=false` (verified: blocks `read_csv`, `glob`, `COPY TO`, `ATTACH`)
- [x] `src/data/loader.py` — CSV → typed DuckDB tables
  - [x] full-scan type inference (`sample_size=-1`)
  - [x] cp1252→UTF-8 transcoding fallback for `HomeCredit_columns_description.csv`
  - [x] `lite` mode: TARGET-stratified client sample, filtered at `CREATE TABLE` time (not load-then-delete)
  - [x] referential integrity preserved across all 7 tables in lite mode
  - [x] row-count verification against Kaggle's published figures (full mode)
  - [x] index creation on join keys
  - [x] `sql/schema.sql` generated from the live catalog
  - [x] rehearsed end-to-end on synthetic data (both header-only and full-column variants)
- [ ] **Run against the real Kaggle dataset** — blocked on Kaggle credentials from user
  - [ ] `python -m src.data.loader` (full mode)
  - [ ] verify real row counts match `SOURCE_FILES.expected_rows`
  - [ ] confirm real default rate ≈ 8.07%
  - [ ] measure real build time and `.duckdb` file size
  - [ ] commit real `sql/schema.sql` (currently holds synthetic-data schema, must be regenerated)

## Phase 2 — EDA

- [ ] `src/eda/analysis.py` — pure functions, one per insight, each returning `(figure_data, chart_spec)`
  - [ ] dataset summary: row/column counts per table, dtypes, memory footprint
  - [ ] feature categorisation (demographic / financial / credit-history / behavioural / document-flag / external-score)
  - [ ] missing-value analysis: % missing per column, grouped by category
  - [ ] data-quality catalogue, confirmed against real data: `DAYS_EMPLOYED=365243` rate, `CODE_GENDER='XNA'` count, `AMT_INCOME_TOTAL` outlier, negative `DAYS_*` sanity, building-stats missingness %
  - [ ] insight 1: default rate by EXT_SOURCE decile
  - [ ] insight 2: age/employment tenure vs default
  - [ ] insight 3: loan-to-income / annuity-to-income burden vs default
  - [ ] insight 4: contract type & income source segments vs default
  - [ ] insight 5: bureau history depth & active-overdue exposure vs default
  - [ ] insight 6: previous-application refusal history vs default
  - [ ] insight 7: installment lateness vs default
- [ ] `src/eda/charts.py` — matplotlib rendering, saved to `models/charts/*.png`
- [ ] `models/eda_artifacts.json` — precomputed figures the API serves
- [ ] `notebooks/eda.py` (jupytext light-format) importing `src.eda.analysis`
- [ ] pair `notebooks/eda.py` ↔ `notebooks/eda.ipynb` via jupytext, execute once, save outputs
- [ ] `docs/EDA_FINDINGS.md` — the 7 insights written up with real numbers and chart references
- [ ] unit tests for `src/eda/analysis.py` functions against the synthetic fixture

## Phase 3 — ML training

- [ ] extend `src/data/features.py` tests: leakage check (no post-application-date info), null-rate sanity per feature block
- [ ] `src/ml/train.py`
  - [ ] baseline: `Pipeline` (median impute → scale → one-hot) + `LogisticRegression`
  - [ ] primary: LightGBM, stratified 5-fold CV, early stopping
  - [ ] imbalance comparison harness: none / `scale_pos_weight` / SMOTE / random undersampling
  - [ ] select strategy on PR-AUC + calibration curve, document the choice
  - [ ] isotonic calibration of the shipped model
  - [ ] risk-band cutoffs derived from validation score distribution → `models/bands.json`
  - [ ] cost-based operating threshold (state the FN:FP ratio assumption)
  - [ ] save `models/model.txt` (LightGBM native format), `models/calibrator.pkl`, `models/feature_list.json`
- [ ] `src/ml/evaluate.py`
  - [ ] ROC-AUC, PR-AUC, KS statistic, Brier score
  - [ ] calibration curve
  - [ ] confusion matrix at chosen threshold
  - [ ] gains/lift table by score decile
  - [ ] `models/metrics.json`
- [ ] `src/ml/predict.py` — inference: raw applicant → feature row → probability → band
- [ ] `docs/MODEL_CARD.md` — model selection rationale, imbalance strategy, metrics, limitations
- [ ] tests: `tests/test_train.py` (fast, tiny synthetic slice), `tests/test_predict.py`

## Phase 4 — Explainable AI

- [ ] `src/ml/explain.py`
  - [ ] SHAP `TreeExplainer` wired to the trained LightGBM model
  - [ ] global explanation at train time: mean-|SHAP| ranking + beeswarm data → `models/eda_artifacts.json` or its own file
  - [ ] per-applicant local explanation: signed top-N contributions
  - [ ] Gemini narrative generation from contributions only (reuse `src/llm/gemini.py`)
  - [ ] `EXPLANATION_SYSTEM_PROMPT` / `EXPLANATION_USER_PROMPT` already exist in `prompt_templates.py` — wire them in
- [ ] tests: contributions sum ≈ (log-odds − base rate); grounding test (narrative never names a factor absent from the input list)

## Phase 5 — Business rules

- [ ] `src/ml/rules.py`
  - [ ] shallow surrogate `DecisionTreeClassifier` (depth 3–4) fit on LightGBM scores
  - [ ] root-to-leaf path extraction as IF-THEN rules
  - [ ] per-rule: population share, observed default rate, lift over base rate
  - [ ] surrogate fidelity report (R²/AUC vs. full model)
  - [ ] `models/rules.json`
- [ ] tests: each rule's stated default rate re-derived independently via a raw DuckDB query (`tests/test_rules.py`)

## Phase 6 — Talk-to-data (mostly done, pending real data)

- [x] `src/talk_to_data/catalog.py` — live schema + Kaggle glossary
- [x] `src/talk_to_data/sql_validator.py` — AST validation (36 adversarial tests)
- [x] `src/talk_to_data/schema_card.py` — tiered, pruned, domain-prior-weighted (18 tests)
- [x] `src/talk_to_data/prompt_templates.py` — versioned, 10 few-shot examples, refusal protocol
- [x] `src/llm/gemini.py` — client, model fallback chain, token accounting, `thinking_budget=0`
- [x] `src/talk_to_data/query_runner.py` — watchdog timeout, markdown rendering (8 tests)
- [x] `src/talk_to_data/nl_to_sql.py` — orchestrator, one-shot repair loop (11 tests)
- [ ] **Live test against real Gemini API** — blocked on `GOOGLE_API_KEY` from user
  - [ ] run the 8 documented query patterns for real, capture actual SQL + answers
  - [ ] measure real token usage per call (prompt/output/thought/cached)
  - [ ] confirm model fallback chain resolves correctly for the user's key tier
- [ ] `docs/PROMPTS.md` — prompt templates, token-optimisation numbers (measured), sample transcripts

## Phase 7 — FastAPI service

- [ ] `src/api/schemas.py` — Pydantic request/response models
- [ ] `src/api/routes/eda.py` — serve `models/eda_artifacts.json` + chart images
- [ ] `src/api/routes/score.py` — `/api/score` (applicant → probability, band)
- [ ] `src/api/routes/explain.py` — `/api/explain` (SHAP + narrative)
- [ ] `src/api/routes/rules.py` — `/api/rules`
- [ ] `src/api/routes/ask.py` — `/api/ask` (talk-to-data)
- [ ] `src/api/main.py` — app factory, static file mount, `/health`, startup checks
- [ ] `tests/test_api.py` — httpx-based route tests (degraded-mode paths included: no DB, no LLM key)

## Phase 8 — Frontend

- [ ] design pass per `frontend-design` skill: token plan (colour/type/layout), review against brief, then build
- [ ] `ui/index.html` — 5 sections: The portfolio / Score an applicant / Why this score / Policy rules / Ask the data
- [ ] `ui/styles.css` — token system, light/dark aware
- [ ] `ui/app.js` — no-build-step vanilla JS, fetch calls to the API
- [ ] self-hosted Newsreader + IBM Plex Sans woff2 subsets in `ui/assets/fonts/`
- [ ] the "risk ruler" component (shared across score/bands/rules/SHAP views)
- [ ] Chrome-driven walkthrough + screenshots of all 5 sections, mobile width, keyboard focus check
- [ ] `docs/DESIGN.md` — token system + rationale, screenshots

## Phase 9 — Docker & deployment

- [ ] `Dockerfile` — multi-stage, non-root user, wheel caching
- [ ] `docker-compose.yml` — `etl` (one-shot) → `api`, volumes for `data/` and `models/`, healthcheck
- [ ] end-to-end `docker compose up` from a clean clone, verified against real data
- [ ] `render.yaml` — web service + persistent disk, background ingestion on first boot, `/health` responsive immediately
- [ ] deploy to Render, verify live URL scores an applicant and answers a chatbot question
- [ ] `docs/DEPLOYMENT.md` — local + Render instructions, cost note

## Phase 10 — Documentation & presentation

- [ ] `README.md` — architecture diagram (Mermaid), setup, model rationale, imbalance strategy, metrics, prompt engineering + token optimisation (measured numbers), rule samples, chatbot transcripts, limitations
- [ ] `documents/project_presentation.pdf` — authored as HTML deck with real screenshots, printed headlessly
- [ ] final consistency pass: every doc reflects the actual, current, verified state of the app
