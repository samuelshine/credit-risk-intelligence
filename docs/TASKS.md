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
- [x] **Run against the real Kaggle dataset**
  - [x] `python -m src.data.loader --force` (full mode) — 78.4s, 2.37 GB
  - [x] verify real row counts match `SOURCE_FILES.expected_rows` — all 8 tables exact
  - [x] confirm real default rate ≈ 8.07% — measured 8.073%
  - [x] measure real build time and `.duckdb` file size — 78.4s, 2.37 GB
  - [x] commit real `sql/schema.sql` (regenerated from the real database)

## Phase 2 — EDA ✅ done

- [x] `src/eda/analysis.py` — pure functions, one per insight
  - [x] dataset summary: row/column counts per table
  - [x] feature categorisation (demographic / financial / credit-history / behavioural / document-flag / external-score / region / contact-flag / housing-detail)
  - [x] missing-value analysis: % missing per column, grouped by category
  - [x] data-quality catalogue, confirmed against real data: `DAYS_EMPLOYED=365243` at 18.0%, `CODE_GENDER='XNA'` (4 rows), `AMT_INCOME_TOTAL` outlier (795x median), `DAYS_*` sign check (clean), building-stats missingness 48.8-69.9%, class imbalance 11.4:1
  - [x] insight 1: default rate by EXT_SOURCE_2 decile (18.4% → 3.0%, 6.2x)
  - [x] insight 2: age vs default (11.4% in 20s → 4.9% in 60s)
  - [x] insight 3: employment tenure vs default (11.2% under 2 years)
  - [x] insight 4: credit-to-income quintile vs default (found non-monotonic — quintile 3 peaks, not quintile 5)
  - [x] insight 5: contract type vs default (cash loans 8.3%)
  - [x] insight 6: bureau overdue exposure vs default (16.2% vs 7.6%)
  - [x] insight 7: previous-application refusal history vs default (10.3%)
  - [x] insight 8: installment lateness vs default (12.1%) — 8 insights total, exceeding the ≥5 requirement
- [x] `src/eda/charts.py` — matplotlib rendering, saved to `models/charts/*.png` (9 charts)
- [x] `models/eda_artifacts.json` — precomputed figures the API serves
- [x] `scripts/build_eda_artifacts.py` — regenerates both from the live database
- [ ] `notebooks/eda.py` (jupytext light-format) importing `src.eda.analysis`
- [ ] pair `notebooks/eda.py` ↔ `notebooks/eda.ipynb` via jupytext, execute once, save outputs
- [x] `docs/EDA_FINDINGS.md` — all 8 insights written up with real numbers and chart references
- [x] unit tests for `src/eda/analysis.py` — 12 tests including 2 regression tests for real bugs found (backwards lift ratio, false monotonicity assumption)

## Phase 3 — ML training ✅ done

- [x] `src/data/features.py`: found and fixed a real reproducibility bug — no `ORDER BY` meant `train_test_split(random_state=42)` produced a *different* split on every query execution; fixed and covered by `tests/test_features.py`
- [x] `src/ml/train.py`
  - [x] baseline: `Pipeline` (median impute → scale → one-hot) + `LogisticRegression` — PR-AUC 0.2518
  - [x] primary: LightGBM, stratified 5-fold CV, early stopping — PR-AUC 0.2790
  - [x] imbalance comparison harness: none / `scale_pos_weight` / SMOTE / random undersampling — run on real data, "none" won honestly
  - [x] select strategy on PR-AUC + Brier, documented in `docs/MODEL_CARD.md`
  - [x] isotonic calibration via `CalibratedClassifierCV` + `FrozenEstimator` (sklearn 1.9 removed `cv="prefit"`)
  - [x] risk-band cutoffs → `models/bands.json` (High 10%/29.9% default, Medium 40%/9.7%, Low 50%/2.4%)
  - [x] cost-based operating threshold (FN:FP = 10:1 stated assumption) → 0.0935
  - [x] all artifacts saved: `model.txt`, `calibrator.pkl`, `feature_list.json`, `baseline_model.pkl`
- [x] `src/ml/evaluate.py` — ROC-AUC 0.7899, PR-AUC 0.2794, KS 0.4414, Brier 0.0654, full lift table, calibration curve, confusion matrix → `models/metrics.json`
- [x] `src/ml/predict.py` — inference: raw applicant → feature row → probability → band (real bug found + fixed: WHERE appended after ORDER BY is invalid SQL; also fixed a pandas DataFrame-fragmentation perf issue)
- [x] `docs/MODEL_CARD.md` — written entirely from real training artifacts
- [x] tests: `tests/test_features.py` (3), `tests/test_predict.py` (8), `tests/test_evaluate.py` (8)

## Phase 4 — Explainable AI ✅ done

- [x] `src/ml/explain.py`
  - [x] SHAP `TreeExplainer` wired directly to the trained LightGBM booster
  - [x] global explanation → `models/shap_global.json` (real ranking: EXT_SOURCE_MEAN #1, CODE_GENDER #2 — flagged as a fairness limitation in MODEL_CARD.md)
  - [x] per-applicant local explanation: signed top-6 contributions
  - [x] Gemini narrative generation from contributions only, degrades gracefully with no key
- [x] tests: `tests/test_explain.py` (7) — grounding test (narrative/factors never reference anything outside the computed SHAP values), signal-recovery test against a fixture with known ground truth, LLM-unavailable degradation path

## Phase 5 — Business rules ✅ done

- [x] `src/ml/rules.py`
  - [x] shallow surrogate `DecisionTreeRegressor` (depth 4) fit to approximate the calibrated score (not the label — see module docstring)
  - [x] root-to-leaf path extraction as IF-THEN rules — 16 rules, 36.3% → 2.6% observed default rate
  - [x] per-rule: population share, observed default rate, lift — measured directly against TARGET, not read off the tree
  - [x] surrogate fidelity report: R²=0.488 vs. calibrated score, ROC-AUC 0.7207 vs. actual target (full model: 0.8766 on the same data)
  - [x] `models/rules.json`
- [x] tests: `tests/test_rules.py` (8) — exact-value assertions against a fixture with a known, checkable split (not just "did not crash")

## Phase 6 — Talk-to-data ✅ done (live-tested against real Gemini API)

- [x] `src/talk_to_data/catalog.py` — live schema + Kaggle glossary
- [x] `src/talk_to_data/sql_validator.py` — AST validation (36 adversarial tests)
- [x] `src/talk_to_data/schema_card.py` — tiered, pruned, domain-prior-weighted (18 tests)
- [x] `src/talk_to_data/prompt_templates.py` — versioned, 10 few-shot examples, refusal protocol
- [x] `src/llm/gemini.py` — client, reactive model fallback, token accounting, `thinking_level="low"`
- [x] `src/talk_to_data/query_runner.py` — watchdog timeout, markdown rendering (8 tests)
- [x] `src/talk_to_data/nl_to_sql.py` — orchestrator, one-shot repair loop (11 tests)
- [x] **Live test against real Gemini API**
  - [x] ran multiple query patterns for real, captured actual SQL + grounded answers (zero hallucinated columns, zero repairs needed)
  - [x] measured real token usage per call (prompt/output/thought/cached) via `tests/test_gemini_live.py`
  - [x] confirmed pinned model ids (`gemini-3.7-flash`, `gemini-3.5-flash-lite`) resolve directly, no fallback needed for this key
  - [x] found and fixed 2 real bugs: `thinking_budget=0` rejected on Gemini 3.x (use `thinking_level` instead); summariser hallucinating truncation from seeing `LIMIT` in SQL it didn't need to see
- [ ] `docs/PROMPTS.md` — prompt templates, token-optimisation numbers (measured), sample transcripts (real transcripts already captured this session, need writing up)

## Phase 7 — FastAPI service ✅ done

- [x] `src/api/schemas.py` — Pydantic request/response models
- [x] `src/api/routes/eda.py` — serve `models/eda_artifacts.json` + chart images (path-traversal guarded)
- [x] `src/api/routes/score.py` — `/api/score` (by id or hand-entered fields), `/api/applicants/sample`
- [x] `src/api/routes/explain.py` — `/api/explain` (SHAP + narrative), re-scores server-side rather than trusting a client-supplied probability
- [x] `src/api/routes/rules.py` — `/api/rules`
- [x] `src/api/routes/ask.py` — `/api/ask` (talk-to-data)
- [x] `src/api/main.py` — app factory, static file mount, `/health`, degrades gracefully with no DB/model/LLM
- [x] `tests/test_api.py` — 20 tests, TestClient-based, all degraded-mode paths (no DB, no model, no LLM key) plus real trained-model success paths
- [x] found and fixed 3 more real bugs via live testing (see docs/PROGRESS.md): explanation narrative silently truncated by Gemini 3.x's mandatory "thinking" consuming the entire token budget; the narrative prompt's "Portfolio average" was populated with SHAP's log-odds base value instead of the real base rate; EDA insights crashed the whole build with IndexError when a joined child table legitimately had zero rows

## Phase 8 — Frontend ✅ done

- [x] design pass per `frontend-design` skill: token plan (colour/type/layout), reviewed against the brief for generic AI-tool defaults, then built — `docs/DESIGN.md`
- [x] `ui/index.html` — 5 sections: The portfolio / Score an applicant / Why this score / Policy rules / Ask the data
- [x] `ui/styles.css` — full token system, light/dark aware (auto + explicit `data-theme`)
- [x] `ui/app.js` — no-build-step vanilla JS, fetch calls to the API, no framework
- [x] fonts loaded from Google Fonts (Newsreader + IBM Plex Sans) with full system fallback stack — self-hosting judged not worth the added build complexity for a Docker deployment with normal internet access; documented as a deliberate choice in `docs/DESIGN.md`
- [x] the "risk ruler" component, shared across the score result and the SHAP explanation view
- [x] Chrome-driven walkthrough of all 5 sections against the real trained model and real data: portfolio charts, applicant picker, real-id scoring, manual-entry scoring, SHAP factor chart, policy rules, and a live chatbot round-trip — zero console errors throughout
- [x] `docs/DESIGN.md` — token system + rationale
- [x] found and fixed 2 real UI bugs during the Chrome walkthrough (see docs/PROGRESS.md): a CSS rule collapsed the padding between adjacent table columns, and several places used middot-joined text where the design's own stated anti-patterns ruled that out

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
