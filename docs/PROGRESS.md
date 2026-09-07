# Progress Tracker

Live pointer into `docs/TASKS.md`. Updated at every step — after each task is
completed, when blocked, and whenever the plan changes. If you're picking this
back up, read **Current task** and **What's next** below; everything else is
context.

Last updated: 2026-09-07 10:25 (local)

---

## Current phase

**Phases 1-6 are all done**, verified against the real dataset and (where
applicable) the live Gemini API. Moving into **Phase 7 — FastAPI service**
next, which wires everything built so far behind HTTP routes for the UI.

## Current task

Starting `src/api/schemas.py` and `src/api/main.py` — the FastAPI app
factory and the request/response models the routes will use.

## What's next (in order)

1. `src/api/schemas.py`, `src/api/main.py` — app factory, `/health`, static mount
2. `src/api/routes/*.py` — eda, score, explain, rules, ask
3. `tests/test_api.py` — httpx-based route tests, including degraded-mode paths
4. `docs/PROMPTS.md` — write up using the real transcripts and token counts already captured this session
5. Frontend design pass (Phase 8), then the UI itself
6. Docker + Render deployment (Phase 9)
7. `notebooks/eda.py`/`.ipynb` (small remaining Phase 2 item, low priority)
8. Final README + presentation PDF (Phase 10)

## Blockers

None.

| Blocker | Resolved |
|---|---|
| Kaggle credentials | ✅ token provided, competition rules accepted after one 403 round-trip, real dataset downloaded (2.68 GB) and ingested |
| `GOOGLE_API_KEY` | ✅ key provided, live-tested against the real API (see below) |

## ML training results (real data, real run)

- **Model:** LightGBM, PR-AUC 0.2790 (5-fold CV), beating a logistic-regression
  baseline (0.2518) by +10.8% relative.
- **Imbalance:** compared 4 strategies on identical CV splits; "none" won
  honestly (`scale_pos_weight` was the *worst*, 0.2000 PR-AUC). Full numbers
  and the reasoning in `docs/MODEL_CARD.md`.
- **Holdout (n=61,503):** ROC-AUC 0.7899, PR-AUC 0.2794, KS 0.4414, Brier
  0.0654. Reviewing the riskiest 30% of applicants catches 70.4% of all
  eventual defaulters (lift table in `models/metrics.json`).
- **Risk bands:** High (top 10%) = 29.9% default rate, 3.7x base rate; Low
  (bottom 50%) = 2.4%, 0.3x base rate.
- **SHAP:** `EXT_SOURCE_MEAN` is the #1 global feature; `CODE_GENDER` is #2 —
  flagged as a fair-lending limitation in `docs/MODEL_CARD.md`, not hidden.
- **Rules:** 16 surrogate rules, 36.3% → 2.6% observed default rate, fidelity
  R²=0.488 / ROC-AUC 0.7207 vs. the full model's 0.8766 (both on training
  data) — reported honestly as a moderate-fidelity summary, not a full
  substitute for the model.

## Real bugs found during Phase 3-5 (data/model work), beyond the Phase 1-2 list above

6. **`build_feature_sql` had no `ORDER BY`.** DuckDB gives no row-order
   guarantee without one, and `train_test_split(random_state=42)` splits by
   *position* - so "the same random_state" silently produced a *different*
   train/holdout split on every query execution. Caught by cross-checking a
   holdout evaluation reconstructed from a fresh query call (0.855 ROC-AUC)
   against the number the original training run measured on its own
   in-process split (0.785) - the mismatch was rows leaking across the split,
   not a modelling difference. Fixed by adding `ORDER BY app.SK_ID_CURR`;
   training was rerun in full afterward and every number in this file and in
   `docs/MODEL_CARD.md` is from that corrected run.
7. **`_load_single_applicant` appended `WHERE` after the new `ORDER BY`** -
   invalid SQL, caught immediately when testing `predict.py` against a real
   applicant id. Fixed by wrapping the feature query as a subquery instead of
   string-appending a clause.
8. **`CalibratedClassifierCV(cv="prefit")`** was removed in sklearn 1.6+;
   this environment has 1.9.0. Fixed with the documented replacement,
   `CalibratedClassifierCV(FrozenEstimator(model))`.
9. **Categorical `.map().fillna(-1)`** raised `TypeError` in the SMOTE/
   undersample resampling path - mapping a pandas `Categorical` returns
   another `Categorical`, whose `fillna` rejects a value outside its known
   categories even when replacing the dtype entirely. Fixed by casting to
   `object` before mapping.
10. **`RiskModel._prepare_frame`** built its aligned frame by assigning into
    an empty `DataFrame` one column at a time (~200 single-column inserts),
    which pandas itself flags as `PerformanceWarning: DataFrame is highly
    fragmented`. Fixed to build one dict then one `DataFrame(...)` call.

## Real data, confirmed

- ETL run against the actual Kaggle files: all 8 data tables' row counts match
  Kaggle's published figures exactly (58,489,893 rows + 219-row column
  glossary = 58,490,112 total). Built in 78.4s, 2.37 GB DuckDB file.
- Real default rate: **8.073%** (matches the widely-documented ~8.07% for this
  dataset). Imbalance ratio 11.4:1.
- Real EDA run end-to-end: `docs/EDA_FINDINGS.md`, `models/eda_artifacts.json`,
  and 9 charts in `models/charts/` are all built from the real data, not
  synthetic. Two real bugs were found and fixed while checking this output
  against the actual numbers (see commit `Fix two data-driven bugs...` and
  the "Real bugs found" section below) — the plan's estimates (≈18%
  DAYS_EMPLOYED sentinel rate, 50-70% building-stats missingness) both landed
  almost exactly on the real figures (18.0%, 48.8-69.9%).
- `sql/schema.sql` regenerated from the real database (no longer the
  synthetic-rehearsal placeholder).

## Real bugs found by testing against real systems (not synthetic data)

Keeping this list because it's evidence the "rehearse on synthetic data first"
strategy caught what it could, and testing against the real thing caught the
rest — worth being honest about in the final documentation.

1. **Gemini 3.x rejects `thinking_budget=0`** with `400 INVALID_ARGUMENT`.
   The 2.x-era "fully disable thinking" mechanism doesn't exist on 3.x; only
   `thinking_level` (low/medium/high) does, and even `low` costs 58-77 thought
   tokens on a trivial prompt. Fixed in `src/llm/gemini.py`; the token-
   optimisation docs will report this honestly rather than claim thinking is
   disabled.
2. **`client.models.list()` overstates availability.** `gemini-2.5-flash`
   appeared in the listing but 404'd at generation time ("no longer available
   to new users"). Model resolution is now reactive: a real call, not a list
   membership check, decides whether a fallback candidate works.
3. **Kaggle's API returns a bare 403** for a valid token whose account hasn't
   accepted the competition rules — indistinguishable from a bad token
   without inspecting the status code. Now caught and given an actionable
   message.
4. **Summariser hallucinated "truncated to 500 rows"** for a single-row
   aggregate query, because it was shown the SQL text (which had a `LIMIT
   500` injected by the validator) and inferred truncation from the clause
   itself rather than the actual row count. Fixed by dropping the SQL from
   the summary prompt entirely and stating completeness in one explicit,
   trust-this-over-anything-else sentence.
5. **Two EDA insight functions assumed monotonic relationships** that don't
   hold on the real data: the `EXT_SOURCE_2` decile lift was computed
   backwards (reported "0.2x" for what is actually a 6.2x risk gap), and the
   loan-burden-by-quintile insight assumed the first/last quintile were the
   safest/riskiest when the real relationship peaks in the middle quintile
   (quintile 3 defaults most, not quintile 5). Both fixed with regression
   tests that assert against a purpose-built fixture with a known, checkable
   answer — not just "did not crash."

## Phase status at a glance

| Phase | Status |
|---|---|
| 0 — Foundation | ✅ done |
| 1 — Data acquisition & ETL | ✅ done, real data confirmed |
| 2 — EDA | ✅ done, real numbers in `docs/EDA_FINDINGS.md` |
| 3 — ML training | ✅ done, real numbers in `docs/MODEL_CARD.md` |
| 4 — Explainable AI | ✅ done, real SHAP rankings + narrative generation |
| 5 — Business rules | ✅ done, 16 real rules extracted |
| 6 — Talk-to-data | ✅ done, live-tested against real Gemini API |
| 7 — FastAPI service | 🔵 starting now |
| 8 — Frontend | ⬜ not started |
| 9 — Docker & deployment | ⬜ not started |
| 10 — Documentation & presentation | 🔵 in progress (TASKS/PROGRESS/EDA_FINDINGS/MODEL_CARD done; README etc. pending) |

## Test suite state

`PYTHONPATH=. .venv/bin/python -m pytest tests/ -q --deselect tests/test_gemini_live.py`
→ 123 passed (fully offline, no network/API key required).

`PYTHONPATH=. .venv/bin/python -m pytest tests/test_gemini_live.py -v`
→ 5 passed (live, needs `GOOGLE_API_KEY`; makes real, tiny-cost API calls).

- `test_sql_validator.py` — 36 (adversarial: injection, filesystem escape, hallucinated column)
- `test_schema_card.py` — 18 (pruning correctness, token reduction, domain prior)
- `test_query_runner.py` — 8 (limits, timeout, markdown rendering)
- `test_nl_to_sql.py` — 11 (repair loop, refusal, grounding, degraded mode)
- `test_analysis.py` — 12 (exact-value EDA assertions + 2 regression tests for the monotonicity bugs above)
- `test_charts.py` — 4 (chart rendering)
- `test_features.py` — 3 (SQL determinism regression test + `load_features` contract)
- `test_predict.py` — 8 (loading, band assignment, manual-entry scoring, real signal recovery)
- `test_explain.py` — 7 (SHAP grounding, signal recovery, LLM-unavailable degradation)
- `test_rules.py` — 8 (exact-value fidelity/lift assertions against a known-answer fixture)
- `test_evaluate.py` — 8 (KS/lift/confusion-matrix against known constructed cases)
- `test_gemini_live.py` — 5 (live API contract checks, skipped without a key)

`tests/conftest_ml.py` is a shared (non-auto-loaded) helper: trains and saves
a small, genuinely real LightGBM + isotonic-calibrator model in milliseconds,
through the exact same save code paths `train.py` uses, so `predict.py`/
`explain.py`/`rules.py` are tested against real artifacts without needing the
~4 minute full training run.

## Known gaps to close before calling this done

- `tests/test_config.py`, `tests/test_database.py`, `tests/test_loader.py`
  don't exist yet as committed pytest files — those modules were verified via
  ad hoc scripts in this session (see commit messages) and now against the
  real dataset, but should get proper regression tests before the final pass.
- `docs/PROMPTS.md` not yet written, despite having real transcripts and
  token counts already captured in this session's testing — needs those
  numbers assembled into the document.
- `notebooks/eda.py`/`.ipynb` (jupytext pairing) not yet done — low priority,
  the same analysis functions are already exercised by the API and tests.
