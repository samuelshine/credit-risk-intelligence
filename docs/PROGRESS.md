# Progress Tracker

Live pointer into `docs/TASKS.md`. Updated at every step — after each task is
completed, when blocked, and whenever the plan changes. If you're picking this
back up, read **Current task** and **What's next** below; everything else is
context.

Last updated: 2026-09-07 09:10 (local)

---

## Current phase

**Both blockers are now resolved** — the user provided a Kaggle token
(competition rules accepted) and a Gemini API key. The real dataset is
downloaded, the real DuckDB database is built and verified, real EDA has
run, and the talk-to-data pipeline has been tested end-to-end against the
live Gemini API. Moving into **Phase 3 — ML training** next.

## Current task

Starting `src/ml/train.py`: baseline Logistic Regression + primary LightGBM,
with the four-way imbalance comparison, against the real feature matrix.

## What's next (in order)

1. `src/ml/train.py` — baseline, LightGBM, imbalance comparison, calibration, risk bands
2. `src/ml/evaluate.py` — ROC-AUC/PR-AUC/KS/Brier, calibration curve, lift table
3. `src/ml/predict.py` — inference path
4. `docs/MODEL_CARD.md` — written from the real training run
5. `src/ml/explain.py` — SHAP (Phase 4)
6. `src/ml/rules.py` — surrogate tree (Phase 5)
7. `docs/PROMPTS.md` — write up using the real transcripts and token counts already captured this session (see below)
8. FastAPI service (Phase 7), then UI (Phase 8), then Docker/deploy (Phase 9)

## Blockers

None currently. Both have been resolved:

| Blocker | Resolved |
|---|---|
| Kaggle credentials | ✅ token provided, competition rules accepted after one 403 round-trip, real dataset downloaded (2.68 GB) and ingested |
| `GOOGLE_API_KEY` | ✅ key provided, live-tested against the real API (see below) |

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
| 3 — ML training | 🔵 starting now |
| 4 — Explainable AI | ⬜ not started |
| 5 — Business rules | ⬜ not started |
| 6 — Talk-to-data | ✅ done, live-tested against real Gemini API |
| 7 — FastAPI service | ⬜ not started |
| 8 — Frontend | ⬜ not started |
| 9 — Docker & deployment | ⬜ not started |
| 10 — Documentation & presentation | 🔵 in progress (TASKS/PROGRESS/EDA_FINDINGS done; README etc. pending) |

## Test suite state

`PYTHONPATH=. .venv/bin/python -m pytest tests/ -q --deselect tests/test_gemini_live.py`
→ 89 passed (fully offline, no network/API key required).

`PYTHONPATH=. .venv/bin/python -m pytest tests/test_gemini_live.py -v`
→ 5 passed (live, needs `GOOGLE_API_KEY`; makes real, tiny-cost API calls).

- `test_sql_validator.py` — 36 (adversarial: injection, filesystem escape, hallucinated column)
- `test_schema_card.py` — 18 (pruning correctness, token reduction, domain prior)
- `test_query_runner.py` — 8 (limits, timeout, markdown rendering)
- `test_nl_to_sql.py` — 11 (repair loop, refusal, grounding, degraded mode)
- `test_analysis.py` — 12 (exact-value EDA assertions + 2 regression tests for the monotonicity bugs above)
- `test_charts.py` — 4 (chart rendering)
- `test_gemini_live.py` — 5 (live API contract checks, skipped without a key)

## Known gaps to close before calling this done

- `tests/test_config.py`, `tests/test_database.py`, `tests/test_loader.py`,
  `tests/test_features.py` don't exist yet as committed pytest files — those
  modules were verified via ad hoc scripts in this session (see commit
  messages) and now against the real dataset, but should get proper
  regression tests before the final pass.
- `docs/PROMPTS.md` not yet written, despite having real transcripts and
  token counts already captured in this session's testing — needs those
  numbers assembled into the document.
