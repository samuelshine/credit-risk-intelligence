# Progress Tracker

Live pointer into `docs/TASKS.md`. Updated at every step — after each task is
completed, when blocked, and whenever the plan changes. If you're picking this
back up, read **Current task** and **What's next** below; everything else is
context.

Last updated: 2026-09-07 08:34 (local)

---

## Current phase

**Phase 6 — Talk-to-data**, wrapping up: the code and its offline test suite
(80 tests) are complete and passing. The remaining item in this phase needs
a real Gemini API key, which is not yet available, so I'm continuing into
Phase 2 (EDA) and Phase 3 (ML) in parallel while waiting on both blockers.

## Current task

Starting `src/eda/analysis.py` (Phase 2) — dataset summary and feature
categorisation functions, built against the synthetic fixture until real data
arrives.

## What's next (in order)

1. `src/eda/analysis.py` — summary stats, feature categorisation, missing-value analysis
2. `src/eda/charts.py` — matplotlib rendering
3. `src/ml/train.py` — baseline + LightGBM + imbalance comparison (can start on synthetic data to prove the pipeline shape, must rerun on real data once available)
4. Once real data lands: rerun ETL, regenerate `sql/schema.sql`, rerun EDA and training for real, replace every synthetic-data number in docs with real ones
5. Once Gemini key lands: live-test the 8 query patterns, capture real transcripts and token counts for `docs/PROMPTS.md`

## Blockers

| Blocker | Needed for | Status |
|---|---|---|
| Kaggle credentials (`KAGGLE_API_TOKEN`, or manual ZIP in `data/`) | Real ETL run, real EDA numbers, real model metrics | Waiting on user |
| `GOOGLE_API_KEY` | Live chatbot test, real token-usage numbers in `docs/PROMPTS.md` | Waiting on user |

Neither blocks writing code: everything through Phase 6 has been built and
verified against synthetic data shaped like the real dataset (see the
"rehearsed end-to-end" notes in `docs/TASKS.md` and the commit log). Numbers
that depend on the real data are explicitly marked TODO in this tracker and
will not be asserted as final until re-measured for real.

## Phase status at a glance

| Phase | Status |
|---|---|
| 0 — Foundation | ✅ done |
| 1 — Data acquisition & ETL | ✅ code done & rehearsed · ⏸ real run blocked on Kaggle creds |
| 2 — EDA | 🔵 starting now |
| 3 — ML training | ⬜ not started |
| 4 — Explainable AI | ⬜ not started |
| 5 — Business rules | ⬜ not started |
| 6 — Talk-to-data | ✅ code done & tested (80 tests) · ⏸ live run blocked on Gemini key |
| 7 — FastAPI service | ⬜ not started |
| 8 — Frontend | ⬜ not started |
| 9 — Docker & deployment | ⬜ not started |
| 10 — Documentation & presentation | ⬜ started (this file + TASKS.md); README etc. pending |

## Test suite state

80 tests, all passing, no network/API key required:
`PYTHONPATH=. .venv/bin/python -m pytest tests/ -q`

- `test_sql_validator.py` — 36 (adversarial: injection, filesystem escape, hallucinated column)
- `test_schema_card.py` — 18 (pruning correctness, token reduction, domain prior)
- `test_query_runner.py` — 8 (limits, timeout, markdown rendering)
- `test_nl_to_sql.py` — 11 (repair loop, refusal, grounding, degraded mode)
- inline smoke tests for config/logger/helpers/database (not yet formalised as pytest files — see below)

## Known gaps to close before calling this done

- `tests/test_config.py`, `tests/test_database.py`, `tests/test_loader.py`,
  `tests/test_features.py` don't exist yet as committed pytest files — those
  modules were verified via ad hoc scripts in this session (see commit
  messages) but should get proper regression tests before the final pass.
- `sql/schema.sql` currently in the repo reflects the *synthetic* rehearsal
  database, not the real one. Must regenerate after the real ETL run.
