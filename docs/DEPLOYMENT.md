# Deployment

Two supported paths: **Docker Compose** locally (the primary one, and what
the assignment's "runs with one command" requirement means), and **Render**
for a hosted instance.

## What the app needs to run

| Thing | Needed for | Without it |
|---|---|---|
| Nothing (fresh clone) | EDA, policy rules, scoring a hand-entered applicant, SHAP explanations | — these work immediately, because the trained model and EDA artifacts are committed |
| `GOOGLE_API_KEY` | "Ask the data" chatbot, plain-English SHAP narratives | Those two degrade with a clear message; nothing else is affected |
| The Kaggle dataset (ETL) | Picking a *real* applicant by id, the chatbot's SQL | Those routes return a clear 503 |

This split is deliberate. `models/model.txt`, `calibrator.pkl`, `bands.json`,
`rules.json`, `metrics.json`, `shap_global.json` and `eda_artifacts.json` are
all committed (see the negated entries in `.gitignore`), so most of the
platform is demonstrable on a clone with no credentials at all. Only the
live-database features need the 2.7 GB download.

## Local: Docker Compose

```bash
cp .env.example .env      # then fill in GOOGLE_API_KEY, optionally KAGGLE_API_TOKEN
docker compose up
```

Open http://localhost:8000.

Two services from one image:

- **`etl`** — runs once, builds the DuckDB database from the Kaggle CSVs, exits.
- **`api`** — FastAPI + the UI on port 8000.

`api` does **not** wait for `etl` to succeed. That is intentional: it lets the
app come up and serve everything in the first table above even when the
dataset is unavailable, and when `etl` does finish, the DB-dependent routes
start working on the next request with no restart. `docker compose ps` will
show `etl` as exited (code 0) once it is done — that is the expected end state,
not a failure.

### Getting the dataset

Home Credit Default Risk is a Kaggle **competition** dataset, so the account
must accept its rules in a browser before any download works — including via
the API. A valid token alone still returns **403** until you have:

1. Accept the rules: https://www.kaggle.com/competitions/home-credit-default-risk/rules
2. Generate a token at https://www.kaggle.com/settings/api and put it in `.env`
   as `KAGGLE_API_TOKEN`.

Or skip the API entirely: download the zip by hand and drop
`home-credit-default-risk.zip` into `./data/`. The loader finds it there,
extracts it, and deletes the zip to reclaim the ~700 MB.

Either way the first ETL run takes about 80 seconds on a modern laptop and
produces a ~3.2 GB DuckDB file. It verifies every table's row count against
Kaggle's published figures and fails loudly on a mismatch rather than
silently building a model on a truncated download.

### Rebuilding

```bash
docker compose run --rm etl python -m src.data.loader --force   # full rebuild
docker compose run --rm etl python -m src.data.loader --mode lite  # sampled, ~50k applicants
```

## Local: plain Python

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env         # fill in keys

python -m src.data.loader              # build the database
python scripts/build_eda_artifacts.py  # (optional) regenerate EDA from your data
python -m src.ml.train                 # (optional) retrain, ~4.5 min
python -m src.ml.rules                 # (optional) re-derive policy rules

uvicorn src.api.main:app --reload --port 8000
```

## Render

`render.yaml` is a Blueprint for a single web service. Push it, then open:

```
https://dashboard.render.com/blueprint/new?repo=https://github.com/samuelshine/credit-risk-intelligence
```

Fill in the two secrets (`GOOGLE_API_KEY`, `KAGGLE_API_TOKEN`) and apply.

### Why one service, not two

Render's persistent disks attach to a **single** instance and are **not
mounted during build**, so there is no way to run the compose file's separate
`etl` container against the same disk. Instead the API ingests for itself:
`AUTO_INGEST_ON_STARTUP=true` makes it check for a database at startup and, if
there is none, build one in a background thread (`src/api/main.py`).

A thread, not an asyncio task, because `build_database()` is blocking DuckDB
work — on the event loop it would stall every request for the ~80 seconds it
runs. `/health` answers immediately throughout, which is what stops Render's
health check from killing the container mid-ingest, and every DB-dependent
route returns its normal 503 until the build finishes.

### Readiness is a marker file, not file existence

`database_exists()` checks for a `.ready` marker written only after a full,
verified ETL run — not for the `.duckdb` file. DuckDB creates that file the
moment a connection opens, long before the tables are populated. This was a
real bug: during a concurrent rebuild a request got a raw 500
(`CatalogException: bureau_balance does not exist`) because file-existence
had already reported the database as available. The marker is cleared at the
start of every build and written at the end, so mid-build the answer is
honestly "not ready".

### Sizing and cost

| | |
|---|---|
| Plan | `standard` (2 GB RAM) — **not** free or starter |
| Disk | 15 GB at `/app/data` |
| First boot | ~4–8 minutes (2.7 GB download, then ~80 s ingest) |

The free and starter tiers are 512 MB and will be OOM-killed during ingestion;
free instances also have no persistent disk, so the database would be lost on
every restart and rebuilt from scratch. `DUCKDB_MEMORY_LIMIT` is set to `1GB`
in the Blueprint so DuckDB spills to disk rather than being killed.

At the time of writing this is a paid tier. If you would rather not pay for a
hosted instance, `docker compose up` gives an identical app locally, and the
committed artifacts mean a reviewer can exercise most of it with no dataset
and no cost.

### Trade-offs of this setup

- **Single instance only.** DuckDB takes a file lock, and the disk attaches to
  one instance, so this cannot scale horizontally as configured. Fine for a
  demo; a real deployment would move the warehouse out of the container.
- **No zero-downtime deploys.** Render stops the old instance before starting
  the new one whenever a disk is attached.
- **Re-ingestion on disk loss.** The disk persists across deploys, so this is
  a first-boot cost, not a per-deploy one.
- **No authentication.** The API is public. Anyone with the URL can score
  applicants and spend the Gemini key's quota. Fine for an assignment demo,
  not for anything real.
