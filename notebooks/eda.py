# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#     main_language: python
# ---

# # Home Credit Default Risk — Exploratory Data Analysis
#
# This notebook is a thin presentation layer over `src/eda/analysis.py`. Every
# figure below is computed by the same functions the API serves and that
# `docs/EDA_FINDINGS.md` is written from, so the three cannot drift apart —
# there is exactly one definition of every number in this project.
#
# **Prerequisite:** the database must be built (`python -m src.data.loader`).
#
# Run from the repo root:
# ```
# PYTHONPATH=. jupyter notebook notebooks/eda.ipynb
# ```

# +
import sys
from pathlib import Path

# Allow running from either the repo root or the notebooks/ directory.
ROOT = Path.cwd()
if not (ROOT / "src").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.data.database import database_exists, get_readonly_connection
from src.eda import analysis
from src.utils.config import get_settings

pd.set_option("display.max_columns", 50)
pd.set_option("display.width", 160)

settings = get_settings()
assert database_exists(settings), (
    "No database found. Build it first with: python -m src.data.loader"
)
conn = get_readonly_connection()
print("connected to", settings.duckdb_path)
# -

# ## 1. Dataset summary
#
# Eight tables, one row per loan application in `application_train`, everything
# else a per-client history table joined on `SK_ID_CURR` (except
# `bureau_balance`, which joins through `bureau` on `SK_ID_BUREAU`).

summary = analysis.dataset_summary(conn)
pd.DataFrame([s.__dict__ for s in summary])

# ## 2. Feature categorisation
#
# `application_train`'s columns grouped by business meaning. Note how large the
# housing-detail block is — 42 of the 122 columns — and compare that against
# the missingness table below.

categories = analysis.feature_categories(conn)
pd.DataFrame(
    [{"category": k, "columns": len(v), "examples": ", ".join(v[:3])}
     for k, v in sorted(categories.items(), key=lambda kv: -len(kv[1]))]
)

# ## 3. Missing values
#
# Worst-affected columns first. The top of this list is entirely the
# building/apartment statistics block: those columns are absent for applicants
# whose housing type has no associated building record, so the missingness is
# *informative* rather than random. That is a large part of why the model is
# LightGBM (native NaN handling) rather than something requiring imputation.

missing = analysis.missing_value_report(conn, tables=("application_train",))
pd.DataFrame([m.__dict__ for m in missing[:20]])

# ## 4. Data quality
#
# Six checks, each run against the live database rather than asserted from
# memory. `days_employed_sentinel` and `class_imbalance` are the two that
# directly changed how the model was built.

for finding in analysis.data_quality_findings(conn):
    print(f"[{finding.severity:>13}] {finding.id}")
    print(f"                {finding.description}")
    print(f"                {finding.value}\n")

# ## 5. Business insights
#
# Each insight is one SQL query plus a headline composed from its own result —
# the text below is generated, not typed, so it cannot disagree with the data.

insights = analysis.run_all_insights(conn)
for insight in insights:
    print(f"### {insight.title}")
    print(f"    {insight.headline}")
    print(f"    So what: {insight.so_what}\n")

# ### The insights as tables
#
# Same data, in full. `loan_burden_vs_default` is the interesting one: the
# relationship is *not* monotonic — the middle quintile of credit-to-income
# ratio defaults most, not the heaviest-burden quintile.

for insight in insights:
    print(f"\n=== {insight.id} ===")
    display(pd.DataFrame(insight.rows, columns=insight.columns))

# ### Charts
#
# The same charts the API serves, rendered from these same insight objects.

# +
from IPython.display import Image, display as ipy_display

from src.eda import charts

chart_dir = Path(settings.charts_dir)
if not chart_dir.exists():
    charts.render_all(insights, missing, chart_dir)

for insight in insights:
    path = chart_dir / f"{insight.id}.png"
    if path.exists():
        print(insight.title)
        ipy_display(Image(filename=str(path)))
# -

# ## 6. What this means for the model
#
# The EDA drove four concrete decisions, all carried through into
# `src/data/features.py` and `src/ml/train.py`:
#
# 1. **`DAYS_EMPLOYED = 365243`** (18.0% of rows) is a "not employed" sentinel,
#    not a duration. Left alone it becomes a 1,000-year tenure that dominates
#    every split, so it is mapped to `NULL` with the fact kept as
#    `FLAG_NOT_EMPLOYED`.
# 2. **Missingness is informative**, not random — so it is left as `NaN` for
#    LightGBM rather than imputed away.
# 3. **`EXT_SOURCE_*` dominate**, with a clean monotonic 6.2x risk gradient
#    across deciles, so their mean/min/max/product are made explicit features
#    rather than left for the trees to rediscover.
# 4. **8.07% positive rate (11.4:1)** is the central modelling constraint, and
#    is why the imbalance study in `docs/MODEL_CARD.md` selects on PR-AUC
#    rather than accuracy.
#
# Full write-up: `docs/EDA_FINDINGS.md`. Model decisions: `docs/MODEL_CARD.md`.
