#!/usr/bin/env python
"""Build models/eda_artifacts.json and models/charts/*.png from the live
database. Run after the ETL:

    python scripts/build_eda_artifacts.py

Idempotent - safe to rerun any time the data or the analysis code changes.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.database import get_readonly_connection  # noqa: E402
from src.eda.analysis import build_eda_artifacts, missing_value_report, run_all_insights  # noqa: E402
from src.eda.charts import render_all  # noqa: E402
from src.utils.config import get_settings  # noqa: E402
from src.utils.helpers import write_json  # noqa: E402
from src.utils.logger import configure_logging, get_logger, log_duration  # noqa: E402


def main() -> int:
    configure_logging()
    log = get_logger("build_eda_artifacts")
    settings = get_settings()
    conn = get_readonly_connection()

    with log_duration(log, "build EDA artifacts + charts"):
        artifacts = build_eda_artifacts(conn)
        out_path = write_json(settings.models_dir / "eda_artifacts.json", artifacts)
        log.info("wrote %s", out_path)

        insights = run_all_insights(conn)
        missing = missing_value_report(conn, tables=("application_train",))
        charts = render_all(insights, missing, settings.charts_dir)
        log.info("rendered %d charts to %s", len(charts), settings.charts_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
